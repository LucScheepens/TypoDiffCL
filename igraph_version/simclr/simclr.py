import math
import os
import sys
import random
import torch
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


# Make the diffusion package importable from the sibling directory
_DIFF_DIR = Path(__file__).resolve().parent.parent / "diffusion"

if str(_DIFF_DIR) not in sys.path:
    sys.path.insert(0, str(_DIFF_DIR))


BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR.parent)) 

from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
import time

import torch.nn as nn
import torch.nn.functional as F

from augmentation import augment_network_view_fast, build_igraph_from_transactions
from diffusion.diff_util import network_to_dense




def prepare_networks(networks, full_df):
    full_graph = build_igraph_from_transactions(full_df)

    for net in networks:
        if "graph" not in net or net["graph"] is None:
            net["graph"] = build_igraph_from_transactions(net["transactions"])

    return full_graph


def network_to_pyg_data_fast(network):
    g = network["graph"]
    n = g.vcount()
    has_names = "name" in g.vs.attributes()
    names = [int(g.vs[i]["name"]) for i in range(n)] if has_names else list(range(n))
    laundering = network["laundering_nodes"]

    elist = g.get_edgelist()
    if elist:
        srcs, dsts = zip(*elist)
        edge_index = torch.tensor(
            [list(srcs) + list(dsts), list(dsts) + list(srcs)],
            dtype=torch.long,
        )
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long)

    # Use precomputed features if available, size matches, and dim matches current schema
    _TARGET_DIM = 11
    if ("x_dense" in network and network["x_dense"] is not None
            and network["x_dense"].shape[0] == n
            and network["x_dense"].shape[1] == _TARGET_DIM):
        x = network["x_dense"].clone()
        # Refresh degree and laundering in case augmentation changed them
        degrees = g.degree()
        max_deg = max(max(degrees), 1)
        for i in range(n):
            x[i, 0] = float(names[i] in laundering)
            x[i, 1] = degrees[i] / max_deg
    else:
        # Recompute from network_to_dense (handles all feature dimensions correctly)
        from diffusion.diff_util import network_to_dense as _ntd
        x, _ = _ntd(network)
        # Refresh laundering flag in case augmentation changed the graph
        degrees = g.degree()
        max_deg = max(max(degrees), 1)
        for i in range(n):
            x[i, 0] = float(names[i] in laundering)
            x[i, 1] = degrees[i] / max_deg
        # Cache for future calls within this session
        network["x_dense"] = x

    # Strip col 0 (laundering flag) — the label must not flow through the GCN
    # layers, only through the SupCon loss term where it is used as a label signal.
    # NOTE: this changes encoder in_dim from 11→10 (IBM) or 6→5 (Elliptic).
    # Existing checkpoints trained with the flag included must be retrained.
    return Data(x=x[:, 1:], edge_index=edge_index)


class GraphEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, out_dim=128):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.lin = nn.Linear(hidden_dim, out_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = global_mean_pool(x, batch)
        x = self.lin(x)

        return x


class ProjectionHead(nn.Module):
    def __init__(self, in_dim, proj_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, proj_dim)
        )

    def forward(self, z):
        return self.net(z)



def nt_xent_loss(z1, z2, temperature=0.5):
    """
    z1, z2: (batch_size, dim)
    """
    z1 = z1.float()
    z2 = z2.float()
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    batch_size = z1.size(0)

    z = torch.cat([z1, z2], dim=0)

    sim = torch.matmul(z, z.t()) / temperature
    sim_exp = torch.exp(sim)

    mask = ~torch.eye(2 * batch_size, dtype=bool, device=z.device)

    sim_exp = sim_exp * mask

    pos_sim = torch.exp(torch.sum(z1 * z2, dim=1) / temperature)
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)

    denom = sim_exp.sum(dim=1)

    loss = -torch.log(pos_sim / denom)

    return loss.mean()


def sup_con_loss(z, labels, temperature=0.07):
    """
    Supervised Contrastive Loss (Khosla et al., 2020).

    z      : [N, D]  embeddings (will be L2-normalised internally)
    labels : [N]     integer class labels (0 = clean, 1 = laundering)

    All same-class pairs act as positives; all cross-class pairs act as
    negatives.  Uses a lower temperature than NT-Xent (0.07 vs 0.5) to
    create tighter, better-separated clusters.

    Returns scalar loss, or 0 if the batch has no valid positive pairs
    (e.g. all samples belong to the same class).
    """
    z = z.float()
    z = F.normalize(z, dim=1)
    N = z.shape[0]

    sim = torch.matmul(z, z.T) / temperature           # [N, N]

    self_mask = torch.eye(N, dtype=torch.bool, device=z.device)
    labels    = labels.view(-1)
    pos_mask  = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~self_mask  # [N, N]

    if pos_mask.sum() == 0:
        return torch.tensor(0.0, device=z.device)

    # Numerical stability: subtract per-row max (excluding self)
    sim_no_self = sim.masked_fill(self_mask, -1e9)
    sim = sim - sim_no_self.max(dim=1, keepdim=True).values.detach()

    exp_sim   = torch.exp(sim).masked_fill(self_mask, 0.0)   # zero out self
    log_denom = torch.log(exp_sim.sum(dim=1, keepdim=True).clamp(min=1e-8))

    log_prob        = sim - log_denom                          # [N, N]
    n_pos           = pos_mask.float().sum(dim=1).clamp(min=1)
    loss_per_anchor = -(log_prob * pos_mask.float()).sum(dim=1) / n_pos

    # Only average over anchors that actually have a positive pair
    has_pos = pos_mask.any(dim=1)
    return loss_per_anchor[has_pos].mean()





# ─────────────────────────────────────────────────────────────────────────────
# Shared dense-tensor helpers used by all diffusion view functions
# ─────────────────────────────────────────────────────────────────────────────

def _prepare_dense_input(network_or_data6, max_nodes, x_mean, x_std, device, is_pyg=False):
    """
    Convert an IBM network dict or an Elliptic PyG Data object (6-dim features)
    into normalised padded tensors ready for the diffusion model.

    Returns (x_norm, adj_pad, mask, n, x_mean_d, x_std_d)  or  None if too large.
    """
    if is_pyg:
        data6 = network_or_data6
        n     = data6.x.shape[0]
        x     = data6.x.clone()
        ei    = data6.edge_index
        adj   = torch.zeros(n, n)
        if ei.shape[1] > 0:
            valid = (ei[0] != ei[1])
            adj[ei[0][valid], ei[1][valid]] = 1.0
    else:
        if "x_dense" in network_or_data6 and "adj_dense" in network_or_data6:
            x   = network_or_data6["x_dense"]
            adj = network_or_data6["adj_dense"]
        else:
            x, adj = network_to_dense(network_or_data6)
        n = x.shape[0]

    if n > max_nodes:
        return None

    # Pad only to actual graph size (model is fully masked — any N works).
    # Avoids allocating the full [1, 300, 300] adj tensor for small graphs.
    node_dim = x.shape[1]
    x_pad    = torch.zeros(1, n, node_dim, device=device)
    adj_pad  = torch.zeros(1, n, n,        device=device)
    mask     = torch.ones(1, n,            device=device)
    x_pad[0]       = x.to(device)
    adj_pad[0] = adj.to(device)

    x_mean_d = x_mean.to(device)
    x_std_d  = x_std.to(device)
    x_norm   = x_pad.clone()
    x_norm[:, :, 1:] = (x_pad[:, :, 1:] - x_mean_d[1:]) / x_std_d[1:]

    return x_norm, adj_pad, mask, n, x_mean_d, x_std_d


def _dense_to_pyg(x_t, adj_t, n, x_mean_d, x_std_d, label=None):
    """
    Convert the final denoised dense tensors [1, MAX, node_dim] back to a
    PyG Data object with denormalised features and thresholded edge_index.
    """
    x0 = x_t[0, :n].cpu()
    x0[:, 1:] = x0[:, 1:] * x_std_d[1:].cpu() + x_mean_d[1:].cpu()
    x0[:, 0]  = x0[:, 0].clamp(0.0, 1.0)
    deg       = adj_t[0, :n, :n].cpu().sum(dim=-1)
    x0[:, 1]  = deg / deg.max().clamp(min=1.0)
    ei = (adj_t[0, :n, :n].cpu() > 0.5).nonzero(as_tuple=False).T.contiguous()
    # Strip col 0 (laundering flag) so diffusion views have the same input dim
    # as structural views produced by network_to_pyg_data_fast.
    d  = Data(x=x0[:, 1:], edge_index=ei)
    if label is not None:
        d.y = torch.tensor([label], dtype=torch.long)
    return d


# ─────────────────────────────────────────────────────────────────────────────
# Option A — Multi-step DDIM view
# ─────────────────────────────────────────────────────────────────────────────

def _diffusion_view_multistep(
    network_or_data6, diff_model, diffusion, x_mean, x_std, device,
    t_start_frac=0.5, n_steps=15, max_nodes=300, is_pyg=False,
):
    """
    Option A: Multi-step DDIM view.

    Forward-noise to t_start, then run n_steps evenly-spaced DDIM denoising
    steps (deterministic, no added noise).  Much more diverse than the current
    single forward-pass reconstruction while staying on the diffusion manifold.

    t_start_frac : fraction of total T to use as starting noise level (default 0.5)
    n_steps      : number of denoising steps (default 15)
    """
    prep = _prepare_dense_input(network_or_data6, max_nodes, x_mean, x_std, device, is_pyg)
    if prep is None:
        return None
    x_norm, adj_pad, mask, n, x_mean_d, x_std_d = prep

    t_start = max(n_steps, int(t_start_frac * diffusion.num_timesteps))
    t_vec   = torch.tensor([t_start], device=device)
    x_t, adj_t = diffusion.q_sample(x_norm, t_vec, node_mask=mask, adj_start=adj_pad)

    # Evenly spaced schedule from t_start → 0
    step     = max(1, t_start // n_steps)
    schedule = list(range(t_start, -1, -step))
    if schedule[-1] != 0:
        schedule.append(0)

    was_training = diff_model.training
    diff_model.eval()

    with torch.no_grad():
        for i, t_curr in enumerate(schedule):
            t_next = schedule[i + 1] if i + 1 < len(schedule) else -1
            t_c    = torch.tensor([t_curr], device=device)
            t_sc   = diffusion._scale_timesteps(t_c)

            eps_pred, adj_pred, _ = diff_model(x_t, t_sc, adj=adj_t, node_mask=mask)

            ab_t       = float(diffusion.alphas_cumprod[t_curr])
            sqrt_ab_t  = math.sqrt(ab_t + 1e-8)
            sqrt_1m_t  = math.sqrt(max(1.0 - ab_t, 1e-8))

            # Predict x0 from epsilon
            x0_cont  = (x_t[..., 1:] - sqrt_1m_t * eps_pred[..., 1:]) / sqrt_ab_t
            x0_bin   = eps_pred[..., 0:1].clamp(0, 1)
            x0_pred  = torch.cat([x0_bin, x0_cont], dim=-1) * mask.unsqueeze(-1)

            if t_next < 0:                               # final step → take x0
                x_t   = x0_pred
                adj_t = (adj_pred.clamp(0, 1) > 0.5).float()
            else:
                ab_s       = float(diffusion.alphas_cumprod[t_next])
                sqrt_ab_s  = math.sqrt(ab_s + 1e-8)
                sqrt_1m_s  = math.sqrt(max(1.0 - ab_s, 1e-8))
                # DDIM deterministic direction: noise vector estimated at t_curr
                noise_cont = (x_t[..., 1:] - sqrt_ab_t * x0_pred[..., 1:]) / sqrt_1m_t
                x_next_cont = sqrt_ab_s * x0_pred[..., 1:] + sqrt_1m_s * noise_cont
                x_t   = torch.cat([x0_bin, x_next_cont], dim=-1) * mask.unsqueeze(-1)
                adj_t = torch.bernoulli(adj_pred.clamp(0, 1))
                adj_t = (adj_t + adj_t.transpose(-1, -2)) / 2 * mask[:, :, None] * mask[:, None, :]

    if was_training:
        diff_model.train()

    # label=None: SimCLR builds the labels tensor separately; no .y needed on the Data object
    return _dense_to_pyg(x_t, adj_t, n, x_mean_d, x_std_d, label=None)


# ─────────────────────────────────────────────────────────────────────────────
# Option C — Guided multi-step DDIM view
# ─────────────────────────────────────────────────────────────────────────────

def _diffusion_view_guided(
    network_or_data6, diff_model, diffusion, x_mean, x_std,
    encoder, probe, device,
    t_start_frac=0.5, n_steps=15, guidance_scale=1.5,
    guide_from_frac=0.4, guide_every=3,
    max_nodes=300, is_pyg=False,
):
    """
    Option C: Class-conditional guided multi-step DDIM view.

    Like _diffusion_view_multistep but in the later denoising steps, applies
    classifier guidance from the probe to steer the view toward the original
    graph's class label.  This trains the encoder to be invariant to
    semantically faithful, class-preserving deformations.

    guide_from_frac : fraction of total denoising steps before guidance starts
    guide_every     : apply guidance every N steps (caching between steps)
    """
    prep = _prepare_dense_input(network_or_data6, max_nodes, x_mean, x_std, device, is_pyg)
    if prep is None:
        return None
    x_norm, adj_pad, mask, n, x_mean_d, x_std_d = prep

    target_label = (float(network_or_data6.y.item()) if is_pyg
                    else float(len(network_or_data6.get("laundering_nodes", set())) > 0))

    t_start = max(n_steps, int(t_start_frac * diffusion.num_timesteps))
    t_vec   = torch.tensor([t_start], device=device)
    x_t, adj_t = diffusion.q_sample(x_norm, t_vec, node_mask=mask, adj_start=adj_pad)

    step     = max(1, t_start // n_steps)
    schedule = list(range(t_start, -1, -step))
    if schedule[-1] != 0:
        schedule.append(0)
    guide_from = int(guide_from_frac * len(schedule))

    was_diff_training = diff_model.training
    was_enc_training  = encoder.training
    diff_model.eval()
    encoder.eval()

    cached_grad = None

    for i, t_curr in enumerate(schedule):
        t_next   = schedule[i + 1] if i + 1 < len(schedule) else -1
        t_c      = torch.tensor([t_curr], device=device)
        t_sc     = diffusion._scale_timesteps(t_c)
        do_guide = (i >= guide_from) and (i % guide_every == 0)

        if do_guide:
            # Compute guidance gradient w.r.t. x_t (in score space)
            with torch.enable_grad():
                x_t_g   = x_t.detach().requires_grad_(True)
                eps_g, adj_g, _ = diff_model(x_t_g, t_sc, adj=adj_t, node_mask=mask)

                ab_t      = float(diffusion.alphas_cumprod[t_curr])
                sqrt_ab_t = math.sqrt(ab_t + 1e-8)
                sqrt_1m_t = math.sqrt(max(1.0 - ab_t, 1e-8))
                x0_cont_g = (x_t_g[..., 1:] - sqrt_1m_t * eps_g[..., 1:]) / sqrt_ab_t
                x0_bin_g  = eps_g[..., 0:1].clamp(0, 1)
                x0_g      = torch.cat([x0_bin_g, x0_cont_g], dim=-1) * mask.unsqueeze(-1)

                # Build a temporary PyG view from x0_pred for probe evaluation
                x_feat = x0_g[0, :n].clone()
                x_feat[:, 1:] = x_feat[:, 1:] * x_std_d[1:] + x_mean_d[1:]
                x_feat[:, 0]  = x0_g[0, :n, 0].clamp(0, 1)
                deg_s         = adj_g[0, :n, :n].sum(dim=-1)
                x_feat[:, 1]  = deg_s / deg_s.detach().max().clamp(min=1.0)
                ei_g = (adj_g[0, :n, :n].detach() > 0.5).nonzero(as_tuple=False).T.contiguous()
                if ei_g.shape[1] == 0:
                    ei_g = torch.zeros(2, 0, dtype=torch.long, device=device)
                # Strip col 0 before passing to encoder (matches fixed network_to_pyg_data_fast)
                pyg_tmp = Data(x=x_feat[:, 1:], edge_index=ei_g,
                               batch=torch.zeros(n, dtype=torch.long, device=device))

                h     = encoder(pyg_tmp)
                h_n   = F.normalize(h, dim=-1)
                score = torch.sigmoid(probe(h_n)).squeeze()
                g_loss = (-torch.log(score + 1e-8) if target_label == 1.0
                          else -torch.log(1.0 - score + 1e-8))

                # Gradient w.r.t. x_t — does NOT accumulate into .grad of any param
                cached_grad = torch.autograd.grad(g_loss, x_t_g)[0].detach().clamp(-1.0, 1.0)
                eps_pred = eps_g.detach()
                adj_pred = adj_g.detach()
        else:
            with torch.no_grad():
                eps_pred, adj_pred, _ = diff_model(x_t, t_sc, adj=adj_t, node_mask=mask)

        with torch.no_grad():
            ab_t      = float(diffusion.alphas_cumprod[t_curr])
            sqrt_ab_t = math.sqrt(ab_t + 1e-8)
            sqrt_1m_t = math.sqrt(max(1.0 - ab_t, 1e-8))

            # Apply score-space guidance to the continuous features
            if cached_grad is not None:
                sqrt_1m_ab = math.sqrt(max(1.0 - ab_t, 1e-8))
                eps_cont = eps_pred[..., 1:] + guidance_scale * sqrt_1m_ab * cached_grad[..., 1:]
            else:
                eps_cont = eps_pred[..., 1:]

            x0_cont = (x_t[..., 1:] - sqrt_1m_t * eps_cont) / sqrt_ab_t
            x0_bin  = eps_pred[..., 0:1].clamp(0, 1)
            x0_pred = torch.cat([x0_bin, x0_cont], dim=-1) * mask.unsqueeze(-1)

            if t_next < 0:
                x_t   = x0_pred
                adj_t = (adj_pred.clamp(0, 1) > 0.5).float()
            else:
                ab_s       = float(diffusion.alphas_cumprod[t_next])
                sqrt_ab_s  = math.sqrt(ab_s + 1e-8)
                sqrt_1m_s  = math.sqrt(max(1.0 - ab_s, 1e-8))
                noise_cont = (x_t[..., 1:] - sqrt_ab_t * x0_pred[..., 1:]) / sqrt_1m_t
                x_next_cont = sqrt_ab_s * x0_pred[..., 1:] + sqrt_1m_s * noise_cont
                x_t   = torch.cat([x0_bin, x_next_cont], dim=-1) * mask.unsqueeze(-1)
                adj_t = torch.bernoulli(adj_pred.clamp(0, 1))
                adj_t = (adj_t + adj_t.transpose(-1, -2)) / 2 * mask[:, :, None] * mask[:, None, :]

    if was_diff_training:
        diff_model.train()
    if was_enc_training:
        encoder.train()

    # label=None: SimCLR builds the labels tensor separately; no .y needed on the Data object
    return _dense_to_pyg(x_t, adj_t, n, x_mean_d, x_std_d, label=None)


# ─────────────────────────────────────────────────────────────────────────────
# Probe fitting helper  (used by guided view and generation pipeline)
# ─────────────────────────────────────────────────────────────────────────────

def _fit_probe(encoder, networks_or_graphs, device, n_epochs=300, is_pyg=False):
    """
    Train a small MLP probe on frozen encoder embeddings.

    For IBM: networks_or_graphs is a list of network dicts.
    For Elliptic: networks_or_graphs is a list of PyG Data objects (6-dim).

    Returns the probe in eval mode with all parameters frozen.
    """
    was_training = encoder.training
    encoder.eval()

    all_graphs, all_labels = [], []
    with torch.no_grad():
        for item in networks_or_graphs:
            if is_pyg:
                # Elliptic graphs may carry the 6-dim representation used by the
                # diffusion model (col 0 = label).  Strip col 0 so the encoder
                # receives the same 5-dim input it sees during forward passes.
                x_enc = item.x[:, 1:] if item.x.shape[1] > 1 else item.x
                all_graphs.append(Data(x=x_enc, edge_index=item.edge_index.clone()))
                all_labels.append(int(item.y.item()))
            else:
                v = augment_network_view_fast(item)
                all_graphs.append(network_to_pyg_data_fast(v))
                all_labels.append(int(len(item["laundering_nodes"]) > 0))
        H = encoder(Batch.from_data_list(all_graphs).to(device)).cpu()

    if was_training:
        encoder.train()

    H = F.normalize(H, dim=1)
    y = torch.tensor(all_labels, dtype=torch.float32)

    probe = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1)).to(device)
    # weight_decay provides L2 regularisation — prevents overfitting on small
    # training sets which would produce misleading guidance gradients.
    opt   = torch.optim.Adam(probe.parameters(), lr=5e-3, weight_decay=1e-4)
    for _ in range(n_epochs):
        logit = probe(H.to(device)).squeeze(-1)
        loss  = F.binary_cross_entropy_with_logits(logit, y.to(device))
        opt.zero_grad(); loss.backward(); opt.step()

    with torch.no_grad():
        preds = (torch.sigmoid(probe(H.to(device)).squeeze(-1)) > 0.5).cpu()
    acc = (preds == y.bool()).float().mean()
    print(f"  [probe] trained {n_epochs} epochs — acc={acc:.3f}")
    for p in probe.parameters():
        p.requires_grad_(False)
    probe.eval()
    return probe


def _diffusion_view(network, diffusion_model, diffusion, x_mean, x_std,
                    max_nodes=300, t_frac=0.3, device="cpu"):
    """
    Generate one augmented view via the diffusion model.

    Strategy: forward-noise the graph to timestep t, then run a single
    model forward pass to recover x_0 and adj_pred.  This is O(1) inference
    (no iterative denoising loop) so it adds negligible overhead to SimCLR.

    Returns a PyG Data object with 3 node features compatible with GraphEncoder:
        [degree, degree, laundering_probability]
    """

    # Use precomputed features if available (avoids expensive betweenness/pagerank recompute)
    if "x_dense" in network and "adj_dense" in network:
        x   = network["x_dense"]
        adj = network["adj_dense"]
    else:
        x, adj = network_to_dense(network)
    n = x.shape[0]

    if n > max_nodes:
        # Graph too large for the diffusion model — fall back silently
        return None

    # Pad only to actual graph size (model is fully masked — any N works).
    node_dim = x.shape[1]
    x_pad   = x.unsqueeze(0).to(device)        # [1, n, F]
    adj_pad = adj.unsqueeze(0).to(device)      # [1, n, n]
    mask    = torch.ones(1, n, device=device)  # [1, n]

    # Normalise continuous features (match train.py)
    x_norm = x_pad.clone()
    x_norm[:, :, 1:] = (x_pad[:, :, 1:] - x_mean[1:]) / x_std[1:]

    # Choose noise level
    t_abs = max(1, int(t_frac * diffusion.num_timesteps))
    t     = torch.tensor([t_abs], device=device)

    # Forward diffusion: corrupt features and adjacency
    x_t, adj_t = diffusion.q_sample(x_norm, t, node_mask=mask, adj_start=adj_pad)

    # Single-step denoising: one forward pass to predict x_0 and adj
    was_training = diffusion_model.training
    diffusion_model.eval()
    with torch.no_grad():
        eps_pred, adj_pred, _ = diffusion_model(
            x_t, diffusion._scale_timesteps(t),
            adj=adj_t, node_mask=mask,
        )
        # Recover x_0 for continuous features from predicted epsilon
        x0_cont = diffusion._predict_xstart_from_eps(x_t[..., 1:], t, eps_pred[..., 1:])
        # Binary laundering feature: model predicts x_start directly
        x0_bin  = eps_pred[..., 0:1].clamp(0.0, 1.0)
    if was_training:
        diffusion_model.train()

    # Extract valid-node slice
    x0_node  = torch.cat([x0_bin, x0_cont], dim=-1)[0, :n]   # [n, 6]
    adj_node = adj_pred[0, :n, :n]                             # [n, n], values in [0,1]

    # Build 6-D node features matching training scale (denormalize diffusion space)
    x_feat = x0_node.clone()                                     # [n, 6]
    x_feat[:, 1:] = x_feat[:, 1:] * x_std[1:].to(device) + x_mean[1:].to(device)
    x_feat[:, 0]  = x0_node[:, 0].clamp(0.0, 1.0)              # laundering prob
    # Override degree (feature 1) with soft adjacency degree, normalised to [0,1]
    deg   = adj_node.sum(dim=-1)
    max_d = deg.detach().max().clamp(min=1.0)
    x_feat[:, 1] = deg / max_d

    # Build edge_index from thresholded adjacency
    edge_index = (adj_node > 0.5).nonzero(as_tuple=False).T.contiguous()  # [2, E]

    # Strip col 0 (laundering flag) to match network_to_pyg_data_fast output dim
    return Data(x=x_feat[:, 1:].cpu(), edge_index=edge_index.cpu())


def train_simclr_fast(
    networks,
    full_df,
    encoder,
    projector,
    optimizer,
    device,
    batch_size=8,
    epochs=50,
    checkpoint_dir="model_checkpoints",
    checkpoint_interval=10,
    diffusion_model=None,
    diffusion=None,
    x_mean=None,
    x_std=None,
    p_diffusion=0.3,
    diffusion_t_frac=0.3,
    max_nodes=300,
    supcon_weight=0.5,
    supcon_temperature=0.07,
    # ── new: view-type control ────────────────────────────────────────────────
    view_type="single_step",   # "single_step" | "multistep" | "guided" | "multistep_to_guided"
    diff_n_steps=15,           # DDIM steps for multistep / guided views
    diff_t_start_frac=0.5,     # noise level: fraction of total T
    diff_guidance_scale=1.5,   # guidance strength for guided views
    probe_warmup_epochs=20,    # for "multistep_to_guided": epochs of Option A before switching
    probe_update_every=20,     # kept for API compatibility — probe is now fitted once only
):
    encoder.train()
    projector.train()

    os.makedirs(checkpoint_dir, exist_ok=True)

    use_amp = (device.type == "cuda") if hasattr(device, "type") else (str(device) != "cpu")
    scaler  = torch.amp.GradScaler("cuda", enabled=use_amp)
    all_params = list(encoder.parameters()) + list(projector.parameters())

    def _make_view1(net):
        return network_to_pyg_data_fast(augment_network_view_fast(net))

    aug_pool = ThreadPoolExecutor(max_workers=4)

    best_loss = float('inf')
    best_encoder_state = None
    best_projector_state = None
    best_epoch = None

    use_diffusion = (
        diffusion_model is not None
        and diffusion is not None
        and x_mean is not None
        and x_std is not None
    )

    # ── view-type flags ───────────────────────────────────────────────────────
    _use_multistep       = view_type in ("multistep", "guided", "multistep_to_guided")
    _use_guided          = view_type == "guided"
    _use_progressive     = view_type == "multistep_to_guided"
    probe                = None   # maintained across epochs for guided views

    start_time = time.time()
    total_batches = 0

    for epoch in range(epochs):
        epoch_start = time.time()
        encoder.train(); projector.train()
        if diffusion_model is not None:
            diffusion_model.eval()

        # ── decide view strategy for this epoch ──────────────────────────────
        guided_this_epoch = _use_guided or (_use_progressive and epoch >= probe_warmup_epochs)

        if guided_this_epoch and use_diffusion:
            # Fit the probe exactly once when the guided phase begins.
            # Periodic re-fitting on a small training set produces an unstable
            # guidance signal — the probe overfits to different noise realisations
            # of the embeddings on each refit, making gradients inconsistent.
            if probe is None:
                print(f"  [probe] fitting once on {len(networks)} networks …")
                probe = _fit_probe(encoder, networks, device, n_epochs=300, is_pyg=False)

        print(f"Epoch {epoch + 1}/{epochs}"
              + (f"  [guided]"    if guided_this_epoch and use_diffusion else
                 f"  [multistep]" if _use_multistep and use_diffusion else ""))
        total_loss = total_ntxent = total_supcon = 0.0

        for i in range(0, len(networks), batch_size):
            total_batches += 1
            batch = networks[i:i + batch_size]

            views1 = list(aug_pool.map(_make_view1, batch))

            views2 = []
            for net in batch:
                # View 2: select diffusion strategy (GPU — serial)
                if use_diffusion and random.random() < p_diffusion:
                    if guided_this_epoch and probe is not None:
                        diff_view = _diffusion_view_guided(
                            net, diffusion_model, diffusion, x_mean, x_std,
                            encoder, probe, device,
                            t_start_frac=diff_t_start_frac, n_steps=diff_n_steps,
                            guidance_scale=diff_guidance_scale, max_nodes=max_nodes,
                        )
                    elif _use_multistep:
                        diff_view = _diffusion_view_multistep(
                            net, diffusion_model, diffusion, x_mean, x_std, device,
                            t_start_frac=diff_t_start_frac, n_steps=diff_n_steps,
                            max_nodes=max_nodes,
                        )
                    else:
                        diff_view = _diffusion_view(
                            net, diffusion_model, diffusion,
                            x_mean, x_std, max_nodes, diffusion_t_frac, device,
                        )
                    if diff_view is not None:
                        views2.append(diff_view)
                        continue
                # Fallback: structural augmentation (CPU — pre-built in parallel)
                v2 = augment_network_view_fast(net)
                views2.append(network_to_pyg_data_fast(v2))

            # Binary labels for the batch (1 = laundering present, 0 = clean)
            labels = torch.tensor(
                [int(len(net["laundering_nodes"]) > 0) for net in batch],
                dtype=torch.long, device=device,
            )

            data1 = Batch.from_data_list(views1).to(device)
            data2 = Batch.from_data_list(views2).to(device)

            optimizer.zero_grad()

            device_type = device.type if hasattr(device, "type") else str(device).split(":")[0]
            # Only the GNN forward passes run in fp16; losses stay in float32
            # because masked_fill(-1e9) overflows fp16 (max ~65504).
            with torch.amp.autocast(device_type=device_type, enabled=use_amp):
                h1 = encoder(data1)
                h2 = encoder(data2)
                z1 = projector(h1)
                z2 = projector(h2)

            z1 = z1.float()
            z2 = z2.float()

            ntxent = nt_xent_loss(z1, z2)

            # Supervised contrastive term: both views share the same label.
            # Concatenating them doubles the effective batch so every
            # same-class pair (across and within views) acts as a positive.
            if supcon_weight > 0.0:
                z_all      = torch.cat([z1, z2], dim=0)
                labels_all = torch.cat([labels, labels], dim=0)
                sc         = sup_con_loss(z_all, labels_all, temperature=supcon_temperature)
            else:
                sc = torch.tensor(0.0, device=device)

            loss = ntxent + supcon_weight * sc

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss   += loss.item()
            total_ntxent += ntxent.item()
            total_supcon  += sc.item()

        n_batches = (len(networks) + batch_size - 1) // batch_size
        avg_loss   = total_loss   / n_batches
        avg_ntxent = total_ntxent / n_batches
        avg_supcon = total_supcon  / n_batches
        epoch_time = time.time() - epoch_start

        print(f"Epoch {epoch + 1}: loss={avg_loss:.4f} "
              f"(nt_xent={avg_ntxent:.4f}, supcon={avg_supcon:.4f}) "
              f"| time={epoch_time:.2f}s")

        # ✅ Save checkpoint every N epochs
        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch + 1}.pt")
            torch.save({
                'encoder_state_dict': encoder.state_dict(),
                'projector_state_dict': projector.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch + 1,
                'loss': avg_loss
            }, checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")

        # ✅ Track best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_encoder_state = encoder.state_dict()
            best_projector_state = projector.state_dict()
            best_epoch = epoch + 1
            print(f"New best model at epoch {epoch + 1} with loss {best_loss:.4f}")

    # 🔥 Save best model at the end
    best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
    torch.save({
        'encoder_state_dict': best_encoder_state,
        'projector_state_dict': best_projector_state,
        'loss': best_loss
    }, best_model_path)

    aug_pool.shutdown(wait=False)
    total_time = time.time() - start_time

    print(f"Best model saved at {best_model_path} with loss {best_loss:.4f}")
    print(f"Total training time: {total_time:.2f}s")
