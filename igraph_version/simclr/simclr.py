import copy
import math
import os
import sys
import random
import numpy as np
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
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
import time

import torch.nn as nn
import torch.nn.functional as F

from augmentation import (augment_network_view_fast, augment_network_view_smart,
                          build_igraph_from_transactions)
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
    _TARGET_DIM = 19
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
    """3-layer GCN with BatchNorm and concatenated mean+max graph pooling.

    n_layers / use_bn / hidden_dim are stored so load_simclr_encoder can
    reconstruct the exact architecture from the checkpoint's state_dict keys.
    """
    def __init__(self, in_dim, hidden_dim=128, out_dim=128, n_layers=3, use_bn=True):
        super().__init__()
        self.n_layers = n_layers
        self.use_bn   = use_bn

        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        if n_layers >= 3:
            self.conv3 = GCNConv(hidden_dim, hidden_dim)

        if use_bn:
            self.bn1 = nn.BatchNorm1d(hidden_dim)
            self.bn2 = nn.BatchNorm1d(hidden_dim)
            if n_layers >= 3:
                self.bn3 = nn.BatchNorm1d(hidden_dim)

        # Mean + max pooling concatenated → 2× hidden_dim input to the linear
        pool_in = hidden_dim * 2 if n_layers >= 3 else hidden_dim
        self.lin = nn.Linear(pool_in, out_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        if self.use_bn:
            x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        if self.use_bn:
            x = self.bn2(x)
        x = F.relu(x)

        if self.n_layers >= 3:
            x = self.conv3(x, edge_index)
            if self.use_bn:
                x = self.bn3(x)
            x = F.relu(x)
            h = torch.cat([global_mean_pool(x, batch),
                           global_max_pool(x, batch)], dim=-1)
        else:
            h = global_mean_pool(x, batch)

        return self.lin(h)


class ProjectionHead(nn.Module):
    def __init__(self, in_dim, proj_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.BatchNorm1d(in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, proj_dim),
        )

    def forward(self, z):
        return self.net(z)


class OnlineProbeHead(nn.Module):
    """Shallow classification head trained jointly with the encoder (Option 2).

    Gradients flow back through the encoder, directly aligning the learned
    representation with the downstream binary classification task.  Keeping
    the head shallow (2 layers) ensures the encoder — not the probe — learns
    the discriminative structure.
    """
    def __init__(self, in_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, h):
        return self.net(h).squeeze(-1)



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
# Connectivity check
# ─────────────────────────────────────────────────────────────────────────────

def _is_connected_ei(edge_index, n_nodes):
    """Return True iff the undirected graph described by edge_index is connected.

    Uses iterative DFS — O(V+E), no external dependencies.
    edge_index must be a 2×E int tensor (both (i,j) and (j,i) rows expected).
    """
    if n_nodes <= 1:
        return True
    if edge_index.shape[1] == 0:
        return False
    adj = [[] for _ in range(n_nodes)]
    for s, d in edge_index.T.tolist():
        if 0 <= s < n_nodes and 0 <= d < n_nodes:
            adj[s].append(d)
    visited = {0}
    stack = [0]
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v not in visited:
                visited.add(v)
                stack.append(v)
    return len(visited) == n_nodes


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
    adj_cpu   = adj_t[0, :n, :n].cpu()
    adj_cpu.fill_diagonal_(0.0)
    deg       = adj_cpu.sum(dim=-1)
    x0[:, 1]  = deg / deg.max().clamp(min=1.0)
    ei = (adj_cpu > 0.5).nonzero(as_tuple=False).T.contiguous()
    if ei.shape[1] > 0:
        ei = ei[:, ei[0] != ei[1]]
    # Strip col 0 (laundering flag) so diffusion views have the same input dim
    # as structural views produced by network_to_pyg_data_fast.
    d  = Data(x=x0[:, 1:], edge_index=ei)
    if label is not None:
        d.y = torch.tensor([label], dtype=torch.long)
    return d


# ─────────────────────────────────────────────────────────────────────────────
# Multi-step DDIM view
# ─────────────────────────────────────────────────────────────────────────────

def _diffusion_view_multistep(
    network_or_data6, diff_model, diffusion, x_mean, x_std, device,
    t_start_frac=0.5, n_steps=15, max_nodes=300, is_pyg=False,
):
    """
    Multi-step DDIM view.

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
# Guided multi-step DDIM view
# ─────────────────────────────────────────────────────────────────────────────

def _diffusion_view_guided(
    network_or_data6, diff_model, diffusion, x_mean, x_std,
    encoder, probe, device,
    t_start_frac=0.5, n_steps=15, guidance_scale=1.5,
    guide_from_frac=0.4, guide_every=3,
    max_nodes=300, is_pyg=False,
):
    """
    Class-conditional guided multi-step DDIM view.

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
                if ei_g.shape[1] > 0:
                    ei_g = ei_g[:, ei_g[0] != ei_g[1]]
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


def compute_node_saliency(encoder, probe_head, networks, device, max_nets=300):
    """
    Compute per-node importance from probe-gradient magnitude.

    For each (sampled) network, runs a single forward+backward pass through the
    frozen encoder and probe.  The gradient of the probe loss w.r.t. the node
    feature matrix  ∂L/∂x  measures how much each node's features influence the
    prediction — high-magnitude gradients identify discriminatively important
    nodes that should be protected during augmentation.

    Returns a list aligned with `networks`, where each entry is either:
      - dict {account_name (int) -> importance_score [0,1]}, or
      - None for networks not sampled this round.
    """
    was_training = encoder.training
    encoder.eval()

    importances = [None] * len(networks)

    # Subsample for efficiency — full set would be too slow every 20 epochs
    indices = list(range(len(networks)))
    random.shuffle(indices)
    indices = indices[:max_nets]

    with torch.no_grad():
        pass  # ensure params are unfrozen below (we only backprop through x)

    for idx in indices:
        net = networks[idx]
        try:
            data = network_to_pyg_data_fast(net).to(device)
            n    = data.x.shape[0]
            if n == 0:
                continue

            x     = data.x.detach().clone().requires_grad_(True)
            batch = torch.zeros(n, dtype=torch.long, device=device)
            tmp   = Data(x=x, edge_index=data.edge_index, batch=batch)

            # Forward through encoder (BN uses running stats in eval mode)
            h = encoder(tmp)
            score = torch.sigmoid(probe_head(F.normalize(h, dim=-1))).squeeze()
            label = float(len(net["laundering_nodes"]) > 0)
            loss  = (-torch.log(score + 1e-8) if label == 1.0
                     else -torch.log(1.0 - score + 1e-8))
            loss.backward()

            imp = x.grad.detach().abs().sum(dim=-1).cpu()  # [n_nodes]
            if imp.max() > 0:
                imp = imp / imp.max()

            # Key by node name so importance survives subgraph cropping
            has_names = "name" in net["graph"].vs.attributes()
            name_imp  = {}
            for vid in range(n):
                name = (int(net["graph"].vs[vid]["name"]) if has_names
                        else vid)
                name_imp[name] = float(imp[vid])

            importances[idx] = name_imp
        except Exception:
            pass

    if was_training:
        encoder.train()

    return importances


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

    # Build edge_index from thresholded adjacency (no self-loops)
    edge_index = (adj_node > 0.5).nonzero(as_tuple=False).T.contiguous()  # [2, E]
    if edge_index.shape[1] > 0:
        edge_index = edge_index[:, edge_index[0] != edge_index[1]]

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
    probe_weight=0.0,          # Option 2: weight on the online probe classification loss
    warmup_epochs=10,          # linear LR warmup before cosine decay
    use_cosine_schedule=True,  # cosine annealing after warmup
    # ── task-aware augmentation ───────────────────────────────────────────────
    use_curriculum=True,         # ramp aug strength from mild → full
    curriculum_epochs=None,      # epochs to reach full strength (default: epochs//2)
    use_motif_preserving=True,   # betweenness-biased edge dropping
    use_saliency=True,           # probe-gradient node importance
    saliency_update_interval=20, # recompute saliency every N epochs
):
    encoder.train()
    projector.train()

    os.makedirs(checkpoint_dir, exist_ok=True)

    use_amp = (device.type == "cuda") if hasattr(device, "type") else (str(device) != "cpu")
    scaler  = torch.amp.GradScaler("cuda", enabled=use_amp)
    all_params = list(encoder.parameters()) + list(projector.parameters())

    # Option 2: online probe head — trained jointly so gradients reach the encoder
    online_probe = None
    if probe_weight > 0.0:
        online_probe = OnlineProbeHead(in_dim=128).to(device)
        optimizer.add_param_group({"params": list(online_probe.parameters())})
        all_params = all_params + list(online_probe.parameters())

    best_loss = float('inf')
    best_encoder_state   = None
    best_projector_state = None
    best_probe_state     = None
    best_epoch = None

    if use_cosine_schedule:
        def _lr_lambda(ep):
            if ep < warmup_epochs:
                return (ep + 1) / max(warmup_epochs, 1)
            progress = (ep - warmup_epochs) / max(1, epochs - warmup_epochs)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)
    else:
        lr_scheduler = None

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

    # ── task-aware augmentation state ────────────────────────────────────────
    _curriculum_epochs  = curriculum_epochs if curriculum_epochs is not None else max(1, epochs // 2)
    _node_importances   = None   # list[dict|None] populated by compute_node_saliency

    start_time = time.time()
    total_batches = 0

    for epoch in range(epochs):
        epoch_start = time.time()
        encoder.train(); projector.train()
        if diffusion_model is not None:
            diffusion_model.eval()

        # ── curriculum factor: 0 (mild) → 1 (full) over _curriculum_epochs ─────
        curriculum_factor = (min(1.0, epoch / _curriculum_epochs)
                             if use_curriculum else 1.0)

        # ── refresh saliency scores periodically (requires online probe) ──────
        if (use_saliency and online_probe is not None
                and epoch % saliency_update_interval == 0):
            print(f"  [saliency] updating node importance scores …")
            _node_importances = compute_node_saliency(
                encoder, online_probe, networks, device, max_nets=300
            )

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
        total_loss = total_ntxent = total_supcon = total_probe = 0.0

        for i in range(0, len(networks), batch_size):
            total_batches += 1
            batch_nets  = networks[i:i + batch_size]
            batch_idxs  = list(range(i, min(i + batch_size, len(networks))))

            # Build view1 augmentations in parallel (igraph releases GIL)
            # Capture loop-local state via default args to avoid late-binding.
            def _make_view1(args,
                            _cf=curriculum_factor,
                            _ni=_node_importances,
                            _mp=use_motif_preserving):
                idx, net = args
                ni  = _ni[idx] if _ni is not None else None
                aug = augment_network_view_smart(
                    net, aug_strength=_cf, node_importance=ni,
                    use_motif_preserving=_mp,
                )
                return network_to_pyg_data_fast(aug)

            n_workers = min(len(batch_nets), 4)
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                views1 = list(pool.map(_make_view1, zip(batch_idxs, batch_nets)))

            views2 = []
            for k, net in enumerate(batch_nets):
                global_idx = batch_idxs[k]
                ni = _node_importances[global_idx] if _node_importances is not None else None

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
                    # Reject disconnected diffusion views; fall back to structural
                    # augmentation so the encoder never learns from disconnected graphs.
                    if (diff_view is not None
                            and _is_connected_ei(diff_view.edge_index,
                                                 diff_view.x.shape[0])):
                        views2.append(diff_view)
                        continue
                # Fallback: smart structural augmentation
                v2 = augment_network_view_smart(
                    net, aug_strength=curriculum_factor, node_importance=ni,
                    use_motif_preserving=use_motif_preserving,
                )
                views2.append(network_to_pyg_data_fast(v2))

            # Binary labels for the batch (1 = laundering present, 0 = clean)
            labels = torch.tensor(
                [int(len(net["laundering_nodes"]) > 0) for net in batch_nets],
                dtype=torch.long, device=device,
            )

            data1 = Batch.from_data_list(views1).to(device)
            data2 = Batch.from_data_list(views2).to(device)
            _ = batch_nets  # keep reference to avoid gc before here

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
                z_all      = torch.cat([z1, z2], dim=0)
                labels_all = torch.cat([labels, labels], dim=0)
                sc         = torch.tensor(0.0, device=device)

            # Option 2: online probe loss — applied to encoder outputs (128-dim),
            # not the projected embeddings (64-dim), so gradients shape the
            # representation that is actually used for downstream tasks.
            if online_probe is not None:
                h_all        = torch.cat([h1.float(), h2.float()], dim=0)
                probe_logits = online_probe(F.normalize(h_all, dim=-1))
                probe_loss   = F.binary_cross_entropy_with_logits(
                    probe_logits, labels_all.float()
                )
            else:
                probe_loss = torch.tensor(0.0, device=device)

            loss = ntxent + supcon_weight * sc + probe_weight * probe_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss   += loss.item()
            total_ntxent += ntxent.item()
            total_supcon  += sc.item()
            total_probe   += probe_loss.item()

        n_batches  = (len(networks) + batch_size - 1) // batch_size
        avg_loss   = total_loss   / n_batches
        avg_ntxent = total_ntxent / n_batches
        avg_supcon = total_supcon  / n_batches
        avg_probe  = total_probe  / n_batches
        epoch_time = time.time() - epoch_start

        if lr_scheduler is not None:
            lr_scheduler.step()

        probe_str = f", probe={avg_probe:.4f}" if online_probe is not None else ""
        lr_now = lr_scheduler.get_last_lr()[0] if lr_scheduler is not None else optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch + 1}: loss={avg_loss:.4f} "
              f"(nt_xent={avg_ntxent:.4f}, supcon={avg_supcon:.4f}{probe_str}) "
              f"| lr={lr_now:.2e} | time={epoch_time:.2f}s")

        # ✅ Save checkpoint every N epochs
        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch + 1}.pt")
            torch.save({
                'encoder_state_dict':    encoder.state_dict(),
                'projector_state_dict':  projector.state_dict(),
                'probe_state_dict':      online_probe.state_dict() if online_probe is not None else None,
                'optimizer_state_dict':  optimizer.state_dict(),
                'epoch': epoch + 1,
                'loss':  avg_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")

        # ✅ Track best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_encoder_state   = encoder.state_dict()
            best_projector_state = projector.state_dict()
            best_probe_state     = online_probe.state_dict() if online_probe is not None else None
            best_epoch = epoch + 1
            print(f"New best model at epoch {epoch + 1} with loss {best_loss:.4f}")

    # 🔥 Save best model at the end
    best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
    torch.save({
        'encoder_state_dict':   best_encoder_state,
        'projector_state_dict': best_projector_state,
        'probe_state_dict':     best_probe_state,
        'loss': best_loss,
    }, best_model_path)

    total_time = time.time() - start_time

    print(f"Best model saved at {best_model_path} with loss {best_loss:.4f}")
    print(f"Total training time: {total_time:.2f}s")


# ─────────────────────────────────────────────────────────────────────────────
# Population-Based Training
# ─────────────────────────────────────────────────────────────────────────────

def _train_worker_one_gen(
    networks, encoder, projector, online_probe, optimizer, device,
    gen_epochs, aug_config, batch_size=128,
    diffusion_model=None, diffusion=None, x_mean=None, x_std=None,
    max_nodes=300, node_importances=None,
):
    """
    Lightweight training loop for one PBT generation.

    Returns the average online-probe loss over the last epoch — lower is better,
    so this serves directly as the PBT fitness signal.
    """
    p_crop        = aug_config.get("p_crop",        0.6)
    p_edge_drop   = aug_config.get("p_edge_drop",   0.3)
    p_node_delete = aug_config.get("p_node_delete", 0.3)
    p_node_add    = aug_config.get("p_node_add",    0.2)
    p_diffusion   = aug_config.get("p_diffusion",   0.3)
    diff_t        = aug_config.get("diffusion_t_frac", 0.3)
    supcon_w      = aug_config.get("supcon_weight",  2.0)
    supcon_temp   = aug_config.get("supcon_temperature", 0.07)

    use_diff = (diffusion_model is not None and diffusion is not None
                and x_mean is not None and x_std is not None)

    encoder.train(); projector.train(); online_probe.train()
    if diffusion_model is not None:
        diffusion_model.eval()

    all_params = (list(encoder.parameters())
                  + list(projector.parameters())
                  + list(online_probe.parameters()))

    last_probe_loss = float("inf")
    n_batches_total = (len(networks) + batch_size - 1) // batch_size

    for ep in range(gen_epochs):
        ep_probe = 0.0

        for i in range(0, len(networks), batch_size):
            batch_nets = networks[i:i + batch_size]
            batch_idxs = list(range(i, min(i + batch_size, len(networks))))

            def _v1(args, _ni=node_importances,
                    _p_crop=p_crop, _p_ed=p_edge_drop,
                    _p_nd=p_node_delete, _p_na=p_node_add):
                idx, net = args
                ni = _ni[idx] if _ni else None
                aug = augment_network_view_smart(
                    net, aug_strength=1.0, node_importance=ni,
                    p_crop=_p_crop, p_edge_drop=_p_ed,
                    p_node_delete=_p_nd, p_node_add=_p_na,
                    use_motif_preserving=True,
                )
                return network_to_pyg_data_fast(aug)

            n_workers = min(len(batch_nets), 4)
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                views1 = list(pool.map(_v1, zip(batch_idxs, batch_nets)))

            views2 = []
            for k, net in enumerate(batch_nets):
                ni = node_importances[batch_idxs[k]] if node_importances else None
                if use_diff and random.random() < p_diffusion:
                    dv = _diffusion_view(net, diffusion_model, diffusion,
                                         x_mean, x_std, max_nodes, diff_t, device)
                    if dv is not None and _is_connected_ei(dv.edge_index, dv.x.shape[0]):
                        views2.append(dv)
                        continue
                v2 = augment_network_view_smart(
                    net, aug_strength=1.0, node_importance=ni,
                    p_crop=p_crop, p_edge_drop=p_edge_drop,
                    p_node_delete=p_node_delete, p_node_add=p_node_add,
                    use_motif_preserving=True,
                )
                views2.append(network_to_pyg_data_fast(v2))

            labels = torch.tensor(
                [int(len(net["laundering_nodes"]) > 0) for net in batch_nets],
                dtype=torch.long, device=device,
            )
            data1 = Batch.from_data_list(views1).to(device)
            data2 = Batch.from_data_list(views2).to(device)

            optimizer.zero_grad()
            h1 = encoder(data1)
            h2 = encoder(data2)
            z1 = projector(h1).float()
            z2 = projector(h2).float()

            ntxent     = nt_xent_loss(z1, z2)
            z_all      = torch.cat([z1, z2], dim=0)
            labels_all = torch.cat([labels, labels], dim=0)
            sc         = sup_con_loss(z_all, labels_all, temperature=supcon_temp)

            h_all      = torch.cat([h1.float(), h2.float()], dim=0)
            probe_loss = F.binary_cross_entropy_with_logits(
                online_probe(F.normalize(h_all, dim=-1)),
                labels_all.float(),
            )
            loss = ntxent + supcon_w * sc + 0.5 * probe_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            optimizer.step()
            ep_probe += probe_loss.item()

        last_probe_loss = ep_probe / n_batches_total

    return last_probe_loss


def train_simclr_pbt(
    networks,
    full_df,
    device,
    in_dim=10,
    base_config=None,
    n_workers=4,
    gen_epochs=15,
    n_generations=10,
    perturb_factor=0.25,
    batch_size=128,
    checkpoint_dir=None,
    diffusion_model=None,
    diffusion=None,
    x_mean=None,
    x_std=None,
    max_nodes=300,
):
    """
    Population-Based Training of augmentation hyperparameters.

    Maintains `n_workers` independent encoder-projector pairs, each training
    with a different augmentation configuration.  After every generation
    (gen_epochs epochs), the worst-performing worker (highest probe loss)
    inherits weights and a perturbed aug config from the best worker.

    This allows the system to discover aug configs that maximise downstream
    classification quality without manual hyperparameter tuning.

    Returns (best_encoder, best_projector, best_aug_config).
    """
    if base_config is None:
        base_config = {
            "p_crop":              0.6,
            "p_edge_drop":         0.3,
            "p_node_delete":       0.3,
            "p_node_add":          0.2,
            "p_diffusion":         0.3,
            "diffusion_t_frac":    0.3,
            "supcon_weight":       2.0,
            "supcon_temperature":  0.07,
        }

    _BOUNDS = {
        "p_crop":             (0.2, 0.9),
        "p_edge_drop":        (0.0, 0.7),
        "p_node_delete":      (0.0, 0.6),
        "p_node_add":         (0.0, 0.5),
        "p_diffusion":        (0.0, 0.7),
        "diffusion_t_frac":   (0.1, 0.7),
        "supcon_weight":      (0.5, 5.0),
        "supcon_temperature": (0.01, 0.5),
    }

    def _perturb(cfg):
        out = {}
        for k, v in cfg.items():
            lo, hi = _BOUNDS.get(k, (0.0, 10.0))
            noise  = v * perturb_factor * (2 * random.random() - 1)
            out[k] = float(np.clip(v + noise, lo, hi))
        return out

    # ── Initialise workers ────────────────────────────────────────────────────
    workers = []
    for k in range(n_workers):
        enc   = GraphEncoder(in_dim=in_dim, hidden_dim=128, out_dim=128,
                              n_layers=3, use_bn=True).to(device)
        proj  = ProjectionHead(in_dim=128, proj_dim=64).to(device)
        probe = OnlineProbeHead(in_dim=128).to(device)
        opt   = torch.optim.Adam(
            list(enc.parameters()) + list(proj.parameters()) + list(probe.parameters()),
            lr=3e-4, weight_decay=1e-4,
        )
        cfg   = _perturb(base_config) if k > 0 else dict(base_config)
        workers.append({"encoder": enc, "projector": proj, "probe": probe,
                         "optimizer": opt, "aug_config": cfg,
                         "fitness": float("inf")})

    best_enc_sd    = None
    best_proj_sd   = None
    best_probe_sd  = None
    best_cfg       = None
    best_fitness   = float("inf")

    # Saliency is shared (computed from the current best worker)
    node_importances = None

    for gen in range(n_generations):
        print(f"\n[PBT] Generation {gen + 1}/{n_generations}")

        # Refresh saliency from the current best worker every 2 generations
        if gen % 2 == 0 and workers[0]["fitness"] < float("inf"):
            best_w = min(workers, key=lambda w: w["fitness"])
            best_w["encoder"].eval()
            node_importances = compute_node_saliency(
                best_w["encoder"], best_w["probe"], networks, device, max_nets=200
            )
            best_w["encoder"].train()

        for widx, w in enumerate(workers):
            cfg = w["aug_config"]
            print(f"  Worker {widx + 1}: "
                  f"p_crop={cfg['p_crop']:.2f}  p_edge={cfg['p_edge_drop']:.2f}  "
                  f"p_diff={cfg['p_diffusion']:.2f}  supcon={cfg['supcon_weight']:.2f}")

            fitness = _train_worker_one_gen(
                networks=networks,
                encoder=w["encoder"], projector=w["projector"],
                online_probe=w["probe"], optimizer=w["optimizer"],
                device=device, gen_epochs=gen_epochs, aug_config=cfg,
                batch_size=batch_size,
                diffusion_model=diffusion_model, diffusion=diffusion,
                x_mean=x_mean, x_std=x_std, max_nodes=max_nodes,
                node_importances=node_importances,
            )
            w["fitness"] = fitness
            print(f"    → probe_loss={fitness:.4f}")

        workers.sort(key=lambda w: w["fitness"])
        ranking = [f"{w['fitness']:.4f}" for w in workers]
        print(f"  Ranking: {ranking}")

        if workers[0]["fitness"] < best_fitness:
            best_fitness  = workers[0]["fitness"]
            best_enc_sd   = copy.deepcopy(workers[0]["encoder"].state_dict())
            best_proj_sd  = copy.deepcopy(workers[0]["projector"].state_dict())
            best_probe_sd = copy.deepcopy(workers[0]["probe"].state_dict())
            best_cfg      = dict(workers[0]["aug_config"])
            print(f"  New best  fitness={best_fitness:.4f}  cfg={best_cfg}")

        # Evolve: bottom 25% inherits from best worker + perturbation
        n_replace = max(1, n_workers // 4)
        src = workers[0]
        for i in range(n_workers - n_replace, n_workers):
            workers[i]["encoder"].load_state_dict(copy.deepcopy(src["encoder"].state_dict()))
            workers[i]["projector"].load_state_dict(copy.deepcopy(src["projector"].state_dict()))
            workers[i]["probe"].load_state_dict(copy.deepcopy(src["probe"].state_dict()))
            workers[i]["optimizer"] = torch.optim.Adam(
                list(workers[i]["encoder"].parameters())
                + list(workers[i]["projector"].parameters())
                + list(workers[i]["probe"].parameters()),
                lr=3e-4, weight_decay=1e-4,
            )
            workers[i]["aug_config"] = _perturb(src["aug_config"])

    print(f"\n[PBT] Done.  Best fitness={best_fitness:.4f}  Best cfg={best_cfg}")

    # Load best weights back into worker 0 and return
    workers[0]["encoder"].load_state_dict(best_enc_sd)
    workers[0]["projector"].load_state_dict(best_proj_sd)
    workers[0]["probe"].load_state_dict(best_probe_sd)

    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
        path = os.path.join(checkpoint_dir, "pbt_best.pt")
        torch.save({
            "encoder_state_dict":   best_enc_sd,
            "projector_state_dict": best_proj_sd,
            "probe_state_dict":     best_probe_sd,
            "aug_config": best_cfg,
            "fitness":    best_fitness,
        }, path)
        print(f"[PBT] Saved best model → {path}")

    return workers[0]["encoder"], workers[0]["projector"], best_cfg
