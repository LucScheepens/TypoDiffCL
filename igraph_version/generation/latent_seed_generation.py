"""
latent_seed_generation.py — Structurally novel generation via latent extrapolation
                            + structural repulsion.

Why the previous version wasn't novel enough
────────────────────────────────────────────
Interpolation (a·z_laund + (1-a)·z_clean) places the seed TARGET at the
*boundary* of the training distribution — it is still surrounded by real
training networks.  Any output near that point will look like a blend of
existing examples rather than something genuinely new.

Additionally, the only novelty signal during generation was cosine-similarity
repulsion in the 128-D SimCLR embedding space.  That encoder uses global mean
pooling, which is highly lossy: many structurally different graphs map to
nearby embeddings.  "Novel in embedding space" does not mean "different
structure".

Two-part fix
────────────
1. EXTRAPOLATION instead of interpolation.
   Push the target PAST the laundering cluster, into space that no training
   network occupies:

       z_extrap = normalise( z_laund + y · (z_laund - z_clean) )
                = normalise( (1+y)·z_laund - y·z_clean )

   With y=0.5 this goes halfway beyond the furthest laundering network in the
   direction away from the clean cluster.  The physical seed is still a real
   training network (the one closest to z_extrap), but the *target* is outside
   the training distribution.

2. STRUCTURAL REPULSION during generation.
   At each guided denoising step, in addition to the embedding-space novelty
   term, a structural repulsion loss is computed directly from the predicted
   adjacency:

       • degree statistics [mean, std, max, density] are computed from the
         soft adjacency output adj_pred  (differentiable, no igraph needed)
       • for each of the K nearest training networks (by embedding), the same
         statistics are compared via cosine similarity
       • the loss PENALISES structural similarity, so the gradient explicitly
         pushes the degree distribution away from those of the K neighbours

   This operates at the level of actual graph topology, not a compressed
   embedding, so it guarantees structural divergence rather than just
   embedding-space divergence.

Usage:
    python latent_seed_generation.py

Outputs (in results/latent_seed/):
    latent_seed_umap.png
    latent_seed_generated.csv
    latent_seed_calibration.csv
"""

import sys
import pickle
import random as _random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import umap
from tqdm.auto import tqdm

# ── path setup ─────────────────────────────────────────────────────────────────
_GEN_DIR   = Path(__file__).resolve().parent   # igraph_version/generation/
ROOT_DIR   = _GEN_DIR.parent                   # igraph_version/
DIFF_DIR   = ROOT_DIR / "diffusion"
SIMCLR_DIR = ROOT_DIR / "simclr"
CKPT_DIR   = ROOT_DIR / "checkpoints"
DATA_DIR   = ROOT_DIR / "data"

for _p in (str(ROOT_DIR), str(DIFF_DIR), str(SIMCLR_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from util import (
    preprocess_df,
    extract_networks_igraph,
)
from augmentation import build_igraph_from_transactions
from generation.generation import (
    load_simclr_encoder,
    load_diffusion_model,
    encode_all_networks,
    train_mlp_probe,
    _to_pyg,
)
from scoring import (
    fit_training_distribution,
    score_network,
    _print_scores_table,
    _save_scores_csv,
)

# ── config ─────────────────────────────────────────────────────────────────────
CSV_PATH        = r"C:\Users\lucsc\Thesis\grad\grad\data\IBM\HI-Small_Trans.csv"
TARGET          = 1
N_GEN           = 8
T_START         = 250     # higher than interpolation version — target is further away
GUIDE_SCALE     = 3.0
NOVELTY_WEIGHT  = 1.5     # embedding-space repulsion (reduced — structural takes over)
STRUCT_WEIGHT   = 2.0     # structural repulsion weight
GUIDE_EVERY     = 1
GUIDE_FROM      = 0.25
EXTRAP_GAMMA    = 0.5     # how far past the laundering cluster to push the target
N_CALIB         = 5
MAX_NODES       = 64


# ─────────────────────────────────────────────────────────────────────────────
# Structural helpers
# ─────────────────────────────────────────────────────────────────────────────

def _degree_stats_tensor(adj, n):
    """
    Differentiable structural statistics from a soft adjacency matrix.

    adj : Tensor [N, N]  (may be soft/probabilistic, only [:n, :n] is used)
    n   : int            number of real nodes

    Returns Tensor [4]: [mean_degree, std_degree, max_degree, edge_density]
    All values computed from the soft degree sequence so gradients flow through.
    """
    deg  = adj[:n, :n].sum(dim=-1)          # [n]  soft degree per node
    n_f  = float(max(n, 1))
    mean = deg.mean()
    std  = deg.std() + 1e-8
    mx   = deg.max()
    dens = deg.sum() / max(n_f * (n_f - 1), 1.0)
    return torch.stack([mean, std, mx, dens])   # [4]


def precompute_train_degree_stats(networks):
    """
    Compute degree statistics for every training network once, on CPU.

    Returns Tensor [N, 4].
    Each row: [mean_deg, std_deg, max_deg, density] for that network's adjacency.
    """
    from diffusion.diff_util import network_to_dense as _ntd
    stats = []
    for net in networks:
        adj = net["adj_dense"] if "adj_dense" in net else _ntd(net)[1]
        n   = adj.shape[0]
        with torch.no_grad():
            s = _degree_stats_tensor(adj.float(), n)
        stats.append(s.cpu())
    return torch.stack(stats, dim=0)   # [N, 4]


# ─────────────────────────────────────────────────────────────────────────────
# Seed selection — extrapolation
# ─────────────────────────────────────────────────────────────────────────────

def find_extrapolated_seeds(H_all_n, y_all, gamma=0.5):
    """
    For every laundering network find its nearest clean neighbour, then
    extrapolate PAST the laundering network in the direction away from clean:

        z_extrap = normalise( z_laund + γ · (z_laund − z_clean) )
                 = normalise( (1+γ)·z_laund − γ·z_clean )

    Unlike interpolation, which places the target at the boundary of the
    training distribution, extrapolation places it OUTSIDE the distribution —
    in space that no training network occupies.

    Returns list of dicts sorted by ascending nearest_sim (most novel first):
        laund_idx   : int
        clean_idx   : int     nearest clean neighbour of this laundering network
        z_extrap    : Tensor [128]  normalised extrapolated target embedding
        nearest_sim : float   cos-sim of z_extrap to its closest training network
    """
    laund_idxs = (y_all == 1).nonzero(as_tuple=True)[0]
    clean_idxs = (y_all == 0).nonzero(as_tuple=True)[0]

    H_laund = H_all_n[laund_idxs]
    H_clean = H_all_n[clean_idxs]

    cos_sim_lc          = H_laund @ H_clean.T          # [N_laund, N_clean]
    nearest_clean_local = cos_sim_lc.argmax(dim=1)     # [N_laund]

    seeds = []
    for li, ci_local in enumerate(nearest_clean_local):
        laund_idx = int(laund_idxs[li])
        clean_idx = int(clean_idxs[int(ci_local)])

        z_laund  = H_all_n[laund_idx]
        z_clean  = H_all_n[clean_idx]

        # Extrapolate: go gamma steps PAST z_laund in direction away from z_clean
        z_extrap = F.normalize(
            (1.0 + gamma) * z_laund - gamma * z_clean,
            dim=0,
        )

        # How close is the extrapolated point to any training network?
        nearest_sim = float((H_all_n @ z_extrap).max())

        seeds.append({
            "laund_idx":   laund_idx,
            "clean_idx":   clean_idx,
            "z_extrap":    z_extrap,
            "nearest_sim": nearest_sim,
        })

    seeds.sort(key=lambda s: s["nearest_sim"])   # most novel first
    return seeds


def select_diverse_seeds(seed_list, n_seeds):
    """
    Greedy farthest-point sampling over z_extrap embeddings.
    Ensures the selected seeds cover different regions of the extrapolated space
    rather than all collapsing to the same laundering neighbourhood.
    """
    if len(seed_list) <= n_seeds:
        return list(seed_list)

    selected  = [seed_list[0]]
    remaining = list(seed_list[1:])

    while len(selected) < n_seeds and remaining:
        selected_z  = torch.stack([s["z_extrap"] for s in selected])   # [k, 128]
        best_idx    = 0
        best_max_sim = float("inf")

        for i, s in enumerate(remaining):
            # Maximum cosine similarity to any already-selected seed
            max_sim = float((selected_z @ s["z_extrap"]).max())
            if max_sim < best_max_sim:
                best_max_sim = max_sim
                best_idx     = i

        selected.append(remaining.pop(best_idx))

    return selected


def find_nearest_network(z_query, H_all_n):
    """Return (index, cosine_sim) of the training network nearest to z_query."""
    z_q  = F.normalize(z_query.float(), dim=0)
    sims = H_all_n @ z_q
    idx  = int(sims.argmax())
    return idx, float(sims[idx])


# ─────────────────────────────────────────────────────────────────────────────
# Generation with structural repulsion
# ─────────────────────────────────────────────────────────────────────────────

def guided_generate_structurally_novel(
    seed_network,
    encoder,
    probe,
    diff_model,
    diffusion,
    x_mean,
    x_std,
    H_train,
    train_degree_stats,
    device,
    target_label=1,
    t_start=300,
    guidance_scale=2.0,
    novelty_weight=1.5,
    struct_weight=2.0,
    novelty_k=10,
    guide_every=5,
    guide_from=0.25,
    degree_penalty=0.5,
    target_mean_degree=None,
    adj_threshold=0.9,
    adj_gamma=2.5,
    target_density=None,
    n_override=None,
    pbar=None,
):
    """
    Guided reverse diffusion with TWO novelty objectives:

      1. Embedding-space repulsion  (same as generation.py guided_generate)
         — pushes the encoder embedding away from the K nearest training
           network embeddings.

      2. Structural repulsion  (NEW)
         — at each guided step, computes degree statistics directly from the
           predicted soft adjacency adj_pred, then penalises cosine similarity
           between those stats and the degree stats of the K nearest training
           networks.
         — operates on raw graph structure, not a compressed embedding, so it
           guarantees structural divergence rather than embedding divergence.

    The two terms are complementary:
      • Embedding repulsion steers the high-level "type" of network away from
        training examples.
      • Structural repulsion steers the low-level topology (degree sequence,
        density) away from the specific neighbours found by embedding lookup.

    Returns (x_out, adj_out, n_nodes) on CPU.
    """
    from diffusion.diff_util import network_to_dense
    from torch_geometric.data import Data

    if "x_dense" in seed_network and "adj_dense" in seed_network:
        x   = seed_network["x_dense"].to(device)
        adj = seed_network["adj_dense"].to(device)
    else:
        x, adj = network_to_dense(seed_network)
        x, adj = x.to(device), adj.to(device)
    seed_n = x.shape[0]
    # n_override sets the candidate pool size (how many positions the model can
    # activate/deactivate). Seed nodes fill [0:seed_n]; the rest are blank
    # "candidate" slots the model may choose to activate.
    n_pool = n_override if (n_override is not None) else seed_n
    n_pool = max(seed_n, min(n_pool, MAX_NODES))   # always at least seed size

    x_pad   = torch.zeros(1, MAX_NODES, 6,         device=device)
    adj_pad = torch.zeros(1, MAX_NODES, MAX_NODES,  device=device)
    mask    = torch.zeros(1, MAX_NODES,             device=device)
    x_pad[0, :seed_n]             = x
    adj_pad[0, :seed_n, :seed_n]  = adj
    mask[0, :n_pool]              = 1.0   # full candidate pool starts as active

    x_norm = x_pad.clone()
    x_norm[:, :, 1:] = (x_pad[:, :, 1:] - x_mean[1:]) / x_std[1:]
    x_norm  = x_norm  * mask.unsqueeze(-1)
    adj_pad = adj_pad * mask[:, :, None] * mask[:, None, :]

    t_tensor   = torch.tensor([t_start], device=device)
    x_t, adj_t = diffusion.q_sample(x_norm, t_tensor, node_mask=mask, adj_start=adj_pad)

    guide_threshold = int(guide_from * t_start)
    cached_grad     = None
    cached_grad_adj = None
    H_dev           = H_train.to(device) if H_train is not None else None
    deg_stats_dev   = train_degree_stats.to(device) if train_degree_stats is not None else None

    for step_i, t_curr in enumerate(range(t_start, -1, -1)):
        t_vec    = torch.tensor([t_curr], device=device)
        t_scaled = diffusion._scale_timesteps(t_vec)
        eff_guide_every = 1 if t_curr < 100 else guide_every
        do_guide = (t_curr < guide_threshold) and (step_i % eff_guide_every == 0)

        if do_guide:
            with torch.enable_grad():
                x_t_g   = x_t.detach().requires_grad_(True)
                adj_t_g = adj_t.detach().requires_grad_(True)

                eps_pred, adj_pred, node_logits = diff_model(x_t_g, t_scaled,
                                                             adj=adj_t_g, node_mask=mask)
                x0_cont = diffusion._predict_xstart_from_eps(
                              x_t_g[..., 1:], t_vec, eps_pred[..., 1:])
                x0_bin  = eps_pred[..., 0:1].clamp(0, 1)
                x0_pred = torch.cat([x0_bin, x0_cont], dim=-1)

                pyg = _to_pyg(x0_pred[0, :n_pool], adj_pred[0, :n_pool, :n_pool],
                              n_pool, device, x_mean, x_std)
                h   = encoder(pyg)
                h_n = F.normalize(h, dim=-1)           # [1, 128]

                # ── 1. Classification loss ───────────────────────────────────
                score  = torch.sigmoid(probe(h_n)).squeeze()
                g_loss = (-torch.log(score + 1e-8) if target_label == 1
                          else -torch.log(1 - score + 1e-8))

                # ── 2. Embedding-space novelty repulsion ─────────────────────
                # Ramped: zero at high-t (structural phase), full at low-t
                t_frac      = t_curr / max(t_start, 1)
                eff_novelty = novelty_weight * (1.0 - t_frac)
                top_k_idx   = None
                if H_dev is not None and eff_novelty > 0.0:
                    cos_sims    = (H_dev @ h_n.T).squeeze()
                    topk_result = torch.topk(cos_sims, min(novelty_k, len(H_dev)))
                    top_k_idx   = topk_result.indices
                    g_loss      = g_loss + eff_novelty * topk_result.values.mean()

                # ── 3. Structural repulsion ──────────────────────────────────
                # Uses the PREDICTED adjacency (adj_pred), not the noisy one.
                # Degree stats are cheap to compute and directly differentiable.
                # Penalise cosine similarity between generated degree stats and
                # those of the K nearest training neighbours found above.
                if (struct_weight > 0.0
                        and deg_stats_dev is not None
                        and top_k_idx is not None):
                    gen_stats = _degree_stats_tensor(adj_pred[0], n_pool)   # [4]
                    struct_sim = torch.tensor(0.0, device=device)
                    for idx in top_k_idx[:5]:   # top-5 structural neighbours
                        near_stats = deg_stats_dev[idx]                # [4]
                        struct_sim = struct_sim + F.cosine_similarity(
                            gen_stats.unsqueeze(0), near_stats.unsqueeze(0)
                        ).squeeze()
                    g_loss = g_loss + struct_weight * (struct_sim / 5.0)

                # ── 4. Degree penalty ────────────────────────────────────────
                # Penalises excess degree above the training distribution target.
                # One-sided squared loss: only fires when generated mean degree
                # exceeds the training mean, pulling toward realistic density.
                if degree_penalty > 0.0:
                    mean_deg = adj_pred[0, :n_pool, :n_pool].sum(dim=-1).mean()
                    if target_mean_degree is not None:
                        excess = torch.relu(mean_deg - target_mean_degree)
                        g_loss = g_loss + degree_penalty * excess ** 2
                    else:
                        g_loss = g_loss + degree_penalty * mean_deg

                grads           = torch.autograd.grad(g_loss, [x_t_g, adj_t_g])
                cached_grad     = grads[0].detach().clamp(-1.0, 1.0)
                cached_grad_adj = grads[1].detach().clamp(-1.0, 1.0)
        else:
            with torch.no_grad():
                eps_pred, adj_pred, node_logits = diff_model(x_t, t_scaled,
                                                             adj=adj_t, node_mask=mask)
                x0_cont = diffusion._predict_xstart_from_eps(
                              x_t[..., 1:], t_vec, eps_pred[..., 1:])
                x0_bin  = eps_pred[..., 0:1].clamp(0, 1)
                x0_pred = torch.cat([x0_bin, x0_cont], dim=-1)

        with torch.no_grad():
            coef1    = float(diffusion.posterior_mean_coef1[t_curr])
            coef2    = float(diffusion.posterior_mean_coef2[t_curr])
            post_var = float(diffusion.posterior_variance[t_curr])
            post_lv  = float(diffusion.posterior_log_variance_clipped[t_curr])

            x0_d = torch.cat([x0_pred[..., 0:1].clamp(0, 1), x0_pred[..., 1:]], dim=-1).detach()
            x0_d = x0_d * mask.unsqueeze(-1)
            mean = coef1 * x0_d + coef2 * x_t.detach()

            if cached_grad is not None:
                mean = (mean - guidance_scale * post_var * cached_grad) * mask.unsqueeze(-1)

            noise = torch.randn_like(x_t) * mask.unsqueeze(-1)
            x_t   = (mean + (t_curr > 0) * np.exp(0.5 * post_lv) * noise) * mask.unsqueeze(-1)

            ap = adj_pred.detach().clamp(0, 1)
            if cached_grad_adj is not None:
                ap = (ap - guidance_scale * post_var * cached_grad_adj).clamp(0, 1)

            # Gamma compression: squash mid-range probabilities toward zero.
            ap = ap ** adj_gamma

            if t_curr > 0:
                adj_t = torch.bernoulli(ap)
            elif target_density is not None:
                # Density-calibrated threshold: select top-confidence edges until
                # the resulting density matches the training mean.
                n_active = int(mask[0].sum().item())
                n_keep   = max(1, round(target_density * n_active * (n_active - 1) / 2))
                ap_flat  = ap[0, :n_pool, :n_pool].reshape(-1)
                if n_keep * 2 < len(ap_flat):
                    kth   = ap_flat.kthvalue(len(ap_flat) - n_keep * 2).values.item()
                    adj_t = (ap > max(kth, 0.05)).float()
                else:
                    adj_t = (ap > adj_threshold).float()
            else:
                adj_t = (ap > adj_threshold).float()

            adj_t = adj_t * mask[:, :, None] * mask[:, None, :]

            # Per-step edge thinning: randomly drop edges with probability that
            # grows as we approach t=0, but never leave a node with degree < 1.
            if t_curr > 0:
                drop_prob = 0.05 * (1.0 - t_curr / t_start)
                keep_mask = (torch.rand_like(adj_t) > drop_prob).float()
                adj_thin  = adj_t * keep_mask
                # Restore edges for any node that would drop to in-degree 0
                deg_thin = adj_thin.sum(dim=-2)  # [B, N] in-degree
                isolated = (deg_thin < 1).float() * mask  # nodes that lost all incoming edges
                if isolated.any():
                    # For isolated nodes, restore their original adj_t column
                    restore = isolated.unsqueeze(-2) * adj_t  # [B, N, N]
                    adj_thin = (adj_thin + restore).clamp(max=1)
                adj_t = adj_thin

            # ── Node existence update ────────────────────────────────────────
            # Use the model's predicted node logits to evolve the working mask.
            # Only positions within the candidate pool (0..n_pool) are eligible.
            # The model learns from training which nodes "make sense" structurally.
            with torch.no_grad():
                node_probs = torch.sigmoid(node_logits[0, :n_pool])  # [n_pool]
                if t_curr > 0:
                    # Soft Bernoulli sample — stochastic during denoising
                    new_mask = torch.bernoulli(node_probs)
                else:
                    # Hard threshold at the final step
                    new_mask = (node_probs > 0.5).float()

                # Enforce minimum graph size (at least 3 active nodes)
                if new_mask.sum() < 3:
                    _, topk = node_probs.topk(3)
                    new_mask = torch.zeros_like(new_mask)
                    new_mask[topk] = 1.0

                mask = torch.zeros(1, MAX_NODES, device=device)
                mask[0, :n_pool] = new_mask
                # Re-zero features / adj for newly deactivated nodes
                x_t   = x_t   * mask.unsqueeze(-1)
                adj_t = adj_t * mask[:, :, None] * mask[:, None, :]

        if pbar is not None:
            pbar.update(1)

    # Compress output to active nodes only (mask may have gaps — keep order)
    active_idx = mask[0].nonzero(as_tuple=True)[0].cpu()
    n_final    = len(active_idx)
    x_out      = x_t[0][active_idx].cpu()
    adj_out    = adj_t[0][active_idx][:, active_idx].cpu()
    return x_out, adj_out, n_final


# ─────────────────────────────────────────────────────────────────────────────
# Generation orchestration
# ─────────────────────────────────────────────────────────────────────────────

def run_latent_seed_generation(
    networks,
    encoder,
    probe,
    diff_model,
    diffusion,
    x_mean,
    x_std,
    H_all_n,
    train_degree_stats,
    device,
    selected_seeds,
    target_label=1,
    t_start=300,
    guidance_scale=2.0,
    novelty_weight=1.5,
    struct_weight=2.0,
    guide_every=5,
    guide_from=0.25,
    degree_penalty=0.5,
    adj_threshold=0.5,
):
    """
    Generate one network per selected extrapolated seed.

    For each seed:
      1. Retrieve the physical seed: training network with embedding closest
         to the extrapolated target z_extrap.
      2. Run guided_generate_structurally_novel with structural repulsion.

    Returns
    -------
    gen_outputs    : list of (x_denorm, adj, n_nodes)
    gen_embeddings : np.ndarray [n_gen, 128]
    seed_networks  : list of seed network dicts used
    seed_indices   : list of int (index in `networks`)
    """
    from torch_geometric.data import Data
    import numpy as np

    # Derive target mean degree and density from precomputed training stats
    # column 0 = mean_deg,  column 3 = density
    target_mean_degree = float(train_degree_stats[:, 0].mean()) if train_degree_stats is not None else None
    target_density     = float(train_degree_stats[:, 3].mean()) if train_degree_stats is not None else None

    # Build a pool of training network sizes to sample from for size diversity
    train_sizes = [net["graph"].vcount() for net in networks if net["graph"].vcount() >= 3]

    gen_outputs, gen_embeddings, seed_networks, seed_indices = [], [], [], []

    total_steps = len(selected_seeds) * (t_start + 1)
    with tqdm(total=total_steps, desc="Structurally-novel generation",
              unit="step", dynamic_ncols=True) as pbar:

        for i, seed in enumerate(selected_seeds):
            pbar.set_postfix(seed=f"{i+1}/{len(selected_seeds)}")

            seed_idx, seed_sim = find_nearest_network(seed["z_extrap"], H_all_n)
            seed_net = networks[seed_idx]
            seed_networks.append(seed_net)
            seed_indices.append(seed_idx)

            # Sample a candidate pool size from the training distribution.
            # The model will decide which nodes in [0..n_pool] actually survive —
            # this gives the model room to add nodes beyond the seed's own size.
            n_pool = int(_random.choice(train_sizes)) if train_sizes else None

            x_out, adj_out, n_out = guided_generate_structurally_novel(
                seed_net, encoder, probe, diff_model, diffusion,
                x_mean, x_std, H_all_n, train_degree_stats, device,
                target_label=target_label,
                t_start=t_start,
                guidance_scale=guidance_scale,
                novelty_weight=novelty_weight,
                struct_weight=struct_weight,
                guide_every=guide_every,
                guide_from=guide_from,
                degree_penalty=degree_penalty,
                target_mean_degree=target_mean_degree,
                adj_threshold=adj_threshold,
                target_density=target_density,
                n_override=n_pool,
                pbar=pbar,
            )

            # Denormalise — col 0 (laundering flag) excluded to match
            # the encoder's input dimension after the label-leakage fix.
            x_cont_d = x_out[:, 1:] * x_std.cpu()[1:] + x_mean.cpu()[1:]
            deg_g    = adj_out.sum(dim=-1, keepdim=True)
            deg_n    = deg_g / deg_g.max().clamp(min=1.0)
            x_denorm = torch.cat([deg_n, x_cont_d[:, 1:]], dim=-1)   # [n, 5]

            gen_outputs.append((x_denorm, adj_out, n_out))

            ei_g = (adj_out > adj_threshold).nonzero(as_tuple=False).T.contiguous()
            bv_g = torch.zeros(n_out, dtype=torch.long)
            with torch.no_grad():
                h_g = encoder(Data(x=x_denorm, edge_index=ei_g,
                                   batch=bv_g).to(device)).cpu()
            gen_embeddings.append(F.normalize(h_g, dim=-1).squeeze(0).numpy())

    gen_embeddings = np.stack(gen_embeddings, axis=0)
    return gen_outputs, gen_embeddings, seed_networks, seed_indices


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def plot_latent_seeds_umap(
    H_all_n, all_labels, selected_seeds,
    seed_indices, gen_embeddings,
    target_label, t_start, gamma, save_path,
):
    """
    Four-layer UMAP:
      · Dots       — training networks (blue=clean, red=laundering)
      · Circles ○  — physical seed networks (nearest to z_extrap)
      · Diamonds ◆ — extrapolated target embeddings (outside training hull)
      · Stars ★    — generated networks
    Grey lines connect each physical seed to its extrapolated target.
    """
    n_train   = len(all_labels)
    z_extraps = torch.stack([s["z_extrap"] for s in selected_seeds]).numpy()

    H_combined = np.concatenate([H_all_n.numpy(), z_extraps, gen_embeddings], axis=0)
    reducer    = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)
    H_2d       = reducer.fit_transform(H_combined)

    n_seeds   = len(selected_seeds)
    train_2d  = H_2d[:n_train]
    extrap_2d = H_2d[n_train : n_train + n_seeds]
    gen_2d    = H_2d[n_train + n_seeds :]
    seed_2d   = train_2d[seed_indices]
    labels    = np.array(all_labels)

    fig, ax = plt.subplots(figsize=(10, 10))

    sc = ax.scatter(train_2d[:, 0], train_2d[:, 1],
                    c=labels, cmap="coolwarm",
                    s=18, alpha=0.30, zorder=1, label="Training networks")
    ax.scatter(seed_2d[:, 0], seed_2d[:, 1],
               s=120, facecolors="none", edgecolors="black", linewidths=1.5,
               zorder=3, label="Physical seeds")
    ax.scatter(extrap_2d[:, 0], extrap_2d[:, 1],
               marker="D", c="orange", s=100, edgecolors="black", linewidths=0.8,
               zorder=4, label=f"Extrapolated targets (γ={gamma})")

    for j in range(n_seeds):
        ax.plot([seed_2d[j, 0], extrap_2d[j, 0]],
                [seed_2d[j, 1], extrap_2d[j, 1]],
                color="grey", linewidth=0.7, alpha=0.5, zorder=2)

    gen_color = "red" if target_label == 1 else "blue"
    ax.scatter(gen_2d[:, 0], gen_2d[:, 1],
               marker="*", c=gen_color, s=220,
               edgecolors="black", linewidths=1.0,
               zorder=5,
               label=f"Generated ({'laundering' if target_label else 'clean'}, "
                     f"t_start={t_start})")

    cbar = fig.colorbar(sc, ax=ax, ticks=[0, 1])
    cbar.ax.set_yticklabels(["Clean", "Laundering"])
    ax.set_title(
        "Structurally-Novel Latent-Seed Generation\n"
        "◆ extrapolated target  ○ physical seed  ★ generated"
    )
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Per-network structural comparison plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_generated_vs_closest_training(
    gen_outputs,
    gen_embeddings,
    H_all_n,
    networks,
    results_dir,
    seed_networks=None,
):
    """
    For every generated network produce a four-panel comparison figure:

      Panel 1 — Generated network
      Panel 2 — Seed network used to initialise generation
                (falls back to closest-by-cosine if seed_networks is None)
      Panel 3 — Closest training network by cosine similarity to gen embedding
      Panel 4 — Grouped bar chart comparing structural statistics across all three

    Saves one file per generated network to:
        results/latent_seed/comparison/comparison_gen_{i+1:02d}.png
    """
    import networkx as nx
    from matplotlib.patches import Patch

    comp_dir = results_dir / "comparison"
    comp_dir.mkdir(exist_ok=True)

    H_np = H_all_n.numpy()

    # ── helper: build NetworkX graph + metadata from an igraph network dict ──
    def _build_nx(net):
        g        = net["graph"]
        laund    = net.get("laundering_nodes", set())
        coll     = net.get("collapsed_nodes",  set())
        has_names = "name" in g.vs.attributes()
        G = nx.Graph()
        for v in g.vs:
            G.add_node(v.index, name=(v["name"] if has_names else v.index))
        for e in g.es:
            G.add_edge(e.source, e.target)
        colors = []
        for v in g.vs:
            nm = v["name"] if has_names else v.index
            if nm in laund:   colors.append("#e74c3c")
            elif nm in coll:  colors.append("#aab7b8")
            else:             colors.append("#aed6f1")
        n_laund = sum(
            1 for v in g.vs
            if (v["name"] if has_names else v.index) in laund
        )
        return G, colors, g.vcount(), n_laund

    # ── helper: structural stats ──────────────────────────────────────────────
    def _stats(G, n):
        ne   = G.number_of_edges()
        dens = ne / max(n * (n - 1) / 2, 1)
        degs = [d for _, d in G.degree()]
        return {
            "nodes":    n,
            "edges":    ne,
            "density":  dens,
            "mean_deg": float(np.mean(degs)) if degs else 0.0,
            "degs":     degs,
        }

    # ── helper: draw one network panel ───────────────────────────────────────
    def _draw_panel(ax, G, pos, colors, title, stats_str):
        nx.draw_networkx_nodes(G, pos, ax=ax,
                               node_color=colors, node_size=80, alpha=0.90)
        nx.draw_networkx_edges(G, pos, ax=ax,
                               edge_color="#555555", width=0.8, alpha=0.45,
                               arrows=False)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.axis("off")
        ax.text(0.02, 0.02, stats_str,
                transform=ax.transAxes, fontsize=8.5,
                verticalalignment="bottom",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.85))

    # ─────────────────────────────────────────────────────────────────────────

    for i, (gen_out, gen_emb) in enumerate(
            zip(gen_outputs, gen_embeddings)):
        x_denorm, adj_out, n_out = gen_out[0], gen_out[1], gen_out[2]

        # ── Generated network ─────────────────────────────────────────────
        adj_np     = adj_out.cpu().numpy()
        laund_prob = x_denorm[:, 0].cpu().numpy()

        G_gen = nx.Graph()
        G_gen.add_nodes_from(range(n_out))
        for a in range(n_out):
            for b in range(a + 1, n_out):
                if adj_np[a, b] > 0.5:
                    G_gen.add_edge(a, b)
        gen_colors = [
            "#e74c3c" if laund_prob[k] > 0.5 else "#aed6f1"
            for k in range(n_out)
        ]
        pct_laund_gen = 100.0 * float((laund_prob > 0.5).mean())
        s_gen  = _stats(G_gen, n_out)
        pos_gen = nx.spring_layout(G_gen, seed=42,
                                   k=2.0 / max(n_out ** 0.5, 1))
        gen_stats_str = (
            f"nodes={n_out}  edges={s_gen['edges']}\n"
            f"density={s_gen['density']:.3f}  mean deg={s_gen['mean_deg']:.1f}\n"
            f"pred. laundering: {pct_laund_gen:.0f}%"
        )

        # ── Seed network ──────────────────────────────────────────────────
        if seed_networks is not None and i < len(seed_networks):
            seed_net = seed_networks[i]
        else:
            # Fall back to closest training network as seed
            cos_sims  = H_np @ gen_emb
            seed_net  = networks[int(cos_sims.argmax())]

        G_seed, seed_colors, n_seed, n_laund_seed = _build_nx(seed_net)
        s_seed   = _stats(G_seed, n_seed)
        pos_seed = nx.spring_layout(G_seed, seed=42,
                                    k=2.0 / max(n_seed ** 0.5, 1))
        seed_stats_str = (
            f"nodes={n_seed}  edges={s_seed['edges']}\n"
            f"density={s_seed['density']:.3f}  mean deg={s_seed['mean_deg']:.1f}\n"
            f"laundering nodes: {n_laund_seed}"
        )

        # ── Closest training network by cosine sim to generated embedding ─
        cos_sims    = H_np @ gen_emb
        closest_idx = int(cos_sims.argmax())
        cos_sim_val = float(cos_sims[closest_idx])
        close_net   = networks[closest_idx]

        G_close, close_colors, n_close, n_laund_close = _build_nx(close_net)
        s_close   = _stats(G_close, n_close)
        pos_close = nx.spring_layout(G_close, seed=42,
                                     k=2.0 / max(n_close ** 0.5, 1))
        close_stats_str = (
            f"nodes={n_close}  edges={s_close['edges']}\n"
            f"density={s_close['density']:.3f}  mean deg={s_close['mean_deg']:.1f}\n"
            f"laundering nodes: {n_laund_close}\n"
            f"cosine sim = {cos_sim_val:.4f}"
        )

        # ── Bar chart data ────────────────────────────────────────────────
        bar_labels   = ["Density ×100", "Mean degree", "% Laund. nodes"]
        gen_vals  = [
            s_gen["density"]   * 100,
            s_gen["mean_deg"],
            pct_laund_gen,
        ]
        seed_vals = [
            s_seed["density"]  * 100,
            s_seed["mean_deg"],
            100.0 * n_laund_seed / max(n_seed, 1),
        ]
        close_vals = [
            s_close["density"] * 100,
            s_close["mean_deg"],
            100.0 * n_laund_close / max(n_close, 1),
        ]

        # ── Figure: 4 panels ─────────────────────────────────────────────
        fig, axes = plt.subplots(
            1, 4, figsize=(32, 8),
            gridspec_kw={"width_ratios": [2, 2, 2, 1.5]},
        )
        ax_gen, ax_seed, ax_close, ax_bar = axes

        fig.suptitle(
            f"Generated #{i+1}  |  seed → generated → closest training",
            fontsize=13, fontweight="bold",
        )

        _draw_panel(ax_gen,   G_gen,   pos_gen,   gen_colors,   "Generated Network",  gen_stats_str)
        _draw_panel(ax_seed,  G_seed,  pos_seed,  seed_colors,  "Seed Network",       seed_stats_str)
        _draw_panel(ax_close, G_close, pos_close, close_colors, "Closest Training",   close_stats_str)

        # Panel 4 — grouped bar chart
        x      = np.arange(len(bar_labels))
        width  = 0.25
        colors_bar = ["#e74c3c", "#f39c12", "#3498db"]

        ax_bar.bar(x - width, gen_vals,   width, label="Generated",       color=colors_bar[0], alpha=0.85)
        ax_bar.bar(x,         seed_vals,  width, label="Seed",            color=colors_bar[1], alpha=0.85)
        ax_bar.bar(x + width, close_vals, width, label="Closest training",color=colors_bar[2], alpha=0.85)

        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(bar_labels, fontsize=9)
        ax_bar.set_ylabel("Value", fontsize=10)
        ax_bar.set_title("Structural Statistics", fontsize=11, fontweight="bold")
        ax_bar.legend(fontsize=8, loc="upper right")
        ax_bar.spines["top"].set_visible(False)
        ax_bar.spines["right"].set_visible(False)

        # Add value labels on bars
        for bars in ax_bar.containers:
            ax_bar.bar_label(bars, fmt="%.1f", fontsize=7, padding=2)

        # Shared legend for node colours
        legend_els = [
            Patch(facecolor="#e74c3c", label="Laundering node"),
            Patch(facecolor="#aed6f1", label="Regular node"),
            Patch(facecolor="#aab7b8", label="Collapsed hub (training)"),
        ]
        fig.legend(handles=legend_els, loc="lower center",
                   ncol=3, fontsize=9, frameon=True,
                   bbox_to_anchor=(0.40, 0.0))

        fig.tight_layout(rect=[0, 0.06, 1, 0.94])
        out = comp_dir / f"comparison_gen_{i+1:02d}.png"
        fig.savefig(out, dpi=200)
        plt.close(fig)
        print(f"  Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import numpy as np

    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir = ROOT_DIR / "results" / "ibm" / "latent_seed"
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results → {results_dir}\n")

    # ── 1. Load networks ──────────────────────────────────────────────────────
    CACHE_PATH = DATA_DIR / "networks_cache.pkl"
    df_full    = preprocess_df(CSV_PATH)

    if CACHE_PATH.exists():
        print(f"Loading networks from cache: {CACHE_PATH} …")
        with open(CACHE_PATH, "rb") as f:
            networks = pickle.load(f)
        for net in networks:
            net["graph"] = build_igraph_from_transactions(net["transactions"])
    else:
        print("Extracting networks (no cache found) …")
        networks = extract_networks_igraph(
            df_full, max_depth=4, max_networks=4000,
            collapse_threshold=10, max_nodes=MAX_NODES,
        )
        for net in networks:
            net["graph"] = build_igraph_from_transactions(net["transactions"])
        to_cache = [{k: v for k, v in n.items() if k != "graph"} for n in networks]
        with open(CACHE_PATH, "wb") as f:
            pickle.dump(to_cache, f)
        print(f"Saved cache → {CACHE_PATH}")

    n_laund = sum(1 for n in networks if len(n["laundering_nodes"]) > 0)
    print(f"Loaded {len(networks)} networks ({n_laund} laundering, "
          f"{len(networks) - n_laund} clean)\n")

    # ── 2. Load models ────────────────────────────────────────────────────────
    print("Loading models …")
    encoder = load_simclr_encoder(device)
    diff_model, diffusion, x_mean, x_std, _max_n = load_diffusion_model(device)

    # ── 3. Encode + probe ─────────────────────────────────────────────────────
    print("\nEncoding training networks …")
    H_all_n, y_all = encode_all_networks(networks, encoder, device)
    probe = train_mlp_probe(H_all_n, y_all, device)

    # ── 4. Precompute structural stats for every training network ─────────────
    print("\nPrecomputing training degree statistics …")
    train_degree_stats = precompute_train_degree_stats(networks)
    print(f"  Degree stats tensor: {train_degree_stats.shape}")

    # ── 5. Find extrapolated seeds + diverse selection ────────────────────────
    print(f"\nFinding extrapolated seeds (γ={EXTRAP_GAMMA}) …")
    all_seeds      = find_extrapolated_seeds(H_all_n, y_all, gamma=EXTRAP_GAMMA)
    selected_seeds = select_diverse_seeds(all_seeds, n_seeds=N_GEN)

    print(f"Selected {len(selected_seeds)} diverse extrapolated seeds:")
    for i, s in enumerate(selected_seeds):
        print(f"  [{i+1}] laund={s['laund_idx']:>5}  clean={s['clean_idx']:>5}"
              f"  nearest_sim={s['nearest_sim']:.4f}  "
              f"(lower = more outside training distribution)")

    # ── 6. Generate with structural repulsion ────────────────────────────────
    print("\nRunning structurally-novel guided generation …")
    gen_outputs, gen_embeddings, seed_networks, seed_indices = run_latent_seed_generation(
        networks, encoder, probe, diff_model, diffusion,
        x_mean, x_std, H_all_n, train_degree_stats, device,
        selected_seeds=selected_seeds,
        target_label=TARGET,
        t_start=T_START,
        guidance_scale=GUIDE_SCALE,
        novelty_weight=NOVELTY_WEIGHT,
        struct_weight=STRUCT_WEIGHT,
        guide_every=GUIDE_EVERY,
        guide_from=GUIDE_FROM,
    )

    # ── 7. UMAP ───────────────────────────────────────────────────────────────
    print("\nPlotting UMAP …")
    plot_latent_seeds_umap(
        H_all_n, y_all.tolist(),
        selected_seeds, seed_indices, gen_embeddings,
        TARGET, T_START, EXTRAP_GAMMA,
        results_dir / "latent_seed_umap.png",
    )

    # ── 8. Per-network comparison plots ──────────────────────────────────────
    print("\nPlotting generated vs closest training comparisons …")
    plot_generated_vs_closest_training(
        gen_outputs, gen_embeddings, H_all_n, networks, results_dir,
        seed_networks=seed_networks,
    )

    # ── 9. Score ──────────────────────────────────────────────────────────────
    print()
    mu, cov_inv, realism_scale = fit_training_distribution(networks)

    gen_scores = []
    for i, (x_d, adj_d, _) in enumerate(gen_outputs):
        s = score_network(
            x_d, adj_d, encoder, H_all_n, device,
            mu, cov_inv, realism_scale,
            gen_embedding=torch.tensor(gen_embeddings[i]),
        )
        gen_scores.append(s)
    _print_scores_table(gen_scores, "Structurally-novel generated networks")
    _save_scores_csv(gen_scores, results_dir / "latent_seed_generated.csv", "latent_seed")

    # ── 10. Calibration ───────────────────────────────────────────────────────
    from diffusion.diff_util import network_to_dense as _ntd
    calib_scores = []
    for net in _random.sample(networks, min(N_CALIB, len(networks))):
        xr = net["x_dense"]   if "x_dense"   in net else _ntd(net)[0]
        ar = net["adj_dense"] if "adj_dense" in net else _ntd(net)[1]
        calib_scores.append(
            score_network(xr, ar, encoder, H_all_n, device, mu, cov_inv, realism_scale)
        )
    _print_scores_table(calib_scores, "Calibration: real training networks")
    _save_scores_csv(calib_scores, results_dir / "latent_seed_calibration.csv", "training")

    print(f"\nAll results saved to {results_dir}")


if __name__ == "__main__":
    main()
