import math
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from torch_geometric.data import Data, Batch

_GEN_DIR  = Path(__file__).resolve().parent   # igraph_version/generation/
ROOT_DIR  = _GEN_DIR.parent                   # igraph_version/
DIFF_DIR  = ROOT_DIR / "diffusion"
CKPT_DIR  = ROOT_DIR / "checkpoints"
MAX_NODES = 300

for _p in (str(ROOT_DIR), str(DIFF_DIR), str(ROOT_DIR / "simclr")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _enforce_connectivity(adj_bin):
    """
    Connect every isolated component to the main (largest) component via hub
    attachment: each disconnected component is joined by a single edge from its
    highest-degree node to the highest-degree node in the main component.

    Hub attachment is more realistic for transaction graphs than sequential
    chaining (which produces a visual chain/trace in spring-layout plots because
    every isolated node ends up with exactly one edge, causing spring-layout to
    pull them into a long line).

    Returns (connected adjacency, number of edges added).
    """
    G = nx.from_numpy_array(adj_bin.numpy())
    components = sorted(nx.connected_components(G), key=len, reverse=True)
    if len(components) == 1:
        return adj_bin, 0
    adj_conn = adj_bin.clone()
    degrees   = dict(G.degree())
    main_comp = components[0]
    # Highest-degree node in the main component acts as the attachment hub
    hub = max(main_comp, key=lambda u: degrees[u])
    edges_added = 0
    for comp in components[1:]:
        # Best node in the isolated component (highest existing degree, or any if all zero)
        anchor = max(comp, key=lambda u: degrees[u])
        adj_conn[hub, anchor] = 1.0
        adj_conn[anchor, hub] = 1.0
        edges_added += 1
    return adj_conn, edges_added


def _to_pyg(x0_nodes, adj_soft, n, device, x_mean, x_std):
    """Convert dense predicted node features + adjacency to a PyG Data object.

    Col 0 (laundering flag) is excluded so the encoder receives the same
    feature dimension as during training (after the label-leakage fix).
    """
    x_cont = x0_nodes[:, 1:] * x_std[1:] + x_mean[1:]
    deg    = adj_soft.sum(dim=-1, keepdim=True)
    deg_n  = deg / deg.detach().max().clamp(min=1.0)
    x_feat = torch.cat([deg_n, x_cont[:, 1:]], dim=-1)   # [n, 5] — col 0 stripped
    ei = (adj_soft.detach() > 0.5).nonzero(as_tuple=False).T.contiguous()
    if ei.shape[1] > 0:
        ei = ei[:, ei[0] != ei[1]]   # remove self-loops
    if ei.shape[1] == 0:
        ei = torch.zeros(2, 0, dtype=torch.long, device=device)
    return Data(x=x_feat, edge_index=ei,
                batch=torch.zeros(n, dtype=torch.long, device=device))


def load_simclr_encoder(device):
    """Find and load the best SimCLR checkpoint. Returns encoder in eval mode."""
    from simclr import GraphEncoder
    ckpt_dir = CKPT_DIR / "simclr_ibm"
    candidates = list(ckpt_dir.glob("*.pt"))
    best_ckpt_path, best_loss = None, float("inf")
    for p in candidates:
        try:
            c = torch.load(p, map_location="cpu", weights_only=False)
            if isinstance(c, dict) and "loss" in c and c["loss"] < best_loss:
                best_loss, best_ckpt_path = c["loss"], p
        except Exception:
            pass
    print(f"Best SimCLR checkpoint: {best_ckpt_path.name}  (loss={best_loss:.4f})")
    ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    sd         = ckpt["encoder_state_dict"]
    in_dim     = sd["conv1.lin.weight"].shape[1]
    hidden_dim = sd["conv1.lin.weight"].shape[0]
    n_layers   = 3 if "conv3.lin.weight" in sd else 2
    use_bn     = "bn1.weight" in sd
    encoder = GraphEncoder(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=128,
                           n_layers=n_layers, use_bn=use_bn).to(device)
    encoder.load_state_dict(sd)
    encoder.eval()
    return encoder


def load_diffusion_model(device):
    """Load diffusion model and normalisation stats. Returns (model, diffusion, x_mean, x_std)."""
    from diffusion.model import DiffusionGNN
    from diffusion.diff_util import create_diffusion
    ckpt            = torch.load(CKPT_DIR / "diffusion_ibm" / "model.pt", map_location=device, weights_only=False)
    node_dim        = ckpt["model"]["input_proj.weight"].shape[1]
    class_cond_ckpt = ckpt.get("class_conditional", False)
    diff_model = DiffusionGNN(node_dim=node_dim, hidden_dim=128, num_layers=4,
                              class_conditional=class_cond_ckpt).to(device)
    diff_model.load_state_dict(ckpt["model"])
    diff_model.eval()
    diffusion  = create_diffusion(T=500)
    x_mean     = ckpt["x_mean"].to(device)
    x_std      = ckpt["x_std"].to(device)
    print("Diffusion model loaded.")
    return diff_model, diffusion, x_mean, x_std


def encode_all_networks(networks, encoder, device):
    """
    Encode every network with the SimCLR encoder.
    Returns (H_all_n, y_all) — normalised embeddings and binary labels.
    """
    from augmentation import augment_network_view_fast
    from simclr import network_to_pyg_data_fast
    all_graphs, all_labels = [], []
    for net in networks:
        v = augment_network_view_fast(net)
        all_graphs.append(network_to_pyg_data_fast(v))
        all_labels.append(int(len(net["laundering_nodes"]) > 0))
    with torch.no_grad():
        H_all = encoder(Batch.from_data_list(all_graphs).to(device)).cpu()
    H_all_n = F.normalize(H_all, dim=1)           # [N, 128]
    y_all   = torch.tensor(all_labels, dtype=torch.float32)
    return H_all_n, y_all


def train_mlp_probe(H_all_n, y_all, device, n_epochs=500):
    """Train a binary MLP probe on frozen SimCLR embeddings. Returns the probe."""
    probe = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1)).to(device)
    # weight_decay = L2 regularisation — prevents the probe from overfitting to
    # small embedding sets, which would produce misleading classifier guidance.
    opt   = torch.optim.Adam(probe.parameters(), lr=5e-3, weight_decay=1e-4)
    for _ in range(n_epochs):
        logit = probe(H_all_n.to(device)).squeeze(-1)
        loss  = F.binary_cross_entropy_with_logits(logit, y_all.to(device))
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        preds = (torch.sigmoid(probe(H_all_n.to(device)).squeeze(-1)) > 0.5).cpu()
    acc = (preds == y_all.bool()).float().mean()
    print(f"MLP probe accuracy: {acc:.3f}")
    for p in probe.parameters():
        p.requires_grad_(False)
    return probe


def guided_generate(
    seed_network,
    encoder,
    probe,
    diff_model,
    diffusion,
    x_mean,
    x_std,
    H_train,
    device,
    target_label=1,
    t_start=200,
    guidance_scale=2.0,
    novelty_weight=2.0,
    novelty_k=10,
    guide_every=5,
    guide_from=0.25,
    degree_penalty=0.5,
    target_mean_degree=None,
    adj_threshold=0.5,
    adj_gamma=1.5,
    target_density=None,
    max_nodes=None,
    pbar=None,
):
    """
    Generate one network via diffusion guided by:
      1. Classification loss  — steer towards target_label
      2. Novelty repulsion    — push away from K nearest training embeddings

    Returns (x_out, adj_out, n_nodes) all on CPU.
    """
    from diffusion.diff_util import network_to_dense

    _max = max_nodes if max_nodes is not None else MAX_NODES

    if "x_dense" in seed_network and "adj_dense" in seed_network:
        x   = seed_network["x_dense"].to(device)
        adj = seed_network["adj_dense"].to(device)
    else:
        x, adj = network_to_dense(seed_network)
        x, adj = x.to(device), adj.to(device)
    n = x.shape[0]

    x_pad   = torch.zeros(1, _max, x.shape[1], device=device)
    adj_pad = torch.zeros(1, _max, _max,    device=device)
    mask    = torch.zeros(1, _max,           device=device)
    x_pad[0, :n]       = x
    adj_pad[0, :n, :n] = adj
    mask[0, :n]        = 1.0

    x_norm = x_pad.clone()
    x_norm[:, :, 1:] = (x_pad[:, :, 1:] - x_mean[1:]) / x_std[1:]
    x_norm  = x_norm  * mask.unsqueeze(-1)
    adj_pad = adj_pad * mask[:, :, None] * mask[:, None, :]

    t_tensor   = torch.tensor([t_start], device=device)
    x_t, adj_t = diffusion.q_sample(x_norm, t_tensor, node_mask=mask, adj_start=adj_pad)

    guide_threshold = int(guide_from * t_start)
    cached_grad     = None
    cached_grad_adj = None
    H_dev = H_train.to(device) if H_train is not None else None

    for step_i, t_curr in enumerate(range(t_start, -1, -1)):
        t_vec    = torch.tensor([t_curr], device=device)
        t_scaled = diffusion._scale_timesteps(t_vec)
        eff_guide_every = 1 if t_curr < 100 else guide_every
        do_guide = (t_curr < guide_threshold) and (step_i % eff_guide_every == 0)

        if do_guide:
            with torch.enable_grad():
                x_t_g   = x_t.detach().requires_grad_(True)
                adj_t_g = adj_t.detach().requires_grad_(True)
                eps_pred, adj_pred, _ = diff_model(x_t_g, t_scaled,
                                                   adj=adj_t_g, node_mask=mask)
                x0_cont = diffusion._predict_xstart_from_eps(
                              x_t_g[..., 1:], t_vec, eps_pred[..., 1:])
                x0_bin  = eps_pred[..., 0:1].clamp(0, 1)
                x0_pred = torch.cat([x0_bin, x0_cont], dim=-1)

                pyg   = _to_pyg(x0_pred[0, :n], adj_pred[0, :n, :n], n, device, x_mean, x_std)
                h     = encoder(pyg)
                h_n   = F.normalize(h, dim=-1)           # [1, 128]

                score  = torch.sigmoid(probe(h_n)).squeeze()
                g_loss = (-torch.log(score + 1e-8) if target_label == 1
                          else -torch.log(1 - score + 1e-8))

                # Novelty weight ramps up as t decreases so classification
                # dominates the structural phase and novelty dominates fine detail
                t_frac      = t_curr / max(t_start, 1)
                eff_novelty = novelty_weight * (1.0 - t_frac)
                if H_dev is not None and eff_novelty > 0.0:
                    cos_sims   = (H_dev @ h_n.T).squeeze()
                    top_k_sims = torch.topk(cos_sims, min(novelty_k, len(H_dev))).values
                    g_loss     = g_loss + eff_novelty * top_k_sims.mean()

                # Degree penalty — penalises excess degree above training target.
                # Uses squared one-sided loss so it pulls toward realistic density
                # rather than toward zero edges.
                if degree_penalty > 0.0:
                    mean_deg = adj_pred[0, :n, :n].sum(dim=-1).mean()
                    if target_mean_degree is not None:
                        excess = torch.relu(mean_deg - target_mean_degree)
                        g_loss = g_loss + degree_penalty * excess ** 2
                    else:
                        g_loss = g_loss + degree_penalty * mean_deg

                grads = torch.autograd.grad(g_loss, [x_t_g, adj_t_g])
                # Clip to prevent instability from large gradient magnitudes
                cached_grad     = grads[0].detach().clamp(-1.0, 1.0)
                cached_grad_adj = grads[1].detach().clamp(-1.0, 1.0)
                # Cache eps and adj_pred for use in the denoising step below
                cached_eps_pred  = eps_pred.detach()
                cached_adj_pred  = adj_pred.detach()
        else:
            with torch.no_grad():
                eps_pred, adj_pred, _ = diff_model(x_t, t_scaled,
                                                   adj=adj_t, node_mask=mask)
                x0_cont = diffusion._predict_xstart_from_eps(
                              x_t[..., 1:], t_vec, eps_pred[..., 1:])
                x0_bin  = eps_pred[..., 0:1].clamp(0, 1)
                x0_pred = torch.cat([x0_bin, x0_cont], dim=-1)
                cached_eps_pred = eps_pred
                cached_adj_pred = adj_pred

        with torch.no_grad():
            coef1    = float(diffusion.posterior_mean_coef1[t_curr])
            coef2    = float(diffusion.posterior_mean_coef2[t_curr])
            post_lv  = float(diffusion.posterior_log_variance_clipped[t_curr])

            if cached_grad is not None:
                # condition_score guidance (Song et al. 2020):
                # Modify the epsilon prediction in score-function space rather
                # than shifting the posterior mean after the fact.  This is more
                # principled: eps_guided = eps + sqrt(1-ᾱ_t) * ∇_x_t log p(y|x_t)
                # Since cached_grad = d(g_loss)/d(x_t) = -∇ log p(y|x_t),
                # the correction is +sqrt(1-ᾱ_t) * cached_grad.
                sqrt_1m_ab  = math.sqrt(max(1.0 - float(diffusion.alphas_cumprod[t_curr]), 1e-8))
                eps_guided  = (cached_eps_pred[..., 1:].detach()
                               + guidance_scale * sqrt_1m_ab * cached_grad[..., 1:])
                x0_cont_g   = diffusion._predict_xstart_from_eps(
                                  x_t.detach()[..., 1:], t_vec, eps_guided)
                x0_bin_g    = cached_eps_pred[..., 0:1].detach().clamp(0, 1)
                x0_d        = torch.cat([x0_bin_g, x0_cont_g], dim=-1) * mask.unsqueeze(-1)
            else:
                x0_d = torch.cat([x0_pred[..., 0:1].clamp(0, 1),
                                   x0_pred[..., 1:]], dim=-1).detach()
                x0_d = x0_d * mask.unsqueeze(-1)

            mean  = coef1 * x0_d + coef2 * x_t.detach()
            noise = torch.randn_like(x_t) * mask.unsqueeze(-1)
            x_t   = (mean + (t_curr > 0) * np.exp(0.5 * post_lv) * noise) * mask.unsqueeze(-1)

            # Adjacency guidance: operate in logit space for numerical stability.
            # Subtracting guidance_scale * grad moves adj_pred in the direction
            # that decreases g_loss (i.e. toward the target class).
            ap = cached_adj_pred.clamp(0, 1)
            if cached_grad_adj is not None:
                ap_logit = torch.logit(ap.clamp(1e-6, 1.0 - 1e-6))
                ap_logit = ap_logit - guidance_scale * cached_grad_adj
                ap = torch.sigmoid(ap_logit)

            if t_curr > 0:
                # Intermediate steps: sample binary adjacency without gamma
                # compression.  Applying gamma at every step squashes mid-range
                # probabilities repeatedly, producing too many isolated nodes that
                # then get chained by _enforce_connectivity.  The model was trained
                # on binary {0,1} adjacency, so stochastic bernoulli sampling here
                # stays in-distribution.
                adj_t = torch.bernoulli(ap)
            else:
                # Final step only: apply gamma compression to select high-confidence
                # edges before thresholding.  Concentrating it here avoids the
                # feedback loop where the model over-predicts to compensate for
                # artificially sparse intermediate adjacency.
                ap = ap ** adj_gamma
                if target_density is not None:
                    # Density-calibrated threshold: keep only enough edges to match
                    # the training mean density, selecting the highest-confidence ones.
                    n_active = int(mask[0].sum().item())
                    n_keep   = max(1, round(target_density * n_active * (n_active - 1) / 2))
                    ap_flat  = ap[0, :n, :n].reshape(-1)
                    if n_keep * 2 < len(ap_flat):
                        kth   = ap_flat.kthvalue(len(ap_flat) - n_keep * 2).values.item()
                        adj_t = (ap > max(kth, 0.05)).float()
                    else:
                        adj_t = (ap > adj_threshold).float()
                else:
                    adj_t = (ap > adj_threshold).float()

            adj_t = (adj_t + adj_t.transpose(-1, -2)) / 2 * mask[:, :, None] * mask[:, None, :]

            # Final step: connect any 0-degree active nodes to their most
            # confident neighbour according to ap, before the state is locked in.
            # This uses the model's own predictions rather than hub-attachment,
            # producing more realistic local topology.
            if t_curr == 0 and n > 1:
                _deg_f = adj_t[0, :n, :n].sum(dim=-1)          # [n]
                _iso   = (_deg_f < 0.5) & (mask[0, :n] > 0.5)  # isolated active nodes
                if _iso.any():
                    _probs = ap[0, :n, :n].clone()
                    _arange = torch.arange(n, device=_probs.device)
                    _probs[_arange, _arange] = -1.0             # no self-loops
                    _best  = _probs.argmax(dim=-1)              # [n]
                    _i_idx = _iso.nonzero(as_tuple=True)[0]
                    _j_idx = _best[_i_idx]
                    adj_t[0, _i_idx, _j_idx] = 1.0
                    adj_t[0, _j_idx, _i_idx] = 1.0

        if pbar is not None:
            pbar.update(1)

    # Guarantee a single connected component before returning
    adj_out = adj_t[0, :n, :n].cpu()
    adj_out.fill_diagonal_(0.0)   # remove self-loops before connectivity check
    adj_out, n_patches = _enforce_connectivity(adj_out)
    # Return patch count so callers can log/filter graphs that needed heavy repair
    return x_t[0, :n].cpu(), adj_out, n, n_patches


def run_guided_generation(
    networks,
    encoder,
    probe,
    diff_model,
    diffusion,
    x_mean,
    x_std,
    H_all_n,
    device,
    target_label=1,
    n_gen=8,
    t_start=200,
    guidance_scale=2.0,
    novelty_weight=2.0,
    guide_every=5,
    guide_from=0.25,
    degree_penalty=0.5,
    adj_threshold=0.5,
    adj_gamma=1.5,
):
    """
    Generate n_gen networks using guided diffusion.
    Seeds are drawn equally from laundering and clean training networks.

    Returns (gen_outputs, gen_embeddings, seeds):
        gen_outputs    : list of (x_denorm, adj, n_nodes) tuples
        gen_embeddings : np.ndarray [n_gen, 128]  normalised encoder embeddings
        seeds          : list of seed network dicts
    """
    import random as _random
    from diffusion.diff_util import network_to_dense as _ntd

    # Compute training mean degree and edge density for calibrated generation
    mean_degs, densities = [], []
    for net in networks:
        adj = net["adj_dense"] if "adj_dense" in net else _ntd(net)[1]
        n = adj.shape[0]
        n_edges = float(adj.sum()) / 2
        mean_degs.append(float(adj.sum(dim=-1).mean()))
        if n > 1:
            densities.append(n_edges / (n * (n - 1) / 2))
    target_mean_degree = float(np.mean(mean_degs)) if mean_degs else None
    target_density     = float(np.mean(densities)) if densities else None

    laund_nets = [n for n in networks if     len(n["laundering_nodes"]) > 0]
    clean_nets = [n for n in networks if not len(n["laundering_nodes"]) > 0]

    # Seed from the same class as target_label so guidance starts from a
    # realistic example of the desired class rather than fighting from the
    # opposite distribution.  Fall back to mixed if one pool is too small.
    if target_label == 1 and len(laund_nets) >= n_gen:
        seeds = _random.sample(laund_nets, n_gen)
    elif target_label == 0 and len(clean_nets) >= n_gen:
        seeds = _random.sample(clean_nets, n_gen)
    else:
        seeds = (
            _random.sample(laund_nets, min(n_gen // 2, len(laund_nets)))
          + _random.sample(clean_nets, min(n_gen // 2, len(clean_nets)))
        )

    # ── Structural validity thresholds ────────────────────────────────────────
    train_sizes = [
        (net["adj_dense"] if "adj_dense" in net else _ntd(net)[1]).shape[0]
        for net in networks
    ]
    size_5th = float(np.percentile(train_sizes, 5)) if train_sizes else 3.0
    density_ceil = 2.0 * target_density if target_density is not None else None

    gen_outputs, gen_embeddings = [], []
    n_discarded = 0

    with tqdm(total=len(seeds) * (t_start + 1),
              desc=f"Generating {len(seeds)} networks",
              unit="step", dynamic_ncols=True) as pbar:
        for i, seed in enumerate(seeds):
            pbar.set_postfix(network=f"{i+1}/{len(seeds)}")
            x_out, adj_out, n_out, n_patches = guided_generate(
                seed, encoder, probe, diff_model, diffusion,
                x_mean, x_std, H_all_n, device,
                target_label=target_label,
                t_start=t_start, guidance_scale=guidance_scale,
                novelty_weight=novelty_weight,
                guide_every=guide_every, guide_from=guide_from,
                degree_penalty=degree_penalty,
                target_mean_degree=target_mean_degree,
                adj_threshold=adj_threshold,
                adj_gamma=adj_gamma,
                target_density=target_density,
                pbar=pbar,
            )

            if n_patches > 0:
                print(f"  [validity] graph {i+1}: connectivity repair added {n_patches} edge(s)")

            # ── Structural validity checks ─────────────────────────────────
            gen_density = float(adj_out.sum()) / max(n_out * (n_out - 1), 1)
            if density_ceil is not None and gen_density > density_ceil:
                print(f"  [validity] graph {i+1} discarded: density {gen_density:.3f} "
                      f"> 2× target ({target_density:.3f})")
                n_discarded += 1
                continue
            # Discard graphs that are far too sparse — these are degenerate outputs
            # where most edges were lost during denoising, forcing _enforce_connectivity
            # to add many artificial hub-to-node edges.
            density_floor = (target_density * 0.2) if target_density is not None else 0.0
            if gen_density < density_floor:
                print(f"  [validity] graph {i+1} discarded: density {gen_density:.3f} "
                      f"< 20% of target ({target_density:.3f})")
                n_discarded += 1
                continue
            if n_out < size_5th:
                print(f"  [validity] graph {i+1} discarded: "
                      f"n_nodes {n_out} < 5th-pct ({size_5th:.0f})")
                n_discarded += 1
                continue

            # Denormalise node features — col 0 (laundering flag) is excluded
            # to match the encoder's input dimension after the leakage fix.
            x_cont_d = x_out[:, 1:] * x_std.cpu()[1:] + x_mean.cpu()[1:]
            deg_g    = adj_out.sum(dim=-1, keepdim=True)
            deg_n    = deg_g / deg_g.max().clamp(min=1.0)
            x_denorm = torch.cat([deg_n, x_cont_d[:, 1:]], dim=-1)   # [n, 5]

            ei_g = (adj_out > adj_threshold).nonzero(as_tuple=False).T.contiguous()
            bv_g = torch.zeros(n_out, dtype=torch.long)
            with torch.no_grad():
                h_g   = encoder(Data(x=x_denorm, edge_index=ei_g,
                                     batch=bv_g).to(device)).cpu()
                h_g_n = F.normalize(h_g, dim=-1)

            # Label is the guided target — we explicitly steered generation
            # toward target_label, so that is the correct label.
            gen_outputs.append((x_denorm, adj_out, n_out, target_label))
            gen_embeddings.append(h_g_n.squeeze(0).numpy())

    if n_discarded:
        print(f"  [validity] {n_discarded}/{len(seeds)} graphs discarded by structural checks")

    gen_embeddings = np.stack(gen_embeddings, axis=0)
    return gen_outputs, gen_embeddings, seeds


# ─────────────────────────────────────────────────────────────────────────────
# Direction 4 — Learnable augmentation policy
# Bayesian / random search over guidance hyperparameters using Q-score as proxy.
# ─────────────────────────────────────────────────────────────────────────────

def tune_guidance_params(
    networks,
    encoder,
    probe,
    diff_model,
    diffusion,
    x_mean,
    x_std,
    H_train,
    train_laund_data,
    H_train_laund,
    device,
    n_trials=15,
    n_gen_per_trial=6,
    t_start=150,
    search_space=None,
    results_dir=None,
):
    """
    Search for guided-generation hyperparameters that maximise mean Q-score
    on a small batch of generated laundering graphs.  Q-score is a cheap proxy
    for downstream classifier utility (correlation r ≈ 0.6 empirically).

    Uses Optuna (TPE sampler) when available; falls back to uniform random
    search otherwise.  Bayesian optimisation converges faster because it builds
    a probabilistic model of the objective and samples from high-expected-
    improvement regions — effectively learning which combinations of
    guidance_scale / novelty_weight / degree_penalty produce realistic graphs.

    Parameters
    ----------
    networks          : list of IBM network dicts (training set)
    encoder           : frozen SimCLR GraphEncoder
    probe             : frozen MLP probe (trained on encoder embeddings)
    diff_model        : trained DiffusionGNN
    diffusion         : GaussianDiffusion scheduler
    x_mean, x_std     : feature normalisation tensors
    H_train           : Tensor [N_train, 128]  all training embeddings
    train_laund_data  : list[PyG Data]  real laundering training graphs
    H_train_laund     : Tensor [M, 128]  laundering-only training embeddings
    device            : torch.device
    n_trials          : number of hyperparameter candidates to evaluate
    n_gen_per_trial   : graphs generated per candidate (more = less variance, slower)
    t_start           : diffusion noise start level (fixed across trials)
    search_space      : dict of (lo, hi) bounds for each parameter; defaults to
                        {"guidance_scale": (0.5, 5.0),
                         "novelty_weight": (0.0, 4.0),
                         "degree_penalty": (0.0, 2.0)}
    results_dir       : optional Path — saves trial CSV to results_dir/tuning_trials.csv

    Returns
    -------
    best_params : dict  {"guidance_scale": …, "novelty_weight": …, "degree_penalty": …}
    history     : list[dict]  all trials with their Q-scores, sorted best-first
    """
    try:
        from generation.graph_quality_metrics import score_generated_graphs
    except ImportError:
        from graph_quality_metrics import score_generated_graphs

    from torch_geometric.data import Data as _Data

    if search_space is None:
        search_space = {
            "guidance_scale": (0.5, 5.0),
            "novelty_weight": (0.0, 4.0),
            "degree_penalty": (0.0, 2.0),
        }

    _DEFAULT_PARAMS = {
        "guidance_scale": 2.0,
        "novelty_weight": 2.0,
        "degree_penalty": 0.5,
    }

    def _gen_to_pyg(x_denorm, adj_out, n_out, label, adj_threshold=0.5):
        """Minimal (x_denorm, adj, n, label) → PyG Data, mirroring gen_output_to_pyg."""
        adj_np = adj_out[:n_out, :n_out]
        if isinstance(adj_np, torch.Tensor):
            adj_np = adj_np.numpy()
        ei = (torch.tensor(adj_np) > adj_threshold).nonzero(as_tuple=False).T.contiguous()
        if ei.shape[1] == 0:
            ei = torch.zeros(2, 0, dtype=torch.long)
        return _Data(
            x=x_denorm,
            edge_index=ei,
            y=torch.tensor([label], dtype=torch.long),
        )

    def _objective(guidance_scale, novelty_weight, degree_penalty):
        """Generate n_gen_per_trial graphs and return their mean Q-score."""
        try:
            gen_outs, _, _ = run_guided_generation(
                networks, encoder, probe, diff_model, diffusion,
                x_mean, x_std, H_train, device,
                target_label=1,
                n_gen=n_gen_per_trial,
                t_start=t_start,
                guidance_scale=guidance_scale,
                novelty_weight=novelty_weight,
                degree_penalty=degree_penalty,
            )
        except Exception as e:
            print(f"    [tune] generation error: {e}")
            return 0.0

        if not gen_outs:
            return 0.0

        gen_pyg = [_gen_to_pyg(x, a, n, lbl)
                   for (x, a, n, lbl) in gen_outs]

        try:
            quality = score_generated_graphs(
                gen_pyg, train_laund_data, H_train_laund,
                encoder, device, k=min(3, len(train_laund_data)),
            )
            return float(np.mean(quality["Q"]))
        except Exception as e:
            print(f"    [tune] quality scoring error: {e}")
            return 0.0

    best_score  = -1.0
    best_params = dict(_DEFAULT_PARAMS)
    history     = []

    # ── Try Optuna first (TPE Bayesian optimisation) ──────────────────────────
    _use_optuna = False
    try:
        import optuna
        _use_optuna = True
    except ImportError:
        pass

    if _use_optuna:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def _optuna_obj(trial):
            params = {
                k: trial.suggest_float(k, lo, hi)
                for k, (lo, hi) in search_space.items()
            }
            score = _objective(**params)
            print(f"  [tune] trial {trial.number+1}/{n_trials}  "
                  f"gs={params['guidance_scale']:.2f}  "
                  f"nw={params['novelty_weight']:.2f}  "
                  f"dp={params['degree_penalty']:.2f}  →  Q={score:.4f}")
            return score

        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(_optuna_obj, n_trials=n_trials, show_progress_bar=False)
        best_params = study.best_params
        best_score  = study.best_value
        history = [
            {**t.params, "q_score": t.value}
            for t in study.trials if t.value is not None
        ]
        print(f"\n  [tune] Optuna best: {best_params}  Q={best_score:.4f}")

    else:
        # ── Uniform random search (no extra dependencies) ─────────────────────
        print(f"  [tune] Optuna not installed — running random search "
              f"({n_trials} trials)")
        rng = np.random.default_rng(seed=42)
        for trial_i in range(n_trials):
            params = {
                k: float(rng.uniform(lo, hi))
                for k, (lo, hi) in search_space.items()
            }
            score = _objective(**params)
            history.append({**params, "q_score": score})
            print(f"  [tune] trial {trial_i+1}/{n_trials}  "
                  f"gs={params['guidance_scale']:.2f}  "
                  f"nw={params['novelty_weight']:.2f}  "
                  f"dp={params['degree_penalty']:.2f}  →  Q={score:.4f}")
            if score > best_score:
                best_score  = score
                best_params = dict(params)

        print(f"\n  [tune] best: {best_params}  Q={best_score:.4f}")

    # Sort history best-first
    history = sorted(history, key=lambda d: d.get("q_score", 0.0), reverse=True)

    # Save trial log
    if results_dir is not None:
        import csv as _csv_mod
        from pathlib import Path as _Path
        trial_csv = _Path(results_dir) / "tuning_trials.csv"
        trial_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(trial_csv, "w", newline="") as _f:
            _w = _csv_mod.DictWriter(_f, fieldnames=list(search_space.keys()) + ["q_score"])
            _w.writeheader()
            _w.writerows(history)
        print(f"  [tune] trial log saved → {trial_csv}")

    return best_params, history


# ─────────────────────────────────────────────────────────────────────────────
# Elliptic equivalents  (PyG Data objects instead of IBM network dicts)
# ─────────────────────────────────────────────────────────────────────────────

def load_simclr_encoder_elliptic(device):
    """
    Like load_simclr_encoder() but reads from model_checkpoints_elliptic/.
    Run elliptic_simclr_train.py first to produce the checkpoint.
    """
    from simclr import GraphEncoder
    ckpt_dir   = CKPT_DIR / "simclr_elliptic"
    candidates = list(ckpt_dir.glob("*.pt"))
    best_path, best_loss = None, float("inf")
    for p in candidates:
        try:
            c = torch.load(p, map_location="cpu", weights_only=False)
            if isinstance(c, dict) and "loss" in c and c["loss"] < best_loss:
                best_loss, best_path = c["loss"], p
        except Exception:
            pass
    if best_path is None:
        raise FileNotFoundError(
            f"No valid checkpoint found in {ckpt_dir}. "
            "Run elliptic_simclr_train.py first."
        )
    print(f"Best Elliptic SimCLR checkpoint: {best_path.name}  (loss={best_loss:.4f})")
    ckpt       = torch.load(best_path, map_location=device, weights_only=False)
    sd         = ckpt["encoder_state_dict"]
    in_dim     = sd["conv1.lin.weight"].shape[1]
    hidden_dim = sd["conv1.lin.weight"].shape[0]
    n_layers   = 3 if "conv3.lin.weight" in sd else 2
    use_bn     = "bn1.weight" in sd
    encoder = GraphEncoder(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=128,
                           n_layers=n_layers, use_bn=use_bn).to(device)
    encoder.load_state_dict(sd)
    encoder.eval()
    return encoder


def load_diffusion_model_elliptic(device):
    """
    Like load_diffusion_model() but loads diffusion/model_elliptic.pt.
    Run elliptic_diffusion_train.py first to produce the checkpoint.
    """
    from diffusion.model    import DiffusionGNN
    from diffusion.diff_util import create_diffusion
    model_path = CKPT_DIR / "diffusion_elliptic" / "model.pt"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Elliptic diffusion checkpoint not found at {model_path}. "
            "Run elliptic_diffusion_train.py first."
        )
    ckpt       = torch.load(model_path, map_location=device, weights_only=False)
    node_dim   = ckpt["model"]["input_proj.weight"].shape[1]
    diff_model = DiffusionGNN(node_dim=node_dim, hidden_dim=128, num_layers=4).to(device)
    diff_model.load_state_dict(ckpt["model"])
    diff_model.eval()
    diffusion  = create_diffusion(T=500)
    x_mean     = ckpt["x_mean"].to(device)
    x_std      = ckpt["x_std"].to(device)
    print(f"Elliptic diffusion model loaded from {model_path}.")
    return diff_model, diffusion, x_mean, x_std


def encode_all_pyg_graphs(graphs, encoder, device, batch_size=128):
    """
    Encode a list of Elliptic PyG Data objects with the Elliptic SimCLR encoder.

    Each graph's .x [n, 5] is extended to [n, 6] by prepending .y as col 0,
    matching the convention used during elliptic_simclr_train.py training.

    Parameters
    ----------
    graphs     : list[PyG Data]  with .x [n,5] and .y [1]
    encoder    : GraphEncoder (in_dim=5 after label-leakage fix, trained on Elliptic)
    device     : torch.device
    batch_size : graphs per forward pass

    Returns
    -------
    H_all_n : Tensor [N, 128]  L2-normalised embeddings
    y_all   : Tensor [N]       float labels (0 = licit, 1 = illicit)
    """
    from torch_geometric.data import Data as _Data, Batch as _Batch

    ext_graphs, all_labels = [], []
    for g in graphs:
        # No label prepending — the encoder now expects features without col 0.
        ext_graphs.append(_Data(x=g.x.clone(), edge_index=g.edge_index.clone()))
        all_labels.append(float(g.y.item()))

    H_list = []
    with torch.no_grad():
        for i in range(0, len(ext_graphs), batch_size):
            chunk = _Batch.from_data_list(ext_graphs[i : i + batch_size]).to(device)
            H_list.append(encoder(chunk).cpu())

    H_all   = torch.cat(H_list, dim=0)                        # [N, 128]
    H_all_n = F.normalize(H_all, dim=1)
    y_all   = torch.tensor(all_labels, dtype=torch.float32)
    return H_all_n, y_all


def run_guided_generation_elliptic(
    graphs,
    encoder,
    probe,
    diff_model,
    diffusion,
    x_mean,
    x_std,
    H_all_n,
    device,
    target_label=1,
    n_gen=8,
    t_start=350,
    guidance_scale=2.0,
    novelty_weight=2.0,
    guide_every=5,
    guide_from=0.25,
    degree_penalty=0.5,
    adj_threshold=0.5,
    max_nodes=100,
):
    """
    Like run_guided_generation() but seeds are Elliptic PyG Data objects.

    Each seed's .x [n, 5] is extended to [n, 6] (prepend .y as col 0) and
    converted to a minimal IBM-style dict {"x_dense": …, "adj_dense": …} so
    that the existing guided_generate() function can be called without changes.

    Seed selection mirrors the IBM convention:
      n_gen // 2 seeds drawn from illicit  (y=1) graphs
      n_gen // 2 seeds drawn from licit    (y=0) graphs

    Returns (gen_outputs, gen_embeddings, seeds) with the same layout as
    run_guided_generation(), so gen_output_to_pyg() in evaluate_classifiers.py
    works unchanged.
    """
    import random as _random

    def _pyg_to_seed_dict(g):
        """Convert an Elliptic PyG Data (5-D) to a fake IBM dict for guided_generate."""
        n         = g.x.shape[0]
        label_col = torch.full((n, 1), float(g.y.item()))
        x6        = torch.cat([label_col, g.x], dim=1)    # [n, 6]

        adj = torch.zeros(n, n)
        ei  = g.edge_index
        if ei.shape[1] > 0:
            valid = ei[0] != ei[1]
            src, dst = ei[0][valid], ei[1][valid]
            inbounds = (src < n) & (dst < n)
            adj[src[inbounds], dst[inbounds]] = 1.0
            adj = (adj + adj.T).clamp(max=1.0)

        return {"x_dense": x6, "adj_dense": adj}

    # Compute training mean degree and density for calibrated generation
    mean_degs, densities = [], []
    for g in graphs:
        n       = g.x.shape[0]
        ei      = g.edge_index
        n_edges = ei.shape[1] // 2          # bidirectional → halve
        deg     = torch.zeros(n)
        if ei.shape[1] > 0:
            deg.scatter_add_(0, ei[0], torch.ones(ei.shape[1]))
        mean_degs.append(float(deg.mean()))
        if n > 1:
            densities.append(n_edges / (n * (n - 1) / 2))
    target_mean_degree = float(np.mean(mean_degs)) if mean_degs else None
    target_density     = float(np.mean(densities)) if densities else None

    illicit = [g for g in graphs if g.y.item() == 1]
    licit   = [g for g in graphs if g.y.item() == 0]
    seeds   = (
        _random.sample(illicit, min(n_gen // 2, len(illicit)))
      + _random.sample(licit,   min(n_gen // 2, len(licit)))
    )

    # ── Structural validity thresholds ────────────────────────────────────────
    train_sizes = [g.x.shape[0] for g in graphs]
    size_5th    = float(np.percentile(train_sizes, 5)) if train_sizes else 3.0
    density_ceil = 2.0 * target_density if target_density is not None else None

    gen_outputs, gen_embeddings = [], []
    n_discarded = 0

    with tqdm(total=len(seeds) * (t_start + 1),
              desc=f"Generating {len(seeds)} Elliptic graphs",
              unit="step", dynamic_ncols=True) as pbar:
        for i, seed_g in enumerate(seeds):
            pbar.set_postfix(graph=f"{i+1}/{len(seeds)}")

            # Convert to the dict format that guided_generate() expects.
            # It checks for "x_dense"/"adj_dense" first (fast path).
            seed_dict = _pyg_to_seed_dict(seed_g)

            x_out, adj_out, n_out, n_patches = guided_generate(
                seed_dict, encoder, probe, diff_model, diffusion,
                x_mean, x_std, H_all_n, device,
                target_label=target_label,
                t_start=t_start, guidance_scale=guidance_scale,
                novelty_weight=novelty_weight,
                guide_every=guide_every, guide_from=guide_from,
                degree_penalty=degree_penalty,
                target_mean_degree=target_mean_degree,
                adj_threshold=adj_threshold,
                target_density=target_density,
                max_nodes=max_nodes,
                pbar=pbar,
            )

            if n_patches > 0:
                print(f"  [validity] graph {i+1}: connectivity repair added {n_patches} edge(s)")

            # ── Structural validity checks ─────────────────────────────────
            gen_density = float(adj_out.sum()) / max(n_out * (n_out - 1), 1)
            if density_ceil is not None and gen_density > density_ceil:
                print(f"  [validity] graph {i+1} discarded: density {gen_density:.3f} "
                      f"> 2× target ({target_density:.3f})")
                n_discarded += 1
                continue
            if n_out < size_5th:
                print(f"  [validity] graph {i+1} discarded: "
                      f"n_nodes {n_out} < 5th-pct ({size_5th:.0f})")
                n_discarded += 1
                continue

            # Denormalise node features — col 0 (laundering flag) excluded
            # to match the encoder's input dimension after the leakage fix.
            x_cont_d = x_out[:, 1:] * x_std.cpu()[1:] + x_mean.cpu()[1:]
            deg_g    = adj_out.sum(dim=-1, keepdim=True)
            deg_n    = deg_g / deg_g.max().clamp(min=1.0)
            x_denorm = torch.cat([deg_n, x_cont_d[:, 1:]], dim=-1)   # [n, 5]

            ei_g = (adj_out > adj_threshold).nonzero(as_tuple=False).T.contiguous()
            bv_g = torch.zeros(n_out, dtype=torch.long)
            with torch.no_grad():
                h_g   = encoder(Data(x=x_denorm, edge_index=ei_g,
                                     batch=bv_g).to(device)).cpu()
                h_g_n = F.normalize(h_g, dim=-1)

            # Label is the guided target — we explicitly steered generation
            # toward target_label, so that is the correct label.
            gen_outputs.append((x_denorm, adj_out, n_out, target_label))
            gen_embeddings.append(h_g_n.squeeze(0).numpy())

    if n_discarded:
        print(f"  [validity] {n_discarded}/{len(seeds)} graphs discarded by structural checks")

    gen_embeddings = np.stack(gen_embeddings, axis=0)
    return gen_outputs, gen_embeddings, seeds
