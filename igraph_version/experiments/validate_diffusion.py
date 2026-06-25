"""
validate_diffusion.py
─────────────────────
Standalone validation experiment for the learned graph diffusion model.

Produces the following outputs in --out (default: experiments/results/diffusion_validation/):

  forward_corruption.png          node-feature heatmaps at t ∈ {0, T/4, T/2, 3T/4, T-1}
  encode_decode_features.png      corrupt a real graph to t=T_ENC, denoise back — features
  encode_decode_adj.png           same — adjacency matrices
  graph_viz.png                   NetworkX graph coloured by laundering label
  generation_features.png         DDIM-generated graph vs. reference — features
  generation_adj.png              same — adjacency matrices
  recon_quality.png               reconstruction MSE and edge-accuracy vs. noise level
  validation_report.txt           numeric summary of all metrics

Usage (run from igraph_version/ directory):
    python experiments/validate_diffusion.py
    python experiments/validate_diffusion.py --dataset HI-Small
    python experiments/validate_diffusion.py --dataset LI-Small --n-gen 4
    python experiments/validate_diffusion.py --ckpt checkpoints/HI-Small/diffusion/model.pt
"""

import argparse
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F

# ── path setup ────────────────────────────────────────────────────────────────
_HERE      = Path(__file__).resolve().parent          # experiments/
ROOT_DIR   = _HERE.parent                             # igraph_version/
DIFF_DIR   = ROOT_DIR / "diffusion"
CKPT_DIR   = ROOT_DIR / "checkpoints"
DATA_DIR   = ROOT_DIR / "data"

for _p in (str(ROOT_DIR), str(DIFF_DIR), str(ROOT_DIR / "simclr"), str(ROOT_DIR / "generation")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from diffusion.model import DiffusionGNN
from diffusion.masked_diffusion import (
    GaussianDiffusion, ModelMeanType, ModelVarType, LossType
)


def _cosine_beta_schedule(T, s=0.008, max_beta=0.999):
    alpha_bar = lambda t: math.cos((t + s) / (1.0 + s) * math.pi / 2) ** 2
    betas = []
    for i in range(T):
        betas.append(min(1.0 - alpha_bar((i + 1) / T) / alpha_bar(i / T), max_beta))
    return np.array(betas, dtype=np.float64)


def create_diffusion(T=500):
    return GaussianDiffusion(
        betas=_cosine_beta_schedule(T),
        model_mean_type=ModelMeanType.EPSILON,
        model_var_type=ModelVarType.FIXED_SMALL,
        loss_type=LossType.MSE,
        rescale_timesteps=True,
    )

FEATURE_NAMES = ["Laundering", "Degree", "Betweenness", "Clustering", "PageRank", "Assortativity"]

# ── helpers ───────────────────────────────────────────────────────────────────

def _load_model(ckpt_path: Path, device):
    ckpt      = torch.load(ckpt_path, map_location=device, weights_only=False)
    node_dim  = ckpt["model"]["input_proj.weight"].shape[1]
    n_layers  = ckpt.get("num_layers", 4)
    class_c   = ckpt.get("class_conditional", False)
    deg_c     = ckpt.get("degree_conditioning", "degree_proj.weight" in ckpt["model"])
    model     = DiffusionGNN(node_dim=node_dim, hidden_dim=128, num_layers=n_layers,
                             class_conditional=class_c, degree_conditioning=deg_c).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    x_mean    = ckpt["x_mean"].to(device)
    x_std     = ckpt["x_std"].to(device)
    max_nodes = ckpt.get("max_nodes", 64)
    print(f"Model loaded from {ckpt_path.name}  "
          f"(node_dim={node_dim}, layers={n_layers}, max_nodes={max_nodes})")
    return model, x_mean, x_std, max_nodes


def _clip_bounds(x_mean, x_std, device):
    F_cont = x_mean.shape[0] - 1
    lb     = torch.zeros(F_cont)
    ub     = torch.ones(F_cont)
    lb[-1] = -1.0                      # assortativity lower bound
    lo     = ((lb - x_mean[1:].cpu()) / x_std[1:].cpu()).to(device)
    hi     = ((ub - x_mean[1:].cpu()) / x_std[1:].cpu()).to(device)
    return lo, hi


def _to_norm(x, x_mean, x_std):
    """Normalize node features in-place (features 1+); feature 0 untouched."""
    x_n = x.clone()
    x_n[..., 1:] = (x[..., 1:] - x_mean[1:]) / x_std[1:]
    return x_n


def _denorm(x_n, x_mean, x_std):
    out = x_n.clone()
    out[..., 1:] = x_n[..., 1:] * x_std[1:] + x_mean[1:]
    return out


def _pick_sample(dataset, min_nodes=6, min_edges=5):
    """Return the first graph with enough nodes AND edges for meaningful metrics."""
    for x, adj in dataset:
        if x.shape[0] >= min_nodes and float(adj.sum()) >= min_edges:
            return x, adj
    for x, adj in dataset:   # relax edge constraint as fallback
        if x.shape[0] >= min_nodes:
            return x, adj
    return dataset[0]


def _denoise_from_t(model, diffusion, x_t, adj_t, mask, start_t, device, clip_bounds=None):
    x, adj = x_t.clone(), adj_t.clone()
    with torch.no_grad():
        for i in reversed(range(start_t + 1)):
            t   = torch.tensor([i] * x.shape[0], device=device)
            out = diffusion.p_sample(model, x, t,
                                     model_kwargs={"adj": adj, "node_mask": mask},
                                     clip_bounds=clip_bounds)
            x, adj = out["sample"], out["adj_sample"]
    return x, adj


def _adj_from_model(model, diffusion, x_out, a_out, mask, device, n_act,
                    target_density, adj_gamma=2.0):
    """
    Re-run the model at t=0 to get continuous adj_pred, then return both the raw
    probability matrix [n_act, n_act] and the density-calibrated binary adjacency.

    DDIM / p_sample_loop hard-threshold adjacency with adj_gamma compression, which
    produces all-zero outputs for sparse graphs (sparse prior adj_bias ≈ -2.25 keeps
    raw probs below 0.5).  Re-querying at t=0 gives the smooth probability matrix;
    top-k then selects exactly the edges needed to match the training density.

    Returns
    -------
    adj_bin  : [n_act, n_act] float32 binary (density-calibrated)
    adj_prob : [n_act, n_act] float32 raw sigmoid probabilities
    """
    with torch.no_grad():
        t_zero = torch.tensor([0], device=device)
        _, adj_pred, _ = model(
            x_out,
            diffusion._scale_timesteps(t_zero),
            adj=a_out,
            node_mask=mask,
        )
    raw  = adj_pred[0, :n_act, :n_act].cpu()
    binar = _density_calibrated_adj(adj_pred[0].cpu(), n_act, target_density, adj_gamma)
    return binar, raw


def _density_calibrated_adj(adj_prob, n_act, target_density, adj_gamma=2.0):
    """
    Apply gamma compression then keep only the top-k edges matching target_density.

    This mirrors the density-calibrated thresholding in guided_generate() and
    prevents the feedback loop where the adjacency decoder's oversmoothed node
    embeddings produce uniformly high edge probabilities → dense output.
    """
    ap = adj_prob[:n_act, :n_act].clamp(0.0, 1.0) ** adj_gamma
    ap.fill_diagonal_(0.0)
    n_keep  = max(1, round(target_density * n_act * (n_act - 1) / 2))
    ap_flat = ap.reshape(-1)
    n_total = len(ap_flat)
    if n_keep * 2 < n_total:
        # kthvalue(k) = k-th smallest; we want the value above which n_keep*2 entries remain
        threshold = max(float(ap_flat.kthvalue(n_total - n_keep * 2).values), 0.05)
    else:
        threshold = 0.05
    adj_bin = (ap > threshold).float()
    adj_bin.fill_diagonal_(0.0)
    return adj_bin


# ── diagnostic functions ──────────────────────────────────────────────────────

def diag_encode_decode(model, diffusion, x_norm, adj, mask, x_mean, x_std,
                        device, out_dir, t_enc=50, feat_names=None):
    """Corrupt a real graph to t_enc, reverse-diffuse back, compare."""
    if feat_names is None:
        feat_names = FEATURE_NAMES[:x_norm.shape[-1]]

    clip_lo, clip_hi = _clip_bounds(x_mean, x_std, device)

    with torch.no_grad():
        t_vec          = torch.tensor([t_enc], device=device)
        x_noisy, a_noisy = diffusion.q_sample(x_norm, t_vec, node_mask=mask, adj_start=adj)

    x_rec, a_rec = _denoise_from_t(model, diffusion, x_noisy, a_noisy, mask,
                                   t_enc, device, clip_bounds=(clip_lo, clip_hi))

    x_orig_d  = _denorm(x_norm[0].cpu(),  x_mean.cpu(), x_std.cpu())
    x_noisy_d = _denorm(x_noisy[0].cpu(), x_mean.cpu(), x_std.cpu())
    x_rec_d   = _denorm(x_rec[0].cpu(),   x_mean.cpu(), x_std.cpu())

    # feature heatmaps
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, title, data in zip(axes,
                                ["Original", f"Noisy (t={t_enc})", "Reconstructed"],
                                [x_orig_d, x_noisy_d, x_rec_d]):
        im = ax.imshow(data.T.numpy(), aspect="auto", cmap="RdBu_r")
        ax.set_title(title, fontsize=12)
        ax.set_yticks(range(len(feat_names)))
        ax.set_yticklabels(feat_names)
        ax.set_xlabel("Node index")
        plt.colorbar(im, ax=ax)
    mse_n = float(((x_norm - x_rec) ** 2).mean())
    plt.suptitle(f"Encode-decode (t={t_enc})  |  feature MSE (norm.): {mse_n:.4f}", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_dir / "encode_decode_features.png", dpi=100, bbox_inches="tight")
    plt.close(fig)

    # density-calibrated adjacency using raw adj_pred from a t=0 forward pass
    n_act       = int(mask[0].sum().item())
    real_dens   = float(adj[0, :n_act, :n_act].mean())
    adj_rec_bin, adj_rec_prob = _adj_from_model(model, diffusion, x_rec, a_rec, mask, device,
                                                 n_act, real_dens)
    adj_true   = adj[0, :n_act, :n_act].cpu().float()
    adj_pred_b = adj_rec_bin
    tp   = float((adj_pred_b * adj_true).sum())
    fp   = float((adj_pred_b * (1 - adj_true)).sum())
    fn   = float(((1 - adj_pred_b) * adj_true).sum())
    prec = tp / max(tp + fp, 1e-8)
    rec  = tp / max(tp + fn, 1e-8)
    f1   = 2 * prec * rec / max(prec + rec, 1e-8)
    acc  = float((adj_pred_b == adj_true).float().mean())

    # per-feature MSE in both normalized and original scale
    feat_mse_per      = ((x_norm - x_rec) ** 2).mean(dim=(0, 1)).cpu().tolist()
    feat_mse_orig_per = ((x_orig_d - x_rec_d) ** 2).mean(dim=0).tolist()

    # ── 4-panel adjacency figure ──────────────────────────────────────────────
    # Panel 1: original adjacency
    # Panel 2: noisy adjacency at t_enc
    # Panel 3: model's raw edge-probability heatmap (before thresholding)
    # Panel 4: error map — green=TP, orange=FP, red=FN, light-grey=TN
    orig_np = adj_true.numpy()
    pred_np = adj_pred_b.numpy()
    prob_np = adj_rec_prob.numpy()

    err_img = np.ones((*orig_np.shape, 3), dtype=np.float32) * 0.93   # light-grey TN
    tp_mask = (orig_np > 0.5) & (pred_np > 0.5)
    fp_mask = (orig_np < 0.5) & (pred_np > 0.5)
    fn_mask = (orig_np > 0.5) & (pred_np < 0.5)
    err_img[tp_mask] = [0.15, 0.70, 0.15]   # green  — correct edge
    err_img[fp_mask] = [1.00, 0.55, 0.00]   # orange — false positive
    err_img[fn_mask] = [0.85, 0.10, 0.10]   # red    — missed edge

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    axes[0].imshow(orig_np, cmap="Blues", vmin=0, vmax=1)
    axes[0].set_title("Original", fontsize=11)

    axes[1].imshow(a_noisy[0, :n_act, :n_act].cpu().float().numpy(), cmap="Blues", vmin=0, vmax=1)
    axes[1].set_title(f"Noisy (t={t_enc})", fontsize=11)

    im = axes[2].imshow(prob_np, cmap="YlOrRd", vmin=0, vmax=1)
    axes[2].set_title("Model probability\n(raw, before threshold)", fontsize=11)
    plt.colorbar(im, ax=axes[2], fraction=0.046)

    axes[3].imshow(err_img)
    axes[3].set_title("Error map", fontsize=11)
    legend_patches = [
        mpatches.Patch(color=[0.15, 0.70, 0.15], label=f"TP ({int(tp)})"),
        mpatches.Patch(color=[1.00, 0.55, 0.00], label=f"FP ({int(fp)})"),
        mpatches.Patch(color=[0.85, 0.10, 0.10], label=f"FN ({int(fn)})"),
        mpatches.Patch(color=[0.93, 0.93, 0.93], label="TN"),
    ]
    axes[3].legend(handles=legend_patches, loc="lower right", fontsize=8, framealpha=0.9)

    for ax in axes:
        ax.axis("off")
    plt.suptitle(
        f"Adjacency reconstruction (t={t_enc})  |  F1: {f1:.3f}  "
        f"(prec: {prec:.3f}, rec: {rec:.3f})  |  density: real={real_dens:.3f}",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(out_dir / "encode_decode_adj.png", dpi=100, bbox_inches="tight")
    plt.close(fig)

    return {
        "t_enc": t_enc,
        "feat_mse_norm": mse_n,
        "feat_mse_per": feat_mse_per,
        "feat_mse_orig_per": feat_mse_orig_per,
        "feat_names": feat_names,
        "adj_accuracy": acc, "adj_precision": prec, "adj_recall": rec, "adj_f1": f1,
    }


def diag_generation(model, diffusion, x_norm_ref, adj_ref, mask, x_mean, x_std,
                     device, out_dir, n_gen=4, ddim_steps=50, feat_names=None):
    """Generate graphs from pure noise (DDIM) and compare statistics to real graphs."""
    if feat_names is None:
        feat_names = FEATURE_NAMES[:x_norm_ref.shape[-1]]

    clip_lo, clip_hi = _clip_bounds(x_mean, x_std, device)
    n_act      = int(mask[0].sum().item())
    # compute density BEFORE the generation loop so it can be used for calibration
    real_dens  = float(adj_ref[0, :n_act, :n_act].mean().item())
    adj_init_p = max(real_dens, 0.05)

    gen_feats, gen_dens, last_adj_cal = [], [], None
    with torch.no_grad():
        for _ in range(n_gen):
            x_g, a_g = diffusion.ddim_sample_loop(
                model,
                shape=x_norm_ref.shape,
                adj_shape=adj_ref.shape,
                model_kwargs={"node_mask": mask},
                device=device,
                adj_init_p=adj_init_p,
                adj_gamma=2.0,
                ddim_steps=ddim_steps,
                eta=0.0,
                clip_bounds=(clip_lo, clip_hi),
            )
            x_d = _denorm(x_g[0].cpu(), x_mean.cpu(), x_std.cpu())
            gen_feats.append(x_d.numpy())
            # re-query model at t=0 for raw adj_pred, then apply density calibration.
            # a_g from ddim_sample_loop is already thresholded binary (not raw probs),
            # so using it directly for top-k selection misses most edges in sparse graphs.
            last_adj_cal, _ = _adj_from_model(model, diffusion, x_g, a_g, mask, device,
                                               n_act, real_dens)
            gen_dens.append(float(last_adj_cal.mean()))

    gen_feats = np.stack(gen_feats, axis=0)    # [n_gen, N, F]
    gen_mean  = gen_feats.mean(axis=(0, 1))    # [F]
    gen_std   = gen_feats.std(axis=(0, 1))

    x_real_d  = _denorm(x_norm_ref[0].cpu(), x_mean.cpu(), x_std.cpu())
    real_mean  = x_real_d.numpy().mean(axis=0)
    real_std   = x_real_d.numpy().std(axis=0)

    # feature heatmap: last generated vs reference
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, title, data in zip(axes,
                                ["Reference (real graph)", f"Generated (DDIM-{ddim_steps})"],
                                [x_real_d.numpy(), gen_feats[-1]]):
        im = ax.imshow(data.T, aspect="auto", cmap="RdBu_r")
        ax.set_title(title, fontsize=12)
        ax.set_yticks(range(len(feat_names)))
        ax.set_yticklabels(feat_names)
        ax.set_xlabel("Node index")
        plt.colorbar(im, ax=ax)
    plt.suptitle(f"Full generation (DDIM-{ddim_steps}): node features in original scale",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(out_dir / "generation_features.png", dpi=100, bbox_inches="tight")
    plt.close(fig)

    # adjacency: real vs last generated (density-calibrated)
    gen_density_mean = float(np.mean(gen_dens))
    adj_viz = last_adj_cal if last_adj_cal is not None else torch.zeros(n_act, n_act)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, title, mat in zip(
        axes,
        ["Real adjacency", f"Generated (density-calibrated, target={real_dens:.3f})"],
        [adj_ref[0, :n_act, :n_act].cpu().float(), adj_viz],
    ):
        ax.imshow(mat.numpy(), cmap="Blues", vmin=0, vmax=1)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    plt.suptitle(
        f"Adjacency  |  real density: {real_dens:.3f}  gen density: {gen_density_mean:.3f}",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(out_dir / "generation_adj.png", dpi=100, bbox_inches="tight")
    plt.close(fig)

    # feature distribution histograms: real node values vs all generated node values
    n_feat_show = min(len(feat_names), 6)
    cols = 3
    rows = math.ceil(n_feat_show / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes_flat  = list(axes.flat) if hasattr(axes, "flat") else [axes]
    real_flat  = x_real_d.numpy().reshape(-1, x_real_d.shape[-1])
    gen_flat   = gen_feats.reshape(-1, gen_feats.shape[-1])
    for i, (ax, name) in enumerate(zip(axes_flat, feat_names[:n_feat_show])):
        ax.hist(real_flat[:, i], bins=20, alpha=0.6, color="steelblue",
                label="Real", density=True)
        ax.hist(gen_flat[:, i],  bins=20, alpha=0.6, color="darkorange",
                label="Generated", density=True)
        ax.set_title(name, fontsize=10)
        ax.legend(fontsize=8)
    for ax in axes_flat[n_feat_show:]:
        ax.set_visible(False)
    plt.suptitle("Feature distributions: real vs. generated", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_dir / "generation_histograms.png", dpi=100, bbox_inches="tight")
    plt.close(fig)

    return {
        "n_gen": n_gen, "ddim_steps": ddim_steps,
        "real_density": real_dens, "gen_density": gen_density_mean,
        "real_mean": real_mean, "gen_mean": gen_mean,
        "real_std": real_std, "gen_std": gen_std,
        "feat_names": feat_names,
    }


def diag_recon_quality(model, diffusion, x_norm, adj, mask, x_mean, x_std,
                        device, out_dir, T=500, n_probe=15, k=4, feat_names=None):
    """
    Lightweight reconstruction-quality curve: for n_probe evenly-spaced timesteps,
    corrupt the real graph and immediately denoise one step to predict x_0.
    Reports feature MSE and edge accuracy, showing how well the model has learned
    at each noise level.
    """
    clip_lo, clip_hi = _clip_bounds(x_mean, x_std, device)
    timesteps = np.linspace(0, T - 1, n_probe, dtype=int).tolist()
    mses, accs = [], []

    n_act      = int(mask[0].sum().item())
    real_dens  = float(adj[0, :n_act, :n_act].mean())
    adj_true   = adj[0, :n_act, :n_act].cpu().float()

    model.eval()
    with torch.no_grad():
        for t_val in timesteps:
            t_vec = torch.tensor([t_val], device=device)
            t_scaled = diffusion._scale_timesteps(t_vec)
            batch_mse, batch_f1 = [], []
            for _ in range(k):
                x_t, a_t = diffusion.q_sample(x_norm, t_vec, node_mask=mask, adj_start=adj)
                eps_pred, adj_pred, _ = model(x_t, t_scaled, adj=a_t, node_mask=mask)

                # recover predicted x0 for continuous features
                x0_cont = diffusion._predict_xstart_from_eps(
                    x_t[..., 1:], t_vec, eps_pred[..., 1:]
                ).clamp(clip_lo, clip_hi)
                x0_bin  = eps_pred[..., 0:1].clamp(0, 1)
                x0_pred = torch.cat([x0_bin, x0_cont], dim=-1)

                mse = float(((x0_pred - x_norm) ** 2).mean())
                # density-calibrated threshold instead of >0.5 (which always gives 0
                # for sparse graphs because adj_bias ≈ -2.25 keeps raw probs below 0.5)
                a_bin   = _density_calibrated_adj(adj_pred[0].cpu(), n_act, real_dens)
                tp  = float((a_bin * adj_true).sum())
                fp  = float((a_bin * (1 - adj_true)).sum())
                fn  = float(((1 - a_bin) * adj_true).sum())
                p   = tp / max(tp + fp, 1e-8)
                r   = tp / max(tp + fn, 1e-8)
                f1  = 2 * p * r / max(p + r, 1e-8)
                batch_mse.append(mse)
                batch_f1.append(f1)
            mses.append(float(np.mean(batch_mse)))
            accs.append(float(np.mean(batch_f1)))

    # At t=T-1, ᾱ_t → 0 so _predict_xstart_from_eps divides by sqrt(ᾱ) ≈ 0,
    # amplifying any eps prediction error to ~∞. Cap the y-axis at 2× the
    # second-largest value so the rest of the curve stays readable.
    mse_finite = sorted(mses)[:-1]   # all but the max
    mse_cap    = max(mse_finite) * 2 if mse_finite else max(mses)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fig.suptitle("Reconstruction Quality vs. Noise Level", fontsize=16)
    ax1.plot(timesteps, mses, "o-", color="steelblue")
    ax1.set_ylim(0, mse_cap)
    ax1.set_ylabel("Feature MSE (norm. space)", fontsize=14)
    ax2.plot(timesteps, accs, "o-", color="darkorange")
    ax2.set_ylabel("Edge F1", fontsize=14)
    ax2.set_xlabel("Timestep $t$", fontsize=14)
    fig.align_ylabels([ax1, ax2])
    plt.tight_layout()
    plt.savefig(out_dir / "recon_quality.png", dpi=100, bbox_inches="tight")
    plt.close(fig)

    return {"timesteps": timesteps, "mse": mses, "edge_f1": accs}


def diag_graph_viz(model, diffusion, x_norm, adj, x_orig, mask,
                   x_mean, x_std, device, out_dir, t_viz=50, feat_names=None):
    """NetworkX spring-layout graph coloured by original and reconstructed laundering labels."""
    if feat_names is None:
        feat_names = FEATURE_NAMES[:x_norm.shape[-1]]

    clip_lo, clip_hi = _clip_bounds(x_mean, x_std, device)
    with torch.no_grad():
        t_vec = torch.tensor([t_viz], device=device)
        x_noisy, a_noisy = diffusion.q_sample(x_norm, t_vec, node_mask=mask, adj_start=adj)

    # use visualizations helper if available, otherwise roll our own
    try:
        from diffusion.visualizations import _denoise_from_t as _d
        x_rec, _ = _d(model, diffusion, x_noisy, a_noisy, mask, t_viz, device)
    except Exception:
        x_rec, _ = _denoise_from_t(model, diffusion, x_noisy, a_noisy, mask, t_viz, device)

    n = x_orig.shape[0]
    G = nx.from_numpy_array(adj[0, :n, :n].cpu().numpy())
    pos = nx.spring_layout(G, seed=42)

    orig_labels  = x_orig[:, 0].numpy()
    rec_labels   = x_rec[0, :n, 0].detach().cpu().float().numpy()
    rec_binary   = (rec_labels > 0.5).astype(float)
    correct      = float((rec_binary == (orig_labels > 0.5)).mean())

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, labels, title in zip(axes,
                                  [orig_labels, rec_binary],
                                  ["Original laundering labels",
                                   f"Reconstructed (t={t_viz})"]):
        colors = ["red" if l > 0.5 else "steelblue" for l in labels]
        nx.draw_networkx(G, pos=pos, ax=ax, node_color=colors, node_size=400,
                         with_labels=True, font_size=7, edge_color="grey", alpha=0.85)
        ax.set_title(title, fontsize=12)
        ax.axis("off")

    legend = [mpatches.Patch(facecolor="red",       label="Laundering"),
              mpatches.Patch(facecolor="steelblue", label="Clean")]
    fig.legend(handles=legend, loc="lower center", ncol=2, fontsize=11)
    plt.suptitle(f"Laundering node detection  |  accuracy: {correct:.2%}", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_dir / "graph_viz.png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    return {"laund_accuracy": correct}


def diag_forward_corruption(diffusion, x_norm, mask, T, device, out_dir, feat_names=None):
    """Heatmap of node features corrupted to five different noise levels."""
    if feat_names is None:
        feat_names = FEATURE_NAMES[:x_norm.shape[-1]]

    checkpoints = [0, T // 4, T // 2, 3 * T // 4, T - 1]
    fig, axes = plt.subplots(len(checkpoints), 1, figsize=(10, 2.5 * len(checkpoints)))
    with torch.no_grad():
        for ax, t_val in zip(axes, checkpoints):
            t_t = torch.tensor([t_val], device=device)
            x_t = diffusion.q_sample(x_norm, t_t, node_mask=mask)
            im  = ax.imshow(x_t[0].cpu().float().numpy().T, aspect="auto",
                            cmap="RdBu_r", vmin=-3, vmax=3)
            ax.set_title(f"t = {t_val}", fontsize=11)
            ax.set_yticks(range(len(feat_names)))
            ax.set_yticklabels(feat_names)
            ax.set_xlabel("Node index")
            plt.colorbar(im, ax=ax)
    plt.suptitle("Forward diffusion: feature corruption at increasing noise levels", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_dir / "forward_corruption.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


# ── report writer ─────────────────────────────────────────────────────────────

def write_report(out_dir, ed_stats, gen_stats, rq_stats, viz_stats, ckpt_path, dataset_key):
    lines = [
        "=" * 60,
        "Diffusion Model Validation Report",
        f"Checkpoint : {ckpt_path}",
        f"Dataset    : {dataset_key}",
        "=" * 60,
        "",
        "── Encode-Decode Reconstruction ────────────────────────",
        f"  Noise level (t)          : {ed_stats['t_enc']}",
        f"  Feature MSE (norm. avg)  : {ed_stats['feat_mse_norm']:.4f}",
        f"  Edge F1                  : {ed_stats['adj_f1']:.3f}",
        f"  Edge precision           : {ed_stats['adj_precision']:.3f}",
        f"  Edge recall              : {ed_stats['adj_recall']:.3f}",
        f"  Edge accuracy (ref only) : {ed_stats['adj_accuracy']:.2%}",
        "",
        f"  {'Feature':<14} {'MSE (norm)':>12} {'MSE (orig)':>12}",
        "  " + "-" * 40,
    ]
    for name, mse_n, mse_o in zip(
        ed_stats.get("feat_names", []),
        ed_stats.get("feat_mse_per", []),
        ed_stats.get("feat_mse_orig_per", []),
    ):
        lines.append(f"  {name:<14} {mse_n:>12.4f} {mse_o:>12.4f}")
    lines += [
        "",
        "── Full Generation (DDIM) ──────────────────────────────",
        f"  Graphs generated    : {gen_stats['n_gen']}",
        f"  DDIM steps          : {gen_stats['ddim_steps']}",
        f"  Edge density (real) : {gen_stats['real_density']:.4f}",
        f"  Edge density (gen)  : {gen_stats['gen_density']:.4f}",
        "",
        f"  {'Feature':<14} {'real mean':>10} {'gen mean':>10} {'real std':>10} {'gen std':>10} {'|Δmean|':>10}",
        "  " + "-" * 66,
    ]
    for i, name in enumerate(gen_stats["feat_names"]):
        lines.append(
            f"  {name:<14} {gen_stats['real_mean'][i]:>10.3f} {gen_stats['gen_mean'][i]:>10.3f} "
            f"{gen_stats['real_std'][i]:>10.3f} {gen_stats['gen_std'][i]:>10.3f} "
            f"{abs(gen_stats['real_mean'][i] - gen_stats['gen_mean'][i]):>10.3f}"
        )
    lines += [
        "",
        "── Reconstruction Quality vs. Noise Level ──────────────",
        f"  {'Timestep':>10} {'MSE (norm.)':>14} {'Edge F1':>14}",
        "  " + "-" * 40,
    ]
    for t, m, a in zip(rq_stats["timesteps"], rq_stats["mse"], rq_stats["edge_f1"]):
        lines.append(f"  {t:>10d} {m:>14.4f} {a:>14.2%}")
    lines += [
        "",
        "── Graph Visualization ─────────────────────────────────",
        f"  Laundering node detection accuracy: {viz_stats['laund_accuracy']:.2%}",
        "",
        "Output files saved to: " + str(out_dir),
        "=" * 60,
    ]
    report_path = out_dir / "validation_report.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    # print with safe fallback for terminals that can't render box-drawing chars
    try:
        print("\n".join(lines))
    except UnicodeEncodeError:
        safe = "\n".join(lines).encode("ascii", errors="replace").decode("ascii")
        print(safe)


# ── main ──────────────────────────────────────────────────────────────────────

DATASET_MAP = {
    "HI-Small" : ("HI-Small", "cached_dataset_HI-Small_Trans.pt"),
    "HI-Medium": ("HI-Medium", "cached_dataset_HI-Medium_Trans.pt"),
    "LI-Small" : ("LI-Small",  "cached_dataset_LI-Small_Trans.pt"),
    "LI-Large" : ("LI-Large",  "cached_dataset_LI-Large_Trans.pt"),
}


def main():
    parser = argparse.ArgumentParser(description="Validate the learned diffusion model.")
    parser.add_argument("--dataset", default="HI-Small", choices=list(DATASET_MAP.keys()),
                        help="Which IBM dataset checkpoint to use (default: HI-Small)")
    parser.add_argument("--ckpt", default=None,
                        help="Explicit path to a model.pt checkpoint (overrides --dataset)")
    parser.add_argument("--n-gen", type=int, default=4,
                        help="Number of graphs to generate in the generation diagnostic")
    parser.add_argument("--ddim-steps", type=int, default=50,
                        help="DDIM denoising steps for full generation (default: 50)")
    parser.add_argument("--t-enc", type=int, default=150,
                        help="Noise level for encode-decode test (default: 150)")
    parser.add_argument("--out", default=None,
                        help="Output directory (default: experiments/results/diffusion_validation/)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    args = parser.parse_args()

    device_str = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    device     = torch.device(device_str)
    print(f"Device: {device}")

    # ── checkpoint ────────────────────────────────────────────────────────────
    if args.ckpt:
        ckpt_path = Path(args.ckpt)
    else:
        ds_key, _ = DATASET_MAP[args.dataset]
        ckpt_path = CKPT_DIR / ds_key / "diffusion" / "model.pt"
        if not ckpt_path.exists():
            ckpt_path = CKPT_DIR / "diffusion_ibm" / "model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint found at {ckpt_path}")

    model, x_mean, x_std, max_nodes = _load_model(ckpt_path, device)
    T         = 500
    diffusion = create_diffusion(T=T)

    # ── dataset ───────────────────────────────────────────────────────────────
    _, cache_file = DATASET_MAP[args.dataset]
    cache_path    = DATA_DIR / cache_file
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Cached dataset not found at {cache_path}.\n"
            "Run the preprocessing step first (e.g. python diffusion/train.py)."
        )
    raw     = torch.load(str(cache_path), weights_only=False)
    dataset = [(x.float(), adj.float()) for x, adj in raw if 0 < x.shape[0] <= max_nodes]
    print(f"Dataset loaded: {len(dataset)} graphs  (max_nodes={max_nodes})")

    x_orig, adj_orig = _pick_sample(dataset, min_nodes=6)
    n = x_orig.shape[0]

    # adapt feature dimension to match model (in case dataset has more features)
    node_dim = model.input_proj.weight.shape[1]
    x_orig   = x_orig[:, :node_dim]
    x_mean_c = x_mean[:node_dim]
    x_std_c  = x_std[:node_dim]

    feat_names = FEATURE_NAMES[:node_dim] if node_dim <= len(FEATURE_NAMES) else \
                 FEATURE_NAMES + [f"feat{i}" for i in range(len(FEATURE_NAMES), node_dim)]

    x_batch   = x_orig.unsqueeze(0).float().to(device)
    adj_batch = adj_orig.unsqueeze(0).float().to(device)
    mask      = torch.ones(1, n, device=device)

    x_norm = _to_norm(x_batch, x_mean_c, x_std_c)

    # ── output directory ──────────────────────────────────────────────────────
    out_dir = Path(args.out) if args.out else _HERE / "results" / "diffusion_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")

    # ── diagnostics ───────────────────────────────────────────────────────────
    print("\n[1/5] Forward corruption heatmap …")
    diag_forward_corruption(diffusion, x_norm, mask, T, device, out_dir, feat_names)

    print(f"[2/5] Encode-decode (t={args.t_enc}) …")
    ed_stats = diag_encode_decode(model, diffusion, x_norm, adj_batch, mask,
                                   x_mean_c, x_std_c, device, out_dir,
                                   t_enc=args.t_enc, feat_names=feat_names)

    print("[3/5] Graph visualization …")
    viz_stats = diag_graph_viz(model, diffusion, x_norm, adj_batch, x_orig,
                                mask, x_mean_c, x_std_c, device, out_dir,
                                t_viz=args.t_enc, feat_names=feat_names)

    print(f"[4/5] Full generation (DDIM-{args.ddim_steps}, n_gen={args.n_gen}) …")
    gen_stats = diag_generation(model, diffusion, x_norm, adj_batch, mask,
                                 x_mean_c, x_std_c, device, out_dir,
                                 n_gen=args.n_gen, ddim_steps=args.ddim_steps,
                                 feat_names=feat_names)

    print("[5/5] Reconstruction quality vs. noise level …")
    rq_stats = diag_recon_quality(model, diffusion, x_norm, adj_batch, mask,
                                   x_mean_c, x_std_c, device, out_dir,
                                   T=T, n_probe=15, k=4, feat_names=feat_names)

    write_report(out_dir, ed_stats, gen_stats, rq_stats, viz_stats, ckpt_path, args.dataset)
    print(f"\nAll outputs saved to {out_dir}/")


if __name__ == "__main__":
    main()
