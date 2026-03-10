"""
Visualization functions for the graph diffusion model.
All functions save figures to a directory and do not display interactively.
"""

import os

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for scripts
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
import torch


FEATURE_NAMES = ["Laundering", "Degree", "Depth", "Betweenness", "Clustering", "PageRank"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _denorm(x_norm, x_mean, x_std):
    """[*, N, F] normalized → original scale."""
    out = x_norm.clone()
    out[..., 1:] = x_norm[..., 1:] * x_std.cpu()[1:] + x_mean.cpu()[1:]
    return out


def _denoise_from_t(model, diffusion, x_t, adj_t, node_mask, start_t, device):
    """Run reverse diffusion from timestep `start_t` down to 0."""
    x   = x_t.clone()
    adj = adj_t.clone()
    with torch.no_grad():
        for i in reversed(range(start_t + 1)):
            t = torch.tensor([i] * x.shape[0], device=device)
            out = diffusion.p_sample(
                model, x, t,
                model_kwargs={"adj": adj, "node_mask": node_mask},
            )
            x   = out["sample"]
            adj = out["adj_sample"]
    return x, adj


def _pick_sample(dataset, min_nodes=6):
    """Return the first dataset entry with at least `min_nodes` nodes."""
    for x, adj in dataset:
        if x.shape[0] >= min_nodes:
            return x, adj
    return dataset[0]


# ---------------------------------------------------------------------------
# Public visualization functions
# ---------------------------------------------------------------------------

def plot_forward_corruption(diffusion, x_batch_norm, node_mask, timesteps, device, save_path,
                            feature_names=None):
    """
    Plot node features at several noise levels along the forward diffusion chain.

    Args:
        diffusion:      Diffusion object.
        x_batch_norm:   Normalized node features [1, N, F].
        node_mask:      [1, N] binary mask.
        timesteps:      Total number of diffusion timesteps.
        device:         Torch device string.
        save_path:      File path to save the figure.
        feature_names:  List of feature name strings.
    """
    if feature_names is None:
        feature_names = FEATURE_NAMES

    checkpoints = [0,
                   timesteps // 4,
                   timesteps // 2,
                   3 * timesteps // 4,
                   timesteps - 1]

    fig, axes = plt.subplots(len(checkpoints), 1, figsize=(10, 2.5 * len(checkpoints)))

    with torch.no_grad():
        for ax, t_val in zip(axes, checkpoints):
            t_tensor = torch.tensor([t_val], device=device)
            x_t = diffusion.q_sample(x_batch_norm, t_tensor, node_mask=node_mask)
            data = x_t[0].cpu().float().numpy()
            im = ax.imshow(data.T, aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3)
            ax.set_title(f"t = {t_val}", fontsize=11)
            ax.set_yticks(range(len(feature_names)))
            ax.set_yticklabels(feature_names)
            ax.set_xlabel("Node index")
            plt.colorbar(im, ax=ax)

    plt.suptitle("Forward diffusion: node features at increasing noise levels (normalised space)",
                 fontsize=12)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_encode_decode(model, diffusion, x_batch_norm, adj_batch, node_mask,
                       x_mean, x_std, device, save_path,
                       t_encode=50, feature_names=None):
    """
    Corrupt a graph to timestep `t_encode`, then run the reverse chain back to 0.
    Saves a side-by-side comparison of original / noisy / reconstructed features and adjacency.

    Args:
        model, diffusion:  Trained model and diffusion object.
        x_batch_norm:      Normalised node features [1, N, F].
        adj_batch:         Adjacency [1, N, N].
        node_mask:         [1, N] binary mask.
        x_mean, x_std:     Normalisation stats (CPU tensors).
        device:            Torch device string.
        save_path:         File path to save the figure.
        t_encode:          Timestep to corrupt to before denoising.
        feature_names:     List of feature name strings.
    """
    if feature_names is None:
        feature_names = FEATURE_NAMES

    model.eval()
    with torch.no_grad():
        t_tensor = torch.tensor([t_encode], device=device)
        x_noisy_norm, adj_noisy = diffusion.q_sample(
            x_batch_norm, t_tensor, node_mask=node_mask, adj_start=adj_batch
        )

    x_recon_norm, adj_recon = _denoise_from_t(
        model, diffusion, x_noisy_norm, adj_noisy, node_mask, t_encode, device
    )

    x_orig_d  = _denorm(x_batch_norm[0].cpu(), x_mean, x_std)
    x_noisy_d = _denorm(x_noisy_norm[0].cpu(), x_mean, x_std)
    x_recon_d = _denorm(x_recon_norm[0].cpu(), x_mean, x_std)

    # --- node features ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, title, data in zip(
        axes,
        ["Original", f"Noisy (t={t_encode})", "Reconstructed"],
        [x_orig_d, x_noisy_d, x_recon_d],
    ):
        im = ax.imshow(data.T.numpy(), aspect="auto", cmap="RdBu_r")
        ax.set_title(title, fontsize=12)
        ax.set_yticks(range(len(feature_names)))
        ax.set_yticklabels(feature_names)
        ax.set_xlabel("Node index")
        plt.colorbar(im, ax=ax)

    mse_norm = ((x_batch_norm - x_recon_norm) ** 2).mean().item()
    plt.suptitle(f"Encode → decode (t={t_encode})  |  Node MSE (norm. space): {mse_norm:.4f}",
                 fontsize=13)
    plt.tight_layout()
    feat_path = save_path.replace(".png", "_features.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(feat_path, dpi=100, bbox_inches="tight")
    plt.close(fig)

    # --- adjacency ---
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, title, mat in zip(
        axes,
        ["Original adj", f"Noisy adj (t={t_encode})", "Reconstructed adj"],
        [adj_batch, adj_noisy, adj_recon],
    ):
        ax.imshow(mat[0].cpu().float().numpy(), cmap="Blues", vmin=0, vmax=1)
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    adj_acc = ((adj_recon[0] > 0.5).float() == adj_batch[0]).float().mean().item()
    plt.suptitle(f"Adjacency reconstruction  |  Edge accuracy: {adj_acc:.2%}", fontsize=13)
    plt.tight_layout()
    adj_path = save_path.replace(".png", "_adj.png")
    plt.savefig(adj_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_full_generation(model, diffusion, x_batch_norm, adj_batch, node_mask,
                         x_mean, x_std, device, save_path, feature_names=None):
    """
    Generate a new graph from pure noise and compare to the reference graph.

    Args:
        model, diffusion:  Trained model and diffusion object.
        x_batch_norm:      Reference normalised node features [1, N, F] (for shape).
        adj_batch:         Reference adjacency [1, N, N].
        node_mask:         [1, N] binary mask.
        x_mean, x_std:     Normalisation stats (CPU tensors).
        device:            Torch device string.
        save_path:         File path to save the figure.
        feature_names:     List of feature name strings.
    """
    if feature_names is None:
        feature_names = FEATURE_NAMES

    model.eval()
    with torch.no_grad():
        x_gen_norm, adj_gen = diffusion.p_sample_loop(
            model,
            x_batch_norm.shape,
            adj_shape=adj_batch.shape,
            model_kwargs={"node_mask": node_mask},
            device=device,
        )

    x_orig_d = _denorm(x_batch_norm[0].cpu(), x_mean, x_std)
    x_gen_d  = _denorm(x_gen_norm[0].cpu(), x_mean, x_std)

    # --- node features ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, title, data in zip(
        axes,
        ["Original node features", "Generated node features"],
        [x_orig_d, x_gen_d],
    ):
        im = ax.imshow(data.T.numpy(), aspect="auto", cmap="RdBu_r")
        ax.set_title(title, fontsize=12)
        ax.set_yticks(range(len(feature_names)))
        ax.set_yticklabels(feature_names)
        ax.set_xlabel("Node index")
        plt.colorbar(im, ax=ax)
    plt.suptitle("Full generation: node features (original scale)", fontsize=13)
    plt.tight_layout()
    feat_path = save_path.replace(".png", "_features.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(feat_path, dpi=100, bbox_inches="tight")
    plt.close(fig)

    # --- adjacency ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, title, mat in zip(
        axes,
        ["Original adj", "Generated adj"],
        [adj_batch, (adj_gen > 0.5).float()],
    ):
        ax.imshow(mat[0].cpu().float().numpy(), cmap="Blues", vmin=0, vmax=1)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    plt.suptitle("Full generation: adjacency matrix", fontsize=13)
    plt.tight_layout()
    adj_path = save_path.replace(".png", "_adj.png")
    plt.savefig(adj_path, dpi=100, bbox_inches="tight")
    plt.close(fig)

    # --- stats table ---
    orig_np = x_orig_d.numpy()
    gen_np  = x_gen_d.numpy()
    orig_density = adj_batch[0].mean().item()
    gen_density  = (adj_gen[0] > 0.5).float().mean().item()

    stats_path = save_path.replace(".png", "_stats.txt")
    with open(stats_path, "w") as f:
        f.write(f"{'Feature':<14} {'orig mean':>10} {'gen mean':>10} {'orig std':>10} {'gen std':>10}\n")
        for i, name in enumerate(feature_names):
            f.write(f"{name:<14} {orig_np[:, i].mean():>10.3f} {gen_np[:, i].mean():>10.3f} "
                    f"{orig_np[:, i].std():>10.3f} {gen_np[:, i].std():>10.3f}\n")
        f.write(f"\nEdge density — original: {orig_density:.3f}  generated: {gen_density:.3f}\n")


def plot_loss_curve(model, diffusion, x_batch_norm, adj_batch, node_mask,
                    timesteps, adj_loss_w, deg_loss_w, device, save_path,
                    laund_loss_w=2.0, k_samples=8):
    """
    Plot the denoising loss as a function of timestep t.
    Each timestep is averaged over `k_samples` noise draws to reduce variance.

    Args:
        model, diffusion:   Trained model and diffusion object.
        x_batch_norm:       Normalised node features [1, N, F].
        adj_batch:          Adjacency [1, N, N].
        node_mask:          [1, N] binary mask.
        timesteps:          Total number of diffusion timesteps.
        adj_loss_w:         Adjacency loss weight passed to training_losses.
        deg_loss_w:         Degree loss weight passed to training_losses.
        device:             Torch device string.
        save_path:          File path to save the figure.
        k_samples:          Noise samples averaged per timestep.
    """
    model.eval()
    losses_per_t = []

    with torch.no_grad():
        for t_val in range(timesteps):
            t_tensor = torch.tensor([t_val], device=device)
            k_losses = []
            for _ in range(k_samples):
                loss_dict = diffusion.training_losses(
                    model,
                    x_start=x_batch_norm,
                    t=t_tensor,
                    adj_start=adj_batch,
                    model_kwargs={"node_mask": node_mask},
                    adj_loss_weight=adj_loss_w,
                    deg_loss_weight=deg_loss_w,
                    laund_loss_weight=laund_loss_w,
                )
                k_losses.append(loss_dict["loss"].item())
            losses_per_t.append(sum(k_losses) / k_samples)

    window   = max(1, timesteps // 20)
    smoothed = np.convolve(losses_per_t, np.ones(window) / window, mode="valid")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(timesteps), losses_per_t, alpha=0.35, linewidth=1,
            color="steelblue", label=f"Loss per t (avg {k_samples} samples)")
    ax.plot(range(window - 1, timesteps), smoothed, linewidth=2,
            color="red", label=f"Smoothed (w={window})")
    ax.set_xlabel("Timestep t")
    ax.set_ylabel("Combined loss")
    ax.set_title(f"Denoising loss vs timestep (K={k_samples} samples/t)")
    ax.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_graph_viz(model, diffusion, x_batch_norm, adj_batch, adj_orig, x_orig,
                   node_mask, device, save_path, t_viz=50, feature_names=None):
    """
    Draw the original graph and the encode→decode reconstruction side by side,
    with nodes coloured by laundering label.

    Args:
        model, diffusion:   Trained model and diffusion object.
        x_batch_norm:       Normalised node features [1, N, F].
        adj_batch:          Adjacency [1, N, N].
        adj_orig:           CPU adjacency [N, N] (for NetworkX graph).
        x_orig:             CPU raw node features [N, F] (for original labels).
        node_mask:          [1, N] binary mask.
        device:             Torch device string.
        save_path:          File path to save the figure.
        t_viz:              Timestep used for encode→decode.
        feature_names:      List of feature name strings.
    """
    if feature_names is None:
        feature_names = FEATURE_NAMES

    model.eval()
    with torch.no_grad():
        t_tensor = torch.tensor([t_viz], device=device)
        x_noisy_viz, adj_noisy_viz = diffusion.q_sample(
            x_batch_norm, t_tensor, node_mask=node_mask, adj_start=adj_batch
        )

    x_recon_viz, _ = _denoise_from_t(
        model, diffusion, x_noisy_viz, adj_noisy_viz, node_mask, t_viz, device
    )

    G   = nx.from_numpy_array(adj_orig.cpu().numpy())
    pos = nx.spring_layout(G, seed=42)

    orig_labels  = x_orig[:, 0].numpy()
    recon_labels = x_recon_viz[0, :, 0].cpu().float().numpy()
    recon_binary = (recon_labels > 0.5).astype(float)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, labels, title in zip(
        axes,
        [orig_labels, recon_binary],
        ["Original laundering labels", f"Reconstructed (t={t_viz})"],
    ):
        colors = ["red" if l > 0.5 else "steelblue" for l in labels]
        nx.draw_networkx(
            G, pos=pos, ax=ax,
            node_color=colors, node_size=400,
            with_labels=True, font_size=7,
            edge_color="grey", alpha=0.85,
        )
        ax.set_title(title, fontsize=12)
        ax.axis("off")

    legend_elements = [
        mpatches.Patch(facecolor="red",       label="Laundering"),
        mpatches.Patch(facecolor="steelblue", label="Clean"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2, fontsize=11)

    correct = ((recon_binary > 0.5) == (orig_labels > 0.5)).mean()
    plt.suptitle(f"Laundering node detection  |  accuracy: {correct:.2%}", fontsize=13)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)



def run_all_visualizations(model, diffusion, dataset, x_mean, x_std,
                            timesteps, adj_loss_w, deg_loss_w, device,
                            graphs_dir, epoch, laund_loss_w=2.0, feature_names=None):
    """
    Pick one sample from the dataset and run all five visualizations.
    Files are saved as  graphs/<name>_epoch{epoch:04d}.png

    Args:
        model, diffusion:   Trained model and diffusion object.
        dataset:            CachedDataset instance.
        x_mean, x_std:      Normalisation stats (CPU tensors, shape [F]).
        timesteps:          Total diffusion timesteps.
        adj_loss_w:         Adjacency loss weight.
        deg_loss_w:         Degree loss weight.
        device:             Torch device string.
        graphs_dir:         Directory to write images into.
        epoch:              Current epoch number (used in filenames).
        feature_names:      Optional list of feature name strings.
    """
    os.makedirs(graphs_dir, exist_ok=True)

    x_orig, adj_orig = _pick_sample(dataset, min_nodes=6)
    N = x_orig.shape[0]

    x_batch   = x_orig.unsqueeze(0).float().to(device)
    adj_batch = adj_orig.unsqueeze(0).float().to(device)
    node_mask = torch.ones(1, N, device=device)

    x_batch_norm = x_batch.clone()
    x_batch_norm[:, :, 1:] = (x_batch[:, :, 1:] - x_mean.to(device)[1:]) / x_std.to(device)[1:]

    tag = f"epoch{epoch:04d}"

    plot_forward_corruption(
        diffusion, x_batch_norm, node_mask, timesteps, device,
        save_path=os.path.join(graphs_dir, f"forward_corruption_{tag}.png"),
        feature_names=feature_names,
    )

    plot_encode_decode(
        model, diffusion, x_batch_norm, adj_batch, node_mask,
        x_mean, x_std, device,
        save_path=os.path.join(graphs_dir, f"encode_decode_{tag}.png"),
        feature_names=feature_names,
    )

    plot_full_generation(
        model, diffusion, x_batch_norm, adj_batch, node_mask,
        x_mean, x_std, device,
        save_path=os.path.join(graphs_dir, f"full_generation_{tag}.png"),
        feature_names=feature_names,
    )

    plot_loss_curve(
        model, diffusion, x_batch_norm, adj_batch, node_mask,
        timesteps, adj_loss_w, deg_loss_w, device,
        save_path=os.path.join(graphs_dir, f"loss_curve_{tag}.png"),
        laund_loss_w=laund_loss_w,
    )

    plot_graph_viz(
        model, diffusion, x_batch_norm, adj_batch, adj_orig, x_orig,
        node_mask, device,
        save_path=os.path.join(graphs_dir, f"graph_viz_{tag}.png"),
        feature_names=feature_names,
    )

    print(f"  [viz] saved 7 figures to {graphs_dir}/ (epoch {epoch})")
