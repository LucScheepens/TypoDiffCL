"""
test.py — evaluate guided generation and save plots + scores to results/

Run:
    # IBM (default)
    python test.py

    # Elliptic Bitcoin Dataset
    python test.py --dataset elliptic

Outputs (in results_ibm/ or results_elliptic/):
    simclr_latent_space.png                     SimCLR encoder space (illicit vs licit)
    simclr_guided_generation.png                UMAP with generated networks highlighted
    comparison/comparison_gen_NN.png            4-panel comparison per generated network
    generated_scores.csv                        Realism / novelty scores for generated nets
    calibration_scores.csv                      Same scores for N_CALIB sampled training nets
"""

import sys
import argparse
import random as _random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive — must come before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import umap

# ── path setup ────────────────────────────────────────────────────────────────
SIMCLR_DIR = Path(__file__).resolve().parent
DIFF_DIR   = SIMCLR_DIR.parent / "diffusion"

for _p in (str(SIMCLR_DIR.parent), str(DIFF_DIR), str(SIMCLR_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from util import (
    preprocess_df,
    extract_laundering_networks_igraph,
    extract_non_laundering_networks_igraph,
)
from augmentation import build_igraph_from_transactions
from plotting_helpers import plot_simclr_latent_space_laundering_vs_clean
from grad.igraph_version.delete.generation import (
    load_simclr_encoder,
    load_diffusion_model,
    encode_all_networks,
    train_mlp_probe,
    run_guided_generation,
    # Elliptic variants
    load_simclr_encoder_elliptic,
    load_diffusion_model_elliptic,
    encode_all_pyg_graphs,
    run_guided_generation_elliptic,
)
from scoring import (
    fit_training_distribution,
    score_network,
    graph_feature_vector,
    _print_scores_table,
    _save_scores_csv,
)
from latent_seed_generation import plot_generated_vs_closest_training

# ── config ────────────────────────────────────────────────────────────────────
IBM_CSV_PATH   = r"C:\Users\lucsc\Thesis\grad\grad\data\IBM\LI-Small_Trans.csv"
TARGET         = 1       # 1 = generate illicit/laundering-like, 0 = clean-like
N_GEN          = 8
T_START        = 150
GUIDE_SCALE    = 2.0
NOVELTY_WEIGHT = 2.0
GUIDE_EVERY    = 5
GUIDE_FROM     = 0.25
N_CALIB        = 5


# ── shared helpers ────────────────────────────────────────────────────────────

def _plot_umap(H_all_n, all_labels, gen_embeddings, seeds, target_label,
               t_start, novelty_weight, save_path, pos_label="illicit/laundering"):
    n_train    = len(all_labels)
    H_combined = np.concatenate([H_all_n.numpy(), gen_embeddings], axis=0)
    is_gen     = np.array([False] * n_train + [True] * len(seeds))
    labels_all = np.array(list(all_labels) + [target_label] * len(seeds))

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)
    H_2d    = reducer.fit_transform(H_combined)

    fig, ax = plt.subplots(figsize=(9, 9))
    sc = ax.scatter(H_2d[~is_gen, 0], H_2d[~is_gen, 1],
                    c=labels_all[~is_gen], cmap="coolwarm",
                    s=20, alpha=0.35, label="Original")
    ax.scatter(H_2d[is_gen, 0], H_2d[is_gen, 1],
               c="red" if target_label == 1 else "blue",
               s=160, edgecolors="black", linewidths=1.5,
               marker="*", zorder=5,
               label=f"Generated ({pos_label if target_label else 'clean/licit'})")
    cbar = fig.colorbar(sc, ax=ax, ticks=[0, 1])
    cbar.ax.set_yticklabels(["Clean / licit", "Illicit / laundering"])
    ax.set_title(
        f"SimCLR Guided Generation  (t_start={t_start}, nov_w={novelty_weight})\n"
        "★ = diffusion-generated, guided away from training neighbours"
    )
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved → {save_path}")


def _plot_latent_space_pyg(graphs, encoder, device, save_path):
    """
    UMAP of the Elliptic SimCLR encoder space (illicit vs licit).
    Works directly with PyG Data objects extended to 6-D.
    """
    from torch_geometric.data import Data as _Data, Batch as _Batch

    ext, labels = [], []
    for g in graphs:
        n         = g.x.shape[0]
        label_col = torch.full((n, 1), float(g.y.item()))
        x6        = torch.cat([label_col, g.x], dim=1)
        ext.append(_Data(x=x6, edge_index=g.edge_index))
        labels.append(int(g.y.item()))

    H_list = []
    encoder.eval()
    with torch.no_grad():
        for i in range(0, len(ext), 256):
            chunk = _Batch.from_data_list(ext[i : i + 256]).to(device)
            H_list.append(encoder(chunk).cpu())
    H   = torch.cat(H_list, dim=0).numpy()
    H   = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-8)
    lbl = np.array(labels)

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)
    H_2d    = reducer.fit_transform(H)

    fig, ax = plt.subplots(figsize=(8, 8))
    sc = ax.scatter(H_2d[:, 0], H_2d[:, 1], c=lbl, cmap="coolwarm", s=30, alpha=0.85)
    ax.set_title("SimCLR Encoder Latent Space (Elliptic)\n"
                 "Red = Illicit ego-subgraphs | Blue = Licit ego-subgraphs")
    ax.set_xticks([]); ax.set_yticks([])
    cbar = fig.colorbar(sc, ax=ax, ticks=[0, 1])
    cbar.ax.set_yticklabels(["Licit", "Illicit"])
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved → {save_path}")


def _plot_comparison_pyg(gen_outputs, gen_embeddings, H_all_n, graphs, results_dir,
                         seed_graphs=None):
    """
    Four-panel comparison figure for each Elliptic generated network.

    Panel 1 — Generated network (built from diffusion output adj + x_denorm col 0)
    Panel 2 — Seed ego-subgraph (PyG Data object used to initialise generation)
    Panel 3 — Closest training ego-subgraph by cosine sim to generated embedding
    Panel 4 — Grouped bar chart of structural statistics across all three

    One PNG per generated network saved to results_dir/comparison/.

    Differences from the IBM version in latent_seed_generation.py:
    - Seed / closest-training panels built from PyG edge_index (no igraph needed)
    - No "collapsed hub" node type — all training nodes coloured by graph label
      (red = illicit ego anchor, blue = licit ego anchor)
    - "% pred illicit nodes" replaces "% laundering nodes" for the bar chart
      (generated: col 0 of x_denorm; training: graph-level .y label broadcast)
    """
    import networkx as nx
    from matplotlib.patches import Patch

    comp_dir = results_dir / "comparison"
    comp_dir.mkdir(exist_ok=True)

    H_np = H_all_n.numpy()   # [N, 128]

    # ── helper: build NetworkX graph from a PyG Data object ──────────────────
    def _build_nx_pyg(g):
        """Returns (G, node_colors, n_nodes, graph_label)."""
        n  = g.x.shape[0]
        lbl = int(g.y.item())
        G  = nx.Graph()
        G.add_nodes_from(range(n))
        ei = g.edge_index
        if ei.shape[1] > 0:
            for u, v in zip(ei[0].tolist(), ei[1].tolist()):
                if u < v:                           # undirected — add once
                    G.add_edge(u, v)
        node_color = "#e74c3c" if lbl == 1 else "#aed6f1"
        colors     = [node_color] * n
        return G, colors, n, lbl

    # ── helper: structural stats ──────────────────────────────────────────────
    def _stats(G, n):
        ne   = G.number_of_edges()
        dens = ne / max(n * (n - 1) / 2, 1)
        degs = [d for _, d in G.degree()]
        return {
            "edges":    ne,
            "density":  dens,
            "mean_deg": float(np.mean(degs)) if degs else 0.0,
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

    for i, ((x_denorm, adj_out, n_out), gen_emb) in enumerate(
            zip(gen_outputs, gen_embeddings)):

        # ── Panel 1: Generated network ────────────────────────────────────
        adj_np     = adj_out.cpu().numpy()
        laund_prob = x_denorm[:, 0].cpu().numpy()   # col 0 = predicted illicit prob

        G_gen = nx.Graph()
        G_gen.add_nodes_from(range(n_out))
        for a in range(n_out):
            for b in range(a + 1, n_out):
                if adj_np[a, b] > 0.5:
                    G_gen.add_edge(a, b)
        gen_colors    = ["#e74c3c" if laund_prob[k] > 0.5 else "#aed6f1"
                         for k in range(n_out)]
        pct_illicit   = 100.0 * float((laund_prob > 0.5).mean())
        s_gen         = _stats(G_gen, n_out)
        pos_gen       = nx.spring_layout(G_gen, seed=42, k=2.0 / max(n_out ** 0.5, 1))
        gen_stats_str = (
            f"nodes={n_out}  edges={s_gen['edges']}\n"
            f"density={s_gen['density']:.3f}  mean deg={s_gen['mean_deg']:.1f}\n"
            f"pred. illicit nodes: {pct_illicit:.0f}%"
        )

        # ── Panel 2: Seed ego-subgraph ────────────────────────────────────
        if seed_graphs is not None and i < len(seed_graphs):
            seed_g = seed_graphs[i]
        else:
            cos_sims = H_np @ gen_emb
            seed_g   = graphs[int(cos_sims.argmax())]

        G_seed, seed_colors, n_seed, lbl_seed = _build_nx_pyg(seed_g)
        s_seed        = _stats(G_seed, n_seed)
        pos_seed      = nx.spring_layout(G_seed, seed=42, k=2.0 / max(n_seed ** 0.5, 1))
        seed_stats_str = (
            f"nodes={n_seed}  edges={s_seed['edges']}\n"
            f"density={s_seed['density']:.3f}  mean deg={s_seed['mean_deg']:.1f}\n"
            f"graph label: {'illicit' if lbl_seed == 1 else 'licit'}"
        )

        # ── Panel 3: Closest training graph by cosine sim ─────────────────
        cos_sims    = H_np @ gen_emb
        closest_idx = int(cos_sims.argmax())
        cos_sim_val = float(cos_sims[closest_idx])
        close_g     = graphs[closest_idx]

        G_close, close_colors, n_close, lbl_close = _build_nx_pyg(close_g)
        s_close        = _stats(G_close, n_close)
        pos_close      = nx.spring_layout(G_close, seed=42,
                                          k=2.0 / max(n_close ** 0.5, 1))
        close_stats_str = (
            f"nodes={n_close}  edges={s_close['edges']}\n"
            f"density={s_close['density']:.3f}  mean deg={s_close['mean_deg']:.1f}\n"
            f"graph label: {'illicit' if lbl_close == 1 else 'licit'}\n"
            f"cosine sim = {cos_sim_val:.4f}"
        )

        # ── Panel 4: Grouped bar chart ────────────────────────────────────
        bar_labels = ["Density ×100", "Mean degree", "% Illicit nodes (pred/label)"]
        gen_vals   = [s_gen["density"]   * 100, s_gen["mean_deg"],   pct_illicit]
        seed_vals  = [s_seed["density"]  * 100, s_seed["mean_deg"],  100.0 * lbl_seed]
        close_vals = [s_close["density"] * 100, s_close["mean_deg"], 100.0 * lbl_close]

        # ── Figure ────────────────────────────────────────────────────────
        fig, axes = plt.subplots(
            1, 4, figsize=(32, 8),
            gridspec_kw={"width_ratios": [2, 2, 2, 1.5]},
        )
        ax_gen, ax_seed, ax_close, ax_bar = axes

        fig.suptitle(
            f"Generated #{i+1}  |  seed → generated → closest training  (Elliptic)",
            fontsize=13, fontweight="bold",
        )

        _draw_panel(ax_gen,   G_gen,   pos_gen,   gen_colors,   "Generated Network",   gen_stats_str)
        _draw_panel(ax_seed,  G_seed,  pos_seed,  seed_colors,  "Seed Ego-Subgraph",   seed_stats_str)
        _draw_panel(ax_close, G_close, pos_close, close_colors, "Closest Training",    close_stats_str)

        x      = np.arange(len(bar_labels))
        width  = 0.25
        ax_bar.bar(x - width, gen_vals,   width, label="Generated",        color="#e74c3c", alpha=0.85)
        ax_bar.bar(x,         seed_vals,  width, label="Seed",             color="#f39c12", alpha=0.85)
        ax_bar.bar(x + width, close_vals, width, label="Closest training", color="#3498db", alpha=0.85)
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(bar_labels, fontsize=9)
        ax_bar.set_ylabel("Value", fontsize=10)
        ax_bar.set_title("Structural Statistics", fontsize=11, fontweight="bold")
        ax_bar.legend(fontsize=8, loc="upper right")
        ax_bar.spines["top"].set_visible(False)
        ax_bar.spines["right"].set_visible(False)
        for bars in ax_bar.containers:
            ax_bar.bar_label(bars, fmt="%.1f", fontsize=7, padding=2)

        legend_els = [
            Patch(facecolor="#e74c3c", label="Illicit ego / pred. illicit node"),
            Patch(facecolor="#aed6f1", label="Licit ego / pred. licit node"),
        ]
        fig.legend(handles=legend_els, loc="lower center",
                   ncol=2, fontsize=9, frameon=True,
                   bbox_to_anchor=(0.40, 0.0))

        fig.tight_layout(rect=[0, 0.06, 1, 0.94])
        out = comp_dir / f"comparison_gen_{i+1:02d}.png"
        fig.savefig(out, dpi=200)
        plt.close(fig)
        print(f"  Saved → {out}")


def _fit_training_distribution_pyg(graphs):
    """
    Like fit_training_distribution() but accepts Elliptic PyG Data objects.
    Converts each graph to dense (x6, adj) on the fly for graph_feature_vector().
    """
    from scipy.spatial.distance import mahalanobis
    FEAT_DIM = 10

    print("Computing Elliptic training graph-feature distribution …")
    feats = []
    for g in graphs:
        n   = g.x.shape[0]
        adj = torch.zeros(n, n)
        ei  = g.edge_index
        if ei.shape[1] > 0:
            valid = ei[0] != ei[1]
            src, dst = ei[0][valid], ei[1][valid]
            inbounds = (src < n) & (dst < n)
            adj[src[inbounds], dst[inbounds]] = 1.0

        # Build 6-D x matching IBM convention: col 0 = label, cols 1-5 = structural
        label_col = torch.full((n, 1), float(g.y.item()))
        x6        = torch.cat([label_col, g.x], dim=1)

        feats.append(graph_feature_vector(x6, adj))

    F_train = np.stack(feats, axis=0)
    mu      = F_train.mean(axis=0)
    cov     = np.cov(F_train.T) + np.eye(FEAT_DIM) * 1e-4
    cov_inv = np.linalg.inv(cov)

    train_mah     = np.array([mahalanobis(fv, mu, cov_inv) for fv in F_train])
    mah_p95       = np.percentile(train_mah, 95)
    realism_scale = mah_p95 * 2.0
    print(f"  Mahalanobis  mean={train_mah.mean():.2f}  "
          f"p95={mah_p95:.2f}  scale={realism_scale:.2f}")
    return mu, cov_inv, realism_scale


# ─────────────────────────────────────────────────────────────────────────────
# IBM path
# ─────────────────────────────────────────────────────────────────────────────

def run_ibm(device, results_dir, ibm_csv):
    import pickle

    csv_stem   = Path(ibm_csv).stem
    cache_path = SIMCLR_DIR / f"networks_cache_{csv_stem}.pkl"
    df_full    = preprocess_df(ibm_csv)

    # -- 1. Load / extract networks ------------------------------------------
    if cache_path.exists():
        print(f"Loading networks from cache: {cache_path} …")
        with open(cache_path, "rb") as f:
            networks = pickle.load(f)
        for net in networks:
            net["graph"] = build_igraph_from_transactions(net["transactions"])
        n_laund = sum(1 for n in networks if len(n["laundering_nodes"]) > 0)
        print(f"Loaded {len(networks)} networks "
              f"({n_laund} laundering, {len(networks) - n_laund} clean) from cache\n")
    else:
        print("Extracting networks from CSV (this may take a while) …")
        with_laund = extract_laundering_networks_igraph(
            df_full, max_depth=4, max_networks=2000,
            collapse_threshold=10, max_nodes=300,
        )
        non_laund = extract_non_laundering_networks_igraph(
            df_full, max_depth=4, max_networks=len(with_laund),
            collapse_threshold=10, max_nodes=300,
        )
        networks = with_laund + non_laund
        for net in networks:
            net["graph"] = build_igraph_from_transactions(net["transactions"])
        networks_to_cache = [{k: v for k, v in net.items() if k != "graph"}
                             for net in networks]
        with open(cache_path, "wb") as f:
            pickle.dump(networks_to_cache, f)
        print(f"Saved network cache → {cache_path}")
        print(f"Loaded {len(networks)} networks "
              f"({len(with_laund)} laundering, {len(non_laund)} clean)\n")

    # -- 2. SimCLR latent space plot -----------------------------------------
    print("Plotting SimCLR latent space …")
    fig = plot_simclr_latent_space_laundering_vs_clean(networks, df_full)
    out = results_dir / "simclr_latent_space.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved → {out}\n")
    
    # -- 3. Load models -------------------------------------------------------
    print("Loading models …")
    encoder = load_simclr_encoder(device)
    diff_model, diffusion, x_mean, x_std = load_diffusion_model(device)

    # -- 4. Encode + train probe ---------------------------------------------
    print("\nEncoding training networks …")
    H_all_n, y_all = encode_all_networks(networks, encoder, device)
    probe = train_mlp_probe(H_all_n, y_all, device)

    # -- 5. Guided generation -------------------------------------------------
    print("\nRunning guided generation …")
    gen_outputs, gen_embeddings, seeds = run_guided_generation(
        networks, encoder, probe, diff_model, diffusion,
        x_mean, x_std, H_all_n, device,
        target_label=TARGET,
        n_gen=N_GEN,
        t_start=T_START,
        guidance_scale=GUIDE_SCALE,
        novelty_weight=NOVELTY_WEIGHT,
        guide_every=GUIDE_EVERY,
        guide_from=GUIDE_FROM,
    )

    # -- 6. UMAP plot ---------------------------------------------------------
    print("\nPlotting UMAP …")
    _plot_umap(
        H_all_n, y_all.tolist(), gen_embeddings, seeds,
        TARGET, T_START, NOVELTY_WEIGHT,
        results_dir / "simclr_guided_generation.png",
        pos_label="laundering",
    )

    # -- 7. Per-network comparison plots -------------------------------------
    print("\nPlotting generated vs closest training comparisons …")
    plot_generated_vs_closest_training(
        gen_outputs, gen_embeddings, H_all_n, networks, results_dir,
        seed_networks=seeds,
    )

    # -- 8. Fit training distribution ----------------------------------------
    print()
    mu, cov_inv, realism_scale = fit_training_distribution(networks)

    # -- 9. Score generated networks -----------------------------------------
    gen_scores = []
    for i, (x_d, adj_d, _) in enumerate(gen_outputs):
        s = score_network(x_d, adj_d, encoder, H_all_n, device,
                          mu, cov_inv, realism_scale,
                          gen_embedding=torch.tensor(gen_embeddings[i]))
        gen_scores.append(s)
    _print_scores_table(gen_scores, "Generated networks")
    _save_scores_csv(gen_scores, results_dir / "generated_scores.csv", "generated")

    # -- 10. Calibration: score real training networks -----------------------
    from diffusion.diff_util import network_to_dense as _ntd
    calib_nets   = _random.sample(networks, min(N_CALIB, len(networks)))
    calib_scores = []
    for net in calib_nets:
        xr, adjr = (net["x_dense"], net["adj_dense"]) \
                   if ("x_dense" in net and "adj_dense" in net) else _ntd(net)
        s = score_network(xr, adjr, encoder, H_all_n, device,
                          mu, cov_inv, realism_scale)
        calib_scores.append(s)
    _print_scores_table(calib_scores, f"Calibration: {N_CALIB} real training networks")
    _save_scores_csv(calib_scores, results_dir / "calibration_scores.csv", "training")


# ─────────────────────────────────────────────────────────────────────────────
# Elliptic path
# ─────────────────────────────────────────────────────────────────────────────

def run_elliptic(device, results_dir):
    from grad.igraph_version.archive.elliptic_adapter import load_elliptic_pyg_graphs

    # -- 1. Load ego subgraphs -----------------------------------------------
    print("Loading Elliptic ego subgraphs …")
    graphs  = load_elliptic_pyg_graphs(max_nodes=100)
    n_ill   = sum(g.y.item() == 1 for g in graphs)
    print(f"Loaded {len(graphs)} graphs ({n_ill} illicit, {len(graphs)-n_ill} licit)\n")

    # -- 2. SimCLR latent space plot -----------------------------------------
    print("Loading Elliptic SimCLR encoder …")
    encoder = load_simclr_encoder_elliptic(device)
    print("Plotting SimCLR latent space …")
    _plot_latent_space_pyg(graphs, encoder, device,
                           results_dir / "simclr_latent_space.png")
    print()

    # -- 3. Load diffusion model ----------------------------------------------
    print("Loading Elliptic diffusion model …")
    diff_model, diffusion, x_mean, x_std = load_diffusion_model_elliptic(device)

    # -- 4. Encode + train probe ---------------------------------------------
    print("\nEncoding graphs …")
    H_all_n, y_all = encode_all_pyg_graphs(graphs, encoder, device)
    probe = train_mlp_probe(H_all_n, y_all, device)

    # -- 5. Guided generation ------------------------------------------------
    print("\nRunning guided generation …")
    gen_outputs, gen_embeddings, seeds = run_guided_generation_elliptic(
        graphs, encoder, probe, diff_model, diffusion,
        x_mean, x_std, H_all_n, device,
        target_label=TARGET,
        n_gen=N_GEN,
        t_start=T_START,
        guidance_scale=GUIDE_SCALE,
        novelty_weight=NOVELTY_WEIGHT,
        guide_every=GUIDE_EVERY,
        guide_from=GUIDE_FROM,
    )

    # -- 6. UMAP plot --------------------------------------------------------
    print("\nPlotting UMAP …")
    _plot_umap(
        H_all_n, y_all.tolist(), gen_embeddings, seeds,
        TARGET, T_START, NOVELTY_WEIGHT,
        results_dir / "simclr_guided_generation.png",
        pos_label="illicit",
    )

    # -- 7. Per-network comparison plots -------------------------------------
    print("\nPlotting generated vs closest training comparisons …")
    _plot_comparison_pyg(gen_outputs, gen_embeddings, H_all_n, graphs,
                         results_dir, seed_graphs=seeds)

    # -- 8. Fit training distribution ----------------------------------------
    print()
    mu, cov_inv, realism_scale = _fit_training_distribution_pyg(graphs)

    # -- 8. Score generated networks -----------------------------------------
    gen_scores = []
    for i, (x_d, adj_d, _) in enumerate(gen_outputs):
        s = score_network(x_d, adj_d, encoder, H_all_n, device,
                          mu, cov_inv, realism_scale,
                          gen_embedding=torch.tensor(gen_embeddings[i]))
        gen_scores.append(s)
    _print_scores_table(gen_scores, "Generated Elliptic networks")
    _save_scores_csv(gen_scores, results_dir / "generated_scores.csv", "generated")

    # -- 9. Calibration: score real training graphs --------------------------
    calib_graphs = _random.sample(graphs, min(N_CALIB, len(graphs)))
    calib_scores = []
    for g in calib_graphs:
        n   = g.x.shape[0]
        adj = torch.zeros(n, n)
        ei  = g.edge_index
        if ei.shape[1] > 0:
            valid = ei[0] != ei[1]
            src, dst = ei[0][valid], ei[1][valid]
            inbounds = (src < n) & (dst < n)
            adj[src[inbounds], dst[inbounds]] = 1.0
        label_col = torch.full((n, 1), float(g.y.item()))
        x6        = torch.cat([label_col, g.x], dim=1)
        s = score_network(x6, adj, encoder, H_all_n, device,
                          mu, cov_inv, realism_scale)
        calib_scores.append(s)
    _print_scores_table(calib_scores, f"Calibration: {N_CALIB} real Elliptic graphs")
    _save_scores_csv(calib_scores, results_dir / "calibration_scores.csv", "training")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["ibm", "elliptic"], default="ibm",
                        help="Dataset to use: ibm (default) or elliptic")
    parser.add_argument("--ibm-csv", type=str, default=None, metavar="PATH",
                        help="Override the IBM CSV path (default: LI-Small_Trans.csv)")
    args = parser.parse_args()

    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    suffix      = args.dataset
    results_dir = SIMCLR_DIR / f"results_{suffix}"
    results_dir.mkdir(exist_ok=True)
    print(f"Device: {device}")
    print(f"Results will be saved to {results_dir}\n")

    if args.dataset == "ibm":
        ibm_csv = args.ibm_csv if args.ibm_csv else IBM_CSV_PATH
        run_ibm(device, results_dir, ibm_csv)
    else:
        run_elliptic(device, results_dir)

    print(f"\nAll results saved to {results_dir}")


if __name__ == "__main__":
    main()
