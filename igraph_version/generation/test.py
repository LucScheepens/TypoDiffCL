"""
test.py — evaluate guided generation and save plots + scores to results/

Run:
    # IBM — baseline generation
    python test.py

    # IBM — with embedding separation check
    python test.py --sep-check

    # IBM — tune guidance params before generating
    python test.py --tune-guidance --tune-trials 15

    # Elliptic
    python test.py --dataset elliptic

    # Override guidance hyperparams directly
    python test.py --guidance-scale 3.0 --novelty-weight 1.5 --degree-penalty 0.3

Outputs (in results/<dataset>/):
    simclr_latent_space.png                 SimCLR encoder space (illicit vs licit)
    simclr_guided_generation.png            UMAP with generated networks highlighted
    comparison/comparison_gen_NN.png        4-panel comparison per generated network
    generated_scores.csv                    Realism / novelty scores for generated nets
    calibration_scores.csv                  Same scores for N_CALIB sampled training nets
    graph_quality_<dataset>.csv             Tier-1 Q-score report (emb dist, Wass, density, KL)
    quality_extremes_<dataset>.png          Top / bottom generated graphs by Q score
    tuning_trials.csv                       Hyperparameter search history (if --tune-guidance)
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

# ── path setup ────────────────────────────────────────────────────────────────
_GEN_DIR   = Path(__file__).resolve().parent   # igraph_version/generation/
ROOT_DIR   = _GEN_DIR.parent                   # igraph_version/
DIFF_DIR   = ROOT_DIR / "diffusion"
SIMCLR_DIR = ROOT_DIR / "simclr"
CKPT_DIR   = ROOT_DIR / "checkpoints"
DATA_DIR   = ROOT_DIR / "data"

for _p in (str(ROOT_DIR), str(DIFF_DIR), str(SIMCLR_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from util import preprocess_df, extract_networks_igraph
from augmentation import build_igraph_from_transactions
from plotting_helpers import plot_simclr_latent_space_laundering_vs_clean
from generation.generation import (
    load_simclr_encoder,
    load_diffusion_model,
    encode_all_networks,
    train_mlp_probe,
    run_guided_generation,
    tune_guidance_params,
    # Elliptic variants
    load_simclr_encoder_elliptic,
    load_diffusion_model_elliptic,
    encode_all_pyg_graphs,
    run_guided_generation_elliptic,
)
from generation.graph_quality_metrics import (
    score_generated_graphs,
    print_quality_report,
    save_quality_csv,
    plot_quality_extremes,
    compute_embedding_separation,
)
from scoring import (
    fit_training_distribution,
    score_network,
    graph_feature_vector,
    _print_scores_table,
    _save_scores_csv,
)
from generation.latent_seed_generation import plot_generated_vs_closest_training

# ── defaults (overridden by CLI args) ─────────────────────────────────────────
IBM_CSV_PATH   = r"C:\Users\lucsc\Thesis\grad\grad\data\IBM\LI-Small_Trans.csv"
TARGET         = 1       # 1 = generate illicit/laundering-like, 0 = clean-like
N_GEN          = 8
T_START        = 150
GUIDE_SCALE    = 2.0
NOVELTY_WEIGHT = 2.0
DEGREE_PENALTY = 0.5
GUIDE_EVERY    = 5
GUIDE_FROM     = 0.25
N_CALIB        = 5


# ── PyG conversion helper (shared by IBM and Elliptic quality scoring) ────────

def _gen_to_pyg(x_denorm, adj_out, n_out, label, adj_threshold=0.5):
    """Convert a diffusion output triple to a minimal PyG Data object."""
    from torch_geometric.data import Data
    adj_np = adj_out[:n_out, :n_out]
    if isinstance(adj_np, torch.Tensor):
        adj_np = adj_np.numpy()
    np.fill_diagonal(adj_np, 0.0)   # no self-loops
    ei = (torch.tensor(adj_np) > adj_threshold).nonzero(as_tuple=False).T.contiguous()
    if ei.shape[1] == 0:
        ei = torch.zeros(2, 0, dtype=torch.long)
    return Data(
        x=x_denorm,
        edge_index=ei,
        y=torch.tensor([label], dtype=torch.long),
    )


# ── UMAP plot (shared) ────────────────────────────────────────────────────────

def _plot_umap(H_all_n, all_labels, gen_embeddings, seeds, target_label,
               t_start, novelty_weight, save_path, pos_label="illicit/laundering"):
    try:
        import umap
    except ImportError:
        print("  [skip] umap-learn not installed — skipping UMAP plot")
        return

    n_train    = len(all_labels)
    n_gen      = len(gen_embeddings)
    H_combined = np.concatenate([H_all_n.numpy(), gen_embeddings], axis=0)
    is_gen     = np.array([False] * n_train + [True] * n_gen)
    labels_all = np.array(list(all_labels) + [target_label] * n_gen)

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


# ── Elliptic-specific helpers ─────────────────────────────────────────────────

def _plot_latent_space_pyg(graphs, encoder, device, save_path):
    try:
        import umap
    except ImportError:
        print("  [skip] umap-learn not installed — skipping latent space plot")
        return

    from torch_geometric.data import Data as _Data, Batch as _Batch

    ext, labels = [], []
    for g in graphs:
        ext.append(_Data(x=g.x.clone(), edge_index=g.edge_index))
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
    import networkx as nx
    from matplotlib.patches import Patch

    comp_dir = results_dir / "comparison"
    comp_dir.mkdir(exist_ok=True)

    H_np = H_all_n.numpy()

    def _build_nx_pyg(g):
        n   = g.x.shape[0]
        lbl = int(g.y.item())
        G   = nx.Graph()
        G.add_nodes_from(range(n))
        ei  = g.edge_index
        if ei.shape[1] > 0:
            for u, v in zip(ei[0].tolist(), ei[1].tolist()):
                if u < v:
                    G.add_edge(u, v)
        colors = ["#e74c3c" if lbl == 1 else "#aed6f1"] * n
        return G, colors, n, lbl

    def _stats(G, n):
        ne   = G.number_of_edges()
        dens = ne / max(n * (n - 1) / 2, 1)
        degs = [d for _, d in G.degree()]
        return {"edges": ne, "density": dens, "mean_deg": float(np.mean(degs)) if degs else 0.0}

    def _draw_panel(ax, G, pos, colors, title, stats_str):
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors, node_size=80, alpha=0.90)
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#555555", width=0.8,
                               alpha=0.45, arrows=False)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.axis("off")
        ax.text(0.02, 0.02, stats_str, transform=ax.transAxes, fontsize=8.5,
                verticalalignment="bottom",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.85))

    for i, ((x_denorm, adj_out, n_out, *_), gen_emb) in enumerate(
            zip(gen_outputs, gen_embeddings)):

        adj_np     = adj_out.cpu().numpy() if isinstance(adj_out, torch.Tensor) else adj_out
        laund_prob = x_denorm[:, 0].cpu().numpy() if isinstance(x_denorm, torch.Tensor) \
                     else x_denorm[:, 0]

        # Symmetrise: treat either directed edge as an undirected connection so
        # nodes with only lower-triangle edges are not drawn as isolated.
        adj_sym = np.maximum(adj_np, adj_np.T)
        G_gen = nx.Graph()
        G_gen.add_nodes_from(range(n_out))
        for a in range(n_out):
            for b in range(a + 1, n_out):
                if adj_sym[a, b] > 0.5:
                    G_gen.add_edge(a, b)
        gen_colors  = ["#e74c3c" if laund_prob[k] > 0.5 else "#aed6f1" for k in range(n_out)]
        pct_illicit = 100.0 * float((laund_prob > 0.5).mean())
        s_gen       = _stats(G_gen, n_out)
        pos_gen     = nx.spring_layout(G_gen, seed=42, k=2.0 / max(n_out ** 0.5, 1))
        gen_stats   = (f"nodes={n_out}  edges={s_gen['edges']}\n"
                       f"density={s_gen['density']:.3f}  mean deg={s_gen['mean_deg']:.1f}\n"
                       f"pred. illicit nodes: {pct_illicit:.0f}%")

        seed_g = seed_graphs[i] if (seed_graphs is not None and i < len(seed_graphs)) \
                 else graphs[int((H_np @ gen_emb).argmax())]
        G_seed, seed_colors, n_seed, lbl_seed = _build_nx_pyg(seed_g)
        s_seed    = _stats(G_seed, n_seed)
        pos_seed  = nx.spring_layout(G_seed, seed=42, k=2.0 / max(n_seed ** 0.5, 1))
        seed_stats = (f"nodes={n_seed}  edges={s_seed['edges']}\n"
                      f"density={s_seed['density']:.3f}  mean deg={s_seed['mean_deg']:.1f}\n"
                      f"graph label: {'illicit' if lbl_seed == 1 else 'licit'}")

        closest_idx  = int((H_np @ gen_emb).argmax())
        cos_sim_val  = float((H_np @ gen_emb)[closest_idx])
        close_g      = graphs[closest_idx]
        G_close, close_colors, n_close, lbl_close = _build_nx_pyg(close_g)
        s_close   = _stats(G_close, n_close)
        pos_close = nx.spring_layout(G_close, seed=42, k=2.0 / max(n_close ** 0.5, 1))
        close_stats = (f"nodes={n_close}  edges={s_close['edges']}\n"
                       f"density={s_close['density']:.3f}  mean deg={s_close['mean_deg']:.1f}\n"
                       f"graph label: {'illicit' if lbl_close == 1 else 'licit'}\n"
                       f"cosine sim = {cos_sim_val:.4f}")

        fig, axes = plt.subplots(1, 4, figsize=(32, 8),
                                 gridspec_kw={"width_ratios": [2, 2, 2, 1.5]})
        ax_gen, ax_seed, ax_close, ax_bar = axes
        fig.suptitle(f"Generated #{i+1}  |  seed → generated → closest training  (Elliptic)",
                     fontsize=13, fontweight="bold")

        _draw_panel(ax_gen,   G_gen,   pos_gen,   gen_colors,   "Generated Network",  gen_stats)
        _draw_panel(ax_seed,  G_seed,  pos_seed,  seed_colors,  "Seed Ego-Subgraph",  seed_stats)
        _draw_panel(ax_close, G_close, pos_close, close_colors, "Closest Training",   close_stats)

        bar_labels = ["Density ×100", "Mean degree", "% Illicit nodes"]
        gen_vals   = [s_gen["density"]   * 100, s_gen["mean_deg"],   pct_illicit]
        seed_vals  = [s_seed["density"]  * 100, s_seed["mean_deg"],  100.0 * lbl_seed]
        close_vals = [s_close["density"] * 100, s_close["mean_deg"], 100.0 * lbl_close]
        x = np.arange(len(bar_labels)); width = 0.25
        ax_bar.bar(x - width, gen_vals,   width, label="Generated",        color="#e74c3c", alpha=0.85)
        ax_bar.bar(x,         seed_vals,  width, label="Seed",             color="#f39c12", alpha=0.85)
        ax_bar.bar(x + width, close_vals, width, label="Closest training", color="#3498db", alpha=0.85)
        ax_bar.set_xticks(x); ax_bar.set_xticklabels(bar_labels, fontsize=9)
        ax_bar.set_ylabel("Value"); ax_bar.set_title("Structural Statistics", fontsize=11, fontweight="bold")
        ax_bar.legend(fontsize=8, loc="upper right")
        ax_bar.spines["top"].set_visible(False); ax_bar.spines["right"].set_visible(False)
        for bars in ax_bar.containers:
            ax_bar.bar_label(bars, fmt="%.1f", fontsize=7, padding=2)

        from matplotlib.patches import Patch
        fig.legend(handles=[Patch(facecolor="#e74c3c", label="Illicit / pred. illicit"),
                             Patch(facecolor="#aed6f1", label="Licit / pred. licit")],
                   loc="lower center", ncol=2, fontsize=9,
                   bbox_to_anchor=(0.40, 0.0))
        fig.tight_layout(rect=[0, 0.06, 1, 0.94])
        out = comp_dir / f"comparison_gen_{i+1:02d}.png"
        fig.savefig(out, dpi=200)
        plt.close(fig)
        print(f"  Saved → {out}")


def _fit_training_distribution_pyg(graphs):
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
        label_col = torch.full((n, 1), float(g.y.item()))
        x6        = torch.cat([label_col, g.x], dim=1)
        feats.append(graph_feature_vector(x6, adj))

    F_train       = np.stack(feats, axis=0)
    mu            = F_train.mean(axis=0)
    cov           = np.cov(F_train.T) + np.eye(FEAT_DIM) * 1e-4
    cov_inv       = np.linalg.inv(cov)
    train_mah     = np.array([mahalanobis(fv, mu, cov_inv) for fv in F_train])
    mah_p95       = np.percentile(train_mah, 95)
    realism_scale = mah_p95 * 2.0
    print(f"  Mahalanobis  mean={train_mah.mean():.2f}  p95={mah_p95:.2f}")
    return mu, cov_inv, realism_scale


# ── Augmented file helpers ────────────────────────────────────────────────────

def _gen_network_to_transactions(x_denorm, adj_out, n_out, target_label, net_idx,
                                 adj_threshold=0.5):
    """Build synthetic CSV-format transaction rows for one generated network.

    Each directed edge in adj_out becomes one transaction row.
    Node IDs are namespaced by net_idx to avoid collisions with the originals.
    """
    import random as _rand
    from datetime import datetime, timedelta

    _rand.seed(net_idx * 137 + 42)

    CURRENCIES = ["US Dollar", "Euro", "Australian Dollar", "Bitcoin", "Swiss Franc"]
    FORMATS    = ["ACH", "Wire", "Cheque", "Credit Card", "Bitcoin"]
    WINDOW_START = datetime(2022, 9, 1)
    WINDOW_END   = datetime(2022, 9, 6, 11, 31)
    window_mins  = int((WINDOW_END - WINDOW_START).total_seconds() // 60)

    deg_norm = x_denorm[:, 0].numpy()
    max_deg  = float(deg_norm.max()) if float(deg_norm.max()) > 0 else 1.0

    banks    = [f"{net_idx * 100 + i:06d}" for i in range(n_out)]
    accounts = [f"F{net_idx * 10000 + i:09X}0" for i in range(n_out)]

    edge_list = (adj_out > adj_threshold).nonzero(as_tuple=False).tolist()
    edge_list = [(s, d) for s, d in edge_list if s != d]

    rows = []
    for edge_count, (src, dst) in enumerate(edge_list):
        ts = WINDOW_START + timedelta(minutes=_rand.randint(0, window_mins))
        amount   = round(_rand.uniform(1000, 50000) * (0.5 + deg_norm[src] / max_deg), 2)
        currency = _rand.choice(CURRENCIES)
        fmt      = _rand.choice(FORMATS)
        rows.append({
            "Timestamp":          ts.strftime("%Y/%m/%d %H:%M"),
            "From Bank":          banks[src],
            "Account":            accounts[src],
            "To Bank":            banks[dst],
            "To Account":         accounts[dst],
            "Amount Received":    f"{amount:.2f}",
            "Receiving Currency": currency,
            "Amount Paid":        f"{amount:.2f}",
            "Payment Currency":   currency,
            "Payment Format":     fmt,
            "Is Laundering":      target_label,
        })
    return rows


def _save_augmented_files(gen_outputs, ibm_csv):
    """Copy LI-Small_Trans.csv and LI-Small_Patterns.txt, then append
    synthetic transactions and pattern blocks from gen_outputs.

    Output files:
        <data_dir>/LI-Small_Trans_augmented.csv
        <data_dir>/LI-Small_Patterns_augmented.txt
    """
    import shutil
    import csv as _csv

    csv_path      = Path(ibm_csv)
    patterns_path = csv_path.parent / (
        csv_path.stem.replace("Trans", "Patterns") + ".txt"
    )
    csv_aug = csv_path.parent / (csv_path.stem + "_augmented.csv")
    pat_aug = patterns_path.parent / (patterns_path.stem + "_augmented.txt")

    shutil.copy2(csv_path,      csv_aug)
    shutil.copy2(patterns_path, pat_aug)
    print(f"  Copied originals → {csv_aug.name}, {pat_aug.name}")

    with open(csv_aug, "a", newline="", encoding="utf-8") as csv_f, \
         open(pat_aug,  "a",             encoding="utf-8") as pat_f:

        writer = _csv.writer(csv_f)

        for net_idx, (x_denorm, adj_out, n_out, target_label) in enumerate(gen_outputs):
            rows = _gen_network_to_transactions(
                x_denorm, adj_out, n_out, target_label, net_idx)
            if not rows:
                continue

            for r in rows:
                writer.writerow([
                    r["Timestamp"], r["From Bank"], r["Account"],
                    r["To Bank"],   r["To Account"],
                    r["Amount Received"],   r["Receiving Currency"],
                    r["Amount Paid"],       r["Payment Currency"],
                    r["Payment Format"],    r["Is Laundering"],
                ])

            if target_label == 1:
                pat_f.write(
                    f"\nBEGIN LAUNDERING ATTEMPT - AUGMENTED: "
                    f"Generated network {net_idx + 1}\n"
                )
                for r in rows:
                    pat_f.write(
                        f"{r['Timestamp']},{r['From Bank']},{r['Account']},"
                        f"{r['To Bank']},{r['To Account']},"
                        f"{r['Amount Received']},{r['Receiving Currency']},"
                        f"{r['Amount Paid']},{r['Payment Currency']},"
                        f"{r['Payment Format']},{r['Is Laundering']}\n"
                    )
                pat_f.write("END LAUNDERING ATTEMPT - AUGMENTED\n")

    print(f"  Saved → {csv_aug}")
    print(f"  Saved → {pat_aug}")


# ─────────────────────────────────────────────────────────────────────────────
# IBM path
# ─────────────────────────────────────────────────────────────────────────────

def run_ibm(args, device, results_dir):
    import pickle

    ibm_csv  = args.ibm_csv if args.ibm_csv else IBM_CSV_PATH
    csv_stem = Path(ibm_csv).stem

    # -- 1. Load / extract networks ------------------------------------------
    cache_path = DATA_DIR / f"networks_cache_{csv_stem}.pkl"
    df_full    = preprocess_df(ibm_csv)

    if cache_path.exists():
        print(f"Loading networks from cache: {cache_path} …")
        with open(cache_path, "rb") as f:
            networks = pickle.load(f)
        for net in networks:
            net["graph"] = build_igraph_from_transactions(net["transactions"])
    else:
        print("Extracting networks from CSV …")
        networks = extract_networks_igraph(
            df_full, max_depth=4, max_networks=4000,
            collapse_threshold=10, max_nodes=64,
        )
        for net in networks:
            net["graph"] = build_igraph_from_transactions(net["transactions"])
        networks_to_cache = [{k: v for k, v in net.items() if k != "graph"}
                             for net in networks]
        with open(cache_path, "wb") as f:
            pickle.dump(networks_to_cache, f)
        print(f"Saved network cache → {cache_path}")

    n_laund = sum(1 for n in networks if len(n["laundering_nodes"]) > 0)
    print(f"Loaded {len(networks)} networks ({n_laund} laundering, "
          f"{len(networks) - n_laund} clean)\n")

    # -- 2. Load models -------------------------------------------------------
    _simclr_ckpt_dir = (Path(args.ckpt_dir) / "simclr") if args.ckpt_dir else None
    _diff_ckpt_path  = (Path(args.ckpt_dir) / "diffusion" / "model.pt") if args.ckpt_dir else None

    # Resolve the SimCLR checkpoint file that will be selected (best loss)
    _sc_search_dir = _simclr_ckpt_dir if _simclr_ckpt_dir else CKPT_DIR / "simclr_ibm"
    _sc_candidates = sorted(Path(_sc_search_dir).glob("*.pt")) if Path(_sc_search_dir).exists() else []
    _sc_best = None; _sc_best_loss = float("inf")
    for _p in _sc_candidates:
        try:
            _c = torch.load(_p, map_location="cpu", weights_only=False)
            if isinstance(_c, dict) and "loss" in _c and _c["loss"] < _sc_best_loss:
                _sc_best_loss = _c["loss"]; _sc_best = _p
        except Exception:
            pass

    # Resolve the diffusion checkpoint path
    _df_path = Path(_diff_ckpt_path) if _diff_ckpt_path else CKPT_DIR / "diffusion_ibm" / "model.pt"

    print("=" * 62)
    print("  MODEL CHECKPOINT SUMMARY")
    print("=" * 62)
    print(f"  SimCLR search dir : {_sc_search_dir}")
    if _sc_best:
        _sc_meta = torch.load(_sc_best, map_location="cpu", weights_only=False)
        _sc_epoch = _sc_meta.get("epoch", "?")
        print(f"  SimCLR checkpoint : {_sc_best}  (epoch={_sc_epoch}, loss={_sc_best_loss:.4f})")
    else:
        print(f"  SimCLR checkpoint : NOT FOUND in {_sc_search_dir}")
    if _df_path.exists():
        _df_meta = torch.load(_df_path, map_location="cpu", weights_only=False)
        _df_nd   = _df_meta["model"]["input_proj.weight"].shape[1]
        _df_cc   = _df_meta.get("class_conditional", False)
        print(f"  Diffusion ckpt    : {_df_path}")
        print(f"                      node_dim={_df_nd}, class_conditional={_df_cc}")
    else:
        print(f"  Diffusion ckpt    : NOT FOUND at {_df_path}")
    print("=" * 62 + "\n")

    print("Loading models …")
    encoder = load_simclr_encoder(device, ckpt_dir=_simclr_ckpt_dir)
    diff_model, diffusion, x_mean, x_std, _max_n = load_diffusion_model(device, ckpt_path=_diff_ckpt_path)

    # -- 3. SimCLR latent space plot -----------------------------------------
    print("Plotting SimCLR latent space …")
    fig = plot_simclr_latent_space_laundering_vs_clean(networks, df_full, encoder=encoder)
    out = results_dir / "simclr_latent_space.png"
    fig.savefig(out, dpi=300); plt.close(fig)
    print(f"Saved → {out}\n")

    # -- 4. Encode + train probe ---------------------------------------------
    print("Encoding training networks …")
    H_all_n, y_all = encode_all_networks(networks, encoder, device)
    probe = train_mlp_probe(H_all_n, y_all, device)

    # -- 5. Direction 3: embedding separation diagnostic ----------------------
    if args.sep_check:
        print("\n[Direction 3] SimCLR embedding separation diagnostic …")
        from torch_geometric.data import Data as _D
        _pyg_nets = []
        for net in networks:
            try:
                from diffusion.diff_util import network_to_dense as _ntd
                xd, ad = (net["x_dense"], net["adj_dense"]) \
                         if ("x_dense" in net and "adj_dense" in net) else _ntd(net)
                n   = xd.shape[0]
                ei  = (ad > 0.5).nonzero(as_tuple=False).T.contiguous()
                x18 = xd[:, 1:].float()   # cols 1-18: all features (strip laundering flag)
                _pyg_nets.append(_D(x=x18, edge_index=ei,
                                    y=torch.tensor([1 if len(net["laundering_nodes"]) > 0 else 0])))
            except Exception:
                continue
        if _pyg_nets:
            _sep_labels = [d.y.item() for d in _pyg_nets]
            _sep        = compute_embedding_separation(_pyg_nets, _sep_labels, encoder, device)
            print(f"  Silhouette score    : {_sep['silhouette']:.4f}  "
                  f"(>0.05 good, <0.05 poor class separation)")
            print(f"  Linear probe AUC    : {_sep['linear_probe_auc']:.4f}  "
                  f"(>0.65 good, <0.65 guidance signal too weak)")
            if _sep["silhouette"] < 0.05:
                print("  WARNING: silhouette < 0.05 — retrain SimCLR with higher supcon_weight.")
            if _sep["linear_probe_auc"] < 0.65:
                print("  WARNING: linear probe AUC < 0.65 — guidance will be unreliable.")
        print()

    # -- 6. Direction 4: tune guidance hyperparameters ------------------------
    gs = args.guidance_scale if args.guidance_scale is not None else GUIDE_SCALE
    nw = args.novelty_weight if args.novelty_weight is not None else NOVELTY_WEIGHT
    dp = args.degree_penalty if args.degree_penalty is not None else DEGREE_PENALTY
    t_start = args.t_start if args.t_start is not None else T_START

    if args.tune_guidance:
        print(f"\n[Direction 4] Tuning guidance params "
              f"({args.tune_trials} trials × {args.tune_gen_per_trial} graphs) …")
        laund_nets   = [n for n in networks if len(n["laundering_nodes"]) > 0]
        H_laund      = H_all_n[y_all == 1]
        from torch_geometric.data import Data as _D2
        train_laund_pyg = []
        for net in laund_nets:
            try:
                from diffusion.diff_util import network_to_dense as _ntd2
                xd, ad = (net["x_dense"], net["adj_dense"]) \
                         if ("x_dense" in net and "adj_dense" in net) else _ntd2(net)
                n   = xd.shape[0]
                ei  = (ad > 0.5).nonzero(as_tuple=False).T.contiguous()
                x18 = xd[:, 1:].float()
                train_laund_pyg.append(_D2(x=x18, edge_index=ei,
                                           y=torch.tensor([1])))
            except Exception:
                continue
        best_params, _ = tune_guidance_params(
            networks, encoder, probe, diff_model, diffusion,
            x_mean, x_std, H_all_n,
            train_laund_pyg, H_laund, device,
            n_trials=args.tune_trials,
            n_gen_per_trial=args.tune_gen_per_trial,
            t_start=t_start,
            results_dir=results_dir,
        )
        gs = best_params.get("guidance_scale", gs)
        nw = best_params.get("novelty_weight", nw)
        dp = best_params.get("degree_penalty", dp)
        print(f"  Using tuned params: guidance_scale={gs:.3f}  "
              f"novelty_weight={nw:.3f}  degree_penalty={dp:.3f}\n")

    # -- 7. Guided generation -------------------------------------------------
    n_gen = args.n_gen
    print(f"\nRunning guided generation  "
          f"(n={n_gen}  t_start={t_start}  gs={gs:.2f}  nw={nw:.2f}  dp={dp:.2f}) …")
    gen_outputs, gen_embeddings, seeds = run_guided_generation(
        networks, encoder, probe, diff_model, diffusion,
        x_mean, x_std, H_all_n, device,
        target_label=TARGET,
        n_gen=n_gen,
        t_start=t_start,
        guidance_scale=gs,
        novelty_weight=nw,
        degree_penalty=dp,
        guide_every=GUIDE_EVERY,
        guide_from=GUIDE_FROM,
    )

    # -- 7b. Save augmented CSV and patterns files ----------------------------
    print("\nSaving augmented transaction files …")
    _save_augmented_files(gen_outputs, ibm_csv)

    # -- 8. UMAP plot ---------------------------------------------------------
    print("\nPlotting UMAP …")
    _plot_umap(
        H_all_n, y_all.tolist(), gen_embeddings, seeds,
        TARGET, t_start, nw,
        results_dir / "simclr_guided_generation.png",
        pos_label="laundering",
    )

    # -- 9. Per-network comparison plots -------------------------------------
    print("\nPlotting generated vs closest training comparisons …")
    plot_generated_vs_closest_training(
        gen_outputs, gen_embeddings, H_all_n, networks, results_dir,
        seed_networks=seeds,
    )

    # -- 10. Fit training distribution + score --------------------------------
    print()
    mu, cov_inv, realism_scale = fit_training_distribution(networks)

    gen_scores = []
    for i, (x_d, adj_d, *_) in enumerate(gen_outputs):
        s = score_network(x_d, adj_d, encoder, H_all_n, device,
                          mu, cov_inv, realism_scale,
                          gen_embedding=torch.tensor(gen_embeddings[i]))
        gen_scores.append(s)
    _print_scores_table(gen_scores, "Generated networks")
    _save_scores_csv(gen_scores, results_dir / "generated_scores.csv", "generated")

    from diffusion.diff_util import network_to_dense as _ntd_c
    calib_nets   = _random.sample(networks, min(N_CALIB, len(networks)))
    calib_scores = []
    for net in calib_nets:
        xr, adjr = (net["x_dense"], net["adj_dense"]) \
                   if ("x_dense" in net and "adj_dense" in net) else _ntd_c(net)
        s = score_network(xr, adjr, encoder, H_all_n, device, mu, cov_inv, realism_scale)
        calib_scores.append(s)
    _print_scores_table(calib_scores, f"Calibration: {N_CALIB} real training networks")
    _save_scores_csv(calib_scores, results_dir / "calibration_scores.csv", "training")

    # -- 11. Direction 5: Tier-1 Q-score quality report ----------------------
    if gen_outputs:
        print("\n[Direction 5] Computing Tier-1 quality metrics (Q-score) …")
        laund_nets_all = [n for n in networks if len(n["laundering_nodes"]) > 0]
        from torch_geometric.data import Data as _Dq
        train_laund_pyg_q = []
        for net in laund_nets_all:
            try:
                from diffusion.diff_util import network_to_dense as _ntd_q
                xd, ad = (net["x_dense"], net["adj_dense"]) \
                         if ("x_dense" in net and "adj_dense" in net) else _ntd_q(net)
                n   = xd.shape[0]
                ei  = (ad > 0.5).nonzero(as_tuple=False).T.contiguous()
                x18 = xd[:, 1:].float()
                train_laund_pyg_q.append(_Dq(x=x18, edge_index=ei, y=torch.tensor([1])))
            except Exception:
                continue

        H_laund_q = H_all_n[y_all == 1]
        gen_pyg   = [_gen_to_pyg(x, a, n, lbl) for (x, a, n, lbl) in gen_outputs]

        quality   = score_generated_graphs(
            gen_pyg, train_laund_pyg_q, H_laund_q, encoder, device)
        print_quality_report(quality)

        _q_suffix = "ibm"
        save_quality_csv(quality, results_dir / f"graph_quality_{_q_suffix}.csv")
        plot_quality_extremes(quality, gen_pyg,
                              results_dir / f"quality_extremes_{_q_suffix}.png",
                              n=min(20, len(gen_pyg)))


# ─────────────────────────────────────────────────────────────────────────────
# Elliptic path
# ─────────────────────────────────────────────────────────────────────────────

def run_elliptic(args, device, results_dir):
    from data.elliptic_adapter import load_elliptic_pyg_graphs

    # -- 1. Load ego subgraphs -----------------------------------------------
    print("Loading Elliptic ego subgraphs …")
    graphs = load_elliptic_pyg_graphs(max_nodes=100)
    n_ill  = sum(g.y.item() == 1 for g in graphs)
    print(f"Loaded {len(graphs)} graphs ({n_ill} illicit, {len(graphs)-n_ill} licit)\n")

    # -- 2. Load models -------------------------------------------------------
    _ell_sc_dir  = CKPT_DIR / "simclr_elliptic"
    _ell_df_path = CKPT_DIR / "diffusion_elliptic" / "model.pt"

    _ell_sc_best = None; _ell_sc_best_loss = float("inf")
    for _p in sorted(_ell_sc_dir.glob("*.pt")) if _ell_sc_dir.exists() else []:
        try:
            _c = torch.load(_p, map_location="cpu", weights_only=False)
            if isinstance(_c, dict) and "loss" in _c and _c["loss"] < _ell_sc_best_loss:
                _ell_sc_best_loss = _c["loss"]; _ell_sc_best = _p
        except Exception:
            pass

    print("=" * 62)
    print("  MODEL CHECKPOINT SUMMARY  (Elliptic)")
    print("=" * 62)
    if _ell_sc_best:
        _ell_sc_meta  = torch.load(_ell_sc_best, map_location="cpu", weights_only=False)
        _ell_sc_epoch = _ell_sc_meta.get("epoch", "?")
        print(f"  SimCLR checkpoint : {_ell_sc_best}  "
              f"(epoch={_ell_sc_epoch}, loss={_ell_sc_best_loss:.4f})")
    else:
        print(f"  SimCLR checkpoint : NOT FOUND in {_ell_sc_dir}")
    if _ell_df_path.exists():
        _ell_df_meta = torch.load(_ell_df_path, map_location="cpu", weights_only=False)
        _ell_df_nd   = _ell_df_meta["model"]["input_proj.weight"].shape[1]
        _ell_df_cc   = _ell_df_meta.get("class_conditional", False)
        print(f"  Diffusion ckpt    : {_ell_df_path}")
        print(f"                      node_dim={_ell_df_nd}, class_conditional={_ell_df_cc}")
    else:
        print(f"  Diffusion ckpt    : NOT FOUND at {_ell_df_path}")
    print("=" * 62 + "\n")

    print("Loading Elliptic SimCLR encoder …")
    encoder = load_simclr_encoder_elliptic(device)
    print("Plotting SimCLR latent space …")
    _plot_latent_space_pyg(graphs, encoder, device,
                           results_dir / "simclr_latent_space.png")

    print("\nLoading Elliptic diffusion model …")
    diff_model, diffusion, x_mean, x_std, _max_n_e = load_diffusion_model_elliptic(device)

    # -- 3. Encode + train probe ---------------------------------------------
    print("\nEncoding graphs …")
    H_all_n, y_all = encode_all_pyg_graphs(graphs, encoder, device)
    probe = train_mlp_probe(H_all_n, y_all, device)

    # -- 4. Direction 3: embedding separation diagnostic ----------------------
    if args.sep_check:
        print("\n[Direction 3] SimCLR embedding separation diagnostic …")
        _sep_labels_e = [g.y.item() for g in graphs]
        _sep_e        = compute_embedding_separation(graphs, _sep_labels_e, encoder, device)
        print(f"  Silhouette score    : {_sep_e['silhouette']:.4f}")
        print(f"  Linear probe AUC    : {_sep_e['linear_probe_auc']:.4f}")
        if _sep_e["silhouette"] < 0.05:
            print("  WARNING: silhouette < 0.05 — guidance will be poorly class-conditioned.")
        print()

    # -- 5. Direction 4: tune guidance hyperparameters ------------------------
    gs = args.guidance_scale if args.guidance_scale is not None else GUIDE_SCALE
    nw = args.novelty_weight if args.novelty_weight is not None else NOVELTY_WEIGHT
    dp = args.degree_penalty if args.degree_penalty is not None else DEGREE_PENALTY
    t_start = args.t_start if args.t_start is not None else T_START

    if args.tune_guidance:
        print(f"\n[Direction 4] Tuning guidance params (Elliptic, {args.tune_trials} trials) …")
        train_laund_e = [g for g in graphs if g.y.item() == 1]
        H_laund_e     = H_all_n[y_all == 1]
        best_params_e, _ = tune_guidance_params(
            graphs, encoder, probe, diff_model, diffusion,
            x_mean, x_std, H_all_n,
            train_laund_e, H_laund_e, device,
            n_trials=args.tune_trials,
            n_gen_per_trial=args.tune_gen_per_trial,
            t_start=t_start,
            results_dir=results_dir,
        )
        gs = best_params_e.get("guidance_scale", gs)
        nw = best_params_e.get("novelty_weight", nw)
        dp = best_params_e.get("degree_penalty", dp)
        print(f"  Using tuned params: guidance_scale={gs:.3f}  "
              f"novelty_weight={nw:.3f}  degree_penalty={dp:.3f}\n")

    # -- 6. Guided generation ------------------------------------------------
    n_gen = args.n_gen
    print(f"\nRunning guided generation  "
          f"(n={n_gen}  t_start={t_start}  gs={gs:.2f}  nw={nw:.2f}  dp={dp:.2f}) …")
    gen_outputs, gen_embeddings, seeds = run_guided_generation_elliptic(
        graphs, encoder, probe, diff_model, diffusion,
        x_mean, x_std, H_all_n, device,
        target_label=TARGET,
        n_gen=n_gen,
        t_start=t_start,
        guidance_scale=gs,
        novelty_weight=nw,
        degree_penalty=dp,
        guide_every=GUIDE_EVERY,
        guide_from=GUIDE_FROM,
    )

    # -- 7. UMAP plot --------------------------------------------------------
    print("\nPlotting UMAP …")
    _plot_umap(
        H_all_n, y_all.tolist(), gen_embeddings, seeds,
        TARGET, t_start, nw,
        results_dir / "simclr_guided_generation.png",
        pos_label="illicit",
    )

    # -- 8. Per-network comparison plots -------------------------------------
    print("\nPlotting generated vs closest training comparisons …")
    _plot_comparison_pyg(gen_outputs, gen_embeddings, H_all_n, graphs,
                         results_dir, seed_graphs=seeds)

    # -- 9. Fit training distribution + score --------------------------------
    print()
    mu, cov_inv, realism_scale = _fit_training_distribution_pyg(graphs)

    gen_scores = []
    for i, (x_d, adj_d, *_) in enumerate(gen_outputs):
        s = score_network(x_d, adj_d, encoder, H_all_n, device,
                          mu, cov_inv, realism_scale,
                          gen_embedding=torch.tensor(gen_embeddings[i]))
        gen_scores.append(s)
    _print_scores_table(gen_scores, "Generated Elliptic networks")
    _save_scores_csv(gen_scores, results_dir / "generated_scores.csv", "generated")

    calib_graphs = _random.sample(graphs, min(N_CALIB, len(graphs)))
    calib_scores = []
    for g in calib_graphs:
        n   = g.x.shape[0]
        adj = torch.zeros(n, n)
        ei  = g.edge_index
        if ei.shape[1] > 0:
            valid    = ei[0] != ei[1]
            src, dst = ei[0][valid], ei[1][valid]
            inbounds = (src < n) & (dst < n)
            adj[src[inbounds], dst[inbounds]] = 1.0
        label_col = torch.full((n, 1), float(g.y.item()))
        x6        = torch.cat([label_col, g.x], dim=1)
        s = score_network(x6, adj, encoder, H_all_n, device, mu, cov_inv, realism_scale)
        calib_scores.append(s)
    _print_scores_table(calib_scores, f"Calibration: {N_CALIB} real Elliptic graphs")
    _save_scores_csv(calib_scores, results_dir / "calibration_scores.csv", "training")

    # -- 10. Direction 5: Tier-1 Q-score quality report ----------------------
    if gen_outputs:
        print("\n[Direction 5] Computing Tier-1 quality metrics (Q-score) …")
        train_laund_e_q = [g for g in graphs if g.y.item() == 1]
        H_laund_e_q     = H_all_n[y_all == 1]
        gen_pyg_e       = [_gen_to_pyg(x, a, n, lbl) for (x, a, n, lbl) in gen_outputs]
        quality_e       = score_generated_graphs(
            gen_pyg_e, train_laund_e_q, H_laund_e_q, encoder, device)
        print_quality_report(quality_e)
        save_quality_csv(quality_e, results_dir / "graph_quality_elliptic.csv")
        plot_quality_extremes(quality_e, gen_pyg_e,
                              results_dir / "quality_extremes_elliptic.png",
                              n=min(20, len(gen_pyg_e)))


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate guided generation: plots, scores, quality metrics."
    )
    parser.add_argument("--dataset", choices=["ibm", "elliptic"], default="ibm")
    parser.add_argument("--ibm-csv", type=str, default=None, metavar="PATH",
                        help="Override the IBM CSV path (default: LI-Small_Trans.csv)")
    parser.add_argument("--n-gen", type=int, default=N_GEN,
                        help=f"Number of networks to generate (default {N_GEN})")
    parser.add_argument("--t-start", type=int, default=None,
                        help=f"Diffusion noise start level (default {T_START})")
    # ── Guidance hyperparams ─────────────────────────────────────────────────
    parser.add_argument("--guidance-scale", type=float, default=None,
                        help=f"Classification guidance scale (default {GUIDE_SCALE})")
    parser.add_argument("--novelty-weight", type=float, default=None,
                        help=f"Novelty repulsion weight (default {NOVELTY_WEIGHT})")
    parser.add_argument("--degree-penalty", type=float, default=None,
                        help=f"Degree/density penalty (default {DEGREE_PENALTY})")
    # ── Direction 3: embedding separation ───────────────────────────────────
    parser.add_argument("--sep-check", action="store_true",
                        help="Compute silhouette score and linear probe AUC on training "
                             "embeddings before generation. Low scores signal weak guidance.")
    # ── Direction 4: guidance tuning ─────────────────────────────────────────
    parser.add_argument("--tune-guidance", action="store_true",
                        help="Bayesian/random search over guidance hyperparameters before "
                             "generating. Uses Q-score as the objective. Overrides "
                             "--guidance-scale / --novelty-weight / --degree-penalty.")
    parser.add_argument("--tune-trials", type=int, default=15, metavar="N",
                        help="Number of hyperparameter candidates (default 15)")
    parser.add_argument("--tune-gen-per-trial", type=int, default=6, metavar="K",
                        help="Graphs per tuning trial for Q-score estimation (default 6)")
    parser.add_argument("--results-dir", type=str, default=None, metavar="PATH",
                        help="Override the output directory (default: results/<dataset>/)")
    parser.add_argument("--ckpt-dir", type=str, default=None, metavar="PATH",
                        help="Root checkpoint directory; expects <path>/simclr/ and "
                             "<path>/diffusion/model.pt (default: checkpoints/simclr_ibm + "
                             "checkpoints/diffusion_ibm)")
    args = parser.parse_args()

    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        suffix      = args.dataset
        results_dir = ROOT_DIR / "results" / suffix
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}")
    print(f"Results will be saved to {results_dir}\n")

    if args.dataset == "ibm":
        run_ibm(args, device, results_dir)
    else:
        run_elliptic(args, device, results_dir)

    print(f"\nAll results saved to {results_dir}")


if __name__ == "__main__":
    main()
