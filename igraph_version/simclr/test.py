"""
test.py — evaluate guided generation and save plots + scores to results/

Run:
    python test.py

Outputs (in results/):
    simclr_latent_space.png          SimCLR encoder space (laundering vs clean)
    simclr_guided_generation.png     UMAP with generated networks highlighted
    generated_scores.csv             Realism / novelty scores for generated nets
    calibration_scores.csv           Same scores for 5 sampled training networks
"""

import sys
import random as _random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive — must come before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import torch
import umap

# ── path setup ────────────────────────────────────────────────────────────────
SIMCLR_DIR = Path(__file__).resolve().parent
if str(SIMCLR_DIR.parent) not in sys.path:
    sys.path.insert(0, str(SIMCLR_DIR.parent))
if str(SIMCLR_DIR) not in sys.path:
    sys.path.insert(0, str(SIMCLR_DIR))

from util import (
    preprocess_df,
    extract_laundering_networks_igraph,
    extract_non_laundering_networks_igraph,
)
from augmentation import build_igraph_from_transactions
from plotting_helpers import plot_simclr_latent_space_laundering_vs_clean
from generation import (
    load_simclr_encoder,
    load_diffusion_model,
    encode_all_networks,
    train_mlp_probe,
    run_guided_generation,
)
from scoring import (
    fit_training_distribution,
    score_network,
    _print_scores_table,
    _save_scores_csv,
)
from latent_seed_generation import plot_generated_vs_closest_training

# ── config ────────────────────────────────────────────────────────────────────
CSV_PATH       = r"C:\Users\lucsc\Thesis\grad\grad\data\IBM\HI-Small_Trans.csv"
TARGET         = 1       # 1 = generate laundering-like, 0 = clean-like
N_GEN          = 8
T_START        = 150
GUIDE_SCALE    = 2.0
NOVELTY_WEIGHT = 2.0
GUIDE_EVERY    = 5
GUIDE_FROM     = 0.25
N_CALIB        = 5


def _plot_umap(H_all_n, all_labels, gen_embeddings, seeds, target_label,
               t_start, novelty_weight, save_path):
    n_train = len(all_labels)
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
               label=f"Generated ({'laundering' if target_label else 'clean'})")
    cbar = fig.colorbar(sc, ax=ax, ticks=[0, 1])
    cbar.ax.set_yticklabels(["Clean", "Laundering"])
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


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir = SIMCLR_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    print(f"Results will be saved to {results_dir}\n")

    # ── 1. Load and prepare data (with cache) ────────────────────────────────
    import pickle
    CACHE_PATH = SIMCLR_DIR / "networks_cache.pkl"

    df_full = preprocess_df(CSV_PATH)

    if CACHE_PATH.exists():
        print(f"Loading networks from cache: {CACHE_PATH} …")
        with open(CACHE_PATH, "rb") as f:
            networks = pickle.load(f)
        # igraph graphs are not cached — rebuild from transactions
        for net in networks:
            net["graph"] = build_igraph_from_transactions(net["transactions"])
        n_laund = sum(1 for n in networks if len(n["laundering_nodes"]) > 0)
        print(f"Loaded {len(networks)} networks ({n_laund} laundering, "
              f"{len(networks) - n_laund} clean) from cache\n")
    else:
        print("Extracting networks from data (this may take a while) …")
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
        # Save without graph objects (rebuilt on load)
        networks_to_cache = [{k: v for k, v in net.items() if k != "graph"}
                             for net in networks]
        with open(CACHE_PATH, "wb") as f:
            pickle.dump(networks_to_cache, f)
        print(f"Saved network cache → {CACHE_PATH}")
        print(f"Loaded {len(networks)} networks "
              f"({len(with_laund)} laundering, {len(non_laund)} clean)\n")

    # ── 2. SimCLR latent space plot ───────────────────────────────────────────
    print("Plotting SimCLR latent space …")
    fig = plot_simclr_latent_space_laundering_vs_clean(networks, df_full)
    out_path = results_dir / "simclr_latent_space.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved → {out_path}\n")

    # ── 3. Load models ────────────────────────────────────────────────────────
    print("Loading models …")
    encoder = load_simclr_encoder(device)
    diff_model, diffusion, x_mean, x_std = load_diffusion_model(device)

    # ── 4. Encode training networks + train binary probe ─────────────────────
    print("\nEncoding training networks …")
    H_all_n, y_all = encode_all_networks(networks, encoder, device)
    probe = train_mlp_probe(H_all_n, y_all, device)

    # ── 5. Guided generation ──────────────────────────────────────────────────
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

    # ── 6. UMAP plot of generation results ───────────────────────────────────
    print("\nPlotting UMAP …")
    _plot_umap(
        H_all_n, y_all.tolist(), gen_embeddings, seeds,
        TARGET, T_START, NOVELTY_WEIGHT,
        results_dir / "simclr_guided_generation.png",
    )

    # ── 7. Per-network comparison plots ──────────────────────────────────────
    print("\nPlotting generated vs closest training comparisons …")
    plot_generated_vs_closest_training(
        gen_outputs, gen_embeddings, H_all_n, networks, results_dir,
        seed_networks=seeds,
    )

    # ── 8. Fit training distribution for scoring ──────────────────────────────
    print()
    mu, cov_inv, realism_scale = fit_training_distribution(networks)

    # ── 9. Score generated networks ───────────────────────────────────────────
    gen_scores = []
    for i, (x_d, adj_d, _) in enumerate(gen_outputs):
        s = score_network(
            x_d, adj_d, encoder, H_all_n, device,
            mu, cov_inv, realism_scale,
            gen_embedding=torch.tensor(gen_embeddings[i]),
        )
        gen_scores.append(s)
    _print_scores_table(gen_scores, "Generated networks")
    _save_scores_csv(gen_scores, results_dir / "generated_scores.csv", "generated")

    # ── 10. Calibration: score a sample of real training networks ─────────────
    from diffusion.diff_util import network_to_dense as _ntd
    calib_nets  = _random.sample(networks, min(N_CALIB, len(networks)))
    calib_scores = []
    for net in calib_nets:
        if "x_dense" in net and "adj_dense" in net:
            xr, adjr = net["x_dense"], net["adj_dense"]
        else:
            xr, adjr = _ntd(net)
        s = score_network(xr, adjr, encoder, H_all_n, device,
                          mu, cov_inv, realism_scale)
        calib_scores.append(s)
    _print_scores_table(calib_scores, f"Calibration: {N_CALIB} real training networks")
    _save_scores_csv(calib_scores, results_dir / "calibration_scores.csv", "training")

    print(f"\nAll results saved to {results_dir}")


if __name__ == "__main__":
    main()
