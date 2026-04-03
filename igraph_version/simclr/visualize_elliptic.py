"""
visualize_elliptic.py
─────────────────────────────────────────────────────────────────────────────
Quick visualization of a handful of Elliptic ego-subgraphs.

Layout
──────
  A grid of N_SHOW networks — illicit ones on the left, licit on the right.
  Each panel shows:
    - The ego-subgraph topology (spring layout)
    - Node colour = PageRank (viridis), scaled per graph
    - Node size   = normalised degree
    - Edge colour = grey
    - Title bar:  label | nodes | edges | density | mean degree

Usage
─────
  python visualize_elliptic.py              # default: 4 illicit + 4 licit
  python visualize_elliptic.py --n 6        # 6 of each class
  python visualize_elliptic.py --depth 2    # 2-hop ego (faster)
  python visualize_elliptic.py --seed 7     # reproducible random pick
"""

import argparse
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx
import numpy as np
import torch

# ── path setup ────────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
for _p in (str(HERE.parent), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from grad.igraph_version.archive.elliptic_adapter import load_elliptic_pyg_graphs


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def pyg_to_nx(g):
    """Build a NetworkX graph from a PyG Data object."""
    n  = g.x.shape[0]
    G  = nx.Graph()
    G.add_nodes_from(range(n))
    ei = g.edge_index
    if ei.shape[1] > 0:
        for u, v in zip(ei[0].tolist(), ei[1].tolist()):
            if u < v:
                G.add_edge(u, v)
    return G


def graph_stats(G, n):
    ne   = G.number_of_edges()
    dens = ne / max(n * (n - 1) / 2, 1)
    degs = [d for _, d in G.degree()]
    return ne, dens, float(np.mean(degs)) if degs else 0.0


def draw_ego(ax, g, title_prefix=""):
    """Draw one ego-subgraph on `ax`."""
    n   = g.x.shape[0]
    G   = pyg_to_nx(g)
    ne, dens, mean_deg = graph_stats(G, n)

    # Node attributes from .x  [degree, betweenness, clustering, pagerank, assort]
    pagerank = g.x[:, 3].numpy()           # col 3
    norm_deg = g.x[:, 0].numpy()           # col 0  (already normalised)

    # Layout
    pos = nx.spring_layout(G, seed=42, k=2.0 / max(n ** 0.5, 1))

    # Colours: PageRank → viridis (per-graph normalised)
    pr_min, pr_max = pagerank.min(), pagerank.max()
    pr_norm = (pagerank - pr_min) / max(pr_max - pr_min, 1e-8)
    node_colors = cm.viridis(pr_norm)

    # Sizes: normalised degree, clamped to [30, 250]
    node_sizes = np.clip(norm_deg * 220 + 30, 30, 250)

    nx.draw_networkx_nodes(G, pos, ax=ax,
                           node_color=node_colors, node_size=node_sizes,
                           alpha=0.92, linewidths=0.4,
                           edgecolors="white")
    nx.draw_networkx_edges(G, pos, ax=ax,
                           edge_color="#888888", width=0.6, alpha=0.5,
                           arrows=False)

    label_str = "ILLICIT" if g.y.item() == 1 else "licit"
    color_str = "#c0392b" if g.y.item() == 1 else "#2471a3"
    ax.set_title(
        f"{title_prefix}{label_str}\n"
        f"n={n}  e={ne}  dens={dens:.3f}  <deg>={mean_deg:.1f}",
        fontsize=9, color=color_str, fontweight="bold",
        pad=4,
    )
    ax.axis("off")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",     type=int, default=4,
                        help="Number of graphs per class to show (default 4)")
    parser.add_argument("--depth", type=int, default=4,
                        help="BFS hop depth for ego extraction (default 4)")
    parser.add_argument("--seed",  type=int, default=0,
                        help="Random seed for graph selection (default 0)")
    parser.add_argument("--out",   type=str, default=None,
                        help="Output PNG path (default: visualize_elliptic.png next to script)")
    args = parser.parse_args()

    out_path = Path(args.out) if args.out else HERE / "visualize_elliptic.png"
    random.seed(args.seed)

    print(f"Loading Elliptic graphs (depth={args.depth}) …")
    graphs = load_elliptic_pyg_graphs(max_nodes=100, depth=args.depth)

    illicit = [g for g in graphs if g.y.item() == 1]
    licit   = [g for g in graphs if g.y.item() == 0]
    print(f"  {len(illicit)} illicit  |  {len(licit)} licit")

    n_show = min(args.n, len(illicit), len(licit))
    ill_sample  = random.sample(illicit, n_show)
    lit_sample  = random.sample(licit,   n_show)

    # Interleave: [ill0, lit0, ill1, lit1, ...]
    samples = []
    for a, b in zip(ill_sample, lit_sample):
        samples += [a, b]

    n_cols = min(4, len(samples))
    n_rows = (len(samples) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 4.5, n_rows * 4.2))
    axes = np.array(axes).reshape(-1)   # flatten for easy indexing

    for i, g in enumerate(samples):
        draw_ego(axes[i], g)

    # Hide unused panels
    for j in range(len(samples), len(axes)):
        axes[j].axis("off")

    # Shared colorbar (PageRank, viridis)
    sm = plt.cm.ScalarMappable(cmap="viridis",
                               norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.tolist(), shrink=0.6, pad=0.02)
    cbar.set_label("PageRank (per-graph normalised)", fontsize=10)

    fig.suptitle(
        f"Elliptic Bitcoin Dataset — ego-subgraphs (depth={args.depth}, max_nodes=100)\n"
        "Node colour = PageRank · Node size = normalised degree",
        fontsize=12, fontweight="bold", y=1.01,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
