"""Standalone script: regenerate generated_gallery.png from cache only."""
import pickle, sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
for _p in (_ROOT, _ROOT / "simclr", _ROOT / "diffusion", _ROOT / "generation"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
import torch

# ── inline the only pieces we need from viz_real_generated.py ────────────────

C_LAUND = "#e63946"
C_CLEAN = "#457b9d"
C_EDGE  = "#6c757d"


def load_gen_cache(path):
    with open(path, "rb") as f:
        raw = pickle.load(f)
    graphs = []
    for item in raw:
        if isinstance(item, dict):
            graphs.append(item)
        else:
            graphs.append(item)
    print(f"  {len(raw)} generated graphs loaded from {Path(path).name}")
    return raw


def _draw_generated_panel(ax, data, title):
    if isinstance(data, dict):
        adj         = data["adj"]
        n           = data["n"]
        laund_flags = data["laund_flags"]
        G = nx.DiGraph()
        G.add_nodes_from(range(n))
        ei_np = adj.nonzero(as_tuple=False).numpy()
        for r, c in ei_np:
            if r != c:
                G.add_edge(int(r), int(c))
        sizes = np.array(
            [max(80, 40 * (G.in_degree(v) + G.out_degree(v) + 1)) for v in G.nodes()],
            dtype=float,
        )
        sizes = np.clip(sizes, 80, 500).tolist()
    else:
        pyg_data = data
        ei = pyg_data.edge_index
        n  = pyg_data.x.shape[0]
        x_feat = pyg_data.x
        if n == 0 or (ei.shape[1] == 0 and n < 2):
            ax.text(0.5, 0.5, "empty", ha="center", va="center",
                    transform=ax.transAxes, color="grey", fontsize=9)
            ax.axis("off")
            return
        G = nx.DiGraph()
        G.add_nodes_from(range(n))
        if ei.shape[1] > 0:
            ei_np = ei.cpu().numpy()
            for s, t in zip(ei_np[0], ei_np[1]):
                if s != t:
                    G.add_edge(int(s), int(t))
        try:
            bc = nx.betweenness_centrality(G.to_undirected())
        except Exception:
            bc = {node: 0.0 for node in G.nodes()}
        bc_vals   = np.array([bc[v] for v in G.nodes()])
        bc_range  = bc_vals.max() - bc_vals.min()
        threshold = (bc_vals.min() + 0.3 * bc_range) if bc_range > 0 else bc_vals.max() + 1
        laund_flags = bc_vals >= threshold
        deg_norm = x_feat[:, 0].cpu().numpy()
        sizes    = np.clip(deg_norm * 400 + 80, 80, 500).tolist()

    if G.number_of_nodes() == 0:
        ax.axis("off")
        return

    k = 2.0 / max(n ** 0.5, 1)
    try:
        pos = nx.spring_layout(G, seed=42, k=k, iterations=80)
    except Exception:
        pos = nx.random_layout(G, seed=42)

    colors = [C_LAUND if lf else C_CLEAN for lf in laund_flags]
    nx.draw_networkx_nodes(G, pos, ax=ax,
                           node_color=colors, node_size=sizes,
                           edgecolors="white", linewidths=1.4, alpha=0.92)
    if G.number_of_edges() > 0:
        nx.draw_networkx_edges(G, pos, ax=ax,
                               edge_color=C_EDGE, arrows=True,
                               arrowstyle="-|>", arrowsize=9, width=1.2,
                               connectionstyle="arc3,rad=0.06", alpha=0.60)
    ax.set_title(title, fontsize=18, fontweight="bold", pad=6)
    ax.axis("off")


def make_generated_gallery(gen_data, out_dir, nrows=3, ncols=3, seed=42):
    import random as _random
    rng   = _random.Random(seed)
    laund = [d for d in gen_data if (d.y.item() == 1 if hasattr(d, "y") else True)]
    if not laund:
        print("[skip] No generated laundering graphs.")
        return
    n_panels = nrows * ncols
    sel = rng.sample(laund, min(n_panels, len(laund)))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5.5, nrows * 5.5))
    axes = np.array(axes).flatten()

    for i, (ax, item) in enumerate(zip(axes, sel)):
        _draw_generated_panel(ax, item, title=f"Generated #{i+1}")

    for ax in axes[len(sel):]:
        ax.set_visible(False)

    handles = [
        mpatches.Patch(color=C_LAUND, label="Laundering node (top-30% betweenness)"),
        mpatches.Patch(color=C_CLEAN, label="Non-laundering node"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2,
               fontsize=16, frameon=True, edgecolor="#dee2e6",
               bbox_to_anchor=(0.5, -0.01))

    fig.suptitle(
        "Gallery: Diffusion-Generated Laundering Subgraphs (IBM AML)\n"
        "All graphs steered toward label=1 via SimCLR-guided reverse diffusion",
        fontsize=22, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    out_path = Path(out_dir) / "generated_gallery.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    gen_data = load_gen_cache(
        "data/gen_cache_LI-Small_Trans_n500_t150_ld10_encenc_full.pkl"
    )
    out_dir = Path("figures_real")
    out_dir.mkdir(exist_ok=True)
    make_generated_gallery(gen_data, out_dir, seed=42)
