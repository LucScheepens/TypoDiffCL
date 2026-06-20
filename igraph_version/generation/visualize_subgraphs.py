"""
visualize_subgraphs.py — plot the first N extracted ego-subgraph networks.

Shows a gallery of the raw subgraphs as they come out of
extract_networks_igraph(), before any augmentation or diffusion step.

Node colours
────────────
  Red         (#e74c3c)   laundering node
  Light blue  (#aed6f1)   clean node
  Gold ring                start / anchor node (ego centre)
  Orange fill (#f39c12)   collapsed hub node

Usage
─────
    # from igraph_version/ :
    python generation/visualize_subgraphs.py

    # show 16 laundering + 8 clean networks:
    python generation/visualize_subgraphs.py --n-laund 16 --n-clean 8

    # use a different CSV / cache:
    python generation/visualize_subgraphs.py \\
        --csv  "data/IBM/HI-Small_Trans.csv" \\
        --cache data/networks_cache_HI-Small_Trans.pkl

    # save to a custom location:
    python generation/visualize_subgraphs.py --out results/ibm/subgraph_gallery.png

Outputs
───────
    results/ibm/subgraph_gallery.png   (default)
"""

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np

# ── path setup ────────────────────────────────────────────────────────────────
_GEN_DIR   = Path(__file__).resolve().parent
ROOT_DIR   = _GEN_DIR.parent
DIFF_DIR   = ROOT_DIR / "diffusion"
SIMCLR_DIR = ROOT_DIR / "simclr"
DATA_DIR   = ROOT_DIR / "data"

for _p in (str(ROOT_DIR), str(DIFF_DIR), str(SIMCLR_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DEFAULT_CSV   = r"C:\Users\lucsc\Thesis\grad\grad\data\IBM\LI-Small_Trans.csv"
DEFAULT_CACHE = DATA_DIR / "networks_cache_LI-Small_Trans.pkl"
DEFAULT_OUT   = ROOT_DIR / "results" / "ibm" / "subgraph_gallery.png"
NCOLS         = 4


# ── network → networkx ───────────────────────────────────────────────────────

def _net_to_nx(net) -> nx.DiGraph:
    """Convert an IBM network dict to a directed NetworkX graph."""
    g = net.get("graph")
    if g is not None:
        # igraph → networkx via edge list
        G = nx.DiGraph()
        G.add_nodes_from(range(g.vcount()))
        names = None
        if "name" in g.vs.attributes():
            names = [v["name"] for v in g.vs]
            G = nx.DiGraph()
            G.add_nodes_from(names)
        for e in g.es:
            src = names[e.source] if names else e.source
            dst = names[e.target] if names else e.target
            G.add_edge(src, dst)
        return G

    # Fallback: build from transactions DataFrame
    txs = net.get("transactions")
    if txs is not None and len(txs) > 0:
        G = nx.DiGraph()
        for _, row in txs.iterrows():
            src = int(row["From_Account_int"])
            dst = int(row["To_Account_int"])
            if src != dst:
                G.add_edge(src, dst)
        return G

    return nx.DiGraph()


def _node_colours(G: nx.DiGraph, net: dict) -> list[str]:
    laundering   = net.get("laundering_nodes",  set())
    collapsed    = net.get("collapsed_nodes",    set())
    start        = net.get("start_node",         None)

    colours = []
    for node in G.nodes():
        if node in laundering:
            colours.append("#e74c3c")   # red — laundering
        elif node in collapsed:
            colours.append("#f39c12")   # orange — collapsed hub
        else:
            colours.append("#aed6f1")   # light blue — clean
    return colours


def _node_edge_colours(G: nx.DiGraph, net: dict) -> list[str]:
    """Gold border for the anchor/start node, grey otherwise."""
    start = net.get("start_node", None)
    return ["#f1c40f" if n == start else "#555555" for n in G.nodes()]


def _stats_str(G: nx.DiGraph, net: dict) -> str:
    n  = G.number_of_nodes()
    e  = G.number_of_edges()
    nl = len(net.get("laundering_nodes", set()) & set(G.nodes()))
    nc = len(net.get("collapsed_nodes",  set()) & set(G.nodes()))
    dens = e / max(n * (n - 1), 1)
    depths = net.get("node_depths", {})
    max_d  = max(depths.values()) if depths else "?"
    lines = [
        f"nodes={n}  edges={e}",
        f"laundering={nl} ({100*nl/max(n,1):.0f}%)",
        f"collapsed hubs={nc}",
        f"density={dens:.3f}  max depth={max_d}",
    ]
    return "\n".join(lines)


# ── single-network panel ─────────────────────────────────────────────────────

def _draw_network(ax, net: dict, idx: int, label: str):
    G = _net_to_nx(net)

    if G.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "empty graph", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="grey")
        ax.axis("off")
        return

    colours    = _node_colours(G, net)
    ec         = _node_edge_colours(G, net)
    n          = G.number_of_nodes()
    node_sizes = np.array([max(60, 20 * (G.in_degree(v) + G.out_degree(v) + 1))
                           for v in G.nodes()], dtype=float)
    node_sizes = np.clip(node_sizes, 40, 400)

    # spring layout; fix seed for reproducibility
    try:
        pos = nx.spring_layout(G, seed=42, k=2.5 / max(n ** 0.5, 1), iterations=60)
    except Exception:
        pos = nx.random_layout(G, seed=42)

    nx.draw_networkx_nodes(G, pos, ax=ax,
                           node_color=colours, node_size=node_sizes,
                           edgecolors=ec, linewidths=1.2, alpha=0.92)
    nx.draw_networkx_edges(G, pos, ax=ax,
                           edge_color="#777777", width=0.7, alpha=0.55,
                           arrows=True, arrowsize=8,
                           connectionstyle="arc3,rad=0.08",
                           min_source_margin=6, min_target_margin=6)

    laundering = net.get("laundering_nodes", set())
    is_laund   = len(laundering) > 0
    title_col  = "#c0392b" if is_laund else "#2471a3"
    ax.set_title(f"#{idx+1} — {label}", fontsize=8.5, fontweight="bold",
                 color=title_col, pad=3)
    ax.text(0.02, 0.02, _stats_str(G, net),
            transform=ax.transAxes, fontsize=6.5,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#cccccc", alpha=0.88))
    ax.axis("off")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv",     default=DEFAULT_CSV,
                        help="IBM AML CSV path (used to build cache if missing)")
    parser.add_argument("--cache",   default=str(DEFAULT_CACHE),
                        help="Pre-built network cache .pkl (default: networks_cache_LI-Small_Trans.pkl)")
    parser.add_argument("--n-laund", type=int, default=8,
                        help="Number of laundering networks to show (default 8)")
    parser.add_argument("--n-clean", type=int, default=4,
                        help="Number of clean networks to show (default 4)")
    parser.add_argument("--out",     default=str(DEFAULT_OUT),
                        help="Output PNG path")
    parser.add_argument("--seed",    type=int, default=0,
                        help="Random seed for sampling (default 0)")
    args = parser.parse_args()

    cache_path = Path(args.cache)
    out_path   = Path(args.out)

    # ── load networks ──────────────────────────────────────────────────────
    if cache_path.exists():
        print(f"Loading network cache: {cache_path}")
        with open(cache_path, "rb") as f:
            networks = pickle.load(f)
        # rebuild igraph objects (stripped before caching)
        try:
            from augmentation import build_igraph_from_transactions
            for net in networks:
                if "graph" not in net:
                    net["graph"] = build_igraph_from_transactions(net["transactions"])
        except Exception:
            pass
    else:
        print(f"Cache not found — extracting from {args.csv} …")
        from util import preprocess_df, extract_networks_igraph
        from augmentation import build_igraph_from_transactions
        df = preprocess_df(args.csv)
        networks = extract_networks_igraph(
            df, max_depth=4, max_networks=4000, collapse_threshold=10, max_nodes=64
        )
        for net in networks:
            net["graph"] = build_igraph_from_transactions(net["transactions"])
        nets_to_cache = [{k: v for k, v in n.items() if k != "graph"} for n in networks]
        with open(cache_path, "wb") as f:
            pickle.dump(nets_to_cache, f)
        print(f"Saved cache → {cache_path}")

    n_laund_total = sum(1 for n in networks if len(n.get("laundering_nodes", set())) > 0)
    n_clean_total = len(networks) - n_laund_total
    print(f"Loaded {len(networks)} networks  "
          f"({n_laund_total} laundering, {n_clean_total} clean)")

    # ── select networks ────────────────────────────────────────────────────
    rng = np.random.default_rng(args.seed)

    laund_nets = [n for n in networks if len(n.get("laundering_nodes", set())) > 0]
    clean_nets = [n for n in networks if len(n.get("laundering_nodes", set())) == 0]

    n_l = min(args.n_laund, len(laund_nets))
    n_c = min(args.n_clean, len(clean_nets))

    sel_laund = [laund_nets[i] for i in rng.choice(len(laund_nets), n_l, replace=False)]
    sel_clean = [clean_nets[i] for i in rng.choice(len(clean_nets), n_c, replace=False)]

    # Interleave: laundering first, then clean
    nets_to_plot = [(net, "LAUNDERING") for net in sel_laund] + \
                   [(net, "CLEAN")      for net in sel_clean]
    total = len(nets_to_plot)

    print(f"Plotting {n_l} laundering + {n_c} clean networks …")

    # ── figure layout ──────────────────────────────────────────────────────
    ncols = min(NCOLS, total)
    nrows = (total + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 4.5, nrows * 4.0))
    axes = np.array(axes).flatten() if total > 1 else [axes]

    for idx, (net, label) in enumerate(nets_to_plot):
        _draw_network(axes[idx], net, idx, label)

    # hide unused panels
    for ax in axes[total:]:
        ax.set_visible(False)

    # ── legend ────────────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(facecolor="#e74c3c", edgecolor="#555", label="Laundering node"),
        mpatches.Patch(facecolor="#aed6f1", edgecolor="#555", label="Clean node"),
        mpatches.Patch(facecolor="#f39c12", edgecolor="#555", label="Collapsed hub"),
        mpatches.Patch(facecolor="white",   edgecolor="#f1c40f",
                       linewidth=2, label="Anchor / start node (gold border)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=4, fontsize=8.5, framealpha=0.9,
               bbox_to_anchor=(0.5, 0.0))

    fig.suptitle(
        f"Extracted Ego-Subgraph Networks — LI-Small\n"
        f"First {n_l} laundering  +  {n_c} clean  "
        f"(cache: {cache_path.name})",
        fontsize=11, y=1.01,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
