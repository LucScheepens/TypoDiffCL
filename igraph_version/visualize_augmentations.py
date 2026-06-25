"""
visualize_augmentations.py
──────────────────────────────────────────────────────────────────────────────
Standalone script that generates thesis-quality figures for the four structural
graph augmentations used in the SimCLR pipeline:

  1. Subgraph Crop     – BFS from a laundering node, keep a fraction of nodes
  2. Edge Deletion     – Remove random non-bridge edges
  3. Node Deletion     – Remove random non-articulation-point, non-laundering nodes
  4. Node Addition     – Insert synthetic nodes by subdividing existing edges

No dataset loading required: all figures use a synthetic AML-pattern subgraph.

Outputs (saved to --out directory, default: figures/):
  augmentation_overview.{pdf,png}      – 2×3 panel grid
  augmentation_before_after.{pdf,png}  – 4-row before / after comparison

Usage:
    python visualize_augmentations.py
    python visualize_augmentations.py --out thesis_figures/
"""

import argparse
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── Colour palette (colour-blind friendly) ─────────────────────────────────────
C_LAUND    = "#e63946"   # red    – laundering / suspicious nodes
C_CLEAN    = "#457b9d"   # blue   – clean nodes
C_ADDED    = "#2dc653"   # green  – newly added synthetic nodes
C_GHOST    = "#ced4da"   # grey   – cropped-out ghost nodes
C_EDGE     = "#343a40"   # dark   – normal edges
C_REM_EDGE = "#e63946"   # red    – removed edges (dashed)
C_GHOST_E  = "#ced4da"   # grey   – ghost edges behind crop


# ── Synthetic graph ───────────────────────────────────────────────────────────

def build_synthetic_aml_graph():
    """
    Build a small synthetic scatter-gather AML transaction subgraph.

    Topology
    --------
    Feeder layer  →  Source (L)  →  3 Intermediaries (L)  →  Sink (L)  →  Exit nodes
    Plus a small side chain off one intermediary.

    Returns
    -------
    G          : nx.DiGraph – the full 12-node subgraph
    laundering : set        – node IDs flagged as suspicious
    pos        : dict       – fixed layout positions (shared across all panels)
    """
    G = nx.DiGraph()

    # Feeder → Source
    G.add_edge(0, 1)
    G.add_edge(9, 1)

    # Source → Intermediaries
    G.add_edge(1, 2)
    G.add_edge(1, 3)
    G.add_edge(1, 4)

    # Intermediaries → Sink
    G.add_edge(2, 5)
    G.add_edge(3, 5)
    G.add_edge(4, 5)

    # Sink → Exit
    G.add_edge(5, 6)
    G.add_edge(6, 7)
    G.add_edge(6, 8)

    # Side chain off intermediary 2
    G.add_edge(2, 10)
    G.add_edge(10, 11)

    laundering = {1, 2, 3, 4, 5}

    # Fixed hierarchical layout
    pos = {
        9:  (-2.5,  1.6),
        0:  (-2.5, -0.1),
        1:  (-1.2,  0.75),
        2:  ( 0.0,  1.7),
        3:  ( 0.0,  0.75),
        4:  ( 0.0, -0.2),
        5:  ( 1.3,  0.75),
        6:  ( 2.5,  0.75),
        7:  ( 3.5,  1.4),
        8:  ( 3.5,  0.1),
        10: ( 0.8,  2.6),
        11: ( 1.6,  3.2),
    }

    return G, laundering, pos


# ── Drawing helpers ───────────────────────────────────────────────────────────

def _color_map(G, laundering, added=None):
    added = added or set()
    return {
        n: C_ADDED if n in added else C_LAUND if n in laundering else C_CLEAN
        for n in G.nodes()
    }


def draw_panel(
    ax, G, pos, laundering, title,
    removed_edges=None,
    added_nodes=None,
    ghost_nodes=None,
    ghost_edges=None,
):
    """
    Draw one augmentation panel.

    Parameters
    ----------
    removed_edges : set of (u,v) – drawn dashed red, then visually removed
    added_nodes   : set of node IDs – drawn green
    ghost_nodes   : set of node IDs – drawn as light grey (excluded by crop)
    ghost_edges   : set of (u,v)    – drawn light grey dashed behind ghost nodes
    """
    removed_edges = removed_edges or set()
    added_nodes   = added_nodes   or set()
    ghost_nodes   = ghost_nodes   or set()
    ghost_edges   = ghost_edges   or set()

    color_map  = _color_map(G, laundering, added=added_nodes)
    all_nodes  = list(G.nodes())
    active     = [n for n in all_nodes if n not in ghost_nodes]
    ghosts     = [n for n in all_nodes if n in ghost_nodes]

    # ── Ghost edges (light grey dashed) ──────────────────────────────
    ghost_edge_list = [e for e in G.edges() if e in ghost_edges]
    if ghost_edge_list:
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            edgelist=ghost_edge_list,
            edge_color=C_GHOST_E,
            style="dashed",
            arrows=False,
            width=1.0,
            alpha=0.4,
        )

    # ── Ghost nodes ───────────────────────────────────────────────────
    if ghosts:
        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            nodelist=ghosts,
            node_color=C_GHOST,
            node_size=380,
            alpha=0.35,
            edgecolors=C_GHOST,
            linewidths=1.0,
        )
        nx.draw_networkx_labels(
            G, pos, ax=ax,
            labels={n: str(n) for n in ghosts},
            font_size=7,
            font_color="#adb5bd",
        )

    # ── Normal active edges ───────────────────────────────────────────
    normal_edges  = [e for e in G.edges()
                     if e not in removed_edges
                     and e[0] not in ghost_nodes
                     and e[1] not in ghost_nodes]
    if normal_edges:
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            edgelist=normal_edges,
            edge_color=C_EDGE,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=14,
            width=1.8,
            connectionstyle="arc3,rad=0.07",
        )

    # ── Removed edges (dashed red, shown in place) ───────────────────
    deleted_shown = [e for e in G.edges()
                     if e in removed_edges
                     and e[0] not in ghost_nodes
                     and e[1] not in ghost_nodes]
    if deleted_shown:
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            edgelist=deleted_shown,
            edge_color=C_REM_EDGE,
            style="dashed",
            arrows=True,
            arrowstyle="-|>",
            arrowsize=12,
            width=1.6,
            alpha=0.50,
            connectionstyle="arc3,rad=0.07",
        )

    # ── Active nodes ──────────────────────────────────────────────────
    if active:
        node_colors = [color_map[n] for n in active]
        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            nodelist=active,
            node_color=node_colors,
            node_size=500,
            edgecolors="white",
            linewidths=1.8,
        )
        nx.draw_networkx_labels(
            G, pos, ax=ax,
            labels={n: str(n) for n in active},
            font_size=8,
            font_color="white",
            font_weight="bold",
        )

    ax.set_title(title, fontsize=10, fontweight="bold", pad=7)
    ax.set_xlim(-3.2, 4.2)
    ax.set_ylim(-0.9, 3.9)
    ax.axis("off")


# ── Augmentation functions (pure NetworkX, deterministic) ─────────────────────

def aug_crop(G, laundering, ratio=0.72, seed=42):
    """BFS from a laundering node; keep the first `ratio` fraction of visited nodes."""
    rng = random.Random(seed)
    start = rng.choice(sorted(laundering))

    visited, queue, seen = [], [start], {start}
    while queue:
        n = queue.pop(0)
        visited.append(n)
        for nbr in sorted(set(G.successors(n)) | set(G.predecessors(n))):
            if nbr not in seen:
                seen.add(nbr)
                queue.append(nbr)

    keep_n    = max(2, int(len(visited) * ratio))
    kept      = set(visited[:keep_n])
    not_reached = set(G.nodes()) - seen
    cropped_out = (set(visited[keep_n:]) | not_reached)

    ghost_edges = {
        (u, v) for u, v in G.edges()
        if u in cropped_out or v in cropped_out
    }
    return kept, cropped_out, ghost_edges


def aug_edge_drop(G, frac=0.35, seed=42):
    """Remove a fraction of non-bridge directed edges at random."""
    rng = random.Random(seed)
    G_undir  = G.to_undirected()
    bridges  = set(nx.bridges(G_undir))
    bridge_pairs = {(min(u, v), max(u, v)) for u, v in bridges}

    non_bridges = [
        (u, v) for u, v in G.edges()
        if (min(u, v), max(u, v)) not in bridge_pairs
    ]
    rng.shuffle(non_bridges)
    n_del = max(1, int(len(non_bridges) * frac))
    return set(non_bridges[:n_del])


def aug_node_delete(G, laundering, frac=0.30, seed=42):
    """Remove random non-articulation-point, non-laundering nodes."""
    rng = random.Random(seed)
    G_undir   = G.to_undirected()
    art_pts   = set(nx.articulation_points(G_undir))
    deletable = [n for n in G.nodes() if n not in art_pts and n not in laundering]
    rng.shuffle(deletable)
    n_del = max(1, int(len(deletable) * frac))
    return set(deletable[:n_del])


def aug_node_add(G, pos, n_new=3, seed=42):
    """
    Subdivide `n_new` random edges, inserting a synthetic node on each.
    Returns (G_new, added_nodes_set, pos_new).
    """
    rng   = random.Random(seed)
    G2    = G.copy()
    edges = list(G2.edges())
    rng.shuffle(edges)

    new_id   = max(G2.nodes()) + 100
    added    = set()
    pos_new  = dict(pos)

    for u, v in edges[:n_new]:
        if not G2.has_edge(u, v):
            continue
        G2.remove_edge(u, v)
        G2.add_node(new_id)
        G2.add_edge(u, new_id)
        G2.add_edge(new_id, v)
        if u in pos and v in pos:
            pos_new[new_id] = (
                (pos[u][0] + pos[v][0]) / 2,
                (pos[u][1] + pos[v][1]) / 2 + 0.35,
            )
        added.add(new_id)
        new_id += 1

    return G2, added, pos_new


# ── Legend elements ───────────────────────────────────────────────────────────

def _legend_handles():
    return [
        mpatches.Patch(color=C_LAUND,              label="Laundering node"),
        mpatches.Patch(color=C_CLEAN,              label="Clean node"),
        mpatches.Patch(color=C_ADDED,              label="Added synthetic node"),
        mpatches.Patch(color=C_GHOST,   alpha=0.5, label="Cropped-out node (ghost)"),
        mpatches.Patch(color=C_REM_EDGE, alpha=0.6, label="Removed edge (dashed)"),
    ]


# ── Figure 1: Overview 2×3 grid ───────────────────────────────────────────────

def make_overview(out_dir):
    """2×3 panel: original + 4 augmentations + description box."""
    G, laundering, pos = build_synthetic_aml_graph()

    kept, cropped_out, ghost_edges = aug_crop(G, laundering)
    removed_edges                  = aug_edge_drop(G)
    removed_nodes                  = aug_node_delete(G, laundering)
    G_add, added_nodes, pos_ext    = aug_node_add(G, pos)

    G_del = G.copy()
    G_del.remove_nodes_from(removed_nodes)
    pos_del = {n: p for n, p in pos.items() if n not in removed_nodes}
    laund_del = laundering - removed_nodes

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    # (a) Original
    draw_panel(axes[0], G, pos, laundering, "(a)  Original Subgraph")

    # (b) Crop
    draw_panel(
        axes[1], G, pos, laundering,
        f"(b)  Subgraph Crop  (kept {len(kept)}/{G.number_of_nodes()} nodes)",
        ghost_nodes=cropped_out,
        ghost_edges=ghost_edges,
    )

    # (c) Edge Deletion
    draw_panel(
        axes[2], G, pos, laundering,
        f"(c)  Edge Deletion  ({len(removed_edges)} non-bridge edges removed)",
        removed_edges=removed_edges,
    )

    # (d) Node Deletion
    draw_panel(
        axes[3], G_del, pos_del, laund_del,
        f"(d)  Node Deletion  ({len(removed_nodes)} non-critical nodes removed)",
    )

    # (e) Node Addition
    draw_panel(
        axes[4], G_add, pos_ext, laundering,
        f"(e)  Node Addition  ({len(added_nodes)} synthetic nodes inserted)",
        added_nodes=added_nodes,
    )

    # (f) Description
    axes[5].axis("off")
    desc = (
        "Two augmented views of the same\n"
        "subgraph are created per training step.\n\n"
        "Each augmentation is applied\n"
        "stochastically with independent\n"
        "random seeds, producing diverse\n"
        "positive pairs for SimCLR."
    )
    axes[5].text(
        0.5, 0.5, desc,
        ha="center", va="center",
        fontsize=11, style="italic", linespacing=1.6,
        transform=axes[5].transAxes,
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#f8f9fa", edgecolor="#dee2e6", lw=1.5),
    )

    fig.legend(
        handles=_legend_handles(),
        loc="lower center", ncol=5,
        fontsize=9.5, frameon=True,
        bbox_to_anchor=(0.5, -0.01),
        edgecolor="#dee2e6",
    )

    fig.suptitle(
        "Structural Graph Augmentation Strategies for Contrastive Learning",
        fontsize=14, fontweight="bold", y=1.01,
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    _save(fig, out_dir, "augmentation_overview")


# ── Figure 2: Before / After 4-row comparison ─────────────────────────────────

def make_before_after(out_dir):
    """4 rows × 2 cols: left = original, right = augmented with changes highlighted."""
    G, laundering, pos = build_synthetic_aml_graph()

    kept, cropped_out, ghost_edges = aug_crop(G, laundering)
    removed_edges                  = aug_edge_drop(G)
    removed_nodes                  = aug_node_delete(G, laundering)
    G_add, added_nodes, pos_ext    = aug_node_add(G, pos)

    G_del     = G.copy()
    G_del.remove_nodes_from(removed_nodes)
    pos_del   = {n: p for n, p in pos.items() if n not in removed_nodes}
    laund_del = laundering - removed_nodes

    row_configs = [
        # (row_label, after_G, after_pos, after_laund, kwargs for after draw_panel, after_title)
        (
            "Subgraph\nCrop",
            G, pos, laundering,
            dict(ghost_nodes=cropped_out, ghost_edges=ghost_edges),
            f"After Crop  ({len(kept)}/{G.number_of_nodes()} nodes kept via BFS)",
        ),
        (
            "Edge\nDeletion",
            G, pos, laundering,
            dict(removed_edges=removed_edges),
            f"After Edge Deletion  ({len(removed_edges)} non-bridge edges removed)",
        ),
        (
            "Node\nDeletion",
            G_del, pos_del, laund_del,
            {},
            f"After Node Deletion  ({len(removed_nodes)} non-critical nodes removed)",
        ),
        (
            "Node\nAddition",
            G_add, pos_ext, laundering,
            dict(added_nodes=added_nodes),
            f"After Node Addition  ({len(added_nodes)} synthetic nodes inserted via edge subdivision)",
        ),
    ]

    fig, axes = plt.subplots(4, 2, figsize=(14, 17))

    for row, (label, aG, apos, alaund, akwargs, after_title) in enumerate(row_configs):
        # Left: original
        draw_panel(axes[row, 0], G, pos, laundering, "Original Subgraph")

        # Right: augmented
        draw_panel(axes[row, 1], aG, apos, alaund, after_title, **akwargs)

        # Row label on left margin
        axes[row, 0].set_ylabel(
            label, fontsize=11, fontweight="bold",
            rotation=0, labelpad=55,
            va="center",
        )

        # Separator line between rows (except last)
        if row < 3:
            line_y = axes[row, 0].get_position().y0 - 0.005
            fig.add_artist(
                plt.Line2D([0.05, 0.95], [line_y, line_y],
                           transform=fig.transFigure,
                           color="#dee2e6", linewidth=0.8, linestyle="--")
            )

    # Add column headers above first row
    axes[0, 0].set_title("Original Subgraph", fontsize=11, fontweight="bold", pad=7)

    fig.legend(
        handles=_legend_handles(),
        loc="lower center", ncol=5,
        fontsize=9.5, frameon=True,
        bbox_to_anchor=(0.5, -0.015),
        edgecolor="#dee2e6",
    )

    fig.suptitle(
        "Before vs. After: Graph Augmentation Transformations",
        fontsize=14, fontweight="bold",
    )

    plt.tight_layout(rect=[0.06, 0.04, 1, 0.97])
    _save(fig, out_dir, "augmentation_before_after")


# ── Save helper ───────────────────────────────────────────────────────────────

def _save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        path = os.path.join(out_dir, f"{name}.{ext}")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"  saved -> {path}")
    plt.close(fig)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate thesis augmentation figures.")
    parser.add_argument("--out", default="figures", help="Output directory (default: figures/)")
    args = parser.parse_args()

    print("Generating augmentation overview …")
    make_overview(args.out)

    print("Generating before/after comparison …")
    make_before_after(args.out)

    print(f"\nDone — figures written to '{args.out}/'")


if __name__ == "__main__":
    main()
