"""
elliptic_adapter.py
─────────────────────────────────────────────────────────────────────────────
Convert the Elliptic Bitcoin Dataset into PyG Data objects that are
compatible with evaluate_classifiers.py.

Dataset description
───────────────────
  Source   : https://www.kaggle.com/datasets/ellipticco/elliptic-data-set
  Nodes    : ~203 k Bitcoin transactions
  Edges    : ~234 k transaction flows
  Labels   : 1 = illicit, 2 = licit, "unknown" = unlabeled
  Time     : 49 discrete timesteps

Structure discovery
───────────────────
  Each timestep forms ONE giant connected component (all nodes are connected
  within their timestep).  Simple connected-component extraction therefore
  cannot be used — every component exceeds any reasonable max_nodes limit.

Subgraph extraction strategy
─────────────────────────────
  For each LABELED transaction (class 1 = illicit, 2 = licit) we extract its
  2-hop ego network as one subgraph sample, capped at max_nodes.

  This mirrors how the IBM AML dataset creates per-laundering-account
  subgraphs, but is anchored on the labeled transaction itself.

  Graph label
    1 (illicit) — anchor transaction is class-1
    0 (licit)   — anchor transaction is class-2

  Deduplication: ego subgraphs that share the exact same node set as an
  already-extracted graph are skipped, removing near-duplicate samples that
  arise when neighbouring transactions have overlapping 2-hop vicinities.

Node features (5) — same as the IBM AML dataset
  degree      : normalised by max degree in the subgraph
  betweenness : normalised by (n-1)(n-2)/2  (undirected)
  clustering  : local clustering coefficient
  PageRank    : α = 0.85
  assortativity: degree assortativity of the subgraph (graph-level constant,
                 same value for all nodes — identical convention to
                 network_to_dense in diff_util.py)

Usage
─────
  from elliptic_adapter import load_elliptic_pyg_graphs
  graphs = load_elliptic_pyg_graphs()      # list[PyG Data]

  python elliptic_adapter.py               # quick self-test
"""

import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import torch
from torch_geometric.data import Data

_THIS_DIR    = Path(__file__).resolve().parent
ELLIPTIC_DIR = _THIS_DIR.parent.parent / "data" / "elliptic_bitcoin_dataset"


# ─────────────────────────────────────────────────────────────────────────────
# Subgraph extraction
# ─────────────────────────────────────────────────────────────────────────────
def _bfs_ego(G: nx.Graph, anchor: int, max_nodes: int) -> set:
    """
    Collect the 2-hop neighbourhood of `anchor` via BFS, capped at max_nodes.
    Returns a set of node IDs (always includes anchor).
    """
    visited = {anchor}
    frontier = [anchor]
    for _ in range(2):                              # 2 hops
        next_frontier = []
        for u in frontier:
            for v in G.neighbors(u):
                if v not in visited:
                    visited.add(v)
                    next_frontier.append(v)
                    if len(visited) >= max_nodes:
                        return visited
        frontier = next_frontier
    return visited


# ─────────────────────────────────────────────────────────────────────────────
# Feature computation  (identical convention to network_to_dense)
# ─────────────────────────────────────────────────────────────────────────────
def _compute_features(G: nx.Graph):
    """
    Compute 5 structural features for every node in G.

    Returns
    -------
    feat  : float32 ndarray [n, 5]
    nodes : list of node IDs (row order matches feat)
    """
    nodes      = list(G.nodes())
    n          = len(nodes)
    degrees    = dict(G.degree())
    max_deg    = max(max(degrees.values()), 1)
    betw_denom = max(1.0, (n - 1) * (n - 2) / 2.0)

    betweenness = nx.betweenness_centrality(G, normalized=False)
    clustering  = nx.clustering(G)
    pagerank    = nx.pagerank(G, alpha=0.85, max_iter=200)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        try:
            assort = nx.degree_assortativity_coefficient(G)
            if math.isnan(assort):
                assort = 0.0
        except Exception:
            assort = 0.0

    feat = np.zeros((n, 5), dtype=np.float32)
    for i, v in enumerate(nodes):
        feat[i, 0] = degrees[v] / max_deg
        feat[i, 1] = betweenness[v] / betw_denom
        feat[i, 2] = clustering[v]
        feat[i, 3] = pagerank[v]
        feat[i, 4] = assort

    return feat, nodes


# ─────────────────────────────────────────────────────────────────────────────
# PyG conversion
# ─────────────────────────────────────────────────────────────────────────────
def _to_pyg(G_sub: nx.Graph, feat: np.ndarray, label: int, timestep: int = -1) -> Data:
    nodes    = list(G_sub.nodes())
    node_idx = {v: i for i, v in enumerate(nodes)}

    src_list, dst_list = [], []
    for u, v in G_sub.edges():
        src_list += [node_idx[u], node_idx[v]]
        dst_list += [node_idx[v], node_idx[u]]

    if src_list:
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    else:
        idx        = torch.arange(len(nodes))
        edge_index = torch.stack([idx, idx])    # self-loops for isolated nodes

    return Data(
        x          = torch.tensor(feat, dtype=torch.float),
        edge_index = edge_index,
        y          = torch.tensor([label], dtype=torch.long),
        timestep   = torch.tensor(timestep, dtype=torch.long),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main loader
# ─────────────────────────────────────────────────────────────────────────────
def load_elliptic_pyg_graphs(
    data_dir        = ELLIPTIC_DIR,
    min_nodes : int = 3,
    max_nodes : int = 100,
    deduplicate: bool = True,
    verbose   : bool = True,
) -> list:
    """
    Load the Elliptic Bitcoin Dataset and return a list of PyG Data objects.

    For each labeled transaction a 2-hop ego subgraph (capped at max_nodes)
    is extracted from the timestep graph.  The graph label equals the
    anchor transaction's class (1 = illicit, 0 = licit).

    Parameters
    ----------
    data_dir    : directory containing the three Elliptic CSV files
    min_nodes   : minimum ego size to include (default 3)
    max_nodes   : BFS cap on ego size (default 100)
    deduplicate : skip egos whose node-set duplicates one already seen
    verbose     : print progress

    Returns
    -------
    list[torch_geometric.data.Data]
        .x          float [n, 5]  structural features
        .edge_index long  [2, E]
        .y          long  [1]     0 = licit, 1 = illicit
    """
    data_dir = Path(data_dir)
    if verbose:
        print("Loading Elliptic Bitcoin dataset …")

    # ── 1. Node classes ──────────────────────────────────────────────────────
    cls_df    = pd.read_csv(data_dir / "elliptic_txs_classes.csv")
    class_map: dict[int, int] = {}
    for _, row in cls_df.iterrows():
        c = str(row["class"]).strip()
        class_map[int(row["txId"])] = 1 if c == "1" else (0 if c == "2" else -1)

    # ── 2. Node → timestep ───────────────────────────────────────────────────
    feat_df = pd.read_csv(
        data_dir / "elliptic_txs_features.csv",
        header  = None,
        usecols = [0, 1],
        dtype   = {0: int, 1: int},
    )
    feat_df.columns = ["txId", "timestep"]
    node_to_ts: dict[int, int] = dict(
        zip(feat_df["txId"].tolist(), feat_df["timestep"].tolist())
    )

    # ── 3. Edges ─────────────────────────────────────────────────────────────
    edge_df = pd.read_csv(data_dir / "elliptic_txs_edgelist.csv")
    edge_df.columns = ["src", "dst"]
    edge_df["src"] = edge_df["src"].astype(int)
    edge_df["dst"] = edge_df["dst"].astype(int)

    # ── 4. Build one graph per timestep ──────────────────────────────────────
    ts_to_nodes: dict[int, list] = {}
    for txId, ts in node_to_ts.items():
        ts_to_nodes.setdefault(ts, []).append(txId)

    ts_to_edges: dict[int, list] = {}
    edge_df["ts"] = edge_df["src"].map(node_to_ts)
    for ts, grp in edge_df.groupby("ts"):
        ts_to_edges[int(ts)] = list(
            zip(grp["src"].tolist(), grp["dst"].tolist())
        )

    if verbose:
        n_ill = sum(1 for c in class_map.values() if c == 1)
        n_lit = sum(1 for c in class_map.values() if c == 0)
        print(f"  {len(ts_to_nodes)} timesteps | "
              f"{n_ill} illicit labeled | {n_lit} licit labeled")

    # ── 5. Extract ego subgraphs around labeled transactions ─────────────────
    graphs:       list  = []
    seen_nodesets: set  = set()
    n_too_small : int   = 0
    n_too_large : int   = 0   # ego already capped so this counts > cap before
    n_dup       : int   = 0

    for ts in sorted(ts_to_nodes.keys()):
        G = nx.Graph()
        G.add_nodes_from(ts_to_nodes[ts])
        if ts in ts_to_edges:
            G.add_edges_from(ts_to_edges[ts])

        labeled_in_ts = [
            v for v in ts_to_nodes[ts]
            if class_map.get(v, -1) >= 0
        ]

        for anchor in labeled_in_ts:
            label    = class_map[anchor]      # 0 or 1
            ego_set  = _bfs_ego(G, anchor, max_nodes)
            n        = len(ego_set)

            if n < min_nodes:
                n_too_small += 1
                continue

            if deduplicate:
                key = frozenset(ego_set)
                if key in seen_nodesets:
                    n_dup += 1
                    continue
                seen_nodesets.add(key)

            G_sub       = G.subgraph(ego_set).copy()
            feat, _     = _compute_features(G_sub)
            graphs.append(_to_pyg(G_sub, feat, label, timestep=ts))

    if verbose:
        n_ill = sum(d.y.item() == 1 for d in graphs)
        n_lit = sum(d.y.item() == 0 for d in graphs)
        print(f"  Extracted {len(graphs)} ego subgraphs "
              f"({n_ill} illicit, {n_lit} licit)"
              f"  [too small: {n_too_small}, duplicates: {n_dup}]")

    return graphs


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    graphs = load_elliptic_pyg_graphs()
    if graphs:
        g = graphs[0]
        print(f"\nSample graph 0:  nodes={g.x.shape[0]}  "
              f"edges={g.edge_index.shape[1]//2}  label={g.y.item()}")
        print(f"Feature range:   min={g.x.min():.4f}  max={g.x.max():.4f}")
        sizes = [d.x.shape[0] for d in graphs]
        import numpy as np
        print(f"Ego size:        mean={np.mean(sizes):.1f}  "
              f"median={np.median(sizes):.1f}  max={max(sizes)}")
