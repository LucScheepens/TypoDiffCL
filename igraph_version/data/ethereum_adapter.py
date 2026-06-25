"""
ethereum_adapter.py
─────────────────────────────────────────────────────────────────────────────
Convert the Ethereum Phishing Transaction Network (EPTransNet) into PyG Data
objects compatible with the existing AML graph generation pipeline.

Dataset description
───────────────────
  Source   : http://xblock.pro/ethereum/
  File     : MulDiGraph.pkl  (networkx.classes.multidigraph.MultiDiGraph)
  Nodes    : 2,973,489 Ethereum accounts
  Edges    : 13,551,303 transactions (directed, multi-edge allowed)
  Labels   : G.nodes[n]['isp'] — 1 = phishing, 0 = clean
  Labeled  : ~1,165 nodes (all others are unlabeled background context)

Subgraph extraction
───────────────────
  For each labeled node, extract a `depth`-hop ego subgraph via BFS (capped
  at max_nodes).  BFS explores both successors and predecessors so that
  transaction flows in both directions are captured.  The subgraph label
  equals the anchor node's 'isp' value (1 = phishing, 0 = clean).

  The full 3M-node graph is never converted to an undirected copy in memory.
  Instead, BFS operates directly on the MultiDiGraph and only the small ego
  subgraphs (≤ max_nodes) are converted to simple graphs for feature
  computation.

Node features (5) — same structural features as the Elliptic adapter
  degree        : normalised by max degree in the ego subgraph (undirected)
  betweenness   : normalised by (n-1)(n-2)/2
  clustering    : local clustering coefficient (undirected simple graph)
  PageRank      : α = 0.85 (undirected simple graph)
  assortativity : degree assortativity (graph-level constant)

  These match the Elliptic adapter's 5-dimensional feature space so that
  Ethereum graphs are compatible with diffusion/SimCLR models trained on
  either dataset (NODE_DIM = 6 including the label column).

Caching
───────
  On first load the extracted PyG graphs are cached to
  igraph_version/data/ethereum_graphs_cache.pt so that subsequent runs skip
  the expensive BFS + feature computation step.

Usage
─────
  from data.ethereum_adapter import load_ethereum_pyg_graphs
  graphs = load_ethereum_pyg_graphs()

  python ethereum_adapter.py          # quick self-test
"""

import math
import pickle
import random
import warnings
from pathlib import Path

import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data
from tqdm.auto import tqdm

_THIS_DIR    = Path(__file__).resolve().parent
ETHEREUM_DIR = _THIS_DIR.parent.parent / "data" / "Ethereum Phishing Transaction Network"
PKL_PATH     = ETHEREUM_DIR / "MulDiGraph.pkl"
CACHE_PATH   = _THIS_DIR / "ethereum_graphs_cache.pt"


# ─────────────────────────────────────────────────────────────────────────────
# BFS ego extraction (operates directly on MultiDiGraph — no full conversion)
# ─────────────────────────────────────────────────────────────────────────────

def _bfs_ego(G, anchor, max_nodes: int, depth: int) -> set:
    """
    BFS on a directed multigraph exploring both directions (sent + received).
    Always includes the anchor node.  Stops when max_nodes is reached.
    """
    visited  = {anchor}
    frontier = [anchor]
    for _ in range(depth):
        next_f = []
        for u in frontier:
            for v in G.successors(u):
                if v not in visited:
                    visited.add(v)
                    next_f.append(v)
                    if len(visited) >= max_nodes:
                        return visited
            for v in G.predecessors(u):
                if v not in visited:
                    visited.add(v)
                    next_f.append(v)
                    if len(visited) >= max_nodes:
                        return visited
        frontier = next_f
    return visited


# ─────────────────────────────────────────────────────────────────────────────
# Simple-graph construction  (avoids parallel-edge blowup)
# ─────────────────────────────────────────────────────────────────────────────

def _build_simple_subgraphs(G, ego_set: set):
    """
    Build a simple undirected Graph and a simple DiGraph from a MultiDiGraph ego.

    Iterates G.successors() which yields UNIQUE neighbour nodes — not one entry
    per parallel edge.  nx.Graph(G.subgraph(ego)) calls edges() internally and
    yields one tuple per parallel edge, which is catastrophically slow for
    phishing accounts that send thousands of repeated transactions to the same
    counterparty.
    """
    ego_s   = set(ego_set)
    G_undir = nx.Graph()
    G_undir.add_nodes_from(ego_s)
    G_dir   = nx.DiGraph()
    G_dir.add_nodes_from(ego_s)
    for u in ego_s:
        for v in G.successors(u):      # unique successors — no parallel duplication
            if v in ego_s and v != u:
                G_undir.add_edge(u, v)
                G_dir.add_edge(u, v)
    return G_undir, G_dir


# ─────────────────────────────────────────────────────────────────────────────
# Feature computation  (same 5 structural features as the Elliptic adapter)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_features(G_simple: nx.Graph):
    """
    Compute 5 structural features on a simple undirected subgraph.

    Returns
    -------
    feat  : float32 ndarray [n, 5]
    nodes : list of node IDs (row order matches feat)
    """
    nodes      = list(G_simple.nodes())
    n          = len(nodes)
    degrees    = dict(G_simple.degree())
    max_deg    = max(max(degrees.values()), 1)
    betw_denom = max(1.0, (n - 1) * (n - 2) / 2.0)

    # Approximate betweenness (k pivot nodes) — exact when k >= n, otherwise
    # samples k source nodes.  Indistinguishable in quality for structural
    # embeddings while being ~n/k× faster on large subgraphs.
    k_betw      = min(n, 15)
    betweenness = nx.betweenness_centrality(G_simple, normalized=False, k=k_betw)
    clustering  = nx.clustering(G_simple)
    pagerank    = nx.pagerank(G_simple, alpha=0.85, max_iter=200)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        try:
            assort = nx.degree_assortativity_coefficient(G_simple)
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

def _to_pyg(G_dir: nx.DiGraph, feat: np.ndarray, label: int,
            anchor_local_idx: int = 0) -> Data:
    """Convert a directed simple subgraph + feature matrix to a PyG Data object."""
    nodes    = list(G_dir.nodes())
    node_idx = {v: i for i, v in enumerate(nodes)}

    src_list, dst_list = [], []
    for u, v in G_dir.edges():
        src_list.append(node_idx[u])
        dst_list.append(node_idx[v])

    if src_list:
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    else:
        idx        = torch.arange(len(nodes))
        edge_index = torch.stack([idx, idx])   # self-loops for isolated nodes

    return Data(
        x                = torch.tensor(feat, dtype=torch.float),
        edge_index       = edge_index,
        y                = torch.tensor([label], dtype=torch.long),
        anchor_local_idx = torch.tensor([anchor_local_idx], dtype=torch.long),
        timestep         = torch.tensor([-1], dtype=torch.long),
        timestamp_val    = -1.0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main loader
# ─────────────────────────────────────────────────────────────────────────────

def load_ethereum_pyg_graphs(
    pkl_path      = PKL_PATH,
    cache_path    = CACHE_PATH,
    min_nodes   : int  = 3,
    max_nodes   : int  = 100,
    max_subgraphs: int = 20_000,
    depth       : int  = 2,
    deduplicate : bool = True,
    verbose     : bool = True,
    use_cache   : bool = True,
) -> list:
    """
    Load the Ethereum Phishing Transaction Network and return PyG Data objects.

    For each labeled node a `depth`-hop ego subgraph (capped at max_nodes) is
    extracted.  The subgraph label = anchor node's 'isp' value.

    Parameters
    ----------
    pkl_path      : path to MulDiGraph.pkl
    cache_path    : path for caching the extracted PyG graphs (avoids re-extraction)
    min_nodes     : minimum ego size to include (default 3)
    max_nodes     : BFS cap on ego size (default 100)
    max_subgraphs : hard cap on total extracted subgraphs (default 20 000)
    depth         : BFS hop depth (default 2 — the Ethereum graph is denser than
                    Elliptic so depth 2 already yields rich context)
    deduplicate   : skip egos whose node-set duplicates one already seen
    verbose       : print progress
    use_cache     : load from / save to cache_path (default True)

    Returns
    -------
    list[torch_geometric.data.Data]
        .x                float [n, 5]  structural features
        .edge_index       long  [2, E]  directed edges
        .y                long  [1]     0 = clean, 1 = phishing
        .anchor_local_idx long  [1]     row index of anchor in .x
    """
    cache_path = Path(cache_path)
    pkl_path   = Path(pkl_path)

    # ── Try cache ─────────────────────────────────────────────────────────────
    if use_cache and cache_path.exists():
        if verbose:
            print(f"Loading Ethereum graphs from cache: {cache_path.name} …")
        graphs = torch.load(str(cache_path), weights_only=False)
        n_phish = sum(d.y.item() == 1 for d in graphs)
        n_clean = len(graphs) - n_phish
        if verbose:
            print(f"  {len(graphs)} graphs loaded ({n_phish} phishing, {n_clean} clean)")
        return graphs

    # ── Load pickle ───────────────────────────────────────────────────────────
    if not pkl_path.exists():
        raise FileNotFoundError(
            f"Ethereum pickle not found: {pkl_path}\n"
            f"Expected: {ETHEREUM_DIR / 'MulDiGraph.pkl'}\n"
            "Download from http://xblock.pro/ethereum/ and place in the data directory."
        )

    if verbose:
        print(f"Loading Ethereum Phishing Transaction Network …")
        print(f"  File: {pkl_path.name}  (~1.2 GB, may take 1-2 minutes)")

    with open(pkl_path, "rb") as f:
        G = pickle.load(f)

    phishing_nodes = [n for n in G.nodes() if G.nodes[n].get("isp") == 1]
    clean_nodes    = [n for n in G.nodes() if G.nodes[n].get("isp") == 0]

    if verbose:
        print(f"  Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
        print(f"  Labeled: {len(phishing_nodes) + len(clean_nodes)} nodes "
              f"({len(phishing_nodes)} phishing, {len(clean_nodes)} clean)")

    # Stratified random sample so we never iterate more than max_subgraphs anchors.
    # Each class gets at most half the budget; if one class is smaller the other
    # can take the remainder.
    half          = max_subgraphs // 2
    n_phish_cap   = min(len(phishing_nodes), half)
    n_clean_cap   = min(len(clean_nodes),    max_subgraphs - n_phish_cap)
    # If phishing nodes are fewer than half, give the remainder to clean
    n_clean_cap   = min(len(clean_nodes),    max_subgraphs - n_phish_cap)

    sampled_phish = random.sample(phishing_nodes, n_phish_cap)
    sampled_clean = random.sample(clean_nodes,    n_clean_cap)
    anchors       = sampled_phish + sampled_clean
    random.shuffle(anchors)

    if verbose:
        print(f"  Sampled {len(sampled_phish)} phishing + {len(sampled_clean)} clean anchors "
              f"(max_subgraphs={max_subgraphs:,})")
        print(f"  Extracting ego subgraphs "
              f"(depth={depth}, max_nodes={max_nodes}) …")

    # ── Ego extraction ────────────────────────────────────────────────────────
    graphs        = []
    seen_nodesets = set()
    n_too_small   = 0
    n_dup         = 0

    for anchor in tqdm(anchors, desc="Ethereum ego extraction", disable=not verbose):
        label   = int(G.nodes[anchor]["isp"])
        ego_set = _bfs_ego(G, anchor, max_nodes, depth)

        if len(ego_set) < min_nodes:
            n_too_small += 1
            continue

        if deduplicate:
            key = frozenset(ego_set)
            if key in seen_nodesets:
                n_dup += 1
                continue
            seen_nodesets.add(key)

        # Build simple undirected + directed subgraphs without touching parallel
        # edges — nx.Graph(G.subgraph()) iterates ALL parallel edges which is
        # orders of magnitude slower for dense phishing-node neighbourhoods.
        G_sub_undir, G_sub_dir = _build_simple_subgraphs(G, ego_set)

        feat, feat_nodes = _compute_features(G_sub_undir)
        anchor_local = feat_nodes.index(anchor) if anchor in feat_nodes else 0

        graphs.append(_to_pyg(G_sub_dir, feat, label,
                              anchor_local_idx=anchor_local))

    if verbose:
        n_ph = sum(d.y.item() == 1 for d in graphs)
        n_cl = len(graphs) - n_ph
        print(f"  Extracted {len(graphs)} ego subgraphs "
              f"({n_ph} phishing, {n_cl} clean)"
              f"  [too small: {n_too_small}, duplicates: {n_dup}]")

    # ── Save cache ────────────────────────────────────────────────────────────
    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(graphs, str(cache_path))
        if verbose:
            print(f"  Cache saved → {cache_path.name}")

    return graphs


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    graphs = load_ethereum_pyg_graphs()
    if graphs:
        g = graphs[0]
        print(f"\nSample graph 0:  nodes={g.x.shape[0]}  "
              f"edges={g.edge_index.shape[1]}  label={g.y.item()}")
        print(f"Feature range:   min={g.x.min():.4f}  max={g.x.max():.4f}")
        sizes = [d.x.shape[0] for d in graphs]
        import numpy as _np
        print(f"Ego size:        mean={_np.mean(sizes):.1f}  "
              f"median={_np.median(sizes):.1f}  max={max(sizes)}")
        n_ph = sum(d.y.item() == 1 for d in graphs)
        print(f"Label balance:   {n_ph} phishing / {len(graphs) - n_ph} clean")
