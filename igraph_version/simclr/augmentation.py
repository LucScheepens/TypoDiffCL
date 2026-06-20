import random
from collections import deque
import igraph as ig
import numpy as np
import time
import copy


def precompute_augmentation_cache(networks):
    """
    Precompute expensive graph features for all original networks and store
    them in each network dict so augmentation calls can skip recomputation.

    Sets three keys on every network dict:
      _bridge_name_pairs : set of (min_name, max_name) tuples that are bridges
      _eb_by_name_pair   : {(min_name, max_name) -> normalised edge betweenness}
      _ap_names          : set of node names that are articulation points

    These are automatically inherited by augmented copies via dict unpacking
    ({**network, ...}), so crop / edge-delete / node-delete functions can use
    them on subgraphs without recomputation.
    """
    for net in networks:
        g = net.get("graph")
        if g is None or g.vcount() < 2 or g.ecount() == 0:
            net["_bridge_name_pairs"] = set()
            net["_eb_by_name_pair"]   = {}
            net["_ap_names"]          = set()
            continue

        has_names = "name" in g.vs.attributes()
        g_undir   = g.as_undirected(combine_edges="first")

        # ── bridges ──────────────────────────────────────────────────────────
        bridge_name_pairs: set = set()
        for eid in g_undir.bridges():
            e  = g_undir.es[eid]
            n1 = g_undir.vs[e.source]["name"] if has_names else e.source
            n2 = g_undir.vs[e.target]["name"] if has_names else e.target
            bridge_name_pairs.add((min(n1, n2), max(n1, n2)))
        net["_bridge_name_pairs"] = bridge_name_pairs

        # ── edge betweenness ──────────────────────────────────────────────────
        eb     = g_undir.edge_betweenness(directed=False)
        max_eb = max(eb) if eb and max(eb) > 0 else 1.0
        eb_by_name_pair: dict = {}
        for eid, e in enumerate(g_undir.es):
            n1   = g_undir.vs[e.source]["name"] if has_names else e.source
            n2   = g_undir.vs[e.target]["name"] if has_names else e.target
            pair = (min(n1, n2), max(n1, n2))
            eb_by_name_pair[pair] = eb[eid] / max_eb
        net["_eb_by_name_pair"] = eb_by_name_pair

        # ── articulation points ───────────────────────────────────────────────
        net["_ap_names"] = {
            (g_undir.vs[vid]["name"] if has_names else vid)
            for vid in g_undir.articulation_points()
        }


def build_igraph_from_transactions(tx_df):
    """
    Build a directed igraph graph from transactions dataframe.
    """
    edges = tx_df[["From_Account_int", "To_Account_int"]]
    edges = edges[edges["From_Account_int"] != edges["To_Account_int"]]
    g = ig.Graph.DataFrame(edges, directed=True, use_vids=False)
    return g


def _get_or_build_graph(network):
    """
    Return a copy of the network's igraph graph.
    If the network dict has no 'graph' key (as produced by util.py),
    build a directed graph from the 'transactions' DataFrame instead.
    Node 'name' attributes are integer account IDs.
    """
    if "graph" in network:
        return network["graph"].copy()

    txs = network["transactions"]
    nodes = network["nodes"]

    if len(txs) > 0:
        edges = txs[["From_Account_int", "To_Account_int"]]
        edges = edges[edges["From_Account_int"] != edges["To_Account_int"]]
        g = ig.Graph.DataFrame(edges, directed=True, use_vids=False)
        # Add isolated nodes that appear in the node set but have no transactions
        existing = {v["name"] for v in g.vs}
        for n in nodes:
            if n not in existing:
                g.add_vertex(name=n)
    else:
        g = ig.Graph(directed=True)
        for n in sorted(nodes):
            g.add_vertex(name=n)

    return g


def crop_network(network, crop_ratio=0.8, random_seed=None):
    if random_seed is not None:
        random.seed(random_seed)

    g = network["graph"]
    graph_nodes = set(v["name"] for v in g.vs)
    nodes = list(set(network["nodes"]) & graph_nodes)

    if len(nodes) < 2:
        return network

    target_size = max(2, int(len(nodes) * crop_ratio))
    start_node = random.choice(list(network["laundering_nodes"])) if network["laundering_nodes"] else random.choice(nodes)

    try:
        start_vid = g.vs.find(name=start_node).index
    except ValueError:
        return network

    order = g.bfs(start_vid)[0]
    bfs_nodes = [g.vs[v]["name"] for v in order if v != -1]
    cropped_nodes = set(bfs_nodes[:target_size])

    # --- convert names to indices ---
    cropped_vids = [v.index for v in g.vs if v["name"] in cropped_nodes]
    g_sub = g.subgraph(cropped_vids)

    return {
        **network,
        "start_node": start_node,
        "nodes": cropped_nodes,
        "laundering_nodes": network["laundering_nodes"] & cropped_nodes,
        "collapsed_nodes": network["collapsed_nodes"] & cropped_nodes,
        "node_depths": {n: d for n, d in network["node_depths"].items() if n in cropped_nodes},
        "graph": g_sub
    }



def delete_random_edges(network, delete_frac=0.15, random_seed=None):
    """
    Delete random edges without disconnecting the network.
    Bridge detection runs on the undirected view so it works correctly
    for directed graphs produced by util.py.
    """
    if random_seed is not None:
        random.seed(random_seed)

    g = _get_or_build_graph(network)
    if g.ecount() < 2:
        return {**network, "graph": g}

    has_names     = "name" in g.vs.attributes()
    _bridge_cache = network.get("_bridge_name_pairs")

    if _bridge_cache is not None:
        # Fast path: use precomputed bridge pairs keyed by node names.
        # Works for subgraphs (after crop) because node names are preserved.
        non_bridges = []
        for e in g.es:
            n1 = g.vs[e.source]["name"] if has_names else e.source
            n2 = g.vs[e.target]["name"] if has_names else e.target
            if (min(n1, n2), max(n1, n2)) not in _bridge_cache:
                non_bridges.append(e.index)
    else:
        # Slow path: compute bridges on-the-fly
        g_undir     = g.as_undirected(combine_edges="first")
        bridge_pairs = set()
        for eid in g_undir.bridges():
            e = g_undir.es[eid]
            bridge_pairs.add((min(e.source, e.target), max(e.source, e.target)))
        non_bridges = [
            e.index for e in g.es
            if (min(e.source, e.target), max(e.source, e.target)) not in bridge_pairs
        ]

    if not non_bridges:
        return {**network, "graph": g}

    target = max(1, int(len(non_bridges) * delete_frac))
    random.shuffle(non_bridges)
    g.delete_edges(non_bridges[:target])

    return {**network, "graph": g}


def delete_nodes(network, delete_frac=0.15, random_seed=None):
    """
    Delete random nodes from the network without disconnecting it.
    Articulation points (whose removal would split the graph) and
    laundering nodes are never deleted.
    """
    if random_seed is not None:
        random.seed(random_seed)

    g = _get_or_build_graph(network)
    if g.vcount() < 3:
        return {**network, "graph": g}

    has_names        = "name" in g.vs.attributes()
    laundering_names = network["laundering_nodes"]
    _ap_cache        = network.get("_ap_names")

    if _ap_cache is not None:
        # Fast path: use precomputed AP names (conservative approximation for subgraphs)
        deletable_vids = [
            v.index for v in g.vs
            if (g.vs[v.index]["name"] if has_names else v.index) not in _ap_cache
            and (g.vs[v.index]["name"] if has_names else v.index) not in laundering_names
        ]
    else:
        # Slow path: find articulation points on-the-fly
        g_undir        = g.as_undirected(combine_edges="first")
        art_point_vids = set(g_undir.articulation_points())
        deletable_vids = [
            v.index for v in g.vs
            if v.index not in art_point_vids
            and (g.vs[v.index]["name"] if has_names else v.index) not in laundering_names
        ]

    if not deletable_vids:
        return {**network, "graph": g}

    target    = max(1, int(len(deletable_vids) * delete_frac))
    random.shuffle(deletable_vids)
    to_delete = deletable_vids[:target]

    deleted_names = {(g.vs[vid]["name"] if has_names else vid) for vid in to_delete}
    g.delete_vertices(to_delete)
    remaining = network["nodes"] - deleted_names

    return {
        **network,
        "nodes": remaining,
        "laundering_nodes": network["laundering_nodes"] - deleted_names,
        "collapsed_nodes": network["collapsed_nodes"] - deleted_names,
        "node_depths": {n: d for n, d in network["node_depths"].items() if n not in deleted_names},
        "graph": g,
    }


def add_nodes(network, num_new_nodes=5, random_seed=None):
    """
    Insert new nodes within the network by subdividing random edges.
    Each selected edge (u → v) is replaced by (u → w, w → v) where w
    is a new synthetic node. This places nodes *inside* the existing
    topology rather than appending external ones.
    """
    if random_seed is not None:
        random.seed(random_seed)

    g = _get_or_build_graph(network)
    if g.ecount() == 0:
        return {**network, "graph": g}

    # Pick a synthetic ID range that cannot clash with real account IDs
    existing_names = {v["name"] for v in g.vs}
    new_id = max(existing_names) + 1

    new_nodes = set()
    node_depths = dict(network["node_depths"])

    # Build name→vid dict once — O(V) instead of O(V) per find() call
    name_to_vid = {v["name"]: v.index for v in g.vs}

    # Snapshot (src_name, tgt_name) pairs before any structural changes
    all_eids = list(range(g.ecount()))
    random.shuffle(all_eids)
    edge_pairs = [
        (g.vs[g.es[eid].source]["name"], g.vs[g.es[eid].target]["name"])
        for eid in all_eids[:min(num_new_nodes, len(all_eids))]
    ]

    for src_name, tgt_name in edge_pairs:
        src_vid = name_to_vid.get(src_name)
        tgt_vid = name_to_vid.get(tgt_name)
        if src_vid is None or tgt_vid is None:
            continue

        try:
            eid = g.get_eid(src_vid, tgt_vid)
        except ig.InternalError:
            continue

        # add_vertex always appends → new index is vcount()-1, no find() needed
        g.add_vertex(name=new_id)
        new_vid = g.vcount() - 1
        name_to_vid[new_id] = new_vid

        g.delete_edges(eid)
        g.add_edges([(src_vid, new_vid), (new_vid, tgt_vid)])

        new_nodes.add(new_id)
        d_src = node_depths.get(src_name)
        d_tgt = node_depths.get(tgt_name)
        node_depths[new_id] = (d_src + d_tgt + 1) // 2 if (d_src is not None and d_tgt is not None) else None

        new_id += 1

    return {
        **network,
        "nodes": network["nodes"] | new_nodes,
        "node_depths": node_depths,
        "graph": g,
    }


def augment_network_view_fast(
    network,
    p_crop=0.6,
    p_edge_drop=0.2,
    p_node_delete=0.2,
    p_node_add=0.2,
    crop_ratio_range=(0.6, 0.9),
    edge_drop_range=(0.05, 0.2),
    node_delete_frac=0.15,
    num_new_nodes=5,
    random_seed=None,
):
    if random_seed is not None:
        random.seed(random_seed)

    aug_net = {**network}  # shallow copy — augmentation fns always return new dicts/graphs

    if random.random() < p_crop:
        ratio = random.uniform(*crop_ratio_range)
        aug_net = crop_network(aug_net, crop_ratio=ratio)

    if random.random() < p_edge_drop:
        frac = random.uniform(*edge_drop_range)
        aug_net = delete_random_edges(aug_net, delete_frac=frac)

    if random.random() < p_node_delete:
        aug_net = delete_nodes(aug_net, delete_frac=node_delete_frac)

    if random.random() < p_node_add:
        aug_net = add_nodes(aug_net, num_new_nodes=num_new_nodes)

    return aug_net


# ─────────────────────────────────────────────────────────────────────────────
# Motif-preserving edge deletion
# ─────────────────────────────────────────────────────────────────────────────

def delete_edges_motif_preserving(network, delete_frac=0.15, random_seed=None):
    """
    Delete edges biased toward structurally unimportant ones.

    Uses edge betweenness centrality as an importance proxy.  High-betweenness
    edges sit on many shortest paths; they are given a low deletion probability.
    Bridges are never deleted.

    Uses precomputed caches (_bridge_name_pairs, _eb_by_name_pair) when
    available (set by precompute_augmentation_cache), avoiding expensive
    O(VE) betweenness recomputation on every augmented view.
    """
    if random_seed is not None:
        random.seed(random_seed)

    g = _get_or_build_graph(network)
    if g.ecount() < 2:
        return {**network, "graph": g}

    has_names     = "name" in g.vs.attributes()
    _bridge_cache = network.get("_bridge_name_pairs")
    _eb_cache     = network.get("_eb_by_name_pair")
    g_undir       = None  # built lazily — only if caches are missing

    # ── identify non-bridge edges ─────────────────────────────────────────────
    if _bridge_cache is not None:
        non_bridges = []
        for e in g.es:
            n1 = g.vs[e.source]["name"] if has_names else e.source
            n2 = g.vs[e.target]["name"] if has_names else e.target
            if (min(n1, n2), max(n1, n2)) not in _bridge_cache:
                non_bridges.append(e.index)
    else:
        g_undir      = g.as_undirected(combine_edges="first")
        bridge_pairs = set()
        for eid in g_undir.bridges():
            e = g_undir.es[eid]
            bridge_pairs.add((min(e.source, e.target), max(e.source, e.target)))
        non_bridges = [
            e.index for e in g.es
            if (min(e.source, e.target), max(e.source, e.target)) not in bridge_pairs
        ]

    if not non_bridges:
        return {**network, "graph": g}

    # ── build importance weights ──────────────────────────────────────────────
    if _eb_cache is not None:
        # Fast path: look up betweenness by node name — works for subgraphs
        weights = []
        for eid in non_bridges:
            e  = g.es[eid]
            n1 = g.vs[e.source]["name"] if has_names else e.source
            n2 = g.vs[e.target]["name"] if has_names else e.target
            importance = _eb_cache.get((min(n1, n2), max(n1, n2)), 0.5)
            weights.append(1.0 - importance + 0.01)
    else:
        # Slow path: compute edge betweenness on-the-fly
        if g_undir is None:
            g_undir = g.as_undirected(combine_edges="first")
        eb     = g_undir.edge_betweenness(directed=False)
        max_eb = max(eb) if max(eb) > 0 else 1.0
        eb_by_pair = {
            (min(e.source, e.target), max(e.source, e.target)): eb[eid] / max_eb
            for eid, e in enumerate(g_undir.es)
        }
        weights = []
        for eid in non_bridges:
            e    = g.es[eid]
            pair = (min(e.source, e.target), max(e.source, e.target))
            weights.append(1.0 - eb_by_pair.get(pair, 0.5) + 0.01)

    probs  = np.array(weights) / sum(weights)
    target = max(1, int(len(non_bridges) * delete_frac))
    chosen = list(np.random.choice(non_bridges, size=min(target, len(non_bridges)),
                                   replace=False, p=probs))
    g.delete_edges(chosen)
    return {**network, "graph": g}


# ─────────────────────────────────────────────────────────────────────────────
# Saliency-guided node deletion
# ─────────────────────────────────────────────────────────────────────────────

def delete_nodes_saliency_guided(network, delete_frac=0.15,
                                  node_importance=None, random_seed=None):
    """
    Delete nodes with a bias toward low-importance ones.

    node_importance: dict {account_name (int) -> importance_score in [0,1]}.
      High scores → keep.  Falls back to uniform random deletion when None.
    Laundering nodes and articulation points are never deleted.

    Uses precomputed _ap_names cache when available to avoid O(V+E)
    articulation-point recomputation on every augmented view.
    """
    if random_seed is not None:
        random.seed(random_seed)

    g = _get_or_build_graph(network)
    if g.vcount() < 3:
        return {**network, "graph": g}

    has_names  = "name" in g.vs.attributes()
    laundering = network["laundering_nodes"]
    _ap_cache  = network.get("_ap_names")

    if _ap_cache is not None:
        # Fast path: use precomputed AP names (conservative approximation)
        deletable_vids = [
            v.index for v in g.vs
            if (g.vs[v.index]["name"] if has_names else v.index) not in _ap_cache
            and (g.vs[v.index]["name"] if has_names else v.index) not in laundering
        ]
    else:
        # Slow path: compute articulation points on-the-fly
        g_undir   = g.as_undirected(combine_edges="first")
        art_vids  = set(g_undir.articulation_points())
        deletable_vids = [
            v.index for v in g.vs
            if v.index not in art_vids
            and (g.vs[v.index]["name"] if has_names else v.index) not in laundering
        ]

    if not deletable_vids:
        return {**network, "graph": g}

    target = max(1, int(len(deletable_vids) * delete_frac))

    if node_importance is not None:
        weights = []
        for vid in deletable_vids:
            name = g.vs[vid]["name"] if has_names else vid
            importance = node_importance.get(int(name), 0.5)
            weights.append(1.0 - importance + 0.01)
        probs = np.array(weights) / sum(weights)
        chosen_vids = list(np.random.choice(deletable_vids,
                                             size=min(target, len(deletable_vids)),
                                             replace=False, p=probs))
    else:
        random.shuffle(deletable_vids)
        chosen_vids = deletable_vids[:target]

    deleted_names = {(g.vs[vid]["name"] if has_names else vid) for vid in chosen_vids}
    g.delete_vertices(chosen_vids)
    remaining = network["nodes"] - deleted_names

    return {
        **network,
        "nodes": remaining,
        "laundering_nodes": network["laundering_nodes"] - deleted_names,
        "collapsed_nodes":  network["collapsed_nodes"]  - deleted_names,
        "node_depths": {n: d for n, d in network["node_depths"].items()
                        if n not in deleted_names},
        "graph": g,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Smart augmentation: curriculum + motif-preserving + saliency
# ─────────────────────────────────────────────────────────────────────────────

def augment_network_view_smart(
    network,
    aug_strength=1.0,
    node_importance=None,
    p_crop=0.6,
    p_edge_drop=0.3,
    p_node_delete=0.3,
    p_node_add=0.2,
    crop_ratio_range=(0.6, 0.9),
    edge_drop_range=(0.05, 0.25),
    node_delete_frac=0.20,
    num_new_nodes=5,
    use_motif_preserving=True,
    random_seed=None,
):
    """
    Augmentation combining three task-aware ideas:

    1. Curriculum strength (aug_strength ∈ [0,1]): at 0, transformations are
       very mild (the network is barely changed); at 1, full-strength augmentation
       is applied.  This allows the encoder to learn stable representations before
       harder views are introduced.

    2. Motif-preserving edge dropping: instead of uniform random edge deletion,
       edges are sampled with probability inversely proportional to their edge
       betweenness centrality, so structurally critical edges (bridges between
       communities, flow bottlenecks) are less likely to be removed.

    3. Saliency-guided node deletion: nodes with high probe-gradient importance
       scores are protected from deletion, preserving the most discriminative
       structural elements for the downstream classification task.
    """
    if random_seed is not None:
        random.seed(random_seed)

    # Scale from mild (0.3) at the start to full (1.0) at the end
    base_scale = 0.3 + 0.7 * aug_strength

    aug_net = {**network}

    if random.random() < p_crop * base_scale:
        lo = crop_ratio_range[0] + (1.0 - aug_strength) * (1.0 - crop_ratio_range[0])
        hi = crop_ratio_range[1] + (1.0 - aug_strength) * (1.0 - crop_ratio_range[1])
        ratio = random.uniform(lo, hi)
        aug_net = crop_network(aug_net, crop_ratio=ratio)

    if random.random() < p_edge_drop * base_scale:
        lo_d  = edge_drop_range[0] * aug_strength
        hi_d  = edge_drop_range[1] * aug_strength
        frac  = random.uniform(max(0.01, lo_d), max(0.02, hi_d))
        if use_motif_preserving:
            aug_net = delete_edges_motif_preserving(aug_net, delete_frac=frac)
        else:
            aug_net = delete_random_edges(aug_net, delete_frac=frac)

    if random.random() < p_node_delete * base_scale:
        frac = node_delete_frac * max(0.1, aug_strength)
        aug_net = delete_nodes_saliency_guided(aug_net, delete_frac=frac,
                                               node_importance=node_importance)

    if random.random() < p_node_add * base_scale:
        n_new = max(1, int(num_new_nodes * aug_strength))
        aug_net = add_nodes(aug_net, num_new_nodes=n_new)

    return aug_net