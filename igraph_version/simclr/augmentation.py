import random
from collections import deque
import igraph as ig
import time
import copy


def build_igraph_from_transactions(tx_df):
    """
    Build an undirected igraph graph from transactions dataframe.
    """
    g = ig.Graph.DataFrame(
        tx_df[["From_Account_int", "To_Account_int"]],
        directed=False,
        use_vids=False
    )
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
        g = ig.Graph.DataFrame(
            txs[["From_Account_int", "To_Account_int"]],
            directed=True,
            use_vids=False,
        )
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
    """Delete a random fraction of edges — no bridge detection overhead."""
    if random_seed is not None:
        random.seed(random_seed)

    g = _get_or_build_graph(network)
    if g.ecount() < 2:
        return {**network, "graph": g}

    all_eids = list(range(g.ecount()))
    target = max(1, int(len(all_eids) * delete_frac))
    random.shuffle(all_eids)
    g.delete_edges(all_eids[:target])

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

    # Find articulation points on the undirected version
    g_undir = g.as_undirected(combine_edges="first")
    art_point_vids = set(g_undir.articulation_points())

    laundering_names = network["laundering_nodes"]

    # Eligible: not an articulation point and not a laundering node
    deletable_vids = [
        v.index for v in g.vs
        if v.index not in art_point_vids
        and v["name"] not in laundering_names
    ]

    if not deletable_vids:
        return {**network, "graph": g}

    target = max(1, int(len(deletable_vids) * delete_frac))
    random.shuffle(deletable_vids)
    to_delete = deletable_vids[:target]

    deleted_names = {g.vs[vid]["name"] for vid in to_delete}
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
    crop_ratio_range=(0.6, 0.9),
    edge_drop_range=(0.05, 0.2),
    random_seed=None,
):
    if random_seed is not None:
        random.seed(random_seed)

    aug_net = {**network}

    if random.random() < p_crop:
        ratio = random.uniform(*crop_ratio_range)
        aug_net = crop_network(aug_net, crop_ratio=ratio)

    if random.random() < p_edge_drop:
        frac = random.uniform(*edge_drop_range)
        aug_net = delete_random_edges(aug_net, delete_frac=frac)

    return aug_net