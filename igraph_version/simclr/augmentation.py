import random
from collections import deque
import igraph as ig 
import pandas as pd
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
    """Delete edges in the graph but skip updating transactions."""
    if random_seed is not None:
        random.seed(random_seed)

    g = network["graph"].copy()
    if g.ecount() < 2:
        return network

    bridges = set(g.bridges())
    non_bridges = [i for i in range(g.ecount()) if i not in bridges]
    if not non_bridges:
        return network

    target_deletions = int(len(non_bridges) * delete_frac)
    random.shuffle(non_bridges)
    delete_eids = non_bridges[:target_deletions]

    g.delete_edges(delete_eids)

    return {
        **network,
        "graph": g
    }


def add_nodes(
    network,
    full_graph,
    max_new_nodes=10,
    max_depth=2,
    collapse_threshold=10,
    random_seed=None
):
    if random_seed is not None:
        random.seed(random_seed)

    g = network["graph"].copy()
    current_nodes = set(network["nodes"])
    collapsed_nodes = set(network["collapsed_nodes"])
    new_nodes = set()

    from collections import deque
    boundary = list(current_nodes)
    random.shuffle(boundary)
    queue = deque((n, 0) for n in boundary)

    while queue and len(new_nodes) < max_new_nodes:
        node, depth = queue.popleft()
        if depth >= max_depth:
            continue
        try:
            vid = full_graph.vs.find(name=node).index
        except ValueError:
            continue

        neighbors = {full_graph.vs[v]["name"] for v in full_graph.neighbors(vid)}
        if len(neighbors) > collapse_threshold:
            collapsed_nodes.add(node)
            continue

        for nbr in neighbors:
            if nbr not in current_nodes and nbr not in new_nodes:
                new_nodes.add(nbr)
                queue.append((nbr, depth + 1))
                if len(new_nodes) >= max_new_nodes:
                    break

    for n in new_nodes:
        if n not in g.vs["name"]:
            g.add_vertex(name=n)

    # Add edges
    name_to_vid = {v["name"]: v.index for v in g.vs}
    edges_to_add = []
    for n in new_nodes:
        try:
            vid = full_graph.vs.find(name=n).index
        except ValueError:
            continue
        for nbr_vid in full_graph.neighbors(vid):
            nbr = full_graph.vs[nbr_vid]["name"]
            if nbr in current_nodes or nbr in new_nodes:
                if n in name_to_vid and nbr in name_to_vid:
                    edges_to_add.append((name_to_vid[n], name_to_vid[nbr]))
    if edges_to_add:
        g.add_edges(edges_to_add)

    return {
        **network,
        "nodes": current_nodes | new_nodes,
        "collapsed_nodes": collapsed_nodes,
        "node_depths": {**network["node_depths"], **{n: None for n in new_nodes}},
        "graph": g
    }


def augment_network_view_fast(
    network,
    full_graph,
    p_crop=0.6,
    p_edge_drop=0.2,
    p_node_add=0.2,
    crop_ratio_range=(0.6, 0.9),
    edge_drop_range=(0.05, 0.2),
    max_new_nodes=10,
    random_seed=None
):
    if random_seed is not None:
        random.seed(random_seed)

    aug_net = copy.deepcopy(network)

    if random.random() < p_crop:
        ratio = random.uniform(*crop_ratio_range)
        aug_net = crop_network(aug_net, crop_ratio=ratio)

    # if random.random() < p_edge_drop:
    #     frac = random.uniform(*edge_drop_range)
    #     aug_net = delete_random_edges(aug_net, delete_frac=frac)    
    # if random.random() < p_node_add:
    #     aug_net = add_nodes(
    #         aug_net, full_graph=full_graph, max_new_nodes=max_new_nodes
    #     )

    # --- Rebuild transactions once at the end ---
    edges = [(e.source, e.target) for e in aug_net["graph"].es]
    tx = pd.DataFrame([
        {"From_Account_int": aug_net["graph"].vs[s]["name"],
         "To_Account_int": aug_net["graph"].vs[t]["name"]}
        for s, t in edges
    ])

    aug_net["transactions"] = tx

    return aug_net