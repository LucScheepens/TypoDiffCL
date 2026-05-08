"""
AML typological pattern features.

Computes 8 per-node features that encode membership in canonical AML structural
patterns (fan-out, fan-in, stack, bipartite scatter-gather, cycle).

Output tensor [n, 8]:
  col 0  out_degree_norm      — fan-out signal: normalised directed out-degree
  col 1  in_degree_norm       — fan-in signal:  normalised directed in-degree
  col 2  degree_asymmetry     — |out-in| / (out+in+1), 0=balanced 1=pure source/sink
  col 3  is_passthrough       — in==1 & out==1 (interior node of a stack/chain)
  col 4  stack_depth_norm     — length of the passthrough chain this node belongs to, /n
  col 5  in_cycle             — in an SCC of size > 1 (round-tripping / U-turn)
  col 6  scatter_gather_score — node lies between a fan-out node (≤2 hops upstream)
                                and a fan-in node (≤2 hops downstream)
  col 7  bipartite_score      — graph-level: fraction of edges crossing the greedy
                                2-cut (1.0 = perfectly bipartite, <1 = has odd cycles)

All features are in [0, 1].  When directed transaction data is unavailable (e.g.
purely synthetic graphs from the diffusion model), cols 0-6 are 0; col 7 is still
computed from the undirected graph.
"""

import igraph as ig
import numpy as np
import torch
from collections import defaultdict, deque


# ── Threshold for classifying a node as a fan-out / fan-in hub ───────────────
# Requires at least this many unique directed neighbours in the dominant direction
# AND the dominant degree must be at least 2× the non-dominant degree.
_FAN_THRESH = 3
_FAN_RATIO  = 2.0


def compute_pattern_features(net) -> torch.Tensor:
    """
    Compute AML typological pattern features for one network dict.

    Parameters
    ----------
    net : dict
        Network dict produced by extract_transaction_ego_networks() or
        extract_networks_igraph().  Must contain:
          "graph"        — undirected igraph.Graph
        Optionally:
          "transactions" — pd.DataFrame with From_Account_int, To_Account_int

    Returns
    -------
    torch.Tensor of shape [n, 8]
    """
    g = net["graph"]
    n = g.vcount()

    if n == 0:
        return torch.zeros(0, 8)

    feats = torch.zeros(n, 8)

    # ── Node ID ↔ vertex-index mapping ───────────────────────────────────────
    has_names = "name" in g.vs.attributes()
    node_ids  = [int(g.vs[i]["name"]) if has_names else i for i in range(n)]
    node_to_vid = {nid: i for i, nid in enumerate(node_ids)}

    # ── Build directed neighbour sets from transactions ───────────────────────
    tx_df = net.get("transactions", None)
    has_directed = (
        tx_df is not None
        and len(tx_df) > 0
        and "From_Account_int" in tx_df.columns
        and "To_Account_int"   in tx_df.columns
    )

    if has_directed:
        frm_arr = tx_df["From_Account_int"].to_numpy(dtype=np.int64)
        to_arr  = tx_df["To_Account_int"].to_numpy(dtype=np.int64)

        # Only keep edges whose both endpoints are nodes in this subgraph
        node_id_set = set(node_ids)
        keep = np.isin(frm_arr, list(node_id_set)) & np.isin(to_arr, list(node_id_set))
        frm_arr = frm_arr[keep]
        to_arr  = to_arr[keep]

        # Unique directed neighbours per node (deduplicate multi-edges)
        out_neighbors: defaultdict = defaultdict(set)
        in_neighbors:  defaultdict = defaultdict(set)
        for f, t in zip(frm_arr.tolist(), to_arr.tolist()):
            if f != t:
                out_neighbors[f].add(t)
                in_neighbors[t].add(f)

        out_degs = np.array([len(out_neighbors[nid]) for nid in node_ids], dtype=np.float32)
        in_degs  = np.array([len(in_neighbors[nid])  for nid in node_ids], dtype=np.float32)

        max_out = float(max(out_degs.max(), 1))
        max_in  = float(max(in_degs.max(),  1))

        # cols 0-2: degree-based features
        feats[:, 0] = torch.from_numpy(out_degs / max_out)
        feats[:, 1] = torch.from_numpy(in_degs  / max_in)
        feats[:, 2] = torch.from_numpy(
            np.abs(out_degs - in_degs) / (out_degs + in_degs + 1.0)
        )

        # col 3: is_passthrough (exactly one predecessor, one successor)
        is_pass = (out_degs == 1) & (in_degs == 1)
        feats[:, 3] = torch.from_numpy(is_pass.astype(np.float32))

        # col 4: stack_depth_norm
        # For each passthrough node, BFS through the passthrough-only subgraph
        # to find the full chain length, then normalise by n.
        if is_pass.any():
            passthrough_set = set(int(i) for i in np.where(is_pass)[0])
            chain_lengths   = np.zeros(n, dtype=np.float32)
            visited_chain   = np.zeros(n, dtype=bool)

            for start_vid in passthrough_set:
                if visited_chain[start_vid]:
                    continue
                component = []
                queue = deque([start_vid])
                visited_chain[start_vid] = True
                while queue:
                    v = queue.popleft()
                    component.append(v)
                    nid_v = node_ids[v]
                    for direction_map in (out_neighbors, in_neighbors):
                        for nb_nid in direction_map[nid_v]:
                            if nb_nid in node_to_vid:
                                nb_v = node_to_vid[nb_nid]
                                if nb_v in passthrough_set and not visited_chain[nb_v]:
                                    visited_chain[nb_v] = True
                                    queue.append(nb_v)
                clen = float(len(component))
                for v in component:
                    chain_lengths[v] = clen

            feats[:, 4] = torch.from_numpy(chain_lengths / max(n, 1))

        # col 5: in_cycle (node belongs to an SCC of size > 1)
        # Build a directed igraph from deduplicated edges and compute SCCs.
        edge_vids = list({
            (node_to_vid[int(f)], node_to_vid[int(t)])
            for f, t in zip(frm_arr.tolist(), to_arr.tolist())
            if int(f) in node_to_vid and int(t) in node_to_vid
               and node_to_vid[int(f)] != node_to_vid[int(t)]
        })
        if edge_vids:
            g_dir = ig.Graph(n=n, edges=edge_vids, directed=True)
            scc   = g_dir.clusters(mode="strong")
            scc_sizes   = np.array(scc.sizes(),      dtype=np.int32)
            memberships = np.array(scc.membership,   dtype=np.int32)
            feats[:, 5] = torch.from_numpy(
                (scc_sizes[memberships] > 1).astype(np.float32)
            )

        # col 6: scatter_gather_score
        # Fan-out node: out_deg >= _FAN_THRESH and out_deg / (in_deg+1) >= _FAN_RATIO
        # Fan-in  node: in_deg  >= _FAN_THRESH and in_deg  / (out_deg+1) >= _FAN_RATIO
        # A node scores 1 if it can be reached (≤2 forward hops) from a fan-out node
        # AND it can reach (≤2 forward hops) a fan-in node.
        fanout_vids = {
            i for i in range(n)
            if out_degs[i] >= _FAN_THRESH
            and out_degs[i] / (in_degs[i] + 1.0) >= _FAN_RATIO
        }
        fanin_vids = {
            i for i in range(n)
            if in_degs[i] >= _FAN_THRESH
            and in_degs[i] / (out_degs[i] + 1.0) >= _FAN_RATIO
        }

        if fanout_vids and fanin_vids:
            # Forward BFS from every fan-out node up to 2 hops
            reachable_from_fanout: set = set(fanout_vids)
            for fo_vid in fanout_vids:
                frontier = {fo_vid}
                for _ in range(2):
                    nxt = set()
                    for v in frontier:
                        for nb_nid in out_neighbors[node_ids[v]]:
                            if nb_nid in node_to_vid:
                                nb_v = node_to_vid[nb_nid]
                                if nb_v not in reachable_from_fanout:
                                    nxt.add(nb_v)
                    reachable_from_fanout.update(nxt)
                    frontier = nxt

            # Backward BFS from every fan-in node up to 2 hops
            can_reach_fanin: set = set(fanin_vids)
            for fi_vid in fanin_vids:
                frontier = {fi_vid}
                for _ in range(2):
                    nxt = set()
                    for v in frontier:
                        for nb_nid in in_neighbors[node_ids[v]]:
                            if nb_nid in node_to_vid:
                                nb_v = node_to_vid[nb_nid]
                                if nb_v not in can_reach_fanin:
                                    nxt.add(nb_v)
                    can_reach_fanin.update(nxt)
                    frontier = nxt

            for vid in reachable_from_fanout & can_reach_fanin:
                feats[vid, 6] = 1.0

    # col 7: bipartite_score (graph-level, computed on undirected graph)
    # Greedy BFS 2-coloring; score = fraction of edges crossing the partition.
    # Perfect bipartite graph → 1.0.  Triangle → ~0.67.
    if g.ecount() > 0:
        coloring = np.full(n, -1, dtype=np.int8)
        for start in range(n):
            if coloring[start] != -1:
                continue
            coloring[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                for v in g.neighbors(u):
                    if coloring[v] == -1:
                        coloring[v] = 1 - coloring[u]
                        queue.append(v)

        satisfied = sum(
            1 for u, v in g.get_edgelist() if coloring[u] != coloring[v]
        )
        bipartite_score = float(satisfied) / g.ecount()
        feats[:, 7] = bipartite_score

    return feats
