import igraph as ig
import math
import os
import random
import numpy as np
import pandas as pd
from collections import deque, defaultdict

def preprocess_df(CSV_PATH):
    """
    Preprocess the IBM Hi-Small Transactions dataset.
    Returns:
        pd.DataFrame: Preprocessed DataFrame with additional columns.
    """

    current_dir = os.path.dirname(os.getcwd())
    FILE_PATH = os.path.join(current_dir, "data", "IBM", "HI-SmallTransactions.txt")
    
    needed_cols = ["Timestamp", "Is Laundering",
                   "From Bank", "Account", "To Bank", "Account.1",
                   "Amount Received", "Amount Paid", "Payment Format"]
    col_dtypes = {
        "Is Laundering":    str,
        "From Bank":        "category",
        "Account":          "category",
        "To Bank":          "category",
        "Account.1":        "category",
        "Payment Format":   "category",
    }
    df_full = pd.read_csv(CSV_PATH, usecols=needed_cols, dtype=col_dtypes)
    df_full["Timestamp"] = pd.to_datetime(df_full["Timestamp"], format="mixed")
    df_full["Is Laundering"] = pd.to_numeric(df_full["Is Laundering"], errors="coerce").fillna(2).astype("int8")

    # Log-transform amounts (add 1 to handle 0-amount transactions)
    df_full["log_amount_received"] = np.log1p(
        pd.to_numeric(df_full["Amount Received"], errors="coerce").fillna(0)
    ).astype("float32")
    df_full["log_amount_paid"] = np.log1p(
        pd.to_numeric(df_full["Amount Paid"], errors="coerce").fillna(0)
    ).astype("float32")
    df_full.drop(columns=["Amount Received", "Amount Paid"], inplace=True)

    # Encode payment format as integer (kept as category codes)
    df_full["payment_format_code"] = df_full["Payment Format"].cat.codes.astype("int8")
    df_full.drop(columns=["Payment Format"], inplace=True)

    # Build string keys for factorization then immediately release them
    keys  = df_full["To Bank"].astype(str) + "|" + df_full["Account.1"].astype(str)
    keys2 = df_full["From Bank"].astype(str) + "|" + df_full["Account"].astype(str)

    all_keys = pd.concat([keys, keys2])
    codes, uniques = pd.factorize(all_keys)
    mapping = dict(zip(uniques, range(len(uniques))))

    df_full["To_Account_int"]   = keys.map(mapping).astype("int32")
    df_full["From_Account_int"] = keys2.map(mapping).astype("int32")

    # Rename for consistency then drop heavy string/category columns
    df_full = df_full.rename(columns={"Account.1": "To Account", "Account": "From Account"})
    df_full.drop(columns=["From Bank", "To Bank", "From Account", "To Account"], inplace=True)

    # Sort by time so temporal splits are simply index-based
    df_full = df_full.sort_values("Timestamp").reset_index(drop=True)

    return df_full


def build_igraph_from_df(df):
    g = ig.Graph.DataFrame(
        df[["From_Node", "To_Node"]],
        directed=True
    )
    return g

def _build_tx_index(df):
    """
    Build a dict mapping each account int -> set of row-index positions
    where that account appears as the sender.  Used to avoid full-DataFrame
    scans when extracting per-network transactions.
    """
    from collections import defaultdict
    idx = defaultdict(set)
    from_vals = df["From_Account_int"].to_numpy()
    row_ids    = df.index.to_numpy()
    for row_id, acct in zip(row_ids, from_vals):
        idx[acct].add(row_id)
    return idx


def _build_laundering_adj(df):
    """
    Build an undirected adjacency mapping for laundering-only transactions.
    Used to trace the full connected laundering component from a focal transaction.
    """
    adj = defaultdict(set)
    laund = df[df["Is Laundering"] == 1]
    senders   = laund["From_Account_int"].to_numpy()
    receivers = laund["To_Account_int"].to_numpy()
    for s, r in zip(senders, receivers):
        s, r = int(s), int(r)
        adj[s].add(r)
        adj[r].add(s)
    return adj


def _trace_laundering_component(sender, receiver, laundering_adj, max_nodes):
    """
    BFS through the laundering-only graph from both endpoints of the focal
    transaction.  Returns the full set of accounts reachable via laundering
    edges, capped at max_nodes.
    """
    visited = set()
    queue = deque([sender, receiver])
    while queue and (max_nodes is None or len(visited) < max_nodes):
        node = queue.popleft()
        if node in visited:
            continue
        visited.add(node)
        for nbr in laundering_adj.get(node, ()):
            if nbr not in visited:
                queue.append(nbr)
    return visited


def _extract_transactions(df, tx_index, component_nodes, cutoff_time=None):
    """
    Return the subset of df whose From_Account_int AND To_Account_int are
    both in component_nodes, using the pre-built tx_index for speed.
    If cutoff_time is given, only transactions at or before that time are included.
    """
    candidate_rows = set()
    for node in component_nodes:
        candidate_rows.update(tx_index.get(node, ()))

    if not candidate_rows:
        return df.iloc[:0].copy()

    sub = df.loc[list(candidate_rows)]
    if cutoff_time is not None:
        sub = sub[sub["Timestamp"] <= cutoff_time]
    mask = sub["To_Account_int"].isin(component_nodes)
    return sub[mask].copy()


def extract_transaction_ego_networks(
    df,
    max_depth=2,
    max_nodes=50,
    n_pos=2000,
    neg_pos_ratio=10,
    collapse_threshold=50,
    bfs_mode="all",
    random_seed=42,
):
    """
    Training-mode subgraph extraction anchored on individual transactions.

    Positive examples (tx_label=1):
        Trace the full connected component of laundering transactions reachable
        from the focal transaction's sender and receiver via laundering-only
        edges.  This gives structurally complete laundering patterns rather than
        arbitrary BFS slices through the middle of a chain.  Up to max_nodes,
        the laundering core is then padded with 1-hop clean context so the model
        also sees the boundary between laundering and legitimate activity.
        node_depths: 0 = laundering core, 1 = clean context.

    Negative examples (tx_label=0):
        Standard BFS ego network from both endpoints up to max_depth hops,
        identical to before.  No laundering component to trace.

    Use extract_networks_igraph for evaluation: it samples at the natural class
    ratio and places laundering nodes at random structural positions, mirroring
    real-world inference conditions.

    Args:
        df                 : preprocessed DataFrame (Timestamp column, sorted ascending)
        max_depth          : BFS depth for negative ego networks (default 2)
        max_nodes          : max nodes per subgraph
        n_pos              : number of laundering transactions to sample
        neg_pos_ratio      : clean transactions per laundering one
        collapse_threshold : hub degree above which a node is collapsed
        bfs_mode           : igraph neighbor mode ("all", "out", "in")
        random_seed        : reproducibility seed

    Returns list of network dicts with keys:
        transactions, nodes, laundering_nodes, collapsed_nodes,
        node_depths, start_node, timestamp (pd.Timestamp), tx_label (int)
    """
    rng = random.Random(random_seed)

    g = ig.Graph.DataFrame(
        df[["From_Account_int", "To_Account_int"]],
        directed=True,
        use_vids=True,
    )
    tx_index = _build_tx_index(df)

    pos_idx = df.index[df["Is Laundering"] == 1].tolist()
    # Deduplicate by (sender, receiver) so near-identical ego networks
    # don't inflate the training set and collapse the contrastive loss.
    pos_rows = df.loc[pos_idx, ["From_Account_int", "To_Account_int"]]
    pos_idx = pos_rows.drop_duplicates().index.tolist()

    neg_idx = df.index[df["Is Laundering"] == 0].tolist()

    n_pos   = min(n_pos, len(pos_idx))
    n_neg   = min(n_pos * neg_pos_ratio, len(neg_idx))

    sampled_pos = rng.sample(pos_idx, n_pos)
    sampled_neg = rng.sample(neg_idx, n_neg)
    sampled     = [(i, 1) for i in sampled_pos] + [(i, 0) for i in sampled_neg]
    rng.shuffle(sampled)

    # Pre-compute once — scanning df twice per iteration for 22k+ samples is expensive.
    all_laundering_accounts = set(
        df.loc[df["Is Laundering"] == 1, "From_Account_int"]
    ).union(df.loc[df["Is Laundering"] == 1, "To_Account_int"])

    laundering_adj = _build_laundering_adj(df)

    networks = []

    for row_idx, tx_label in sampled:
        row          = df.loc[row_idx]
        sender       = int(row["From_Account_int"])
        receiver     = int(row["To_Account_int"])
        tx_timestamp = row["Timestamp"]

        if tx_label == 1:
            # Trace the full connected laundering component from both endpoints,
            # then pad with 1-hop clean context up to max_nodes.
            core = _trace_laundering_component(
                sender, receiver, laundering_adj, max_nodes
            )
            component_nodes = set(core)
            collapsed_nodes = set()

            for node in list(core):
                if max_nodes is not None and len(component_nodes) >= max_nodes:
                    break
                nbrs = g.neighbors(node, mode=bfs_mode)
                if len(nbrs) > collapse_threshold:
                    collapsed_nodes.add(node)
                    continue
                for nbr in nbrs:
                    if nbr not in component_nodes:
                        if max_nodes is not None and len(component_nodes) >= max_nodes:
                            break
                        component_nodes.add(nbr)

            # depth 0 = laundering core, depth 1 = clean context
            node_depths = {n: 0 for n in core}
            node_depths.update({n: 1 for n in component_nodes - core})
        else:
            # Standard BFS ego network for negatives.
            visited = {}
            collapsed_nodes = set()
            queue = deque([(sender, 0), (receiver, 0)])

            while queue and (max_nodes is None or len(visited) < max_nodes):
                node, depth = queue.popleft()
                if node in visited:
                    continue
                visited[node] = depth
                neighbors = set(g.neighbors(node, mode=bfs_mode))
                if len(neighbors) > collapse_threshold:
                    collapsed_nodes.add(node)
                    continue
                if depth >= max_depth:
                    continue
                for nbr in neighbors:
                    if nbr not in visited:
                        queue.append((nbr, depth + 1))

            component_nodes = set(visited.keys())
            node_depths = visited

        if len(component_nodes) < 3:
            continue

        # Restrict to transactions at or before the focal tx to prevent future leakage.
        transactions = _extract_transactions(
            df, tx_index, component_nodes, cutoff_time=tx_timestamp
        )
        laundering_in_comp = component_nodes & all_laundering_accounts

        networks.append({
            "start_node":      sender,
            "nodes":           component_nodes,
            "laundering_nodes": laundering_in_comp,
            "collapsed_nodes": collapsed_nodes,
            "node_depths":     node_depths,
            "transactions":    transactions,
            "timestamp":       tx_timestamp,
            "tx_label":        tx_label,
        })

    pos = sum(1 for n in networks if n["tx_label"] == 1)
    print(f"  Extracted {len(networks)} transaction ego-networks: "
          f"{pos} laundering ({pos/len(networks)*100:.1f}%), "
          f"{len(networks)-pos} clean")
    return networks


def extract_networks_igraph(
    df,
    max_depth=4,
    max_networks=4000,
    collapse_threshold=10,
    min_size=5,
    max_nodes=64,
    bfs_mode="all",
    sim_threshold=0.5,
    random_seed=42,
):
    """
    Evaluation-mode subgraph extraction at the natural class ratio.

    BFS starts from random nodes so laundering accounts appear at arbitrary
    structural positions — exactly as at inference time, where you have no
    prior knowledge of which transactions are laundering.  The natural class
    imbalance is preserved, giving a realistic measure of detector performance.

    Do NOT use this for training.  The laundering signal in positive subgraphs
    can range from a complete laundering chain to a single laundering account
    whose connections fall outside the subgraph boundary, making many positive
    labels structurally ambiguous.  For training, use extract_transaction_ego_networks,
    which anchors each positive on a known laundering transaction and traces the
    full laundering component so the model sees complete patterns.

    Label assignment:
        1  — subgraph contains at least one laundering transaction (edge-based)
        0  — subgraph contains no laundering transactions

    Returns a list of network dicts compatible with network_to_dense().
    """
    random.seed(random_seed)

    laundering_nodes = set(
        df.loc[df["Is Laundering"] == 1, "From_Account_int"]
    ).union(
        df.loc[df["Is Laundering"] == 1, "To_Account_int"]
    )

    g = ig.Graph.DataFrame(
        df[["From_Account_int", "To_Account_int"]],
        directed=True,
        use_vids=True,
    )

    tx_index = _build_tx_index(df)

    all_nodes = list(range(g.vcount()))
    random.shuffle(all_nodes)

    networks = []
    used_nodes = set()

    for start_node in all_nodes:
        if start_node in used_nodes:
            continue

        visited = {}
        collapsed_nodes = set()
        queue = deque([(start_node, 0)])

        while queue and (max_nodes is None or len(visited) < max_nodes):
            node, depth = queue.popleft()

            if node in visited:
                continue

            visited[node] = depth

            neighbors = set(g.neighbors(node, mode=bfs_mode))

            if len(neighbors) > collapse_threshold:
                collapsed_nodes.add(node)
                continue

            if depth >= max_depth:
                continue

            for nbr in neighbors:
                if nbr not in visited:
                    queue.append((nbr, depth + 1))

        component_nodes = set(visited.keys())

        if len(component_nodes) < min_size:
            continue

        # Reject subgraphs too similar to recently extracted ones.
        # Checking only the last 50 keeps this O(1) amortised per subgraph.
        if any(
            len(component_nodes & n["nodes"]) / len(component_nodes | n["nodes"])
            > sim_threshold
            for n in networks[-50:]
        ):
            continue

        transactions = _extract_transactions(df, tx_index, component_nodes)

        laundering_in_component = component_nodes & laundering_nodes
        has_laundering = len(transactions[transactions["Is Laundering"] == 1]) > 0

        network = {
            "start_node":      start_node,
            "nodes":           component_nodes,
            "laundering_nodes": laundering_in_component if has_laundering else set(),
            "collapsed_nodes": collapsed_nodes,
            "node_depths":     visited,
            "transactions":    transactions,
            "tx_label":        1 if has_laundering else 0,
        }

        networks.append(network)
        used_nodes.update(component_nodes)

        if len(networks) >= max_networks:
            break

    pos = sum(1 for n in networks if n["tx_label"] == 1)
    print(f"  Extracted {len(networks)} subgraphs: {pos} laundering, "
          f"{len(networks) - pos} clean  "
          f"(natural ratio {pos/max(len(networks),1)*100:.1f}% positive)")

    return networks

