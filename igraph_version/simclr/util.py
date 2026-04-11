import igraph as ig
import math
import os
import numpy as np
import pandas as pd
from collections import deque

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


def _extract_transactions(df, tx_index, component_nodes):
    """
    Return the subset of df whose From_Account_int AND To_Account_int are
    both in component_nodes, using the pre-built tx_index for speed.
    """
    # Candidate rows: any row whose sender is in the component
    candidate_rows = set()
    for node in component_nodes:
        candidate_rows.update(tx_index.get(node, ()))

    if not candidate_rows:
        return df.iloc[:0].copy()

    sub = df.loc[list(candidate_rows)]
    mask = sub["To_Account_int"].isin(component_nodes)
    return sub[mask].copy()


import random

def extract_transaction_ego_networks(
    df,
    max_depth=2,
    max_nodes=50,
    n_pos=2000,
    neg_pos_ratio=10,
    random_seed=42,
):
    """
    Transaction-centric graph extraction for fair comparison with
    transaction-level AML benchmarks (GFP, MultiGNN, FraudGT, etc.).

    For each sampled transaction we extract the 2-hop ego network of
    BOTH the sender and receiver accounts, merged into one subgraph.
    The graph label = that transaction's Is Laundering flag.

    Why this is more comparable to the paper:
      - Label is per-transaction, not per-subgraph (same as the benchmark)
      - Positive rate in the extracted set matches n_pos / (n_pos + n_neg),
        and the TEST set can be kept at the natural dataset ratio
      - The temporal ordering is preserved via net["timestamp"], enabling a
        temporal train/test split identical to the paper's protocol

    Args:
        df            : preprocessed DataFrame (must have Timestamp column,
                        sorted ascending)
        max_depth     : BFS depth from sender+receiver (default 2)
        max_nodes     : max nodes per subgraph
        n_pos         : number of laundering transactions to sample
        neg_pos_ratio : how many clean transactions per laundering one for
                        training; test set uses the natural ratio
        random_seed   : reproducibility seed

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
    neg_idx = df.index[df["Is Laundering"] == 0].tolist()

    n_pos   = min(n_pos, len(pos_idx))
    n_neg   = min(n_pos * neg_pos_ratio, len(neg_idx))

    sampled_pos = rng.sample(pos_idx, n_pos)
    sampled_neg = rng.sample(neg_idx, n_neg)
    sampled     = [(i, 1) for i in sampled_pos] + [(i, 0) for i in sampled_neg]
    rng.shuffle(sampled)

    networks = []

    for row_idx, tx_label in sampled:
        row           = df.loc[row_idx]
        sender        = int(row["From_Account_int"])
        receiver      = int(row["To_Account_int"])
        tx_timestamp  = row["Timestamp"]

        # BFS from both endpoints simultaneously
        visited        = {}
        collapsed_nodes = set()
        queue          = deque([(sender, 0), (receiver, 0)])

        while queue and (max_nodes is None or len(visited) < max_nodes):
            node, depth = queue.popleft()
            if node in visited:
                continue
            visited[node] = depth
            neighbors = set(g.neighbors(node, mode="all"))
            if len(neighbors) > 50:          # collapse hubs
                collapsed_nodes.add(node)
                continue
            if depth >= max_depth:
                continue
            for nbr in neighbors:
                if nbr not in visited:
                    queue.append((nbr, depth + 1))

        component_nodes = set(visited.keys())
        if len(component_nodes) < 3:
            continue

        transactions         = _extract_transactions(df, tx_index, component_nodes)
        laundering_in_comp   = component_nodes & set(
            df.loc[df["Is Laundering"] == 1, "From_Account_int"].tolist() +
            df.loc[df["Is Laundering"] == 1, "To_Account_int"].tolist()
        )

        networks.append({
            "start_node":      sender,
            "nodes":           component_nodes,
            # laundering_nodes drives the x[:,0] feature in network_to_dense;
            # use the actual laundering accounts in this subgraph so the
            # node features are still informative for the diffusion model,
            # but the GRAPH label comes from tx_label (the focal transaction)
            "laundering_nodes": laundering_in_comp if tx_label == 1 else set(),
            "collapsed_nodes": collapsed_nodes,
            "node_depths":     visited,
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
    max_nodes=300,
    random_seed=42,
):
    """
    Extract subgraphs by BFS from random starting nodes and label each
    subgraph by whether it contains any laundering transactions.

    This avoids the structural leakage of the old approach (which always
    started BFS from laundering nodes, making them artificially central).
    Here laundering nodes appear at random structural positions — just as
    they do in the real graph — so the classifier must learn genuine
    laundering patterns rather than "is there a hub in the center?".

    Label assignment:
        1  — subgraph contains at least one laundering transaction
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

            neighbors = set(g.neighbors(node, mode="all"))

            if len(neighbors) > collapse_threshold:
                collapsed_nodes.add(node)
                continue

            if depth >= max_depth:
                continue

            for nbr in neighbors:
                if nbr not in visited and nbr not in used_nodes:
                    queue.append((nbr, depth + 1))

        component_nodes = set(visited.keys())

        if len(component_nodes) < min_size:
            continue

        transactions = _extract_transactions(df, tx_index, component_nodes)

        # Label: does this subgraph contain any laundering transaction?
        laundering_in_component = component_nodes & laundering_nodes
        has_laundering = len(
            transactions[transactions["Is Laundering"] == 1]
        ) > 0

        network = {
            "start_node":      start_node,
            "nodes":           component_nodes,
            # Use transaction-based label so isolated laundering nodes that
            # happen to be in the subgraph but have no laundering edges don't
            # produce false positives.
            "laundering_nodes": laundering_in_component if has_laundering else set(),
            "collapsed_nodes": collapsed_nodes,
            "node_depths":     visited,
            "transactions":    transactions,
        }

        networks.append(network)
        used_nodes.update(component_nodes)

        if len(networks) >= max_networks:
            break

    pos = sum(1 for n in networks if len(n["laundering_nodes"]) > 0)
    print(f"  Extracted {len(networks)} subgraphs: {pos} laundering, "
          f"{len(networks) - pos} clean  "
          f"(natural ratio {pos/max(len(networks),1)*100:.1f}% positive)")

    return networks

