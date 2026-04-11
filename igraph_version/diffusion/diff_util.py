import math
import numpy as np
import torch
from masked_diffusion import GaussianDiffusion
from masked_diffusion import ModelMeanType, ModelVarType, LossType
import igraph as ig

def cosine_beta_schedule(T, s=0.008, max_beta=0.999):
    """
    Cosine beta schedule from Nichol & Dhariwal (2021).

    Decays ᾱ_t more slowly than the linear schedule, preserving more signal
    at intermediate timesteps.  The offset `s` prevents betas from being
    too small near t=0.
    """
    alpha_bar = lambda t: math.cos((t + s) / (1.0 + s) * math.pi / 2) ** 2
    betas = []
    for i in range(T):
        betas.append(min(1.0 - alpha_bar((i + 1) / T) / alpha_bar(i / T), max_beta))
    return np.array(betas, dtype=np.float64)


def linear_beta_schedule(T):
    return torch.linspace(1e-4, 2e-2, T, dtype=torch.float64).numpy()


def create_diffusion(T=1000, schedule="cosine"):
    """
    Build a GaussianDiffusion object.

    Parameters
    ----------
    T        : total diffusion timesteps
    schedule : "cosine" (default — Nichol & Dhariwal 2021, slower signal decay)
               or "linear" (Ho et al. 2020)
    """
    betas = cosine_beta_schedule(T) if schedule == "cosine" else linear_beta_schedule(T)

    diffusion = GaussianDiffusion(
        betas=betas,
        model_mean_type=ModelMeanType.EPSILON,
        model_var_type=ModelVarType.FIXED_SMALL,
        loss_type=LossType.MSE,
        rescale_timesteps=True,
    )

    return diffusion

def network_to_dense(net):
    """
    Convert a single network dict to (x [n, 11], adj [n,n]) dense tensors.

    Node features (11 total):
      col 0  — laundering flag (binary, excluded from classifier input)
      col 1  — normalised degree
      col 2  — normalised betweenness
      col 3  — local clustering coefficient
      col 4  — PageRank
      col 5  — assortativity (graph-level, same for all nodes)
      col 6  — log mean transaction amount (sender)
      col 7  — log max transaction amount (sender)
      col 8  — transaction count (normalised)
      col 9  — payment format entropy (diversity of payment types used)
      col 10 — temporal span in hours (normalised by 30 days)

    The last 5 are derived from net["transactions"] when available;
    they default to 0 for networks loaded from old caches that lack
    the amount / payment-format columns.
    """
    graph      = net["graph"]
    n          = graph.vcount()
    laundering = net["laundering_nodes"]

    x = torch.zeros(n, 11)

    # ── Topology features (cols 0-5) ─────────────────────────────────────────
    degrees_raw     = graph.degree()
    max_deg         = max(max(degrees_raw), 1) if degrees_raw else 1
    betweenness_raw = graph.betweenness(directed=False)
    betw_denom      = max(1.0, (n - 1) * (n - 2) / 2)
    clustering_raw  = graph.transitivity_local_undirected(mode="zero")
    pagerank_raw    = graph.pagerank()
    _assort         = graph.assortativity_degree(directed=False)
    assortativity   = 0.0 if (_assort is None or math.isnan(_assort)) else _assort

    # ── Transaction features per node (cols 6-10) ────────────────────────────
    tx_df = net.get("transactions", None)
    has_tx_feats = (tx_df is not None
                    and "log_amount_received" in tx_df.columns
                    and len(tx_df) > 0)

    # Build per-node lookup: account_int -> list of (log_amount, fmt_code, timestamp)
    node_tx_map = {}   # int node_id -> dict of lists
    if has_tx_feats:
        for _, row in tx_df.iterrows():
            amt   = float(row["log_amount_received"])
            fmt   = int(row.get("payment_format_code", 0))
            ts    = row["Timestamp"].timestamp() if hasattr(row["Timestamp"], "timestamp") else 0.0
            for acct_col in ("From_Account_int", "To_Account_int"):
                acct = int(row[acct_col])
                if acct not in node_tx_map:
                    node_tx_map[acct] = {"amts": [], "fmts": [], "ts": []}
                node_tx_map[acct]["amts"].append(amt)
                node_tx_map[acct]["fmts"].append(fmt)
                node_tx_map[acct]["ts"].append(ts)

        all_ts  = [t for v in node_tx_map.values() for t in v["ts"]]
        ts_span = (max(all_ts) - min(all_ts)) / 3600.0 if len(all_ts) > 1 else 0.0
        max_tx  = max(len(v["amts"]) for v in node_tx_map.values()) if node_tx_map else 1
    else:
        ts_span = 0.0
        max_tx  = 1

    for i in range(n):
        if "name" in graph.vs.attributes():
            node_id = int(graph.vs[i]["name"])
        else:
            node_id = i

        x[i, 0] = int(node_id in laundering)
        x[i, 1] = degrees_raw[i] / max_deg
        x[i, 2] = betweenness_raw[i] / betw_denom
        x[i, 3] = clustering_raw[i]
        x[i, 4] = pagerank_raw[i]
        x[i, 5] = assortativity

        if has_tx_feats and node_id in node_tx_map:
            nd   = node_tx_map[node_id]
            amts = nd["amts"]
            fmts = nd["fmts"]
            tss  = nd["ts"]

            x[i, 6] = float(np.mean(amts))                    # log mean amount
            x[i, 7] = float(np.max(amts))                     # log max amount

            n_tx    = len(amts)
            x[i, 8] = n_tx / max(max_tx, 1)                   # tx count (normalised)

            # Payment format entropy: 0 = single format, high = mixed formats
            if n_tx > 1:
                fmt_counts = np.bincount(np.array(fmts, dtype=int),
                                         minlength=max(fmts) + 1).astype(float)
                fmt_probs  = fmt_counts / fmt_counts.sum()
                fmt_probs  = fmt_probs[fmt_probs > 0]
                x[i, 9]   = float(-np.sum(fmt_probs * np.log(fmt_probs + 1e-12)))
            # else x[i,9] stays 0

            # Node-level temporal span (hours), normalised by 30 days
            if len(tss) > 1:
                node_span  = (max(tss) - min(tss)) / 3600.0
                x[i, 10]  = node_span / (30 * 24)

    # clamp to [0,1]: IBM data can have parallel edges (multigraph) so
    # get_adjacency() may return counts > 1 — we want a binary adjacency.
    adj = torch.from_numpy(np.array(graph.get_adjacency().data, dtype=np.float32)).clamp(0, 1)

    return x, adj


def preprocess(networks, save_path="cached_dataset.pt"):
    valid = [net for net in networks if net["graph"].vcount() > 0]
    if len(valid) < len(networks):
        print(f"  [preprocess] Skipped {len(networks) - len(valid)} empty-graph networks.")
    data = [network_to_dense(net) for net in valid]

    torch.save(data, save_path)

    print(f"Saved {len(data)} graphs to {save_path}")


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
