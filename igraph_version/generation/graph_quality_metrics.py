"""
graph_quality_metrics.py
────────────────────────────────────────────────────────────────────────────
Tier-1 per-generated-graph quality metrics.

For each generated graph three complementary scores are computed — all
without retraining any classifier:

1. SimCLR embedding distance  (cosine, k-NN)
   Distance in the encoder's representation space to the k nearest real
   laundering training graphs.  Small → realistic; large → out-of-distribution.

2. Feature-space Wasserstein distance
   Mean Wasserstein-1 distance (averaged over the 5 structural node
   features: degree, betweenness, clustering, pagerank, assortativity)
   between this graph's node feature distribution and the pooled
   distribution of real laundering training graph nodes.
   Small → feature-realistic; large → distribution-shifted.

3. Edge density
   |E| / (N * (N-1)).  Values near 1.0 flag the density pathology
   documented in the diffusion model — dense graphs hurt GIN/Transformer
   but are invisible to DeepSets.

Composite quality score Q ∈ [0, 1]:
   Q = mean(1 - norm_emb_dist,  1 - norm_wass_dist,  1 - norm_density)
   Higher Q = graph is closer to the real laundering distribution AND has
   realistic density.  Useful as a cheap proxy for data value before any
   retraining.
"""

import csv
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from scipy.stats import wasserstein_distance
from torch_geometric.data import Data, Batch


_FEAT_NAMES = ["degree", "betweenness", "clustering", "pagerank", "assortativity"]


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _embed_gen_data(gen_data, encoder, device, batch_size=64):
    """
    Embed generated PyG graphs through the SimCLR encoder.

    Generated graphs may have been padded to IN_CHANNELS (IBM feature count)
    by gen_output_to_pyg.  encoder.in_dim is used to strip that padding and,
    for legacy 6-D Elliptic checkpoints, to prepend the graph label column.

    Returns
    -------
    H : Tensor [N_gen, 128]  L2-normalised embeddings
    """
    enc_dim     = encoder.in_dim
    needs_label = (enc_dim == 6)
    n_real      = enc_dim - 1 if needs_label else enc_dim

    ext_graphs = []
    for g in gen_data:
        x = g.x[:, :n_real]            # strip padding if present
        if needs_label:
            lc = torch.full((x.shape[0], 1), float(g.y.item()))
            x  = torch.cat([lc, x], dim=1)
        ext_graphs.append(Data(x=x, edge_index=g.edge_index.clone()))

    H_list = []
    with torch.no_grad():
        for i in range(0, len(ext_graphs), batch_size):
            chunk = Batch.from_data_list(ext_graphs[i : i + batch_size]).to(device)
            H_list.append(encoder(chunk).cpu())

    H = torch.cat(H_list, dim=0)       # [N_gen, 128]
    return F.normalize(H, dim=1)


def _normalize_01(arr):
    """Min-max normalise to [0, 1]; returns zeros if range is negligible."""
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-9:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Metric 1 — SimCLR embedding distance
# ─────────────────────────────────────────────────────────────────────────────

def compute_embedding_distances(gen_data, H_train_laund, encoder, device, k=5):
    """
    Mean cosine distance from each generated graph to its k nearest real
    laundering training graph embeddings.

    Parameters
    ----------
    gen_data      : list[PyG Data]  generated graphs (5-dim node features)
    H_train_laund : Tensor [M, 128] L2-normalised embeddings of real
                    laundering training graphs only
    encoder       : SimCLR GraphEncoder (in_dim=5 after label-leakage fix)
    device        : torch.device
    k             : number of nearest neighbours to average over

    Returns
    -------
    distances : np.ndarray [N_gen]
        Mean cosine distance to k-NN in [0, 2].
        Lower = closer to real laundering distribution = more realistic.
    """
    if len(gen_data) == 0:
        return np.array([], dtype=np.float32)

    H_gen = _embed_gen_data(gen_data, encoder, device)     # [N_gen, 128]
    H_ref = H_train_laund.to(H_gen.device)                 # [M, 128]

    # cosine similarity matrix — both sides are already L2-normalised
    sim   = H_gen @ H_ref.T                                # [N_gen, M]
    k_eff = min(k, H_ref.shape[0])
    top_k = torch.topk(sim, k_eff, dim=1).values          # [N_gen, k]
    mean_sim = top_k.mean(dim=1).numpy()                   # [N_gen]
    return (1.0 - mean_sim).astype(np.float32)             # distance ∈ [0, 2]


# ─────────────────────────────────────────────────────────────────────────────
# Metric 2 — Feature-space Wasserstein distance
# ─────────────────────────────────────────────────────────────────────────────

def compute_wasserstein_distances(gen_data, train_laund_data):
    """
    Mean per-feature Wasserstein-1 distance between each generated graph's
    node feature distribution and the pooled real laundering training
    distribution.

    Parameters
    ----------
    gen_data         : list[PyG Data]  generated graphs (5-dim node features)
    train_laund_data : list[PyG Data]  real laundering training graphs (5-dim)

    Returns
    -------
    distances : np.ndarray [N_gen]
        Mean Wasserstein-1 across 5 features.
        Lower = node feature distribution matches real laundering data better.
    """
    if len(gen_data) == 0:
        return np.array([], dtype=np.float32)
    if len(train_laund_data) == 0:
        return np.full(len(gen_data), np.nan, dtype=np.float32)

    # Pool all real laundering training node features → [M_nodes, 5]
    ref_feats = np.vstack([g.x.numpy() for g in train_laund_data])
    n_feats   = ref_feats.shape[1]

    dists = []
    for g in gen_data:
        gen_feats = g.x.numpy()       # [n_nodes, 5]
        per_feat  = [
            wasserstein_distance(gen_feats[:, f], ref_feats[:, f])
            for f in range(n_feats)
        ]
        dists.append(float(np.mean(per_feat)))
    return np.array(dists, dtype=np.float32)


def compute_wasserstein_per_feature(gen_data, train_laund_data):
    """
    Same as compute_wasserstein_distances but returns the full
    per-feature breakdown as a [N_gen, 5] array.
    Useful for diagnosing which feature dimension is most OOD.
    """
    if len(gen_data) == 0:
        return np.empty((0, len(_FEAT_NAMES)), dtype=np.float32)
    if len(train_laund_data) == 0:
        return np.full((len(gen_data), len(_FEAT_NAMES)), np.nan, dtype=np.float32)

    ref_feats = np.vstack([g.x.numpy() for g in train_laund_data])
    n_feats   = ref_feats.shape[1]

    out = []
    for g in gen_data:
        gen_feats = g.x.numpy()
        out.append([
            wasserstein_distance(gen_feats[:, f], ref_feats[:, f])
            for f in range(n_feats)
        ])
    return np.array(out, dtype=np.float32)   # [N_gen, 5]


# ─────────────────────────────────────────────────────────────────────────────
# Metric 3 — Edge density
# ─────────────────────────────────────────────────────────────────────────────

def compute_edge_densities(gen_data):
    """
    Edge density = |E| / (N * (N - 1)) for each generated graph.

    Edge_index counts directed edges; for an undirected graph stored with
    both (u→v) and (v→u), the formula still gives the correct density
    because the denominator is also for directed pairs.

    Returns
    -------
    densities : np.ndarray [N_gen]  ∈ [0, 1]
        Higher values indicate denser (potentially pathological) graphs.
    """
    densities = []
    for g in gen_data:
        n         = g.x.shape[0]
        e         = g.edge_index.shape[1]
        max_edges = max(n * (n - 1), 1)   # directed pairs, no self-loops
        densities.append(e / max_edges)
    return np.array(densities, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Metric 4 — Degree-distribution KL divergence  (Direction 5)
# ─────────────────────────────────────────────────────────────────────────────

def compute_degree_kl_divergence(gen_data, train_laund_data):
    """
    Per-graph KL divergence KL(gen || real) between each generated graph's
    node-degree distribution and the pooled degree distribution of real
    laundering training graphs.

    Uses Laplace-smoothed histograms so no bin is ever zero.
    Lower divergence means the generated graph's degree sequence is closer
    to real laundering data — a good proxy for realistic topology.

    Parameters
    ----------
    gen_data         : list[PyG Data]  generated graphs
    train_laund_data : list[PyG Data]  real laundering training graphs

    Returns
    -------
    kl_divs : np.ndarray [N_gen]  KL divergence ≥ 0.  Lower is better.
    """
    if len(gen_data) == 0:
        return np.array([], dtype=np.float32)
    if len(train_laund_data) == 0:
        return np.full(len(gen_data), np.nan, dtype=np.float32)

    def _node_degrees(data_list):
        """Collect per-node degree (out-degree from edge_index[0]) across all graphs."""
        degs = []
        for g in data_list:
            n = g.num_nodes
            if g.edge_index.shape[1] > 0:
                src = g.edge_index[0].numpy()
                degs.extend(np.bincount(src, minlength=n).tolist())
            else:
                degs.extend([0] * n)
        return np.array(degs, dtype=np.float32)

    real_degs = _node_degrees(train_laund_data)
    # Cap bins at 99th-percentile to avoid sparse high-degree tails
    max_deg = max(int(np.percentile(real_degs, 99)), 1)
    bins    = np.arange(0, max_deg + 2)

    # Reference histogram with Laplace smoothing
    real_hist, _ = np.histogram(real_degs, bins=bins)
    real_prob    = (real_hist.astype(float) + 1.0)
    real_prob   /= real_prob.sum()

    kl_divs = []
    for g in gen_data:
        n = g.num_nodes
        if g.edge_index.shape[1] > 0:
            src = g.edge_index[0].numpy()
            deg = np.bincount(src, minlength=n)
        else:
            deg = np.zeros(n, dtype=np.int64)

        gen_hist, _ = np.histogram(deg, bins=bins)
        gen_prob    = (gen_hist.astype(float) + 1.0)
        gen_prob   /= gen_prob.sum()

        kl = float(np.sum(gen_prob * np.log(gen_prob / real_prob)))
        kl_divs.append(kl)

    return np.array(kl_divs, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic — SimCLR embedding class separation  (Direction 3)
# ─────────────────────────────────────────────────────────────────────────────

def compute_embedding_separation(data_list, labels, encoder, device):
    """
    Measure how well the SimCLR encoder separates laundering from clean graphs
    BEFORE generation.  Poor separation means the probe will produce weak guidance
    signals, so the generated graphs will not be meaningfully class-conditional.

    Two complementary metrics:
    - Silhouette score  ∈ [-1, 1]  — higher → cleaner cluster separation in
      embedding space (computed with cosine distance).
    - Linear probe AUC  ∈ [0, 1]  — 5-fold CV AUC of a logistic regression
      trained on the frozen embeddings.  Measures whether class signal is
      linearly decodable (the minimum bar for useful guidance).

    Call this after loading the encoder and before running generation.
    If silhouette < 0.05 or linear_probe_auc < 0.65, consider retraining
    SimCLR with stronger SupCon loss (supcon_weight > 0.5) or more epochs.

    Parameters
    ----------
    data_list : list[PyG Data]  training graphs (all classes)
    labels    : list[int]       binary label per graph
    encoder   : GraphEncoder in eval mode
    device    : torch.device

    Returns
    -------
    dict with keys "silhouette" and "linear_probe_auc"
    """
    from sklearn.metrics import silhouette_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    labels_arr = np.array(labels, dtype=int)
    if len(np.unique(labels_arr)) < 2 or len(data_list) < 10:
        return {"silhouette": float("nan"), "linear_probe_auc": float("nan")}

    # Encode all graphs — handle legacy 6-D encoder (prepend label)
    enc_dim      = encoder.in_dim
    needs_label  = (enc_dim == 6)
    ext_graphs = []
    for g in data_list:
        x = g.x.clone()
        if needs_label:
            lc = torch.full((x.shape[0], 1), float(g.y.item()))
            x  = torch.cat([lc, x], dim=1)
        ext_graphs.append(Data(x=x, edge_index=g.edge_index.clone()))
    H_list = []
    with torch.no_grad():
        for i in range(0, len(ext_graphs), 64):
            chunk = Batch.from_data_list(ext_graphs[i : i + 64]).to(device)
            H_list.append(encoder(chunk).cpu())
    H   = torch.cat(H_list, dim=0)
    H_n = F.normalize(H, dim=1).numpy()  # [N, 128]

    # Silhouette score — subsample to ≤2000 for speed
    try:
        n = len(labels_arr)
        if n > 2000:
            rng = np.random.default_rng(42)
            idx = rng.choice(n, 2000, replace=False)
            sil = float(silhouette_score(H_n[idx], labels_arr[idx], metric="cosine"))
        else:
            sil = float(silhouette_score(H_n, labels_arr, metric="cosine"))
    except Exception:
        sil = float("nan")

    # Linear probe AUC (5-fold CV), cap folds by minority class size
    try:
        n_pos  = int(labels_arr.sum())
        n_folds = min(5, max(2, n_pos))
        clf    = LogisticRegression(max_iter=500, C=1.0, random_state=42)
        aucs   = cross_val_score(clf, H_n, labels_arr, cv=n_folds, scoring="roc_auc")
        lp_auc = float(aucs.mean())
    except Exception:
        lp_auc = float("nan")

    return {"silhouette": sil, "linear_probe_auc": lp_auc}


# ─────────────────────────────────────────────────────────────────────────────
# Composite scoring
# ─────────────────────────────────────────────────────────────────────────────

def score_generated_graphs(gen_data, train_laund_data, H_train_laund,
                           encoder, device, k=5):
    """
    Compute all Tier-1 quality metrics for a list of generated graphs.

    Parameters
    ----------
    gen_data         : list[PyG Data]  generated graphs (5-dim node features)
    train_laund_data : list[PyG Data]  real laundering training graphs (5-dim)
    H_train_laund    : Tensor [M, 128] L2-normalised SimCLR embeddings of
                       real laundering training graphs (label == 1 subset only)
    encoder          : SimCLR GraphEncoder (in_dim=5 after label-leakage fix)
    device           : torch.device
    k                : k-NN for embedding distance

    Returns
    -------
    metrics : dict with keys
        "emb_dist"      np.ndarray [N_gen]  cosine distance to k-NN (↓ better)
        "wass_dist"     np.ndarray [N_gen]  mean Wasserstein-1 (↓ better)
        "wass_per_feat" np.ndarray [N_gen, 5]  per-feature breakdown
        "density"       np.ndarray [N_gen]  edge density (↓ better)
        "degree_kl"     np.ndarray [N_gen]  degree-distribution KL (↓ better)
        "Q"             np.ndarray [N_gen]  composite quality ∈ [0,1] (↑ better)
    """
    print("  [quality] computing SimCLR embedding distances …")
    emb_dist      = compute_embedding_distances(
                        gen_data, H_train_laund, encoder, device, k=k)

    print("  [quality] computing Wasserstein feature distances …")
    wass_dist     = compute_wasserstein_distances(gen_data, train_laund_data)
    wass_per_feat = compute_wasserstein_per_feature(gen_data, train_laund_data)

    print("  [quality] computing edge densities …")
    density       = compute_edge_densities(gen_data)

    print("  [quality] computing degree-distribution KL divergence …")
    degree_kl     = compute_degree_kl_divergence(gen_data, train_laund_data)

    # Composite score: 4 components, each normalised to [0,1] then inverted
    # (higher = better), then averaged equally.
    # degree_kl replaces nothing — it adds topology information absent from
    # the Wasserstein distance (which operates on node features, not graph structure).
    emb_score  = 1.0 - _normalize_01(emb_dist)
    wass_score = 1.0 - _normalize_01(wass_dist)
    dens_score = 1.0 - _normalize_01(density)
    kl_score   = 1.0 - _normalize_01(np.nan_to_num(degree_kl, nan=0.0))
    Q          = (emb_score + wass_score + dens_score + kl_score) / 4.0

    return {
        "emb_dist":      emb_dist,
        "wass_dist":     wass_dist,
        "wass_per_feat": wass_per_feat,
        "density":       density,
        "degree_kl":     degree_kl,
        "Q":             Q,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Reporting helpers
# ─────────────────────────────────────────────────────────────────────────────

def print_quality_report(metrics, top_n=None):
    """
    Print a ranked per-graph quality table plus aggregate summary.

    Parameters
    ----------
    metrics : dict returned by score_generated_graphs()
    top_n   : how many rows to show (None = all)
    """
    Q = metrics["Q"]
    n = len(Q)
    if n == 0:
        print("No generated graphs to score.")
        return

    top_n = n if top_n is None else min(top_n, n)
    rank  = np.argsort(-Q)      # descending by quality

    sep = "─" * 86
    print("\n" + sep)
    print("GENERATED GRAPH QUALITY REPORT  "
          "(Tier-1 metrics  |  Q: higher = better quality)")
    print(sep)
    print(f"{'Rank':>4}  {'Graph':>6}  {'Q':>6}  "
          f"{'EmbDist':>9}  {'WassDist':>9}  {'Density':>8}  {'DegKL':>8}")
    print(sep)

    has_kl = "degree_kl" in metrics and len(metrics["degree_kl"]) == n
    for pos in range(top_n):
        i = rank[pos]
        kl_str = f"{metrics['degree_kl'][i]:>8.4f}" if has_kl else "       —"
        print(f"{pos+1:>4}  {i:>6}  {Q[i]:>6.3f}  "
              f"{metrics['emb_dist'][i]:>9.4f}  "
              f"{metrics['wass_dist'][i]:>9.4f}  "
              f"{metrics['density'][i]:>8.4f}  "
              f"{kl_str}")

    if n > top_n:
        print(f"  … ({n - top_n} more graphs not shown, use top_n=None to see all)")

    print(sep)
    # Per-metric summary
    extra_keys = [("DegKL    ", "degree_kl")] if has_kl else []
    for label, key in [("Q        ", "Q"),
                        ("EmbDist  ", "emb_dist"),
                        ("WassDist ", "wass_dist"),
                        ("Density  ", "density")] + extra_keys:
        arr = metrics[key]
        print(f"  {label}  mean={arr.mean():.4f}  std={arr.std():.4f}  "
              f"min={arr.min():.4f}  max={arr.max():.4f}")

    # Per-feature Wasserstein breakdown
    wf = metrics["wass_per_feat"]
    if wf.size > 0:
        print(f"\n  Wasserstein breakdown by feature (mean over {n} graphs):")
        for fi, fname in enumerate(_FEAT_NAMES):
            print(f"    {fname:<14}  {wf[:, fi].mean():.4f}")

    print(sep)


def save_quality_csv(metrics, path):
    """
    Save per-graph quality metrics (including per-feature Wasserstein) to CSV.

    Parameters
    ----------
    metrics : dict returned by score_generated_graphs()
    path    : str or Path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    wf     = metrics["wass_per_feat"]
    n      = len(metrics["Q"])
    has_kl = "degree_kl" in metrics and len(metrics["degree_kl"]) == n

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        header = (["graph_idx", "Q", "emb_dist", "wass_dist", "density"]
                  + (["degree_kl"] if has_kl else [])
                  + [f"wass_{fn}" for fn in _FEAT_NAMES])
        writer.writerow(header)
        for i in range(n):
            per_feat_vals = list(wf[i]) if wf.size > 0 else [""] * len(_FEAT_NAMES)
            kl_val = [f"{metrics['degree_kl'][i]:.6f}"] if has_kl else []
            writer.writerow([
                i,
                f"{metrics['Q'][i]:.6f}",
                f"{metrics['emb_dist'][i]:.6f}",
                f"{metrics['wass_dist'][i]:.6f}",
                f"{metrics['density'][i]:.6f}",
                *kl_val,
                *[f"{v:.6f}" for v in per_feat_vals],
            ])
    print(f"Quality metrics saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Quality-extremes visualisation
# ─────────────────────────────────────────────────────────────────────────────

def plot_quality_extremes(metrics, gen_data, path, n=50):
    """
    Draw a grid of the top-n and bottom-n generated graphs ranked by Q score.

    Each cell shows the graph drawn with spring layout, coloured by
    laundering probability (node feature col 0: red=high, blue=low),
    annotated with Q / Wass / density.

    Parameters
    ----------
    metrics  : dict returned by score_generated_graphs()
    gen_data : list[PyG Data]  same order as metrics arrays
    path     : str or Path  output file (e.g. results/quality_extremes_ibm.png)
    n        : how many top / bottom graphs to show (default 50)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    Q    = metrics["Q"]
    rank = np.argsort(-Q)           # best → worst
    n    = min(n, len(Q))

    top_idx    = rank[:n]
    bottom_idx = rank[-n:][::-1]    # worst first → reverse so worst is last

    def _draw_cell(ax, data, q, wass, dens, title_prefix):
        ei  = data.edge_index.numpy()
        nf  = data.x.numpy()
        N   = data.num_nodes

        G = nx.Graph()
        G.add_nodes_from(range(N))
        for s, t in ei.T:
            if s < t:
                G.add_edge(int(s), int(t))

        # Node colour: col 0 = laundering probability
        lp = nf[:, 0] if nf.shape[1] > 0 else np.zeros(N)
        colors = [
            "#e74c3c" if lp[k] > 0.5 else "#aed6f1"
            for k in range(N)
        ]

        try:
            pos = nx.spring_layout(G, seed=42)
        except Exception:
            pos = {k: (k % 5, k // 5) for k in range(N)}

        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors,
                               node_size=30, alpha=0.9)
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#666",
                               width=0.5, alpha=0.4, arrows=False)
        ax.set_title(
            f"{title_prefix}\nQ={q:.3f}  W={wass:.3f}  d={dens:.3f}",
            fontsize=6, pad=2,
        )
        ax.axis("off")

    # Layout: two big groups side by side (top-n | bottom-n), each in a grid
    cols      = 10
    rows      = int(np.ceil(n / cols))
    fig_w     = cols * 2 * 2 + 1          # two groups × cols × cell width + gap
    fig_h     = rows * 2 + 0.8            # rows × cell height + header

    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.suptitle(
        f"Top {n} (left) vs Bottom {n} (right) generated graphs by Q score",
        fontsize=13, fontweight="bold", y=1.01,
    )

    # Build a gridspec with a small gap between the two halves
    import matplotlib.gridspec as gridspec
    outer = gridspec.GridSpec(1, 2, figure=fig, wspace=0.08)
    gs_top = gridspec.GridSpecFromSubplotSpec(rows, cols, subplot_spec=outer[0],
                                              hspace=0.6, wspace=0.3)
    gs_bot = gridspec.GridSpecFromSubplotSpec(rows, cols, subplot_spec=outer[1],
                                              hspace=0.6, wspace=0.3)

    for pos_in_group, idx in enumerate(top_idx):
        r, c = divmod(pos_in_group, cols)
        ax = fig.add_subplot(gs_top[r, c])
        d  = gen_data[idx]
        _draw_cell(ax, d,
                   metrics["Q"][idx],
                   metrics["wass_dist"][idx],
                   metrics["density"][idx],
                   f"#{pos_in_group+1} (g{idx})")

    for pos_in_group, idx in enumerate(bottom_idx):
        r, c = divmod(pos_in_group, cols)
        ax = fig.add_subplot(gs_bot[r, c])
        d  = gen_data[idx]
        _draw_cell(ax, d,
                   metrics["Q"][idx],
                   metrics["wass_dist"][idx],
                   metrics["density"][idx],
                   f"#{len(Q)-n+pos_in_group+1} (g{idx})")

    # Group labels
    for ax_group, label in [(fig.add_subplot(outer[0]), f"Top {n}  (Q ↑)"),
                             (fig.add_subplot(outer[1]), f"Bottom {n}  (Q ↓)")]:
        ax_group.set_title(label, fontsize=11, fontweight="bold", pad=8)
        ax_group.axis("off")

    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Quality extremes plot saved → {path}")
