"""
scoring.py — shared metric helpers used by test.py and latent_seed_generation.py.
"""

import csv
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.distance import mahalanobis
from torch_geometric.data import Data

FEAT_DIM  = 10
NOVELTY_K = 10


def graph_feature_vector(x, adj):
    """
    Compute a 10-D graph-level feature vector from node features + adjacency.
    Betweenness (x[:,3]) is excluded — the diffusion model does not predict it
    reliably and it was zeroed during SimCLR encoder training.

    x   : [n, 7]  node features in ORIGINAL scale
    adj : [n, n]  binary adjacency
    """
    if isinstance(x, torch.Tensor):
        x   = x.cpu().numpy()
        adj = adj.cpu().numpy()

    n     = x.shape[0]
    edges = float(adj.sum()) / 2
    dens  = edges / max(n * (n - 1) / 2, 1)
    deg   = adj.sum(axis=1)
    clust = x[:, 4]
    pr    = x[:, 5]
    assort = float(x[0, 5])

    return np.array([
        n,                       # 0  n_nodes
        edges,                   # 1  n_edges
        dens,                    # 2  density
        deg.mean(),              # 3  mean degree
        deg.std()   + 1e-8,      # 4  std degree
        clust.mean(),            # 5  mean clustering
        clust.std() + 1e-8,      # 6  std clustering
        pr.mean(),               # 7  mean pagerank
        pr.std()    + 1e-8,      # 8  std pagerank
        assort,                  # 9  assortativity
    ], dtype=np.float64)


def fit_training_distribution(networks):
    """
    Fit a multivariate Gaussian over graph-level features of all training networks.
    Returns (mu, cov_inv, realism_scale).
    """
    from diffusion.diff_util import network_to_dense as _ntd
    print("Computing training graph-feature distribution …")
    feats = []
    for net in networks:
        if "x_dense" in net and "adj_dense" in net:
            fv = graph_feature_vector(net["x_dense"], net["adj_dense"])
        else:
            _x, _adj = _ntd(net)
            fv = graph_feature_vector(_x, _adj)
        feats.append(fv)

    F_train = np.stack(feats, axis=0)                          # [N, FEAT_DIM]
    mu      = F_train.mean(axis=0)
    cov     = np.cov(F_train.T) + np.eye(FEAT_DIM) * 1e-4
    cov_inv = np.linalg.inv(cov)

    train_mah     = np.array([mahalanobis(fv, mu, cov_inv) for fv in F_train])
    mah_p95       = np.percentile(train_mah, 95)
    realism_scale = mah_p95 * 2.0
    print(f"  Mahalanobis  mean={train_mah.mean():.2f}  "
          f"p95={mah_p95:.2f}  scale={realism_scale:.2f}")
    return mu, cov_inv, realism_scale


def score_network(x, adj, encoder, H_all_n, device, mu, cov_inv, realism_scale,
                  gen_embedding=None):
    """
    Score a single network on realism (Mahalanobis) and novelty (cosine distance).

    Returns a dict with keys: score, realism, novelty, mah_distance,
                               max_cosine_sim, mean_top10_sim.
    """
    fv      = graph_feature_vector(x, adj)
    mah     = mahalanobis(fv, mu, cov_inv)
    realism = float(np.exp(-mah / realism_scale))

    if gen_embedding is None:
        xt = x   if isinstance(x,   torch.Tensor) else torch.tensor(x,   dtype=torch.float32)
        at = adj if isinstance(adj, torch.Tensor) else torch.tensor(adj, dtype=torch.float32)
        n  = xt.shape[0]
        x_pyg       = xt.clone().float()
        deg_g       = at.sum(dim=-1).float()
        x_pyg[:, 1] = deg_g / deg_g.max().clamp(min=1.0)
        ei = (at > 0.5).nonzero(as_tuple=False).T.contiguous()
        if ei.shape[1] == 0:
            ei = torch.zeros(2, 0, dtype=torch.long)
        bv = torch.zeros(n, dtype=torch.long)
        # Strip col 0 (laundering flag) — encoder expects features without label
        x_enc = x_pyg[:, 1:] if x_pyg.shape[1] > 1 else x_pyg
        with torch.no_grad():
            h = encoder(Data(x=x_enc, edge_index=ei, batch=bv).to(device)).cpu()
            gen_embedding = F.normalize(h, dim=-1).squeeze(0)

    if isinstance(gen_embedding, torch.Tensor):
        gen_embedding = gen_embedding.cpu().numpy()

    cos_sims    = H_all_n.numpy() @ gen_embedding          # [N_train]
    top_k_sims  = np.sort(cos_sims)[-NOVELTY_K:][::-1]    # K highest
    mean_top_k  = float(top_k_sims.mean())
    max_cos_sim = float(cos_sims.max())
    novelty     = float(1.0 - mean_top_k)

    return {
        "score":          round(realism * novelty, 4),
        "realism":        round(realism,            4),
        "novelty":        round(novelty,            4),
        "mah_distance":   round(mah,                3),
        "max_cosine_sim": round(max_cos_sim,         4),
        "mean_top10_sim": round(mean_top_k,          4),
    }


def _print_scores_table(scores, header):
    print(f"\n── {header} " + "─" * max(0, 68 - len(header)))
    print(f"{'#':>3}  {'score':>7}  {'realism':>9}  {'novelty':>9}"
          f"  {'mah_dist':>9}  {'max_sim':>8}  {'mean_top10':>10}")
    print("─" * 70)
    for i, s in enumerate(scores):
        print(f"{i+1:>3}  {s['score']:>7.4f}  {s['realism']:>9.4f}  "
              f"{s['novelty']:>9.4f}  {s['mah_distance']:>9.3f}  "
              f"{s['max_cosine_sim']:>8.4f}  {s['mean_top10_sim']:>10.4f}")


def _save_scores_csv(scores, path, label):
    fieldnames = ["#", "label", "score", "realism", "novelty",
                  "mah_distance", "max_cosine_sim", "mean_top10_sim"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for i, s in enumerate(scores):
            w.writerow({"#": i + 1, "label": label, **s})
    print(f"Saved → {path}")
