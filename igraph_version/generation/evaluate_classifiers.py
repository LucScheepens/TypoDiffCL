"""
evaluate_classifiers.py
────────────────────────────────────────────────────────────────────────────
Compare four graph classifiers with fundamentally different inductive biases
on the subgraph-level AML task, with and without diffusion-generated
training data augmentation.

Models
──────
  GIN               — Graph Isomorphism Network (Xu et al. 2019)
                      Local SUM aggregation; provably as powerful as WL.
                      Topology-heavy: the generated edge structure matters.
                      EXPECT: augmentation hurts — sum scales with density.

  GraphTransformer  — Transformer-style attention along edges (Shi et al. 2021)
                      Multi-head attention weights each neighbour differently.
                      Residual + LayerNorm stabilises noisy edges.
                      EXPECT: augmentation nearly neutral — attention suppresses
                      spurious edges.

  GraphSAGE         — Inductive mean aggregation (Hamilton et al. 2017)
                      Mean-normalises over neighbours, so doubling edges does
                      NOT double activations (unlike GIN SUM).
                      EXPECT: augmentation mildly positive — degree-normalised
                      aggregation tolerates dense generated adj better.

  DeepSets          — No message passing; MLP on mean+max-pooled node features
                      (Zaheer et al. 2017, graph-level variant).
                      Completely ignores edge structure — only node features
                      (degree, betweenness, clustering, PageRank, assortativity)
                      are used.  Node features are the well-generated part of
                      the diffusion output.
                      EXPECT: augmentation helps — topology noise is irrelevant;
                      the generated feature distributions add genuine diversity.

Conditions
──────────
  baseline   — real training networks only
  augmented  — real training networks + diffusion+SimCLR generated laundering
               networks (requires trained diffusion and SimCLR models)

  --low-data FRAC  subsample the training set to FRAC of its original size
               (e.g. 0.2 = 20 %).  Augmentation benefit is most visible when
               labelled data is scarce.

Input node features (5, structural — laundering label excluded to prevent leakage)
  degree, betweenness, clustering, PageRank, assortativity

Graph label
  1 = network contains at least one laundering node
  0 = clean network
  All generated networks are labelled 1.

Usage
─────
  # full data, baseline only
  python evaluate_classifiers.py

  # full data, with augmentation
  python evaluate_classifiers.py --augment --n-gen 40

  # low-data regime (20 % training), with augmentation
  python evaluate_classifiers.py --augment --n-gen 40 --low-data 0.2
"""

import sys
import argparse
import csv
import functools
import pickle
import random
from pathlib import Path

import pandas as pd

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, TransformerConv, SAGEConv
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedShuffleSplit

# ── path setup ────────────────────────────────────────────────────────────────

_GEN_DIR   = Path(__file__).resolve().parent   # igraph_version/generation/
ROOT_DIR   = _GEN_DIR.parent                   # igraph_version/
DIFF_DIR   = ROOT_DIR / "diffusion"
SIMCLR_DIR = ROOT_DIR / "simclr"
CKPT_DIR   = ROOT_DIR / "checkpoints"
DATA_DIR   = ROOT_DIR / "data"

for _p in (str(ROOT_DIR), str(DIFF_DIR), str(SIMCLR_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from util import preprocess_df, extract_networks_igraph, extract_transaction_ego_networks
from augmentation import build_igraph_from_transactions

# ── hyper-parameters ──────────────────────────────────────────────────────────
CSV_PATH    = r"C:\Users\lucsc\Thesis\grad\grad\data\IBM\LI-Small_Trans.csv"  # default
IN_CHANNELS        = 10       # topology (5) + transaction features (5)
FRAUDGT_IN_CHANNELS = IN_CHANNELS + 1   # +1 for ego_id feature
HIDDEN      = 64
NUM_LAYERS  = 3
HEADS       = 4        # attention heads for GraphTransformer
DROPOUT     = 0.3
LR          = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS      = 80
BATCH_SIZE  = 32
N_RUNS      = 3        # seeds averaged to produce mean ± std
TEST_FRAC   = 0.20
VAL_FRAC    = 0.10


# ─────────────────────────────────────────────────────────────────────────────
# Model 1 — GIN  (local sum aggregation)
# ─────────────────────────────────────────────────────────────────────────────
class GINClassifier(nn.Module):
    """
    Graph Isomorphism Network for graph-level classification.

    Key properties
    - SUM aggregation: captures multiset of neighbourhood features
    - train_eps: learnable self-loop weight
    - BatchNorm between layers for training stability
    - Global SUM readout: sensitive to graph size and total feature mass

    Why it differs from GraphTransformer
    GIN treats each node's neighbourhood uniformly — every neighbour
    contributes equally to the aggregate.  It is maximally sensitive to
    local topology (isomorphism-complete for finite graphs) but cannot
    weight neighbours by relevance or capture distant interactions without
    deep stacking.
    """
    def __init__(self, in_channels=IN_CHANNELS, hidden=HIDDEN,
                 num_layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        for i in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(in_channels if i == 0 else hidden, hidden),
                nn.BatchNorm1d(hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
            )
            self.convs.append(GINConv(mlp, train_eps=True))
            self.bns.append(nn.BatchNorm1d(hidden))
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 2),
        )

    def forward(self, x, edge_index, batch):
        for conv, bn in zip(self.convs, self.bns):
            x = bn(conv(x, edge_index)).relu()
        x = global_add_pool(x, batch)
        return self.head(x)


# ─────────────────────────────────────────────────────────────────────────────
# Model 2 — Graph Transformer  (multi-head attention along edges)
# ─────────────────────────────────────────────────────────────────────────────
class GraphTransformerClassifier(nn.Module):
    """
    Graph Transformer for graph-level classification.

    Key properties
    - Multi-head attention along edges: each neighbour is weighted by
      learned query-key compatibility, not treated uniformly
    - Residual + LayerNorm after every layer (transformer-style stability)
    - Global MEAN readout: less sensitive to graph size than SUM

    Why it differs from GIN
    The attention mechanism allows the model to dynamically suppress
    irrelevant neighbours and amplify informative ones.  This is especially
    useful for AML where a few central laundering nodes matter far more
    than peripheral participants.  Unlike GIN's fixed aggregation,
    attention weights change per-input, giving the model an implicit
    node-importance ranking that GIN cannot learn.
    """
    def __init__(self, in_channels=IN_CHANNELS, hidden=HIDDEN,
                 num_layers=NUM_LAYERS, heads=HEADS, dropout=DROPOUT):
        super().__init__()
        assert hidden % heads == 0, "hidden must be divisible by heads"
        head_dim = hidden // heads
        self.input_proj = nn.Linear(in_channels, hidden)
        self.convs = nn.ModuleList([
            TransformerConv(hidden, head_dim, heads=heads,
                            dropout=dropout, concat=True)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(num_layers)])
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 2),
        )

    def forward(self, x, edge_index, batch):
        x = self.input_proj(x).relu()
        for conv, norm in zip(self.convs, self.norms):
            x = norm(conv(x, edge_index) + x)      # residual + LayerNorm
        x = global_mean_pool(x, batch)
        return self.head(x)


# ─────────────────────────────────────────────────────────────────────────────
# Model 3 — GraphSAGE  (degree-normalised mean aggregation)
# ─────────────────────────────────────────────────────────────────────────────
class GraphSAGEClassifier(nn.Module):
    """
    GraphSAGE for graph-level classification.

    Key properties
    - MEAN aggregation with degree normalisation: h = W * mean(neighbours)
      Doubling edge count does NOT double activations — unlike GIN SUM.
    - Concatenates self + aggregated neighbour into the update (SAGEConv default)
    - BatchNorm + ReLU between layers
    - Global MEAN readout

    Why it should tolerate generated data better than GIN
    The dense adjacency produced by the current diffusion model means that
    in generated graphs almost every pair of nodes is connected.  GIN's SUM
    pooling makes every node representation scale linearly with the number of
    neighbours, which completely distorts the embedding space.  SAGE's MEAN
    pooling produces roughly the same value regardless of whether 5 or 50
    neighbours exist — the representation is dominated by the average feature
    of the neighbourhood rather than its size.  This robustness to density
    means SAGE can still extract a meaningful laundering signal from the
    generated node features even when the adjacency is near-complete.
    """
    def __init__(self, in_channels=IN_CHANNELS, hidden=HIDDEN,
                 num_layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(SAGEConv(in_channels if i == 0 else hidden, hidden))
            self.bns.append(nn.BatchNorm1d(hidden))
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 2),
        )

    def forward(self, x, edge_index, batch):
        for conv, bn in zip(self.convs, self.bns):
            x = bn(conv(x, edge_index)).relu()
        x = global_mean_pool(x, batch)
        return self.head(x)


# ─────────────────────────────────────────────────────────────────────────────
# Model 4 — DeepSets  (no message passing — node features only)
# ─────────────────────────────────────────────────────────────────────────────
class DeepSetsClassifier(nn.Module):
    """
    DeepSets-style graph classifier — no GNN, no edge structure.

    Key properties
    - Applies a shared per-node MLP φ to each node independently
    - Reads out the graph as concat(mean, max) of transformed node features
    - Applies a second MLP ρ to the graph-level representation
    - Edge index is accepted in forward() for API compatibility but IGNORED

    Why it should benefit most from augmentation
    The diffusion model generates node features (degree, betweenness,
    clustering, PageRank, assortativity) that closely match the real
    laundering distribution after the clipping fix.  DeepSets is the only
    model here that uses those features exclusively — the dense/unrealistic
    adjacency has zero effect on its predictions.  Every generated graph
    therefore contributes a genuine laundering-like feature distribution to
    training, which should improve generalisation especially in the low-data
    regime.  No other model in this comparison isolates the feature signal
    so cleanly.
    """
    def __init__(self, in_channels=IN_CHANNELS, hidden=HIDDEN, dropout=DROPOUT):
        super().__init__()
        # φ: per-node embedding
        self.phi = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        # ρ: graph-level classifier (input = mean || max → 2*hidden)
        self.rho = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

    def forward(self, x, _edge_index, batch):   # edge_index intentionally unused
        h    = self.phi(x)
        h_m  = global_mean_pool(h, batch)       # [B, hidden]
        h_mx = global_max_pool(h, batch)        # [B, hidden]
        return self.rho(torch.cat([h_m, h_mx], dim=-1))


# ─────────────────────────────────────────────────────────────────────────────
# Model 5 — FraudGT  (bidirectional Graph Transformer + ego ID)
# ─────────────────────────────────────────────────────────────────────────────
class FraudGTClassifier(nn.Module):
    """
    FraudGT-inspired Graph Transformer (Ju et al., ACM FinSys 2024).

    Core contributions adapted for subgraph-level AML classification:

    1. Bidirectional message passing — separate TransformerConv streams for
       forward (A→B) and reverse (B→A) directions, combined per layer via a
       learned linear projection.  Money flows follow directed patterns;
       separating directions lets the model distinguish sender/receiver roles.

    2. Ego ID — a binary node feature (col −1 of x) marking the focal account
       at the centre of the ego subgraph.  Lets the model weight whether the
       hub node IS the suspicious account or merely a transit node.

    3. Mean+Max readout — captures both average connectivity and the most
       extreme node embedding (the potential hub launderer).

    Difference from existing GraphTransformerClassifier:
    - Two separate conv streams (fwd / rev) instead of one
    - Streams combined with a linear projection + residual before LayerNorm
    - Extra ego_id input feature (in_channels = IN_CHANNELS + 1)
    """
    def __init__(self, in_channels=FRAUDGT_IN_CHANNELS, hidden=HIDDEN,
                 num_layers=NUM_LAYERS, heads=HEADS, dropout=DROPOUT):
        super().__init__()
        assert hidden % heads == 0, "hidden must be divisible by heads"
        head_dim = hidden // heads
        self.input_proj = nn.Linear(in_channels, hidden)
        self.fwd_convs = nn.ModuleList([
            TransformerConv(hidden, head_dim, heads=heads, dropout=dropout, concat=True)
            for _ in range(num_layers)
        ])
        self.rev_convs = nn.ModuleList([
            TransformerConv(hidden, head_dim, heads=heads, dropout=dropout, concat=True)
            for _ in range(num_layers)
        ])
        self.combines = nn.ModuleList([
            nn.Linear(hidden * 2, hidden) for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(num_layers)])
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

    def forward(self, x, edge_index, batch):
        rev_ei = edge_index[[1, 0]]          # flip src/dst for reverse pass
        x = self.input_proj(x).relu()
        for fwd, rev, combine, norm in zip(
                self.fwd_convs, self.rev_convs, self.combines, self.norms):
            msg = combine(torch.cat([fwd(x, edge_index), rev(x, rev_ei)], dim=-1))
            x   = norm(msg + x)
        h = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=-1)
        return self.head(h)


# ─────────────────────────────────────────────────────────────────────────────
# Data conversion helpers
# ─────────────────────────────────────────────────────────────────────────────
def _adj_to_edge_index(adj):
    """Dense [N, N] adjacency (tensor or ndarray) → PyG edge_index [2, E]."""
    if isinstance(adj, torch.Tensor):
        adj = adj.numpy()
    src, dst = np.where(adj > 0.5)
    return torch.tensor(np.stack([src, dst]), dtype=torch.long)


def _fallback_edges(n):
    """Self-loops for isolated graphs so every GNN layer can operate."""
    idx = torch.arange(n)
    return torch.stack([idx, idx])


def network_to_pyg(x_dense, adj_dense, label):
    """
    Convert a dense (x, adj) pair from network_to_dense() to a PyG Data object.

    x_dense[:, 0]  = laundering label  — EXCLUDED from node features
    x_dense[:, 1:] = structural features — used as input
    """
    if isinstance(x_dense, torch.Tensor):
        x_np   = x_dense[:, 1:].float().numpy()
        adj_np = adj_dense.numpy() if isinstance(adj_dense, torch.Tensor) else adj_dense
    else:
        x_np   = x_dense[:, 1:].astype(np.float32)
        adj_np = adj_dense

    ei = _adj_to_edge_index(adj_np)
    if ei.shape[1] == 0:
        ei = _fallback_edges(x_np.shape[0])

    return Data(
        x             = torch.tensor(x_np, dtype=torch.float),
        edge_index    = ei,
        y             = torch.tensor([label], dtype=torch.long),
        timestep      = torch.tensor([-1], dtype=torch.long),
        timestamp_val = -1.0,
    )


def gen_output_to_pyg(x_denorm, adj_out, n_out, label=1):
    """
    Convert a diffusion-generated triple to a PyG Data object.

    Node features and adjacency are used exactly as produced by the diffusion
    model — no post-hoc recomputation.  The model is responsible for learning
    to generate internally consistent (feature, topology) pairs.  Any gap
    between predicted features and the actual adjacency is a signal that the
    generation process needs improvement, not something to paper over here.

    # ── AUGMENTATION IMPROVEMENT DIRECTIONS ──────────────────────────────────
    #
    # The core challenge: the diffusion model must jointly learn to generate
    # realistic node features AND meaningful graph topology.  Current pain points
    # and concrete directions to address them:
    #
    # 1. FEATURE-TOPOLOGY COUPLING (highest priority)
    #    Problem: the model generates features and adjacency largely independently,
    #    so e.g. a node predicted to have high PageRank may sit in a low-degree
    #    position in the generated adj — an impossible combination in real data.
    #    Direction: add a consistency regularisation term to the diffusion training
    #    loss that penalises divergence between predicted structural features and
    #    the features implied by the generated adjacency (e.g. soft L2 on
    #    recomputed degree/clustering vs predicted degree/clustering).
    #
    # 2. TOPOLOGY-AWARE DIFFUSION TRAINING
    #    Problem: the diffusion model is trained with a generic MSE loss on
    #    (x, adj) jointly, giving equal weight to all elements of the adjacency
    #    matrix even though most entries are 0 in sparse real graphs.
    #    Direction: weight the adjacency reconstruction loss by edge rarity
    #    (class-balanced BCE on adj entries), or use a graph-structured denoising
    #    objective (e.g. GDSS, NVDiff) that directly models the adjacency as a
    #    discrete structure rather than a continuous density matrix.
    #
    # 3. CLASS-CONDITIONAL GENERATION QUALITY
    #    Problem: guidance is applied via a probe trained on SimCLR embeddings,
    #    but the probe signal is weak if the encoder has not learned to separate
    #    laundering vs clean graphs in embedding space.
    #    Direction: evaluate embedding-space class separation (e.g. silhouette
    #    score, linear probe accuracy) before generation; use that as a diagnostic
    #    for when SimCLR needs retraining.  Consider hard-negative mining or a
    #    supervised contrastive loss (SupCon) to improve class separation.
    #
    # 4. AUGMENTATION AS A LEARNABLE POLICY (long-term)
    #    Problem: the current approach generates graphs and then selects useful
    #    ones — but the generator never receives a signal about *which* generated
    #    graphs were actually useful for the downstream classifier.
    #    Direction: bilevel optimisation — treat the generation parameters
    #    (guidance scale, novelty weight) as meta-parameters and optimise them
    #    to maximise downstream val F1 via a differentiable proxy or Bayesian
    #    optimisation loop over the (guidance_scale, novelty_weight) space.
    #
    # 5. GRAPH STRUCTURE DISTRIBUTION MATCHING
    #    Problem: generated graphs are often too dense (near-complete adjacency),
    #    which makes GNN aggregation behave very differently from real sparse graphs.
    #    Direction: add a degree-distribution regulariser during generation (already
    #    partly done via degree_penalty); measure KL divergence between real and
    #    generated degree distributions and report it as a generation quality metric
    #    alongside the existing Q-score.
    # ─────────────────────────────────────────────────────────────────────────
    """
    x_np = x_denorm[:n_out].float().numpy().copy()

    adj_np = adj_out[:n_out, :n_out]
    adj_np = adj_np.numpy() if isinstance(adj_np, torch.Tensor) else np.array(adj_np, dtype=float)

    # Symmetrise and threshold the generated adjacency for edge_index extraction.
    # The model output is a continuous density — treat values > 0.5 as edges.
    adj_s = np.clip(adj_np + adj_np.T, 0, 1).astype(float)
    np.fill_diagonal(adj_s, 0)

    ei = _adj_to_edge_index(adj_s)
    if ei.shape[1] == 0:
        ei = _fallback_edges(n_out)

    return Data(
        x             = torch.tensor(x_np, dtype=torch.float),
        edge_index    = ei,
        y             = torch.tensor([label], dtype=torch.long),
        timestep      = torch.tensor([-1], dtype=torch.long),
        timestamp_val = -1.0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# FraudGT data helpers
# ─────────────────────────────────────────────────────────────────────────────

def network_to_pyg_fraudgt(x_dense, adj_dense, label, net):
    """
    Like network_to_pyg but appends an ego_id feature (last column of x).

    ego_id = 1 for the focal/root account (net["start_node"]), 0 for all
    other nodes.  Falls back to highest-degree node when start_node is absent.
    """
    if isinstance(x_dense, torch.Tensor):
        x_np   = x_dense[:, 1:].float().numpy()
        adj_np = adj_dense.numpy() if isinstance(adj_dense, torch.Tensor) else adj_dense
    else:
        x_np   = x_dense[:, 1:].astype(np.float32)
        adj_np = np.asarray(adj_dense, dtype=np.float32)

    n      = x_np.shape[0]
    ego_id = np.zeros((n, 1), dtype=np.float32)

    start_node = net.get("start_node", None)
    g          = net.get("graph", None)
    if start_node is not None and g is not None and "name" in g.vs.attributes():
        names = [int(g.vs[i]["name"]) for i in range(min(n, g.vcount()))]
        if start_node in names:
            ego_id[names.index(start_node)] = 1.0
        else:
            ego_id[int(np.argmax(adj_np.sum(axis=1)))] = 1.0
    elif n > 0:
        ego_id[int(np.argmax(adj_np.sum(axis=1)))] = 1.0

    ei = _adj_to_edge_index(adj_np)
    if ei.shape[1] == 0:
        ei = _fallback_edges(n)

    return Data(
        x             = torch.tensor(np.concatenate([x_np, ego_id], axis=1), dtype=torch.float),
        edge_index    = ei,
        y             = torch.tensor([label], dtype=torch.long),
        timestep      = torch.tensor([-1], dtype=torch.long),
        timestamp_val = -1.0,
    )


def _to_fraudgt_format(data):
    """Append ego_id=0 column to a standard Data object (for generated graphs)."""
    n = data.x.shape[0]
    return Data(
        x             = torch.cat([data.x, torch.zeros(n, 1)], dim=1),
        edge_index    = data.edge_index,
        y             = data.y,
        timestep      = getattr(data, "timestep",      torch.tensor([-1])),
        timestamp_val = getattr(data, "timestamp_val", -1.0),
    )


# ─────────────────────────────────────────────────────────────────────────────
# ExSTraQt — feature extraction + sklearn classifier
# ─────────────────────────────────────────────────────────────────────────────

_EXSTRAQT_FEAT_NAMES = [
    # Account role counts (Tariq et al. §3.2)
    "num_sources", "num_targets", "num_passthrough", "num_transactions",
    # Transaction amount statistics
    "amount_mean", "amount_max", "amount_std", "amount_median",
    "log_amount_mean", "log_amount_std",
    # Temporal spread
    "ts_range_hours", "ts_std_hours",
    # Net money flow imbalance
    "turnover",
    # Graph topology
    "num_nodes", "num_edges", "max_degree", "mean_degree", "density",
    "assortativity", "num_biconn", "num_articulation_points",
]


def _extract_exstraqt_features(net):
    """
    Compute ExSTraQt-style feature vector for one IBM network dict.

    Mirrors the feature set described in Tariq et al. (arXiv 2604.02899):
    account role fractions, transaction amount/temporal statistics, turnover,
    and graph topology metrics.  Returns a fixed-length float32 array.
    """
    feat = dict.fromkeys(_EXSTRAQT_FEAT_NAMES, 0.0)
    txs  = net.get("transactions", None)
    g    = net.get("graph", None)

    if txs is not None and len(txs) > 0 and "From_Account_int" in txs.columns:
        srcs = set(txs["From_Account_int"])
        dsts = set(txs["To_Account_int"])
        both = srcs & dsts
        feat["num_sources"]      = float(len(srcs - dsts))
        feat["num_targets"]      = float(len(dsts - srcs))
        feat["num_passthrough"]  = float(len(both))
        feat["num_transactions"] = float(len(txs))

        amt_col = next((c for c in ("Amount_Paid", "amount", "Amount") if c in txs.columns), None)
        if amt_col:
            amounts = np.abs(txs[amt_col].values.astype(float)) + 1e-8
            feat["amount_mean"]     = float(np.mean(amounts))
            feat["amount_max"]      = float(np.max(amounts))
            feat["amount_std"]      = float(np.std(amounts))
            feat["amount_median"]   = float(np.median(amounts))
            log_a = np.log1p(amounts)
            feat["log_amount_mean"] = float(np.mean(log_a))
            feat["log_amount_std"]  = float(np.std(log_a))

            out_sum = txs.groupby("From_Account_int")[amt_col].sum()
            in_sum  = txs.groupby("To_Account_int")[amt_col].sum()
            all_acc = set(out_sum.index) | set(in_sum.index)
            feat["turnover"] = float(
                np.sum([abs(out_sum.get(a, 0.0) - in_sum.get(a, 0.0)) for a in all_acc])
            )

        ts_col = next((c for c in ("Timestamp", "timestamp") if c in txs.columns), None)
        if ts_col:
            try:
                ts = pd.to_datetime(txs[ts_col]).astype("int64") / 1e9
                feat["ts_range_hours"] = float((ts.max() - ts.min()) / 3600.0)
                feat["ts_std_hours"]   = float(ts.std() / 3600.0)
            except Exception:
                pass

    if g is not None:
        n    = g.vcount()
        degs = g.degree()
        feat["num_nodes"]   = float(n)
        feat["num_edges"]   = float(g.ecount())
        feat["max_degree"]  = float(max(degs)) if degs else 0.0
        feat["mean_degree"] = float(np.mean(degs)) if degs else 0.0
        feat["density"]     = float(g.density()) if n > 1 else 0.0
        a = g.assortativity_degree(directed=False)
        feat["assortativity"] = 0.0 if (a is None or np.isnan(float(a))) else float(a)
        try:
            feat["num_biconn"] = float(len(list(g.biconnected_components())))
        except Exception:
            feat["num_biconn"] = 1.0
        try:
            feat["num_articulation_points"] = float(len(g.articulation_points()))
        except Exception:
            pass

    return np.array([feat[k] for k in _EXSTRAQT_FEAT_NAMES], dtype=np.float32)


def _get_net_label(net):
    """Extract binary laundering label from a network dict."""
    if "tx_label" in net:
        return int(net["tx_label"])
    return int(len(net.get("laundering_nodes", set())) > 0)


def run_experiment_exstraqt(train_nets, val_nets, test_nets, seed=0):
    """
    Train and evaluate an ExSTraQt-style gradient-boosted classifier.

    Uses XGBoost if available, otherwise sklearn HistGradientBoostingClassifier.
    Threshold is tuned on the validation set (same protocol as GNN experiments).

    Returns a metrics dict with keys: auc, f1, precision, recall.
    """
    X_train = np.stack([_extract_exstraqt_features(n) for n in train_nets])
    y_train = np.array([_get_net_label(n) for n in train_nets])
    X_val   = np.stack([_extract_exstraqt_features(n) for n in val_nets])
    y_val   = np.array([_get_net_label(n) for n in val_nets])
    X_test  = np.stack([_extract_exstraqt_features(n) for n in test_nets])
    y_test  = np.array([_get_net_label(n) for n in test_nets])

    n_pos       = int(y_train.sum())
    n_neg       = len(y_train) - n_pos
    scale_pos   = max(1.0, n_neg / max(n_pos, 1))

    try:
        import xgboost as xgb
        clf = xgb.XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale_pos, random_state=seed,
            eval_metric="logloss", verbosity=0,
        )
        clf.fit(X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=25, verbose=False)
        _clf_name = "XGBoost"
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        clf = HistGradientBoostingClassifier(
            max_iter=400, max_depth=5, learning_rate=0.05,
            class_weight={0: 1.0, 1: scale_pos},
            random_state=seed,
            early_stopping=True, validation_fraction=0.1, n_iter_no_change=25,
        )
        clf.fit(X_train, y_train)
        _clf_name = "HistGBT"

    val_probs  = clf.predict_proba(X_val)[:, 1]
    best_thresh, _ = _best_f1_threshold(y_val, val_probs)

    test_probs = clf.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= best_thresh).astype(int)

    auc  = roc_auc_score(y_test, test_probs) if len(np.unique(y_test)) > 1 else 0.5
    f1   = f1_score(y_test, test_preds, average="binary", zero_division=0)
    prec = precision_score(y_test, test_preds, average="binary", zero_division=0)
    rec  = recall_score(y_test, test_preds, average="binary", zero_division=0)
    return {"auc": auc, "f1": f1, "precision": prec, "recall": rec,
            "_clf": _clf_name}


# ─────────────────────────────────────────────────────────────────────────────
# Training / evaluation
# ─────────────────────────────────────────────────────────────────────────────

def _focal_loss(logits, targets, weight=None, gamma=2.0):
    """
    Focal loss: FL(p) = -α(1-p)^γ log(p).
    Downweights easy negatives more aggressively than weighted CE,
    which helps when positives are rare (8-11% here).
    `weight` is the per-class weight tensor (same as F.cross_entropy weight).
    """
    ce   = F.cross_entropy(logits, targets, weight=weight, reduction="none")
    probs = F.softmax(logits, dim=-1)
    pt   = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    return ((1 - pt) ** gamma * ce).mean()


def _train_epoch(model, loader, optimizer, device, pos_weight=None, focal=False):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch.x, batch.edge_index, batch.batch)
        if focal:
            loss = _focal_loss(logits, batch.y, weight=pos_weight)
        else:
            loss = F.cross_entropy(logits, batch.y, weight=pos_weight)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)


@torch.no_grad()
def _collect_probs(model, loader, device):
    """Return (labels, probs) arrays without computing any metrics."""
    model.eval()
    all_labels, all_probs = [], []
    for batch in loader:
        batch  = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.batch)
        probs  = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        all_labels.extend(batch.y.cpu().numpy())
        all_probs.extend(probs)
    return np.array(all_labels), np.array(all_probs)


def _best_f1_threshold(labels, probs, thresholds=None):
    """
    Sweep decision thresholds and return the one maximising F1 on the
    given labels/probs arrays.  Thresholds default to 50 values in [0.05, 0.95].
    """
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 50)
    best_t, best_f1 = 0.5, 0.0
    for t in thresholds:
        preds = (probs >= t).astype(int)
        f = f1_score(labels, preds, average="binary", zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t
    return best_t, best_f1


def _evaluate(model, loader, device, threshold=0.5):
    """Compute metrics at a fixed threshold (use after threshold is tuned on val)."""
    labels, probs = _collect_probs(model, loader, device)
    preds = (probs >= threshold).astype(int)
    auc  = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.5
    f1   = f1_score(labels, preds, average="binary", zero_division=0)
    prec = precision_score(labels, preds, average="binary", zero_division=0)
    rec  = recall_score(labels, preds, average="binary", zero_division=0)
    return {"auc": auc, "f1": f1, "precision": prec, "recall": rec}


_PROXY_MAX_TRAIN = 2000   # cap proxy training set size to keep greedy selection fast

def _proxy_val_f1(train_data, val_data, device, epochs=20, seed=0,
                  focal=False, n_seeds=1):
    """
    Train `n_seeds` small GIN proxies for `epochs` epochs each and return
    the mean val F1 (at the per-run tuned threshold).

    train_data is subsampled to _PROXY_MAX_TRAIN graphs (stratified) so that
    each proxy run stays fast even on large datasets — the signal for whether
    one extra graph helps is the same regardless of training set size.
    """
    # Stratified subsample to keep proxy fast
    rng = np.random.default_rng(seed)
    if len(train_data) > _PROXY_MAX_TRAIN:
        pos_idx = [i for i, d in enumerate(train_data) if d.y.item() == 1]
        neg_idx = [i for i, d in enumerate(train_data) if d.y.item() == 0]
        n_pos_keep = max(1, int(_PROXY_MAX_TRAIN * len(pos_idx) / len(train_data)))
        n_neg_keep = _PROXY_MAX_TRAIN - n_pos_keep
        keep = (list(rng.choice(pos_idx, min(n_pos_keep, len(pos_idx)), replace=False)) +
                list(rng.choice(neg_idx, min(n_neg_keep, len(neg_idx)), replace=False)))
        proxy_train = [train_data[i] for i in keep]
    else:
        proxy_train = train_data

    f1_scores = []
    for s in range(n_seeds):
        torch.manual_seed(seed * 100 + s)
        np.random.seed(seed * 100 + s)

        proxy = GINClassifier(hidden=32, num_layers=2, dropout=0.0).to(device)
        opt   = torch.optim.Adam(proxy.parameters(), lr=1e-3)

        n_pos = sum(d.y.item() == 1 for d in proxy_train)
        n_neg = len(proxy_train) - n_pos
        pw    = (torch.tensor([1.0, n_neg / n_pos], dtype=torch.float, device=device)
                 if n_pos > 0 and n_neg > 0 else None)

        tr_loader  = DataLoader(proxy_train, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_data,    batch_size=BATCH_SIZE)

        for _ in range(epochs):
            _train_epoch(proxy, tr_loader, opt, device, pos_weight=pw, focal=focal)

        labels, probs = _collect_probs(proxy, val_loader, device)
        _, f1         = _best_f1_threshold(labels, probs)
        f1_scores.append(f1)

    return float(np.mean(f1_scores))


def greedy_select_generated(gen_data, train_data, val_data, device,
                            proxy_epochs=20, min_delta=0.0, focal=False,
                            proxy_seeds=3):
    """
    Greedily select the subset of generated graphs that maximally improves
    validation F1 (at tuned threshold) of a fast proxy GIN.

    Each proxy score is the mean over `proxy_seeds` independent runs to reduce
    variance — without this, the ranking is dominated by noise from a single run.

    Algorithm
    ---------
    1. Train proxy_seeds proxies on `train_data` alone → mean baseline val F1.
    2. For each generated graph, train proxy_seeds proxies on train_data + {graph}
       and record the mean val F1 gain over baseline.
    3. Sort candidates by gain (descending).  Keep all graphs whose
       gain > min_delta (default 0 = keep anything that doesn't hurt).
    4. Return the selected subset and a list of (graph_idx, f1_gain) pairs.

    Parameters
    ----------
    gen_data     : list[PyG Data]  candidate generated graphs
    train_data   : list[PyG Data]  real training graphs
    val_data     : list[PyG Data]  validation graphs (not touched during main training)
    device       : torch.device
    proxy_epochs : int   epochs for each proxy training (default 20)
    min_delta    : float minimum F1 gain to keep a graph (default 0.0)
    focal        : bool  use focal loss in proxy (mirrors main training)
    proxy_seeds  : int   number of independent proxy runs to average (default 3)

    Returns
    -------
    selected   : list[PyG Data]  filtered subset of gen_data
    gains      : list[tuple]  [(original_idx, f1_gain), ...] sorted best-first
    """
    print(f"\nGreedy selection: evaluating {len(gen_data)} generated graphs "
          f"with a {proxy_epochs}-epoch proxy GIN "
          f"(averaged over {proxy_seeds} seeds) …")

    baseline = _proxy_val_f1(train_data, val_data, device,
                             epochs=proxy_epochs, focal=focal, n_seeds=proxy_seeds)
    print(f"  Baseline val F1 (no augmentation): {baseline:.4f}")

    gains = []
    for i, g in enumerate(gen_data):
        f1 = _proxy_val_f1(train_data + [g], val_data, device,
                           epochs=proxy_epochs, seed=i, focal=focal,
                           n_seeds=proxy_seeds)
        gains.append((i, f1 - baseline))
        if (i + 1) % 10 == 0 or (i + 1) == len(gen_data):
            print(f"  [{i+1:>3}/{len(gen_data)}]  "
                  f"graph {i:>4}: Δ F1 = {f1 - baseline:+.4f}")

    gains_sorted = sorted(gains, key=lambda x: x[1], reverse=True)
    selected_idx = {idx for idx, delta in gains_sorted if delta > min_delta}
    selected     = [g for i, g in enumerate(gen_data) if i in selected_idx]

    n_kept    = len(selected)
    mean_gain = float(np.mean([d for _, d in gains_sorted if d > min_delta])) if selected else 0.0
    print(f"  Selected {n_kept}/{len(gen_data)} graphs  "
          f"(mean Δ F1 among kept: {mean_gain:+.4f})\n")

    return selected, gains_sorted


def run_experiment(train_data, val_data, test_data, model_cls, device, seed,
                   focal=False):
    """
    Train one model with checkpoint selection by val F1 (tuned threshold).
    The threshold that maximises val F1 on the best checkpoint is then applied
    to the test set, so precision/recall/F1 are all computed at that threshold.
    AUC is threshold-independent and reported alongside.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_data,   batch_size=BATCH_SIZE)
    test_loader  = DataLoader(test_data,  batch_size=BATCH_SIZE)

    # Class-weighted loss: upweight the minority (laundering) class so the
    # model doesn't collapse to always predicting "clean" under real imbalance.
    n_pos = sum(d.y.item() == 1 for d in train_data)
    n_neg = len(train_data) - n_pos
    if n_pos > 0 and n_neg > 0:
        pos_weight = torch.tensor([1.0, n_neg / n_pos], dtype=torch.float, device=device)
    else:
        pos_weight = None

    model     = model_cls().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_f1  = -1.0
    best_state   = None
    best_thresh  = 0.5

    for epoch in range(1, EPOCHS + 1):
        _train_epoch(model, train_loader, optimizer, device,
                     pos_weight=pos_weight, focal=focal)
        scheduler.step()
        val_labels, val_probs = _collect_probs(model, val_loader, device)
        thresh, val_f1 = _best_f1_threshold(val_labels, val_probs)
        if val_f1 > best_val_f1:
            best_val_f1  = val_f1
            best_thresh  = thresh
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return _evaluate(model, test_loader, device, threshold=best_thresh)


def _mean_std(values):
    arr = np.array(values)
    return float(arr.mean()), float(arr.std())


# ─────────────────────────────────────────────────────────────────────────────
# Pretty-print helpers
# ─────────────────────────────────────────────────────────────────────────────
def _print_table(results):
    col_w = 28
    metrics = ["auc", "f1", "precision", "recall"]
    headers = ["Model / Condition"] + ["AUC-ROC", "F1", "Precision", "Recall"]
    sep = "─" * (col_w + len(metrics) * 16)

    print("\n" + sep)
    print(f"{'RESULTS  (mean ± std over ' + str(N_RUNS) + ' seeds, held-out test set)'}")
    print(sep)
    print(f"{'Model / Condition':<{col_w}}", end="")
    for h in headers[1:]:
        print(f"  {h:>12}", end="")
    print()
    print(sep)

    prev_model = None
    for key, res in results.items():
        model_name, condition = key.rsplit("_", 1)
        if model_name != prev_model and prev_model is not None:
            print()
        prev_model = model_name
        label = f"{model_name} [{condition}]"
        print(f"{label:<{col_w}}", end="")
        for m in metrics:
            mu, sd = res[m]
            print(f"  {mu:.3f}±{sd:.3f}", end="")
        print()

    print(sep)

    model_names = list(dict.fromkeys(k.rsplit("_", 1)[0] for k in results))
    has_delta   = any(f"{mn}_augmented" in results for mn in model_names)
    if has_delta:
        print("\nAugmentation delta (augmented − baseline):")
        for model_name in model_names:
            base_key = f"{model_name}_baseline"
            aug_key  = f"{model_name}_augmented"
            if base_key in results and aug_key in results:
                print(f"  {model_name}:")
                for m in metrics:
                    delta = results[aug_key][m][0] - results[base_key][m][0]
                    sign  = "+" if delta >= 0 else ""
                    print(f"    {m:<12} {sign}{delta:.3f}")


def _save_csv(results, path):
    path.parent.mkdir(exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "condition",
                         "auc_mean", "auc_std",
                         "f1_mean",  "f1_std",
                         "prec_mean","prec_std",
                         "rec_mean", "rec_std"])
        for key, res in results.items():
            model_name, condition = key.rsplit("_", 1)
            writer.writerow([model_name, condition,
                             *res["auc"], *res["f1"],
                             *res["precision"], *res["recall"]])
    print(f"Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--augment", action="store_true",
                        help="Augment training set with diffusion-generated networks")
    parser.add_argument("--focal-loss", action="store_true",
                        help="Use focal loss (γ=2) instead of weighted cross-entropy. "
                             "Downweights easy negatives more aggressively — helps with "
                             "the 8%% positive rate in this dataset.")
    parser.add_argument("--augment-select", action="store_true",
                        help="Like --augment but greedily selects only generated graphs "
                             "that improve val AUC of a fast proxy GIN. "
                             "Implies --augment.")
    parser.add_argument("--proxy-epochs", type=int, default=20, metavar="E",
                        help="Epochs for the fast proxy GIN used in greedy selection "
                             "(default 20). Lower = faster but noisier.")
    parser.add_argument("--proxy-seeds", type=int, default=1, metavar="S",
                        help="Number of independent proxy runs to average per candidate "
                             "graph during greedy selection (default 1). Higher = more "
                             "reliable ranking but linearly more compute.")
    parser.add_argument("--min-delta", type=float, default=0.0, metavar="D",
                        help="Minimum val AUC gain to keep a generated graph during "
                             "greedy selection (default 0.0 = keep anything that "
                             "doesn't hurt).")
    parser.add_argument("--n-gen", type=int, default=40,
                        help="Number of networks to generate for augmentation (default 40)")
    parser.add_argument("--low-data", type=float, default=1.0, metavar="FRAC",
                        help="Subsample training set to this fraction before augmenting "
                             "(e.g. 0.2 = 20%%). Useful to show augmentation benefit "
                             "under data scarcity. Default: 1.0 (full training set)")
    parser.add_argument("--dataset", choices=["ibm", "elliptic", "both"],
                        default="ibm",
                        help="Dataset to use: ibm (default), elliptic, or both combined")
    parser.add_argument("--ibm-csv", type=str, default=None, metavar="PATH",
                        help="Override the IBM CSV path. Use this to switch between "
                             "HI-Small_Trans.csv (default) and LI-Large_Trans.csv, e.g. "
                             r"--ibm-csv C:\path\to\LI-Large_Trans.csv")
    # ── Ablation / encoder override ──────────────────────────────────────────
    parser.add_argument("--encoder-dir", type=str, default=None, metavar="DIR",
                        help="Override the SimCLR encoder checkpoint directory. "
                             "Must contain a .pt file with 'encoder_state_dict' and 'loss'. "
                             "Default: checkpoints/simclr_elliptic (Elliptic) or "
                             "checkpoints/simclr_ibm (IBM).")
    parser.add_argument("--ablation-label", type=str, default=None, metavar="LABEL",
                        help="Short label appended to the results CSV name, e.g. "
                             "'no_supcon' → classifier_comparison_elliptic_no_supcon.csv")
    # ── Generation hyperparameter overrides ──────────────────────────────────
    parser.add_argument("--guidance-scale", type=float, default=None,
                        help="Override guidance_scale in guided generation (default 2.0). "
                             "Set 0 for unguided pure diffusion.")
    parser.add_argument("--novelty-weight", type=float, default=None,
                        help="Override novelty_weight in guided generation (default 2.0). "
                             "Set 0 to disable novelty repulsion.")
    parser.add_argument("--degree-penalty", type=float, default=None,
                        help="Override degree_penalty in guided generation (default 0.5). "
                             "Set 0 to disable density constraint.")
    parser.add_argument("--q-threshold", type=float, default=0.4, metavar="Q",
                        help="Minimum composite quality score Q ∈ [0,1] required to keep "
                             "a generated graph. Graphs below this threshold are discarded "
                             "before being added to the training set. "
                             "Q is the mean of (1-norm_emb_dist, 1-norm_wass_dist, "
                             "1-norm_density). Default: 0.4. Set 0.0 to keep all.")
    parser.add_argument("--t-start", type=int, default=None,
                        help="Override t_start (starting diffusion timestep) in guided "
                             "generation (default 150 for Elliptic).")
    # ── Direction 3: embedding separation diagnostic ──────────────────────────
    parser.add_argument("--sep-check", action="store_true",
                        help="After loading the SimCLR encoder, compute and print silhouette "
                             "score and linear probe AUC on the training embeddings. "
                             "Low scores (silhouette < 0.05, AUC < 0.65) indicate the encoder "
                             "does not separate classes well — retrain SimCLR before generating.")
    # ── Direction 4: learnable augmentation policy ────────────────────────────
    parser.add_argument("--tune-guidance", action="store_true",
                        help="Before generating augmentation data, run a Bayesian/random "
                             "search over (guidance_scale, novelty_weight, degree_penalty) "
                             "to find the combination that maximises mean Q-score. "
                             "Overrides --guidance-scale / --novelty-weight / --degree-penalty "
                             "with the tuned values.  Implies --augment.")
    parser.add_argument("--tune-trials", type=int, default=15, metavar="N",
                        help="Number of hyperparameter candidates to evaluate during guidance "
                             "tuning (default 15). More trials = better optimum but slower. "
                             "Each trial generates --tune-gen-per-trial graphs.")
    parser.add_argument("--tune-gen-per-trial", type=int, default=6, metavar="K",
                        help="Graphs generated per tuning trial for Q-score estimation "
                             "(default 6). Higher = less variance but linearly more compute.")
    args = parser.parse_args()

    if args.augment_select:
        args.augment = True
    if args.tune_guidance:
        args.augment = True   # tuning implies we'll generate augmentation data

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    from generation.graph_quality_metrics import (
        score_generated_graphs, print_quality_report, save_quality_csv,
        plot_quality_extremes, compute_embedding_separation,
    )

    # ── 1. Load PyG graphs ───────────────────────────────────────────────────
    all_data: list = []
    networks: list = []   # IBM network dicts — needed for augmentation

    # ibm_data_networks[i] is the IBM network dict that produced all_data[i]
    # (only populated for IBM graphs; used later to restrict probe to training fold)
    ibm_data_networks: list = []

    # Parallel FraudGT data list — same graphs with ego_id appended as last feature
    # Only populated for IBM data (ego_id requires start_node from network dicts)
    fraudgt_all_data: list = []

    if args.dataset in ("ibm", "both"):
        ibm_csv = args.ibm_csv if args.ibm_csv else CSV_PATH
        # Cache filename is tied to the CSV so switching datasets doesn't silently
        # reuse a stale cache built from a different file.
        csv_stem   = Path(ibm_csv).stem
        CACHE_PATH = DATA_DIR / f"networks_cache_{csv_stem}_v2.pkl"
        df_full    = preprocess_df(ibm_csv)

        if CACHE_PATH.exists():
            print(f"Loading IBM networks from cache ({CACHE_PATH.name}) …")
            with open(CACHE_PATH, "rb") as f:
                networks = pickle.load(f)
            for net in networks:
                net["graph"] = build_igraph_from_transactions(net["transactions"])
        else:
            print(f"Extracting IBM networks from {ibm_csv} …")
            networks = extract_transaction_ego_networks(
                df_full,
                max_depth=2,
                max_nodes=50,
                n_pos=2000,
                neg_pos_ratio=10,
            )
            for net in networks:
                net["graph"] = build_igraph_from_transactions(net["transactions"])
            # Save cache — exclude igraph objects which are not picklable
            networks_to_cache = [{k: v for k, v in net.items() if k != "graph"}
                                 for net in networks]
            with open(CACHE_PATH, "wb") as f:
                pickle.dump(networks_to_cache, f)
            print(f"Saved IBM network cache → {CACHE_PATH.name}")

        print(f"IBM networks: {len(networks)}")
        from diffusion.diff_util import network_to_dense as _ntd

        skipped = 0
        for net in networks:
            try:
                x_d, adj_d = _ntd(net)
            except Exception:
                skipped += 1
                continue
            if x_d.shape[0] < 3:
                skipped += 1
                continue
            # Use tx_label (the focal transaction's laundering flag) when
            # available; fall back to subgraph-level label for cached nets.
            label = net.get("tx_label",
                            1 if len(net.get("laundering_nodes", set())) > 0 else 0)
            pyg = network_to_pyg(x_d, adj_d, label)
            pyg.timestamp_val = float(net["timestamp"].timestamp()) \
                if "timestamp" in net else -1.0
            all_data.append(pyg)
            ibm_data_networks.append(net)
            _fgt_pyg = network_to_pyg_fraudgt(x_d, adj_d, label, net)
            _fgt_pyg.timestamp_val = pyg.timestamp_val
            fraudgt_all_data.append(_fgt_pyg)

        if skipped:
            print(f"  [{skipped} IBM networks skipped]")

    if args.dataset in ("elliptic", "both"):
        from data.elliptic_adapter import load_elliptic_pyg_graphs
        elliptic_graphs = load_elliptic_pyg_graphs()
        all_data.extend(elliptic_graphs)

    all_labels_np = np.array([d.y.item() for d in all_data])
    n_laund = int(all_labels_np.sum())
    print(f"\nTotal PyG graphs: {len(all_data)}  "
          f"({n_laund} illicit/laundering, {len(all_data) - n_laund} clean)")

    if len(all_data) == 0:
        raise RuntimeError(
            "No graphs were loaded. "
            "For --dataset ibm: ensure networks_cache.pkl exists or IBM CSV is accessible. "
            "For --dataset elliptic: ensure the Elliptic CSV files are in the data directory."
        )
    if len(np.unique(all_labels_np)) < 2:
        raise RuntimeError(
            f"All {len(all_data)} graphs have the same label — "
            "stratified splitting requires both classes."
        )

    # ── 3. Train / val / test split ──────────────────────────────────────────
    #
    # Elliptic  → temporal split: train+val on timesteps 1-34, test on 35-49.
    #   This is the standard benchmark protocol from Weber et al. (2019) and
    #   prevents ego-subgraph overlap from inflating scores: graphs from the
    #   same timestep share nodes, so a random split would leak structure.
    #
    # IBM / combined → stratified random split (no timestep signal available).
    #
    ELLIPTIC_TRAIN_TS_MAX = 34   # timesteps 1-34 for train+val, 35-49 for test

    if args.dataset == "elliptic":
        # Temporal split — use the .timestep attribute added by elliptic_adapter
        trainval_pool = [g for g in all_data if g.timestep.item() <= ELLIPTIC_TRAIN_TS_MAX]
        data_test     = [g for g in all_data if g.timestep.item() >  ELLIPTIC_TRAIN_TS_MAX]

        if len(data_test) == 0:
            raise RuntimeError(
                "Temporal split produced an empty test set. "
                "Check that Elliptic graphs have a valid .timestep attribute."
            )

        tv_labels = np.array([g.y.item() for g in trainval_pool])
        val_size  = VAL_FRAC / (1.0 - TEST_FRAC)
        sss_val   = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=42)
        idx_tr, idx_val = next(sss_val.split(np.arange(len(trainval_pool)), tv_labels))

        data_train = [trainval_pool[i] for i in idx_tr]
        data_val   = [trainval_pool[i] for i in idx_val]

        # idx_trainval and idx_test (into all_data) for probe training below
        # Build a timestep-based mapping back to all_data indices
        _ts_map = [(i, g.timestep.item()) for i, g in enumerate(all_data)]
        idx_trainval = np.array([i for i, ts in _ts_map if ts <= ELLIPTIC_TRAIN_TS_MAX])
        idx_test_arr = np.array([i for i, ts in _ts_map if ts >  ELLIPTIC_TRAIN_TS_MAX])

        print(f"Temporal split (ts≤{ELLIPTIC_TRAIN_TS_MAX} train+val | "
              f"ts>{ELLIPTIC_TRAIN_TS_MAX} test): "
              f"{len(data_train)} train | {len(data_val)} val | {len(data_test)} test")

    else:
        # Temporal split for IBM — mirrors the benchmark protocol.
        # Graphs are sorted by transaction timestamp; we use the first 70%
        # of time for train+val and the last 20% for test (10% gap = val).
        # This tests generalisation to FUTURE transactions, which is the
        # operationally relevant setting and what the paper measures.
        ts_vals = np.array([d.timestamp_val if hasattr(d, "timestamp_val") else -1.0
                            for d in all_data])

        if ts_vals.min() >= 0:
            # All graphs have timestamps — do a proper temporal split
            sorted_idx  = np.argsort(ts_vals)
            n           = len(sorted_idx)
            n_trainval  = int(n * (1.0 - TEST_FRAC))
            n_train     = int(n_trainval * (1.0 - VAL_FRAC / (1.0 - TEST_FRAC)))

            idx_tr       = sorted_idx[:n_train]
            idx_val      = sorted_idx[n_train:n_trainval]
            idx_test_arr = sorted_idx[n_trainval:]
            idx_trainval = sorted_idx[:n_trainval]
        else:
            # Fallback to stratified random split (e.g. loaded from old cache)
            indices = np.arange(len(all_data))
            sss_test = StratifiedShuffleSplit(n_splits=1, test_size=TEST_FRAC, random_state=42)
            idx_trainval, idx_test_arr = next(sss_test.split(indices, all_labels_np))
            labels_trainval = all_labels_np[idx_trainval]
            val_size = VAL_FRAC / (1.0 - TEST_FRAC)
            sss_val  = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=42)
            idx_tr, idx_val = next(sss_val.split(np.arange(len(idx_trainval)), labels_trainval))
            idx_tr  = idx_trainval[idx_tr]
            idx_val = idx_trainval[idx_val]

        data_train = [all_data[i] for i in idx_tr]
        data_val   = [all_data[i] for i in idx_val]
        data_test  = [all_data[i] for i in idx_test_arr]

        test_pos  = sum(d.y.item() == 1 for d in data_test)
        train_pos = sum(d.y.item() == 1 for d in data_train)
        print(f"Temporal split: {len(data_train)} train | {len(data_val)} val | {len(data_test)} test")
        print(f"  train positive rate: {train_pos/len(data_train)*100:.1f}%  "
              f"test positive rate: {test_pos/len(data_test)*100:.1f}%")

    # ── 3b. Optional low-data subsampling ────────────────────────────────────
    if args.low_data < 1.0:
        n_train_before_ld = len(data_train)
        n_keep = max(10, int(n_train_before_ld * args.low_data))
        train_labels_np = np.array([d.y.item() for d in data_train])
        sss_ld = StratifiedShuffleSplit(n_splits=1, train_size=n_keep, random_state=42)
        idx_ld, _ = next(sss_ld.split(np.arange(len(data_train)), train_labels_np))
        data_train = [data_train[i] for i in idx_ld]
        print(f"Low-data mode: keeping {len(data_train)} / {n_train_before_ld} "
              f"training graphs ({args.low_data:.0%})")
        print()

    # ── 3c. FraudGT and ExSTraQt splits (IBM only) ───────────────────────────
    # Mirror the main all_data split; idx_tr / idx_val / idx_test_arr are set above.
    n_ibm = len(ibm_data_networks)
    if fraudgt_all_data:
        fgt_train = [fraudgt_all_data[i] for i in idx_tr  if i < n_ibm]
        fgt_val   = [fraudgt_all_data[i] for i in idx_val if i < n_ibm]
        fgt_test  = [fraudgt_all_data[i] for i in idx_test_arr if i < n_ibm]
    else:
        fgt_train = fgt_val = fgt_test = []

    xq_train_nets = [ibm_data_networks[i] for i in idx_tr      if i < n_ibm]
    xq_val_nets   = [ibm_data_networks[i] for i in idx_val     if i < n_ibm]
    xq_test_nets  = [ibm_data_networks[i] for i in idx_test_arr if i < n_ibm]

    # ── 4. Optionally generate augmentation data ─────────────────────────────
    gen_data = []
    if args.augment:
        if args.dataset == "elliptic":
            # ── Elliptic augmentation path ────────────────────────────────
            # Requires elliptic_diffusion_train.py and elliptic_simclr_train.py
            # to have been run first.
            print(f"[Elliptic] Generating {args.n_gen} augmentation graphs …")
            from generation.generation import (
                load_simclr_encoder_elliptic,
                load_diffusion_model_elliptic,
                encode_all_pyg_graphs,
                train_mlp_probe,
                run_guided_generation_elliptic,
            )
            if args.encoder_dir is not None:
                # Load encoder from an alternative directory (for SimCLR ablations)
                from simclr import GraphEncoder as _GE
                _ckpt_dir = Path(args.encoder_dir)
                _candidates = list(_ckpt_dir.glob("*.pt"))
                _best_path, _best_loss = None, float("inf")
                for _p in _candidates:
                    try:
                        _c = torch.load(_p, map_location="cpu", weights_only=False)
                        if isinstance(_c, dict) and "loss" in _c and _c["loss"] < _best_loss:
                            _best_loss, _best_path = _c["loss"], _p
                    except Exception:
                        pass
                if _best_path is None:
                    raise FileNotFoundError(f"No valid checkpoint in {args.encoder_dir}")
                print(f"Ablation encoder: {_best_path.name}  (loss={_best_loss:.4f})")
                _ckpt = torch.load(_best_path, map_location=device, weights_only=False)
                # in_dim=5 after label-leakage fix (col 0 stripped before encoder)
                encoder_e = _GE(in_dim=5, hidden_dim=64, out_dim=128).to(device)
                encoder_e.load_state_dict(_ckpt["encoder_state_dict"])
                encoder_e.eval()
            else:
                encoder_e = load_simclr_encoder_elliptic(device)

            diff_model_e, diffusion_e, x_mean_e, x_std_e = load_diffusion_model_elliptic(device)
            # Only encode training-fold graphs to avoid test leakage in the probe.
            # data_train has already been split off from the temporal train pool.
            H_all_e, y_all_e = encode_all_pyg_graphs(data_train, encoder_e, device)
            probe_e = train_mlp_probe(H_all_e, y_all_e, device)

            # ── Direction 3: embedding separation diagnostic (Elliptic) ──────
            if args.sep_check:
                print("\n[Direction 3] SimCLR embedding separation diagnostic …")
                _sep_labels_e = [d.y.item() for d in data_train]
                _sep_e        = compute_embedding_separation(
                                    data_train, _sep_labels_e, encoder_e, device)
                print(f"  Silhouette score    : {_sep_e['silhouette']:.4f}")
                print(f"  Linear probe AUC    : {_sep_e['linear_probe_auc']:.4f}")
                if _sep_e["silhouette"] < 0.05:
                    print("  WARNING: silhouette < 0.05 — guidance will be poorly class-conditioned.")
                print()

            _e_t_start = args.t_start if args.t_start is not None else 150
            _e_gs = args.guidance_scale
            _e_nw = args.novelty_weight
            _e_dp = args.degree_penalty

            # ── Direction 4: tune guidance hyperparameters (Elliptic) ────────
            if args.tune_guidance:
                from generation.generation import tune_guidance_params as _tune_e
                print(f"\n[Direction 4] Tuning guidance params (Elliptic, "
                      f"{args.tune_trials} trials) …")
                _train_laund_e_pre = [g for g in data_train if g.y.item() == 1]
                _H_laund_e_pre     = H_all_e[y_all_e == 1]
                # Elliptic uses PyG graphs not IBM network dicts — pass as networks=None;
                # the objective falls back gracefully if run_guided_generation can't seed.
                _best_e, _ = _tune_e(
                    list(data_train), encoder_e, probe_e,
                    diff_model_e, diffusion_e, x_mean_e, x_std_e, H_all_e,
                    _train_laund_e_pre, _H_laund_e_pre, device,
                    n_trials=args.tune_trials,
                    n_gen_per_trial=args.tune_gen_per_trial,
                    t_start=_e_t_start,
                    results_dir=ROOT_DIR / "results",
                )
                _e_gs = _best_e.get("guidance_scale", 2.0)
                _e_nw = _best_e.get("novelty_weight", 2.0)
                _e_dp = _best_e.get("degree_penalty", 0.5)
                print(f"  Using tuned params: guidance_scale={_e_gs:.3f}  "
                      f"novelty_weight={_e_nw:.3f}  degree_penalty={_e_dp:.3f}\n")

            _gen_kwargs_shared = dict(
                n_gen=args.n_gen // 2,   # half per class for balanced augmentation
                t_start=_e_t_start,
            )
            if _e_gs is not None:
                _gen_kwargs_shared["guidance_scale"] = _e_gs
            if _e_nw is not None:
                _gen_kwargs_shared["novelty_weight"] = _e_nw
            if _e_dp is not None:
                _gen_kwargs_shared["degree_penalty"] = _e_dp

            # Generate for both classes to match the real class balance and
            # avoid biasing the augmented training set toward class 1.
            _gen_raw_e = []
            for _lbl in (1, 0):
                _outs, _, _ = run_guided_generation_elliptic(
                    data_train, encoder_e, probe_e, diff_model_e, diffusion_e,
                    x_mean_e, x_std_e, H_all_e, device,
                    target_label=_lbl,
                    **_gen_kwargs_shared,
                )
                _gen_raw_e.extend(_outs)

            for (x_denorm, adj_out, n_out, label) in _gen_raw_e:
                gen_data.append(gen_output_to_pyg(x_denorm, adj_out, n_out, label))

            n_ill_gen = sum(d.y.item() == 1 for d in gen_data)
            n_lit_gen = len(gen_data) - n_ill_gen
            print(f"Generated {len(gen_data)} augmentation graphs "
                  f"(illicit={n_ill_gen}, licit={n_lit_gen})\n")

            # ── Tier-1 quality scoring (Elliptic) ────────────────────────
            _train_laund_e   = [g for g in data_train if g.y.item() == 1]
            _H_train_laund_e = H_all_e[y_all_e == 1]
            _quality_e = score_generated_graphs(
                gen_data, _train_laund_e, _H_train_laund_e, encoder_e, device)
            print_quality_report(_quality_e)
            _q_suffix_e = f"elliptic{'_ld' + str(int(args.low_data * 100)) if args.low_data < 1.0 else ''}"
            save_quality_csv(_quality_e,
                             ROOT_DIR / "results" / f"graph_quality_{_q_suffix_e}.csv")
            plot_quality_extremes(_quality_e, gen_data,
                                  ROOT_DIR / "results" / f"quality_extremes_{_q_suffix_e}.png")

            # ── Q-score filtering ─────────────────────────────────────────
            if args.q_threshold > 0.0:
                Q_scores = _quality_e["Q"]
                before   = len(gen_data)
                gen_data = [g for g, q in zip(gen_data, Q_scores) if q >= args.q_threshold]
                print(f"Q-filter (threshold={args.q_threshold:.2f}): "
                      f"kept {len(gen_data)}/{before} generated graphs\n")

        elif not networks:
            # ── IBM networks missing ──────────────────────────────────────
            print("WARNING: --augment requires IBM networks for the diffusion+SimCLR "
                  "pipeline (the model was trained on IBM data). "
                  "Re-run with --dataset ibm or --dataset both to enable augmentation.")
            args.augment = False

        else:
            # ── IBM augmentation path (original) ──────────────────────────
            _ld_tag  = f"_ld{int(args.low_data * 100)}" if args.low_data < 1.0 else ""
            _t_start = args.t_start if args.t_start is not None else 150
            GEN_CACHE_PATH = DATA_DIR / f"gen_cache_{csv_stem}_n{args.n_gen}_t{_t_start}{_ld_tag}.pkl"

            _cache_valid = False
            if GEN_CACHE_PATH.exists():
                with open(GEN_CACHE_PATH, "rb") as _f:
                    gen_data = pickle.load(_f)
                # Invalidate cache if feature dim changed (expected IN_CHANNELS)
                _sample_dim = gen_data[0].x.shape[1] if gen_data else 0
                if _sample_dim == IN_CHANNELS:
                    _cache_valid = True
                    n_laund_gen = sum(d.y.item() == 1 for d in gen_data)
                    n_clean_gen = len(gen_data) - n_laund_gen
                    print(f"Loading generated graphs from cache ({GEN_CACHE_PATH.name}) …")
                    print(f"Loaded {len(gen_data)} cached augmentation graphs "
                          f"(laundering={n_laund_gen}, clean={n_clean_gen})\n")
                else:
                    print(f"Stale cache (dim={_sample_dim}, expected {IN_CHANNELS}) — regenerating …")
                    GEN_CACHE_PATH.unlink()
            if not _cache_valid:
                gen_data = []
                print(f"Generating {args.n_gen} augmentation networks …")
                from generation.generation import (
                    load_simclr_encoder, load_diffusion_model,
                    encode_all_networks, train_mlp_probe, run_guided_generation,
                    tune_guidance_params,
                )
                encoder = load_simclr_encoder(device)
                diff_model, diffusion, x_mean, x_std = load_diffusion_model(device)

                # Only encode networks that ended up in the TRAINING fold.
                # Using all networks (including test) would leak test-set embeddings
                # into the probe, biasing guided generation toward the test distribution.
                n_ibm = len(ibm_data_networks)
                train_ibm_nets = [
                    ibm_data_networks[i]
                    for i in idx_tr
                    if i < n_ibm
                ]
                if not train_ibm_nets:
                    train_ibm_nets = networks   # fallback if mapping fails

                H_all_n, y_all = encode_all_networks(train_ibm_nets, encoder, device)
                probe = train_mlp_probe(H_all_n, y_all, device)

                # ── Direction 3: embedding separation diagnostic ───────────
                if args.sep_check:
                    print("\n[Direction 3] SimCLR embedding separation diagnostic …")
                    _sep_labels = [d.y.item() for d in data_train]
                    _sep        = compute_embedding_separation(
                                    data_train, _sep_labels, encoder, device)
                    print(f"  Silhouette score    : {_sep['silhouette']:.4f}  "
                          f"(>0.05 good, <0.05 poor class separation)")
                    print(f"  Linear probe AUC    : {_sep['linear_probe_auc']:.4f}  "
                          f"(>0.65 good, <0.65 guidance signal too weak)")
                    if _sep["silhouette"] < 0.05:
                        print("  WARNING: silhouette < 0.05 — encoder may not separate classes. "
                              "Retrain SimCLR with higher supcon_weight (e.g. 1.0).")
                    if _sep["linear_probe_auc"] < 0.65:
                        print("  WARNING: linear probe AUC < 0.65 — guidance will be unreliable. "
                              "Consider more SimCLR epochs or a larger encoder.")
                    print()

                # ── Direction 4: tune guidance hyperparameters ─────────────
                _gs = args.guidance_scale
                _nw = args.novelty_weight
                _dp = args.degree_penalty

                if args.tune_guidance:
                    print(f"\n[Direction 4] Tuning guidance params "
                          f"({args.tune_trials} trials × {args.tune_gen_per_trial} graphs) …")
                    _train_laund_ibm_pre = [g for g in data_train if g.y.item() == 1]
                    _H_laund_ibm_pre     = H_all_n[y_all == 1]
                    _best_params, _history = tune_guidance_params(
                        networks, encoder, probe, diff_model, diffusion,
                        x_mean, x_std, H_all_n,
                        _train_laund_ibm_pre, _H_laund_ibm_pre, device,
                        n_trials=args.tune_trials,
                        n_gen_per_trial=args.tune_gen_per_trial,
                        t_start=_t_start,
                        results_dir=ROOT_DIR / "results",
                    )
                    _gs = _best_params.get("guidance_scale", 2.0)
                    _nw = _best_params.get("novelty_weight", 2.0)
                    _dp = _best_params.get("degree_penalty", 0.5)
                    print(f"  Using tuned params: guidance_scale={_gs:.3f}  "
                          f"novelty_weight={_nw:.3f}  degree_penalty={_dp:.3f}\n")

                # Generate for both classes (balanced) — avoids biasing the
                # augmented training set toward class 1 only.
                _gen_kwargs = dict(t_start=_t_start)
                if _gs is not None: _gen_kwargs["guidance_scale"] = _gs
                if _nw is not None: _gen_kwargs["novelty_weight"] = _nw
                if _dp is not None: _gen_kwargs["degree_penalty"] = _dp

                for _lbl in (1, 0):
                    _outs, _, _ = run_guided_generation(
                        networks, encoder, probe, diff_model, diffusion,
                        x_mean, x_std, H_all_n, device,
                        target_label=_lbl,
                        n_gen=args.n_gen // 2,
                        **_gen_kwargs,
                    )
                    for (x_denorm, adj_out, n_out, label) in _outs:
                        gen_data.append(gen_output_to_pyg(x_denorm, adj_out, n_out, label))

                n_laund_gen = sum(d.y.item() == 1 for d in gen_data)
                n_clean_gen = len(gen_data) - n_laund_gen
                print(f"Generated {len(gen_data)} augmentation graphs "
                      f"(laundering={n_laund_gen}, clean={n_clean_gen})\n")

                GEN_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
                with open(GEN_CACHE_PATH, "wb") as _f:
                    pickle.dump(gen_data, _f)
                print(f"Generated graphs cached to {GEN_CACHE_PATH.name}")

            # ── Tier-1 quality scoring (IBM) — skipped when loading from cache ──
            if not GEN_CACHE_PATH.exists() or 'encoder' in dir():
                try:
                    _train_laund_ibm   = [g for g in data_train if g.y.item() == 1]
                    _H_train_laund_ibm = H_all_n[y_all == 1]
                    _quality_ibm = score_generated_graphs(
                        gen_data, _train_laund_ibm, _H_train_laund_ibm, encoder, device)
                    print_quality_report(_quality_ibm)
                    _q_suffix_ibm = f"ibm{'_ld' + str(int(args.low_data * 100)) if args.low_data < 1.0 else ''}"
                    save_quality_csv(_quality_ibm,
                                     ROOT_DIR / "results" / f"graph_quality_{_q_suffix_ibm}.csv")
                    plot_quality_extremes(_quality_ibm, gen_data,
                                         ROOT_DIR / "results" / f"quality_extremes_{_q_suffix_ibm}.png")
                except Exception as _qe:
                    print(f"[quality scoring skipped: {_qe}]")
            else:
                # When loading from cache, quality scoring was already done in the
                # previous run.  Apply Q filtering based on a saved CSV if it exists,
                # otherwise skip (the user can re-generate without cache to re-filter).
                _q_csv = ROOT_DIR / "results" / f"graph_quality_ibm.csv"
                if args.q_threshold > 0.0 and _q_csv.exists():
                    import csv as _csv_mod
                    with open(_q_csv) as _qf:
                        _q_rows = list(_csv_mod.DictReader(_qf))
                    if len(_q_rows) == len(gen_data):
                        _Q_cached = [float(r["Q"]) for r in _q_rows]
                        before    = len(gen_data)
                        gen_data  = [g for g, q in zip(gen_data, _Q_cached)
                                     if q >= args.q_threshold]
                        print(f"Q-filter (threshold={args.q_threshold:.2f}, from cache): "
                              f"kept {len(gen_data)}/{before} generated graphs\n")

            # Apply Q filtering when quality scores were freshly computed
            if 'encoder' in dir() and '_quality_ibm' in dir():
                if args.q_threshold > 0.0:
                    Q_scores = _quality_ibm["Q"]
                    before   = len(gen_data)
                    gen_data = [g for g, q in zip(gen_data, Q_scores)
                                if q >= args.q_threshold]
                    print(f"Q-filter (threshold={args.q_threshold:.2f}): "
                          f"kept {len(gen_data)}/{before} generated graphs\n")

    # ── 5. Greedy selection (if requested) ──────────────────────────────────
    gen_data_selected = []
    selection_gains   = []
    if args.augment and args.augment_select and gen_data:
        gen_data_selected, selection_gains = greedy_select_generated(
            gen_data, list(data_train), data_val, device,
            proxy_epochs=args.proxy_epochs,
            min_delta=args.min_delta,
            focal=args.focal_loss,
            proxy_seeds=args.proxy_seeds,
        )
        # Save gain report
        gains_path = ROOT_DIR / "results" / "greedy_selection_gains.csv"
        gains_path.parent.mkdir(exist_ok=True)
        with open(gains_path, "w", newline="") as _f:
            _w = csv.writer(_f)
            _w.writerow(["graph_idx", "f1_gain", "kept"])
            kept_set = {i for i, _ in selection_gains
                        if _ > args.min_delta}
            for idx, gain in selection_gains:
                _w.writerow([idx, f"{gain:.6f}", int(idx in kept_set)])
        print(f"Selection gains saved → {gains_path}")

    # ── 6. Run experiments ───────────────────────────────────────────────────
    models     = [
        ("GIN",              GINClassifier),
        ("GraphTransformer", GraphTransformerClassifier),
        ("GraphSAGE",        GraphSAGEClassifier),
        ("DeepSets",         DeepSetsClassifier),
    ]
    conditions = ["baseline"]
    if args.augment:
        conditions.append("augmented")
    if args.augment_select and gen_data_selected:
        conditions.append("selected")
    results    = {}

    for model_name, model_cls in models:
        for condition in conditions:
            train_set = list(data_train)
            if condition == "augmented":
                train_set = train_set + gen_data
                random.shuffle(train_set)
            elif condition == "selected":
                train_set = train_set + gen_data_selected
                random.shuffle(train_set)

            label_counts = {0: 0, 1: 0}
            for d in train_set:
                label_counts[d.y.item()] += 1
            print(f"[{model_name} / {condition}]  "
                  f"train={len(train_set)} "
                  f"(clean={label_counts[0]}, laund={label_counts[1]})  "
                  f"running {N_RUNS} seeds …")

            run_metrics = {k: [] for k in ["auc", "f1", "precision", "recall"]}
            for seed in range(N_RUNS):
                m = run_experiment(train_set, data_val, data_test, model_cls, device, seed,
                                   focal=args.focal_loss)
                for k in run_metrics:
                    run_metrics[k].append(m[k])
                print(f"  seed {seed}: AUC={m['auc']:.3f}  F1={m['f1']:.3f}")

            results[f"{model_name}_{condition}"] = {
                k: _mean_std(v) for k, v in run_metrics.items()
            }

    # ── 6b. FraudGT (IBM only — requires ego_id feature) ────────────────────
    if fgt_train:
        _fgt_cls = functools.partial(FraudGTClassifier, in_channels=FRAUDGT_IN_CHANNELS)
        for condition in conditions:
            if condition == "baseline":
                fgt_tr = list(fgt_train)
            elif condition == "augmented":
                fgt_gen = [_to_fraudgt_format(g) for g in gen_data]
                fgt_tr  = list(fgt_train) + fgt_gen
                random.shuffle(fgt_tr)
            elif condition == "selected":
                fgt_gen = [_to_fraudgt_format(g) for g in gen_data_selected]
                fgt_tr  = list(fgt_train) + fgt_gen
                random.shuffle(fgt_tr)
            else:
                fgt_tr = list(fgt_train)

            label_counts = {0: 0, 1: 0}
            for d in fgt_tr:
                label_counts[d.y.item()] += 1
            print(f"[FraudGT / {condition}]  "
                  f"train={len(fgt_tr)} "
                  f"(clean={label_counts[0]}, laund={label_counts[1]})  "
                  f"running {N_RUNS} seeds …")

            run_metrics = {k: [] for k in ["auc", "f1", "precision", "recall"]}
            for seed in range(N_RUNS):
                m = run_experiment(fgt_tr, fgt_val, fgt_test, _fgt_cls, device, seed,
                                   focal=args.focal_loss)
                for k in run_metrics:
                    run_metrics[k].append(m[k])
                print(f"  seed {seed}: AUC={m['auc']:.3f}  F1={m['f1']:.3f}")

            results[f"FraudGT_{condition}"] = {
                k: _mean_std(v) for k, v in run_metrics.items()
            }
    else:
        print("[FraudGT] skipped — only runs on IBM data (ego_id requires network dicts)")

    # ── 6c. ExSTraQt (IBM only — requires transaction-level network dicts) ───
    if xq_train_nets:
        print(f"\n[ExSTraQt / baseline]  "
              f"train={len(xq_train_nets)}  "
              f"running {N_RUNS} seeds …")
        run_metrics_xq = {k: [] for k in ["auc", "f1", "precision", "recall"]}
        for seed in range(N_RUNS):
            m = run_experiment_exstraqt(xq_train_nets, xq_val_nets, xq_test_nets, seed=seed)
            for k in run_metrics_xq:
                run_metrics_xq[k].append(m[k])
            print(f"  seed {seed}: AUC={m['auc']:.3f}  F1={m['f1']:.3f}  [{m['_clf']}]")
        results["ExSTraQt_baseline"] = {
            k: _mean_std(v) for k, v in run_metrics_xq.items()
            if k != "_clf"
        }
    else:
        print("[ExSTraQt] skipped — only runs on IBM data (requires transaction dicts)")

    # ── 6. Report ────────────────────────────────────────────────────────────
    _print_table(results)
    suffix = args.dataset
    if args.low_data < 1.0:
        suffix += f"_ld{int(args.low_data * 100)}"
    if args.ablation_label:
        suffix += f"_{args.ablation_label}"
    _save_csv(results, ROOT_DIR / "results" / f"classifier_comparison_{suffix}.csv")


if __name__ == "__main__":
    main()
