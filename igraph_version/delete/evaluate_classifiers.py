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
import pickle
import random
from pathlib import Path

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

SIMCLR_DIR = Path(__file__).resolve().parent
DIFF_DIR = SIMCLR_DIR.parent / "diffusion"

if str(DIFF_DIR) not in sys.path:
    sys.path.insert(0, str(DIFF_DIR))
if str(SIMCLR_DIR.parent) not in sys.path:
    sys.path.insert(0, str(SIMCLR_DIR.parent))
if str(SIMCLR_DIR) not in sys.path:
    sys.path.insert(0, str(SIMCLR_DIR))

from util import preprocess_df, extract_laundering_networks_igraph, extract_non_laundering_networks_igraph
from augmentation import build_igraph_from_transactions

# ── hyper-parameters ──────────────────────────────────────────────────────────
CSV_PATH    = r"C:\Users\lucsc\Thesis\grad\grad\data\IBM\LI-Small_Trans.csv"  # default
IN_CHANNELS = 5        # structural features only: degree, betweenness, clustering, pagerank, assortativity
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
        x          = torch.tensor(x_np, dtype=torch.float),
        edge_index = ei,
        y          = torch.tensor([label], dtype=torch.long),
    )


def gen_output_to_pyg(x_denorm, adj_out, n_out):
    """
    Convert a diffusion-generated triple to a PyG Data object.
    x_denorm[:, 0] = predicted laundering probability — excluded.
    Label is always 1 (all generated networks target the laundering class).
    """
    x_np   = x_denorm[:n_out, 1:].float().numpy()
    adj_np = adj_out[:n_out, :n_out]
    adj_np = adj_np.numpy() if isinstance(adj_np, torch.Tensor) else adj_np

    ei = _adj_to_edge_index(adj_np)
    if ei.shape[1] == 0:
        ei = _fallback_edges(n_out)

    return Data(
        x          = torch.tensor(x_np, dtype=torch.float),
        edge_index = ei,
        y          = torch.tensor([1], dtype=torch.long),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Training / evaluation
# ─────────────────────────────────────────────────────────────────────────────
def _train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch.x, batch.edge_index, batch.batch)
        loss   = F.cross_entropy(logits, batch.y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)


@torch.no_grad()
def _evaluate(model, loader, device):
    model.eval()
    all_labels, all_probs = [], []
    for batch in loader:
        batch  = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.batch)
        probs  = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        all_labels.extend(batch.y.cpu().numpy())
        all_probs.extend(probs)

    labels = np.array(all_labels)
    probs  = np.array(all_probs)
    preds  = (probs >= 0.5).astype(int)

    auc  = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.5
    f1   = f1_score(labels, preds, average="macro", zero_division=0)
    prec = precision_score(labels, preds, average="macro", zero_division=0)
    rec  = recall_score(labels, preds, average="macro", zero_division=0)
    return {"auc": auc, "f1": f1, "precision": prec, "recall": rec}


def run_experiment(train_data, val_data, test_data, model_cls, device, seed):
    """Train one model; pick the checkpoint with best val AUC; report test metrics."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_data,   batch_size=BATCH_SIZE)
    test_loader  = DataLoader(test_data,  batch_size=BATCH_SIZE)

    model     = model_cls().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_auc = -1.0
    best_state   = None

    for epoch in range(1, EPOCHS + 1):
        _train_epoch(model, train_loader, optimizer, device)
        scheduler.step()
        val_m = _evaluate(model, val_loader, device)
        if val_m["auc"] > best_val_auc:
            best_val_auc = val_m["auc"]
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return _evaluate(model, test_loader, device)


def _mean_std(values):
    arr = np.array(values)
    return float(arr.mean()), float(arr.std())


# ─────────────────────────────────────────────────────────────────────────────
# Pretty-print helpers
# ─────────────────────────────────────────────────────────────────────────────
def _print_table(results):
    col_w = 28
    metrics = ["auc", "f1", "precision", "recall"]
    headers = ["Model / Condition"] + ["AUC-ROC", "Macro-F1", "Precision", "Recall"]
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
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # ── 1. Load PyG graphs ───────────────────────────────────────────────────
    all_data: list = []
    networks: list = []   # IBM network dicts — needed for augmentation

    # ibm_data_networks[i] is the IBM network dict that produced all_data[i]
    # (only populated for IBM graphs; used later to restrict probe to training fold)
    ibm_data_networks: list = []

    if args.dataset in ("ibm", "both"):
        ibm_csv = args.ibm_csv if args.ibm_csv else CSV_PATH
        # Cache filename is tied to the CSV so switching datasets doesn't silently
        # reuse a stale cache built from a different file.
        csv_stem   = Path(ibm_csv).stem
        CACHE_PATH = SIMCLR_DIR / f"networks_cache_{csv_stem}.pkl"
        df_full    = preprocess_df(ibm_csv)

        if CACHE_PATH.exists():
            print(f"Loading IBM networks from cache ({CACHE_PATH.name}) …")
            with open(CACHE_PATH, "rb") as f:
                networks = pickle.load(f)
            for net in networks:
                net["graph"] = build_igraph_from_transactions(net["transactions"])
        else:
            print(f"Extracting IBM networks from {ibm_csv} (slow — run test.py first to cache) …")
            with_laund = extract_laundering_networks_igraph(
                df_full, max_depth=4, max_networks=2000,
                collapse_threshold=10, max_nodes=300,
            )
            non_laund = extract_non_laundering_networks_igraph(
                df_full, max_depth=4, max_networks=len(with_laund),
                collapse_threshold=10, max_nodes=300,
            )
            networks = with_laund + non_laund
            for net in networks:
                net["graph"] = build_igraph_from_transactions(net["transactions"])

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
            label = 1 if len(net.get("laundering_nodes", set())) > 0 else 0
            all_data.append(network_to_pyg(x_d, adj_d, label))
            ibm_data_networks.append(net)

        if skipped:
            print(f"  [{skipped} IBM networks skipped]")

    if args.dataset in ("elliptic", "both"):
        from grad.igraph_version.archive.elliptic_adapter import load_elliptic_pyg_graphs
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
        # Stratified random split for IBM / combined
        indices = np.arange(len(all_data))

        sss_test = StratifiedShuffleSplit(n_splits=1, test_size=TEST_FRAC, random_state=42)
        idx_trainval, idx_test_arr = next(sss_test.split(indices, all_labels_np))

        data_trainval   = [all_data[i]    for i in idx_trainval]
        labels_trainval = all_labels_np[idx_trainval]
        data_test       = [all_data[i]    for i in idx_test_arr]

        val_size = VAL_FRAC / (1.0 - TEST_FRAC)
        sss_val  = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=42)
        idx_tr, idx_val = next(sss_val.split(np.arange(len(data_trainval)), labels_trainval))

        data_train = [data_trainval[i] for i in idx_tr]
        data_val   = [data_trainval[i] for i in idx_val]

        print(f"Split: {len(data_train)} train | {len(data_val)} val | {len(data_test)} test")

    # ── 3b. Optional low-data subsampling ────────────────────────────────────
    if args.low_data < 1.0:
        n_keep = max(10, int(len(data_train) * args.low_data))
        train_labels_np = np.array([d.y.item() for d in data_train])
        sss_ld = StratifiedShuffleSplit(n_splits=1, train_size=n_keep, random_state=42)
        idx_ld, _ = next(sss_ld.split(np.arange(len(data_train)), train_labels_np))
        data_train = [data_train[i] for i in idx_ld]
        print(f"Low-data mode: keeping {len(data_train)} / {len(data_trainval)} "
              f"training graphs ({args.low_data:.0%})")
    print()

    # ── 4. Optionally generate augmentation data ─────────────────────────────
    gen_data = []
    if args.augment:
        if args.dataset == "elliptic":
            # ── Elliptic augmentation path ────────────────────────────────
            # Requires elliptic_diffusion_train.py and elliptic_simclr_train.py
            # to have been run first.
            print(f"[Elliptic] Generating {args.n_gen} augmentation graphs …")
            from grad.igraph_version.delete.generation import (
                load_simclr_encoder_elliptic,
                load_diffusion_model_elliptic,
                encode_all_pyg_graphs,
                train_mlp_probe,
                run_guided_generation_elliptic,
            )
            encoder_e = load_simclr_encoder_elliptic(device)
            diff_model_e, diffusion_e, x_mean_e, x_std_e = load_diffusion_model_elliptic(device)
            # Only encode training-fold graphs to avoid test leakage in the probe.
            # data_train has already been split off from the temporal train pool.
            H_all_e, y_all_e = encode_all_pyg_graphs(data_train, encoder_e, device)
            probe_e = train_mlp_probe(H_all_e, y_all_e, device)

            gen_outputs, _, _ = run_guided_generation_elliptic(
                data_train, encoder_e, probe_e, diff_model_e, diffusion_e,
                x_mean_e, x_std_e, H_all_e, device,
                target_label=1,
                n_gen=args.n_gen,
                t_start=150,
            )
            for (x_denorm, adj_out, n_out) in gen_outputs:
                gen_data.append(gen_output_to_pyg(x_denorm, adj_out, n_out))

            print(f"Generated {len(gen_data)} augmentation graphs (label = illicit)\n")

        elif not networks:
            # ── IBM networks missing ──────────────────────────────────────
            print("WARNING: --augment requires IBM networks for the diffusion+SimCLR "
                  "pipeline (the model was trained on IBM data). "
                  "Re-run with --dataset ibm or --dataset both to enable augmentation.")
            args.augment = False

        else:
            # ── IBM augmentation path (original) ──────────────────────────
            print(f"Generating {args.n_gen} augmentation networks …")
            from grad.igraph_version.delete.generation import (
                load_simclr_encoder, load_diffusion_model,
                encode_all_networks, train_mlp_probe, run_guided_generation,
            )
            encoder = load_simclr_encoder(device)
            diff_model, diffusion, x_mean, x_std = load_diffusion_model(device)

            # Only encode networks that ended up in the TRAINING fold.
            # Using all networks (including test) would leak test-set embeddings
            # into the probe, biasing guided generation toward the test distribution.
            n_ibm = len(ibm_data_networks)
            if args.dataset == "ibm":
                # idx_tr indexes into data_trainval; idx_trainval indexes into all_data
                train_ibm_nets = [
                    ibm_data_networks[idx_trainval[i]]
                    for i in idx_tr
                    if idx_trainval[i] < n_ibm
                ]
            else:
                # "both" dataset: IBM graphs are all_data[:n_ibm]
                train_ibm_nets = [
                    ibm_data_networks[idx_trainval[i]]
                    for i in idx_tr
                    if idx_trainval[i] < n_ibm
                ]
            if not train_ibm_nets:
                train_ibm_nets = networks   # fallback if mapping fails

            H_all_n, y_all = encode_all_networks(train_ibm_nets, encoder, device)
            probe = train_mlp_probe(H_all_n, y_all, device)

            gen_outputs, _, _ = run_guided_generation(
                networks, encoder, probe, diff_model, diffusion,
                x_mean, x_std, H_all_n, device,
                target_label=1,
                n_gen=args.n_gen,
                t_start=150,
            )
            for (x_denorm, adj_out, n_out) in gen_outputs:
                gen_data.append(gen_output_to_pyg(x_denorm, adj_out, n_out))

            print(f"Generated {len(gen_data)} augmentation graphs (label = laundering)\n")

    # ── 5. Run experiments ───────────────────────────────────────────────────
    models     = [
        ("GIN",              GINClassifier),
        ("GraphTransformer", GraphTransformerClassifier),
        ("GraphSAGE",        GraphSAGEClassifier),
        ("DeepSets",         DeepSetsClassifier),
    ]
    conditions = ["baseline"] + (["augmented"] if args.augment else [])
    results    = {}

    for model_name, model_cls in models:
        for condition in conditions:
            train_set = list(data_train)
            if condition == "augmented":
                train_set = train_set + gen_data
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
                m = run_experiment(train_set, data_val, data_test, model_cls, device, seed)
                for k in run_metrics:
                    run_metrics[k].append(m[k])
                print(f"  seed {seed}: AUC={m['auc']:.3f}  F1={m['f1']:.3f}")

            results[f"{model_name}_{condition}"] = {
                k: _mean_std(v) for k, v in run_metrics.items()
            }

    # ── 6. Report ────────────────────────────────────────────────────────────
    _print_table(results)
    suffix = args.dataset
    if args.low_data < 1.0:
        suffix += f"_ld{int(args.low_data * 100)}"
    _save_csv(results, SIMCLR_DIR / "results" / f"classifier_comparison_{suffix}.csv")


if __name__ == "__main__":
    main()
