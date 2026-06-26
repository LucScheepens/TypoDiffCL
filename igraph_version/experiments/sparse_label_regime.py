"""
sparse_label_regime.py
──────────────────────
Experiment: how do augmentation methods perform when only a fraction of
training graphs have their labels revealed (sparse-label / semi-supervised
setting)?

Unlike the low-data experiment (which removes minority-class examples),
here the full set of training graphs exists but only label_frac of them
are annotated.  Augmentation methods operate on the labeled subset only.

Conditions
──────────
  baseline    — labeled graphs only, no augmentation
  diffusion   — labeled + N_GEN diffusion-generated laundering graphs
  graphsmote  — labeled + N_GEN GraphSMOTE synthetic graphs
  gan         — labeled + N_GEN WGAN-GP synthetic graphs
  diga        — labeled + N_GEN DiGa synthetic graphs

Label fractions
───────────────
  5% | 10% | 25% | 50% | 100% of all training graphs are labeled

Seeds: 3 independent runs per (condition, fraction) combination
Metric: AUC-ROC + F1 (threshold tuned on validation set)

Usage
─────
    # from igraph_version/ directory:
    python experiments/sparse_label_regime.py

    # with options:
    python experiments/sparse_label_regime.py \\
        --csv data/IBM/LI-Small_Trans.csv \\
        --n-gen 50 \\
        --fractions 0.05 0.1 0.25 0.5 1.0 \\
        --seeds 3 \\
        --classifier gin \\
        --out experiments/results

Outputs
───────
    experiments/results/sparse_label_results.csv
    experiments/results/sparse_label_curves.png
"""

import argparse
import csv
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── path setup ────────────────────────────────────────────────────────────────
_HERE      = Path(__file__).resolve().parent
ROOT_DIR   = _HERE.parent
DIFF_DIR   = ROOT_DIR / "diffusion"
SIMCLR_DIR = ROOT_DIR / "simclr"
GEN_DIR    = ROOT_DIR / "generation"
DATA_DIR   = ROOT_DIR / "data"

for _p in (str(ROOT_DIR), str(DIFF_DIR), str(SIMCLR_DIR), str(GEN_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── constants ─────────────────────────────────────────────────────────────────
DEFAULT_CSV      = str(ROOT_DIR.parent / "data" / "IBM" / "LI-Small_Trans.csv")
DIFF_CKPT        = ROOT_DIR / "checkpoints" / "LI-Small" / "diffusion" / "model.pt"
SIMCLR_CKPT_DIR  = ROOT_DIR / "checkpoints" / "simclr_ibm"

IN_CHANNELS  = 18
HIDDEN       = 64
NUM_LAYERS   = 3
DROPOUT      = 0.3
LR           = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS       = 80
BATCH_SIZE   = 32

DEFAULT_FRACTIONS  = [0.05, 0.10, 0.25, 0.50, 1.0]
DEFAULT_N_GEN      = 50
DEFAULT_SEEDS      = 3
DEFAULT_CLASSIFIER = "gin"


# ── PyG imports ───────────────────────────────────────────────────────────────
from torch_geometric.nn import (
    GINConv, SAGEConv, TransformerConv,
    global_add_pool, global_mean_pool,
)
from torch_geometric.loader import DataLoader as PygDataLoader
from torch_geometric.data import Data
from torch_geometric.utils import degree
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit


# ── Classifier models (same as low_data_regime) ───────────────────────────────

class _GIN(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        for i in range(NUM_LAYERS):
            mlp = nn.Sequential(
                nn.Linear(IN_CHANNELS if i == 0 else HIDDEN, HIDDEN),
                nn.BatchNorm1d(HIDDEN), nn.ReLU(),
                nn.Linear(HIDDEN, HIDDEN),
            )
            self.convs.append(GINConv(mlp, train_eps=True))
            self.bns.append(nn.BatchNorm1d(HIDDEN))
        self.head = nn.Sequential(
            nn.Linear(HIDDEN, HIDDEN // 2), nn.ReLU(),
            nn.Dropout(DROPOUT), nn.Linear(HIDDEN // 2, 2),
        )

    def forward(self, x, edge_index, batch):
        for conv, bn in zip(self.convs, self.bns):
            x = bn(conv(x, edge_index)).relu()
        return self.head(global_add_pool(x, batch))


class _GraphSAGE(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList([
            SAGEConv(IN_CHANNELS if i == 0 else HIDDEN, HIDDEN)
            for i in range(NUM_LAYERS)
        ])
        self.bns = nn.ModuleList([nn.BatchNorm1d(HIDDEN) for _ in range(NUM_LAYERS)])
        self.head = nn.Sequential(
            nn.Linear(HIDDEN, HIDDEN // 2), nn.ReLU(),
            nn.Dropout(DROPOUT), nn.Linear(HIDDEN // 2, 2),
        )

    def forward(self, x, edge_index, batch):
        for conv, bn in zip(self.convs, self.bns):
            x = bn(conv(x, edge_index)).relu()
        return self.head(global_mean_pool(x, batch))


class _GraphTransformer(nn.Module):
    _HEADS = 4

    def __init__(self):
        super().__init__()
        heads = self._HEADS
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        for i in range(NUM_LAYERS):
            in_ch = IN_CHANNELS if i == 0 else HIDDEN
            self.convs.append(
                TransformerConv(in_ch, HIDDEN // heads, heads=heads, dropout=DROPOUT)
            )
            self.bns.append(nn.BatchNorm1d(HIDDEN))
        self.head = nn.Sequential(
            nn.Linear(HIDDEN, HIDDEN // 2), nn.ReLU(),
            nn.Dropout(DROPOUT), nn.Linear(HIDDEN // 2, 2),
        )

    def forward(self, x, edge_index, batch):
        for conv, bn in zip(self.convs, self.bns):
            x = bn(conv(x, edge_index)).relu()
        return self.head(global_mean_pool(x, batch))


class _FraudGT(nn.Module):
    _HEADS   = 4
    _MAX_DEG = 63

    def __init__(self):
        super().__init__()
        heads = self._HEADS
        self.input_proj = nn.Linear(IN_CHANNELS, HIDDEN)
        self.deg_enc    = nn.Embedding(self._MAX_DEG + 1, HIDDEN)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.ffs   = nn.ModuleList()
        for _ in range(NUM_LAYERS):
            self.convs.append(
                TransformerConv(HIDDEN, HIDDEN // heads, heads=heads, dropout=DROPOUT)
            )
            self.norms.append(nn.LayerNorm(HIDDEN))
            self.ffs.append(nn.Sequential(
                nn.Linear(HIDDEN, HIDDEN * 2), nn.GELU(),
                nn.Dropout(DROPOUT), nn.Linear(HIDDEN * 2, HIDDEN),
            ))
        self.norm_out = nn.LayerNorm(HIDDEN)
        self.head = nn.Sequential(
            nn.Linear(HIDDEN, HIDDEN // 2), nn.GELU(),
            nn.Dropout(DROPOUT), nn.Linear(HIDDEN // 2, 2),
        )

    def forward(self, x, edge_index, batch):
        x = self.input_proj(x)
        deg = degree(edge_index[0], x.size(0), dtype=torch.long).clamp(max=self._MAX_DEG)
        x = x + self.deg_enc(deg)
        for conv, norm, ff in zip(self.convs, self.norms, self.ffs):
            x = norm(x + conv(x, edge_index))
            x = x + ff(x)
        return self.head(global_mean_pool(self.norm_out(x), batch))


CLASSIFIERS: dict[str, type] = {
    "gin":              _GIN,
    "graphtransformer": _GraphTransformer,
    "graphsage":        _GraphSAGE,
    "fraudgt":          _FraudGT,
}


# ── Training / evaluation helpers (identical to low_data_regime) ───────────────

def _train_epoch(model, loader, opt, device, pos_weight):
    model.train()
    for batch in loader:
        batch = batch.to(device)
        opt.zero_grad()
        logits = model(batch.x, batch.edge_index, batch.batch)
        loss = F.cross_entropy(logits, batch.y, weight=pos_weight)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()


@torch.no_grad()
def _collect_probs(model, loader, device):
    model.eval()
    labels, probs = [], []
    for batch in loader:
        batch = batch.to(device)
        p = F.softmax(model(batch.x, batch.edge_index, batch.batch), dim=-1)[:, 1]
        labels.extend(batch.y.cpu().numpy())
        probs.extend(p.cpu().numpy())
    return np.array(labels), np.array(probs)


def _best_f1(labels, probs):
    best_t, best_f1 = 0.5, 0.0
    for t in np.linspace(0.05, 0.95, 50):
        f1 = f1_score(labels, (probs >= t).astype(int), average="binary", zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1


def train_and_eval(train_data, val_data, test_data, device, seed=0, model_cls=None):
    if model_cls is None:
        model_cls = _GIN
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    n_pos = sum(d.y.item() == 1 for d in train_data)
    n_neg = len(train_data) - n_pos
    pos_w = (torch.tensor([1.0, n_neg / n_pos], dtype=torch.float, device=device)
             if n_pos > 0 and n_neg > 0 else None)

    model = model_cls().to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    tr_ld  = PygDataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_ld = PygDataLoader(val_data,   batch_size=BATCH_SIZE)
    te_ld  = PygDataLoader(test_data,  batch_size=BATCH_SIZE)

    best_f1, best_thresh, best_state = -1.0, 0.5, None
    for _ in range(EPOCHS):
        _train_epoch(model, tr_ld, opt, device, pos_w)
        sched.step()
        vl, vp = _collect_probs(model, val_ld, device)
        t, f1  = _best_f1(vl, vp)
        if f1 > best_f1:
            best_f1, best_thresh = f1, t
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    tl, tp = _collect_probs(model, te_ld, device)
    preds  = (tp >= best_thresh).astype(int)
    auc    = roc_auc_score(tl, tp) if len(np.unique(tl)) > 1 else 0.5
    f1_te  = f1_score(tl, preds, average="binary", zero_division=0)
    return auc, f1_te


# ── Data loading (identical to low_data_regime) ───────────────────────────────

def load_pyg_graphs(csv_path: Path, device):
    from util import preprocess_df, extract_transaction_ego_networks
    from augmentation import build_igraph_from_transactions
    from diffusion.diff_util import network_to_dense as ntd

    cache_path = DATA_DIR / f"networks_cache_{csv_path.stem}_v2.pkl"

    if cache_path.exists():
        print(f"  Loading network cache: {cache_path.name}")
        with open(cache_path, "rb") as f:
            networks = pickle.load(f)
        for net in networks:
            net["graph"] = build_igraph_from_transactions(net["transactions"])
    else:
        print(f"  Extracting networks from {csv_path.name} ...")
        df = preprocess_df(str(csv_path))
        networks = extract_transaction_ego_networks(
            df, max_depth=2, max_nodes=50, n_pos=2000, neg_pos_ratio=10,
        )
        for net in networks:
            net["graph"] = build_igraph_from_transactions(net["transactions"])
        to_cache = [{k: v for k, v in n.items() if k != "graph"} for n in networks]
        with open(cache_path, "wb") as f:
            pickle.dump(to_cache, f)
        print(f"  Saved cache -> {cache_path.name}")

    pyg_list       = []
    networks_valid = []
    skipped        = 0
    for net in networks:
        try:
            x_d, adj_d = ntd(net)
        except Exception:
            skipped += 1
            continue
        if x_d.shape[0] < 3:
            skipped += 1
            continue
        label = net.get("tx_label",
                        1 if len(net.get("laundering_nodes", set())) > 0 else 0)
        x_np  = x_d[:, 1:].float().numpy()
        adj_np = adj_d.numpy() if isinstance(adj_d, torch.Tensor) else np.asarray(adj_d)
        src, dst = np.where(adj_np > 0.5)
        ei = torch.tensor(np.stack([src, dst]), dtype=torch.long)
        if ei.shape[1] == 0:
            n = x_np.shape[0]; idx = torch.arange(n)
            ei = torch.stack([idx, idx])
        ts_val = float(net["timestamp"].timestamp()) if "timestamp" in net else -1.0
        g = Data(x=torch.tensor(x_np, dtype=torch.float),
                 edge_index=ei,
                 y=torch.tensor([label], dtype=torch.long),
                 timestamp_val=ts_val,
                 net_idx=len(networks_valid))
        pyg_list.append(g)
        net["x_dense"]   = x_d
        net["adj_dense"] = adj_d
        networks_valid.append(net)

    if skipped:
        print(f"  [{skipped} networks skipped]")
    print(f"  {len(pyg_list)} PyG graphs loaded")
    return pyg_list, networks_valid


def temporal_split(pyg_list, test_frac=0.20, val_frac=0.10):
    ts = np.array([getattr(d, "timestamp_val", -1.0) for d in pyg_list])
    labels = np.array([d.y.item() for d in pyg_list])
    n = len(pyg_list)

    if ts.min() >= 0:
        idx = np.argsort(ts)
        n_tv = int(n * (1.0 - test_frac))
        n_tr = int(n_tv * (1.0 - val_frac / (1.0 - test_frac)))
        idx_tr, idx_val, idx_te = idx[:n_tr], idx[n_tr:n_tv], idx[n_tv:]
    else:
        sss = StratifiedShuffleSplit(1, test_size=test_frac, random_state=42)
        idx_tv, idx_te = next(sss.split(np.arange(n), labels))
        sss2 = StratifiedShuffleSplit(1, test_size=val_frac / (1.0 - test_frac), random_state=42)
        idx_sub_tr, idx_sub_val = next(sss2.split(np.arange(len(idx_tv)), labels[idx_tv]))
        idx_tr, idx_val = idx_tv[idx_sub_tr], idx_tv[idx_sub_val]

    return ([pyg_list[i] for i in idx_tr],
            [pyg_list[i] for i in idx_val],
            [pyg_list[i] for i in idx_te])


# ── Sparse-label subsampling ──────────────────────────────────────────────────

def reveal_labels(train_data: list, frac: float, seed: int = 42) -> list:
    """
    Randomly reveal only `frac` of all training graph labels (both classes).
    The rest are treated as unlabeled and excluded from training.
    Stratified so both classes are represented proportionally.
    """
    if frac >= 1.0:
        return train_data
    labels = np.array([d.y.item() for d in train_data])
    n_keep = max(2, int(len(train_data) * frac))
    # stratified sample to preserve class ratio in the revealed set
    rng = np.random.default_rng(seed)
    laund_idx = np.where(labels == 1)[0]
    clean_idx = np.where(labels == 0)[0]
    n_laund_keep = max(1, round(n_keep * len(laund_idx) / len(train_data)))
    n_clean_keep = max(1, n_keep - n_laund_keep)
    n_laund_keep = min(n_laund_keep, len(laund_idx))
    n_clean_keep = min(n_clean_keep, len(clean_idx))
    kept = np.concatenate([
        rng.choice(laund_idx, n_laund_keep, replace=False),
        rng.choice(clean_idx, n_clean_keep, replace=False),
    ])
    return [train_data[i] for i in sorted(kept)]


# ── Augmentation helpers (identical to low_data_regime) ───────────────────────

def load_generation_bundle(train_nets: list, device):
    if not DIFF_CKPT.exists() and not SIMCLR_CKPT_DIR.exists():
        return None
    try:
        from generation import (
            load_simclr_encoder,
            load_diffusion_model,
            encode_all_networks,
            train_mlp_probe,
        )
        encoder = load_simclr_encoder(device, ckpt_dir=SIMCLR_CKPT_DIR)
        diff_model, diffusion, x_mean, x_std, _ = load_diffusion_model(
            device, ckpt_path=DIFF_CKPT if DIFF_CKPT.exists() else None)
        H_train, y_train = encode_all_networks(train_nets, encoder, device)
        probe = train_mlp_probe(H_train, y_train, device)
        return encoder, probe, diff_model, diffusion, x_mean, x_std, H_train
    except Exception as e:
        print(f"  [WARN] Could not load generation bundle: {e}")
        return None


def generate_guided_graphs(train_nets: list, bundle, n_gen: int, device) -> list[Data]:
    from generation import run_guided_generation
    encoder, probe, diff_model, diffusion, x_mean, x_std, H_train = bundle

    gen_outputs, _, _ = run_guided_generation(
        train_nets, encoder, probe, diff_model, diffusion,
        x_mean, x_std, H_train, device,
        target_label=1, n_gen=n_gen,
    )

    result = []
    for x_denorm, adj_out, n_out, _lbl in gen_outputs:
        adj_np = adj_out[:n_out, :n_out]
        if isinstance(adj_np, torch.Tensor):
            adj_np = adj_np.numpy()
        np.fill_diagonal(adj_np, 0.0)
        ei = (torch.tensor(adj_np) > 0.5).nonzero(as_tuple=False).T.contiguous()
        if ei.shape[1] == 0:
            ei = torch.zeros(2, 0, dtype=torch.long)
        x = x_denorm[:n_out].float()
        if x.shape[1] < IN_CHANNELS:
            x = torch.cat([x, torch.zeros(x.shape[0], IN_CHANNELS - x.shape[1])], dim=1)
        x = x[:, :IN_CHANNELS]
        result.append(Data(x=x, edge_index=ei,
                           y=torch.tensor([1], dtype=torch.long),
                           timestamp_val=-1.0, net_idx=-1))
    return result


def generate_graphsmote(train_data: list, n_gen: int) -> list[Data]:
    sys.path.insert(0, str(_HERE))
    from graphsmote_augmentation import GraphSMOTEAugmenter
    laund = [d for d in train_data if d.y.item() == 1]
    if len(laund) < 2:
        return []
    aug = GraphSMOTEAugmenter(k=min(5, len(laund) - 1), random_state=42)
    graphs = aug.fit_generate(laund, n_gen)
    for g in graphs:
        if not hasattr(g, "net_idx"):
            g.net_idx = -1
    return graphs


def generate_gan(train_data: list, n_gen: int, device) -> list[Data]:
    sys.path.insert(0, str(_HERE))
    from gan_augmentation import GraphGANAugmenter
    laund = [d for d in train_data if d.y.item() == 1]
    if len(laund) < 4:
        return []
    aug = GraphGANAugmenter(
        latent_dim=32, epochs=300, batch_size=min(32, len(laund)),
        lr=1e-4, n_critic=5, hidden=128,
        random_state=42, device=device, verbose=0,
    )
    return aug.fit_generate(laund, n_gen)


def generate_diga(train_data: list, n_gen: int, device) -> list[Data]:
    sys.path.insert(0, str(_HERE))
    from diga_augmentation import DiGaAugmenter
    laund = [d for d in train_data if d.y.item() == 1]
    if len(laund) < 4:
        return []
    aug = DiGaAugmenter(
        T=300, epochs=400, batch_size=min(32, len(laund)),
        lr=1e-3, hidden=128, time_dim=32,
        random_state=42, device=device, verbose=0,
    )
    return aug.fit_generate(laund, n_gen)


# ── Main experiment loop ───────────────────────────────────────────────────────

def run_experiment(csv_path: Path, fractions: list[float], n_gen: int,
                   n_seeds: int, out_dir: Path, device: torch.device,
                   classifier_name: str = "gin"):

    model_cls = CLASSIFIERS[classifier_name]

    print(f"\nLoading data from {csv_path.name} ...")
    pyg_list, networks_valid = load_pyg_graphs(csv_path, device)

    all_labels = np.array([d.y.item() for d in pyg_list])
    n_laund    = int(all_labels.sum())
    print(f"  {len(pyg_list)} graphs  ({n_laund} laundering, "
          f"{len(pyg_list) - n_laund} clean)\n")

    train_full, val_data, test_data = temporal_split(pyg_list)
    print(f"Split: {len(train_full)} train | {len(val_data)} val | {len(test_data)} test")

    train_full_nets = [networks_valid[d.net_idx] for d in train_full]

    print("\nLoading guided generation bundle ...")
    gen_bundle = load_generation_bundle(train_full_nets, device)
    if gen_bundle is not None:
        print("  Guided generation bundle ready.")
    else:
        print("  Could not load generation bundle -- 'diffusion' condition will be skipped.")

    CONDITIONS = ["baseline", "graphsmote", "gan", "diga"]
    if gen_bundle is not None:
        CONDITIONS.append("diffusion")

    records = []

    for frac in fractions:
        for condition in CONDITIONS:
            aucs, f1s = [], []
            for seed in range(n_seeds):
                labeled = reveal_labels(train_full, frac, seed=seed)
                n_pos = sum(d.y.item() == 1 for d in labeled)
                n_total = len(labeled)

                if condition == "baseline":
                    train_aug = labeled
                elif condition == "diffusion":
                    labeled_nets = [networks_valid[d.net_idx] for d in labeled
                                    if d.net_idx >= 0]
                    gen = generate_guided_graphs(labeled_nets, gen_bundle, n_gen, device)
                    train_aug = labeled + gen
                elif condition == "graphsmote":
                    gen = generate_graphsmote(labeled, n_gen)
                    train_aug = labeled + gen
                elif condition == "gan":
                    gen = generate_gan(labeled, n_gen, device)
                    train_aug = labeled + gen
                elif condition == "diga":
                    gen = generate_diga(labeled, n_gen, device)
                    train_aug = labeled + gen
                else:
                    train_aug = labeled

                n_pos_aug = sum(d.y.item() == 1 for d in train_aug)
                if n_pos_aug < 2:
                    print(f"  [SKIP] frac={frac:.0%} cond={condition} seed={seed}: "
                          f"only {n_pos_aug} positive(s) after augmentation")
                    continue

                auc, f1 = train_and_eval(
                    train_aug, val_data, test_data, device,
                    seed=seed, model_cls=model_cls,
                )
                aucs.append(auc)
                f1s.append(f1)

                print(f"  frac={frac:.0%}  cond={condition:<12}  seed={seed}  "
                      f"labeled={n_total} (pos={n_pos})  "
                      f"AUC={auc:.4f}  F1={f1:.4f}")

            if aucs:
                records.append({
                    "fraction":    frac,
                    "condition":   condition,
                    "auc_mean":    float(np.mean(aucs)),
                    "auc_std":     float(np.std(aucs)),
                    "f1_mean":     float(np.mean(f1s)),
                    "f1_std":      float(np.std(f1s)),
                    "n_seeds_run": len(aucs),
                })

    return records, CONDITIONS


# ── Output helpers ────────────────────────────────────────────────────────────

def save_csv(records: list[dict], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        print("[WARN] No records to save.")
        return
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print(f"\nResults saved -> {out_path}")


def plot_curves(records: list[dict], conditions: list[str], out_path: Path,
                classifier_name: str = "gin"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [SKIP] matplotlib not available")
        return

    fracs = sorted(set(r["fraction"] for r in records))
    colours = {
        "baseline":   "#4a90d9",
        "diffusion":  "#e05c2e",
        "graphsmote": "#2ecc71",
        "gan":        "#9b59b6",
        "diga":       "#1abc9c",
    }
    markers = {
        "baseline":   "o",
        "diffusion":  "s",
        "graphsmote": "^",
        "gan":        "D",
        "diga":       "X",
    }

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for ax, metric, ylabel in [
        (axes[0], "auc", "AUC-ROC"),
        (axes[1], "f1",  "F1 Score"),
    ]:
        for cond in conditions:
            xs, ys, errs = [], [], []
            for frac in fracs:
                m = [r for r in records if r["fraction"] == frac and r["condition"] == cond]
                if m:
                    xs.append(frac * 100)
                    ys.append(m[0][f"{metric}_mean"])
                    errs.append(m[0][f"{metric}_std"])
            if xs:
                ax.errorbar(xs, ys, yerr=errs, label=cond,
                            color=colours.get(cond, "grey"),
                            marker=markers.get(cond, "o"),
                            linewidth=1.8, markersize=6, capsize=3)

        ax.set_xlabel("Labeled fraction (%)", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f"{ylabel} vs. label fraction", fontsize=11)
        ax.set_xlim(0, 105)
        ax.set_xticks([5, 10, 25, 50, 100])
        ax.set_xticklabels(["5", "10", "25%", "50%", "100%"])
        ax.legend(fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle(f"Sparse-Label Regime: Augmentation Recovery Curves\n"
                 f"LI-Small AML Dataset  ({classifier_name.upper()})",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    print(f"Plot saved -> {out_path}")
    plt.close(fig)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=str, default=DEFAULT_CSV)
    parser.add_argument("--n-gen", type=int, default=DEFAULT_N_GEN,
                        help=f"Generated graphs per condition (default {DEFAULT_N_GEN})")
    parser.add_argument("--fractions", nargs="+", type=float, default=DEFAULT_FRACTIONS,
                        help="Label fractions to evaluate")
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--classifier", type=str, default=DEFAULT_CLASSIFIER,
                        choices=list(CLASSIFIERS.keys()),
                        help=f"Downstream classifier (default: {DEFAULT_CLASSIFIER})")
    parser.add_argument("--out", type=str, default=str(_HERE / "results"))
    args = parser.parse_args()

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir  = Path(args.out)
    csv_path = Path(args.csv)

    print(f"Device     : {device}")
    print(f"CSV        : {csv_path}")
    print(f"N_gen      : {args.n_gen}")
    print(f"Fractions  : {args.fractions}")
    print(f"Seeds      : {args.seeds}")
    print(f"Classifier : {args.classifier}")
    print(f"Output dir : {out_dir}\n")

    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV not found: {csv_path}\n"
            "Set --csv to the path of LI-Small_Trans.csv."
        )

    records, conditions = run_experiment(
        csv_path, args.fractions, args.n_gen, args.seeds,
        out_dir, device, classifier_name=args.classifier,
    )

    save_csv(records, out_dir / "sparse_label_results.csv")
    plot_curves(records, conditions,
                out_dir / "sparse_label_curves.png",
                classifier_name=args.classifier)
    print("\nDone.")


if __name__ == "__main__":
    main()
