"""
low_data_regime.py
──────────────────
Experiment 3: how much does each augmentation method recover performance when
labelled training data is scarce?

Conditions
──────────
  baseline    — real training graphs only (no augmentation)
  diffusion   — real + N_GEN diffusion-generated laundering graphs
                (skipped gracefully if no checkpoint is found)
  graphsmote  — real + N_GEN GraphSMOTE synthetic laundering graphs

Training fractions
──────────────────
  5% | 10% | 25% | 50% | 100% of the original laundering training examples

Classifier: GIN (fast; correlates well with the full suite on IBM data)
Seeds: 3 independent runs per (condition, fraction) combination
Metric: AUC-ROC + F1 (threshold tuned on validation set)

Usage
─────
    # from igraph_version/ directory:
    python experiments/low_data_regime.py

    # with options:
    python experiments/low_data_regime.py \\
        --csv data/IBM/LI-Small_Trans.csv \\
        --n-gen 50 \\
        --fractions 0.05 0.1 0.25 0.5 1.0 \\
        --seeds 3 \\
        --out experiments/results

Outputs
───────
    experiments/results/low_data_results.csv     — full table (condition × fraction)
    experiments/results/low_data_curves.png      — recovery curve plot
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

IN_CHANNELS   = 18     # laundering flag col excluded from classifier input
HIDDEN        = 64
NUM_LAYERS    = 3
DROPOUT       = 0.3
LR            = 1e-3
WEIGHT_DECAY  = 1e-4
EPOCHS        = 80
BATCH_SIZE    = 32

DEFAULT_FRACTIONS = [0.05, 0.10, 0.25, 0.50, 1.0]
DEFAULT_N_GEN     = 50
DEFAULT_SEEDS     = 3


# ── GIN classifier (minimal copy from evaluate_classifiers.py) ────────────────

from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.loader import DataLoader as PygDataLoader
from torch_geometric.data import Data
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit


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


def train_and_eval(train_data, val_data, test_data, device, seed=0):
    """Train one GIN and return test AUC and F1."""
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    n_pos = sum(d.y.item() == 1 for d in train_data)
    n_neg = len(train_data) - n_pos
    pos_w = (torch.tensor([1.0, n_neg / n_pos], dtype=torch.float, device=device)
             if n_pos > 0 and n_neg > 0 else None)

    model = _GIN().to(device)
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


# ── Data loading ───────────────────────────────────────────────────────────────

def load_pyg_graphs(csv_path: Path, device):
    """
    Load IBM graphs from cache or build from CSV.
    Returns (pyg_list, networks_list) where networks_list is used for
    FraudGT / generation — may be None for cached PyG-only loading.
    """
    from util import preprocess_df, extract_transaction_ego_networks
    from augmentation import build_igraph_from_transactions
    from diffusion.diff_util import network_to_dense as ntd

    # evaluate_classifiers.py cache format (v2)
    cache_path = DATA_DIR / f"networks_cache_{csv_path.stem}_v2.pkl"

    if cache_path.exists():
        print(f"  Loading network cache: {cache_path.name}")
        with open(cache_path, "rb") as f:
            networks = pickle.load(f)
        for net in networks:
            net["graph"] = build_igraph_from_transactions(net["transactions"])
    else:
        print(f"  Extracting networks from {csv_path.name} …")
        df = preprocess_df(str(csv_path))
        networks = extract_transaction_ego_networks(
            df, max_depth=2, max_nodes=50, n_pos=2000, neg_pos_ratio=10,
        )
        for net in networks:
            net["graph"] = build_igraph_from_transactions(net["transactions"])
        to_cache = [{k: v for k, v in n.items() if k != "graph"} for n in networks]
        with open(cache_path, "wb") as f:
            pickle.dump(to_cache, f)
        print(f"  Saved cache → {cache_path.name}")

    pyg_list       = []
    networks_valid = []   # aligned with pyg_list (skipped entries excluded)
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
        # net_idx allows retrieving the corresponding network dict after splits/subsamples
        g = Data(x=torch.tensor(x_np, dtype=torch.float),
                 edge_index=ei,
                 y=torch.tensor([label], dtype=torch.long),
                 timestamp_val=ts_val,
                 net_idx=len(networks_valid))
        pyg_list.append(g)
        # Cache dense tensors on the network dict (used by guided_generate)
        net["x_dense"]   = x_d
        net["adj_dense"] = adj_d
        networks_valid.append(net)

    if skipped:
        print(f"  [{skipped} networks skipped]")
    print(f"  {len(pyg_list)} PyG graphs loaded")
    return pyg_list, networks_valid


def temporal_split(pyg_list, test_frac=0.20, val_frac=0.10):
    """Temporal train/val/test split, with stratified fallback."""
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

    tr  = [pyg_list[i] for i in idx_tr]
    val = [pyg_list[i] for i in idx_val]
    te  = [pyg_list[i] for i in idx_te]
    return tr, val, te


def subsample_laundering(train_data: list, frac: float, seed: int = 42) -> list:
    """
    Subsample the training set to `frac` of its laundering examples.
    Clean graphs are kept in full to preserve the realistic class ratio.
    """
    if frac >= 1.0:
        return train_data
    laund = [d for d in train_data if d.y.item() == 1]
    clean = [d for d in train_data if d.y.item() == 0]
    n_keep = max(2, int(len(laund) * frac))
    rng = np.random.default_rng(seed)
    kept_laund = [laund[i] for i in rng.choice(len(laund), n_keep, replace=False)]
    return kept_laund + clean


# ── Augmentation helpers ───────────────────────────────────────────────────────

def load_generation_bundle(train_nets: list, device):
    """
    Load SimCLR encoder + probe + diffusion model for guided generation.
    Uses the same pipeline as generation/test.py (run_guided_generation).
    Returns (encoder, probe, diff_model, diffusion, x_mean, x_std, H_train) or None.
    """
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
    """
    Generate n_gen laundering graphs using the full guided reverse diffusion
    (identical pipeline to generation/test.py run_guided_generation).
    """
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
                           timestamp_val=-1.0,
                           net_idx=-1))
    return result


def generate_graphsmote(train_data: list, n_gen: int) -> list[Data]:
    """Generate n_gen GraphSMOTE synthetic laundering graphs."""
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
    """Generate n_gen synthetic laundering graphs using a WGAN-GP in feature space."""
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


def generate_vae(train_data: list, n_gen: int, device) -> list[Data]:
    """Generate n_gen synthetic laundering graphs using a β-VAE in feature space."""
    sys.path.insert(0, str(_HERE))
    from vae_augmentation import GraphVAEAugmenter
    laund = [d for d in train_data if d.y.item() == 1]
    if len(laund) < 4:
        return []
    aug = GraphVAEAugmenter(
        latent_dim=16, epochs=300, batch_size=min(64, len(laund)),
        lr=1e-3, beta=0.5, hidden=128,
        random_state=42, device=device, verbose=0,
    )
    return aug.fit_generate(laund, n_gen)


# ── Main experiment loop ───────────────────────────────────────────────────────

def run_experiment(csv_path: Path, fractions: list[float], n_gen: int,
                   n_seeds: int, out_dir: Path, device: torch.device):

    print(f"\nLoading data from {csv_path.name} …")
    pyg_list, networks_valid = load_pyg_graphs(csv_path, device)

    all_labels = np.array([d.y.item() for d in pyg_list])
    n_laund    = int(all_labels.sum())
    print(f"  {len(pyg_list)} graphs  ({n_laund} laundering, "
          f"{len(pyg_list) - n_laund} clean)\n")

    train_full, val_data, test_data = temporal_split(pyg_list)
    print(f"Split: {len(train_full)} train | {len(val_data)} val | {len(test_data)} test")

    # Retrieve network dicts for the training split (needed for guided generation)
    train_full_nets = [networks_valid[d.net_idx] for d in train_full]

    # Load full guided generation bundle (encoder + probe + diffusion) once.
    # The probe is fitted on ALL training networks so it has a stable sense of
    # laundering direction regardless of which fraction the GIN classifier sees.
    print("\nLoading guided generation bundle (encoder + probe + diffusion) …")
    gen_bundle = load_generation_bundle(train_full_nets, device)
    if gen_bundle is not None:
        print("  Guided generation bundle ready.")
    else:
        print("  Could not load generation bundle — 'diffusion' condition will be skipped.")

    CONDITIONS = ["baseline", "graphsmote", "gan", "vae"]
    if gen_bundle is not None:
        CONDITIONS.append("diffusion")

    # Results storage
    records = []  # list of dicts

    for frac in fractions:
        for condition in CONDITIONS:
            aucs, f1s = [], []
            for seed in range(n_seeds):
                train_sub = subsample_laundering(train_full, frac, seed=seed)
                n_laund_sub = sum(d.y.item() == 1 for d in train_sub)

                # Build augmented training set
                if condition == "baseline":
                    train_aug = train_sub
                elif condition == "diffusion":
                    # Use network dicts for the subsample so guided generation
                    # seeds from the same limited data the classifier sees.
                    train_sub_nets = [networks_valid[d.net_idx] for d in train_sub]
                    gen = generate_guided_graphs(train_sub_nets, gen_bundle, n_gen, device)
                    train_aug = train_sub + gen
                elif condition == "graphsmote":
                    gen = generate_graphsmote(train_sub, n_gen)
                    train_aug = train_sub + gen
                elif condition == "gan":
                    gen = generate_gan(train_sub, n_gen, device)
                    train_aug = train_sub + gen
                elif condition == "vae":
                    gen = generate_vae(train_sub, n_gen, device)
                    train_aug = train_sub + gen

                # Skip if too few positives
                n_pos = sum(d.y.item() == 1 for d in train_aug)
                if n_pos < 2:
                    print(f"  [SKIP] frac={frac:.0%} cond={condition} seed={seed}: "
                          f"only {n_pos} positive(s)")
                    continue

                auc, f1 = train_and_eval(train_aug, val_data, test_data, device, seed=seed)
                aucs.append(auc); f1s.append(f1)

                print(f"  frac={frac:.0%}  cond={condition:<12}  "
                      f"seed={seed}  laund_train={n_laund_sub}  "
                      f"AUC={auc:.4f}  F1={f1:.4f}")

            if aucs:
                records.append({
                    "fraction":       frac,
                    "condition":      condition,
                    "auc_mean":       float(np.mean(aucs)),
                    "auc_std":        float(np.std(aucs)),
                    "f1_mean":        float(np.mean(f1s)),
                    "f1_std":         float(np.std(f1s)),
                    "n_seeds_run":    len(aucs),
                })

    return records, CONDITIONS


# ── CSV + plot ────────────────────────────────────────────────────────────────

def save_csv(records: list[dict], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        print("[WARN] No records to save.")
        return
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print(f"\nResults saved → {out_path}")


def print_table(records: list[dict], conditions: list[str]):
    fracs = sorted(set(r["fraction"] for r in records))
    col_w = 12
    header = f"{'Fraction':<10}" + "".join(
        f"  {c+' AUC':>{col_w}}  {c+' F1':>{col_w}}" for c in conditions
    )
    sep = "─" * len(header)
    print("\n" + sep)
    print("LOW-DATA REGIME RESULTS  (mean ± std across seeds)")
    print(sep)
    print(header)
    print(sep)
    for frac in fracs:
        row_str = f"{frac:<10.0%}"
        for cond in conditions:
            match = [r for r in records if r["fraction"] == frac and r["condition"] == cond]
            if match:
                r = match[0]
                row_str += f"  {r['auc_mean']:.3f}±{r['auc_std']:.3f}"
                row_str += f"  {r['f1_mean']:.3f}±{r['f1_std']:.3f}"
            else:
                row_str += f"  {'N/A':>{col_w}}  {'N/A':>{col_w}}"
        print(row_str)
    print(sep)


def plot_curves(records: list[dict], conditions: list[str], out_path: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [SKIP] matplotlib not available — skipping plot")
        return

    fracs = sorted(set(r["fraction"] for r in records))
    colours = {
        "baseline":   "#4a90d9",
        "diffusion":  "#e05c2e",
        "graphsmote": "#2ecc71",
        "gan":        "#9b59b6",
        "vae":        "#f39c12",
    }
    markers = {
        "baseline":   "o",
        "diffusion":  "s",
        "graphsmote": "^",
        "gan":        "D",
        "vae":        "P",
    }

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for metric, ax, ylabel in [
        ("auc", axes[0], "AUC-ROC"),
        ("f1",  axes[1], "F1 Score"),
    ]:
        for cond in conditions:
            xs, ys, errs = [], [], []
            for frac in fracs:
                match = [r for r in records
                         if r["fraction"] == frac and r["condition"] == cond]
                if match:
                    xs.append(frac * 100)
                    ys.append(match[0][f"{metric}_mean"])
                    errs.append(match[0][f"{metric}_std"])
            if xs:
                colour = colours.get(cond, "grey")
                marker = markers.get(cond, "o")
                ax.errorbar(xs, ys, yerr=errs,
                            label=cond, color=colour, marker=marker,
                            linewidth=1.8, markersize=6, capsize=3)

        ax.set_xlabel("Training laundering fraction (%)", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(f"{ylabel} vs. training fraction", fontsize=11)
        ax.legend(fontsize=9)
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
        ax.grid(True, which="both", linestyle="--", alpha=0.4)

    fig.suptitle("Low-Data Regime: Augmentation Recovery Curves\n"
                 "LI-Small AML Dataset", fontsize=12, y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    print(f"  Plot saved → {out_path}")
    plt.close(fig)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=str, default=DEFAULT_CSV,
                        help="Path to IBM AML CSV (default: LI-Small_Trans.csv)")
    parser.add_argument("--n-gen", type=int, default=DEFAULT_N_GEN,
                        help=f"Generated graphs added per condition (default {DEFAULT_N_GEN})")
    parser.add_argument("--fractions", nargs="+", type=float, default=DEFAULT_FRACTIONS,
                        help="Training laundering fractions to evaluate")
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS,
                        help=f"Seeds per (condition, fraction) combo (default {DEFAULT_SEEDS})")
    parser.add_argument("--out", type=str, default=str(_HERE / "results"),
                        help="Output directory")
    args = parser.parse_args()

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out)
    csv_path = Path(args.csv)

    print(f"Device     : {device}")
    print(f"CSV        : {csv_path}")
    print(f"N_gen      : {args.n_gen}")
    print(f"Fractions  : {args.fractions}")
    print(f"Seeds      : {args.seeds}")
    print(f"Output dir : {out_dir}\n")

    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV not found: {csv_path}\n"
            "Set --csv to the path of LI-Small_Trans.csv (or another IBM AML CSV)."
        )

    records, conditions = run_experiment(
        csv_path, args.fractions, args.n_gen, args.seeds, out_dir, device,
    )

    print_table(records, conditions)
    save_csv(records, out_dir / "low_data_results.csv")
    plot_curves(records, conditions, out_dir / "low_data_curves.png")
    print("\nDone.")


if __name__ == "__main__":
    main()
