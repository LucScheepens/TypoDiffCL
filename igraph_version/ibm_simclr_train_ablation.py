"""
ibm_simclr_train_ablation.py
──────────────────────────────────────────────────────────────────────────────
Parametrized IBM SimCLR training for ablation studies.

Saves the best checkpoint to:
  checkpoints/simclr_ibm_ablation/<condition>/best_model.pt

The IBM augmentation pipeline uses augment_network_view_fast() (crop, edge
delete, node delete, node add) for both views.  With p_diffusion > 0, view 2
is replaced by the diffusion model's single-step noisy reconstruction with
probability p_diffusion.

Usage examples
──────────────
  # Full system — replicates default IBM SimCLR training
  python ibm_simclr_train_ablation.py --condition full

  # No supervised contrastive loss
  python ibm_simclr_train_ablation.py --condition no_supcon --supcon-weight 0.0

  # No diffusion augmentation (structural aug only)
  python ibm_simclr_train_ablation.py --condition no_diffusion_aug --p-diffusion 0.0

  # Diffusion-only views (structural aug disabled, p_diffusion=1.0)
  python ibm_simclr_train_ablation.py --condition diffusion_aug_only --p-diffusion 1.0

  # NT-Xent only — no SupCon, no diffusion aug
  python ibm_simclr_train_ablation.py --condition ntxent_only \\
      --supcon-weight 0.0 --p-diffusion 0.0
"""

import argparse
import sys
import time
from pathlib import Path

import torch

# ── path setup ─────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent          # igraph_version/
DIFF_DIR = BASE_DIR / "diffusion"

for _p in (str(BASE_DIR), str(DIFF_DIR), str(BASE_DIR / "simclr"),
           str(BASE_DIR / "data")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from simclr import GraphEncoder, ProjectionHead, train_simclr_fast
from augmentation import build_igraph_from_transactions
from util import preprocess_df, extract_networks_igraph

CKPT_ROOT    = BASE_DIR / "checkpoints" / "simclr_ibm_ablation"
DATA_DIR     = BASE_DIR / "data"
DEFAULT_CSV  = str(Path(__file__).resolve().parent.parent / "data" / "IBM" / "LI-Small_Trans.csv")
MAX_NODES    = 64
MAX_NETWORKS = 2000   # laundering networks; same count of clean extracted


# ── helpers ────────────────────────────────────────────────────────────────────

def _cache_stale(nets):
    """Return True if cached networks have wrong node feature dimension."""
    for n in nets:
        xd = n.get("x_dense")
        if xd is not None:
            return xd.shape[1] != 19
    return False


def _load_networks(ibm_csv):
    """Load IBM networks from cache or re-extract from CSV."""
    cache_pt = DATA_DIR / "simclr_networks_cache.pt"

    if cache_pt.exists():
        print(f"Loading networks from cache: {cache_pt}")
        networks = torch.load(str(cache_pt), weights_only=False)
        if _cache_stale(networks):
            print("  Stale cache (wrong node_dim) — rebuilding …")
            cache_pt.unlink()
            networks = None
        else:
            return networks

    print(f"Extracting IBM networks from {ibm_csv} …")
    df_full  = preprocess_df(ibm_csv)
    networks = extract_networks_igraph(
        df_full,
        max_depth=4,
        max_networks=MAX_NETWORKS * 2,
        collapse_threshold=10,
        max_nodes=MAX_NODES,
    )
    for net in networks:
        net["graph"] = build_igraph_from_transactions(net["transactions"])
    torch.save(networks, str(cache_pt))
    print(f"Saved {len(networks)} networks → {cache_pt}")
    return networks


def _load_diffusion(device):
    """
    Load the IBM diffusion model.
    Returns (diff_model, diffusion, x_mean, x_std, node_dim, max_nodes).
    All model values are None if the checkpoint is missing or fails to load.
    """
    model_path = BASE_DIR / "checkpoints" / "diffusion_ibm" / "model.pt"
    if not model_path.exists():
        print(f"[diffusion] Checkpoint not found at {model_path} — diffusion views disabled.")
        return None, None, None, None, 19, MAX_NODES

    try:
        from diffusion.model     import DiffusionGNN
        from diffusion.diff_util import create_diffusion

        ckpt       = torch.load(model_path, map_location=device, weights_only=False)
        node_dim   = ckpt["model"]["input_proj.weight"].shape[1]
        diff_model = DiffusionGNN(node_dim=node_dim, hidden_dim=128, num_layers=4).to(device)
        diff_model.load_state_dict(ckpt["model"])
        diff_model.eval()
        diffusion  = create_diffusion(T=500)
        x_mean     = ckpt["x_mean"].to(device)
        x_std      = ckpt["x_std"].to(device)
        max_nodes  = ckpt.get("max_nodes", MAX_NODES)
        print(f"[diffusion] Loaded {model_path}  (node_dim={node_dim}, max_nodes={max_nodes})")
        return diff_model, diffusion, x_mean, x_std, node_dim, max_nodes

    except Exception as e:
        print(f"[diffusion] Load failed ({e}) — diffusion views disabled.")
        return None, None, None, None, 19, MAX_NODES


# ── pattern feature masking ────────────────────────────────────────────────────

# Column indices of the 8 AML pattern features inside the 19-dim x_dense tensor.
# These match the layout defined in network_to_dense() / detector.py.
_PAT_COLS = {
    "fan_out":       [11],        # out_degree_norm
    "fan_in":        [12],        # in_degree_norm
    "fan_asymmetry": [13],        # degree_asymmetry
    "stack":         [14, 15],    # is_passthrough + stack_depth_norm
    "cycle":         [16],        # in_cycle (SCC size > 1)
    "scatter_gather":[17],        # scatter_gather_score
    "bipartite":     [18],        # bipartite_score
}
_PAT_ALL_COLS = list(range(11, 19))


def _apply_pattern_masking(networks, cols_to_zero):
    """
    Force-compute x_dense for every network then zero out the specified
    pattern feature columns.  This must be called before training so that
    network_to_pyg_data_fast picks up the masked cache instead of
    recomputing from network_to_dense.
    """
    if not cols_to_zero:
        return
    from diffusion.diff_util import network_to_dense

    for net in networks:
        if "x_dense" not in net or net["x_dense"] is None:
            x, _ = network_to_dense(net)
            net["x_dense"] = x
        for col in cols_to_zero:
            net["x_dense"][:, col] = 0.0


# ── training ──────────────────────────────────────────────────────────────────

def train(args, device):
    print(f"\n{'='*60}")
    print(f"IBM SimCLR ablation: {args.condition}")
    print(f"  supcon_weight={args.supcon_weight}  p_diffusion={args.p_diffusion}")
    print(f"  view_type={args.view_type}  diff_n_steps={args.diff_n_steps}"
          f"  diff_t_start={args.diff_t_start}  diff_guidance={args.diff_guidance_scale}")
    print(f"  epochs={args.epochs}  batch_size={args.batch_size}  lr={args.lr}")
    print(f"{'='*60}\n")

    # ── data ──────────────────────────────────────────────────────────────────
    networks = _load_networks(args.ibm_csv)
    n_laund  = sum(len(n["laundering_nodes"]) > 0 for n in networks)
    print(f"  {len(networks)} networks  ({n_laund} laundering, {len(networks)-n_laund} clean)")

    # ── pattern feature masking (ablation) ────────────────────────────────────
    cols_to_zero = []
    if args.no_pattern_features:
        cols_to_zero = _PAT_ALL_COLS
    else:
        if args.no_fan_features:        cols_to_zero += _PAT_COLS["fan_out"] + _PAT_COLS["fan_in"] + _PAT_COLS["fan_asymmetry"]
        if args.no_stack_features:      cols_to_zero += _PAT_COLS["stack"]
        if args.no_cycle_feature:       cols_to_zero += _PAT_COLS["cycle"]
        if args.no_sg_feature:          cols_to_zero += _PAT_COLS["scatter_gather"]
        if args.no_bipartite_feature:   cols_to_zero += _PAT_COLS["bipartite"]
    if cols_to_zero:
        print(f"  [pattern ablation] zeroing cols {sorted(set(cols_to_zero))} in x_dense …")
        _apply_pattern_masking(networks, list(set(cols_to_zero)))

    # ── diffusion model ────────────────────────────────────────────────────────
    diff_model, diffusion, x_mean, x_std, node_dim, _diff_max_nodes = _load_diffusion(device)

    # Honour p_diffusion=0 even if a checkpoint was found
    if args.p_diffusion == 0.0:
        diff_model = None

    # ── models ─────────────────────────────────────────────────────────────────
    ckpt_dir = CKPT_ROOT / args.condition
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # node_dim is the diffusion model's full feature dim (col 0 = laundering flag included).
    # The encoder receives col 0 stripped, so in_dim = node_dim - 1.
    encoder   = GraphEncoder(in_dim=node_dim - 1, hidden_dim=64, out_dim=128).to(device)
    projector = ProjectionHead(in_dim=128, proj_dim=64).to(device)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(projector.parameters()),
        lr=args.lr,
    )

    # ── train via shared SimCLR loop ───────────────────────────────────────────
    # train_simclr_fast handles IBM-specific augmentation internally via
    # augment_network_view_fast (crop / edge-delete / node-delete / node-add).
    # p_diffusion controls whether view 2 is replaced by a diffusion view.
    train_simclr_fast(
        networks            = networks,
        full_df             = None,
        encoder             = encoder,
        projector           = projector,
        optimizer           = optimizer,
        device              = device,
        batch_size          = args.batch_size,
        epochs              = args.epochs,
        checkpoint_dir      = str(ckpt_dir),
        checkpoint_interval = args.ckpt_interval,
        diffusion_model     = diff_model,
        diffusion           = diffusion,
        x_mean              = x_mean,
        x_std               = x_std,
        p_diffusion         = args.p_diffusion,
        diffusion_t_frac    = 0.3,
        max_nodes           = _diff_max_nodes,
        supcon_weight       = args.supcon_weight,
        supcon_temperature  = args.supcon_temp,
        # view-type control
        view_type           = args.view_type,
        diff_n_steps        = args.diff_n_steps,
        diff_t_start_frac   = args.diff_t_start,
        diff_guidance_scale = args.diff_guidance_scale,
        probe_warmup_epochs = args.probe_warmup_epochs,
        probe_update_every  = args.probe_update_every,
    )

    print(f"\nBest checkpoint → {ckpt_dir / 'best_model.pt'}")
    return str(ckpt_dir)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parametrized IBM SimCLR training for ablation studies."
    )
    parser.add_argument("--condition",     type=str,   required=True,
                        help="Unique name for this ablation condition, e.g. 'no_diffusion_aug'")
    # Loss weights
    parser.add_argument("--supcon-weight", type=float, default=0.5,
                        help="Supervised contrastive loss weight (default 0.5; 0 to disable)")
    parser.add_argument("--supcon-temp",   type=float, default=0.07,
                        help="Temperature for supervised contrastive loss (default 0.07)")
    # Augmentation
    parser.add_argument("--p-diffusion",   type=float, default=0.30,
                        help="Probability of using diffusion view for view 2 (default 0.3)")
    # ── view-type control ─────────────────────────────────────────────────────
    parser.add_argument("--view-type", type=str, default="single_step",
                        choices=["single_step", "multistep", "guided", "multistep_to_guided"],
                        help="Diffusion view strategy (default: single_step)")
    parser.add_argument("--diff-n-steps",        type=int,   default=15)
    parser.add_argument("--diff-t-start",        type=float, default=0.5)
    parser.add_argument("--diff-guidance-scale", type=float, default=1.5)
    parser.add_argument("--probe-warmup-epochs", type=int,   default=20)
    parser.add_argument("--probe-update-every",  type=int,   default=20)
    # Training hyperparameters
    parser.add_argument("--epochs",        type=int,   default=100)
    parser.add_argument("--batch-size",    type=int,   default=128)
    parser.add_argument("--lr",            type=float, default=1e-3)
    parser.add_argument("--ckpt-interval", type=int,   default=10,
                        help="Save a periodic checkpoint every N epochs (default 10)")
    # Data path override
    parser.add_argument("--ibm-csv",       type=str,   default=DEFAULT_CSV,
                        help="Path to IBM transactions CSV (default: LI-Small_Trans.csv)")
    # ── AML pattern feature ablation flags ────────────────────────────────────
    parser.add_argument("--no-pattern-features",  action="store_true",
                        help="Zero out ALL AML pattern features (cols 11-18) — full removal ablation")
    parser.add_argument("--no-fan-features",      action="store_true",
                        help="Zero out fan-out/fan-in/asymmetry features (cols 11-13)")
    parser.add_argument("--no-stack-features",    action="store_true",
                        help="Zero out stack/passthrough/chain-depth features (cols 14-15)")
    parser.add_argument("--no-cycle-feature",     action="store_true",
                        help="Zero out in-cycle feature (col 16)")
    parser.add_argument("--no-sg-feature",        action="store_true",
                        help="Zero out scatter-gather score (col 17)")
    parser.add_argument("--no-bipartite-feature", action="store_true",
                        help="Zero out bipartite score (col 18)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    train(args, device)
