"""
elliptic_simclr_train_ablation.py
──────────────────────────────────────────────────────────────────────────────
Parametrized Elliptic SimCLR training for ablation studies.

Saves the best checkpoint to:
  checkpoints/simclr_elliptic_ablation/<condition>/best_model.pt

Usage examples
──────────────
  # Full system (replicates default training)
  python elliptic_simclr_train_ablation.py --condition full

  # No supervised contrastive loss
  python elliptic_simclr_train_ablation.py --condition no_supcon --supcon-weight 0.0

  # No diffusion augmentation
  python elliptic_simclr_train_ablation.py --condition no_diffusion --p-diffusion 0.0

  # Edge-drop only (no feature masking, no diffusion)
  python elliptic_simclr_train_ablation.py --condition edge_drop_only \\
      --p-feat-mask 0.0 --p-diffusion 0.0

  # Feature-mask only
  python elliptic_simclr_train_ablation.py --condition feat_mask_only \\
      --p-edge-drop 0.0 --p-diffusion 0.0

  # No structural augmentation (diffusion-only views)
  python elliptic_simclr_train_ablation.py --condition diffusion_only \\
      --p-edge-drop 0.0 --p-feat-mask 0.0 --p-diffusion 1.0

  # NT-Xent only (no SupCon, no augmentation variety)
  python elliptic_simclr_train_ablation.py --condition ntxent_only \\
      --supcon-weight 0.0 --p-diffusion 0.0
"""

import argparse
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch

# ── path setup (mirrors generation/generation.py convention) ─────────────────
BASE_DIR = Path(__file__).resolve().parent          # igraph_version/
DIFF_DIR = BASE_DIR / "diffusion"                   # igraph_version/diffusion/

# Insert in reverse priority order so simclr/ ends up first in sys.path,
# meaning `import simclr` resolves to igraph_version/simclr/simclr.py
# (the full module) rather than the package __init__.py.
for _p in (str(BASE_DIR), str(DIFF_DIR), str(BASE_DIR / "simclr"),
           str(BASE_DIR / "data")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from simclr import (GraphEncoder, ProjectionHead, nt_xent_loss, sup_con_loss,
                    _diffusion_view_multistep, _diffusion_view_guided, _fit_probe)
from elliptic_adapter import load_elliptic_pyg_graphs

CKPT_ROOT = BASE_DIR / "checkpoints" / "simclr_elliptic_ablation"
MAX_NODES  = 100


# ── augmentation helpers ──────────────────────────────────────────────────────

def _extend_to_6d(data):
    n         = data.x.shape[0]
    label_col = torch.full((n, 1), float(data.y.item()))
    x6        = torch.cat([label_col, data.x], dim=1)
    return Data(x=x6, edge_index=data.edge_index.clone(), y=data.y.clone())


def _augment_pyg(data, p_edge_drop=0.20, p_feat_mask=0.15):
    """Structural augmentation: edge dropout + feature masking."""
    x  = data.x.clone()
    ei = data.edge_index.clone()

    if p_feat_mask > 0.0 and x.shape[1] > 1:
        feat_mask       = torch.rand(x.shape[1]) < p_feat_mask
        feat_mask[0]    = False   # never mask label column
        x[:, feat_mask] = 0.0

    if ei.shape[1] > 1 and p_edge_drop > 0.0:
        E_half    = ei.shape[1] // 2
        keep      = torch.rand(E_half) > p_edge_drop
        keep_full = keep.repeat_interleave(2)
        if ei.shape[1] % 2 == 1:
            keep_full = torch.cat([keep_full, torch.ones(1, dtype=torch.bool)])
        ei = ei[:, keep_full]

    # Strip col 0 (laundering flag) — encoder must not receive the label
    # as a node feature, only through the SupCon label signal.
    return Data(x=x[:, 1:], edge_index=ei, y=data.y.clone())


def _load_diffusion(device):
    model_path = BASE_DIR / "checkpoints" / "diffusion_elliptic" / "model.pt"
    if not model_path.exists():
        print(f"[diffusion] Not found at {model_path} — diffusion views disabled.")
        return None, None, None, None
    try:
        from diffusion.model     import DiffusionGNN
        from diffusion.diff_util import create_diffusion
        ckpt       = torch.load(model_path, map_location=device, weights_only=False)
        diff_model = DiffusionGNN(node_dim=6, hidden_dim=128, num_layers=4).to(device)
        diff_model.load_state_dict(ckpt["model"])
        diff_model.eval()
        diffusion  = create_diffusion(T=500)
        x_mean     = ckpt["x_mean"].to(device)
        x_std      = ckpt["x_std"].to(device)
        print(f"[diffusion] Loaded {model_path}")
        return diff_model, diffusion, x_mean, x_std
    except Exception as e:
        print(f"[diffusion] Load failed ({e}) — diffusion views disabled.")
        return None, None, None, None


def _diffusion_view(data6, diff_model, diffusion, x_mean, x_std, device, t_frac=0.3):
    n = data6.x.shape[0]
    if n > MAX_NODES:
        return None

    ei  = data6.edge_index
    adj = torch.zeros(n, n)
    if ei.shape[1] > 0:
        valid = ei[0] != ei[1]
        adj[ei[0][valid], ei[1][valid]] = 1.0

    x_pad     = torch.zeros(1, MAX_NODES, 6)
    adj_pad   = torch.zeros(1, MAX_NODES, MAX_NODES)
    node_mask = torch.zeros(1, MAX_NODES)
    x_pad[0, :n]       = data6.x
    adj_pad[0, :n, :n] = adj
    node_mask[0, :n]   = 1.0

    x_pad, adj_pad, node_mask = (
        x_pad.to(device), adj_pad.to(device), node_mask.to(device)
    )
    x_norm           = x_pad.clone()
    x_norm[:, :, 1:] = (x_pad[:, :, 1:] - x_mean[1:]) / x_std[1:]
    x_norm           = x_norm  * node_mask.unsqueeze(-1)
    adj_pad          = adj_pad * node_mask[:, :, None] * node_mask[:, None, :]

    t_abs = max(1, int(t_frac * diffusion.num_timesteps))
    t     = torch.tensor([t_abs], device=device)
    x_t, adj_t = diffusion.q_sample(x_norm, t, node_mask=node_mask, adj_start=adj_pad)

    with torch.no_grad():
        eps_pred, adj_pred, _ = diff_model(
            x_t, diffusion._scale_timesteps(t), adj=adj_t, node_mask=node_mask
        )
        x0_cont = diffusion._predict_xstart_from_eps(x_t[..., 1:], t, eps_pred[..., 1:])
        x0_bin  = eps_pred[..., 0:1].clamp(0.0, 1.0)

    x0 = torch.cat([x0_bin, x0_cont], dim=-1)[0, :n].cpu()
    x0[:, 1:] = x0[:, 1:] * x_std[1:].cpu() + x_mean[1:].cpu()
    deg = adj_pred[0, :n, :n].cpu().sum(dim=-1)
    x0[:, 1] = deg / deg.max().clamp(min=1.0)

    ei_out = (adj_pred[0, :n, :n].cpu() > 0.5).nonzero(as_tuple=False).T.contiguous()
    # Strip col 0 so the local diffusion view has the same dim as _augment_pyg output
    return Data(x=x0[:, 1:], edge_index=ei_out, y=data6.y.clone())


# ── training ──────────────────────────────────────────────────────────────────

def train(args, device):
    print(f"\n{'='*60}")
    print(f"SimCLR ablation: {args.condition}")
    print(f"  supcon_weight={args.supcon_weight}  p_diffusion={args.p_diffusion}")
    print(f"  p_edge_drop={args.p_edge_drop}  p_feat_mask={args.p_feat_mask}")
    print(f"  view_type={args.view_type}  diff_n_steps={args.diff_n_steps}"
          f"  diff_t_start={args.diff_t_start}  diff_guidance={args.diff_guidance_scale}")
    print(f"{'='*60}\n")

    print("Loading Elliptic graphs …")
    raw_graphs = load_elliptic_pyg_graphs(max_nodes=MAX_NODES)
    graphs     = [_extend_to_6d(g) for g in raw_graphs]
    n_ill = sum(g.y.item() == 1 for g in graphs)
    print(f"  {len(graphs)} graphs  ({n_ill} illicit, {len(graphs)-n_ill} licit)")

    diff_model, diffusion, x_mean, x_std = _load_diffusion(device)
    use_diffusion = (
        diff_model is not None and args.p_diffusion > 0.0
    )

    ckpt_dir = CKPT_ROOT / args.condition
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # in_dim=5: col 0 (laundering flag) is stripped before the encoder sees any view
    encoder   = GraphEncoder(in_dim=5, hidden_dim=64, out_dim=128).to(device)
    projector = ProjectionHead(in_dim=128, proj_dim=64).to(device)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(projector.parameters()), lr=args.lr
    )

    best_loss         = float("inf")
    best_encoder_sd   = None
    best_projector_sd = None

    # view-type flags (mirrors train_simclr_fast logic)
    _use_multistep   = args.view_type in ("multistep", "guided", "multistep_to_guided")
    _use_guided      = args.view_type == "guided"
    _use_progressive = args.view_type == "multistep_to_guided"
    probe            = None

    for epoch in range(args.epochs):
        t0 = time.time()
        encoder.train();  projector.train()
        if diff_model is not None:
            diff_model.eval()

        # ── decide view strategy for this epoch ──────────────────────────────
        guided_this_epoch = _use_guided or (_use_progressive and epoch >= args.probe_warmup_epochs)

        if guided_this_epoch and use_diffusion:
            # Fit once at the start of the guided phase — periodic re-fitting
            # overfits to noise in the embeddings and destabilises guidance.
            if probe is None:
                print(f"  [probe] fitting once on {len(graphs)} Elliptic graphs …")
                probe = _fit_probe(encoder, graphs, device, n_epochs=300, is_pyg=True)

        random.shuffle(graphs)
        total_loss = total_ntxent = total_sc = 0.0
        n_batches  = 0

        for i in range(0, len(graphs), args.batch_size):
            batch = graphs[i : i + args.batch_size]
            views1, views2, labels_list = [], [], []

            for g in batch:
                views1.append(_augment_pyg(g,
                    p_edge_drop=args.p_edge_drop,
                    p_feat_mask=args.p_feat_mask,
                ))

                diff_v2 = None
                if use_diffusion and random.random() < args.p_diffusion:
                    if guided_this_epoch and probe is not None:
                        diff_v2 = _diffusion_view_guided(
                            g, diff_model, diffusion, x_mean, x_std,
                            encoder, probe, device,
                            t_start_frac=args.diff_t_start,
                            n_steps=args.diff_n_steps,
                            guidance_scale=args.diff_guidance_scale,
                            max_nodes=MAX_NODES, is_pyg=True,
                        )
                    elif _use_multistep:
                        diff_v2 = _diffusion_view_multistep(
                            g, diff_model, diffusion, x_mean, x_std, device,
                            t_start_frac=args.diff_t_start,
                            n_steps=args.diff_n_steps,
                            max_nodes=MAX_NODES, is_pyg=True,
                        )
                    else:
                        diff_v2 = _diffusion_view(g, diff_model, diffusion, x_mean, x_std, device)

                views2.append(diff_v2 if diff_v2 is not None else _augment_pyg(
                    g, p_edge_drop=args.p_edge_drop, p_feat_mask=args.p_feat_mask,
                ))
                labels_list.append(int(g.y.item()))

            labels = torch.tensor(labels_list, dtype=torch.long, device=device)
            b1 = Batch.from_data_list(views1).to(device)
            b2 = Batch.from_data_list(views2).to(device)

            optimizer.zero_grad()
            h1 = encoder(b1);  h2 = encoder(b2)
            z1 = projector(h1); z2 = projector(h2)

            ntxent = nt_xent_loss(z1, z2)

            if args.supcon_weight > 0.0:
                z_all   = torch.cat([z1, z2], dim=0)
                lbl_all = torch.cat([labels, labels], dim=0)
                sc      = sup_con_loss(z_all, lbl_all, temperature=args.supcon_temp)
            else:
                sc = torch.tensor(0.0, device=device)

            loss = ntxent + args.supcon_weight * sc
            loss.backward()
            optimizer.step()

            total_loss   += loss.item()
            total_ntxent += ntxent.item()
            total_sc     += sc.item() if isinstance(sc, torch.Tensor) else float(sc)
            n_batches    += 1

        avg    = total_loss   / max(1, n_batches)
        avg_nx = total_ntxent / max(1, n_batches)
        avg_sc = total_sc     / max(1, n_batches)
        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"loss={avg:.4f} (nt_xent={avg_nx:.4f}, supcon={avg_sc:.4f}) | "
              f"{time.time()-t0:.1f}s")

        if (epoch + 1) % args.ckpt_interval == 0:
            p = ckpt_dir / f"epoch_{epoch+1}.pt"
            torch.save({
                "encoder_state_dict":   encoder.state_dict(),
                "projector_state_dict": projector.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch + 1, "loss": avg,
            }, p)

        if avg < best_loss:
            best_loss         = avg
            best_encoder_sd   = {k: v.cpu().clone() for k, v in encoder.state_dict().items()}
            best_projector_sd = {k: v.cpu().clone() for k, v in projector.state_dict().items()}
            print(f"  ↑ new best  loss={best_loss:.4f}")

    best_path = ckpt_dir / "best_model.pt"
    torch.save({
        "encoder_state_dict":   best_encoder_sd,
        "projector_state_dict": best_projector_sd,
        "loss": best_loss,
    }, best_path)
    print(f"\nBest model → {best_path}  (loss={best_loss:.4f})")
    return str(ckpt_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parametrized Elliptic SimCLR training for ablation studies."
    )
    parser.add_argument("--condition", type=str, required=True,
                        help="Name for this ablation condition, e.g. 'no_supcon'")
    # Loss weights
    parser.add_argument("--supcon-weight",  type=float, default=0.5,
                        help="Supervised contrastive loss weight (default 0.5; set 0 to disable)")
    parser.add_argument("--supcon-temp",    type=float, default=0.07,
                        help="Temperature for supervised contrastive loss (default 0.07)")
    # Augmentation probabilities
    parser.add_argument("--p-diffusion",  type=float, default=0.30,
                        help="Probability of using diffusion view for view 2 (default 0.3)")
    parser.add_argument("--p-edge-drop",  type=float, default=0.20,
                        help="Per-edge drop probability in structural augmentation (default 0.2)")
    parser.add_argument("--p-feat-mask",  type=float, default=0.15,
                        help="Per-feature mask probability in structural augmentation (default 0.15)")
    # Training hyperparameters
    parser.add_argument("--epochs",        type=int,   default=100)
    parser.add_argument("--batch-size",    type=int,   default=128)
    parser.add_argument("--lr",            type=float, default=1e-3)
    parser.add_argument("--ckpt-interval", type=int,   default=10,
                        help="Save a periodic checkpoint every N epochs")
    # ── new: view-type control ────────────────────────────────────────────────
    parser.add_argument("--view-type", type=str, default="single_step",
                        choices=["single_step", "multistep", "guided", "multistep_to_guided"],
                        help="Diffusion view strategy: single_step (current), "
                             "multistep (Option A: DDIM), guided (Option C: class-guided DDIM), "
                             "multistep_to_guided (warmup on A then switch to C)")
    parser.add_argument("--diff-n-steps",       type=int,   default=15,
                        help="DDIM denoising steps for multistep/guided views (default 15)")
    parser.add_argument("--diff-t-start",       type=float, default=0.5,
                        help="Noise start level as fraction of T (default 0.5)")
    parser.add_argument("--diff-guidance-scale",type=float, default=1.5,
                        help="Guidance strength for guided views (default 1.5)")
    parser.add_argument("--probe-warmup-epochs",type=int,   default=20,
                        help="Epochs of multistep before switching to guided (multistep_to_guided)")
    parser.add_argument("--probe-update-every", type=int,   default=20,
                        help="Re-fit probe every N epochs once guided phase starts")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    train(args, device)
