"""
elliptic_simclr_train.py
────────────────────────────────────────────────────────────────────────────────
Train the SimCLR graph encoder on ego subgraphs from the Elliptic Bitcoin
Dataset.

Because Elliptic data is stored as PyG Data objects (not IBM-style network
dicts with igraph graphs), PyG-level augmentations replace the igraph-based
ones used in simclr/main.py:

  Edge dropout   — drop each edge independently with probability p_edge_drop
  Feature mask   — zero each structural feature independently with prob p_feat_mask

Node features are extended from 5-D to 6-D before encoding, matching the IBM
training convention (GraphEncoder in_dim=6):

  col 0  — graph-level label broadcast to all nodes  (1 = illicit, 0 = licit)
  col 1  — normalised degree
  col 2  — betweenness centrality
  col 3  — clustering coefficient
  col 4  — PageRank
  col 5  — degree assortativity

If elliptic_diffusion_train.py has already been run, diffusion augmentation is
loaded automatically and applied with probability p_diffusion=0.3 per batch.

Outputs
───────
  simclr/model_checkpoints_elliptic/best_model.pt   best checkpoint by loss
  simclr/model_checkpoints_elliptic/epoch_N.pt      periodic checkpoints

Usage
─────
  python elliptic_simclr_train.py
"""

import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch

# ── path setup ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DIFF_DIR = BASE_DIR.parent / "diffusion"

for _p in (str(BASE_DIR.parent), str(DIFF_DIR), str(BASE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from simclr import GraphEncoder, ProjectionHead, nt_xent_loss, sup_con_loss
from elliptic_adapter import load_elliptic_pyg_graphs

# ── hyperparameters ───────────────────────────────────────────────────────────
EPOCHS         = 100
BATCH_SIZE     = 128
LR             = 1e-3
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
SUPCON_W       = 0.5
SUPCON_T       = 0.07
CKPT_INTERVAL  = 10
MAX_NODES      = 100

P_EDGE_DROP    = 0.20   # probability of dropping any individual edge
P_FEAT_MASK    = 0.15   # probability of zeroing any individual feature
P_DIFFUSION    = 0.30   # probability of using diffusion view instead of structural

CKPT_DIR = BASE_DIR / "model_checkpoints_elliptic"


# ── augmentations ─────────────────────────────────────────────────────────────

def _extend_to_6d(data):
    """
    Prepend the graph-level label as col 0 → [n, 6].
    Result is compatible with GraphEncoder(in_dim=6).
    Returns a new Data object; original is not modified.
    """
    n         = data.x.shape[0]
    label_col = torch.full((n, 1), float(data.y.item()))
    x6        = torch.cat([label_col, data.x], dim=1)
    return Data(x=x6, edge_index=data.edge_index.clone(), y=data.y.clone())


def _augment_pyg(data, p_edge_drop=P_EDGE_DROP, p_feat_mask=P_FEAT_MASK):
    """
    Return an augmented view of a 6-D PyG Data object.

    Augmentations applied independently:
      1. Feature masking  — zero individual features with prob p_feat_mask
                            col 0 (class label) is never masked to preserve the
                            class signal that the supervised contrastive loss uses.
      2. Edge dropout     — drop each undirected edge with prob p_edge_drop;
                            both directions are dropped together to keep the
                            graph undirected.
    """
    x  = data.x.clone()
    ei = data.edge_index.clone()

    # Feature masking (skip col 0)
    if p_feat_mask > 0.0 and x.shape[1] > 1:
        feat_mask        = torch.rand(x.shape[1]) < p_feat_mask
        feat_mask[0]     = False  # never mask the label column
        x[:, feat_mask]  = 0.0

    # Edge dropout — preserve undirected symmetry by dropping edges in pairs.
    # elliptic_adapter stores edges as interleaved pairs (u→v, v→u), so
    # edges at positions 2k and 2k+1 form one undirected edge.
    if ei.shape[1] > 1 and p_edge_drop > 0.0:
        E_half = ei.shape[1] // 2
        keep   = torch.rand(E_half) > p_edge_drop          # [E_half] bool
        # Expand to full (both directions): keep[k] applies to positions 2k and 2k+1
        keep_full = keep.repeat_interleave(2)               # [E]
        if ei.shape[1] % 2 == 1:
            # Odd edge count (shouldn't happen for undirected; handle defensively)
            keep_full = torch.cat([keep_full, torch.ones(1, dtype=torch.bool)])
        ei = ei[:, keep_full]

    return Data(x=x, edge_index=ei, y=data.y.clone())


# ── diffusion augmentation view ───────────────────────────────────────────────

def _load_diffusion_for_simclr():
    """
    Try to load the Elliptic diffusion model for optional view augmentation.
    Falls back gracefully with (None, None, None, None) if not yet trained.
    """
    model_path = DIFF_DIR / "model_elliptic.pt"
    if not model_path.exists():
        print("[diffusion] model_elliptic.pt not found — diffusion views disabled.")
        return None, None, None, None
    try:
        from diffusion.model    import DiffusionGNN
        from diffusion.diff_util import create_diffusion
        ckpt       = torch.load(model_path, map_location=DEVICE, weights_only=False)
        diff_model = DiffusionGNN(node_dim=6, hidden_dim=128, num_layers=4).to(DEVICE)
        diff_model.load_state_dict(ckpt["model"])
        diff_model.eval()
        diffusion  = create_diffusion(T=500)
        x_mean     = ckpt["x_mean"].to(DEVICE)
        x_std      = ckpt["x_std"].to(DEVICE)
        print(f"[diffusion] Loaded model_elliptic.pt for SimCLR augmentation.")
        return diff_model, diffusion, x_mean, x_std
    except Exception as e:
        print(f"[diffusion] Could not load model_elliptic.pt ({e}) — diffusion views disabled.")
        return None, None, None, None


def _diffusion_view(data6, diff_model, diffusion, x_mean, x_std, t_frac=0.3):
    """
    Generate one augmented view via a forward-noise + single-step denoising pass.
    Identical strategy to simclr.py:_diffusion_view but takes a 6-D PyG Data
    object as input (Elliptic convention) instead of an IBM network dict.

    Returns a 6-D PyG Data object, or None if the graph is too large.
    """
    n = data6.x.shape[0]
    if n > MAX_NODES:
        return None

    # Build dense adj from edge_index (filtering self-loops)
    ei  = data6.edge_index
    adj = torch.zeros(n, n)
    if ei.shape[1] > 0:
        valid = ei[0] != ei[1]
        adj[ei[0][valid], ei[1][valid]] = 1.0

    # Pad to MAX_NODES
    x_pad    = torch.zeros(1, MAX_NODES, 6)
    adj_pad  = torch.zeros(1, MAX_NODES, MAX_NODES)
    node_mask = torch.zeros(1, MAX_NODES)
    x_pad[0, :n]       = data6.x
    adj_pad[0, :n, :n] = adj
    node_mask[0, :n]   = 1.0

    x_pad, adj_pad, node_mask = (
        x_pad.to(DEVICE), adj_pad.to(DEVICE), node_mask.to(DEVICE)
    )

    # Normalise (same convention as diffusion/train.py and generation.py)
    x_norm        = x_pad.clone()
    x_norm[:, :, 1:] = (x_pad[:, :, 1:] - x_mean[1:]) / x_std[1:]
    x_norm        = x_norm  * node_mask.unsqueeze(-1)
    adj_pad       = adj_pad * node_mask[:, :, None] * node_mask[:, None, :]

    t_abs = max(1, int(t_frac * diffusion.num_timesteps))
    t     = torch.tensor([t_abs], device=DEVICE)
    x_t, adj_t = diffusion.q_sample(x_norm, t, node_mask=node_mask, adj_start=adj_pad)

    with torch.no_grad():
        eps_pred, adj_pred, _ = diff_model(
            x_t, diffusion._scale_timesteps(t), adj=adj_t, node_mask=node_mask
        )
        x0_cont = diffusion._predict_xstart_from_eps(x_t[..., 1:], t, eps_pred[..., 1:])
        x0_bin  = eps_pred[..., 0:1].clamp(0.0, 1.0)

    x0    = torch.cat([x0_bin, x0_cont], dim=-1)[0, :n].cpu()   # [n, 6]
    x0[:, 1:] = x0[:, 1:] * x_std[1:].cpu() + x_mean[1:].cpu()

    deg = adj_pred[0, :n, :n].cpu().sum(dim=-1)
    x0[:, 1] = deg / deg.max().clamp(min=1.0)

    ei_out = (adj_pred[0, :n, :n].cpu() > 0.5).nonzero(as_tuple=False).T.contiguous()
    return Data(x=x0, edge_index=ei_out, y=data6.y.clone())


# ── training ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    start = time.time()

    print("Loading Elliptic ego subgraphs …")
    raw_graphs = load_elliptic_pyg_graphs(max_nodes=MAX_NODES)
    graphs     = [_extend_to_6d(g) for g in raw_graphs]
    n_ill = sum(g.y.item() == 1 for g in graphs)
    print(f"Loaded {len(graphs)} graphs ({n_ill} illicit, {len(graphs)-n_ill} licit)  "
          f"[{time.time()-start:.1f}s]")

    diff_model, diffusion, x_mean, x_std = _load_diffusion_for_simclr()
    use_diffusion = diff_model is not None

    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    encoder   = GraphEncoder(in_dim=6, hidden_dim=64, out_dim=128).to(DEVICE)
    projector = ProjectionHead(in_dim=128, proj_dim=64).to(DEVICE)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(projector.parameters()), lr=LR
    )

    best_loss         = float("inf")
    best_encoder_sd   = None
    best_projector_sd = None

    for epoch in range(EPOCHS):
        epoch_start = time.time()
        encoder.train()
        projector.train()
        if use_diffusion:
            diff_model.eval()

        random.shuffle(graphs)
        total_loss = total_ntxent = total_sc = 0.0
        n_batches  = 0

        for i in range(0, len(graphs), BATCH_SIZE):
            batch = graphs[i : i + BATCH_SIZE]

            views1, views2, labels_list = [], [], []
            for g in batch:
                views1.append(_augment_pyg(g))

                if use_diffusion and random.random() < P_DIFFUSION:
                    v2 = _diffusion_view(g, diff_model, diffusion, x_mean, x_std)
                    views2.append(v2 if v2 is not None else _augment_pyg(g))
                else:
                    views2.append(_augment_pyg(g))

                labels_list.append(int(g.y.item()))

            labels = torch.tensor(labels_list, dtype=torch.long, device=DEVICE)
            b1 = Batch.from_data_list(views1).to(DEVICE)
            b2 = Batch.from_data_list(views2).to(DEVICE)

            optimizer.zero_grad()
            h1 = encoder(b1);  h2 = encoder(b2)
            z1 = projector(h1); z2 = projector(h2)

            ntxent = nt_xent_loss(z1, z2)

            z_all   = torch.cat([z1, z2], dim=0)
            lbl_all = torch.cat([labels, labels], dim=0)
            sc      = sup_con_loss(z_all, lbl_all, temperature=SUPCON_T)

            loss = ntxent + SUPCON_W * sc
            loss.backward()
            optimizer.step()

            total_loss   += loss.item()
            total_ntxent += ntxent.item()
            total_sc     += sc.item() if isinstance(sc, torch.Tensor) else float(sc)
            n_batches    += 1

        avg      = total_loss   / max(1, n_batches)
        avg_nx   = total_ntxent / max(1, n_batches)
        avg_sc   = total_sc     / max(1, n_batches)
        print(f"Epoch {epoch+1:3d}/{EPOCHS} | "
              f"loss={avg:.4f} (nt_xent={avg_nx:.4f}, supcon={avg_sc:.4f}) | "
              f"{time.time()-epoch_start:.1f}s")

        if (epoch + 1) % CKPT_INTERVAL == 0:
            ckpt_path = CKPT_DIR / f"epoch_{epoch+1}.pt"
            torch.save({
                "encoder_state_dict":   encoder.state_dict(),
                "projector_state_dict": projector.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch + 1,
                "loss":  avg,
            }, ckpt_path)
            print(f"  Checkpoint → {ckpt_path}")

        if avg < best_loss:
            best_loss         = avg
            best_encoder_sd   = {k: v.cpu().clone() for k, v in encoder.state_dict().items()}
            best_projector_sd = {k: v.cpu().clone() for k, v in projector.state_dict().items()}
            print(f"  New best  loss={best_loss:.4f}")

    best_path = CKPT_DIR / "best_model.pt"
    torch.save({
        "encoder_state_dict":   best_encoder_sd,
        "projector_state_dict": best_projector_sd,
        "loss": best_loss,
    }, best_path)
    print(f"\nBest model saved → {best_path}  (loss={best_loss:.4f})")
    print(f"Total time: {time.time()-start:.0f}s")
