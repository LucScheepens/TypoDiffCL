"""
elliptic_diffusion_train.py
────────────────────────────────────────────────────────────────────────────────
Train the joint (node-feature + adjacency) diffusion model on ego subgraphs
extracted from the Elliptic Bitcoin Dataset.

Uses the identical architecture and loss as diffusion/train.py (node_dim=6,
hidden_dim=128, T=500, cosine schedule) — only the data source changes.

Node feature layout (6-D, matching IBM convention so all downstream generation
code is reusable without modification)
  col 0  — broadcast graph label  (1.0 = illicit anchor, 0.0 = licit anchor)
  col 1  — normalised degree
  col 2  — betweenness centrality
  col 3  — clustering coefficient
  col 4  — PageRank
  col 5  — degree assortativity (graph-level constant per ego)

Outputs
───────
  diffusion/model_elliptic.pt          trained diffusion checkpoint
  simclr/elliptic_diffusion_cache_d4.pt  dense (x, adj) pairs for fast reloads

Usage
─────
  python elliptic_diffusion_train.py
"""

import math
import sys
import time
from pathlib import Path

import torch
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

# ── path setup ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent          # simclr/
DIFF_DIR = BASE_DIR.parent / "diffusion"

for _p in (str(BASE_DIR.parent), str(DIFF_DIR), str(BASE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from diffusion.collate    import collate_fn
from diffusion.diff_util  import create_diffusion
from diffusion.model      import DiffusionGNN
from diffusion.resample   import LossSecondMomentResampler
from elliptic_adapter     import load_elliptic_pyg_graphs

# ── hyperparameters ───────────────────────────────────────────────────────────
BATCH_SIZE        = 32
LR                = 2e-4
EPOCHS            = 200
WARMUP_EPOCHS     = 10
DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"

NODE_DIM          = 6       # col 0 = broadcast label, cols 1-5 = structural
HIDDEN            = 128
TIMESTEPS         = 500
MAX_NODES         = 100     # matches elliptic_adapter default
ADJ_LOSS_W        = 0.3
DENSITY_LOSS_W    = 3.0
NODE_EXIST_LOSS_W = 1.0
MASK_DROPOUT_RATE = 0.15
GHOST_NODE_RATE   = 0.10
LAUND_LOSS_W      = 2.0     # upweight illicit BCE — mirrors IBM convention
MAX_GRAD_NORM     = 1.0
VIZ_INTERVAL      = 50

CACHE_PATH = BASE_DIR / "elliptic_diffusion_cache_d4.pt"
CKPT_PATH  = DIFF_DIR / "model_elliptic.pt"


# ── dense conversion ──────────────────────────────────────────────────────────

def pyg_to_dense_elliptic(data):
    """
    Convert one Elliptic PyG Data object to (x [n,6], adj [n,n]) tensors.

    Broadcasting the graph-level label to col 0 of every node mirrors the IBM
    per-node laundering flag.  The diffusion model learns to associate the
    illicit feature distribution with col-0 ≈ 1, enabling guided generation to
    steer toward illicit-like subgraphs via the same probe mechanism.
    """
    n     = data.x.shape[0]
    label = float(data.y.item())

    x = torch.zeros(n, 6, dtype=torch.float32)
    x[:, 0] = label     # broadcast graph label
    x[:, 1:] = data.x   # 5 structural features (degree … assortativity)

    adj = torch.zeros(n, n, dtype=torch.float32)
    ei  = data.edge_index
    if ei.shape[1] > 0:
        # Filter self-loops (adapter adds them for isolated graphs)
        valid = ei[0] != ei[1]
        src, dst = ei[0][valid], ei[1][valid]
        # Keep only in-bounds indices
        inbounds = (src < n) & (dst < n)
        adj[src[inbounds], dst[inbounds]] = 1.0

    return x, adj


# ── dataset ───────────────────────────────────────────────────────────────────

class EllipticDenseDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


# ── cache helpers ─────────────────────────────────────────────────────────────

def _needs_rebuild():
    if not CACHE_PATH.exists():
        return True
    sample = torch.load(CACHE_PATH, weights_only=False)
    if len(sample) == 0 or sample[0][0].shape[1] != NODE_DIM:
        print(f"Cache feature dim {sample[0][0].shape[1]} != NODE_DIM={NODE_DIM}. Rebuilding …")
        CACHE_PATH.unlink()
        return True
    return False


def _build_cache():
    print("Loading Elliptic ego subgraphs …")
    pyg_graphs = load_elliptic_pyg_graphs(max_nodes=MAX_NODES)
    print(f"Converting {len(pyg_graphs)} graphs to dense format …")
    pairs = []
    for data in pyg_graphs:
        if data.x.shape[0] < 3:
            continue
        x6, adj = pyg_to_dense_elliptic(data)
        pairs.append((x6, adj))
    torch.save(pairs, CACHE_PATH)
    print(f"Saved {len(pairs)} dense pairs → {CACHE_PATH}")
    return pairs


# ── LR schedule ───────────────────────────────────────────────────────────────

def lr_lambda(epoch):
    if epoch < WARMUP_EPOCHS:
        return (epoch + 1) / WARMUP_EPOCHS
    progress = (epoch - WARMUP_EPOCHS) / max(1, EPOCHS - WARMUP_EPOCHS)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    USE_AMP    = (DEVICE == "cuda")
    scaler     = torch.amp.GradScaler("cuda", enabled=USE_AMP)
    start_time = time.time()

    # -- Dataset ---------------------------------------------------------------
    if _needs_rebuild():
        pairs = _build_cache()
    else:
        print(f"Loading dense cache from {CACHE_PATH} …")
        pairs = torch.load(CACHE_PATH, weights_only=False)
        print(f"Loaded {len(pairs)} graphs.")

    # Filter any that slipped through over MAX_NODES
    before = len(pairs)
    pairs  = [(x, adj) for x, adj in pairs if x.shape[0] <= MAX_NODES]
    if len(pairs) < before:
        print(f"Filtered: {before} → {len(pairs)} (max_nodes={MAX_NODES})")

    dataset = EllipticDenseDataset(pairs)
    print(f"Dataset: {len(dataset)} graphs  [{time.time()-start_time:.1f}s]")

    # -- Feature normalisation stats (skip col 0 — binary label) ---------------
    all_x  = torch.cat([x for x, _ in pairs], dim=0)
    x_mean = torch.zeros(NODE_DIM)
    x_std  = torch.ones(NODE_DIM)
    x_mean[1:] = all_x[:, 1:].mean(0)
    x_std[1:]  = all_x[:, 1:].std(0).clamp(min=1e-6)
    x_mean = x_mean.to(DEVICE)
    x_std  = x_std.to(DEVICE)
    print(f"Feature means: {x_mean.tolist()}")
    print(f"Feature stds : {x_std.tolist()}")

    # -- DataLoader ------------------------------------------------------------
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,          # Elliptic dataset is small; no worker overhead needed
        pin_memory=(DEVICE == "cuda"),
    )

    steps_per_epoch = len(loader)
    print(f"Batches/epoch: {steps_per_epoch}  |  Total: {steps_per_epoch * EPOCHS}")

    # -- Model -----------------------------------------------------------------
    model     = DiffusionGNN(node_dim=NODE_DIM, hidden_dim=HIDDEN, num_layers=4).to(DEVICE)
    diffusion = create_diffusion(TIMESTEPS)
    print(f"Model on {DEVICE}  |  params: {sum(p.numel() for p in model.parameters()):,}")

    schedule_sampler = LossSecondMomentResampler(diffusion)
    optimizer        = Adam(model.parameters(), lr=LR)
    scheduler        = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    CKPT_PATH.parent.mkdir(exist_ok=True)

    # -- Training loop ---------------------------------------------------------
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        for step, (x, adj, node_mask) in enumerate(loader):
            x         = x.to(DEVICE, dtype=torch.float32, non_blocking=True)
            adj       = adj.to(DEVICE, dtype=torch.float32, non_blocking=True)
            node_mask = node_mask.to(DEVICE, dtype=torch.float32, non_blocking=True)

            x_norm         = x.clone()
            x_norm[:, :, 1:] = (x[:, :, 1:] - x_mean[1:]) / x_std[1:]
            x_norm         = x_norm * node_mask.unsqueeze(-1)
            adj            = adj * node_mask[:, :, None] * node_mask[:, None, :]

            B = x_norm.shape[0]
            t, iw = schedule_sampler.sample(B, DEVICE)

            with torch.amp.autocast(device_type=DEVICE, enabled=USE_AMP):
                loss_dict = diffusion.training_losses(
                    model,
                    x_start=x_norm,
                    t=t,
                    adj_start=adj,
                    model_kwargs={"node_mask": node_mask},
                    adj_loss_weight=ADJ_LOSS_W,
                    density_loss_weight=DENSITY_LOSS_W,
                    node_exist_loss_weight=NODE_EXIST_LOSS_W,
                    mask_dropout_rate=MASK_DROPOUT_RATE,
                    ghost_node_rate=GHOST_NODE_RATE,
                    laund_loss_weight=LAUND_LOSS_W,
                )
                loss = (loss_dict["loss"] * iw).mean()

            scaler.scale(loss).backward()
            schedule_sampler.update(t.tolist(), loss_dict["loss"].detach().tolist())

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            total_loss += loss.item()

        scheduler.step()

        avg_loss   = total_loss / len(loader)
        current_lr = scheduler.get_last_lr()[0]

        if (epoch + 1) % 10 == 0 or epoch < 3:
            print(f"Epoch {epoch+1:4d}/{EPOCHS} | loss={avg_loss:.4f} | "
                  f"lr={current_lr:.2e} | {time.time()-start_time:.0f}s")

        torch.save({
            "model":  model.state_dict(),
            "x_mean": x_mean.cpu(),
            "x_std":  x_std.cpu(),
        }, CKPT_PATH)

    print(f"Training complete — checkpoint saved → {CKPT_PATH}  "
          f"[{time.time()-start_time:.0f}s]")
