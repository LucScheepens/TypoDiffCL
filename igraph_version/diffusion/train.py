import math
import os
import sys
import time
from pathlib import Path

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR.parent))   # igraph_version/ — makes simclr importable

from collate import collate_fn
from dataset import CachedDataset
from diff_util import create_diffusion, preprocess
from model import DiffusionGNN
from visualizations import run_all_visualizations

from simclr.augmentation import build_igraph_from_transactions
from simclr.util import (
    extract_laundering_networks_igraph,
    extract_non_laundering_networks_igraph,
    preprocess_df,
)


BATCH_SIZE    = 16
ACCUM_STEPS   = 1
LR            = 2e-4
EPOCHS        = 300
WARMUP_EPOCHS = 15
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

NODE_DIM      = 7
HIDDEN        = 128
TIMESTEPS     = 500
MAX_NODES     = 300
ADJ_LOSS_W    = 0.3
DEG_LOSS_W    = 0.1
LAUND_LOSS_W  = 2.0    # upweight laundering BCE — rare class needs stronger signal
MAX_GRAD_NORM = 1.0

VIZ_INTERVAL  = 50
GRAPHS_DIR    = str(BASE_DIR / "graphs")
CACHE_PATH    = str(BASE_DIR / "cached_dataset.pt")


def _needs_rebuild():
    if not os.path.exists(CACHE_PATH):
        return True
    sample = torch.load(CACHE_PATH)
    if len(sample) == 0 or sample[0][0].shape[1] != NODE_DIM:
        print(f"Cache feature dim {sample[0][0].shape[1]} != NODE_DIM={NODE_DIM}. Rebuilding...")
        os.remove(CACHE_PATH)
        return True
    return False


def lr_lambda(epoch):
    if epoch < WARMUP_EPOCHS:
        return (epoch + 1) / WARMUP_EPOCHS
    progress = (epoch - WARMUP_EPOCHS) / max(1, EPOCHS - WARMUP_EPOCHS)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


if __name__ == "__main__":

    USE_AMP    = (DEVICE == "cuda")
    scaler     = torch.amp.GradScaler("cuda", enabled=USE_AMP)
    start_time = time.time()

    # -- Dataset -------------------------------------------------------------

    if _needs_rebuild():
        CSV_PATH = str(BASE_DIR.parents[2] / 'grad' / "data" / "IBM" / "Hi-Small_Trans.csv")
        df_full  = preprocess_df(CSV_PATH)
        print(f"Loaded {len(df_full)} rows  [{time.time()-start_time:.1f}s]")

        with_laund = extract_laundering_networks_igraph(
            df_full, max_depth=4, max_networks=2000,
            collapse_threshold=10, max_nodes=MAX_NODES,
        )
        print(f"Laundering networks: {len(with_laund)}  [{time.time()-start_time:.1f}s]")

        non_laund = extract_non_laundering_networks_igraph(
            df_full, max_depth=4, max_networks=len(with_laund),
            collapse_threshold=10, max_nodes=MAX_NODES,
        )
        print(f"Non-laundering networks: {len(non_laund)}  [{time.time()-start_time:.1f}s]")

        networks = with_laund + non_laund
        for net in networks:
            net["graph"] = build_igraph_from_transactions(net["transactions"])
        preprocess(networks, save_path=CACHE_PATH)

    dataset = CachedDataset(CACHE_PATH, max_nodes=MAX_NODES)
    print(f"Dataset: {len(dataset)} graphs  [{time.time()-start_time:.1f}s]")

    # -- Feature normalisation stats -----------------------------------------

    all_x  = torch.cat([x for x, _ in dataset], dim=0)
    x_mean = torch.zeros(NODE_DIM)
    x_std  = torch.ones(NODE_DIM)
    x_mean[1:] = all_x[:, 1:].mean(0)
    x_std[1:]  = all_x[:, 1:].std(0).clamp(min=1e-6)
    x_mean = x_mean.to(DEVICE)
    x_std  = x_std.to(DEVICE)
    print(f"Feature means: {x_mean.tolist()}")
    print(f"Feature stds : {x_std.tolist()}")

    # -- DataLoader ----------------------------------------------------------

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=(DEVICE == "cuda"),
        persistent_workers=True,
    )

    steps_per_epoch = max(1, len(loader) // ACCUM_STEPS)
    print(f"Batches/epoch: {len(loader)}  |  Opt steps/epoch: {steps_per_epoch}  "
          f"|  Total: {steps_per_epoch * EPOCHS}")

    # -- Model ---------------------------------------------------------------

    model     = DiffusionGNN(node_dim=NODE_DIM, hidden_dim=HIDDEN, num_layers=4).to(DEVICE)
    diffusion = create_diffusion(TIMESTEPS)
    print(f"Model on {DEVICE}  |  params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    os.makedirs(GRAPHS_DIR, exist_ok=True)

    # -- Training loop -------------------------------------------------------

    for epoch in range(EPOCHS):

        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        for step, (x, adj, node_mask) in enumerate(loader):

            x         = x.to(DEVICE, dtype=torch.float32, non_blocking=True)
            adj       = adj.to(DEVICE, dtype=torch.float32, non_blocking=True)
            node_mask = node_mask.to(DEVICE, dtype=torch.float32, non_blocking=True)

            x_norm = x.clone()
            x_norm[:, :, 1:] = (x[:, :, 1:] - x_mean[1:]) / x_std[1:]
            x_norm = x_norm * node_mask.unsqueeze(-1)

            adj = adj * node_mask[:, :, None] * node_mask[:, None, :]

            B = x_norm.shape[0]
            t = torch.randint(0, diffusion.num_timesteps, (B,), device=DEVICE)

            with torch.amp.autocast(device_type=DEVICE, enabled=USE_AMP):
                loss_dict = diffusion.training_losses(
                    model,
                    x_start=x_norm,
                    t=t,
                    adj_start=adj,
                    model_kwargs={"node_mask": node_mask},
                    adj_loss_weight=ADJ_LOSS_W,
                    deg_loss_weight=DEG_LOSS_W,
                    laund_loss_weight=LAUND_LOSS_W,
                )
                loss = loss_dict["loss"].mean() / ACCUM_STEPS

            scaler.scale(loss).backward()

            if (step + 1) % ACCUM_STEPS == 0 or (step + 1) == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item() * ACCUM_STEPS

        scheduler.step()

        avg_loss   = total_loss / len(loader)
        current_lr = scheduler.get_last_lr()[0]

        if (epoch + 1) % 10 == 0 or epoch < 3:
            print(f"Epoch {epoch+1:4d}/{EPOCHS} | loss={avg_loss:.4f} | lr={current_lr:.2e} "
                  f"| {time.time()-start_time:.0f}s")

        torch.save({
            "model": model.state_dict(),
            "x_mean": x_mean.cpu(),
            "x_std":  x_std.cpu(),
        }, BASE_DIR / "model.pt")

        if (epoch + 1) % VIZ_INTERVAL == 0 or (epoch + 1) == EPOCHS:
            model.eval()
            run_all_visualizations(
                model, diffusion, dataset,
                x_mean.cpu(), x_std.cpu(),
                timesteps=TIMESTEPS,
                adj_loss_w=ADJ_LOSS_W,
                deg_loss_w=DEG_LOSS_W,
                laund_loss_w=LAUND_LOSS_W,
                device=DEVICE,
                graphs_dir=GRAPHS_DIR,
                epoch=epoch + 1,
            )
            model.train()

    print(f"Training complete  [{time.time()-start_time:.0f}s]")

