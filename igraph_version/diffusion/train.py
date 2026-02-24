import torch
import os
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim import Adam

from torch.cuda.amp import autocast, GradScaler


from dataset import CachedDataset
from collate import collate_fn

BASE_DIR = Path(__file__).resolve().parent

from augmentation import build_igraph_from_transactions
from model import DiffusionGNN
from diff_util import create_diffusion, preprocess
from util import extract_laundering_networks_igraph, extract_non_laundering_networks_igraph, preprocess_df

scaler = GradScaler()

# ----------------------------------
# Config
# ----------------------------------

BATCH_SIZE = 32          # you can increase now
LR = 2e-4
EPOCHS = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NODE_DIM = 3
HIDDEN = 128
TIMESTEPS = 1000


# ----------------------------------
# Dataset
# ----------------------------------
if "cached_dataset.pt" not in os.listdir():
    current_dir = os.path.dirname(os.getcwd())
    CSV_PATH = os.path.join(current_dir, "data", "IBM", "Hi-Small_Trans.csv")
    df_full = preprocess_df(CSV_PATH)

    with_laund_networks = extract_laundering_networks_igraph(
    df_full,
    max_depth=2,
    max_networks=2000,
    collapse_threshold=10
    ) 

    non_laundering_networks = extract_non_laundering_networks_igraph(
    df_full,
    max_depth=2,
    max_networks=len(with_laund_networks),
    collapse_threshold=10)

    networks = with_laund_networks + non_laundering_networks # laundering + non-laundering

    for net in networks:
        net["graph"] = build_igraph_from_transactions(net["transactions"])
    
    preprocess(networks)

dataset = CachedDataset("cached_dataset.pt")


loader = DataLoader(

    dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    collate_fn=collate_fn,

    num_workers=4,      # IMPORTANT
    pin_memory=True,    # IMPORTANT

    persistent_workers=True,
)


# ----------------------------------
# Model
# ----------------------------------

model = DiffusionGNN(
    node_dim=NODE_DIM,
    hidden_dim=HIDDEN,
    num_layers=4
).to(DEVICE)


# ----------------------------------
# Diffusion
# ----------------------------------

diffusion = create_diffusion(TIMESTEPS)


# ----------------------------------
# Optimizer
# ----------------------------------

optimizer = Adam(model.parameters(), lr=LR)


# ----------------------------------
# Training
# ----------------------------------

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0


    for x, adj, node_mask in loader:

        x = x.to(DEVICE, non_blocking=True)
        adj = adj.to(DEVICE, non_blocking=True)
        node_mask = node_mask.to(DEVICE, non_blocking=True)


        # Mask adj
        adj = adj * node_mask[:, :, None] * node_mask[:, None, :]


        B = x.shape[0]


        t = torch.randint(
            0,
            diffusion.num_timesteps,
            (B,),
            device=DEVICE
        )


        with autocast():

            loss_dict = diffusion.training_losses(
                model,
                x_start=x,
                t=t,
                model_kwargs={
                    "adj": adj,
                    "node_mask": node_mask
                }
            )

            loss = loss_dict["loss"].mean()


        optimizer.zero_grad()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


        total_loss += loss.item()


    avg_loss = total_loss / len(loader)

    print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f}")


    torch.save(model.state_dict(), "model.pt")