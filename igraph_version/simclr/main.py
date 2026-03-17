from augmentation import augment_network_view_fast
from simclr import *
import sys

import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import igraph as ig
import torch

from util import *
from plotting_helpers import *

from simclr import *
from augmentation import *

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DIFF_DIR = BASE_DIR.parent / "diffusion"

# Add igraph_version/ so that `diffusion.model` / `diffusion.diff_util` resolve as a package
if str(DIFF_DIR.parent) not in sys.path:
    sys.path.insert(0, str(DIFF_DIR.parent))

from diffusion.model import DiffusionGNN
from diffusion.diff_util import create_diffusion


def load_diffusion(device):
    """Load the trained diffusion model from the sibling diffusion directory."""
    ckpt_path = DIFF_DIR / "model.pt"
    if not ckpt_path.exists():
        print(f"[diffusion] No checkpoint found at {ckpt_path} — diffusion augmentation disabled.")
        return None, None, None, None

    ckpt = torch.load(ckpt_path, map_location=device)

    diff_model = DiffusionGNN(node_dim=7, hidden_dim=128, num_layers=4).to(device)
    diff_model.load_state_dict(ckpt["model"])
    diff_model.eval()

    diffusion = create_diffusion(T=500)
    x_mean   = ckpt["x_mean"].to(device)
    x_std    = ckpt["x_std"].to(device)

    print(f"[diffusion] Loaded checkpoint from {ckpt_path}")
    return diff_model, diffusion, x_mean, x_std


if __name__ == "__main__":
    CSV_PATH = BASE_DIR.parents[1] / "data" / "IBM" / "Hi-Small_Trans.csv"
    df_full  = preprocess_df(str(CSV_PATH))

    with_laund_networks = extract_laundering_networks_igraph(
        df_full,
        max_depth=4,
        max_networks=2000,
        collapse_threshold=10,
    )

    non_laundering_networks = extract_non_laundering_networks_igraph(
        df_full,
        max_depth=5,
        max_networks=len(with_laund_networks),
        collapse_threshold=10,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the trained diffusion model (optional — falls back gracefully if absent)
    diff_model, diffusion, x_mean, x_std = load_diffusion(device)

    encoder   = GraphEncoder(in_dim=3, hidden_dim=64, out_dim=128).to(device)
    projector = ProjectionHead(in_dim=128, proj_dim=64).to(device)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(projector.parameters()),
        lr=1e-3,
    )

    networks = with_laund_networks + non_laundering_networks

    train_simclr_fast(
        networks=networks,
        full_df=df_full,
        encoder=encoder,
        projector=projector,
        optimizer=optimizer,
        device=device,
        batch_size=128,
        epochs=1000,
        # Diffusion augmentation (ignored if diff_model is None)
        diffusion_model=diff_model,
        diffusion=diffusion,
        x_mean=x_mean,
        x_std=x_std,
        p_diffusion=0.3,
        diffusion_t_frac=0.3,
        max_nodes=300,
    )

    plot_simclr_latent_space_laundering_vs_clean(networks, df_full)
