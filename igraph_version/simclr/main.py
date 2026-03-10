from augmentation import augment_network_view_fast
from simclr import *
import os
import re

import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from io import StringIO
import pandas as pd
import pickle
import igraph as ig
import torch

from util import *
from plotting_helpers import *

from simclr import * 
from augmentation import *

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

if __name__ == "__main__":
    current_dir = os.path.dirname(os.getcwd())
    file_path = os.path.join(current_dir, 'grad', "data", "IBM", "Hi-Small_Trans.csv")
    CSV_PATH = BASE_DIR / file_path
    df_full = preprocess_df(CSV_PATH)

    with_laund_networks = extract_laundering_networks_igraph(
    df_full,
    max_depth=4,
    max_networks=2000,
    collapse_threshold=10
    )    

    non_laundering_networks = extract_non_laundering_networks_igraph(
    df_full,
    max_depth=5,
    max_networks=len(with_laund_networks),
    collapse_threshold=10)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = GraphEncoder(in_dim=3, hidden_dim=64, out_dim=128).to(device)
    projector = ProjectionHead(in_dim=128, proj_dim=64).to(device)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(projector.parameters()),
        lr=1e-3
    )

    networks = with_laund_networks + non_laundering_networks # laundering + non-laundering

    train_simclr_fast(
        networks=networks,     
        full_df=df_full,
        encoder=encoder,
        projector=projector,
        optimizer=optimizer,
        device=device,
        batch_size=128,
        epochs=1000
    )

    plot_simclr_latent_space_laundering_vs_clean(
        networks,
        df_full
    )