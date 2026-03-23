import os
import sys
import torch
from tqdm import tqdm

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

BASE_DIR = Path(__file__).resolve().parent
DIFF_DIR = BASE_DIR.parent / "diffusion"

# igraph_version/ — makes `diffusion.*` and `simclr.*` resolve as packages
if str(DIFF_DIR.parent) not in sys.path:
    sys.path.insert(0, str(DIFF_DIR.parent))
# diffusion/ itself — lets bare intra-package imports (masked_diffusion etc.) resolve
if str(DIFF_DIR) not in sys.path:
    sys.path.insert(0, str(DIFF_DIR))
# simclr/ itself — lets bare intra-module imports inside simclr.py resolve
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from diffusion.model import DiffusionGNN
from diffusion.diff_util import create_diffusion, network_to_dense

from simclr import GraphEncoder, ProjectionHead, train_simclr_fast
from augmentation import build_igraph_from_transactions
from util import preprocess_df, extract_laundering_networks_igraph, extract_non_laundering_networks_igraph

# Separate cache from the diffusion (x,adj)-tuple cache — stores full network dicts with igraph graphs
SIMCLR_CACHE = BASE_DIR / "simclr_networks_cache.pt"
MAX_NODES = 300

def load_or_build_networks(df_full):
    """
    Load pre-built networks from disk, or extract + precompute them on first run.

    Each network dict includes:
      - graph:           igraph.Graph  (for augmentation)
      - x_dense:        Tensor [n, 7]  (precomputed node features)
      - adj_dense:      Tensor [n, n]  (precomputed adjacency)
      - laundering_nodes, collapsed_nodes, node_depths, nodes, start_node
    """
    if SIMCLR_CACHE.exists():
        print(f"[cache] Loading networks from {SIMCLR_CACHE} ...")
        networks = torch.load(SIMCLR_CACHE, weights_only=False)
        print(f"[cache] Loaded {len(networks)} networks.")
        return networks

    print("[cache] No cache found — extracting networks from scratch ...")

    with_laund = extract_laundering_networks_igraph(
        df_full, max_depth=4, max_networks=2000, collapse_threshold=10, max_nodes=MAX_NODES
    )
    non_laund = extract_non_laundering_networks_igraph(
        df_full, max_depth=5, max_networks=len(with_laund), collapse_threshold=10, max_nodes=MAX_NODES
    )
    networks = with_laund + non_laund

    print(f"[cache] Extracted {len(networks)} networks. Building igraph graphs and precomputing features ...")

    def _build_one(net):
        net["graph"]     = build_igraph_from_transactions(net["transactions"])
        x, adj           = network_to_dense(net)
        net["x_dense"]   = x
        net["adj_dense"]  = adj
        del net["transactions"]
        return net

    workers = min(os.cpu_count() or 4, 8)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        networks = list(tqdm(pool.map(_build_one, networks), total=len(networks), desc="Building networks"))
    print(f"[cache] All {len(networks)} networks processed.")

    torch.save(networks, SIMCLR_CACHE)
    print(f"[cache] Saved to {SIMCLR_CACHE}")
    return networks


def load_diffusion(device):
    """Load the trained diffusion model from the sibling diffusion directory."""
    ckpt_path = DIFF_DIR / "model.pt"
    if not ckpt_path.exists():
        print(f"[diffusion] No checkpoint at {ckpt_path} — diffusion augmentation disabled.")
        return None, None, None, None

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    networks = load_or_build_networks(df_full)

    # Load the trained diffusion model (optional — falls back gracefully if absent)
    diff_model, diffusion, x_mean, x_std = load_diffusion(device)

    encoder   = GraphEncoder(in_dim=7, hidden_dim=64, out_dim=128).to(device)
    projector = ProjectionHead(in_dim=128, proj_dim=64).to(device)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(projector.parameters()),
        lr=1e-3,
    )

    train_simclr_fast(
        networks=networks,
        full_df=df_full,
        encoder=encoder,
        projector=projector,
        optimizer=optimizer,
        device=device,
        batch_size=128,
        epochs=100,
        checkpoint_dir=str(BASE_DIR / "model_checkpoints"),
        # Diffusion augmentation (ignored if diff_model is None)
        diffusion_model=diff_model,
        diffusion=diffusion,
        x_mean=x_mean,
        x_std=x_std,
        p_diffusion=0.3,
        diffusion_t_frac=0.3,
        max_nodes=300,
    )
