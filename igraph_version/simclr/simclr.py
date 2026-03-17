import os
import sys
import random
import torch
from pathlib import Path


# Make the diffusion package importable from the sibling directory
_DIFF_DIR = Path(__file__).resolve().parent.parent / "diffusion"

if str(_DIFF_DIR) not in sys.path:
    sys.path.insert(0, str(_DIFF_DIR))


BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR.parent)) 

from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
import time

import torch.nn as nn
import torch.nn.functional as F

from augmentation import augment_network_view_fast, build_igraph_from_transactions
from diffusion.diff_util import network_to_dense




def prepare_networks(networks, full_df):
    full_graph = build_igraph_from_transactions(full_df)

    for net in networks:
        net["graph"] = build_igraph_from_transactions(net["transactions"])

    return full_graph


def network_to_pyg_data_fast(network):
    nodes = list(network["nodes"])
    node_idx = {n: i for i, n in enumerate(nodes)}

    tx = network["transactions"]

    src = tx["From_Account_int"].map(node_idx)
    dst = tx["To_Account_int"].map(node_idx)

    mask = src.notna() & dst.notna()
    src = src[mask].astype(int)
    dst = dst[mask].astype(int)

    edge_index = torch.stack([
        torch.cat([torch.tensor(src.values), torch.tensor(dst.values)]),
        torch.cat([torch.tensor(dst.values), torch.tensor(src.values)])
    ], dim=0).long()

    in_deg = tx["To_Account_int"].value_counts()
    out_deg = tx["From_Account_int"].value_counts()

    x = torch.tensor([
        [
            in_deg.get(n, 0),
            out_deg.get(n, 0),
            1 if n in network["laundering_nodes"] else 0
        ]
        for n in nodes
    ], dtype=torch.float)

    return Data(x=x, edge_index=edge_index)


class GraphEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, out_dim=128):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.lin = nn.Linear(hidden_dim, out_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = global_mean_pool(x, batch)
        x = self.lin(x)

        return x


class ProjectionHead(nn.Module):
    def __init__(self, in_dim, proj_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, proj_dim)
        )

    def forward(self, z):
        return self.net(z)



def nt_xent_loss(z1, z2, temperature=0.5):
    """
    z1, z2: (batch_size, dim)
    """

    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    batch_size = z1.size(0)

    z = torch.cat([z1, z2], dim=0)

    sim = torch.matmul(z, z.t()) / temperature
    sim_exp = torch.exp(sim)

    mask = ~torch.eye(2 * batch_size, dtype=bool, device=z.device)

    sim_exp = sim_exp * mask

    pos_sim = torch.exp(torch.sum(z1 * z2, dim=1) / temperature)
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)

    denom = sim_exp.sum(dim=1)

    loss = -torch.log(pos_sim / denom)

    return loss.mean()





def _diffusion_view(network, diffusion_model, diffusion, x_mean, x_std,
                    max_nodes=300, t_frac=0.3, device="cpu"):
    """
    Generate one augmented view via the diffusion model.

    Strategy: forward-noise the graph to timestep t, then run a single
    model forward pass to recover x_0 and adj_pred.  This is O(1) inference
    (no iterative denoising loop) so it adds negligible overhead to SimCLR.

    Returns a PyG Data object with 3 node features compatible with GraphEncoder:
        [degree, degree, laundering_probability]
    """

    x, adj = network_to_dense(network)
    n = x.shape[0]

    if n > max_nodes:
        # Graph too large for the diffusion model — fall back silently
        return None

    # Pad to max_nodes and build node mask
    x_pad    = torch.zeros(1, max_nodes, 7)
    adj_pad  = torch.zeros(1, max_nodes, max_nodes)
    mask     = torch.zeros(1, max_nodes)
    x_pad[0, :n]       = x
    adj_pad[0, :n, :n] = adj
    mask[0, :n]        = 1.0

    # Normalise continuous features (match train.py)
    x_norm = x_pad.clone()
    x_norm[:, :, 1:] = (x_pad[:, :, 1:] - x_mean[1:]) / x_std[1:]
    x_norm = x_norm * mask.unsqueeze(-1)
    adj_pad = adj_pad * mask[:, :, None] * mask[:, None, :]

    x_norm  = x_norm.to(device)
    adj_pad = adj_pad.to(device)
    mask    = mask.to(device)

    # Choose noise level
    t_abs = max(1, int(t_frac * diffusion.num_timesteps))
    t     = torch.tensor([t_abs], device=device)

    # Forward diffusion: corrupt features and adjacency
    x_t, adj_t = diffusion.q_sample(x_norm, t, node_mask=mask, adj_start=adj_pad)

    # Single-step denoising: one forward pass to predict x_0 and adj
    was_training = diffusion_model.training
    diffusion_model.eval()
    with torch.no_grad():
        eps_pred, adj_pred = diffusion_model(
            x_t, diffusion._scale_timesteps(t),
            adj=adj_t, node_mask=mask,
        )
        # Recover x_0 for continuous features from predicted epsilon
        x0_cont = diffusion._predict_xstart_from_eps(x_t[..., 1:], t, eps_pred[..., 1:])
        # Binary laundering feature: model predicts x_start directly
        x0_bin  = eps_pred[..., 0:1].clamp(0.0, 1.0)
    if was_training:
        diffusion_model.train()

    # Extract valid-node slice
    x0_node  = torch.cat([x0_bin, x0_cont], dim=-1)[0, :n]   # [n, 7]
    adj_node = adj_pred[0, :n, :n]                             # [n, n], values in [0,1]

    # Build PyG-compatible node features: (degree, degree, laundering_prob)
    deg   = adj_node.sum(dim=-1)           # soft degree [n]
    laund = x0_node[:, 0]                  # laundering probability [n]
    x_pyg = torch.stack([deg, deg, laund], dim=-1)  # [n, 3]

    # Build edge_index from thresholded adjacency
    edge_index = (adj_node > 0.5).nonzero(as_tuple=False).T.contiguous()  # [2, E]

    return Data(x=x_pyg.cpu(), edge_index=edge_index.cpu())


def train_simclr_fast(
    networks,
    full_df,
    encoder,
    projector,
    optimizer,
    device,
    batch_size=8,
    epochs=50,
    checkpoint_dir="model_checkpoints",
    checkpoint_interval=10,
    diffusion_model=None,
    diffusion=None,
    x_mean=None,
    x_std=None,
    p_diffusion=0.3,
    diffusion_t_frac=0.3,
    max_nodes=300,
):
    encoder.train()
    projector.train()

    # 🔥 Build graphs ONCE
    full_graph = prepare_networks(networks, full_df)

    os.makedirs(checkpoint_dir, exist_ok=True)

    best_loss = float('inf')
    best_encoder_state = None
    best_projector_state = None
    best_epoch = None

    # ⏱️ Timing containers
    start_time = time.time()
    epoch_times = []
    total_batches = 0

    for epoch in range(epochs):
        epoch_start = time.time()

        print(f"Epoch {epoch + 1}/{epochs}")
        random.shuffle(networks)
        total_loss = 0.0

        for i in range(0, len(networks), batch_size):
            batch_start = time.time()
            total_batches += 1

            batch = networks[i:i + batch_size]

            use_diffusion = (
                diffusion_model is not None
                and diffusion is not None
                and x_mean is not None
                and x_std is not None
            )

            views1 = []
            views2 = []
            for net in batch:
                v1 = augment_network_view_fast(net, full_graph)
                views1.append(network_to_pyg_data_fast(v1))

                # View 2: use diffusion augmentation with probability p_diffusion
                if use_diffusion and random.random() < p_diffusion:
                    diff_view = _diffusion_view(
                        net, diffusion_model, diffusion,
                        x_mean, x_std, max_nodes, diffusion_t_frac, device,
                    )
                    if diff_view is not None:
                        views2.append(diff_view)
                        continue
                # Fall back to standard structural augmentation
                v2 = augment_network_view_fast(net, full_graph)
                views2.append(network_to_pyg_data_fast(v2))

            data1 = Batch.from_data_list(views1).to(device)
            data2 = Batch.from_data_list(views2).to(device)

            optimizer.zero_grad()

            h1 = encoder(data1)
            h2 = encoder(data2)

            z1 = projector(h1)
            z2 = projector(h2)

            loss = nt_xent_loss(z1, z2)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / ((len(networks) + batch_size - 1) // batch_size)
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        print(f"Epoch {epoch + 1}: avg loss = {avg_loss:.4f} | time = {epoch_time:.2f}s")

        # ✅ Save checkpoint every N epochs
        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch + 1}.pt")
            torch.save({
                'encoder_state_dict': encoder.state_dict(),
                'projector_state_dict': projector.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch + 1,
                'loss': avg_loss
            }, checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")

        # ✅ Track best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_encoder_state = encoder.state_dict()
            best_projector_state = projector.state_dict()
            best_epoch = epoch + 1
            print(f"New best model at epoch {epoch + 1} with loss {best_loss:.4f}")

    # 🔥 Save best model at the end
    best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
    torch.save({
        'encoder_state_dict': best_encoder_state,
        'projector_state_dict': best_projector_state,
        'loss': best_loss
    }, best_model_path)

    total_time = time.time() - start_time

    print(f"Best model saved at {best_model_path} with loss {best_loss:.4f}")
    print(f"Total training time: {total_time:.2f}s")
