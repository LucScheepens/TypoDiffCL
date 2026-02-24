import os
import random
import torch
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
import time

import torch.nn as nn
import torch.nn.functional as F

from augmentation import augment_network_view_fast, build_igraph_from_transactions

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
    checkpoint_interval=10
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

            views1 = []
            views2 = []
            for net in batch:
                v1 = augment_network_view_fast(net, full_graph)
                v2 = augment_network_view_fast(net, full_graph)

                views1.append(network_to_pyg_data_fast(v1))
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
