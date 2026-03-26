import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================
# Timestep Embedding
# ======================================================

def timestep_embedding(timesteps, dim, max_period=10000):

    """
    Create sinusoidal timestep embeddings.
    """

    half = dim // 2

    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, dtype=torch.float32)
        / half
    ).to(timesteps.device)

    args = timesteps[:, None].float() * freqs[None]

    emb = torch.cat(
        [torch.cos(args), torch.sin(args)],
        dim=-1
    )

    if dim % 2:
        emb = torch.cat(
            [emb, torch.zeros_like(emb[:, :1])],
            dim=-1
        )

    return emb


# ======================================================
# Masked Graph Convolution
# ======================================================

class MaskedGraphConv(nn.Module):

    """
    Simple masked message passing layer.
    """

    def __init__(self, in_dim, out_dim):

        super().__init__()

        self.lin_self = nn.Linear(in_dim, out_dim)
        self.lin_neigh = nn.Linear(in_dim, out_dim)


    def forward(self, x, adj, node_mask):

        """
        x   : [B, N, F]
        adj : [B, N, N]
        """

        # Mask padded nodes
        mask = node_mask.unsqueeze(-1)

        x = x * mask

        # Aggregate neighbors
        deg = adj.sum(dim=-1, keepdim=True).clamp(min=1)

        neigh = torch.bmm(adj, x) / deg

        out = (
            self.lin_self(x)
            + self.lin_neigh(neigh)
        )

        return out * mask


# ======================================================
# Residual GNN Block
# ======================================================

class GNNBlock(nn.Module):

    def __init__(self, dim, time_dim):

        super().__init__()

        self.conv1 = MaskedGraphConv(dim, dim)
        self.conv2 = MaskedGraphConv(dim, dim)

        # AdaLN: project time to scale+shift for each of the 2 norms
        self.time_proj = nn.Linear(time_dim, dim * 4)

        # elementwise_affine=False — AdaLN supplies the affine transform
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)


    def forward(self, x, adj, t_emb, node_mask):

        """
        x     : [B, N, D]
        t_emb : [B, T]
        """

        # AdaLN scale + shift for norm1 and norm2
        t = self.time_proj(t_emb)               # [B, 4*dim]
        s1, b1, s2, b2 = t.chunk(4, dim=-1)     # each [B, dim]

        h = self.conv1(x, adj, node_mask)
        h = self.norm1(h) * (1 + s1[:, None, :]) + b1[:, None, :]
        h = F.silu(h)

        h = self.conv2(h, adj, node_mask)
        h = self.norm2(h) * (1 + s2[:, None, :]) + b2[:, None, :]
        h = F.silu(h)

        return x + h


# ======================================================
# Full Diffusion GNN
# ======================================================

class DiffusionGNN(nn.Module):

    """
    Masked GNN for DDPM on graphs.
    Predicts epsilon.
    """

    def __init__(
        self,
        node_dim,
        hidden_dim=128,
        time_dim=128,
        num_layers=4,
    ):

        super().__init__()

        self.time_dim = time_dim


        # Time embedding MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )


        # Input projection
        self.input_proj = nn.Linear(node_dim, hidden_dim)


        # GNN Blocks
        self.blocks = nn.ModuleList([
            GNNBlock(hidden_dim, time_dim)
            for _ in range(num_layers)
        ])


        # Output projection
        self.output_proj = nn.Linear(hidden_dim, node_dim)

        # Adjacency decoder: decomposed bilinear form.
        # logit(i,j) = h_i^T W h_j  +  u^T h_i  +  u^T h_j  +  bias
        # Separating pairwise (bilinear) and per-node (unary) terms avoids the
        # symmetric inner-product's collapse: when embeddings over-smooth to the
        # same vector the bilinear term saturates, but unary + bias still
        # controls the global density prior.  W is not constrained to be
        # symmetric, giving strictly more expressiveness than inner-product.
        D_e = hidden_dim // 4                                # 32 for hidden=128
        self.adj_proj  = nn.Linear(hidden_dim, D_e)
        self.adj_bilin = nn.Linear(D_e, D_e, bias=False)    # W in h_i^T W h_j
        self.adj_lin   = nn.Linear(D_e, 1,   bias=False)    # unary: u^T h
        self.adj_bias  = nn.Parameter(torch.tensor(-2.25))  # ≈ logit(0.095) sparse prior
        nn.init.normal_(self.adj_bilin.weight, std=0.01)
        nn.init.normal_(self.adj_lin.weight,   std=0.01)
        nn.init.normal_(self.adj_proj.weight,  std=0.02)

        # Node existence prediction: learned from the same node embeddings.
        # Predicts a logit per node — whether that node should be active.
        self.node_existence_head = nn.Linear(hidden_dim, 1)


    def forward(self, x, t, adj=None, node_mask=None):

        """
        x         : [B, N, F]
        t         : [B]
        adj       : [B, N, N]  — noisy adj used for message passing
        node_mask : [B, N]

        Returns
        -------
        out      : [B, N, F]   predicted noise / x_start per node feature
        adj_pred : [B, N, N]   predicted clean adjacency (sigmoid, symmetric)
        """

        if node_mask is not None:
            x = x * node_mask.unsqueeze(-1)


        # Time embedding
        t_emb = timestep_embedding(t, self.time_dim)

        t_emb = self.time_mlp(t_emb)


        # Project input
        h = self.input_proj(x)


        # Message passing
        for block in self.blocks:
            h = block(h, adj, t_emb, node_mask)


        # Predict node features (noise or x_start depending on mode)
        out = self.output_proj(h)

        if node_mask is not None:
            out = out * node_mask.unsqueeze(-1)


        # Adjacency prediction via decomposed bilinear form (no large intermediate tensors)
        h_e   = F.silu(self.adj_proj(h))                          # [B, N, D_e]
        bilin = torch.bmm(self.adj_bilin(h_e), h_e.transpose(1, 2))  # [B, N, N]
        bilin = (bilin + bilin.transpose(1, 2)) / 2               # symmetrise W
        unary = self.adj_lin(h_e).squeeze(-1)                     # [B, N]
        adj_logits = bilin + unary.unsqueeze(2) + unary.unsqueeze(1) + self.adj_bias
        adj_pred = torch.sigmoid(adj_logits)
        adj_pred = (adj_pred + adj_pred.transpose(1, 2)) / 2     # enforce symmetry

        if node_mask is not None:
            adj_pred = adj_pred * node_mask[:, :, None] * node_mask[:, None, :]

        # Node existence logits — raw (not sigmoid'd), one per padded position
        node_logits = self.node_existence_head(h).squeeze(-1)   # [B, N]

        return out, adj_pred, node_logits