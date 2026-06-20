"""
gan_augmentation.py
───────────────────
Conditional GAN augmentation for laundering ego-subgraph oversampling.

Architecture
────────────
Both Generator and Discriminator operate in the mean-pooled node-feature space
(18-dim, same representation used by GraphSMOTE and the tabular baselines).

  Generator   : z ~ N(0,I)  →  MLP  →  synthetic graph embedding  (18-dim)
  Discriminator: embedding  →  MLP  →  real / fake logit

Training uses Wasserstein loss with gradient penalty (WGAN-GP, Gulrajani 2017)
for stable training without mode collapse.

Graph recovery
──────────────
After sampling a synthetic 18-dim embedding from the trained Generator, we
recover a full graph using the same topology-donor step as GraphSMOTE:
  1. Find the real laundering graph whose mean-pooled features are closest
     to the synthetic embedding.
  2. Shift the donor's per-node features so their column-wise mean matches
     the synthetic embedding.
  3. Return the shifted features + donor topology as a new Data object.

This makes GAN directly comparable to GraphSMOTE — the only difference is how
the target embedding is produced (adversarial vs. k-NN interpolation).

Usage
─────
    from experiments.gan_augmentation import GraphGANAugmenter

    aug = GraphGANAugmenter(latent_dim=32, epochs=200, random_state=42)
    aug.fit(train_laundering_graphs)   # list[Data] with y == 1
    synthetic = aug.generate(n=50)    # list[Data]
    # or:
    synthetic = aug.fit_generate(train_laundering_graphs, n=50)
"""

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data


# ── MLP helpers ────────────────────────────────────────────────────────────────

def _mlp(dims: list[int], act=nn.LeakyReLU, last_act=None) -> nn.Sequential:
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(act(0.2, inplace=True))
            layers.append(nn.LayerNorm(dims[i + 1]))
    if last_act is not None:
        layers.append(last_act())
    return nn.Sequential(*layers)


class _Generator(nn.Module):
    def __init__(self, latent_dim: int, feat_dim: int, hidden: int = 128):
        super().__init__()
        self.net = _mlp([latent_dim, hidden, hidden, feat_dim])

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class _Discriminator(nn.Module):
    def __init__(self, feat_dim: int, hidden: int = 128):
        super().__init__()
        self.net = _mlp([feat_dim, hidden, hidden, 1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── WGAN-GP ────────────────────────────────────────────────────────────────────

def _gradient_penalty(D: nn.Module, real: torch.Tensor, fake: torch.Tensor,
                      device: torch.device, lam: float = 10.0) -> torch.Tensor:
    b = real.size(0)
    alpha = torch.rand(b, 1, device=device)
    interp = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_interp = D(interp)
    grad = torch.autograd.grad(
        outputs=d_interp, inputs=interp,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True, retain_graph=True,
    )[0]
    gp = ((grad.norm(2, dim=1) - 1) ** 2).mean()
    return lam * gp


# ── Main class ─────────────────────────────────────────────────────────────────

class GraphGANAugmenter:
    """
    Conditional GAN augmentation for laundering ego-subgraph oversampling.

    Parameters
    ----------
    latent_dim   : Dimensionality of the Generator's noise input.
    epochs       : Training epochs (each epoch iterates over all laundering graphs).
    batch_size   : Mini-batch size during GAN training.
    lr           : Learning rate (Adam, same for G and D).
    n_critic     : Discriminator updates per Generator update (WGAN-GP default: 5).
    hidden       : Hidden layer width for both G and D.
    random_state : Seed for reproducibility.
    device       : Torch device.  Defaults to CPU if None.
    verbose      : Print training loss every `verbose` epochs (0 = silent).
    """

    def __init__(
        self,
        latent_dim:   int = 32,
        epochs:       int = 300,
        batch_size:   int = 32,
        lr:           float = 1e-4,
        n_critic:     int = 5,
        hidden:       int = 128,
        random_state: int | None = 42,
        device:       torch.device | None = None,
        verbose:      int = 0,
    ):
        self.latent_dim   = latent_dim
        self.epochs       = epochs
        self.batch_size   = batch_size
        self.lr           = lr
        self.n_critic     = n_critic
        self.hidden       = hidden
        self.random_state = random_state
        self.device       = device or torch.device("cpu")
        self.verbose      = verbose
        self._fitted      = False

    # ── fit ──────────────────────────────────────────────────────────────────

    def fit(self, laundering_graphs: list[Data]) -> "GraphGANAugmenter":
        """Train GAN on mean-pooled features of laundering_graphs."""
        graphs = [g for g in laundering_graphs if g.y.item() == 1]
        if len(graphs) < 4:
            raise ValueError(f"Need at least 4 laundering graphs, got {len(graphs)}.")

        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        self._graphs = graphs
        embs = np.stack([g.x.float().mean(dim=0).numpy() for g in graphs])
        feat_dim = embs.shape[1]
        self._embeddings = embs        # [N, feat_dim]  kept for topology recovery

        # Standardise so GAN sees zero-mean unit-variance inputs
        self._mu  = embs.mean(axis=0)
        self._sig = embs.std(axis=0).clip(min=1e-6)
        embs_norm = (embs - self._mu) / self._sig

        real_t = torch.tensor(embs_norm, dtype=torch.float32, device=self.device)

        G = _Generator(self.latent_dim, feat_dim, self.hidden).to(self.device)
        D = _Discriminator(feat_dim, self.hidden).to(self.device)
        opt_G = torch.optim.Adam(G.parameters(), lr=self.lr, betas=(0.0, 0.9))
        opt_D = torch.optim.Adam(D.parameters(), lr=self.lr, betas=(0.0, 0.9))

        n = real_t.size(0)
        for epoch in range(1, self.epochs + 1):
            # ── n_critic Discriminator steps ────────────────────────────────
            for _ in range(self.n_critic):
                idx  = torch.randint(0, n, (min(self.batch_size, n),), device=self.device)
                real = real_t[idx]
                z    = torch.randn(real.size(0), self.latent_dim, device=self.device)
                fake = G(z).detach()

                d_loss = -D(real).mean() + D(fake).mean() + _gradient_penalty(
                    D, real, fake, self.device)
                opt_D.zero_grad(); d_loss.backward(); opt_D.step()

            # ── Generator step ───────────────────────────────────────────────
            z      = torch.randn(self.batch_size, self.latent_dim, device=self.device)
            g_loss = -D(G(z)).mean()
            opt_G.zero_grad(); g_loss.backward(); opt_G.step()

            if self.verbose > 0 and epoch % self.verbose == 0:
                print(f"    [GAN] epoch {epoch}/{self.epochs}  "
                      f"D={d_loss.item():.4f}  G={g_loss.item():.4f}")

        self._G = G
        self._G.eval()
        self._fitted = True
        return self

    # ── generate ─────────────────────────────────────────────────────────────

    def generate(self, n: int) -> list[Data]:
        """Sample n synthetic laundering graphs from the trained GAN."""
        if not self._fitted:
            raise RuntimeError("Call fit() before generate().")

        with torch.no_grad():
            z    = torch.randn(n, self.latent_dim, device=self.device)
            embs = self._G(z).cpu().numpy()

        # Denormalise back to original feature scale
        embs = embs * self._sig + self._mu

        synthetic = []
        for emb in embs:
            # Topology donor: real graph whose mean embedding is closest
            dists = np.linalg.norm(self._embeddings - emb, axis=1)
            donor = self._graphs[int(np.argmin(dists))]

            # Shift donor node features to match the GAN-generated mean
            x_donor  = donor.x.float()
            mu_donor = x_donor.mean(dim=0).numpy()
            shift    = torch.tensor(emb - mu_donor, dtype=torch.float32)
            x_syn    = x_donor + shift.unsqueeze(0)

            synthetic.append(Data(
                x=x_syn,
                edge_index=donor.edge_index.clone(),
                y=torch.tensor([1], dtype=torch.long),
                timestamp_val=-1.0,
                net_idx=-1,
            ))
        return synthetic

    # ── convenience ──────────────────────────────────────────────────────────

    def fit_generate(self, laundering_graphs: list[Data], n: int) -> list[Data]:
        """Fit on laundering_graphs then generate n synthetic samples."""
        return self.fit(laundering_graphs).generate(n)

    def __repr__(self) -> str:
        status = f"fitted on {len(self._graphs)} graphs" if self._fitted else "unfitted"
        return (f"GraphGANAugmenter(latent_dim={self.latent_dim}, "
                f"epochs={self.epochs}, {status})")
