"""
vae_augmentation.py
───────────────────
Variational Autoencoder (VAE) augmentation for laundering ego-subgraph
oversampling.

Architecture
────────────
The VAE operates in the mean-pooled node-feature space (18-dim), identical to
GraphSMOTE and GraphGAN so all three methods are directly comparable.

  Encoder : x (18-dim)  →  µ, log σ²  (latent_dim each)
  Decoder : z (latent_dim) →  x̂ (18-dim)

Training objective: reconstruction MSE + β * KL(q(z|x) || N(0,I))
  β < 1 encourages smoother latent geometry (β-VAE, Higgins et al. 2017)
  which improves generation quality when the training set is small.

Graph recovery
──────────────
After sampling z ~ N(µ, σ) or z ~ N(0, I) and decoding to a synthetic
18-dim embedding, topology recovery follows the same donor step as
GraphSMOTE and GraphGAN:
  1. Find the real laundering graph whose mean features are closest to the
     synthetic embedding.
  2. Shift the donor's per-node features to match the decoded embedding.
  3. Return shifted features + donor edge_index as a new Data object.

This makes VAE directly comparable to GAN and SMOTE — the generation
mechanism differs but the graph-recovery step is identical.

Usage
─────
    from experiments.vae_augmentation import GraphVAEAugmenter

    aug = GraphVAEAugmenter(latent_dim=16, epochs=300, random_state=42)
    aug.fit(train_laundering_graphs)   # list[Data] with y == 1
    synthetic = aug.generate(n=50)    # list[Data]
    # or:
    synthetic = aug.fit_generate(train_laundering_graphs, n=50)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data


# ── VAE modules ────────────────────────────────────────────────────────────────

class _Encoder(nn.Module):
    def __init__(self, feat_dim: int, latent_dim: int, hidden: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
        )
        self.mu_head     = nn.Linear(hidden, latent_dim)
        self.logvar_head = nn.Linear(hidden, latent_dim)

    def forward(self, x: torch.Tensor):
        h      = self.shared(x)
        mu     = self.mu_head(h)
        logvar = self.logvar_head(h).clamp(-10, 10)
        return mu, logvar


class _Decoder(nn.Module):
    def __init__(self, latent_dim: int, feat_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),     nn.ReLU(),
            nn.Linear(hidden, feat_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


# ── Main class ─────────────────────────────────────────────────────────────────

class GraphVAEAugmenter:
    """
    VAE-based augmentation for laundering ego-subgraph oversampling.

    Parameters
    ----------
    latent_dim   : Dimensionality of the VAE latent space.
    epochs       : Training epochs.
    batch_size   : Mini-batch size during training.
    lr           : Adam learning rate.
    beta         : KL weight (β < 1 = β-VAE for smoother latents; default 0.5).
    hidden       : Hidden layer width for encoder and decoder.
    random_state : Seed for reproducibility.
    device       : Torch device.  Defaults to CPU if None.
    verbose      : Print loss every `verbose` epochs (0 = silent).
    """

    def __init__(
        self,
        latent_dim:   int   = 16,
        epochs:       int   = 300,
        batch_size:   int   = 64,
        lr:           float = 1e-3,
        beta:         float = 0.5,
        hidden:       int   = 128,
        random_state: int | None = 42,
        device:       torch.device | None = None,
        verbose:      int   = 0,
    ):
        self.latent_dim   = latent_dim
        self.epochs       = epochs
        self.batch_size   = batch_size
        self.lr           = lr
        self.beta         = beta
        self.hidden       = hidden
        self.random_state = random_state
        self.device       = device or torch.device("cpu")
        self.verbose      = verbose
        self._fitted      = False

    # ── fit ──────────────────────────────────────────────────────────────────

    def fit(self, laundering_graphs: list[Data]) -> "GraphVAEAugmenter":
        """Train VAE on mean-pooled features of laundering_graphs."""
        graphs = [g for g in laundering_graphs if g.y.item() == 1]
        if len(graphs) < 4:
            raise ValueError(f"Need at least 4 laundering graphs, got {len(graphs)}.")

        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        self._graphs = graphs
        embs = np.stack([g.x.float().mean(dim=0).numpy() for g in graphs])
        feat_dim = embs.shape[1]
        self._embeddings = embs

        # Standardise
        self._mu_data  = embs.mean(axis=0)
        self._sig_data = embs.std(axis=0).clip(min=1e-6)
        embs_norm = (embs - self._mu_data) / self._sig_data

        data_t = torch.tensor(embs_norm, dtype=torch.float32, device=self.device)
        n = data_t.size(0)

        enc = _Encoder(feat_dim, self.latent_dim, self.hidden).to(self.device)
        dec = _Decoder(self.latent_dim, feat_dim, self.hidden).to(self.device)
        opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()),
                               lr=self.lr)

        for epoch in range(1, self.epochs + 1):
            idx   = torch.randperm(n, device=self.device)[:self.batch_size]
            x_b   = data_t[idx]

            mu, logvar = enc(x_b)
            std = (0.5 * logvar).exp()
            z   = mu + std * torch.randn_like(std)

            x_hat = dec(z)
            recon = F.mse_loss(x_hat, x_b)
            kl    = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
            loss  = recon + self.beta * kl

            opt.zero_grad(); loss.backward(); opt.step()

            if self.verbose > 0 and epoch % self.verbose == 0:
                print(f"    [VAE] epoch {epoch}/{self.epochs}  "
                      f"recon={recon.item():.4f}  kl={kl.item():.4f}")

        self._dec = dec
        self._dec.eval()
        self._fitted = True
        return self

    # ── generate ─────────────────────────────────────────────────────────────

    def generate(self, n: int) -> list[Data]:
        """Sample n synthetic laundering graphs from the trained VAE prior."""
        if not self._fitted:
            raise RuntimeError("Call fit() before generate().")

        with torch.no_grad():
            z    = torch.randn(n, self.latent_dim, device=self.device)
            embs = self._dec(z).cpu().numpy()

        # Denormalise
        embs = embs * self._sig_data + self._mu_data

        synthetic = []
        for emb in embs:
            # Topology donor: nearest real laundering graph
            dists = np.linalg.norm(self._embeddings - emb, axis=1)
            donor = self._graphs[int(np.argmin(dists))]

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
        return (f"GraphVAEAugmenter(latent_dim={self.latent_dim}, "
                f"beta={self.beta}, epochs={self.epochs}, {status})")
