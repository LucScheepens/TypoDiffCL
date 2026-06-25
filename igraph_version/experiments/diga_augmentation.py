"""
diga_augmentation.py
────────────────────
DiGa-style diffusion augmentation for laundering ego-subgraph oversampling.

This implements a simple unconditional DDPM (Ho et al., 2020) in the
mean-pooled node-feature space (18-dim), serving as a diffusion-based
baseline comparable to GAN, VAE, and GraphSMOTE.

Relationship to our full method
────────────────────────────────
Our method (diffusion + SimCLR) operates on the full (node-feature, adjacency)
pair at the graph level with class-conditional guided reverse diffusion.
DiGa operates only in the 18-dim mean-pooled feature space — no graph topology
is modelled, no SimCLR classifier guides the reverse process.  This isolates
the contribution of (a) graph-level joint generation and (b) SimCLR guidance.

Architecture
────────────
Denoiser: MLP(noisy_x ∥ sinusoidal_time_emb → residual correction)
Schedule: linear β from β_start=1e-4 to β_end=0.02 over T=300 steps
          (shorter schedule than image DDPM; feature space is much simpler)

Graph recovery  (same as GAN / VAE / GraphSMOTE)
────────────────
After sampling a synthetic 18-dim embedding from the trained DDPM:
  1. Find the real laundering graph whose mean-pooled features are closest
     to the synthetic embedding.
  2. Shift the donor's per-node features so their column-wise mean matches
     the synthetic embedding.
  3. Return shifted features + donor topology as a new Data object.

Usage
─────
    from experiments.diga_augmentation import DiGaAugmenter

    aug = DiGaAugmenter(epochs=300, random_state=42)
    aug.fit(train_laundering_graphs)   # list[Data] with y == 1
    synthetic = aug.generate(n=50)    # list[Data]
    # or:
    synthetic = aug.fit_generate(train_laundering_graphs, n=50)
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data


# ── Sinusoidal time embedding (standard DDPM) ─────────────────────────────────

def _sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Sinusoidal position embedding for diffusion timesteps.
    t : [B] integer timesteps
    Returns [B, dim]
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    args = t[:, None].float() * freqs[None]
    emb  = torch.cat([args.sin(), args.cos()], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


# ── MLP denoiser ──────────────────────────────────────────────────────────────

class _Denoiser(nn.Module):
    """
    Simple residual MLP that predicts the noise ε from (noisy_x, t).
    """
    def __init__(self, feat_dim: int, time_dim: int = 32, hidden: int = 128):
        super().__init__()
        self.time_proj = nn.Sequential(
            nn.Linear(time_dim, hidden), nn.SiLU(),
        )
        self.net = nn.Sequential(
            nn.Linear(feat_dim + hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden),            nn.SiLU(),
            nn.Linear(hidden, hidden),            nn.SiLU(),
            nn.Linear(hidden, feat_dim),
        )
        self.time_dim = time_dim

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = _sinusoidal_embedding(t, self.time_dim)
        t_h   = self.time_proj(t_emb)           # [B, hidden]
        inp   = torch.cat([x, t_h], dim=-1)     # [B, feat_dim + hidden]
        return self.net(inp)                    # [B, feat_dim]


# ── DDPM noise schedule ───────────────────────────────────────────────────────

def _make_schedule(T: int, beta_start: float, beta_end: float, device: torch.device):
    betas      = torch.linspace(beta_start, beta_end, T, dtype=torch.float32, device=device)
    alphas     = 1.0 - betas
    alphas_bar = torch.cumprod(alphas, dim=0)
    return betas, alphas, alphas_bar


# ── Main class ─────────────────────────────────────────────────────────────────

class DiGaAugmenter:
    """
    DiGa-style DDPM augmentation for laundering ego-subgraph oversampling.

    Parameters
    ----------
    T            : Diffusion timesteps (default 300 — shorter than image DDPM;
                   18-dim feature space is much simpler than 256×256 images).
    beta_start   : Starting noise variance (default 1e-4).
    beta_end     : Ending noise variance (default 0.02).
    epochs       : Training epochs.
    batch_size   : Mini-batch size during training.
    lr           : Adam learning rate.
    hidden       : Hidden width of the MLP denoiser.
    time_dim     : Dimensionality of sinusoidal time embedding.
    random_state : Seed for reproducibility.
    device       : Torch device.  Defaults to CPU if None.
    verbose      : Print loss every `verbose` epochs (0 = silent).
    """

    def __init__(
        self,
        T:            int   = 300,
        beta_start:   float = 1e-4,
        beta_end:     float = 0.02,
        epochs:       int   = 400,
        batch_size:   int   = 32,
        lr:           float = 1e-3,
        hidden:       int   = 128,
        time_dim:     int   = 32,
        random_state: int | None = 42,
        device:       torch.device | None = None,
        verbose:      int   = 0,
    ):
        self.T            = T
        self.beta_start   = beta_start
        self.beta_end     = beta_end
        self.epochs       = epochs
        self.batch_size   = batch_size
        self.lr           = lr
        self.hidden       = hidden
        self.time_dim     = time_dim
        self.random_state = random_state
        self.device       = device or torch.device("cpu")
        self.verbose      = verbose
        self._fitted      = False

    # ── fit ──────────────────────────────────────────────────────────────────

    def fit(self, laundering_graphs: list[Data]) -> "DiGaAugmenter":
        """Train the DDPM denoiser on mean-pooled features of laundering_graphs."""
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

        # Standardise so the DDPM sees zero-mean unit-variance inputs
        self._mu_data  = embs.mean(axis=0)
        self._sig_data = embs.std(axis=0).clip(min=1e-6)
        embs_norm = (embs - self._mu_data) / self._sig_data

        x0_all = torch.tensor(embs_norm, dtype=torch.float32, device=self.device)
        n      = x0_all.size(0)

        betas, alphas, alphas_bar = _make_schedule(
            self.T, self.beta_start, self.beta_end, self.device)
        self._betas      = betas
        self._alphas     = alphas
        self._alphas_bar = alphas_bar

        denoiser = _Denoiser(feat_dim, self.time_dim, self.hidden).to(self.device)
        opt      = torch.optim.Adam(denoiser.parameters(), lr=self.lr)

        for epoch in range(1, self.epochs + 1):
            idx  = torch.randint(0, n, (min(self.batch_size, n),), device=self.device)
            x0   = x0_all[idx]                                     # [B, feat_dim]
            t    = torch.randint(0, self.T, (x0.size(0),), device=self.device)
            eps  = torch.randn_like(x0)

            ab_t = alphas_bar[t].unsqueeze(-1)                     # [B, 1]
            xt   = x0 * ab_t.sqrt() + eps * (1 - ab_t).sqrt()     # forward diffusion

            eps_pred = denoiser(xt, t)
            loss     = F.mse_loss(eps_pred, eps)

            opt.zero_grad(); loss.backward(); opt.step()

            if self.verbose > 0 and epoch % self.verbose == 0:
                print(f"    [DiGa] epoch {epoch}/{self.epochs}  loss={loss.item():.5f}")

        self._denoiser = denoiser
        self._denoiser.eval()
        self._feat_dim = feat_dim
        self._fitted   = True
        return self

    # ── generate ─────────────────────────────────────────────────────────────

    def generate(self, n: int) -> list[Data]:
        """
        Sample n synthetic laundering graphs using the DDPM reverse process.

        DDPM reverse:
          x_{t-1} = 1/√α_t * (x_t - β_t/√(1-ᾱ_t) * ε_θ(x_t, t)) + σ_t * z
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before generate().")

        betas      = self._betas
        alphas     = self._alphas
        alphas_bar = self._alphas_bar

        with torch.no_grad():
            x = torch.randn(n, self._feat_dim, device=self.device)

            for t_idx in reversed(range(self.T)):
                t_batch = torch.full((n,), t_idx, device=self.device, dtype=torch.long)
                eps_pred = self._denoiser(x, t_batch)

                beta_t  = betas[t_idx]
                alpha_t = alphas[t_idx]
                ab_t    = alphas_bar[t_idx]

                # Predicted mean
                coef = beta_t / (1 - ab_t).sqrt()
                mu   = (x - coef * eps_pred) / alpha_t.sqrt()

                if t_idx > 0:
                    # Posterior variance: β_t * (1 - ᾱ_{t-1}) / (1 - ᾱ_t)
                    ab_prev = alphas_bar[t_idx - 1]
                    sigma   = ((beta_t * (1 - ab_prev) / (1 - ab_t)).sqrt())
                    x = mu + sigma * torch.randn_like(mu)
                else:
                    x = mu

            embs = x.cpu().numpy()

        # Denormalise
        embs = embs * self._sig_data + self._mu_data

        synthetic = []
        for emb in embs:
            # Topology donor: nearest real laundering graph (same as GAN / VAE)
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
        return (f"DiGaAugmenter(T={self.T}, epochs={self.epochs}, {status})")
