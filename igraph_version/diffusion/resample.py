"""
Loss-aware timestep importance sampler.
Single-GPU version of LossSecondMomentResampler from:
  Nichol & Dhariwal, "Improved Denoising Diffusion Probabilistic Models" (2021).

Usage in training loop:
    sampler = LossSecondMomentResampler(diffusion)
    t, weights = sampler.sample(B, device)
    loss = (loss_dict["loss"] * weights).mean()
    ...
    sampler.update(t.tolist(), loss_dict["loss"].detach().tolist())
"""

import numpy as np
import torch


class LossSecondMomentResampler:
    """
    Sample timesteps proportionally to the RMS loss at each t.
    Timesteps that are consistently hard get sampled more often,
    focusing training budget where it matters most.

    After warm-up (history_per_term observations per t), the sample
    weight for timestep t is proportional to sqrt(E[loss(t)^2]).
    Importance-correction weights are returned alongside timesteps so
    the gradient estimate remains unbiased.
    """

    def __init__(self, diffusion, history_per_term=10, uniform_prob=0.001):
        self.num_timesteps    = diffusion.num_timesteps
        self.history_per_term = history_per_term
        self.uniform_prob     = uniform_prob
        self._loss_history    = np.zeros(
            [diffusion.num_timesteps, history_per_term], dtype=np.float64
        )
        self._loss_counts = np.zeros([diffusion.num_timesteps], dtype=np.int64)

    # ------------------------------------------------------------------

    def weights(self):
        """Return a (T,) weight array.  Uniform until warmed up."""
        if not self._warmed_up():
            return np.ones([self.num_timesteps], dtype=np.float64)
        w  = np.sqrt(np.mean(self._loss_history ** 2, axis=-1))
        w /= w.sum()
        w  = w * (1.0 - self.uniform_prob) + self.uniform_prob / self.num_timesteps
        return w

    def sample(self, batch_size, device):
        """
        Draw `batch_size` timesteps.

        Returns
        -------
        t       : LongTensor [B]   — sampled timestep indices
        weights : FloatTensor [B]  — importance-correction weights
                  (multiply loss by these before calling .mean())
        """
        w   = self.weights()
        p   = w / w.sum()
        idx = np.random.choice(self.num_timesteps, size=(batch_size,), p=p)
        iw  = 1.0 / (self.num_timesteps * p[idx])
        t   = torch.from_numpy(idx).long().to(device)
        iw  = torch.from_numpy(iw.astype(np.float32)).to(device)
        return t, iw

    def update(self, ts, losses):
        """
        Record per-timestep losses to refine future sampling weights.

        Parameters
        ----------
        ts     : list[int]   — timestep indices from the last batch
        losses : list[float] — corresponding scalar losses
        """
        for t, loss in zip(ts, losses):
            if self._loss_counts[t] == self.history_per_term:
                self._loss_history[t, :-1] = self._loss_history[t, 1:]
                self._loss_history[t, -1]  = loss
            else:
                self._loss_history[t, self._loss_counts[t]] = loss
                self._loss_counts[t] += 1

    def _warmed_up(self):
        return (self._loss_counts == self.history_per_term).all()
