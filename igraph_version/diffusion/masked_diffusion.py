import enum
import math
import numpy as np
import torch as th

def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.

    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, th.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for th.exp().
    logvar1, logvar2 = [
        x if isinstance(x, th.Tensor) else th.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + th.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * th.exp(-logvar2)
    )

def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (1.0 + th.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * th.pow(x, 3))))

def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.

    :param x: the target images. It is assumed that this was uint8 values,
              rescaled to the range [-1, 1].
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = th.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = th.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = th.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = th.where(
        x < -0.999,
        log_cdf_plus,
        th.where(x > 0.999, log_one_minus_cdf_min, th.log(cdf_delta.clamp(min=1e-12))),
    )
    assert log_probs.shape == x.shape
    return log_probs



def _extract_into_tensor(arr, timesteps, broadcast_shape):

    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()

    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]

    return res.expand(broadcast_shape)



class ModelMeanType(enum.Enum):
    PREVIOUS_X = enum.auto()
    START_X = enum.auto()
    EPSILON = enum.auto()


class ModelVarType(enum.Enum):
    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()
    RESCALED_MSE = enum.auto()
    KL = enum.auto()
    RESCALED_KL = enum.auto()

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


# Masked Gaussian Diffusion

class GaussianDiffusion:

    def __init__(
        self,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        rescale_timesteps=False,
    ):

        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps

        betas = np.array(betas, dtype=np.float64)
        self.betas = betas

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas

        self.alphas_cumprod      = np.cumprod(alphas)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)  # for DDIM reverse

        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1 - self.alphas_cumprod)

        self.sqrt_recip_alphas_cumprod = np.sqrt(1 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1 / self.alphas_cumprod - 1)

        self.posterior_variance = (
            betas
            * (1 - self.alphas_cumprod_prev)
            / (1 - self.alphas_cumprod)
        )

        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )

        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev)
            / (1 - self.alphas_cumprod)
        )

        self.posterior_mean_coef2 = (
            (1 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1 - self.alphas_cumprod)
        )


    # Forward Diffusion (Masked)

    def q_sample(self, x_start, t, noise=None, node_mask=None, adj_start=None):

        if noise is None:
            noise = th.randn_like(x_start)

        if node_mask is not None:

            mask = node_mask.unsqueeze(-1)

            x_start = x_start * mask
            noise = noise * mask

        # Gaussian forward process for continuous features (indices 1+)
        x_t_cont = (
            _extract_into_tensor(
                self.sqrt_alphas_cumprod, t, x_start[..., 1:].shape
            ) * x_start[..., 1:]
            +
            _extract_into_tensor(
                self.sqrt_one_minus_alphas_cumprod, t, x_start[..., 1:].shape
            ) * noise[..., 1:]
        )

        # Bernoulli forward process for binary laundering feature (index 0):
        # with probability alpha_bar_t keep the original bit, else draw uniform {0,1}
        keep_prob = _extract_into_tensor(
            self.alphas_cumprod, t, x_start[..., 0:1].shape
        ).clamp(0.0, 1.0)
        keep     = th.bernoulli(keep_prob)
        rand_bit = th.bernoulli(th.full_like(x_start[..., 0:1], 0.5))
        x_t_bin  = keep * x_start[..., 0:1] + (1 - keep) * rand_bit

        x_t = th.cat([x_t_bin, x_t_cont], dim=-1)

        if adj_start is None:
            return x_t

        # Bernoulli forward process for adjacency (binary edges).
        # Stationary distribution uses the actual batch edge density rather than
        # 0.5: if real graphs are ~5% dense, converging toward 50% noise forces the
        # model to reconstruct sparse from very dense inputs, which biases it toward
        # over-predicting edges. Using the true density as the noise floor avoids this.
        keep_prob_adj = _extract_into_tensor(
            self.alphas_cumprod, t, adj_start.shape
        ).clamp(0.0, 1.0)
        keep_adj = th.bernoulli(keep_prob_adj)
        with th.no_grad():
            if node_mask is not None:
                mask2d_s = node_mask[:, :, None] * node_mask[:, None, :]
                n_act = mask2d_s.sum().clamp(min=1)
                p_edge = (adj_start * mask2d_s).sum() / n_act
            else:
                p_edge = adj_start.mean()
            p_edge = p_edge.clamp(0.01, 0.5)
        rand_adj  = th.bernoulli(th.full_like(adj_start, p_edge.item()))
        adj_t = keep_adj * adj_start + (1 - keep_adj) * rand_adj

        if node_mask is not None:
            adj_t = adj_t * node_mask[:, :, None] * node_mask[:, None, :]

        return x_t, adj_t


    # Reverse Prediction

    def _predict_xstart_from_eps(self, x_t, t, eps):

        return (
            _extract_into_tensor(
                self.sqrt_recip_alphas_cumprod, t, x_t.shape
            ) * x_t
            -
            _extract_into_tensor(
                self.sqrt_recipm1_alphas_cumprod, t, x_t.shape
            ) * eps
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        """Inverse of _predict_xstart_from_eps — re-derives eps given x_0 prediction."""
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(
            self.sqrt_recipm1_alphas_cumprod, t, x_t.shape
        ).clamp(min=1e-8)


    def _scale_timesteps(self, t):

        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)

        return t


    def p_mean_variance(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        model_kwargs=None,
        clip_bounds=None,
    ):

        if model_kwargs is None:
            model_kwargs = {}

        node_mask = model_kwargs.get("node_mask", None)

        if node_mask is not None:
            x = x * node_mask.unsqueeze(-1)

        model_output_x, adj_pred, node_logits = model(
            x,
            self._scale_timesteps(t),
            **model_kwargs
        )

        if self.model_mean_type == ModelMeanType.EPSILON:

            # Feature 0: model predicts x_start directly (binary, clamp to [0,1])
            pred_xstart_bin  = model_output_x[..., 0:1].clamp(0.0, 1.0)
            # Features 1+: model predicts epsilon, recover x_start via standard formula
            pred_xstart_cont = self._predict_xstart_from_eps(
                x[..., 1:], t, model_output_x[..., 1:]
            )
            pred_xstart = th.cat([pred_xstart_bin, pred_xstart_cont], dim=-1)

        elif self.model_mean_type == ModelMeanType.START_X:

            pred_xstart = model_output_x

        else:
            raise NotImplementedError()


        if clip_denoised:
            # Feature 0: binary, always clamp to [0,1].
            # Features 1+: clamp to the valid normalised range derived from the
            # original feature bounds.  Without this, small eps-prediction errors
            # compound over 500 steps and diverge — especially for features with
            # tiny std (PageRank ~0.005, Assortativity near-zero), where a unit
            # normalised error maps to hundreds in original scale.
            cont = pred_xstart[..., 1:]
            if clip_bounds is not None:
                lo, hi = clip_bounds   # each [F-1], broadcastable to [B, N, F-1]
                cont = cont.clamp(lo, hi)
            pred_xstart = th.cat([pred_xstart[..., 0:1].clamp(0.0, 1.0), cont], dim=-1)


        if node_mask is not None:
            pred_xstart = pred_xstart * node_mask.unsqueeze(-1)


        model_mean = (
            _extract_into_tensor(
                self.posterior_mean_coef1, t, x.shape
            ) * pred_xstart
            +
            _extract_into_tensor(
                self.posterior_mean_coef2, t, x.shape
            ) * x
        )


        model_variance = _extract_into_tensor(
            self.posterior_variance, t, x.shape
        )

        model_log_variance = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x.shape
        )


        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
            "adj_pred": adj_pred,
            "node_logits": node_logits,
        }



    def p_sample(
        self,
        model,
        x,
        t,
        model_kwargs=None,
        adj_gamma=1.0,
        clip_bounds=None,
    ):
        """
        adj_gamma : exponent applied to adj_pred before Bernoulli sampling.
            Values > 1 squash mid-range probabilities toward 0 (sparse bias).
            Default 1.0 = no effect.  Set to ~2.0 in the generation loop to
            counteract the decoder's tendency to predict near-uniform mid values.
        """

        if model_kwargs is None:
            model_kwargs = {}

        node_mask = model_kwargs.get("node_mask", None)

        out = self.p_mean_variance(
            model,
            x,
            t,
            model_kwargs=model_kwargs,
            clip_bounds=clip_bounds,
        )


        noise = th.randn_like(x)

        if node_mask is not None:

            mask = node_mask.unsqueeze(-1)

            noise = noise * mask


        nonzero_mask = (
            (t != 0).float()
            .view(-1, *([1] * (len(x.shape) - 1)))
        )


        sample = (
            out["mean"]
            + nonzero_mask
            * th.exp(0.5 * out["log_variance"])
            * noise
        )


        if node_mask is not None:
            sample = sample * mask


        # Adjacency reverse step: predict-x0-and-resample (Bernoulli).
        # Gamma compression pushes mid-range adj_pred values toward 0, preventing
        # the positive-feedback loop where dense initial adj → similar node
        # embeddings → high dot-products → even denser adj_pred → fully connected.
        adj_pred = out["adj_pred"]
        if adj_gamma != 1.0:
            adj_pred = adj_pred.clamp(0.0, 1.0) ** adj_gamma
        nonzero_adj = (t != 0).float().view(-1, 1, 1)
        adj_stoch   = th.bernoulli(adj_pred.clamp(0.0, 1.0))
        adj_determ  = (adj_pred > 0.5).float()
        adj_sample  = th.where(nonzero_adj.bool(), adj_stoch, adj_determ)

        if node_mask is not None:
            adj_sample = adj_sample * node_mask[:, :, None] * node_mask[:, None, :]

        return {
            "sample": sample,
            "pred_xstart": out["pred_xstart"],
            "adj_sample": adj_sample,
        }


    # ========================================================
    # Sampling Loop
    # ========================================================

    def p_sample_loop(
        self,
        model,
        shape,
        adj_shape=None,
        model_kwargs=None,
        device=None,
        adj_init_p=None,
        adj_gamma=2.0,
        clip_bounds=None,
    ):
        """
        shape      : (B, N, F) for node features
        adj_shape  : (B, N, N) — pass to generate adjacency from noise.
                     If None, adj must be supplied in model_kwargs.
        adj_init_p : initial Bernoulli probability for adj noise.
                     Should match the training stationary distribution — i.e.
                     the true edge density of the dataset.  Defaults to 0.5 if
                     not provided, but passing the real density (e.g. 0.05-0.15)
                     avoids the dense-initialisation feedback loop.
        adj_gamma  : exponent applied to adj_pred at each denoising step.
                     Values > 1 squash mid-range predictions toward 0 (sparse
                     bias).  Default 2.0.
        """

        if device is None:
            device = next(model.parameters()).device

        if model_kwargs is None:
            model_kwargs = {}

        x = th.randn(*shape, device=device)

        # Initialise adjacency.  Using the true edge density instead of 0.5
        # avoids the positive-feedback loop: dense init → mixed embeddings →
        # high dot-products → denser adj_pred → fully connected generation.
        adj = None
        if adj_shape is not None:
            init_p = float(adj_init_p) if adj_init_p is not None else 0.5
            adj = th.bernoulli(th.full(adj_shape, init_p, device=device))
            node_mask = model_kwargs.get("node_mask")
            if node_mask is not None:
                adj = adj * node_mask[:, :, None] * node_mask[:, None, :]

        for i in reversed(range(self.num_timesteps)):

            t = th.tensor(
                [i] * shape[0],
                device=device
            )

            current_kwargs = {**model_kwargs}
            if adj is not None:
                current_kwargs["adj"] = adj

            with th.no_grad():

                out = self.p_sample(
                    model,
                    x,
                    t,
                    model_kwargs=current_kwargs,
                    adj_gamma=adj_gamma,
                    clip_bounds=clip_bounds,
                )

                x = out["sample"]
                if adj is not None:
                    adj = out["adj_sample"]

        if adj is not None:
            return x, adj
        return x


    # ========================================================
    # DDIM Sampling Loop
    # ========================================================

    def ddim_sample_loop(
        self,
        model,
        shape,
        adj_shape=None,
        model_kwargs=None,
        device=None,
        adj_init_p=None,
        adj_gamma=2.0,
        ddim_steps=50,
        eta=0.0,
        clip_bounds=None,
    ):
        """
        Generate a sample using DDIM (Song et al. 2020) with `ddim_steps` ≤ T.

        DDIM replaces the stochastic ancestral sampler with a (near-)deterministic
        ODE, allowing generation in far fewer steps than T without retraining.

        Parameters
        ----------
        ddim_steps : number of denoising steps (≤ T).  50 gives ~10× speedup
                     over the full 500-step ancestral chain with minimal quality loss.
        eta        : stochasticity level.  0.0 = fully deterministic DDIM;
                     1.0 = DDPM-equivalent noise at each step.
        adj_init_p : initial Bernoulli density for adj noise (should equal the
                     training dataset's edge density, e.g. 0.10).
        adj_gamma  : gamma compression applied to adj_pred before sampling.

        Note: DDIM applies to the continuous node features only.  The adjacency
        keeps the Bernoulli predict-x0-and-resample step (DDIM is for Gaussians).
        The binary laundering feature (index 0) also uses direct prediction.
        """
        if device is None:
            device = next(model.parameters()).device
        if model_kwargs is None:
            model_kwargs = {}

        # Select ddim_steps evenly-spaced timesteps in [0, T-1] and reverse them
        indices = list(np.linspace(0, self.num_timesteps - 1, ddim_steps, dtype=int))
        timestep_seq = list(reversed(indices))  # descending: T-1, ..., 0

        x = th.randn(*shape, device=device)

        adj = None
        if adj_shape is not None:
            init_p = float(adj_init_p) if adj_init_p is not None else 0.5
            adj    = th.bernoulli(th.full(adj_shape, init_p, device=device))
            node_mask = model_kwargs.get("node_mask")
            if node_mask is not None:
                adj = adj * node_mask[:, :, None] * node_mask[:, None, :]

        for step_idx, t_curr in enumerate(timestep_seq):
            t_prev = timestep_seq[step_idx + 1] if step_idx + 1 < len(timestep_seq) else -1

            t_vec = th.tensor([t_curr] * shape[0], device=device)

            current_kwargs = {**model_kwargs}
            if adj is not None:
                current_kwargs["adj"] = adj

            node_mask = current_kwargs.get("node_mask")

            with th.no_grad():
                model_out, adj_pred, _ = model(
                    x * (node_mask.unsqueeze(-1) if node_mask is not None else 1),
                    self._scale_timesteps(t_vec),
                    **current_kwargs,
                )

                # Predict x0 for continuous features from eps prediction
                pred_xstart_cont = self._predict_xstart_from_eps(
                    x[..., 1:], t_vec, model_out[..., 1:]
                )
                if clip_bounds is not None:
                    lo, hi = clip_bounds
                    pred_xstart_cont = pred_xstart_cont.clamp(lo, hi)
                pred_xstart_bin  = model_out[..., 0:1].clamp(0.0, 1.0)
                pred_xstart      = th.cat([pred_xstart_bin, pred_xstart_cont], dim=-1)
                if node_mask is not None:
                    pred_xstart = pred_xstart * node_mask.unsqueeze(-1)

                # Re-derive eps from pred_xstart (more stable than raw model eps)
                eps = self._predict_eps_from_xstart(
                    x[..., 1:], t_vec, pred_xstart[..., 1:]
                )

                # DDIM update coefficients
                ab_t    = float(self.alphas_cumprod[t_curr])
                ab_prev = float(self.alphas_cumprod[t_prev]) if t_prev >= 0 else 1.0

                sigma = (
                    eta
                    * math.sqrt((1.0 - ab_prev) / max(1.0 - ab_t, 1e-8))
                    * math.sqrt(max(1.0 - ab_t / max(ab_prev, 1e-8), 0.0))
                )
                dir_coef  = math.sqrt(max(1.0 - ab_prev - sigma ** 2, 0.0))

                # Continuous features: DDIM mean + optional noise
                mean_cont = (
                    pred_xstart[..., 1:] * math.sqrt(ab_prev)
                    + dir_coef * eps
                )
                noise = th.randn_like(x[..., 1:]) if t_prev >= 0 else th.zeros_like(x[..., 1:])
                x_cont = mean_cont + sigma * noise

                # Binary laundering feature: use direct prediction (no DDIM)
                x_bin = pred_xstart[..., 0:1]

                x = th.cat([x_bin, x_cont], dim=-1)
                if node_mask is not None:
                    x = x * node_mask.unsqueeze(-1)

                # Adjacency: Bernoulli predict-x0-and-resample (same as p_sample)
                if adj is not None:
                    ap = adj_pred.clamp(0.0, 1.0) ** adj_gamma
                    if t_prev >= 0:
                        adj = th.bernoulli(ap)
                    else:
                        adj = (ap > 0.5).float()
                    if node_mask is not None:
                        adj = adj * node_mask[:, :, None] * node_mask[:, None, :]

        if adj is not None:
            return x, adj
        return x


    def training_losses(
        self,
        model,
        x_start,
        t,
        model_kwargs=None,
        noise=None,
        adj_start=None,
        adj_loss_weight=0.5,
        density_loss_weight=1.0,
        node_exist_loss_weight=1.0,
        mask_dropout_rate=0.4,
        laund_loss_weight=1.0,
        ghost_node_rate=0.3,
        degree_seq_loss_weight=0.5,
        # ── Direction 1: feature-topology consistency ──────────────────────
        # Penalises divergence between structural features implied by adj_pred
        # (degree, clustering) and the ground-truth structural features in x_start.
        # Gradients flow through adj_pred → adjacency decoder, pushing it to
        # produce adjacencies whose structure matches the real data.
        consistency_loss_weight=0.0,   # set > 0 to enable (e.g. 0.2)
        degree_feat_col=1,             # column index of degree in x_start
        clust_feat_col=3,              # column index of clustering in x_start
        x_mean_feat=None,              # [F] normalization mean (required if weight > 0)
        x_std_feat=None,               # [F] normalization std  (required if weight > 0)
        # ── Direction 2: topology-aware adj BCE ────────────────────────────
        # Raising this cap from 2.0 lets the model upweight rare edges more
        # aggressively in very sparse graphs without switching to focal loss.
        adj_pos_weight_max=2.0,        # set to e.g. 10.0 for sparser datasets
        # ── Edge dropout ───────────────────────────────────────────────────
        # Randomly zero a fraction of adj_t edges before the model forward pass.
        # This prevents oversmoothing: when adj_t is always fully observed the GNN
        # aggregates over all neighbours, collapsing embeddings and causing adj_pred
        # to be uniformly high for all pairs.
        edge_drop_rate=0.0,            # set to e.g. 0.15 to enable
    ):

        if model_kwargs is None:
            model_kwargs = {}

        node_mask = model_kwargs.get("node_mask", None)

        if noise is None:
            noise = th.randn_like(x_start)

        # Corrupt node features (and adj when provided)
        if adj_start is not None:
            x_t, adj_t = self.q_sample(
                x_start, t, noise=noise, node_mask=node_mask, adj_start=adj_start
            )
        else:
            x_t = self.q_sample(x_start, t, noise=noise, node_mask=node_mask)
            adj_t = None

        # Node-mask corruption: randomly deactivate active nodes proportional to t.
        # The model must learn to reconstruct the original mask — i.e., decide which
        # nodes should exist based purely on local structural context.
        if node_mask is not None:
            t_frac = (t.float() / max(self.num_timesteps - 1, 1)).unsqueeze(-1)  # [B,1]
            drop_prob     = t_frac * mask_dropout_rate                            # [B,1]
            keep          = (th.rand_like(node_mask) >= drop_prob).float()
            node_mask_t   = node_mask * keep                                      # [B,N]
            # Zero out features / adj rows+cols for dropped nodes
            x_t   = x_t   * node_mask_t.unsqueeze(-1)
            if adj_t is not None:
                adj_t = adj_t * node_mask_t[:, :, None] * node_mask_t[:, None, :]

            # Ghost node injection: activate padding slots with random features and edges.
            # This noise looks like "adding new nodes" to the graph, and the model must
            # learn to mark them as non-existent (node_logits < 0) while also ignoring
            # their spurious edges. It provides a strong, direct training signal that
            # teaches sparsity: the target adjacency is 0 for all ghost-involved pairs.
            if ghost_node_rate > 0.0:
                inactive = 1.0 - node_mask                           # [B, N] padding slots
                ghost_prob = (t_frac * ghost_node_rate).clamp(0.0, 0.8)
                ghost_added = inactive * th.bernoulli(
                    th.ones_like(node_mask) * ghost_prob
                )                                                     # [B, N]

                if ghost_added.any():
                    # Random Gaussian features for ghost nodes; real-node positions
                    # have ghost_added=0 so they are unaffected.
                    ghost_feats = th.randn_like(x_t) * ghost_added.unsqueeze(-1)
                    x_t = x_t + ghost_feats

                    # Activate ghost nodes in the corrupted mask so the model sees them
                    node_mask_t = (node_mask_t + ghost_added).clamp(0.0, 1.0)

                    # Random edges between ghost nodes and all active (real+ghost) nodes
                    if adj_t is not None:
                        g  = ghost_added                            # [B, N]
                        am = node_mask_t                            # [B, N]
                        # edge mask: at least one endpoint is a ghost node
                        ghost_edge_mask = (
                            g[:, :, None] * am[:, None, :]
                            + am[:, :, None] * g[:, None, :]
                        ).clamp(0.0, 1.0)
                        rand_ghost_adj = (
                            th.bernoulli(th.full_like(adj_t, 0.5)) * ghost_edge_mask
                        )
                        adj_t = (adj_t + rand_ghost_adj).clamp(0.0, 1.0)
        else:
            node_mask_t = None

        # Forward pass with the corrupted mask
        training_kwargs = {**model_kwargs, "node_mask": node_mask_t}
        if adj_t is not None:
            adj_t_in = adj_t
            if edge_drop_rate > 0.0 and model.training:
                drop = th.bernoulli(th.full_like(adj_t, 1.0 - edge_drop_rate))
                adj_t_in = adj_t * drop
            training_kwargs["adj"] = adj_t_in

        model_output_x, adj_pred, node_logits = model(
            x_t,
            self._scale_timesteps(t),
            **training_kwargs
        )

        if self.model_mean_type == ModelMeanType.EPSILON:

            # Feature 0 (laundering): model predicts x_start directly (binary recovery)
            # Features 1+ (continuous): model predicts epsilon (Gaussian denoising)
            target = th.cat([x_start[..., 0:1], noise[..., 1:]], dim=-1)

        elif self.model_mean_type == ModelMeanType.START_X:

            target = x_start

        else:
            raise NotImplementedError()


        # --- Feature 0 (laundering label): weighted BCE with logits ---
        # MSE on a binary feature with class imbalance causes the model to
        # predict ~0 for all nodes. Weighted BCE forces it to learn the rare class.
        laund_logits = model_output_x[..., 0]   # [B, N] — treated as logit
        laund_true   = x_start[..., 0]          # [B, N] — binary {0, 1}

        if node_mask is not None:
            n_nodes = node_mask.sum(dim=-1).clamp(min=1)              # [B]
            n_pos   = (laund_true * node_mask).sum(dim=-1).clamp(min=1)
        else:
            n_nodes = th.tensor(float(laund_true.shape[-1]), device=laund_true.device).expand(laund_true.shape[0])
            n_pos   = laund_true.sum(dim=-1).clamp(min=1)

        n_neg       = (n_nodes - n_pos).clamp(min=1)
        pos_w_laund = (n_neg / n_pos).clamp(1.0, 50.0)               # [B]

        eps = 1e-6
        laund_prob = th.sigmoid(laund_logits).clamp(eps, 1.0 - eps)
        laund_bce  = -(
            pos_w_laund[:, None] * laund_true   * th.log(laund_prob)
            + (1.0 - laund_true) * th.log(1.0 - laund_prob)
        )                                                              # [B, N]

        if node_mask is not None:
            laund_loss = (laund_bce * node_mask).sum(dim=-1) / n_nodes
        else:
            laund_loss = laund_bce.mean(dim=-1)

        # --- Features 1+ (continuous): MSE on epsilon / x_start ---
        mse_cont = (target[..., 1:] - model_output_x[..., 1:]) ** 2  # [B, N, F-1]

        if node_mask is not None:
            mask = node_mask.unsqueeze(-1)
            cont_loss = (
                (mse_cont * mask).sum(dim=[1, 2])
                / mask.sum(dim=[1, 2]).clamp(min=1)
            )
        else:
            cont_loss = mse_cont.mean(dim=[1, 2])

        loss = cont_loss + laund_loss_weight * laund_loss


        # Adjacency reconstruction: weighted BCE (handles sparsity imbalance)
        if adj_start is not None:
            if node_mask is not None:
                mask2d = node_mask[:, :, None] * node_mask[:, None, :]
                n_valid = mask2d.sum(dim=[1, 2]).clamp(min=1)
                n_pos   = (adj_start * mask2d).sum(dim=[1, 2]).clamp(min=1)
            else:
                n_valid = th.tensor(
                    float(adj_start.shape[-1] ** 2), device=adj_start.device
                ).expand(adj_start.shape[0])
                n_pos = adj_start.sum(dim=[1, 2]).clamp(min=1)

            # pos_weight: mild upweight of edges to counter sparsity.
            # Clamped at 2 — higher values bias the model toward
            # over-predicting edges, leading to unrealistically dense generation.
            pos_w = ((n_valid - n_pos) / n_pos).clamp(1.0, adj_pos_weight_max)  # [B]

            eps = 1e-6
            adj_pred_c = adj_pred.clamp(eps, 1.0 - eps)
            bce = -(
                pos_w[:, None, None] * adj_start * th.log(adj_pred_c)
                + (1.0 - adj_start) * th.log(1.0 - adj_pred_c)
            )

            if node_mask is not None:
                adj_loss = (bce * mask2d).sum(dim=[1, 2]) / n_valid
            else:
                adj_loss = bce.mean(dim=[1, 2])

            loss = loss + adj_loss_weight * adj_loss

            # Density regularisation: symmetric squared penalty on the gap
            # between predicted and true density.  Two-sided so the model is
            # pulled toward the exact true density rather than just being
            # prevented from exceeding it — this is the primary signal that
            # counteracts the inner-product decoder's bias toward dense outputs.
            true_density = n_pos / n_valid                                   # [B]
            if node_mask is not None:
                pred_density = (adj_pred_c * mask2d).sum(dim=[1, 2]) / n_valid
            else:
                pred_density = adj_pred_c.mean(dim=[1, 2])
            density_loss = (pred_density - true_density) ** 2

            loss = loss + density_loss_weight * density_loss

            # ── Degree sequence adherence loss ────────────────────────────────
            # When degree_seq conditioning is active, explicitly penalise adj_pred
            # for producing per-node degrees that deviate from the target sequence.
            # This ensures the model learns to USE the degree_seq signal rather than
            # ignoring it (which would happen if only the adj BCE provided supervision).
            #
            # degree_seq: [B, N, 1] with values in [0, 1] = degree / n_active.
            # We normalise adj_pred's degree by the same denominator so both sides
            # live on the same scale regardless of graph size.
            if degree_seq_loss_weight > 0.0 and "degree_seq" in model_kwargs:
                deg_seq = model_kwargs["degree_seq"]          # [B, N, 1]
                deg_pred_raw = adj_pred.sum(dim=-1, keepdim=True)   # [B, N, 1]
                if node_mask is not None:
                    n_act = node_mask.sum(dim=-1, keepdim=True).unsqueeze(-1).clamp(min=1.0)  # [B,1,1]
                else:
                    n_act = float(adj_pred.shape[-1])
                deg_pred_norm = deg_pred_raw / n_act          # [B, N, 1] in same scale as deg_seq

                sq = (deg_pred_norm - deg_seq) ** 2           # [B, N, 1]
                if node_mask is not None:
                    nm = node_mask.unsqueeze(-1)               # [B, N, 1]
                    deg_seq_loss = (sq * nm).sum(dim=[1, 2]) / nm.sum(dim=[1, 2]).clamp(min=1.0)
                else:
                    deg_seq_loss = sq.mean(dim=[1, 2])

                loss = loss + degree_seq_loss_weight * deg_seq_loss

            # ── Direction 1: feature-topology consistency regulariser ─────────
            # Forces adj_pred to produce structural statistics (degree, clustering)
            # that match the ground-truth features in x_start.  Both quantities
            # are compared in the same normalised feature space so gradients scale
            # consistently with the other loss terms.
            if (consistency_loss_weight > 0.0
                    and x_mean_feat is not None and x_std_feat is not None):

                x_m = x_mean_feat.to(adj_pred.device)
                x_s = x_std_feat.to(adj_pred.device).clamp(min=1e-6)

                # --- Degree consistency ---
                # Zero the diagonal so self-loops don't inflate degree counts.
                n_sz     = adj_pred.shape[-1]
                eye_mask = th.eye(n_sz, device=adj_pred.device).unsqueeze(0)
                A_nd     = adj_pred * (1.0 - eye_mask)   # [B, N, N]
                if node_mask is not None:
                    A_nd = A_nd * mask2d                  # zero out padding pairs

                deg_raw = A_nd.sum(dim=-1)                # [B, N] — soft degree
                max_deg = deg_raw.detach().max(dim=-1, keepdim=True).values.clamp(min=1.0)
                deg_from_adj = deg_raw / max_deg          # [B, N] ∈ [0, 1]

                # Normalise into the same space as x_start features
                deg_from_adj_n = (deg_from_adj - x_m[degree_feat_col]) / x_s[degree_feat_col]
                deg_target     = x_start[..., degree_feat_col].detach()  # already normalised

                # --- Clustering consistency (differentiable triangle counting) ---
                # clust(i) = Σ_{j,k} A[i,j]·A[j,k]·A[i,k] / (d_i·(d_i-1))
                # = (A @ A * A).sum(-1) / (d·(d-1))
                # Computational note: bmm is O(N³) per batch but fast on GPU;
                # set consistency_loss_weight=0 on CPU-only runs if too slow.
                A2         = th.bmm(A_nd, A_nd)              # [B, N, N]
                clust_num  = (A2 * A_nd).sum(dim=-1)         # [B, N] = 2× expected triangles
                clust_denom = deg_raw * (deg_raw - 1.0) + 1e-8
                clust_from_adj   = (clust_num / clust_denom).clamp(0.0, 1.0)  # [B, N]
                clust_from_adj_n = (clust_from_adj - x_m[clust_feat_col]) / x_s[clust_feat_col]
                clust_target     = x_start[..., clust_feat_col].detach()

                if node_mask is not None:
                    nm         = node_mask.bool()
                    deg_cons   = ((deg_from_adj_n[nm]   - deg_target[nm])   ** 2).mean()
                    clust_cons = ((clust_from_adj_n[nm] - clust_target[nm]) ** 2).mean()
                else:
                    deg_cons   = ((deg_from_adj_n   - deg_target)   ** 2).mean()
                    clust_cons = ((clust_from_adj_n - clust_target) ** 2).mean()

                consistency_loss = (deg_cons + clust_cons) * 0.5
                loss = loss + consistency_loss_weight * consistency_loss


        # Node existence loss: predict original mask from the corrupted input.
        # This teaches the model which nodes *should* be active given the graph
        # structure — the foundation for meaningful node insertion/deletion.
        if node_mask is not None:
            node_exist_loss = th.nn.functional.binary_cross_entropy_with_logits(
                node_logits,   # [B, N] raw logits
                node_mask,     # [B, N] original mask (before corruption)
                reduction="none",
            ).mean(dim=-1)     # [B]
            loss = loss + node_exist_loss_weight * node_exist_loss

        return {
            "loss": loss,
            "mse": loss,
        }