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

        self.alphas_cumprod = np.cumprod(alphas)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])

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

        # Bernoulli forward process for adjacency (binary edges)
        keep_prob_adj = _extract_into_tensor(
            self.alphas_cumprod, t, adj_start.shape
        ).clamp(0.0, 1.0)
        keep_adj = th.bernoulli(keep_prob_adj)
        rand_adj  = th.bernoulli(th.full_like(adj_start, 0.5))
        adj_t = keep_adj * adj_start + (1 - keep_adj) * rand_adj
        adj_t = (adj_t + adj_t.transpose(-1, -2)) / 2          # keep symmetric

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
    ):

        if model_kwargs is None:
            model_kwargs = {}

        node_mask = model_kwargs.get("node_mask", None)

        if node_mask is not None:
            x = x * node_mask.unsqueeze(-1)

        model_output_x, adj_pred = model(
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
            pred_xstart = th.cat([
                pred_xstart[..., 0:1].clamp(0.0, 1.0),
                pred_xstart[..., 1:],
            ], dim=-1)


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
        }



    def p_sample(
        self,
        model,
        x,
        t,
        model_kwargs=None,
    ):

        if model_kwargs is None:
            model_kwargs = {}

        node_mask = model_kwargs.get("node_mask", None)

        out = self.p_mean_variance(
            model,
            x,
            t,
            model_kwargs=model_kwargs,
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


        # Adjacency reverse step: predict-x0-and-resample (Bernoulli)
        adj_pred = out["adj_pred"]
        nonzero_adj = (t != 0).float().view(-1, 1, 1)
        adj_stoch   = th.bernoulli(adj_pred.clamp(0.0, 1.0))
        adj_determ  = (adj_pred > 0.5).float()
        adj_sample  = th.where(nonzero_adj.bool(), adj_stoch, adj_determ)
        adj_sample  = (adj_sample + adj_sample.transpose(-1, -2)) / 2  # symmetric

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
    ):
        """
        shape     : (B, N, F) for node features
        adj_shape : (B, N, N) — pass this to generate the adjacency from noise.
                    If None, adj must be supplied in model_kwargs (old behaviour).
        """

        if device is None:
            device = next(model.parameters()).device

        if model_kwargs is None:
            model_kwargs = {}

        x = th.randn(*shape, device=device)

        # Initialise adjacency from Bernoulli(0.5) when generating graphs
        adj = None
        if adj_shape is not None:
            adj = th.bernoulli(th.full(adj_shape, 0.5, device=device))
            adj = (adj + adj.transpose(-1, -2)) / 2
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
                )

                x = out["sample"]
                if adj is not None:
                    adj = out["adj_sample"]

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
        deg_loss_weight=0.1,
        laund_loss_weight=1.0,
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
            # Pass noisy adj to the model for message passing
            training_kwargs = {**model_kwargs, "adj": adj_t}
        else:
            x_t = self.q_sample(x_start, t, noise=noise, node_mask=node_mask)
            training_kwargs = model_kwargs

        model_output_x, adj_pred = model(
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

            # pos_weight: upweight positive (edge) class to counter sparsity
            pos_w = ((n_valid - n_pos) / n_pos).clamp(1.0, 50.0)  # [B]

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

            # Degree distribution matching: soft degree of adj_pred vs adj_start
            # Normalise by n_nodes so loss is in [0, 1] instead of [0, N²]
            if node_mask is not None:
                n_nodes_float = node_mask.sum(dim=-1, keepdim=True).clamp(min=1)  # [B, 1]
            else:
                n_nodes_float = float(adj_pred.shape[-1])
            deg_pred = adj_pred.sum(dim=-1) / n_nodes_float   # fractional degree [B, N]
            deg_true = adj_start.sum(dim=-1) / n_nodes_float  # fractional degree [B, N]
            deg_mse  = (deg_pred - deg_true) ** 2

            if node_mask is not None:
                deg_loss = (
                    (deg_mse * node_mask).sum(dim=-1)
                    / node_mask.sum(dim=-1).clamp(min=1)
                )
            else:
                deg_loss = deg_mse.mean(dim=-1)

            loss = loss + deg_loss_weight * deg_loss


        return {
            "loss": loss,
            "mse": loss,
        }