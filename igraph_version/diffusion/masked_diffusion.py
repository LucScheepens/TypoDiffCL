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

    def q_sample(self, x_start, t, noise=None, node_mask=None):

        if noise is None:
            noise = th.randn_like(x_start)

        if node_mask is not None:

            mask = node_mask.unsqueeze(-1)

            x_start = x_start * mask
            noise = noise * mask

        return (
            _extract_into_tensor(
                self.sqrt_alphas_cumprod, t, x_start.shape
            ) * x_start
            +
            _extract_into_tensor(
                self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
            ) * noise
        )


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

        model_output = model(
            x,
            self._scale_timesteps(t),
            **model_kwargs
        )

        if self.model_mean_type == ModelMeanType.EPSILON:

            pred_xstart = self._predict_xstart_from_eps(
                x, t, model_output
            )

        elif self.model_mean_type == ModelMeanType.START_X:

            pred_xstart = model_output

        else:
            raise NotImplementedError()


        if clip_denoised:
            pred_xstart = pred_xstart.clamp(-1, 1)


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


        return {
            "sample": sample,
            "pred_xstart": out["pred_xstart"],
        }


    # ========================================================
    # Sampling Loop
    # ========================================================

    def p_sample_loop(
        self,
        model,
        shape,
        model_kwargs=None,
        device=None,
    ):

        if device is None:
            device = next(model.parameters()).device

        if model_kwargs is None:
            model_kwargs = {}

        x = th.randn(*shape, device=device)

        for i in reversed(range(self.num_timesteps)):

            t = th.tensor(
                [i] * shape[0],
                device=device
            )

            with th.no_grad():

                out = self.p_sample(
                    model,
                    x,
                    t,
                    model_kwargs=model_kwargs,
                )

                x = out["sample"]

        return x


    def training_losses(
        self,
        model,
        x_start,
        t,
        model_kwargs=None,
        noise=None,
    ):

        if model_kwargs is None:
            model_kwargs = {}

        node_mask = model_kwargs.get("node_mask", None)
        print("node_mask in training_losses:", node_mask)

        if noise is None:
            noise = th.randn_like(x_start)


        x_t = self.q_sample(
            x_start,
            t,
            noise=noise,
            node_mask=node_mask,
        )


        model_output = model(
            x_t,
            self._scale_timesteps(t),
            **model_kwargs
        )


        if self.model_mean_type == ModelMeanType.EPSILON:

            target = noise

        elif self.model_mean_type == ModelMeanType.START_X:

            target = x_start

        else:
            raise NotImplementedError()


        mse = (target - model_output) ** 2


        if node_mask is not None:

            mask = node_mask.unsqueeze(-1)

            mse = mse * mask

            loss = (
                mse.sum(dim=[1, 2])
                /
                mask.sum(dim=[1, 2]).clamp(min=1)
            )

        else:

            loss = mean_flat(mse)


        return {
            "loss": loss,
            "mse": loss,
        }