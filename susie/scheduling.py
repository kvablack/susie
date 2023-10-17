import jax
import jax.numpy as jnp

# continuous timestep scheduling adapted from
# https://github.com/lucidrains/imagen-pytorch/blob/main/imagen_pytorch/imagen_pytorch.py
#
# this code follows the conventions and notations from the Variation Diffusion Models (VDM)
# paper, where the timestep t is a continuous variable in [0, 1] that is mapped to a log SNR
# lambda using a schedule. the log SNR is what the UNet is conditioned on, and is also used
# to compute the mean and stdev (alpha and sigma) of the diffusion process. I believe
# most of the particular formulas (e.g. the usage of expm1) are from the appendix of the
# VDM paper.


def lnpoch(a, b):
    # Computes the log of the rising factorial (a)_b in a numerically stable way when a >> b.
    # From https://stackoverflow.com/questions/21228076/the-precision-of-scipy-special-gammaln
    # lmao
    return (
        (b**4 / 12 - b**3 / 6 + b**2 / 12) / a**3
        + (-(b**3) / 6 + b**2 / 4 - b / 12) / a**2
        + (b**2 / 2 - b / 2) / a
        - b * jnp.log(1 / a)
    )


def linear_log_snr(t, *, beta_start=0.001, beta_end=0.02, num_timesteps=1000):
    """Computes log SNR from t ~ [0, 1] for a linear beta schedule."""
    m = (beta_end - beta_start) / num_timesteps
    b = 1 - beta_start
    n = t * num_timesteps

    log_alpha_sq = (n + 1) * jnp.log(m) + lnpoch(b / m - n, n + 1)
    return jax.scipy.special.logit(jnp.exp(log_alpha_sq))


def scaled_linear_log_snr(t, *, beta_start=0.00085, beta_end=0.012, num_timesteps=1000):
    """Computes log SNR from t ~ [0, 1] for a scaled (sqrt) linear beta schedule, as used in stable diffusion."""
    m = (beta_end**0.5 - beta_start**0.5) / num_timesteps
    b = beta_start**0.5
    n = t * num_timesteps

    fact = lnpoch((1 - b) / m - n, n + 1) + lnpoch((1 + b) / m, n + 1)
    pow = 2 * (n + 1) * jnp.log(m)
    alpha_sq = jnp.exp(fact + pow)
    return jax.scipy.special.logit(alpha_sq)


def cosine_log_snr(t, s: float = 0.008, d: float = 0.008):
    """Computes log SNR from t ~ [0, 1] for a cosine beta schedule.

    In the original Improved DDPM paper, they add an offset of s=0.008 on the
    *left* side of the schedule, because they found it was hard for the NN to
    predict very small amounts of noise. Without this offset we would have
    alpha=1 and sigma=0 at t=0 and hence log_snr=+inf. However, they leave the
    singularity on the *right* side of the schedule: i.e. at t=1, alpha=0 and
    sigma=1, so log_snr=-inf. They deal with this singularity by clipping beta
    to a maximum value of 0.999. The problem is that in this formulation we
    don't directly calculate alpha or beta -- instead we define the schedule in
    terms of log_snr and calculate all other relevant quantities from that. So,
    to deal with the singularity at t=1, I'm adding a symmetrical offset of
    d=0.008 on the right side of the schedule, so that log_snr is finite at t=1.
    I've never seen this anywhere, but hopefully it works :).
    """
    return -jnp.log((jnp.cos(((t / (1 + d)) + s) / (1 + s) * jnp.pi * 0.5) ** -2) - 1)


def create_log_snr_fn(config):
    """
    Returns a function that maps from t ~ [0, 1] to lambda (log SNR). The log SNR is
    used to condition the neural network as well as compute the mean and stdev of the
    diffusion process.
    """
    schedule_name = config["noise_schedule"]

    if schedule_name == "linear":
        log_snr_fn = linear_log_snr
    elif schedule_name == "cosine":
        log_snr_fn = cosine_log_snr
    elif schedule_name == "scaled_linear":
        log_snr_fn = scaled_linear_log_snr
    else:
        raise ValueError(f"unknown noise schedule {schedule_name}")

    return log_snr_fn


def create_ema_decay_fn(config):
    def ema_decay_schedule(step):
        count = jnp.clip(step - config.start_step - 1, a_min=0.0)
        value = 1 - (1 + count / config.inv_gamma) ** -config.power
        ema_rate = jnp.clip(value, a_min=config.min_decay, a_max=config.max_decay)
        return ema_rate

    return ema_decay_schedule
