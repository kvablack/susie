from functools import partial

import jax
import jax.numpy as jnp


@jax.jit
def q_sample(x_0, log_snr, noise):
    """Samples from the diffusion process at a given timestep.

    Args:
        x_0: the start image.
        log_snr: the log SNR (lambda) for the timestep t.
        noise: the noise, typically N(0, I) or output of the UNet.
    Returns:
        the resulting sample from q(x_t | x_0).
    """
    # based on A.4 of VDM paper
    # the signs are flipped bc log_snr is monotonic decreasing
    # (rather than increasing as is their learned gamma in the paper)
    alpha = jnp.sqrt(jax.nn.sigmoid(log_snr))[:, None, None, None]
    sigma = jnp.sqrt(jax.nn.sigmoid(-log_snr))[:, None, None, None]
    return alpha * x_0 + sigma * noise


@partial(jax.jit, static_argnames="use_ema")
def model_predict(state, x, y, prompt_embeds, t, use_ema=True):
    """Runs forward inference of the model.

    Args:
        state: an EmaTrainState instance.
        x: the input image.
        y: the image context.
        prompt_embeds: the prompt embeddings.
        t: the current timestep in the range [0, 1].
        use_ema: whether to use the exponential moving average of the parameters.
    Returns:
        the raw output of the UNet.
    """
    if use_ema:
        variables = {"params": state.params_ema}
    else:
        variables = {"params": state.params}

    input = jnp.concatenate([x, y], axis=-1)

    return state.apply_fn(variables, input, t * 1000, prompt_embeds, train=False)


@partial(jax.jit, static_argnames="log_snr_fn")
def sample_step(
    rng,
    state,
    x,
    y,
    prompt_embeds,
    uncond_y,
    uncond_prompt_embeds,
    t,
    t_next,
    log_snr_fn,
    context_w,
    prompt_w,
    eta,
):
    """Runs a sampling step.

    Derived from a combination of
    https://github.com/lucidrains/imagen-pytorch/blob/main/imagen_pytorch/imagen_pytorch.py,
    the appendix of the VDM paper, the appendix of the Imagen paper, and DDIM
    paper.

    Args:
        rng: a JAX PRNGKey.
        state: an EmaTrainState instance.
        x: the input image (batched).
        y: the image context (batched).
        prompt_embeds: the prompt embeddings (batched).
        t: the current timestep (t).
        t_next: the next timestep (s, with s < t).
        log_snr_fn: a function that takes a timestep t ~ [0, 1] and returns the log SNR.
        w: the weight to use for classifier-free guidance. 1.0 (default) uses the
            conditional model only.
        eta: the DDIM eta parameter. 0.0 is full deterministic, 1.0 is ancestral
            sampling.
    """
    assert len(x.shape) == 4
    assert len(y.shape) == 4

    batched_t = jnp.full(x.shape[0], t)
    batched_t_next = jnp.full(x.shape[0], t_next)

    uncond_pred = model_predict(state, x, uncond_y, uncond_prompt_embeds, batched_t)
    context_pred = model_predict(state, x, y, uncond_prompt_embeds, batched_t)
    prompt_pred = model_predict(state, x, y, prompt_embeds, batched_t)

    pred_eps = (
        uncond_pred
        + context_w * (context_pred - uncond_pred)
        + prompt_w * (prompt_pred - context_pred)
    )

    # compute log snr
    log_snr = log_snr_fn(batched_t)
    log_snr_next = log_snr_fn(batched_t_next)

    # signs are flipped from VDM paper, see q_sample above
    alpha_t_sq = jax.nn.sigmoid(log_snr)[:, None, None, None]
    alpha_s_sq = jax.nn.sigmoid(log_snr_next)[:, None, None, None]
    sigma_t_sq = jax.nn.sigmoid(-log_snr)[:, None, None, None]
    sigma_s_sq = jax.nn.sigmoid(-log_snr_next)[:, None, None, None]

    # this constant from A.4 of the VDM paper is equal to sigma_{t|s}^2 / sigma_t^2
    c = -jnp.expm1(log_snr - log_snr_next)[:, None, None, None]

    # this is equivalent to the posterior stddev in ancestral sampling
    d = jnp.sqrt(sigma_s_sq * c)
    # DDIM scales this by eta
    d = eta * d

    # fresh noise
    noise = jax.random.normal(rng, x.shape)

    # get predicted x0
    x_0 = (x - jnp.sqrt(sigma_t_sq) * pred_eps) / jnp.sqrt(alpha_t_sq)

    # clip it -- removed bc latent space is not bounded
    # x_0 = jnp.clip(x_0, -1, 1)

    # compute x_s using DDIM formula
    x_s = (
        jnp.sqrt(alpha_s_sq) * x_0
        + jnp.sqrt(sigma_s_sq - d**2) * pred_eps
        + d * noise
    )
    return x_s, x_0


@partial(jax.jit, static_argnames=("num_timesteps", "log_snr_fn"))
def sample_loop(
    rng,
    state,
    y,
    prompt_embeds,
    uncond_y,
    uncond_prompt_embeds,
    *,
    log_snr_fn,
    num_timesteps,
    context_w=1.0,
    prompt_w=1.0,
    eta=0.0,
):
    """Runs the full sampling loop.

    Implements the following loop using a scan:

    ```
        for t, t_next in reversed(zip(t_seq, t_seq_next)):
            rng, step_rng = jax.random.split(rng)
            x, x0 = sample_step(x, x0, ...)
        return x0
    ```

    Args:
        rng: a JAX PRNGKey.
        state: an EmaTrainState instance.
        y: the image context (batched).
        prompt_embeds: the text prompt embeddings (batched).
        log_snr_fn: a function that takes a timestep t ~ [0, 1] and returns the log SNR.
        num_timesteps: the number of timesteps to run.
        w: the weight to use for classifier-free guidance. 1.0 (default) uses the
            conditional model only.
        eta: the DDIM eta parameter. 0.0 (default) is full deterministic, 1.0 is
            ancestral sampling.
    """
    assert len(y.shape) == 4

    def scan_fn(carry, t_combined):
        rng, x, x0 = carry
        t, t_next = t_combined
        rng, step_rng = jax.random.split(rng)
        x, x0 = sample_step(
            step_rng,
            state,
            x,
            y,
            prompt_embeds,
            uncond_y,
            uncond_prompt_embeds,
            t,
            t_next,
            log_snr_fn,
            context_w,
            prompt_w,
            eta,
        )
        return (rng, x, x0), None

    if y.shape[-1] % 4 == 0:
        # vae-encoded
        channel_dim = 4
    elif y.shape[-1] % 3 == 0:
        # full images
        channel_dim = 3
    else:
        raise ValueError(f"Invalid channel dimension {y.shape[-1]}")

    rng, init_rng = jax.random.split(rng)
    x = jax.random.normal(
        init_rng, y.shape[:-1] + (channel_dim,)
    )  # initial image (pure noise)
    x0 = jnp.zeros_like(x)  # unused

    # evenly spaced sequence of timesteps
    t_seq = jnp.linspace(0, 1, num=num_timesteps, endpoint=False, dtype=jnp.float32)
    t_seq_cur = t_seq[1:]
    t_seq_next = t_seq[:-1]
    t_seq_combined = jnp.stack([t_seq_cur, t_seq_next], axis=-1)

    rng, scan_rng = jax.random.split(rng)
    (_, _, final_x0), _ = jax.lax.scan(
        scan_fn, (scan_rng, x, x0), t_seq_combined[::-1], unroll=1
    )

    return final_x0
