import datetime
import functools
import logging
import os
import tempfile
import time
from collections import defaultdict
from functools import partial

import einops as eo
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint
import tensorflow as tf

import tqdm
from absl import app, flags
from flax.training import orbax_utils
from jax.experimental import multihost_utils
from jax.lax import with_sharding_constraint as wsc
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from ml_collections import ConfigDict, config_flags
from PIL import Image

import wandb
from susie import sampling, scheduling
from susie.data.datasets import get_data_loader
from susie.jax_utils import (
    host_broadcast_str,
    initialize_compilation_cache,
)
from susie.model import (
    EmaTrainState,
    create_model_def,
    load_pretrained_unet,
    load_text_encoder,
    load_vae,
)

if jax.process_count() > 1:
    jax.distributed.initialize()

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

try:
    from jax_smi import initialise_tracking  # type: ignore

    initialise_tracking()
except ImportError:
    pass


def fsdp_sharding(mesh: jax.sharding.Mesh, array: jax.ShapeDtypeStruct):
    if array.ndim < 2:
        # replicate scalar and vector arrays
        return NamedSharding(mesh, P())

    # shard matrices and larger tensors across the fsdp dimension. the conv kernels are a little tricky because they
    # vary in which axis is a power of 2, so I'll just search for the first one that works.
    l = []
    for n in array.shape:
        if n % mesh.shape["fsdp"] == 0:
            l.append("fsdp")
            return NamedSharding(mesh, P(*l))
        l.append(None)

    logging.warning(
        f"Could not find a valid sharding for array of shape {array.shape} with mesh of shape {mesh.shape}"
    )
    return NamedSharding(mesh, P())


def train_step(
    rng,
    state,
    batch,
    # static args
    log_snr_fn,
    uncond_prompt_embed,
    text_encode_fn,
    vae_encode_fn,
    curr_drop_rate=0.0,
    goal_drop_rate=0.0,
    prompt_drop_rate=0.0,
    eval_only=False,
):
    batch_size = batch["subgoals"].shape[0]

    # encode stuff
    for key in {"curr", "goals", "subgoals"}.intersection(batch.keys()):
        # VERY IMPORTANT: for some godforsaken reason, the context latents are
        # NOT scaled in InstructPix2Pix
        scale = key == "subgoals"
        rng, encode_rng = jax.random.split(rng)
        batch[key] = vae_encode_fn(encode_rng, batch[key], scale=scale)
    prompt_embeds = text_encode_fn(batch["prompt_ids"])

    if goal_drop_rate == 1.0:
        batch["goals"] = jnp.zeros(
            batch["subgoals"].shape[:-1] + (0,), batch["subgoals"].dtype
        )
    elif goal_drop_rate > 0:
        rng, mask_rng = jax.random.split(rng)
        batch["goals"] = jnp.where(
            jax.random.uniform(mask_rng, shape=(batch_size, 1, 1, 1)) < goal_drop_rate,
            0,
            batch["goals"],
        )

    if curr_drop_rate > 0:
        rng, mask_rng = jax.random.split(rng)
        batch["curr"] = jnp.where(
            jax.random.uniform(mask_rng, shape=(batch_size, 1, 1, 1)) < curr_drop_rate,
            0,
            batch["curr"],
        )

    if prompt_drop_rate > 0:
        rng, mask_rng = jax.random.split(rng)
        prompt_embeds = jnp.where(
            jax.random.uniform(mask_rng, shape=(batch_size, 1, 1)) < prompt_drop_rate,
            uncond_prompt_embed,
            prompt_embeds,
        )

    x = batch["subgoals"]  # the generation target
    y = jnp.concatenate(
        [batch["curr"], batch["goals"]], axis=-1
    )  # the conditioning image(s)

    # sample batch of timesteps from t ~ U[0, num_train_timesteps)
    rng, t_rng = jax.random.split(rng)
    t = jax.random.uniform(t_rng, shape=(batch_size,), dtype=jnp.float32)

    # sample noise (epsilon) from N(0, I)
    rng, noise_rng = jax.random.split(rng)
    noise = jax.random.normal(noise_rng, x.shape)

    log_snr = log_snr_fn(t)

    # generate the noised image from q(x_t | x_0, y)
    x_t = sampling.q_sample(x, log_snr, noise)

    input = jnp.concatenate([x_t, y], axis=-1)

    # seems like remat is actually enabled by default -- this disables it
    # @partial(jax.checkpoint, policy=jax.checkpoint_policies.everything_saveable)
    def loss_fn(params, rng):
        pred = state.apply_fn(
            {"params": params},
            input,
            t * 1000,
            prompt_embeds,
            train=not eval_only,
            rngs={"dropout": rng},
        )
        assert pred.shape == noise.shape
        loss = (pred - noise) ** 2
        return jnp.mean(loss)

    info = {}
    if not eval_only:
        grad_fn = jax.value_and_grad(loss_fn)
        rng, dropout_rng = jax.random.split(rng)
        info["loss"], grads = grad_fn(state.params, dropout_rng)
        info["grad_norm"] = optax.global_norm(grads)

        new_state = state.apply_gradients(grads=grads)
    else:
        rng, dropout_rng = jax.random.split(rng)
        info["loss"] = loss_fn(state.params, dropout_rng)
        rng, dropout_rng = jax.random.split(rng)
        info["loss_ema"] = loss_fn(state.params_ema, dropout_rng)
        new_state = state

    return new_state, info


FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    "config",
    None,
    "File path to the hyperparameter configuration.",
    lock_config=False,
)


def main(_):
    config = FLAGS.config

    assert config.sample.num_contexts % 4 == 0

    # prevent tensorflow from using GPUs
    tf.config.experimental.set_visible_devices([], "GPU")
    tf.random.set_seed(config.seed + jax.process_index())

    # get jax devices
    logging.info(f"JAX process: {jax.process_index()} of {jax.process_count()}")
    logging.info(f"Local devices: {jax.local_device_count()}")
    logging.info(f"Global devices: {jax.device_count()}")

    mesh = jax.sharding.Mesh(
        # create_device_mesh([32, 1]), # can't make contiguous meshes for the v4-64 pod for some reason
        np.array(jax.devices()).reshape(*config.mesh),
        axis_names=["dp", "fsdp"],
    )
    replicated_sharding = NamedSharding(mesh, P())
    # data gets sharded over both dp and fsdp logical axes
    data_sharding = NamedSharding(mesh, P(["dp", "fsdp"]))

    # initial rng
    rng = jax.random.PRNGKey(config.seed + jax.process_index())

    # set up wandb run
    if config.wandb_resume_id is not None:
        run = wandb.Api().run(config.wandb_resume_id)
        old_num_steps = config.num_steps
        config = ConfigDict(run.config)
        config.num_steps = old_num_steps
        config.wandb_resume_id = run.id
        logdir = tf.io.gfile.join(config.logdir, run.name)

        if jax.process_index() == 0:
            wandb.init(
                project=run.project,
                id=run.id,
                resume="must",
            )
    else:
        unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
        unique_id = host_broadcast_str(unique_id)

        if not config.run_name:
            config.run_name = unique_id
        else:
            config.run_name += "_" + unique_id

        logdir = tf.io.gfile.join(config.logdir, config.run_name)

        if jax.process_index() == 0:
            tf.io.gfile.makedirs(logdir)
            wandb.init(
                project=config.wandb_project,
                name=config.run_name,
                config=config.to_dict(),
            )

    checkpointer = orbax.checkpoint.CheckpointManager(
        logdir,
        checkpointers={
            "state": orbax.checkpoint.PyTreeCheckpointer(),
            "params_ema": orbax.checkpoint.PyTreeCheckpointer(),
        },
    )

    log_snr_fn = scheduling.create_log_snr_fn(config.scheduling)
    ema_decay_fn = scheduling.create_ema_decay_fn(config.ema)

    # load vae
    if config.vae is not None:
        vae_encode, vae_decode = load_vae(config.vae)

    # load text encoder
    tokenize, untokenize, text_encode = load_text_encoder(config.text_encoder)
    uncond_prompt_embed = jax.device_get(text_encode(tokenize([""])))  # (1, 77, 768)

    def tokenize_fn(batch):
        lang = [s.decode("utf-8") for s in batch.pop("lang")]
        assert all(s != "" for s in lang)
        batch["prompt_ids"] = tokenize(lang)
        return batch

    # load pretrained model
    if config.model.get("pretrained", None):
        pretrained_model_def, pretrained_params = load_pretrained_unet(
            config.model.pretrained, in_channels=12 if config.goal_drop_rate < 1 else 8
        )
        pretrained_config = ConfigDict(pretrained_model_def.config)
        del config.model.pretrained
        if config.model.keys():
            logging.warning(f"Overriding pretrained config keys: {config.model.keys()}")
            pretrained_config.update(config.model)
        config.model = pretrained_config
    else:
        pretrained_params = None

    # create model def
    config.model.out_channels = 4 if config.vae else 3
    model_def = create_model_def(config.model)

    # create optimizer
    learning_rate_fn = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.optim.lr,
        warmup_steps=config.optim.warmup_steps,
        decay_steps=config.optim.decay_steps,
        end_value=0.0,
    )
    tx = optax.adamw(
        learning_rate=learning_rate_fn,
        b1=config.optim.beta1,
        b2=config.optim.beta2,
        eps=config.optim.epsilon,
        weight_decay=config.optim.weight_decay,
        mu_dtype=jnp.bfloat16,
    )
    tx = optax.chain(
        optax.clip_by_global_norm(config.optim.max_grad_norm),
        tx,
    )

    if config.optim.accumulation_steps > 1:
        tx = optax.MultiSteps(tx, config.optim.accumulation_steps)

    # create data loader
    train_loader, val_loader, num_datasets = get_data_loader(
        config.data, tokenize_fn, mesh
    )
    # warm up loaders
    logging.info("Warming up data loaders...")
    next(train_loader), next(val_loader)

    # initialize parameters
    if pretrained_params is None or config.wandb_resume_id is not None:
        example_batch = next(train_loader)
        print("--------------------------------")
        print(example_batch["subgoals"].shape)

        def init_fn(init_rng):
            if config.goal_drop_rate == 1.0:
                example_batch["goals"] = jnp.zeros(
                    example_batch["subgoals"].shape[:-1] + (0,),
                    example_batch["subgoals"].dtype,
                )
            example_input = jnp.concatenate(
                [
                    example_batch["subgoals"],
                    example_batch["curr"],
                    example_batch["goals"],
                ],
                axis=-1,
            )
            example_timesteps = jnp.zeros(example_input.shape[:1], example_input.dtype)
            example_prompt_embeds = jnp.zeros(
                [example_input.shape[0], 77, 768], example_input.dtype
            )

            if config.vae:
                example_input = vae_encode(rng, example_input)

            params = model_def.init(
                init_rng, example_input, example_timesteps, example_prompt_embeds
            )["params"]
            state = EmaTrainState.create(
                apply_fn=model_def.apply, params=params, params_ema=params, tx=tx
            )
            return state

        state_shape = jax.eval_shape(
            init_fn, rng
        )  # pytree of ShapeDtypeStructs for the TrainState
        state_sharding = jax.tree_map(
            lambda x: fsdp_sharding(mesh, x), state_shape
        )  # pytree of NamedShardings

        if config.wandb_resume_id is None:
            # initialize sharded TrainState
            rng, init_rng = jax.random.split(rng)
            state = jax.jit(init_fn, out_shardings=state_sharding)(init_rng)
        else:
            # restore from checkpoint
            state = checkpointer.restore(
                checkpointer.latest_step(),
                items={
                    "state": state_shape,
                    "params_ema": None,
                },
            )["state"]
            state = jax.tree_map(
                lambda arr, sharding: jax.make_array_from_callback(
                    arr.shape, sharding, lambda index: arr[index]
                ),
                state,
                state_sharding,
            )
    else:
        assert pretrained_params is not None
        state = EmaTrainState.create(
            apply_fn=model_def.apply,
            params=pretrained_params,
            params_ema=pretrained_params,
            tx=tx,
        )
        state = jax.tree_map(np.array, state)
        state_sharding = jax.tree_map(lambda x: fsdp_sharding(mesh, x), state)
        state = jax.tree_map(
            lambda arr, sharding: jax.make_array_from_callback(
                arr.shape, sharding, lambda index: arr[index]
            ),
            state,
            state_sharding,
        )

    # create train and eval step
    train_step_configured = partial(
        train_step,
        log_snr_fn=log_snr_fn,
        uncond_prompt_embed=uncond_prompt_embed,
        text_encode_fn=text_encode,
        vae_encode_fn=vae_encode if config.vae else lambda rng, x, *_, **__: x,
        curr_drop_rate=config.curr_drop_rate,
        goal_drop_rate=config.goal_drop_rate,
        prompt_drop_rate=config.prompt_drop_rate,
    )
    train_in_shardings = (
        replicated_sharding,  # rng
        state_sharding,  # state
        data_sharding,  # batch
    )
    train_out_shardings = (
        state_sharding,  # new_state
        replicated_sharding,  # info
    )
    train_step_jit = jax.jit(
        partial(train_step_configured, eval_only=False),
        in_shardings=train_in_shardings,
        out_shardings=train_out_shardings,
        donate_argnums=1,
    )
    eval_step_jit = jax.jit(
        partial(train_step_configured, eval_only=True),
        in_shardings=train_in_shardings,
        out_shardings=train_out_shardings,
        donate_argnums=1,
    )

    # shard ema decay
    EmaTrainState.apply_ema_decay = jax.jit(
        EmaTrainState.apply_ema_decay,
        in_shardings=(state_sharding, replicated_sharding),  # state, ema_decay
        out_shardings=state_sharding,  # new_state
        donate_argnums=0,  # donate state (have to respecify; it doesn't carry over from the inner jit)
    )

    # shard sample loop
    sample_loop_configured = partial(
        sampling.sample_loop,
        log_snr_fn=log_snr_fn,
        num_timesteps=config.sample.num_steps,
        context_w=config.sample.context_w,
        prompt_w=config.sample.prompt_w,
        eta=config.sample.eta,
    )
    sample_loop_jit = jax.jit(
        sample_loop_configured,
        in_shardings=(
            replicated_sharding,  # rng
            state_sharding,  # state
            replicated_sharding,  # y
            replicated_sharding,  # prompt_embeds
            replicated_sharding,  # uncond_y
            replicated_sharding,  # uncond_prompt_embeds
        ),
        out_shardings=replicated_sharding,  # returned samples
    )

    train_metrics = defaultdict(list)
    last_t = time.time()

    start_step = int(jax.device_get(state.step))
    pbar = tqdm(range(start_step, config.num_steps))
    for step in pbar:
        batch = next(train_loader)

        rng, train_step_rng = jax.random.split(rng)
        state, info = train_step_jit(train_step_rng, state, batch)
        pbar.set_postfix_str(f"loss: {info['loss']:.6f}")
        for k, v in info.items():
            train_metrics[k].append(v)

        # update ema params
        if (step + 1) <= config.ema.start_step:
            state = state.apply_ema_decay(0.0)
        if (step + 1) % config.ema.update_every == 0:
            ema_decay = ema_decay_fn(step)
            state = state.apply_ema_decay(ema_decay)

        # train logging
        if (step + 1) % config.log_interval == 0 and jax.process_index() == 0:
            summary = {f"train/{k}": np.mean(v) for k, v in train_metrics.items()}
            summary["time/seconds_per_step"] = (
                time.time() - last_t
            ) / config.log_interval

            train_metrics = defaultdict(list)
            last_t = time.time()

            summary["train/ema_decay"] = jax.device_get(ema_decay)
            summary["train/lr"] = jax.device_get(learning_rate_fn(step))

            wandb.log(summary, step=step + 1)

        if (step + 1) % config.val_interval == 0:
            # compute and log validation metrics
            val_metrics = defaultdict(list)
            for _ in tqdm(range(config.num_val_batches), desc="val", position=1):
                batch = next(val_loader)
                rng, val_step_rng = jax.random.split(rng)
                state, info = eval_step_jit(val_step_rng, state, batch)
                for k, v in info.items():
                    val_metrics[k].append(v)
            if jax.process_index() == 0:
                summary = {f"val/{k}": np.mean(v) for k, v in val_metrics.items()}
                wandb.log(summary, step=step + 1)

        if (step + 1) % config.sample_interval == 0:
            pbar.set_postfix_str("sampling")

            data = defaultdict(list)
            while not data or len(data["curr"]) < config.sample.num_contexts:
                batch = next(val_loader)
                batch = multihost_utils.process_allgather(batch)
                for key in {"curr", "goals", "prompt_ids"}.intersection(batch.keys()):
                    data[key].extend(batch[key])
            data = {k: np.array(v) for k, v in data.items()}

            data = jax.tree_map(lambda x: x[: config.sample.num_contexts], data)

            # get rid of goals if we're not using them
            if config.goal_drop_rate == 1.0:
                data["goals"] = np.zeros(
                    data["curr"].shape[:-1] + (0,), data["curr"].dtype
                )
            else:
                # make the first half have no prompt
                # data["prompt_ids"][: config.sample.num_contexts // 2] = uncond_prompt_id
                # make the second half have no goal
                # data["goals"][config.sample.num_contexts // 2 :] = 0
                pass

            # concatenate to make context
            contexts = np.concatenate([data["curr"], data["goals"]], axis=-1)

            # encode stuff
            if config.vae:
                rng, encode_rng = jax.random.split(rng)
                contexts = jax.device_get(vae_encode(encode_rng, contexts))
            prompt_embeds = jax.device_get(text_encode(data["prompt_ids"]))

            # repeat
            contexts_repeated = eo.repeat(
                contexts, "n ... -> (n r) ...", r=config.sample.num_samples_per_context
            )
            prompt_embeds_repeated = eo.repeat(
                prompt_embeds,
                "n ... -> (n r) ...",
                r=config.sample.num_samples_per_context,
            )

            # run sample loop
            rng, sample_rng = jax.random.split(rng)
            samples = sample_loop_jit(
                sample_rng,
                state,
                contexts_repeated,
                prompt_embeds_repeated,
                jnp.zeros_like(contexts_repeated),
                jnp.broadcast_to(uncond_prompt_embed, prompt_embeds_repeated.shape),
            )  # (num_contexts * num_samples_per_context, h, w, c)

            if config.vae:
                samples = jax.device_get(vae_decode(samples, scale=True))
                contexts = jax.device_get(vae_decode(contexts, scale=False))

            right = eo.rearrange(
                samples,
                "(n r) h w c -> (n h) (r w) c",
                r=config.sample.num_samples_per_context,
            )
            left = eo.rearrange(contexts, "n h w (x c) -> (n h) (x w) c", c=3)

            final_image = np.concatenate([left, right], axis=1)
            final_image = np.clip(np.round(final_image * 127.5 + 127.5), 0, 255).astype(
                np.uint8
            )

            if jax.process_index() == 0:
                prompts = untokenize(data["prompt_ids"])
                prompt_str = "; ".join(prompts)
                pil = Image.fromarray(final_image)
                with tf.io.gfile.GFile(
                    tf.io.gfile.join(logdir, f"{step + 1}.jpg"), "wb"
                ) as f:
                    pil.save(f, format="jpeg", quality=95)
                with tf.io.gfile.GFile(
                    tf.io.gfile.join(logdir, f"{step + 1}.txt"), "w"
                ) as f:
                    f.write(prompt_str)

                with tempfile.TemporaryDirectory() as tmpdir:
                    pil.save(os.path.join(tmpdir, "image.jpg"), quality=95)
                    wandb.log(
                        {
                            "samples": wandb.Image(
                                os.path.join(tmpdir, "image.jpg"), caption=prompt_str
                            )
                        },
                        step=step + 1,
                    )

        if (step + 1) % config.save_interval == 0:
            checkpointer.save(
                step + 1,
                {"state": state, "params_ema": state.params_ema},
                {
                    "state": {
                        "save_args": orbax_utils.save_args_from_target(state),
                    },
                    "params_ema": {
                        "save_args": orbax_utils.save_args_from_target(
                            state.params_ema
                        ),
                    },
                },
            )


if __name__ == "__main__":
    app.run(main)
