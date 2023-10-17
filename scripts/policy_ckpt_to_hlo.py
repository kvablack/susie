import inspect

import jax
import numpy as np
import orbax.checkpoint
import tensorflow as tf
from absl import app, flags
from jaxrl_m.agents import agents
from jaxrl_m.vision import encoders

import wandb
from susie.jax_utils import serialize_jax_fn

FLAGS = flags.FLAGS

flags.DEFINE_string("checkpoint_path", None, "Path to checkpoint file", required=True)

flags.DEFINE_string(
    "wandb_run_name", None, "Name of wandb run to get config from.", required=True
)

flags.DEFINE_string(
    "outpath", None, "Path to save serialized policy to.", required=True
)

flags.DEFINE_integer(
    "im_size", 256, "Image size, which was unfortunately not saved in config"
)


def load_policy_checkpoint(path, wandb_run_name):
    assert tf.io.gfile.exists(path)

    # load information from wandb
    api = wandb.Api()
    run = api.run(wandb_run_name)
    config = run.config

    # create encoder from wandb config
    encoder_def = encoders[config["encoder"]](**config["encoder_kwargs"])

    act_pred_horizon = run.config["dataset_kwargs"].get("act_pred_horizon")
    obs_horizon = run.config.get("obs_horizon") or run.config["dataset_kwargs"].get(
        "obs_horizon"
    )

    if act_pred_horizon is not None:
        example_actions = np.zeros((1, act_pred_horizon, 7), dtype=np.float32)
    else:
        example_actions = np.zeros((1, 7), dtype=np.float32)

    if obs_horizon is not None:
        example_obs = {
            "image": np.zeros(
                (1, obs_horizon, FLAGS.im_size, FLAGS.im_size, 3), dtype=np.uint8
            )
        }
    else:
        example_obs = {
            "image": np.zeros((1, FLAGS.im_size, FLAGS.im_size, 3), dtype=np.uint8)
        }

    example_goal = {
        "image": np.zeros((1, FLAGS.im_size, FLAGS.im_size, 3), dtype=np.uint8)
    }

    example_batch = {
        "observations": example_obs,
        "actions": example_actions,
        "goals": example_goal,
    }

    # create agent from wandb config
    agent = agents[config["agent"]].create(
        rng=jax.random.PRNGKey(0),
        encoder_def=encoder_def,
        observations=example_batch["observations"],
        goals=example_batch["goals"],
        actions=example_batch["actions"],
        **config["agent_kwargs"],
    )

    # load action metadata from wandb
    # action_metadata = config["bridgedata_config"]["action_metadata"]
    # action_mean = np.array(action_metadata["mean"])
    # action_std = np.array(action_metadata["std"])

    # load action metadata from wandb
    action_proprio_metadata = run.config["bridgedata_config"]["action_proprio_metadata"]
    action_mean = np.array(action_proprio_metadata["action"]["mean"])
    action_std = np.array(action_proprio_metadata["action"]["std"])

    # hydrate agent with parameters from checkpoint
    agent = orbax.checkpoint.PyTreeCheckpointer().restore(
        path,
        item=agent,
    )

    def get_action(rng, obs_image, goal_image):
        obs = {"image": obs_image}
        goal_obs = {"image": goal_image}
        # some agents (e.g. DDPM) don't have argmax
        if inspect.signature(agent.sample_actions).parameters.get("argmax"):
            action = agent.sample_actions(obs, goal_obs, seed=rng, argmax=True)
        else:
            action = agent.sample_actions(obs, goal_obs, seed=rng)
        action = action * action_std + action_mean
        return action

    serialized = serialize_jax_fn(
        get_action,
        jax.random.PRNGKey(0),
        example_obs["image"][0],
        example_goal["image"][0],
    )

    return serialized


def main(_):
    serialized = load_policy_checkpoint(FLAGS.checkpoint_path, FLAGS.wandb_run_name)

    with open(FLAGS.outpath, "wb") as f:
        f.write(serialized)


if __name__ == "__main__":
    app.run(main)
