import datetime
import os
import time
from collections import deque

import imageio
import jax
import numpy as np
from absl import app, flags
from pyquaternion import Quaternion

# bridge_data_robot imports
from widowx_envs.widowx_env_service import WidowXClient, WidowXConfigs

import wandb
from susie.jax_utils import (
    deserialize_jax_fn,
    initialize_compilation_cache,
)

##############################################################################

STEP_DURATION = 0.2
NO_PITCH_ROLL = False
NO_YAW = False
STICKY_GRIPPER_NUM_STEPS = 2
ENV_PARAMS = {
    "camera_topics": [{"name": "/blue/image_raw", "flip": False}],
    "return_full_image": False,
    # forward, left, up
    # wallpaper
    # "override_workspace_boundaries": [
    #     [0.1, -0.15, 0.0, -1.57, 0],
    #     [0.60, 0.25, 0.18, 1.57, 0],
    # ],
    # toysink2
    # "override_workspace_boundaries": [
    #     [0.21, -0.13, 0.06, -1.57, 0],
    #     [0.36, 0.25, 0.18, 1.57, 0],
    # ],
    # microwave
    "override_workspace_boundaries": [
        [0.1, -0.15, 0.05, -1.57, 0],
        [0.31, 0.25, 0.23, 1.57, 0],
    ],
}

FIXED_STD = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


##############################################################################

np.set_printoptions(suppress=True)

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "policy_checkpoint", None, "Path to policy checkpoint", required=True
)
flags.DEFINE_string("video_save_path", None, "Path to save video")

flags.DEFINE_integer("num_timesteps", 120, "num timesteps")
flags.DEFINE_bool("blocking", True, "Use the blocking controller")

flags.DEFINE_spaceseplist("goal_eep", None, "Goal position")
flags.DEFINE_spaceseplist("initial_eep", None, "Initial position")

flags.DEFINE_string("ip", "localhost", "IP address of the robot")
flags.DEFINE_integer("port", 5556, "Port of the robot")


def state_to_eep(xyz_coor, zangle: float):
    """
    Implement the state to eep function.
    Refered to `bridge_data_robot`'s `widowx_controller/widowx_controller.py`
    return a 4x4 matrix
    """
    assert len(xyz_coor) == 3
    DEFAULT_ROTATION = np.array([[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]])
    new_pose = np.eye(4)
    new_pose[:3, -1] = xyz_coor
    new_quat = Quaternion(axis=np.array([0.0, 0.0, 1.0]), angle=zangle) * Quaternion(
        matrix=DEFAULT_ROTATION
    )
    new_pose[:3, :3] = new_quat.rotation_matrix
    # yaw, pitch, roll = quat.yaw_pitch_roll
    return new_pose


def rollout_subgoal(
    widowx_client, get_action, goal_obs, num_timesteps, obs_horizon, im_size
):
    is_gripper_closed = False
    num_consecutive_gripper_change_actions = 0

    last_tstep = time.time()
    images = []
    full_images = []
    t = 0
    actions = None
    rng = jax.random.PRNGKey(int(time.time()))
    if obs_horizon is not None:
        obs_hist = deque(maxlen=obs_horizon)
    try:
        while t < num_timesteps:
            if time.time() > last_tstep + STEP_DURATION or FLAGS.blocking:
                obs = widowx_client.get_observation()
                if obs is None:
                    print("WARNING retrying to get observation...")
                    continue

                full_images.append(obs["full_image"])

                obs = (
                    obs["image"].reshape(3, im_size, im_size).transpose(1, 2, 0) * 255
                ).astype(np.uint8)
                images.append(obs)

                # deal with obs history
                if obs_horizon is not None:
                    if len(obs_hist) == 0:
                        obs_hist.extend([obs] * obs_horizon)
                    else:
                        obs_hist.append(obs)
                    obs = np.stack(obs_hist)

                last_tstep = time.time()

                # deal with mutli-action prediction
                rng, key = jax.random.split(rng)
                pred_actions = jax.device_get(get_action(key, obs, goal_obs))
                if len(pred_actions.shape) == 1:
                    pred_actions = pred_actions[None]
                if actions is None:
                    actions = np.zeros_like(pred_actions)
                    weights = 1 / (np.arange(len(pred_actions)) + 1)
                else:
                    actions = np.concatenate([actions[1:], np.zeros_like(actions[-1:])])
                    weights = np.concatenate([weights[1:], [1 / len(weights)]])
                actions += pred_actions * weights[:, None]

                action = actions[0]

                # sticky gripper logic
                if (action[-1] < 0.5) != is_gripper_closed:
                    num_consecutive_gripper_change_actions += 1
                else:
                    num_consecutive_gripper_change_actions = 0

                if num_consecutive_gripper_change_actions >= STICKY_GRIPPER_NUM_STEPS:
                    is_gripper_closed = not is_gripper_closed
                    num_consecutive_gripper_change_actions = 0

                action[-1] = 0.0 if is_gripper_closed else 1.0

                # remove degrees of freedom
                if NO_PITCH_ROLL:
                    action[3] = 0
                    action[4] = 0
                if NO_YAW:
                    action[5] = 0

                action_norm = np.linalg.norm(action[:3])

                print(
                    f"Timestep {t}, action norm: {action_norm * 100:.1f}cm, gripper state: {action[-1]}"
                )
                widowx_client.step_action(action, blocking=FLAGS.blocking)

                t += 1
    except KeyboardInterrupt:
        return images, full_images, True
    return images, full_images, False


def main(_):
    initialize_compilation_cache()
    get_action = deserialize_jax_fn(FLAGS.policy_checkpoint)

    obs_horizon = get_action.args_info[0][1].aval.shape[0]
    im_size = get_action.args_info[0][1].aval.shape[1]

    print(f"obs horizon: {obs_horizon}, im size: {im_size}")

    # init environment
    env_params = WidowXConfigs.DefaultEnvParams.copy()
    env_params.update(ENV_PARAMS)
    widowx_client = WidowXClient(host=FLAGS.ip, port=FLAGS.port)

    # goal sampling loop
    image_goal = None
    while True:
        widowx_client.init(env_params)

        # ask for new goal
        if image_goal is None:
            print("Taking a new goal...")
            ch = "y"
        else:
            ch = input("Taking a new goal? [y/n]")
        if ch == "y":
            if FLAGS.goal_eep is not None:
                assert isinstance(FLAGS.goal_eep, list)
                goal_eep = [float(e) for e in FLAGS.goal_eep]
            else:
                # pick random goal eep
                low_bound = [0.24, -0.1, 0.05, -1.57, 0]
                high_bound = [0.4, 0.20, 0.15, 1.57, 0]
                goal_eep = np.random.uniform(low_bound[:3], high_bound[:3])
            widowx_client.move_gripper(1.0)  # open gripper
            widowx_client.move_gripper(1.0)  # open gripper

            print(f"Moving to goal position {goal_eep}")
            widowx_client.move(state_to_eep(goal_eep, 0), blocking=True)
            input("Press [Enter] when ready for taking the goal image. ")

            # take goal image
            obs = widowx_client.get_observation()
            while obs is None:
                print("WARNING retrying to get observation...")
                obs = widowx_client.get_observation()
                time.sleep(1)

            image_goal = (
                obs["image"].reshape(3, im_size, im_size).transpose(1, 2, 0) * 255
            ).astype(np.uint8)

        # move to initial position
        if FLAGS.initial_eep is not None:
            assert isinstance(FLAGS.initial_eep, list)
            initial_eep = [float(e) for e in FLAGS.initial_eep]
            print(f"Moving to initial position {initial_eep}")
            widowx_client.move_gripper(1.0)  # open gripper
            widowx_client.move_gripper(1.0)  # open gripper
            widowx_client.move(state_to_eep(initial_eep, 0), blocking=True)

        input("Press [Enter] to start.")

        # do rollout
        images, full_images, done = rollout_subgoal(
            widowx_client,
            get_action,
            image_goal,
            FLAGS.num_timesteps,
            obs_horizon,
            im_size,
        )

        if FLAGS.video_save_path is not None:
            save_path = os.path.join(
                FLAGS.video_save_path,
                datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S.mp4"),
            )
            images = np.array(images)
            video = np.concatenate(
                [np.broadcast_to(image_goal[None], images.shape), images], axis=1
            )
            imageio.mimsave(
                save_path,
                video,
                fps=3.0 / STEP_DURATION,
            )


if __name__ == "__main__":
    app.run(main)
