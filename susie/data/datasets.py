from functools import partial
from typing import Any, Dict, List

import dlimp as dl
import numpy as np
import tensorflow as tf
from jax.experimental import multihost_utils
from jax.sharding import PartitionSpec as P

from . import goal_relabeling


class Transforms:
    """Trajectory-level transforms for each dataset"""

    @staticmethod
    def ego4d(x: Dict[str, Any]) -> Dict[str, Any]:
        return x

    @staticmethod
    def bridge(x: Dict[str, Any]) -> Dict[str, Any]:
        CAMERA_VIEWS = {"images0", "images1", "images2"}
        # pick a random camera view
        views = tf.stack([x["obs"][k] for k in CAMERA_VIEWS])
        lengths = tf.stack([tf.strings.length(x["obs"][k][0]) for k in CAMERA_VIEWS])
        views = views[lengths > 0]
        idx = tf.random.uniform([], minval=0, maxval=tf.shape(views)[0], dtype=tf.int32)
        x["obs"] = views[idx]
        # x["obs"] = x["obs"]["images0"]

        del x["actions"]
        return x

    @staticmethod
    def calvin(x: Dict[str, Any]) -> Dict[str, Any]:
        x["obs"] = x.pop("image_states")
        x["lang"] = x.pop("language_annotation")

        del x["actions"]
        del x["proprioceptive_states"]

        return x

    @staticmethod
    def somethingsomething(x: Dict[str, Any]) -> Dict[str, Any]:
        return x


class GetPaths:
    """Retrieves paths to TFRecord files or each dataset"""

    @staticmethod
    def ego4d(data_path: str, train: bool) -> str:
        return f"{data_path}/{'train' if train else 'val'}"

    @staticmethod
    def bridge(data_path: str, train: bool) -> str:
        return f"{data_path}/{'train' if train else 'val'}"

    @staticmethod
    def somethingsomething(data_path: str, train: bool) -> List[str]:
        return f"{data_path}/{'train' if train else 'val'}"

    @staticmethod
    def calvin(data_path: str, train: bool) -> List[str]:
        if train:
            return (
                tf.io.gfile.glob(f"{data_path}/training/A/*")
                + tf.io.gfile.glob(f"{data_path}/training/B/*")
                + tf.io.gfile.glob(f"{data_path}/training/C/*")
            )
        else:
            return tf.io.gfile.glob(f"{data_path}/validation/D/*")


def make_dataset(
    name: str,
    data_path: str,
    image_size: int,
    shuffle_buffer_size: int,
    train: bool,
    goal_relabeling_fn: str,
    goal_relabeling_kwargs: dict = {},
    augment_kwargs: dict = {},
) -> dl.DLataset:
    paths = getattr(GetPaths, name)(data_path, train)

    dataset = (
        dl.DLataset.from_tfrecords(paths)
        .map(dl.transforms.unflatten_dict)
        .map(getattr(Transforms, name))
        .filter(lambda x: tf.math.reduce_all(x["lang"] != ""))
        .apply(
            partial(
                getattr(goal_relabeling, goal_relabeling_fn), **goal_relabeling_kwargs
            ),
        )
        .unbatch()
        .shuffle(shuffle_buffer_size)
    )

    dataset = dataset.map(
        partial(dl.transforms.decode_images, match=["curr", "goals", "subgoals"])
    ).map(
        partial(
            dl.transforms.resize_images,
            match=["curr", "goals", "subgoals"],
            size=(image_size, image_size),
        )
    )

    if train:
        dataset = dataset.map(
            partial(
                dl.transforms.augment,
                traj_identical=False,
                keys_identical=True,
                match=["curr", "goals", "subgoals"],
                augment_kwargs=augment_kwargs,
            )
        )

    # normalize images to [-1, 1]
    dataset = dataset.map(
        partial(
            dl.transforms.selective_tree_map,
            match=["curr", "goals", "subgoals"],
            map_fn=lambda v: v / 127.5 - 1.0,
        )
    )

    return dataset.repeat()


def get_data_loader(data_config, tokenize_fn, mesh=None):
    data_config = dict(data_config)
    batch_size = data_config.pop("batch_size")

    train_datasets = []
    val_datasets = []
    weights = []
    for data_name, data_kwargs in data_config.items():
        data_kwargs = dict(data_kwargs)
        weights.append(float(data_kwargs.pop("weight")))
        train_datasets.append(make_dataset(data_name, train=True, **data_kwargs))
        val_datasets.append(make_dataset(data_name, train=False, **data_kwargs))

    train = dl.DLataset.sample_from_datasets(
        train_datasets, weights=weights, stop_on_empty_dataset=True
    ).batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE)
    val = dl.DLataset.sample_from_datasets(
        val_datasets, weights=weights, stop_on_empty_dataset=True
    ).batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE)

    def shard(batch):
        return multihost_utils.host_local_array_to_global_array(
            batch,
            mesh,
            P(("dp", "fsdp")),
        )

    # WARNING: for some reason any amount of prefetching is also a total no-go in terms of memory usage...
    train = map(tokenize_fn, train.as_numpy_iterator())
    val = map(tokenize_fn, val.as_numpy_iterator())

    if mesh:
        return map(shard, train), map(shard, val), len(train_datasets)
    else:
        return train, val, len(train_datasets)
