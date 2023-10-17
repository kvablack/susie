import tensorflow as tf


def tail_goals(ds, *, tail_proportion, subgoal_delta):
    """
    Relabels for subgoal training. Removes the `obs` key and adds `subgoals`, `curr`, and `goals` keys.

    The "goal" is selected from the last `tail_proportion` proportion of the trajectory. The "current obs" is selected
    from [0, len * (1 - tail_proportion) - subgoal_delta[0]). The "subgoal" is selected from [curr + subgoal_delta[0],
    min{curr + subgoal_delta[1], goal}).
    """
    assert len(subgoal_delta) == 2

    def filter_fn(traj):
        num_frames = tf.shape(traj["obs"])[0]
        n = tf.cast(
            tf.math.ceil(tf.cast(num_frames, tf.float32) * tail_proportion),
            tf.int32,
        )
        return num_frames > n + subgoal_delta[0]

    def map_fn(traj):
        num_frames = tf.shape(traj["obs"])[0]

        n = tf.cast(
            tf.math.ceil(tf.cast(num_frames, tf.float32) * tail_proportion),
            tf.int32,
        )

        # select the last n transitions to be goals: [len - n, len)
        goal_idxs = tf.range(num_frames - n, num_frames)
        goals = tf.gather(traj["obs"], goal_idxs, name="tail_1")

        # for each goal, select a random state from [0, len - n - subgoal_delta[0])
        rand = tf.random.uniform([n])
        high = tf.cast(num_frames - n - subgoal_delta[0], tf.float32)
        curr_idxs = tf.cast(rand * high, tf.int32)
        curr = tf.gather(traj["obs"], curr_idxs, name="tail_2")

        # for each (curr, goal) pair, select a random subgoal from [curr + subgoal_delta[0], min{curr +
        # subgoal_delta[1], goal})
        rand = tf.random.uniform([n])
        low = tf.cast(curr_idxs + subgoal_delta[0], tf.float32)
        high = tf.cast(tf.minimum(curr_idxs + subgoal_delta[1], goal_idxs), tf.float32)
        subgoal_idxs = tf.cast(low + rand * (high - low), tf.int32)
        subgoals = tf.gather(traj["obs"], subgoal_idxs, name="tail_3")

        extras = {k: v[-n:] for k, v in traj.items() if k != "obs"}

        return {"subgoals": subgoals, "curr": curr, "goals": goals, **extras}

    return ds.filter(filter_fn).map(map_fn)


def delta_goals(ds, *, goal_delta, subgoal_delta):
    """
    Relabels for subgoal training. Removes the `obs` key and adds `subgoals`, `curr`, and `goals` keys.

    The "current obs" is selected from [0, len - goal_delta[0]). The "goal" is then selected from [curr +
    goal_delta[0], min{curr + goal_delta[1], len}).

    The "subgoal" is selected from [curr + subgoal_delta[0], min{curr + subgoal_delta[1], goal}).
    """
    assert len(subgoal_delta) == 2
    assert len(goal_delta) == 2

    def filter_fn(traj):
        num_frames = tf.shape(traj["obs"])[0]
        n = num_frames - goal_delta[0]
        return n >= 1

    def map_fn(traj):
        num_frames = tf.shape(traj["obs"])[0]
        n = num_frames - goal_delta[0]

        # select [0, len - goal_delta[0]) to be the current obs
        curr_idxs = tf.range(n)
        curr = tf.gather(traj["obs"], curr_idxs, name="delta_1")

        # for each current obs, select a random goal from [curr + goal_delta[0], min{curr + goal_delta[1], len})
        rand = tf.random.uniform([n])
        low = tf.cast(curr_idxs + goal_delta[0], tf.float32)
        high = tf.cast(tf.minimum(curr_idxs + goal_delta[1], num_frames), tf.float32)
        goal_idxs = tf.cast(low + rand * (high - low), tf.int32)
        goal_idxs = tf.clip_by_value(goal_idxs, 0, num_frames - 1)
        goals = tf.gather(traj["obs"], goal_idxs, name="delta_2")

        # for each (curr, goal) pair, select a random subgoal from [curr + subgoal_delta[0], min{curr + subgoal_delta[1], goal})
        rand = tf.random.uniform([n])
        low = tf.cast(curr_idxs + subgoal_delta[0], tf.float32)
        high = tf.cast(tf.minimum(curr_idxs + subgoal_delta[1], goal_idxs), tf.float32)
        subgoal_idxs = tf.cast(low + rand * (high - low), tf.int32)
        subgoals = tf.gather(traj["obs"], subgoal_idxs, name="delta_3")

        extras = {k: v[:n] for k, v in traj.items() if k != "obs"}

        return {"subgoals": subgoals, "curr": curr, "goals": goals, **extras}

    return ds.filter(filter_fn).map(map_fn)


def subgoal_only(ds, *, subgoal_delta, truncate=False):
    """
    Relabels for subgoal training. Removes the `obs` key and adds `subgoals` and `curr` keys.

    If truncate == False:
        The "current obs" is selected from [0, len). The "subgoal" is then selected
        from [min{curr + subgoal_delta[0], len - 1}, min{curr + subgoal_delta[1], len}).
    else:
        The "current obs" is selected from [0, len - subgoal_delta[0]). The "subgoal" is then selected
        from [curr + subgoal_delta[0], min{curr + subgoal_delta[1], len}).

    """
    assert len(subgoal_delta) == 2

    def filter_fn(traj):
        num_frames = tf.shape(traj["obs"])[0]
        n = num_frames - subgoal_delta[0]
        return n >= 1

    if truncate:

        def map_fn(traj):
            num_frames = tf.shape(traj["obs"])[0]
            n = num_frames - subgoal_delta[0]

            # select [0, len - subgoal_delta[0]) to be the current obs
            curr_idxs = tf.range(n)
            curr = tf.gather(traj["obs"], curr_idxs, name="subdelta_1")

            # for each current obs, select a random subgoal from [curr + subgoal_delta[0], min{curr + subgoal_delta[1],
            # len})
            rand = tf.random.uniform([n])
            low = tf.cast(curr_idxs + subgoal_delta[0], tf.float32)
            high = tf.cast(
                tf.minimum(curr_idxs + subgoal_delta[1], num_frames), tf.float32
            )
            subgoal_idxs = tf.cast(low + rand * (high - low), tf.int32)
            subgoal_idxs = tf.clip_by_value(subgoal_idxs, 0, num_frames - 1)
            subgoals = tf.gather(traj["obs"], subgoal_idxs, name="subdelta_2")

            extras = {k: v[:n] for k, v in traj.items() if k != "obs"}

            return {"subgoals": subgoals, "curr": curr, **extras}

    else:

        def map_fn(traj):
            num_frames = tf.shape(traj["obs"])[0]

            # select [0, len) to be the current obs
            curr_idxs = tf.range(num_frames)
            curr = traj["obs"]

            # for each current obs, select a random subgoal from [min{curr +
            # subgoal_delta[0], len - 1}, min{curr + subgoal_delta[1], len})
            rand = tf.random.uniform([num_frames])
            low = tf.cast(
                tf.minimum(curr_idxs + subgoal_delta[0], num_frames - 1), tf.float32
            )
            high = tf.cast(
                tf.minimum(curr_idxs + subgoal_delta[1], num_frames), tf.float32
            )
            subgoal_idxs = tf.cast(low + rand * (high - low), tf.int32)
            subgoal_idxs = tf.clip_by_value(subgoal_idxs, 0, num_frames - 1)
            subgoals = tf.gather(traj["obs"], subgoal_idxs, name="subdelta_2")

            extras = {k: v for k, v in traj.items() if k != "obs"}

            return {"subgoals": subgoals, "curr": curr, **extras}

    return ds.filter(filter_fn).map(map_fn)


def uniform(ds, *, dist_norm):
    """
    Relabels with a true uniform distribution over future states.
    """

    def map_fn(traj):
        traj_len = tf.shape(tf.nest.flatten(traj["obs"])[0])[0]

        # select a random future index for each transition i in the range [i + 1, traj_len)
        rand = tf.random.uniform([traj_len])
        low = tf.cast(tf.range(traj_len) + 1, tf.float32)
        high = tf.cast(traj_len, tf.float32)
        goal_idxs = tf.cast(rand * (high - low) + low, tf.int32)

        # sometimes there are floating-point errors that cause an out-of-bounds
        goal_idxs = tf.minimum(goal_idxs, traj_len - 1)

        traj["goals"] = tf.gather(traj["obs"], goal_idxs, name="uniform_1")
        traj["dists"] = goal_idxs - tf.range(traj_len)
        traj["curr"] = traj.pop("obs")

        traj["dists"] = 2 * tf.cast(traj["dists"], tf.float32) / dist_norm - 1

        return traj

    return ds.map(map_fn)
