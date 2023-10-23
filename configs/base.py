from copy import deepcopy

from ml_collections import ConfigDict


def base():
    config = ConfigDict()

    # top-level stuff
    config.seed = 42
    config.wandb_project = "susie"
    config.run_name = ""
    config.logdir = "logs"
    config.num_steps = 40000
    config.log_interval = 100
    config.save_interval = 5000
    config.val_interval = 2500
    config.sample_interval = 2500
    config.num_val_batches = 128
    config.goal_drop_rate = 1.0
    config.curr_drop_rate = 0.0
    config.prompt_drop_rate = 0.0
    config.mesh = [-1, 1]  # dp, fsdp

    config.wandb_resume_id = None

    config.vae = "runwayml/stable-diffusion-v1-5:flax"
    config.text_encoder = "runwayml/stable-diffusion-v1-5:flax"

    # ema
    config.ema = ema = ConfigDict()
    ema.max_decay = 0.999
    ema.min_decay = 0.999
    ema.update_every = 1
    ema.start_step = 0
    ema.inv_gamma = 1.0
    ema.power = 3 / 4

    # optim
    config.optim = optim = ConfigDict()
    optim.optimizer = "adamw"
    optim.lr = 1e-4
    optim.warmup_steps = 800  # linear warmup steps
    optim.decay_steps = 1e9  # cosine decay total steps (reaches 0 at this number)
    optim.weight_decay = (
        1e-2  # adamw weight decay -- pytorch default (which instructpix2pix and SD use)
    )
    optim.beta1 = 0.9
    optim.beta2 = 0.999
    optim.epsilon = 1e-8
    optim.max_grad_norm = 1.0
    optim.accumulation_steps = 1

    # scheduling
    config.scheduling = scheduling = ConfigDict()
    scheduling.noise_schedule = "scaled_linear"

    # sampling
    config.sample = sample = ConfigDict()
    sample.num_contexts = 8
    sample.num_samples_per_context = 8
    sample.num_steps = 50
    sample.context_w = 2.5
    sample.prompt_w = 7.5
    sample.eta = 0.0

    # data
    config.data = ConfigDict()
    config.data.batch_size = 128

    data_base = ConfigDict()
    data_base.image_size = 256
    data_base.shuffle_buffer_size = 100000
    data_base.augment_kwargs = dict(
        random_resized_crop=dict(scale=[0.85, 1.0], ratio=[0.95, 1.05]),
        random_brightness=[0.05],
        random_contrast=[0.95, 1.05],
        random_saturation=[0.95, 1.05],
        random_hue=[0.025],
        augment_order=[
            # "random_flip_left_right",
            # "random_resized_crop",
            # "random_brightness",
            # "random_contrast",
            # "random_saturation",
            # "random_hue",
            # "random_flip_up_down",
            # "random_rot90",
        ],
    )

    # config.data.ego4d = ego4d = deepcopy(data_base)
    # ego4d.weight = 70.0
    # ego4d.data_path = ""
    # ego4d.goal_relabeling_fn = "subgoal_only"
    # ego4d.goal_relabeling_kwargs = dict(
    #     subgoal_delta=(30, 60),
    #     truncate=True,
    # )

    config.data.bridge = bridge = deepcopy(data_base)
    bridge.weight = 45.0
    bridge.data_path = ""
    bridge.goal_relabeling_fn = "subgoal_only"
    bridge.goal_relabeling_kwargs = dict(
        subgoal_delta=(11, 14),
        truncate=False,
    )

    # config.data.calvin = calvin = deepcopy(data_base)
    # calvin.weight = 15.0
    # calvin.data_path = ""
    # calvin.goal_relabeling_fn = "subgoal_only"
    # calvin.goal_relabeling_kwargs = dict(
    #     subgoal_delta=(20, 21),
    #     truncate=False,
    # )

    config.data.somethingsomething = somethingsomething = deepcopy(data_base)
    somethingsomething.weight = 75.0
    somethingsomething.data_path = ""
    somethingsomething.goal_relabeling_fn = "subgoal_only"
    somethingsomething.goal_relabeling_kwargs = dict(
        subgoal_delta=(11, 14),
        truncate=False,
    )

    # model
    config.model = model = ConfigDict()
    config.model.pretrained = "kvablack/instruct-pix2pix-flax"

    return config


def debug():
    config = base()
    config.logdir = "logs"
    config.log_interval = 150
    config.save_interval = 150
    config.val_interval = 150
    config.sample_interval = 10
    config.num_val_batches = 4

    config.vae = "runwayml/stable-diffusion-v1-5:flax"
    config.text_encoder = "runwayml/stable-diffusion-v1-5:flax"

    config.sample.num_contexts = 4
    config.sample.num_samples_per_context = 4
    config.sample.num_steps = 20
    config.sample.w = 1.0

    config.data.batch_size = 16
    for data in [d for d in config.data.values() if isinstance(d, ConfigDict)]:
        data.shuffle_buffer_size = 100
        # data.image_size = 32

    config.model.pretrained = None
    config.model.block_out_channels = (32, 32)
    config.model.down_block_types = (
        "DownBlock2D",
        "DownBlock2D",
        # "DownBlock2D",
        # "CrossAttnDownBlock2D",
        # "DownBlock2D",
    )
    config.model.up_block_types = (
        "UpBlock2D",
        "UpBlock2D",
        # "CrossAttnUpBlock2D",
        # "UpBlock2D",
        # "UpBlock2D",
    )
    config.model.layers_per_block = 1
    config.model.attention_head_dim = 1

    return config


def get_config(name):
    return globals()[name]()
