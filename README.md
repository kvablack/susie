# susie
Code for the paper [Zero-Shot Robotic Manipulation With Pretrained Image-Editing Diffusion Models](https://rail-berkeley.github.io/susie/).

This repository contains the code for training the high-level image-editing diffusion model on video data. For training the low-level policy, head over to the [BridgeData V2](https://github.com/rail-berkeley/bridge_data_v2) repository --- we use the `gc_ddpm_bc` agent, unmodified, with an action prediction horizon of 4 and the `delta_goals(min_delta=0, max_delta=20)` relabeling strategy.

- **Creating datasets**: this repo uses [dlimp](https://github.com/kvablack/dlimp) for dataloading. Check out the `scripts/` directory inside dlimp for creating TFRecords in a compatible format.
- **Installation**: `pip install -r requirements.txt` to install the versions of required packages confirmed to be working with this codebase. Then, `pip install -e .`. Only tested with Python 3.10.
- **Training**: once the missing dataset paths have been filled in inside `base.py`, you can start training by running `python scripts/train.py --config configs/base.py:base`.
- **Evaluation**: robot evaluation scripts are provided in the `scripts/robot` directory. You probably won't be able to run them, since you don't have our robot setup, but they are there for reference. See `create_sample_fn` in `susie/model.py` for canonical sampling code.

## Model Weights
The UNet weights for our best-performing model, trained on BridgeData and Something-Something for 40k steps, are hosted [on HuggingFace](https://huggingface.co/kvablack/susie). They can be loaded using `FlaxUNet2DConditionModel.from_pretrained("kvablack/susie", subfolder="unet")`. Use with the standard Stable Diffusion v1-5 VAE and text encoder.
