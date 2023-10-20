# susie
Code for the paper [Zero-Shot Robotic Manipulation With Pretrained Image-Editing Diffusion Models](https://rail-berkeley.github.io/susie/).

This repository contains the code for training the high-level image-editing diffusion model on video data. For training the low-level policy, head over to the [BridgeData V2](https://github.com/rail-berkeley/bridge_data_v2) repository --- we use the `gc_ddpm_bc` agent, unmodified, with an action prediction horizon of 4 and the `delta_goals` relabeling strategy.

For integration with the CALVIN simulator and reproducing our simulated results, see [our fork of the calvin-sim repo](https://github.com/pranavatreya/calvin-sim) and the [corresponding documentation in the BridgeData V2 repository](https://github.com/rail-berkeley/bridge_data_v2/tree/main/experiments/susie/calvin).

- **Creating datasets**: this repo uses [dlimp](https://github.com/kvablack/dlimp) for dataloading. Check out the `scripts/` directory inside dlimp for creating TFRecords in a compatible format.
- **Installation**: `pip install -r requirements.txt` to install the versions of required packages confirmed to be working with this codebase. Then, `pip install -e .`. Only tested with Python 3.10. You'll also have to manually install Jax for your platform (see the [Jax installation instructions](https://jax.readthedocs.io/en/latest/installation.html)). Make sure you have the Jax version specified in `requirements.txt` (rather than using `--upgrade` as suggested in the Jax docs).
- **Training**: once the missing dataset paths have been filled in inside `base.py`, you can start training by running `python scripts/train.py --config configs/base.py:base`.
- **Evaluation**: robot evaluation scripts are provided in the `scripts/robot` directory. You probably won't be able to run them, since you don't have our robot setup, but they are there for reference. See `create_sample_fn` in `susie/model.py` for canonical sampling code.

## Model Weights
The UNet weights for our best-performing model, trained on BridgeData and Something-Something for 40k steps, are hosted [on HuggingFace](https://huggingface.co/kvablack/susie). They can be loaded using `FlaxUNet2DConditionModel.from_pretrained("kvablack/susie", subfolder="unet")`. Use with the standard Stable Diffusion v1-5 VAE and text encoder.

Here's a quickstart for getting out-of-the-box subgoals using this repo:
```python
from susie.model import create_sample_fn
from susie.jax_utils import initialize_compilation_cache
import requests
import numpy as np
from PIL import Image

initialize_compilation_cache()

IMAGE_URL = "https://rail.eecs.berkeley.edu/datasets/bridge_release/raw/bridge_data_v2/datacol2_toykitchen7/drawer_pnp/01/2023-04-19_09-18-15/raw/traj_group0/traj0/images0/im_12.jpg"

sample_fn = create_sample_fn("kvablack/susie")
image = np.array(Image.open(requests.get(IMAGE_URL, stream=True).raw).resize((256, 256)))
image_out = sample_fn(image, "open the drawer")
display(Image.fromarray(image_out))
```
