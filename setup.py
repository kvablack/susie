from setuptools import find_packages, setup

setup(
    name="susie",
    packages=find_packages(),
    version="0.0.1",
    install_requires=[
        "absl-py",
        "diffusers[flax]",
        "ml_collections",
        "tensorflow",
        "wandb",
        "einops",
    ],
)
