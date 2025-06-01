from pathlib import Path
from setuptools import find_packages, setup

core_requirements = [
    "gymnasium==0.29.1",
    "rich==13.7.1",
    "tqdm==4.66.4",
    "ipdb==0.13.13",
    "mujoco==3.1.6",
    "mujoco-mjx==3.1.6",
    "dm_control==1.0.20",
    "imageio==2.34.2",
    "gymnax==0.0.8",
    "brax==0.9.4",
    "torch==2.3.1",
    "opencv-python==4.10.0.84",
    "natsort==8.4.0",
]

setup(
    name="basketball_env",
    version="0.1",
    python_requires=">3.7",
    install_requires=core_requirements,
)
