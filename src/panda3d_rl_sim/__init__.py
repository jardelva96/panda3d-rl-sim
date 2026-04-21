"""Reinforcement-learning simulation environment built on Panda3D."""

from gymnasium.envs.registration import register

from .config import EnvConfig
from .env import PandaNavEnv

__version__ = "0.1.0"
__all__ = ["PandaNavEnv", "EnvConfig", "__version__"]

register(
    id="Panda3D-Nav-v0",
    entry_point="panda3d_rl_sim.env:PandaNavEnv",
    max_episode_steps=None,  # env handles truncation internally
)
