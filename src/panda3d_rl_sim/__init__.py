"""Reinforcement-learning simulation environment built on Panda3D."""

from __future__ import annotations

import functools

from gymnasium.envs.registration import register

from .config import EnvConfig
from .env import PandaNavEnv

__version__ = "0.6.0"
__all__ = ["PandaNavEnv", "EnvConfig", "make_vec_env", "__version__"]

register(
    id="Panda3D-Nav-v0",
    entry_point="panda3d_rl_sim.env:PandaNavEnv",
    max_episode_steps=None,  # env handles truncation internally
)


def _make_single_env(config: EnvConfig | None, render_mode: str | None) -> PandaNavEnv:
    """Top-level picklable factory used by make_vec_env worker processes."""
    # Importing here ensures the env is registered even in spawn subprocesses.
    import panda3d_rl_sim  # noqa: F401 — side-effect: registers the env
    return PandaNavEnv(config=config, render_mode=render_mode)


def make_vec_env(
    n_envs: int,
    config: EnvConfig | None = None,
    render_mode: str | None = None,
    *,
    async_envs: bool = False,
    context: str = "spawn",
):
    """Create a vectorised batch of :class:`PandaNavEnv` environments.

    Parameters
    ----------
    n_envs:
        Number of parallel environment copies.
    config:
        Shared :class:`EnvConfig` applied to every worker.  A default config
        is used when *None*.
    render_mode:
        Passed through to each worker env.  Use ``None`` (headless) for
        training.
    async_envs:
        When ``True`` each worker runs in a separate subprocess
        (``AsyncVectorEnv``).  When ``False`` (default) all workers share the
        calling process (``SyncVectorEnv``).
    context:
        Multiprocessing start context used by ``AsyncVectorEnv``.  ``"spawn"``
        is the safest default on all platforms and is required on Windows.

    Returns
    -------
    gymnasium.vector.VectorEnv
    """
    from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv

    factories = [
        functools.partial(_make_single_env, config, render_mode)
        for _ in range(n_envs)
    ]
    if async_envs:
        return AsyncVectorEnv(factories, context=context)
    return SyncVectorEnv(factories)
