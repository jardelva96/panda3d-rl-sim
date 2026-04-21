"""Tests for vectorised environment helpers (make_vec_env).

AsyncVectorEnv tests are marked with ``vec_async`` and skipped by default on
CI because they spawn subprocesses which can be flaky on constrained runners.
Run them explicitly with ``pytest -m vec_async``.
"""

from __future__ import annotations

import numpy as np
import pytest

from panda3d_rl_sim import EnvConfig, make_vec_env

# ----------------------------------------------------------------- SyncVectorEnv

def test_sync_vec_reset_shape():
    venv = make_vec_env(3)
    obs, _info = venv.reset(seed=0)
    assert obs.shape == (3, 23)  # 3 envs × (7 pose + 16 LIDAR)
    assert obs.dtype == np.float32
    venv.close()


def test_sync_vec_step_shapes():
    venv = make_vec_env(2)
    venv.reset(seed=1)
    obs, rew, term, trunc, info = venv.step(venv.action_space.sample())
    assert obs.shape == (2, 23)
    assert rew.shape == (2,)
    assert term.shape == (2,)
    assert trunc.shape == (2,)
    venv.close()


def test_sync_vec_custom_config():
    cfg = EnvConfig(num_goals=2, num_obstacles=2)
    venv = make_vec_env(4, config=cfg)
    obs, _info = venv.reset(seed=0)
    assert obs.shape == (4, 23)
    venv.close()


def test_sync_vec_seeds_differ():
    """Different workers should produce different initial observations."""
    venv = make_vec_env(3)
    obs, _ = venv.reset(seed=7)
    venv.close()
    # At least two workers should start at different positions.
    assert not np.allclose(obs[0], obs[1])


def test_sync_vec_dr_config():
    cfg = EnvConfig(dr_num_obstacles=(1, 5), dr_max_speed=(2.0, 4.0))
    venv = make_vec_env(2, config=cfg)
    venv.reset(seed=0)
    venv.close()


# ----------------------------------------------------------------- AsyncVectorEnv

@pytest.mark.vec_async
def test_async_vec_reset_shape():
    venv = make_vec_env(2, async_envs=True)
    obs, _info = venv.reset(seed=0)
    assert obs.shape == (2, 23)
    assert obs.dtype == np.float32
    venv.close()


@pytest.mark.vec_async
def test_async_vec_step():
    venv = make_vec_env(2, async_envs=True)
    venv.reset(seed=0)
    obs, rew, term, trunc, info = venv.step(venv.action_space.sample())
    assert obs.shape == (2, 23)
    assert rew.shape == (2,)
    venv.close()


@pytest.mark.vec_async
def test_async_vec_multi_goal():
    cfg = EnvConfig(num_goals=3, num_obstacles=0)
    venv = make_vec_env(2, config=cfg, async_envs=True)
    obs, info = venv.reset(seed=5)
    assert obs.shape == (2, 23)
    venv.close()
