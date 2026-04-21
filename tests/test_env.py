"""Smoke tests for PandaNavEnv in headless (state) mode.

These tests avoid graphics entirely so they pass on CI runners without a
display or GPU.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

import panda3d_rl_sim  # noqa: F401 — triggers env registration
from panda3d_rl_sim import EnvConfig

EXPECTED_STATE_DIM = 7 + EnvConfig().num_rays


@pytest.fixture()
def env():
    e = gym.make("Panda3D-Nav-v0", observation_mode="state")
    yield e
    e.close()


@pytest.fixture()
def empty_env():
    """Env with no obstacles — for deterministic behavioral tests."""
    e = gym.make(
        "Panda3D-Nav-v0",
        observation_mode="state",
        config=EnvConfig(num_obstacles=0),
    )
    yield e
    e.close()


def test_reset_returns_valid_state(env):
    obs, info = env.reset(seed=42)
    assert obs.shape == (EXPECTED_STATE_DIM,)
    assert obs.dtype == np.float32
    assert np.all(np.isfinite(obs))
    assert info["distance_to_goal"] > 0


def test_step_shapes_and_types(env):
    env.reset(seed=0)
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs.shape == (EXPECTED_STATE_DIM,)
    assert np.isfinite(reward)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_action_clipping(env):
    env.reset(seed=0)
    huge = np.array([10.0, -10.0], dtype=np.float32)
    env.step(huge)  # should not raise — env clips internally


def test_full_episode_terminates(env):
    env.reset(seed=1)
    for _ in range(10_000):
        _obs, _r, terminated, truncated, _info = env.step(
            np.zeros(2, dtype=np.float32)
        )
        if terminated or truncated:
            break
    assert terminated or truncated


def test_determinism(env):
    obs1, _ = env.reset(seed=123)
    obs2, _ = env.reset(seed=123)
    np.testing.assert_allclose(obs1, obs2)


def test_distance_decreases_under_simple_controller(empty_env):
    """A closed-loop P-controller should close the distance to the goal."""
    obs, info = empty_env.reset(seed=7)
    start_dist = info["distance_to_goal"]
    for _ in range(80):
        _x, _y, cos_h, sin_h, gdx, gdy, _d = obs[:7]
        heading = np.arctan2(sin_h, cos_h)
        target = np.arctan2(gdy, gdx)
        err = (target - heading + np.pi) % (2 * np.pi) - np.pi
        turn = float(np.clip(err * 2.0, -1.0, 1.0))
        forward = 1.0 if abs(err) < 0.3 else 0.0
        action = np.array([forward, turn], dtype=np.float32)
        obs, _r, terminated, truncated, info = empty_env.step(action)
        if terminated or truncated:
            break
    assert info["distance_to_goal"] < start_dist


def test_obstacles_are_spawned(env):
    _obs, info = env.reset(seed=2)
    # Default config requests 4 obstacles; rejection sampling may place fewer
    # on very tight layouts but we should get at least one.
    assert info["num_obstacles"] >= 1
    assert info["num_obstacles"] <= EnvConfig().num_obstacles


def test_lidar_distances_are_in_range(env):
    obs, _info = env.reset(seed=3)
    lidar = obs[7:]
    cfg = EnvConfig()
    assert lidar.shape == (cfg.num_rays,)
    assert np.all(lidar >= 0.0)
    assert np.all(lidar <= cfg.lidar_max_range + 1e-4)


def test_empty_env_has_max_range_lidar(empty_env):
    """With no obstacles every ray should return max_range."""
    obs, _info = empty_env.reset(seed=4)
    lidar = obs[7:]
    cfg = EnvConfig()
    np.testing.assert_allclose(lidar, cfg.lidar_max_range)
