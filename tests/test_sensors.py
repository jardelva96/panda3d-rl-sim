"""Unit tests for the pure-Python ray-caster."""

from __future__ import annotations

import numpy as np

from panda3d_rl_sim.sensors import ray_cast_aabb


def test_empty_scene_returns_max_range():
    out = ray_cast_aabb(
        origin=np.array([0.0, 0.0], dtype=np.float32),
        heading=0.0,
        num_rays=5,
        fov_rad=np.pi,
        max_range=10.0,
        obstacles=np.zeros((0, 4), dtype=np.float32),
    )
    assert out.shape == (5,)
    assert np.allclose(out, 10.0)


def test_single_forward_ray_hits_box():
    # A 1x1 box centered at x=5
    obstacles = np.array([[4.5, -0.5, 5.5, 0.5]], dtype=np.float32)
    out = ray_cast_aabb(
        origin=np.array([0.0, 0.0], dtype=np.float32),
        heading=0.0,
        num_rays=1,
        fov_rad=0.0,
        max_range=20.0,
        obstacles=obstacles,
    )
    assert out.shape == (1,)
    assert 4.4 < float(out[0]) < 4.6


def test_ray_behind_box_misses():
    obstacles = np.array([[-1.0, -1.0, 1.0, 1.0]], dtype=np.float32)
    out = ray_cast_aabb(
        origin=np.array([5.0, 0.0], dtype=np.float32),
        heading=0.0,  # pointing +x, away from the box
        num_rays=1,
        fov_rad=0.0,
        max_range=100.0,
        obstacles=obstacles,
    )
    assert float(out[0]) == 100.0


def test_fan_of_rays_shape():
    out = ray_cast_aabb(
        origin=np.array([0.0, 0.0], dtype=np.float32),
        heading=0.0,
        num_rays=16,
        fov_rad=np.pi,
        max_range=5.0,
        obstacles=np.array([[3.0, -0.5, 4.0, 0.5]], dtype=np.float32),
    )
    assert out.shape == (16,)
    # At least one ray in the middle of the fan should hit
    assert float(out.min()) < 5.0
