"""Unit tests for the pure-Python ray-caster."""

from __future__ import annotations

import numpy as np

from panda3d_rl_sim.sensors import (
    cpp_backend_available,
    ray_cast_aabb,
    ray_cast_aabb_fast,
)


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


def test_fast_backend_agrees_with_python():
    """The auto-selected backend must produce the same distances as the
    reference Python implementation (identity holds trivially when C++ is
    not built, and up to float rounding when it is)."""
    rng = np.random.default_rng(0)
    origin = rng.uniform(-2, 2, size=2).astype(np.float32)
    obstacles = np.array(
        [
            [1.0, -1.0, 2.0, 1.0],
            [-3.0, 0.0, -1.0, 2.0],
            [0.0, 2.0, 3.0, 3.0],
        ],
        dtype=np.float32,
    )
    a = ray_cast_aabb(origin, 0.3, 32, np.pi, 8.0, obstacles)
    b = ray_cast_aabb_fast(origin, 0.3, 32, np.pi, 8.0, obstacles)
    np.testing.assert_allclose(a, b, atol=1e-4)


def test_cpp_backend_flag_is_boolean():
    assert isinstance(cpp_backend_available(), bool)
