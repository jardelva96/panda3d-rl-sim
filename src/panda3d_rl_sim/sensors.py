"""
Pure-Python 2D ray-casting against axis-aligned bounding boxes.

This is the fallback implementation used when the optional C++ extension in
``cpp/`` is not built. Both implementations expose the same signature, so
they are drop-in interchangeable.
"""

from __future__ import annotations

import numpy as np


def ray_cast_aabb(
    origin: np.ndarray,
    heading: float,
    num_rays: int,
    fov_rad: float,
    max_range: float,
    obstacles: np.ndarray,
) -> np.ndarray:
    """Cast ``num_rays`` 2-D rays and return the nearest-hit distance for each.

    Parameters
    ----------
    origin : (2,) float array
        Ray origin in world space.
    heading : float
        Center heading of the fan, in radians.
    num_rays : int
        Number of evenly spaced rays across the FOV.
    fov_rad : float
        Total angular field of view, in radians.
    max_range : float
        Maximum ray length; also the value returned when a ray misses.
    obstacles : (N, 4) float array
        Each row is ``(x_min, y_min, x_max, y_max)``.

    Returns
    -------
    (num_rays,) float32 array
        Distance to the nearest hit, or ``max_range`` if the ray misses.
    """
    if num_rays <= 0:
        return np.zeros(0, dtype=np.float32)

    if num_rays == 1:
        angles = np.array([heading], dtype=np.float32)
    else:
        half = fov_rad * 0.5
        angles = heading + np.linspace(-half, half, num_rays, dtype=np.float32)

    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    out = np.full(num_rays, max_range, dtype=np.float32)

    obstacles = np.asarray(obstacles, dtype=np.float32)
    if obstacles.size == 0:
        return out

    ox, oy = float(origin[0]), float(origin[1])
    for i in range(num_rays):
        dx, dy = float(cos_a[i]), float(sin_a[i])
        best = max_range
        for row in obstacles:
            t = _ray_aabb(ox, oy, dx, dy, row[0], row[1], row[2], row[3], best)
            if t < best:
                best = t
        out[i] = best
    return out


def _ray_aabb(ox, oy, dx, dy, x_min, y_min, x_max, y_max, best):
    """Slab-method intersection with early-out. Returns ``best`` on miss."""
    tmin = 0.0
    tmax = best
    for o, d, lo, hi in ((ox, dx, x_min, x_max), (oy, dy, y_min, y_max)):
        if abs(d) < 1e-9:
            if o < lo or o > hi:
                return best
            continue
        inv = 1.0 / d
        t1 = (lo - o) * inv
        t2 = (hi - o) * inv
        if t1 > t2:
            t1, t2 = t2, t1
        if t1 > tmin:
            tmin = t1
        if t2 < tmax:
            tmax = t2
        if tmin > tmax:
            return best
    return tmin if tmin > 0 else best


# ---------------------------------------------------------- backend selection
#
# ``ray_cast_aabb_fast`` is the recommended entry point for hot paths: it uses
# the compiled C++ extension when it is importable, and falls back to the pure
# Python implementation otherwise. The two implementations produce identical
# results up to float rounding, so they are drop-in interchangeable.

try:
    from _raycaster import ray_cast_aabb as _cpp_ray_cast_aabb  # type: ignore[import-not-found]

    def ray_cast_aabb_fast(
        origin: np.ndarray,
        heading: float,
        num_rays: int,
        fov_rad: float,
        max_range: float,
        obstacles: np.ndarray,
    ) -> np.ndarray:
        obstacles = np.asarray(obstacles, dtype=np.float32)
        if obstacles.ndim == 1 and obstacles.size == 0:
            obstacles = obstacles.reshape(0, 4)
        return _cpp_ray_cast_aabb(
            np.asarray(origin, dtype=np.float32),
            float(heading),
            int(num_rays),
            float(fov_rad),
            float(max_range),
            obstacles,
        )

    _CPP_BACKEND = True
except ImportError:
    ray_cast_aabb_fast = ray_cast_aabb
    _CPP_BACKEND = False


def cpp_backend_available() -> bool:
    """True iff the compiled C++ ray-caster is importable on this interpreter."""
    return _CPP_BACKEND
