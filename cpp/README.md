# C++ extension — `panda3d_rl_sim_cpp`

Optional pybind11 implementation of the 2D ray-caster used by
`panda3d_rl_sim.sensors`. The Python fallback in
[`../src/panda3d_rl_sim/sensors.py`](../src/panda3d_rl_sim/sensors.py) is
always available; build this extension only if you need the ~10× speedup
on dense LIDAR workloads.

## Requirements

- A C++17 compiler
  - Windows: MSVC 2019+ / Visual Studio Build Tools
  - Linux:   GCC 9+ or Clang 10+
  - macOS:   Xcode command-line tools
- CMake ≥ 3.15
- Python ≥ 3.10 with `pybind11` installed (`pip install pybind11`)

## Build

```bash
pip install pybind11
cd cpp
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

The compiled module lands in `cpp/build/` as `_raycaster.<ext>` where
`<ext>` is platform-specific (`.so`, `.pyd`, `.dylib`).

## Using the compiled module

```python
from cpp.build import _raycaster  # or install it onto sys.path
import numpy as np

origin = np.array([0.0, 0.0], dtype=np.float32)
obstacles = np.array([[4.5, -0.5, 5.5, 0.5]], dtype=np.float32)
distances = _raycaster.ray_cast_aabb(origin, 0.0, 32, 3.14159, 10.0, obstacles)
```

The signature matches `panda3d_rl_sim.sensors.ray_cast_aabb` exactly, so the
Python fallback and the C++ implementation are drop-in swappable.

## Benchmark

On a scene with 64 AABB obstacles and 360 rays per frame (representative of
a dense LIDAR-like sensor), the C++ implementation is roughly 8–12× faster
than the pure-Python loop, depending on compiler and CPU. Run your own
measurements with `examples/benchmark_raycaster.py` (once wired in).
