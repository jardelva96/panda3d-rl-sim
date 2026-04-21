# panda3d-rl-sim

A reinforcement-learning simulation environment built on the [Panda3D](https://www.panda3d.org/)
3D engine, exposed through a [Gymnasium](https://gymnasium.farama.org/) API.

It lets you train and evaluate agents — classical RL policies, vision-based
models, or LLM-driven agents — inside a scriptable, headless-capable 3D world
written entirely in Python, with an optional C++ extension for performance-
critical sensor math.

## Why Panda3D

Panda3D is a mature, open-source real-time 3D engine originally developed by
Disney and maintained today by Carnegie Mellon's Entertainment Technology
Center. Unlike most game engines, it is driven from Python as a first-class
language and runs fully headless, which makes it well suited to simulation and
machine-learning workloads where the "player" is a training loop rather than
a human.

Properties that matter for this project:

- **Python-native** scene graph, physics, shaders, and input pipeline
- **Headless / offscreen** rendering out of the box — no display required
- **Cross-platform** (Windows, Linux, macOS) with prebuilt wheels on PyPI
- **Bullet physics** integration available; built-in collision system
- **Portable** to older hardware — OpenGL 2.1 baseline, with a software
  `tinydisplay` renderer as a last-resort fallback

## What this project is

`panda3d-rl-sim` wraps a small 3D world in the standard Gymnasium `Env`
interface so you can plug it into any RL training stack
(Stable-Baselines3, RLlib, CleanRL, custom loops, ...) without writing
engine-specific glue.

The reference task — `PandaNavEnv` — is a continuous-control navigation
problem: a differential-drive rover on a bounded plane has to reach a
randomly placed goal. It is simple enough to train in minutes on CPU while
still exercising the full pipeline: action application, scene update,
observation capture, reward shaping, and termination.

### Observations

| Mode | Shape | Use case |
|------|-------|----------|
| `state`  | `Box(7 + num_rays,)` float32 | Fast training, classical RL |
| `pixels` | `Box(H, W, 3)` uint8         | Vision-based RL, multimodal agents |

The `state` vector concatenates two blocks:

1. **Pose + goal** (7 floats): `[x, y, cos(h), sin(h), goal_dx, goal_dy, distance]`
2. **LIDAR fan** (`num_rays` floats): the distance to the nearest obstacle
   hit along each of `num_rays` rays spread evenly across `lidar_fov_rad`
   and anchored to the rover's heading. A ray that hits nothing reports
   `lidar_max_range`.

`pixels` observations are captured from an offscreen Panda3D camera using
`Texture.getRamImageAs("RGB")`, so they cost no display and work on CI
workers or remote servers.

### Sensors and the C++ backend

The LIDAR fan is computed with a 2-D axis-aligned-box ray caster. The module
`panda3d_rl_sim.sensors` ships two interchangeable implementations:

- `ray_cast_aabb` — pure NumPy, used by default and always available;
- `ray_cast_aabb_fast` — same signature, but dispatches to the compiled
  C++ extension in [`cpp/`](cpp/) when it is importable, and falls back to
  the Python implementation otherwise.

Call `panda3d_rl_sim.sensors.cpp_backend_available()` to see which backend
is active at runtime.

### Render modes

| `render_mode` | Panda3D pipe | Typical use |
|---|---|---|
| `None`        | `window-type none`      | Fastest; state-only training |
| `"rgb_array"` | `window-type offscreen` | Video recording, pixel observations |
| `"human"`     | `window-type onscreen`  | Interactive debugging |

## Installation

Requires Python 3.10+.

```bash
git clone https://github.com/jardelva96/panda3d-rl-sim.git
cd panda3d-rl-sim
pip install -e .
```

For the optional pixels observation mode and on-screen rendering you also
need a working OpenGL 2.1+ driver. On headless Linux servers either stay on
`window-type none` (state-only) or run the process under `xvfb-run`.

## Quick start

```python
import panda3d_rl_sim  # registers "Panda3D-Nav-v0"
import gymnasium as gym

env = gym.make("Panda3D-Nav-v0", observation_mode="state")
obs, info = env.reset(seed=0)

for _ in range(200):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

Watch a random policy in an on-screen window:

```bash
python examples/random_agent.py --render human
```

## Project layout

```
panda3d-rl-sim/
├── src/panda3d_rl_sim/       Core Python package
│   ├── env.py                Gymnasium Env wrapper
│   ├── world.py              Panda3D scene and simulation state
│   ├── sensors.py            LIDAR-like ray cast (pure-Python fallback)
│   └── config.py             Dataclass configuration
├── cpp/                      Optional C++ extension (pybind11)
│   ├── src/raycaster.cpp     Vectorized 2D ray casts against AABBs
│   └── CMakeLists.txt
├── examples/                 End-to-end scripts
└── tests/                    Unit tests (state mode; no display needed)
```

## Optional C++ extension

A pure-Python ray-caster ships in `sensors.py` and is used by default.
For performance-critical workloads — dense LIDAR scans, many obstacles, or
massively parallel rollouts — `cpp/` contains a pybind11 implementation of
the same API with a ~10× speedup on representative workloads.

Build and install:

```bash
cd cpp
cmake -B build -S .
cmake --build build --config Release
pip install .
```

See [cpp/README.md](cpp/README.md) for details.

## Roadmap

- [x] Procedurally placed obstacles and AABB collision termination
- [x] LIDAR-like range sensor with optional C++ backend
- [ ] Bullet-backed rigid-body dynamics (slopes, friction, dynamic bodies)
- [ ] Multi-goal and procedurally generated room layouts
- [ ] Domain randomization hooks (textures, lighting, mass, friction)
- [ ] Vectorized `AsyncVectorEnv` support for parallel rollouts
- [ ] Depth and segmentation sensor outputs
- [ ] Example Stable-Baselines3 PPO training config and pre-trained checkpoint

## License

MIT — see [LICENSE](LICENSE).
