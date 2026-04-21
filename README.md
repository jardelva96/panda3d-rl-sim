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

## Vectorised environments

`make_vec_env` creates a batch of environments that share the same
`EnvConfig` and exposes the standard Gymnasium `VectorEnv` API:

```python
from panda3d_rl_sim import make_vec_env, EnvConfig

# 8 envs, same process (fastest on single-core machines)
venv = make_vec_env(8)

# 8 envs in separate subprocesses — useful when env steps are slow
venv = make_vec_env(8, async_envs=True)

# With domain randomisation + multi-goal
cfg = EnvConfig(num_goals=3, dr_num_obstacles=(2, 8), dr_max_speed=(2.0, 5.0))
venv = make_vec_env(4, config=cfg)
```

A throughput benchmark is available in `examples/vec_rollout.py`:

```bash
python examples/vec_rollout.py --n-envs 4 --steps 5000
python examples/vec_rollout.py --n-envs 4 --async-envs --steps 5000
```

## Training a PPO policy

A ready-to-run Stable-Baselines3 training script is shipped in
`examples/train_ppo.py`. Install the optional training dependencies and
train:

```bash
pip install -e ".[train]"
python examples/train_ppo.py --timesteps 200000
```

Training takes a few minutes on a modern CPU with the default four
subprocess workers. A pretrained checkpoint lives at
`examples/checkpoints/ppo_panda_nav.zip` so you can skip straight to
evaluation:

```bash
python examples/eval_trained.py                      # aggregate metrics
python examples/eval_trained.py --render human       # watch rollouts
```

### Baseline results (200 k timesteps, CPU, seed 0)

| Metric | Value |
|---|---|
| Mean episode return | −0.29 |
| Mean episode length | 268 steps |
| Success rate (20 eps) | 5 % |
| Training throughput | ~1 100 fps |
| Wall-clock time | 182 s |

> Early-stage run; the agent has learned to avoid walls but rarely reaches
> the goal. A longer run (≥ 1 M timesteps) or reward-shaping (dense distance
> signal) will push success rate well above 50 %.

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
├── examples/
│   ├── random_agent.py       Baseline — run a uniform random policy
│   ├── train_ppo.py          Stable-Baselines3 PPO training
│   ├── eval_trained.py       Rollout metrics for a trained checkpoint
│   └── checkpoints/          Pretrained PPO weights (committed)
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
- [x] Multi-goal sequential navigation (`num_goals=N`; backward-compatible with single-goal default)
- [x] Domain randomization hooks (num_obstacles, obstacle size, max_speed, max_turn_rate)
- [x] Vectorized `SyncVectorEnv` / `AsyncVectorEnv` support via `make_vec_env()`
- [ ] Depth and segmentation sensor outputs
- [x] Example Stable-Baselines3 PPO training config and pre-trained checkpoint

## License

MIT — see [LICENSE](LICENSE).
