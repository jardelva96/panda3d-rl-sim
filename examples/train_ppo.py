"""Train a PPO policy on PandaNavEnv with Stable-Baselines3.

Install the optional training dependencies first::

    pip install -e ".[train]"

Then run::

    python examples/train_ppo.py --timesteps 200000

The trained checkpoint is written to
``examples/checkpoints/ppo_panda_nav.zip`` by default. Load it back with
``eval_trained.py`` to inspect success rate and watch rollouts.

Each parallel worker runs in its own subprocess because Panda3D's
``ShowBase`` is effectively a per-process singleton; ``DummyVecEnv`` would
force several envs to share one scene graph.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

import panda3d_rl_sim  # noqa: F401 — registers the env id


def _make_env_thunk(seed: int):
    def _thunk():
        env = gym.make("Panda3D-Nav-v0", observation_mode="state")
        env.reset(seed=seed)
        return env
    return _thunk


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--timesteps", type=int, default=200_000)
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("examples/checkpoints/ppo_panda_nav.zip"),
    )
    ap.add_argument(
        "--log-dir",
        type=Path,
        default=Path("examples/checkpoints/logs"),
    )
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Tensorboard logging is optional — the `tensorboard` package is only
    # listed in the [train] extra and may not be installed.
    tb_log = None
    try:
        import tensorboard  # noqa: F401
        args.log_dir.mkdir(parents=True, exist_ok=True)
        tb_log = str(args.log_dir)
    except ImportError:
        print("[train] tensorboard not installed — skipping TB logging.")

    thunks = [_make_env_thunk(args.seed + i) for i in range(args.n_envs)]
    vec_env = SubprocVecEnv(thunks) if args.n_envs > 1 else DummyVecEnv(thunks)

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        seed=args.seed,
        tensorboard_log=tb_log,
        n_steps=256,
        batch_size=256,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.0,
        policy_kwargs={"net_arch": [64, 64]},
    )
    model.learn(total_timesteps=args.timesteps, progress_bar=False)
    model.save(args.out)
    print(f"Saved checkpoint to {args.out.resolve()}")

    vec_env.close()


if __name__ == "__main__":
    main()
