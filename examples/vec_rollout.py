"""Vectorised parallel rollout example using make_vec_env.

Demonstrates both SyncVectorEnv (same-process) and AsyncVectorEnv
(subprocess workers) on the Panda3D-Nav-v0 task.

Usage
-----
    # Sync (default): 4 envs in the calling process
    python examples/vec_rollout.py

    # Async: 4 envs each in their own subprocess
    python examples/vec_rollout.py --async-envs

    # With domain randomisation and multi-goal
    python examples/vec_rollout.py --n-envs 8 --num-goals 3 \\
        --dr-obstacles 2 8 --dr-speed 1.5 4.5
"""

from __future__ import annotations

import argparse
import time

import panda3d_rl_sim  # noqa: F401 — registers Panda3D-Nav-v0
from panda3d_rl_sim import EnvConfig, make_vec_env


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Vectorised rollout benchmark")
    p.add_argument("--n-envs", type=int, default=4, help="Number of parallel envs")
    p.add_argument("--steps", type=int, default=1000, help="Total env-steps to collect")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--async-envs", action="store_true", help="Use AsyncVectorEnv")
    p.add_argument("--num-goals", type=int, default=1)
    p.add_argument("--num-obstacles", type=int, default=4)
    p.add_argument(
        "--dr-obstacles", type=int, nargs=2, metavar=("LO", "HI"), default=None
    )
    p.add_argument(
        "--dr-speed", type=float, nargs=2, metavar=("LO", "HI"), default=None
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg = EnvConfig(
        num_goals=args.num_goals,
        num_obstacles=args.num_obstacles,
        dr_num_obstacles=tuple(args.dr_obstacles) if args.dr_obstacles else None,
        dr_max_speed=tuple(args.dr_speed) if args.dr_speed else None,
    )

    mode = "Async" if args.async_envs else "Sync"
    print(f"[vec_rollout] {mode}VectorEnv × {args.n_envs} envs | "
          f"num_goals={cfg.num_goals} | collecting {args.steps} steps …")

    venv = make_vec_env(args.n_envs, config=cfg, async_envs=args.async_envs)
    obs, _info = venv.reset(seed=args.seed)

    steps_collected = 0
    episodes = 0
    total_reward = 0.0
    t0 = time.perf_counter()

    while steps_collected < args.steps:
        action = venv.action_space.sample()
        obs, rew, term, trunc, _info = venv.step(action)
        steps_collected += args.n_envs
        total_reward += float(rew.sum())
        episodes += int((term | trunc).sum())

    elapsed = time.perf_counter() - t0
    fps = steps_collected / elapsed

    print(f"  steps collected : {steps_collected}")
    print(f"  episodes done   : {episodes}")
    print(f"  mean reward/env : {total_reward / steps_collected:.4f}")
    print(f"  wall-clock time : {elapsed:.2f}s")
    print(f"  throughput      : {fps:.0f} steps/s")

    venv.close()


if __name__ == "__main__":
    main()
