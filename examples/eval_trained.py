"""Evaluate a trained PPO policy on PandaNavEnv.

Usage
-----
    # headless, 50 episodes, print aggregate metrics
    python examples/eval_trained.py

    # open a window and watch the agent
    python examples/eval_trained.py --render human --episodes 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

import panda3d_rl_sim  # noqa: F401


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("examples/checkpoints/ppo_panda_nav.zip"),
    )
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--render", choices=["none", "human"], default="none")
    ap.add_argument("--seed", type=int, default=1000)
    args = ap.parse_args()

    if not args.checkpoint.exists():
        raise SystemExit(
            f"Checkpoint not found: {args.checkpoint}\n"
            "Train one first with:  python examples/train_ppo.py"
        )

    render_mode = None if args.render == "none" else args.render
    env = gym.make("Panda3D-Nav-v0", render_mode=render_mode)
    model = PPO.load(args.checkpoint, env=env)

    returns: list[float] = []
    lengths: list[int] = []
    successes: list[int] = []

    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        total_r = 0.0
        steps = 0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_r += float(reward)
            steps += 1
            if terminated or truncated:
                break
        reached = info["distance_to_goal"] < 0.6
        returns.append(total_r)
        lengths.append(steps)
        successes.append(int(reached))
        print(
            f"ep {ep:3d}  return={total_r:+7.2f}  len={steps:3d}  "
            f"{'reached' if reached else 'miss   '}"
        )

    print("-" * 48)
    print(f"episodes:       {args.episodes}")
    print(f"mean return:    {np.mean(returns):+.3f}")
    print(f"mean length:    {np.mean(lengths):.1f}")
    print(f"success rate:   {np.mean(successes):.0%}")

    env.close()


if __name__ == "__main__":
    main()
