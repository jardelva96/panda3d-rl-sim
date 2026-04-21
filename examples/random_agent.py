"""Run a random policy in PandaNavEnv, optionally with on-screen rendering.

Usage
-----
    python examples/random_agent.py                    # headless, 3 episodes
    python examples/random_agent.py --render human     # open a window
    python examples/random_agent.py --episodes 10
"""

from __future__ import annotations

import argparse

import gymnasium as gym

import panda3d_rl_sim  # noqa: F401 — triggers env registration


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--render",
        choices=["none", "rgb_array", "human"],
        default="none",
        help="Panda3D render mode (default: none = headless)",
    )
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    render_mode = None if args.render == "none" else args.render
    env = gym.make("Panda3D-Nav-v0", render_mode=render_mode)

    try:
        for ep in range(args.episodes):
            obs, info = env.reset(seed=args.seed + ep)
            total = 0.0
            steps = 0
            while True:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                total += reward
                steps += 1
                if terminated or truncated:
                    break
            print(
                f"episode {ep}: steps={steps:3d} return={total:+.3f} "
                f"final_distance={info['distance_to_goal']:.3f} "
                f"{'reached' if info['distance_to_goal'] < 0.6 else 'miss'}"
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
