"""Configuration dataclass for :class:`panda3d_rl_sim.env.PandaNavEnv`."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class EnvConfig:
    # World geometry
    world_size: float = 10.0
    goal_radius: float = 0.6
    min_goal_distance: float = 3.0

    # Rover dynamics
    max_speed: float = 3.0
    max_turn_rate: float = 2.5  # rad/s
    rover_half_extent: float = 0.4

    # Simulation
    dt: float = 1.0 / 20.0
    max_steps: int = 300

    # Obstacles (axis-aligned boxes scattered on the plane)
    num_obstacles: int = 4
    obstacle_half_extent: float = 0.5

    # LIDAR-like range sensor: fan of rays anchored to the rover's heading
    num_rays: int = 16
    lidar_fov_rad: float = math.pi  # 180°
    lidar_max_range: float = 10.0

    # Multi-goal sequential navigation
    # num_goals=1 is equivalent to the classic single-goal task.
    # Goals are visited in random order; each reached goal grants reward_goal
    # and advances goal_index.  Episode terminates when all goals are reached
    # OR on collision / out-of-bounds / timeout.
    num_goals: int = 1
    min_goal_separation: float = 2.0  # minimum distance between any two goals

    # Rewards
    reward_goal: float = 10.0
    reward_out_of_bounds: float = -5.0
    reward_collision: float = -5.0
    reward_step_penalty: float = -0.01

    # Pixel observation / rgb_array render
    pixel_width: int = 84
    pixel_height: int = 84

    # Domain randomisation — set (lo, hi) to enable per-episode sampling.
    # When None the corresponding base value is used unchanged every episode.
    # Sampled independently at every reset() call.
    dr_num_obstacles: tuple[int, int] | None = None
    dr_obstacle_half_extent: tuple[float, float] | None = None
    dr_max_speed: tuple[float, float] | None = None
    dr_max_turn_rate: tuple[float, float] | None = None
