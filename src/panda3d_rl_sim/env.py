"""Gymnasium environment wrapping a Panda3D world."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .config import EnvConfig
from .world import World


class PandaNavEnv(gym.Env):
    """A differential-drive navigation task rendered in Panda3D.

    Parameters
    ----------
    render_mode : {None, "rgb_array", "human"}, optional
        Controls the underlying Panda3D graphics pipe. ``None`` skips graphics
        entirely (fastest); ``"rgb_array"`` renders to an offscreen buffer;
        ``"human"`` opens an on-screen window.
    observation_mode : {"state", "pixels"}
        Shape of the observation returned by :meth:`reset` / :meth:`step`.
    config : EnvConfig, optional
        Environment constants. A default is used if omitted.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(
        self,
        render_mode: str | None = None,
        observation_mode: str = "state",
        config: EnvConfig | None = None,
    ):
        super().__init__()
        self.cfg = config or EnvConfig()
        self.render_mode = render_mode
        self.observation_mode = observation_mode

        if observation_mode not in ("state", "pixels"):
            raise ValueError(
                f"observation_mode must be 'state' or 'pixels', got {observation_mode!r}"
            )
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(
                f"render_mode must be one of {self.metadata['render_modes']} or None, "
                f"got {render_mode!r}"
            )

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        if observation_mode == "state":
            # 7 pose+goal features, followed by num_rays LIDAR distances.
            obs_dim = 7 + max(0, self.cfg.num_rays)
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            )
        else:
            self.observation_space = spaces.Box(
                low=0, high=255,
                shape=(self.cfg.pixel_height, self.cfg.pixel_width, 3),
                dtype=np.uint8,
            )

        needs_graphics = render_mode is not None or observation_mode == "pixels"
        self.world = World(
            render_mode=render_mode,
            needs_graphics=needs_graphics,
            config=self.cfg,
        )

        self._steps = 0
        self._prev_dist = 0.0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.world.reset(self.np_random)
        self._steps = 0
        self._prev_dist = self.world.distance_to_goal()
        return self._get_obs(), self._info()

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).clip(-1.0, 1.0)
        self.world.apply_action(action, dt=self.cfg.dt)

        dist = self.world.distance_to_goal()
        shaping = self._prev_dist - dist  # positive when we get closer
        self._prev_dist = dist

        reward = float(shaping + self.cfg.reward_step_penalty)
        reached = dist < self.cfg.goal_radius
        out_of_bounds = self.world.is_out_of_bounds()
        collided = self.world.is_collided()
        self._steps += 1
        timeout = self._steps >= self.cfg.max_steps

        if reached:
            reward += self.cfg.reward_goal
        if out_of_bounds:
            reward += self.cfg.reward_out_of_bounds
        if collided:
            reward += self.cfg.reward_collision

        terminated = bool(reached or out_of_bounds or collided)
        truncated = bool(timeout and not terminated)
        return self._get_obs(), reward, terminated, truncated, self._info()

    def render(self):
        if self.render_mode == "rgb_array":
            return self.world.get_camera_image()
        return None

    def close(self):
        self.world.close()

    def _get_obs(self):
        if self.observation_mode == "pixels":
            return self.world.get_camera_image()
        return self.world.get_state_vector()

    def _info(self):
        return {
            "distance_to_goal": self.world.distance_to_goal(),
            "rover_pos": self.world.rover_position(),
            "goal_pos": self.world.goal_position(),
            "num_obstacles": self.world.num_obstacles(),
            "collided": self.world.is_collided(),
            "steps": self._steps,
            "ep_max_speed": self.world.ep_max_speed(),
            "ep_max_turn_rate": self.world.ep_max_turn_rate(),
        }
