"""Panda3D scene graph and simulation state for :mod:`panda3d_rl_sim.env`."""

from __future__ import annotations

import builtins
import contextlib

import numpy as np

from .config import EnvConfig


def _configure_panda3d(render_mode: str | None, needs_graphics: bool) -> None:
    """Set Panda3D config variables *before* the first ``ShowBase`` import."""
    from panda3d.core import loadPrcFileData

    if render_mode == "human":
        loadPrcFileData("", "window-type onscreen")
        loadPrcFileData("", "win-size 800 600")
        loadPrcFileData("", "window-title panda3d-rl-sim")
    elif needs_graphics:
        loadPrcFileData("", "window-type offscreen")
    else:
        loadPrcFileData("", "window-type none")

    loadPrcFileData("", "audio-library-name null")
    loadPrcFileData("", "sync-video 0")
    loadPrcFileData("", "notify-level-display error")
    loadPrcFileData("", "notify-level-glgsg error")


def _make_box(parent, half_extent: float, color, name: str):
    """Build a unit cube procedurally and attach it to ``parent``."""
    from panda3d.core import (
        Geom,
        GeomNode,
        GeomTriangles,
        GeomVertexData,
        GeomVertexFormat,
        GeomVertexWriter,
    )

    fmt = GeomVertexFormat.getV3n3()
    vdata = GeomVertexData(name, fmt, Geom.UHStatic)
    vwriter = GeomVertexWriter(vdata, "vertex")
    nwriter = GeomVertexWriter(vdata, "normal")

    s = half_extent
    corners = [
        (-s, -s, -s), (+s, -s, -s), (+s, +s, -s), (-s, +s, -s),
        (-s, -s, +s), (+s, -s, +s), (+s, +s, +s), (-s, +s, +s),
    ]
    faces = [
        ([0, 1, 2, 3], (0, 0, -1)),
        ([4, 5, 6, 7], (0, 0, 1)),
        ([0, 1, 5, 4], (0, -1, 0)),
        ([2, 3, 7, 6], (0, 1, 0)),
        ([1, 2, 6, 5], (1, 0, 0)),
        ([0, 3, 7, 4], (-1, 0, 0)),
    ]
    tris = GeomTriangles(Geom.UHStatic)
    vi = 0
    for idxs, normal in faces:
        for i in idxs:
            x, y, z = corners[i]
            vwriter.addData3(x, y, z)
            nwriter.addData3(*normal)
        tris.addVertices(vi, vi + 1, vi + 2)
        tris.addVertices(vi, vi + 2, vi + 3)
        vi += 4

    geom = Geom(vdata)
    geom.addPrimitive(tris)
    node = GeomNode(name)
    node.addGeom(geom)
    np_ = parent.attachNewNode(node)
    np_.setColor(*color)
    return np_


class World:
    """Owns the Panda3D scene graph and the simulation state.

    Notes
    -----
    ``ShowBase`` is effectively a per-process singleton in Panda3D. This class
    reuses an existing instance if one already exists (useful for test suites
    where multiple environments are instantiated serially).
    """

    def __init__(self, render_mode: str | None, needs_graphics: bool, config: EnvConfig):
        _configure_panda3d(render_mode, needs_graphics)

        existing = getattr(builtins, "base", None)
        if existing is not None:
            self.base = existing
            self._owns_base = False
        else:
            from direct.showbase.ShowBase import ShowBase
            self.base = ShowBase()
            self._owns_base = True

        self.cfg = config
        self.needs_graphics = needs_graphics
        self.render_mode = render_mode

        self._build_scene()

        self.rover_pos = np.zeros(2, dtype=np.float32)
        self.rover_heading = 0.0
        self.goal_pos = np.zeros(2, dtype=np.float32)
        self.obstacle_nps: list = []
        self.obstacles_aabb = np.zeros((0, 4), dtype=np.float32)

        # Per-episode effective values; updated by reset() when DR is enabled.
        self._ep_num_obstacles: int = config.num_obstacles
        self._ep_obstacle_half_extent: float = config.obstacle_half_extent
        self._ep_max_speed: float = config.max_speed
        self._ep_max_turn_rate: float = config.max_turn_rate

    # ---------------------------------------------------------------- setup

    def _build_scene(self):
        from panda3d.core import (
            AmbientLight,
            CardMaker,
            DirectionalLight,
            Vec4,
        )

        root = self.base.render

        cm = CardMaker("ground")
        size = self.cfg.world_size
        cm.setFrame(-size, size, -size, size)
        self.ground_np = root.attachNewNode(cm.generate())
        self.ground_np.setHpr(0, -90, 0)  # face +Z
        self.ground_np.setColor(0.30, 0.50, 0.30, 1)

        self.rover_np = _make_box(root, 0.4, (0.90, 0.20, 0.20, 1), "rover")
        self.rover_np.setZ(0.4)

        self.goal_np = _make_box(root, 0.3, (0.20, 0.90, 0.20, 1), "goal")
        self.goal_np.setZ(0.3)

        alight = AmbientLight("ambient")
        alight.setColor(Vec4(0.4, 0.4, 0.45, 1))
        self._ambient_np = root.attachNewNode(alight)
        root.setLight(self._ambient_np)

        dlight = DirectionalLight("sun")
        dlight.setColor(Vec4(0.9, 0.9, 0.85, 1))
        self._sun_np = root.attachNewNode(dlight)
        self._sun_np.setHpr(-45, -45, 0)
        root.setLight(self._sun_np)

        if self.needs_graphics:
            self.base.disableMouse()
            self.base.camera.setPos(0, -14, 11)
            self.base.camera.lookAt(0, 0, 0)

    # ------------------------------------------------------------ lifecycle

    def reset(self, np_random) -> None:
        cfg = self.cfg
        # Domain randomisation: sample per-episode dynamics / layout parameters.
        if cfg.dr_num_obstacles is not None:
            lo, hi = cfg.dr_num_obstacles
            self._ep_num_obstacles = int(np_random.integers(lo, hi + 1))
        else:
            self._ep_num_obstacles = cfg.num_obstacles
        if cfg.dr_obstacle_half_extent is not None:
            lo_f, hi_f = cfg.dr_obstacle_half_extent
            self._ep_obstacle_half_extent = float(np_random.uniform(lo_f, hi_f))
        else:
            self._ep_obstacle_half_extent = cfg.obstacle_half_extent
        if cfg.dr_max_speed is not None:
            lo_f, hi_f = cfg.dr_max_speed
            self._ep_max_speed = float(np_random.uniform(lo_f, hi_f))
        else:
            self._ep_max_speed = cfg.max_speed
        if cfg.dr_max_turn_rate is not None:
            lo_f, hi_f = cfg.dr_max_turn_rate
            self._ep_max_turn_rate = float(np_random.uniform(lo_f, hi_f))
        else:
            self._ep_max_turn_rate = cfg.max_turn_rate

        size = self.cfg.world_size * 0.8
        self.rover_pos = np_random.uniform(-size, size, size=2).astype(np.float32)
        self.rover_heading = float(np_random.uniform(-np.pi, np.pi))

        goal = np_random.uniform(-size, size, size=2).astype(np.float32)
        for _ in range(32):
            if np.linalg.norm(goal - self.rover_pos) >= self.cfg.min_goal_distance:
                break
            goal = np_random.uniform(-size, size, size=2).astype(np.float32)
        self.goal_pos = goal

        self._spawn_obstacles(np_random)
        self._sync_scene()

    def _spawn_obstacles(self, np_random) -> None:
        # Tear down any obstacles left over from a previous episode.
        for node in self.obstacle_nps:
            with contextlib.suppress(Exception):
                node.removeNode()
        self.obstacle_nps = []

        n = self._ep_num_obstacles
        if n <= 0:
            self.obstacles_aabb = np.zeros((0, 4), dtype=np.float32)
            return

        he = self._ep_obstacle_half_extent
        bound = self.cfg.world_size * 0.75
        placements: list[np.ndarray] = []

        min_clearance_rover = self.cfg.rover_half_extent + he + 0.5
        min_clearance_goal = self.cfg.goal_radius + he + 0.3
        min_clearance_pair = 2.0 * he + 0.3

        for _ in range(n):
            for _attempt in range(32):
                p = np_random.uniform(-bound, bound, size=2).astype(np.float32)
                if np.linalg.norm(p - self.rover_pos) < min_clearance_rover:
                    continue
                if np.linalg.norm(p - self.goal_pos) < min_clearance_goal:
                    continue
                if any(np.linalg.norm(p - q) < min_clearance_pair for q in placements):
                    continue
                placements.append(p)
                break

        aabbs = np.empty((len(placements), 4), dtype=np.float32)
        for i, p in enumerate(placements):
            aabbs[i] = (p[0] - he, p[1] - he, p[0] + he, p[1] + he)
            node = _make_box(self.base.render, he, (0.45, 0.45, 0.55, 1), f"obstacle_{i}")
            node.setPos(float(p[0]), float(p[1]), he)
            self.obstacle_nps.append(node)
        self.obstacles_aabb = aabbs

    def apply_action(self, action: np.ndarray, dt: float) -> None:
        forward = float(action[0]) * self._ep_max_speed
        turn = float(action[1]) * self._ep_max_turn_rate

        self.rover_heading += turn * dt
        self.rover_heading = (self.rover_heading + np.pi) % (2 * np.pi) - np.pi

        self.rover_pos[0] += np.cos(self.rover_heading) * forward * dt
        self.rover_pos[1] += np.sin(self.rover_heading) * forward * dt

        self._sync_scene()
        if self.needs_graphics:
            self.base.graphicsEngine.renderFrame()

    def close(self) -> None:
        # Leave the ShowBase singleton alive — tearing it down reliably in the
        # same process is brittle. Just drop our scene nodes.
        for node in self.obstacle_nps:
            with contextlib.suppress(Exception):
                node.removeNode()
        self.obstacle_nps = []
        for attr in ("rover_np", "goal_np", "ground_np", "_ambient_np", "_sun_np"):
            node = getattr(self, attr, None)
            if node is not None:
                with contextlib.suppress(Exception):
                    node.removeNode()

    # ------------------------------------------------------------- queries

    def distance_to_goal(self) -> float:
        return float(np.linalg.norm(self.rover_pos - self.goal_pos))

    def is_out_of_bounds(self) -> bool:
        return bool(np.any(np.abs(self.rover_pos) > self.cfg.world_size))

    def is_collided(self) -> bool:
        """AABB overlap test between the rover and every obstacle."""
        if self.obstacles_aabb.size == 0:
            return False
        he = self.cfg.rover_half_extent
        rx, ry = self.rover_pos[0], self.rover_pos[1]
        obs = self.obstacles_aabb
        overlap = (
            (rx + he > obs[:, 0])
            & (rx - he < obs[:, 2])
            & (ry + he > obs[:, 1])
            & (ry - he < obs[:, 3])
        )
        return bool(overlap.any())

    def rover_position(self) -> np.ndarray:
        return self.rover_pos.copy()

    def goal_position(self) -> np.ndarray:
        return self.goal_pos.copy()

    def ep_max_speed(self) -> float:
        """Effective max speed for the current episode (may differ from cfg when DR is on)."""
        return self._ep_max_speed

    def ep_max_turn_rate(self) -> float:
        """Effective max turn rate for the current episode (may differ from cfg when DR is on)."""
        return self._ep_max_turn_rate

    def get_lidar(self) -> np.ndarray:
        """Return ``num_rays`` ray-cast distances fanning out from the rover."""
        from .sensors import ray_cast_aabb_fast

        return ray_cast_aabb_fast(
            origin=self.rover_pos,
            heading=self.rover_heading,
            num_rays=self.cfg.num_rays,
            fov_rad=self.cfg.lidar_fov_rad,
            max_range=self.cfg.lidar_max_range,
            obstacles=self.obstacles_aabb,
        )

    def get_state_vector(self) -> np.ndarray:
        delta = self.goal_pos - self.rover_pos
        base = np.array(
            [
                self.rover_pos[0],
                self.rover_pos[1],
                np.cos(self.rover_heading),
                np.sin(self.rover_heading),
                delta[0],
                delta[1],
                float(np.linalg.norm(delta)),
            ],
            dtype=np.float32,
        )
        if self.cfg.num_rays <= 0:
            return base
        return np.concatenate([base, self.get_lidar().astype(np.float32)])

    def get_camera_image(self) -> np.ndarray:
        if not self.needs_graphics or self.base.win is None:
            raise RuntimeError(
                "Camera image requested but the Panda3D world was built without "
                "graphics. Use render_mode='rgb_array' / 'human' or "
                "observation_mode='pixels'."
            )
        self.base.graphicsEngine.renderFrame()
        tex = self.base.win.getScreenshot()
        if tex is None:
            raise RuntimeError("Panda3D window returned no screenshot")

        xs, ys = tex.getXSize(), tex.getYSize()
        data = tex.getRamImageAs("RGB")
        img = np.frombuffer(bytes(data), dtype=np.uint8).reshape((ys, xs, 3))
        img = np.flipud(img)
        return _resize_nearest(img, self.cfg.pixel_width, self.cfg.pixel_height)

    # --------------------------------------------------------------- utils

    def _sync_scene(self) -> None:
        self.rover_np.setPos(float(self.rover_pos[0]), float(self.rover_pos[1]), 0.4)
        self.rover_np.setH(float(np.degrees(self.rover_heading)))
        self.goal_np.setPos(float(self.goal_pos[0]), float(self.goal_pos[1]), 0.3)

    def num_obstacles(self) -> int:
        return int(self.obstacles_aabb.shape[0])


def _resize_nearest(img: np.ndarray, w: int, h: int) -> np.ndarray:
    src_h, src_w = img.shape[:2]
    y_idx = np.linspace(0, src_h - 1, h).astype(np.int32)
    x_idx = np.linspace(0, src_w - 1, w).astype(np.int32)
    return img[y_idx[:, None], x_idx[None, :]].copy()
