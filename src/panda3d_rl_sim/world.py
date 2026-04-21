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
        size = self.cfg.world_size * 0.8
        self.rover_pos = np_random.uniform(-size, size, size=2).astype(np.float32)
        self.rover_heading = float(np_random.uniform(-np.pi, np.pi))

        goal = np_random.uniform(-size, size, size=2).astype(np.float32)
        for _ in range(32):
            if np.linalg.norm(goal - self.rover_pos) >= self.cfg.min_goal_distance:
                break
            goal = np_random.uniform(-size, size, size=2).astype(np.float32)
        self.goal_pos = goal
        self._sync_scene()

    def apply_action(self, action: np.ndarray, dt: float) -> None:
        forward = float(action[0]) * self.cfg.max_speed
        turn = float(action[1]) * self.cfg.max_turn_rate

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

    def rover_position(self) -> np.ndarray:
        return self.rover_pos.copy()

    def goal_position(self) -> np.ndarray:
        return self.goal_pos.copy()

    def get_state_vector(self) -> np.ndarray:
        delta = self.goal_pos - self.rover_pos
        return np.array(
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


def _resize_nearest(img: np.ndarray, w: int, h: int) -> np.ndarray:
    src_h, src_w = img.shape[:2]
    y_idx = np.linspace(0, src_h - 1, h).astype(np.int32)
    x_idx = np.linspace(0, src_w - 1, w).astype(np.int32)
    return img[y_idx[:, None], x_idx[None, :]].copy()
