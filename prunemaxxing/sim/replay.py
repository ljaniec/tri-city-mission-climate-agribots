"""Replay a MissionResult inside the 1:1 MuJoCo world.

The mission is simulated by :mod:`prunemaxxing.planning.mission` (controller +
vehicle dynamics + swarm); this module poses the MuJoCo bodies kinematically
from the logs at any time ``t`` (world coordinates, terrain-following, wheel
spin + Ackermann steering, rotor spin, effector lift, particle FX, mulch
texture), and renders named or cinematic free cameras.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", "/usr/share/glvnd/egl_vendor.d/10_nvidia.json")

import imageio.v2 as imageio  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from prunemaxxing.planning import ms_effectors as fx  # noqa: E402
from prunemaxxing.planning.mission import LABEL_CODES, MissionResult  # noqa: E402
from prunemaxxing.planning.swarm import PHASE_APPLY, PHASE_BACK, PHASE_CLIMB, PHASE_LAND, PHASE_OUT  # noqa: E402
from prunemaxxing.sim import world_builder as wb  # noqa: E402
from prunemaxxing.sim.world_robots import DOCKS, DRONE_SKID_Z, PAD_TOP_Z  # noqa: E402

G = 9.81


def _quat_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])


def _font(size: int):
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


@dataclass
class CamSpec:
    lookat: tuple[float, float, float]
    distance: float
    azimuth: float
    elevation: float


class ReplayScene:
    def __init__(self, res: MissionResult, max_size: tuple[int, int] = (1920, 1080)):
        self.res = res
        self.mode = res.spec.mode.value
        self.spec = wb.build_world(site=res.site, payload_mode=self.mode, swarm_size=res.n_drones)
        self.model = wb.load_world_model(self.spec)
        self.data = mujoco.MjData(self.model)
        wb.set_payload_mode(self.model, self.mode)
        self.frame = self.spec.frame
        self.terrain = self.spec.terrain
        m = self.model
        self.model.vis.global_.offwidth = max(m.vis.global_.offwidth, max_size[0])
        self.model.vis.global_.offheight = max(m.vis.global_.offheight, max_size[1])
        self._renderers: dict[tuple[int, int], mujoco.Renderer] = {}
        self._painter = None
        self._painter_t: float | None = None
        self._painter_ver = 0
        self._mulch_t: dict[tuple[int, int], int] = {}
        self._mulch_grid_t = -1.0
        self._mulch_grid = None
        self._jadr = {}
        for j in range(m.njnt):
            self._jadr[mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)] = int(m.jnt_qposadr[j])
        self._mocap = {}
        for b in range(m.nbody):
            if m.body_mocapid[b] >= 0:
                self._mocap[mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)] = int(m.body_mocapid[b])
        self.dust_ids = [self._mocap[f"fx_dust_{k}"] for k in range(200) if f"fx_dust_{k}" in self._mocap]
        self.cart_ids = [self._mocap[f"fx_cart_{k}"] for k in range(40) if f"fx_cart_{k}" in self._mocap]
        self.blow_ids = self._pool("fx_blow_")
        self.pellet_ids = self._pool("fx_pellet_")
        self.puff_ids = self._pool("fx_puff_")
        self.bdust_ids = self._pool("fx_bdust_")
        self.trail_ids = self._pool("fx_trail_")
        self._sites = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, i): i for i in range(m.nsite)}
        self._det_t = {e["oid"]: e["t"] for e in res.events if e["kind"] == "obstacle_detected"}
        self._det_state = {oid: None for oid in self._det_t}
        cells = res.plan.cells
        self._cell_rc = np.array([[c.row, c.col] for c in cells], dtype=int)
        twin_shape = (int(self._cell_rc[:, 0].max()) + 1, int(self._cell_rc[:, 1].max()) + 1)
        self._grid_shape = (self.spec.cells.nrows, self.spec.cells.ncols) if hasattr(self.spec, "cells") else twin_shape
        # smoothed yaw for chase cameras
        L = res.ms_log
        self._yaw_t = L["t"]
        self._yaw = L["yaw"]
        self.t = 0.0

    # ------------------------------------------------------------ helpers
    def world(self, e: float, n: float) -> tuple[float, float]:
        x, y = self.frame.to_world(e, n)
        return float(x), float(y)

    def ground(self, x: float, y: float) -> float:
        return float(self.terrain.height_at(x, y))

    def _set_free(self, joint: str, pos, quat) -> None:
        a = self._jadr.get(joint)
        if a is None:
            return
        self.data.qpos[a : a + 3] = pos
        self.data.qpos[a + 3 : a + 7] = quat

    def _set_hinge(self, joint: str, val: float) -> None:
        a = self._jadr.get(joint)
        if a is not None:
            self.data.qpos[a] = val

    def smooth_yaw(self, t: float, window: float = 4.0) -> float:
        ts = np.linspace(t - window, t, 9)
        ys = np.interp(ts, self._yaw_t, self._yaw)
        w = np.linspace(0.3, 1.0, 9)
        return float(np.sum(ys * w) / np.sum(w))

    def ms_world(self, t: float) -> dict:
        ms = self.res.ms_state_at(t)
        x, y = self.world(ms["x"], ms["y"])
        ms["wx"], ms["wy"] = x, y
        ms["wz"] = self.ground(x, y)
        return ms

    # ------------------------------------------------------------ pose
    def set_time(self, t: float) -> None:
        self.t = t
        res = self.res
        vp = res.cfg.vehicle
        ms = self.ms_world(t)
        x, y, yaw = ms["wx"], ms["wy"], ms["yaw"]
        c, s = math.cos(yaw), math.sin(yaw)
        hl, ht = vp.wheelbase_m / 2, vp.track_m / 2
        zf = self.ground(x + hl * c, y + hl * s)
        zr = self.ground(x - hl * c, y - hl * s)
        zl = self.ground(x - ht * s, y + ht * c)
        zrt = self.ground(x + ht * s, y - ht * c)
        pitch = -math.atan2(zf - zr, vp.wheelbase_m)
        roll = math.atan2(zl - zrt, vp.track_m)
        z = 0.25 * (zf + zr + zl + zrt)
        q_ms = _quat_from_euler(roll, pitch, yaw)
        self._set_free("ms_free", [x, y, z], q_ms)
        R_ms = np.zeros(9)
        mujoco.mju_quat2Mat(R_ms, q_ms)
        R_ms = R_ms.reshape(3, 3)
        ms_pos = np.array([x, y, z])
        wheel = ms["odo"] / vp.wheel_radius_m
        for jn in ("ms_wheel_fl", "ms_wheel_fr", "ms_wheel_rl", "ms_wheel_rr"):
            self._set_hinge(jn, wheel)
        d = ms["steer"]
        if abs(d) > 1e-4:
            tl = math.tan(d)
            d_left = math.atan(vp.wheelbase_m * tl / (vp.wheelbase_m - ht * tl))
            d_right = math.atan(vp.wheelbase_m * tl / (vp.wheelbase_m + ht * tl))
        else:
            d_left = d_right = 0.0
        self._set_hinge("ms_steer_fl", d_left)
        self._set_hinge("ms_steer_fr", d_right)

        # drones
        n_dust = len(self.dust_ids)
        per_drone_dust = n_dust // max(1, res.n_drones)
        per_drone_cart = len(self.cart_ids) // max(1, res.n_drones)
        dust_pos = np.full((n_dust, 3), [0.0, 0.0, -60.0])
        cart_pos = np.full((len(self.cart_ids), 3), [0.0, 0.0, -60.0])
        for di in range(res.n_drones):
            st = res.drone_state_at(di, t)
            wx, wy = self.world(st["x"], st["y"])
            wz = st["z"]
            vx, vy = st["vx"], st["vy"]
            sp = math.hypot(vx, vy)
            tilt = min(0.28, 0.035 * sp)
            dyaw = st["yaw"]
            # the mission docks on a flat, yaw-only deck model: pin to the real pad pose
            # and blend into the free flight track over the first metres of climb
            dk = DOCKS[di % len(DOCKS)]
            pad = ms_pos + R_ms @ np.array([dk[0], dk[1], PAD_TOP_Z + DRONE_SKID_Z])
            if st["phase"] == 0:
                ref, w = np.array([wx, wy, wz]), 1.0
            else:
                tr = res.trajectory(st["sortie"])
                rx, ry = self.world(tr.actual[0, 0], tr.actual[0, 1])
                ref = np.array([rx, ry, tr.actual[0, 2]])
                w = float(np.clip(1.0 - (wz - ref[2]) / 2.5, 0.0, 1.0))
            wx, wy, wz = np.array([wx, wy, wz]) + w * (pad - ref)
            self._set_free(f"drone{di}_free", [wx, wy, wz],
                           _quat_from_euler(w * roll, w * pitch + (1 - w) * tilt, dyaw))
            flying = st["phase"] != 0
            spin = (t * 41.0 + di * 1.3) if (flying or st["payload"] > 0) else 0.0
            for k in range(6):
                self._set_hinge(f"drone{di}_rotor{k}", spin * (1 if k % 2 == 0 else -1) + k * 0.5)
            ph = st["phase"]
            lift = {PHASE_APPLY: -1.1, PHASE_OUT: -0.35, PHASE_BACK: -0.25, PHASE_CLIMB: -0.1, PHASE_LAND: -0.05}.get(ph, 0.0)
            self._set_hinge(f"drone{di}_lift", lift)
            # FX
            if ph == PHASE_APPLY and self.mode == "dust" and per_drone_dust:
                dust_pos[di * per_drone_dust : (di + 1) * per_drone_dust] = self._dust_particles(
                    di, t, np.array([wx, wy, wz + lift - 0.55]), np.array([vx, vy]), per_drone_dust)
            if self.mode == "cartridge" and per_drone_cart and st["sortie"] >= 0:
                cart_pos[di * per_drone_cart : (di + 1) * per_drone_cart] = self._cart_drops(di, st["sortie"], t,
                                                                                            per_drone_cart)
        for k, mid in enumerate(self.dust_ids):
            self.data.mocap_pos[mid] = dust_pos[k]
        for k, mid in enumerate(self.cart_ids):
            self.data.mocap_pos[mid] = cart_pos[k]
        self._pose_effectors(t)

        for oid, td in self._det_t.items():
            det = t >= td
            if self._det_state[oid] != det:
                wb.set_obstacle_detected(self.model, oid, det)
                self._det_state[oid] = det
        mujoco.mj_forward(self.model, self.data)

    # ------------------------------------------------------------ mothership effectors
    def _pool(self, prefix: str) -> list[int]:
        items = [(int(k[len(prefix):]), v) for k, v in self._mocap.items()
                 if k.startswith(prefix) and k[len(prefix):].isdigit()]
        return [v for _, v in sorted(items)]

    def _site(self, name: str) -> np.ndarray | None:
        sid = self._sites.get(name)
        return None if sid is None else self.data.site_xpos[sid].copy()

    def _pose_effectors(self, t: float) -> None:
        """Chute / turret joints from the mission log, then blower stream, pellets, muzzle puffs."""
        res = self.res
        states = res.ms_effector_state(t)
        ep = res.cfg.effectors
        for st in states:
            if st["kind"] == fx.KIND_BLOWER:
                self._set_hinge("ms_chute_yaw", st["rel_az"])
                self._set_hinge("ms_chute_pitch", 0.55 * st["elev"])
                self._set_hinge("ms_impeller", t * 38.0 if st["firing"] else 0.0)
            else:
                s = st["side"]
                self._set_hinge(f"ms_cannon{s}_yaw", st["rel_az"])
                self._set_hinge(f"ms_cannon{s}_pitch", st["elev"])
                age = t - st["t_shot"]
                for b in (0, 1):
                    rec = -0.12 * max(0.0, 1.0 - age / 0.35) if (b == st["barrel"] and 0 <= age < 0.35) else 0.0
                    self._set_hinge(f"ms_cannon{s}_recoil{b}", rec)
        if not (self.blow_ids or self.pellet_ids or self.puff_ids):
            return
        mujoco.mj_kinematics(self.model, self.data)
        blow = np.full((len(self.blow_ids), 3), [0.0, 0.0, -60.0])
        pel = np.full((len(self.pellet_ids), 3), [0.0, 0.0, -60.0])
        puff = np.full((len(self.puff_ids), 3), [0.0, 0.0, -60.0])
        bdust = np.full((len(self.bdust_ids), 3), [0.0, 0.0, -60.0])
        trail = np.full((len(self.trail_ids), 3), [0.0, 0.0, -60.0])
        n_pel = n_puff = n_tr = 0
        for st in states:
            if st["kind"] == fx.KIND_BLOWER:
                tip = self._site("ms_chute_tip")
                if st["firing"] and st["aim"] is not None and tip is not None and len(self.blow_ids):
                    ax, ay = self.world(*st["aim"])
                    blow[:] = self._blower_stream(t, tip, np.array([ax, ay]), len(self.blow_ids),
                                                  math.radians(ep.blower_elev_deg))
                    if len(self.bdust_ids):
                        bdust[:] = self._blower_stream(0.37 * t + 11.0, tip, np.array([ax, ay]), len(self.bdust_ids),
                                                       math.radians(ep.blower_elev_deg), fan=2.2)
                continue
            s = st["side"]
            for p, tr in zip(st["pellets"], st["trails"]):
                if n_pel < len(pel):
                    wx, wy = self.world(p[0], p[1])
                    pel[n_pel] = [wx, wy, p[2]]
                    n_pel += 1
                for q in tr:
                    if n_tr < len(trail):
                        wx, wy = self.world(q[0], q[1])
                        trail[n_tr] = [wx, wy, q[2]]
                        n_tr += 1
            age = t - st["t_shot"]
            if 0 <= age < 0.45 and n_puff < len(puff):
                m = self._site(f"ms_cannon{s}_muzzle{st['barrel']}")
                if m is not None:
                    d = np.array([math.cos(st["az"]), math.sin(st["az"]), math.tan(st["elev"])])
                    puff[n_puff] = m + d / np.linalg.norm(d) * (0.25 + 2.2 * age)
                    n_puff += 1
        # recently landed cartridges stay visible on the bed
        if self.mode == "cartridge" and n_pel < len(pel):
            for st in states:
                k = st["row"]
                if k < 0:
                    continue
                v = res.ms_volley(k)
                landed = np.flatnonzero(v["t_fire"] + v["t_fly"] <= t)[::-1]
                for i in landed:
                    if n_pel >= len(pel):
                        break
                    wx, wy = self.world(v["land"][i, 0], v["land"][i, 1])
                    pel[n_pel] = [wx, wy, float(self.spec.ground_z(wx, wy)) + 0.03]
                    n_pel += 1
        for ids, arr in ((self.blow_ids, blow), (self.pellet_ids, pel), (self.puff_ids, puff),
                         (self.bdust_ids, bdust), (self.trail_ids, trail)):
            for k, mid in enumerate(ids):
                self.data.mocap_pos[mid] = arr[k]

    def _blower_stream(self, t: float, tip: np.ndarray, aim_xy: np.ndarray, n: int, elev: float,
                       fan: float = 1.0) -> np.ndarray:
        """Stateless chip stream: particles cycle along the ballistic arc, fanning out downrange."""
        gz = float(self.spec.ground_z(aim_xy[0], aim_xy[1]))
        d = aim_xy - tip[:2]
        R = max(float(np.hypot(*d)), 0.5)
        v = fx.launch_speed(R, tip[2] - gz, elev)
        T = R / (v * math.cos(elev))
        j = np.arange(n)
        ph = (j * 0.61803398875) % 1.0
        u = (t / T + ph) % 1.0
        cyc = np.floor(t / T + ph)
        h1 = (np.sin(cyc * 12.9898 + j * 78.233) * 43758.5453) % 1.0
        h2 = (np.sin(cyc * 39.346 + j * 11.135) * 24634.6345) % 1.0
        h3 = (np.sin(cyc * 73.156 + j * 52.235) * 12345.6789) % 1.0
        along = u * (0.85 + 0.3 * h1)
        tt = along * T
        dirn = d / R
        perp = np.array([-dirn[1], dirn[0]])
        spread = (h2 - 0.5) * (0.3 + 1.7 * along) * fan
        xy = tip[None, :2] + (along * R)[:, None] * dirn[None, :] + spread[:, None] * perp[None, :]
        z = tip[2] + v * math.sin(elev) * tt - 0.5 * G * tt**2 + (h3 - 0.5) * (0.12 + 0.45 * along) * fan
        zg = self.spec.ground_z(xy[:, 0], xy[:, 1]) + 0.02
        return np.column_stack([xy, np.maximum(z, zg)])

    def _dust_particles(self, di: int, t: float, src: np.ndarray, vel: np.ndarray, n: int) -> np.ndarray:
        life = 1.1
        j = np.arange(n)
        rng = np.random.default_rng(di * 7919 + j)
        phase = (j * 0.6180339) % 1.0
        age = ((t / life) + phase) % 1.0 * life
        cycle = np.floor((t / life) + phase)
        ang = (np.sin(cycle * 12.9898 + j * 78.233) * 43758.5453) % 1.0 * 2 * np.pi
        r0 = 0.15 + 0.35 * ((np.sin(cycle * 4.1 + j) * 9631.7) % 1.0)
        spd = 2.2 + 1.8 * ((np.cos(cycle * 3.3 + j * 1.7) * 1231.1) % 1.0)
        dirs = np.stack([np.cos(ang), np.sin(ang)], axis=1)
        drag = np.exp(-2.4 * age)
        horiz = dirs * (r0[:, None] + spd[:, None] * (1 - drag)[:, None] / 2.4) - vel[None, :] * age[:, None] * 0.6
        z = src[2] - 0.5 * G * 0.45 * age**2
        pos = np.column_stack([src[0] + horiz[:, 0], src[1] + horiz[:, 1], z])
        gz = self.spec.ground_z(pos[:, 0], pos[:, 1]) + 0.03
        pos[:, 2] = np.maximum(pos[:, 2], gz)
        _ = rng
        return pos

    def _cart_drops(self, di: int, sortie: int, t: float, n: int) -> np.ndarray:
        out = np.full((n, 3), [0.0, 0.0, -60.0])
        tr = self.res.trajectory(sortie)
        if len(tr.drops) == 0:
            return out
        m = tr.drop_t <= t
        idx = np.where(m)[0][-n:]
        for k, i in enumerate(idx):
            wx, wy = self.world(tr.drops[i, 0], tr.drops[i, 1])
            gz = float(self.spec.ground_z(wx, wy)) + 0.06
            age = t - tr.drop_t[i]
            z = max(gz, tr.drops[i, 2] - 0.5 * G * age * age)
            out[k] = [wx, wy, z]
        return out

    # ------------------------------------------------------------ mulch
    def _completion_grid(self, t: float) -> np.ndarray:
        if self._mulch_grid is not None and abs(t - self._mulch_grid_t) < 1e-6:
            return self._mulch_grid
        comp = self.res.completion_at([t])[0]
        grid = np.full(self._grid_shape, np.nan)
        grid[self._cell_rc[:, 0], self._cell_rc[:, 1]] = np.nan_to_num(comp, nan=0.0)
        self._mulch_grid, self._mulch_grid_t = grid, t
        return grid

    def _apply_mulch(self, key, renderer, t: float, min_dt: float) -> None:
        """One painter edits the shared model texture; every GL context gets its own upload."""
        if self._painter is None:
            self._painter = wb.MulchPainter(self.spec, self.model, self.mode)
        if self._painter_t is None or abs(t - self._painter_t) >= min_dt or t < self._painter_t:
            if self._painter_t is not None and t < self._painter_t:
                self._painter.reset(self.model)
            self._painter.apply(self.model, None, self._completion_grid(t), upload=False)
            self._painter_t = t
            self._painter_ver += 1
        if self._mulch_t.get(key) != self._painter_ver:
            renderer._gl_context.make_current()
            mujoco.mjr_uploadTexture(self.model, renderer._mjr_context, self._painter.tex_id)
            self._mulch_t[key] = self._painter_ver

    # ------------------------------------------------------------ render
    def renderer(self, w: int, h: int) -> mujoco.Renderer:
        key = (w, h)
        if key not in self._renderers:
            self._renderers[key] = wb.make_renderer(self.model, height=h, width=w)
        return self._renderers[key]

    def render(self, cam, w: int, h: int, mulch_min_dt: float = 600.0) -> np.ndarray:
        r = self.renderer(w, h)
        self._apply_mulch((w, h), r, self.t, mulch_min_dt)
        if isinstance(cam, CamSpec):
            wb.focus_shadows(r, self.model, self.data, center=cam.lookat,
                             half_extent=float(np.clip(cam.distance * 2.2, 30.0, 255.0)))
            c = mujoco.MjvCamera()
            c.type = mujoco.mjtCamera.mjCAMERA_FREE
            c.lookat[:] = cam.lookat
            c.distance = cam.distance
            c.azimuth = cam.azimuth
            c.elevation = cam.elevation
            r.update_scene(self.data, camera=c)
        else:
            wb.focus_shadows(r, self.model, self.data, center=None)
            r.update_scene(self.data, camera=cam)
        return r.render()

    # ------------------------------------------------------------ cameras
    def chase_cam(self, t: float, distance: float = 15.0, elevation: float = -20.0, side: float = -28.0) -> CamSpec:
        ms = self.ms_world(t)
        yaw = self.smooth_yaw(t)
        az = math.degrees(yaw) + 180.0 + side
        ahead = 2.5
        look = (ms["wx"] + ahead * math.cos(yaw), ms["wy"] + ahead * math.sin(yaw), ms["wz"] + 1.6)
        return CamSpec(look, distance, az, elevation)

    def drone_cam(self, t: float, d: int, distance: float = 7.5) -> CamSpec:
        st = self.res.drone_state_at(d, t)
        wx, wy = self.world(st["x"], st["y"])
        az = math.degrees(st["yaw"]) + 180.0 + 20.0
        return CamSpec((wx, wy, st["z"] - 0.6), distance, az, -24.0)

    def active_drone(self, t: float, prefer: int = 0) -> int:
        order = [prefer] + [d for d in range(self.res.n_drones) if d != prefer]
        for d in order:
            if self.res.drone_state_at(d, t)["phase"] != 0:
                return d
        return prefer

    def dashboard_cams(self, main_size=(1280, 720), second_size=(640, 360)) -> Callable[[float], dict]:
        state = {"drone": 0, "until": -1.0}

        def provider(t: float) -> dict:
            self.set_time(t)
            out = {"main": self.render(self.chase_cam(t), *main_size, mulch_min_dt=900.0)}
            if t > state["until"] or self.res.drone_state_at(state["drone"], t)["phase"] == 0:
                state["drone"] = self.active_drone(t, state["drone"])
                state["until"] = t + 20.0
            d = state["drone"]
            if self.res.drone_state_at(d, t)["phase"] != 0:
                cam = self.drone_cam(t, d, 8.0)
            else:
                ms = self.ms_world(t)
                cam = CamSpec((ms["wx"], ms["wy"], ms["wz"] + 2.4), 9.0, math.degrees(self.smooth_yaw(t)) + 120, -35)
            out["second"] = self.render(cam, *second_size, mulch_min_dt=900.0)
            return out

        return provider


# ---------------------------------------------------------------------------
# Stills + cinematic
# ---------------------------------------------------------------------------


def _caption(img: np.ndarray, title: str, sub: str = "", corner: str = "", alpha: float = 1.0,
             lift: float = 0.0) -> np.ndarray:
    if alpha <= 0.01 or (not title and not corner):
        return img
    im = Image.fromarray(img).convert("RGBA")
    ov = Image.new("RGBA", im.size, (0, 0, 0, 0))
    dr = ImageDraw.Draw(ov)
    W, H = im.size
    s = H / 1080.0
    if title:
        f1, f2 = _font(int(40 * s)), _font(int(24 * s))
        x, y = int(60 * s), int(H - (150 + lift) * s)
        tw = max(dr.textlength(title, font=f1), dr.textlength(sub, font=f2) if sub else 0)
        dr.rounded_rectangle([x - 24 * s, y - 20 * s, x + tw + 30 * s, y + (98 if sub else 60) * s],
                             radius=int(14 * s), fill=(8, 12, 8, int(170 * alpha)))
        dr.rectangle([x - 24 * s, y - 20 * s, x - 16 * s, y + (98 if sub else 60) * s],
                     fill=(224, 151, 47, int(255 * alpha)))
        dr.text((x, y), title, font=f1, fill=(240, 244, 236, int(255 * alpha)))
        if sub:
            dr.text((x, y + 54 * s), sub, font=f2, fill=(190, 205, 186, int(255 * alpha)))
    if corner:
        f3 = _font(int(22 * s))
        dr.rounded_rectangle([W - 30 * s - dr.textlength(corner, font=f3) - 24 * s, 26 * s, W - 30 * s, 70 * s],
                             radius=int(10 * s), fill=(8, 12, 8, 150))
        dr.text((W - 42 * s - dr.textlength(corner, font=f3), 34 * s), corner, font=f3, fill=(240, 200, 120, 255))
    return np.asarray(Image.alpha_composite(im, ov).convert("RGB"))


def _clock(t: float) -> str:
    tot = 7 * 3600 + t
    return f"Day {int(tot // 86400) + 1}  {int(tot % 86400 // 3600):02d}:{int(tot % 3600 // 60):02d}:{int(tot % 60):02d}"


def render_hero_stills(scene: ReplayScene, out_dir: Path, size=(1920, 1080)) -> dict[str, Path]:
    res = scene.res
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    v0 = res.visits[min(2, len(res.visits) - 1)]
    S = res.sorties
    idx = np.where(S["station"] == v0["sid"])[0]
    k = int(idx[min(40, len(idx) - 1)])
    t_fly = float(0.5 * (S["apply_t0"][k] + S["apply_t1"][k]))
    scene.set_time(t_fly)
    ms = scene.ms_world(t_fly)
    yaw = math.degrees(ms["yaw"])
    shots = {
        "hero_swarm": CamSpec((ms["wx"], ms["wy"], ms["wz"] + 2.5), 22.0, yaw + 140, -24),
        "hero_low": CamSpec((ms["wx"], ms["wy"], ms["wz"] + 1.2), 12.0, yaw + 205, -7),
        "hero_drone": scene.drone_cam(t_fly, int(S["drone"][k]), 6.0),
        "hero_aerial": CamSpec((ms["wx"], ms["wy"], ms["wz"]), 90.0, yaw + 160, -50),
    }
    for name, cam in shots.items():
        img = scene.render(cam, *size, mulch_min_dt=0.0)
        p = out_dir / f"{name}.png"
        imageio.imwrite(p, img)
        paths[name] = p
    em = _effector_moment(scene, frac=0.3)
    if em is not None:
        scene.set_time(em[0])
        img = scene.render(em[1](em[0], 0.5), *size, mulch_min_dt=0.0)
        p = out_dir / "hero_ms_effector.png"
        imageio.imwrite(p, img)
        paths["hero_ms_effector"] = p
    # mid-mission overview with mulch
    t_mid = res.t_end * 0.55
    scene.set_time(t_mid)
    cx, cy = scene.world(*res.plan.depot_xy)
    img = scene.render(CamSpec((0.0, 0.0, 2.0), 360.0, 235.0, -38.0), *size, mulch_min_dt=0.0)
    p = out_dir / "hero_overview_mid_mission.png"
    imageio.imwrite(p, img)
    paths["hero_overview"] = p
    # obstacle moment
    if res.detours:
        det = res.detours[0]
        t_o = det["t"] + 12.0
        scene.set_time(t_o)
        ob = next(o for o in res.site.obstacles if o.oid == det["oid"])
        ox, oy = scene.world(ob.polygon.centroid.x, ob.polygon.centroid.y)
        ms = scene.ms_world(t_o)
        mx, my = (ox + ms["wx"]) / 2, (oy + ms["wy"]) / 2
        img = scene.render(CamSpec((mx, my, scene.ground(mx, my) + 1.0), 30.0, math.degrees(ms["yaw"]) + 200, -30),
                           *size, mulch_min_dt=0.0)
        p = out_dir / "hero_obstacle_bypass.png"
        imageio.imwrite(p, img)
        paths["hero_obstacle"] = p
    return paths


def _effector_moment(scene: ReplayScene, min_range: float = 7.0, frac: float = 0.05):
    """(t, CamSpec factory, title, sub) for a long-range mothership blower burst / cannon volley."""
    res = scene.res
    M = res.ms_shots
    if not len(M.get("kg", [])):
        return None
    cand = np.flatnonzero(M["range_m"] > min_range)
    long_ = cand[(M["t_free"][cand] - M["t_fire0"][cand]) > 10.0]
    cand = long_ if len(long_) else cand
    if res.spec.mode.value == "cartridge":  # prefer a moment with both turrets busy
        busy = [k for k in cand if np.any((M["side"] != M["side"][k]) & (M["t0"] < M["t_fire0"][k] + 3)
                                          & (M["t_free"] > M["t_fire0"][k] + 3))]
        cand = np.array(busy) if busy else cand
    if not len(cand):
        cand = np.arange(len(M["kg"]))
    k = int(cand[int(frac * (len(cand) - 1))])
    t = float(M["t_fire0"][k]) + 2.5
    k_share = res.kpis.get("ms_effector_share_pct", 0.0)

    if res.spec.mode.value == "cartridge":
        ax_, ay_ = scene.world(M["aim_x"][k], M["aim_y"][k])

        def cam(tt, u):
            ms = scene.ms_world(tt)
            mx, my = ms["wx"] + 0.5 * (ax_ - ms["wx"]), ms["wy"] + 0.5 * (ay_ - ms["wy"])
            az = math.degrees(math.atan2(ay_ - ms["wy"], ax_ - ms["wx"]))
            return CamSpec((mx, my, ms["wz"] + 0.8), 16.0 - 2.5 * u, az + 34 - 16 * u, -24)
        title = "Twin pneumatic air cannons · 40 ft reach"
        sub = f"seed cartridges fired one by one onto the bed ridges · {k_share:.0f} % placed from the deck"
    else:
        ax_, ay_ = scene.world(M["aim_x"][k], M["aim_y"][k])

        def cam(tt, u):
            ms = scene.ms_world(tt)
            mx, my = 0.5 * (ms["wx"] + ax_), 0.5 * (ms["wy"] + ay_)
            az = math.degrees(math.atan2(ay_ - ms["wy"], ax_ - ms["wx"]))
            return CamSpec((mx, my, ms["wz"] + 1.8), 15.0 - 2.5 * u, az - 115 + 30 * u, -16)
        rate = res.cfg.effectors.blower_rate_kg_s * 3.6
        title = "Snow-blower chip chute · 40 ft reach"
        sub = f"impeller + slewing chute · {rate:.1f} t/h · {k_share:.0f} % of the mulch thrown from the deck"
    return t, cam, title, sub


@dataclass
class Shot:
    t0: float
    seconds: float
    speed: float  # sim seconds per video second
    cam: Callable[[float, float], object]  # (t, u in [0,1]) -> CamSpec | camera name
    title: str = ""
    sub: str = ""
    mulch_dt: float = 900.0


def _ease(u: float) -> float:
    return 0.5 - 0.5 * math.cos(math.pi * min(max(u, 0.0), 1.0))


def plan_cinematic(scene: ReplayScene) -> list[Shot]:
    res = scene.res
    k = res.kpis
    S = res.sorties
    v0 = res.visits[0]
    mode = scene.mode
    shots: list[Shot] = []
    cx, cy = 0.0, 0.0
    shots.append(Shot(0.0, 5.0, 0.6, lambda t, u: CamSpec((cx, cy, 2.0), 420 - 120 * _ease(u), 205 + 35 * u, -34 + 6 * u),
                      "PruneMaxxing", f"{k['area_treated_ha']:.1f} ha pilot · Parc Agrari del Baix Llobregat · "
                      f"1:1 digital twin from 10 m DEM"))
    shots.append(Shot(0.0, 6.0, 4.0, lambda t, u: scene.chase_cam(t, 13.0 - 2 * u, -16.0, -35 + 25 * u),
                      "Mothership leaves the depot", "wood-chip hopper 2.5 t · 4 drones docked on the deck"))
    t_l = v0["t_arrive"] - 3.0

    def orbit(t, u):
        ms = scene.ms_world(t)
        return CamSpec((ms["wx"], ms["wy"], ms["wz"] + 3.0), 19.0 - 3 * u, math.degrees(ms["yaw"]) + 100 + 70 * u, -18)

    shots.append(Shot(t_l, 9.0, 1.0, orbit, f"Drone station 1 of {len(res.plan.stations)}",
                      "swarm launch · wireless deck charging · battery swap every ~25 sorties"))
    idx = np.where((S["station"] == v0["sid"]) & (S["t_launch"] > v0["t_arrive"] + 40))[0]
    if len(idx):
        ks = int(idx[1] if len(idx) > 1 else idx[0])
        dd = int(S["drone"][ks])
        t_s = float(S["t_launch"][ks]) - 1.0
        dur = float(S["t_land"][ks] - S["t_launch"][ks]) + 2.0
        label = ("Dust mode · 25 kg spinner-spread wood shavings" if mode == "dust"
                 else "Cartridge mode · 16 seed + mycelium pucks per sortie")
        shots.append(Shot(t_s, dur / 1.5, 1.5, lambda t, u, dd=dd: scene.drone_cam(t, dd, 7.0 - 1.5 * u), label,
                          "planned sortie vs wind-disturbed actual track · logged at 4 Hz"))
        ta = float(S["apply_t0"][ks]) - 1.5
        tr = res.trajectory(ks)
        m = tr.phase == PHASE_APPLY
        pc = tr.actual[m].mean(axis=0) if m.any() else tr.actual[len(tr.actual) // 2]
        px, py = scene.world(pc[0], pc[1])
        gz = scene.ground(px, py)
        shots.append(Shot(ta, 5.0, 1.0, lambda t, u, px=px, py=py, gz=gz: CamSpec((px, py, gz + 1.6), 9.0 - 2 * u,
                                                                                 230 + 20 * u, -12),
                          "Effector close-up", "cradle lowered on the winch · application at 2-3 m AGL"))
    em = _effector_moment(scene)
    if em is not None:
        shots.append(Shot(em[0] - 1.5, 7.0, 1.0, em[1], em[2], em[3]))
    # lane drive: find a long lane stretch
    L = res.ms_log
    lane = np.where((L["mode"] == LABEL_CODES["lane"]) & (L["v"] > 0.9))[0]
    if len(lane) > 200:
        t_lane = float(L["t"][lane[len(lane) // 3]])
        shots.append(Shot(t_lane, 6.0, 1.5, lambda t, u: scene.chase_cam(t, 7.5, -6.0, 85.0 - 20 * u),
                          "Wheels locked in the furrows",
                          f"pure pursuit + actuator limits · lane cross-track RMS {100 * k['cte_rms_lane_m']:.1f} cm"))
    for det in res.detours[:2]:
        ob = next(o for o in res.site.obstacles if o.oid == det["oid"])
        ev = next(e for e in res.events if e["kind"] == "obstacle_detected" and e["oid"] == det["oid"])
        shots.append(Shot(det["t"] - 4.0, 9.0, 4.0, lambda t, u: scene.chase_cam(t, 34.0 - 8 * u, -38.0, -40 + 30 * u),
                          f"Unknown {ob.kind} — not in the satellite map",
                          f"lidar contact at {ev['range_m']:.1f} m · local re-plan {ev.get('action', '')}"))
    # headland turn
    hl = np.where((L["mode"] == LABEL_CODES["headland"]) & (np.abs(L["steer"]) > 0.25))[0]
    if len(hl):
        t_h = float(L["t"][hl[len(hl) // 2]]) - 10.0
        shots.append(Shot(t_h, 5.0, 4.0, lambda t, u: scene.chase_cam(t, 26.0, -62.0, 0.0 + 40 * u),
                          "Headland U-turn", "6.5 m radius · no wheel on a bed"))
    # time-lapse overview
    T = res.t_end
    shots.append(Shot(0.0, 8.0, T / 8.0, lambda t, u: CamSpec((0.0, -10.0, 2.0), 330.0 - 40 * u, 220 + 25 * u, -42),
                      f"{k['mission_time_h']:.0f} h in 8 s",
                      f"{k['biomass_applied_t']:,.0f} t {'mulch' if mode == 'dust' else 'seed cartridges'} · "
                      f"{k['sorties']:,} sorties · {len(k['obstacles_detected'])} unknown obstacles bypassed",
                      mulch_dt=0.0))
    shots.append(Shot(T - 1.0, 4.0, 0.25, lambda t, u: scene.chase_cam(T - 1.0, 16 + 4 * u, -18, 60 + 40 * u),
                      "Mission complete",
                      f"{k['area_treated_ha']:.2f} ha at {k['dose_kg_per_m2']:.1f} kg/m² · CV {100 * k['dose_cv']:.1f} %",
                      mulch_dt=0.0))
    return shots


def render_cinematic(scene: ReplayScene, out_path: Path, fps: int = 24, size=(1920, 1080),
                     preview: bool = False, kpi_bar: bool = True) -> Path:
    from prunemaxxing.viz.kpi_overlay import KpiOverlay
    from prunemaxxing.viz.video_io import video_writer

    shots = plan_cinematic(scene)
    if preview:
        size = (960, 540)
    overlay = KpiOverlay(scene.res) if kpi_bar else None
    with video_writer(out_path, fps) as writer:
        for k, sh in enumerate(shots):
            n = max(1, int(round(sh.seconds * fps * (0.25 if preview else 1.0))))
            for i in range(n):
                u = i / max(1, n - 1)
                t = min(sh.t0 + (i / fps) * sh.speed * (4.0 if preview else 1.0), scene.res.t_end)
                scene.set_time(t)
                img = scene.render(sh.cam(t, u), *size, mulch_min_dt=sh.mulch_dt)
                fade = min(1.0, i / (0.4 * fps), (n - 1 - i) / (0.4 * fps) + 0.15)
                speed_txt = "real time" if sh.speed <= 1.01 else (f"x{sh.speed:.0f}" if sh.speed < 200 else
                                                                  f"time-lapse x{sh.speed:,.0f}")
                bar_a = 0.0 if overlay is None or k == 0 else (min(1.0, i / (0.6 * fps)) if k == 1 else 1.0)
                if bar_a > 0.0:
                    img = overlay.draw(img, t, compact=True, alpha=bar_a)
                img = _caption(img, sh.title, sh.sub, f"{_clock(t)}  ·  {speed_txt}", alpha=max(0.0, fade),
                               lift=66.0 if overlay is not None and k > 0 else 0.0)
                writer.append_data(img)
    return out_path
