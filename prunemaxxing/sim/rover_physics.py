"""Closed-loop MuJoCo physics validation of the 4 t Ackermann mothership UGV.

The controller never touches the vehicle pose: every 50 ms it reads the MuJoCo
state, computes a bicycle steering angle and a speed command, and writes actuator
targets (two knuckle position servos + four wheel velocity servos). MuJoCo then
integrates a multibody rover (free chassis, oscillating front axle, steering
knuckles, four driven wheels) on a DEM heightfield or a plane. Everything logged
is read back from the physics state.

Vehicle frame: the chassis body origin is the rear-axle midpoint at wheel-centre
height (x forward, y left, z up). Logged ``x, y`` are that point (the pure-pursuit
reference point), ``z`` is its absolute world height (terrain + ~0.45 m).
``cross_track`` is signed, positive = vehicle left of the reference direction.

World frame matches :mod:`prunemaxxing.sim.world_frame`
(x = easting - center_e, y = northing - center_n, z = absolute elevation).
"""

from __future__ import annotations

import csv
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RoverParams:
    """Physical vehicle + low-level actuator model."""

    wheelbase: float = 2.4
    track: float = 1.7
    wheel_radius: float = 0.45
    wheel_half_width: float = 0.18
    wheel_geom: str = "ellipsoid"  # collision shape: ellipsoid | cylinder | sphere
    chassis_size: tuple[float, float, float] = (4.2, 2.0, 0.6)
    chassis_z: float = 0.85  # deck centre above the axle line (deck bottom ~1.0 m above ground)
    chassis_mass: float = 3550.0  # frame + payload
    wheel_mass: float = 85.0
    knuckle_mass: float = 20.0
    front_axle_mass: float = 70.0
    axle_roll_limit: float = 0.10  # oscillating front axle stop (rad)
    # steering: equivalent bicycle limit 0.38 rad -> R_min = L / tan(0.38) = 6.0 m
    steer_limit: float = 0.38
    knuckle_limit: float = 0.46  # Ackermann inner wheel needs ~0.44 rad at R_min
    steer_rate_max: float = 0.40  # rad/s hydraulic valve slew on the command
    steer_kp: float = 30000.0  # N m / rad
    steer_kv: float = 4000.0  # N m s / rad  -> dominant servo lag ~ kv / kp = 0.13 s
    steer_torque_max: float = 8000.0  # N m
    steer_armature: float = 15.0  # kg m^2 (hydraulic cylinder + linkage reflected)
    # drive: 4WD wheel velocity servos with torque saturation
    drive_kv: float = 2500.0  # N m s / rad per wheel
    drive_torque_max: float = 900.0  # N m per wheel: 8 kN total, covers 0.4 m/s^2 on a 9 % grade
    wheel_armature: float = 2.0
    # longitudinal / lateral comfort limits used by the speed profile
    a_lon_max: float = 0.4
    a_dec_max: float = 0.4
    a_lat_max: float = 0.2
    # contact: sliding mu, torsional (m). MuJoCo contact rolling friction (condim 6)
    # makes rolling wheels chatter, so rolling resistance is Coulomb friction on the
    # spin joints instead: frictionloss = c_rr * (static wheel load) * r.
    mu_slide: float = 0.9
    mu_torsion: float = 0.01
    mu_roll: float = 0.0
    wheel_condim: int = 4
    c_rr: float = 0.05  # firm cultivated soil
    contact_solref: tuple[float, float] = (0.02, 1.0)
    contact_solimp: tuple[float, float, float] = (0.95, 0.99, 0.002)
    timestep: float = 0.002
    impratio: float = 10.0

    @property
    def total_mass(self) -> float:
        return self.chassis_mass + 4 * self.wheel_mass + 2 * self.knuckle_mass + self.front_axle_mass


@dataclass(frozen=True)
class ControllerParams:
    """Pure pursuit + delay compensation + speed PI (runs at 1 / dt_ctrl)."""

    k_ld: float = 1.0  # s, adaptive lookahead gain
    ld0: float = 1.6  # m
    ld_min: float = 2.2  # m
    ld_max: float = 5.0  # m
    t_pred: float = 0.15  # s, predict pose over actuator lag before pursuit
    k_ff: float = 0.0  # optional curvature feedforward blend
    ki_cte: float = 0.0  # optional integral on cross-track (1/m^2), straights only
    kp_v: float = 0.6
    ki_v: float = 0.5
    i_v_max: float = 0.3  # m/s anti-windup
    v_creep: float = 0.2  # m/s minimum command until the end is reached
    a_dec_profile: float = 0.3  # m/s^2 used to shape decelerations in the profile
    kappa_window_m: float = 2.0  # +/- window for curvature max-filter in the profile
    plateau_m: float = 6.0  # +/- window for the speed min-filter in the profile


# ---------------------------------------------------------------------------
# Reference path
# ---------------------------------------------------------------------------


class PathBuilder:
    """Analytic straight / arc / cosine-blend detour polyline, densified."""

    def __init__(self, x: float, y: float, heading: float, ds: float = 0.25):
        self.ds = float(ds)
        self.x, self.y, self.h = float(x), float(y), float(heading)
        self._pts: list[tuple[float, float]] = [(self.x, self.y)]
        self._labels: list[str] = [""]

    def _push(self, x: float, y: float, label: str) -> None:
        self._pts.append((x, y))
        self._labels.append(label)
        if self._labels[0] == "":
            self._labels[0] = label

    def straight(self, length: float, label: str = "lane") -> "PathBuilder":
        n = max(1, int(math.ceil(length / (self.ds / 4))))
        c, s = math.cos(self.h), math.sin(self.h)
        x0, y0 = self.x, self.y
        for i in range(1, n + 1):
            u = length * i / n
            self._push(x0 + u * c, y0 + u * s, label)
        self.x, self.y = x0 + length * c, y0 + length * s
        return self

    def arc(self, radius: float, angle: float, label: str = "headland") -> "PathBuilder":
        """Circular arc; ``angle`` > 0 turns left (CCW)."""
        sgn = 1.0 if angle >= 0 else -1.0
        cx = self.x - sgn * radius * math.sin(self.h)
        cy = self.y + sgn * radius * math.cos(self.h)
        n = max(2, int(math.ceil(radius * abs(angle) / (self.ds / 4))))
        for i in range(1, n + 1):
            hh = self.h + angle * i / n
            self._push(cx + sgn * radius * math.sin(hh), cy - sgn * radius * math.cos(hh), label)
        self.h += angle
        self.x, self.y = self._pts[-1]
        return self

    def detour(self, offset: float, blend: float, hold: float, label: str = "detour") -> "PathBuilder":
        """Lateral S-curve: cosine blend to ``offset`` (left > 0), hold, blend back."""
        total = 2 * blend + hold
        n = max(4, int(math.ceil(total / (self.ds / 4))))
        c, s = math.cos(self.h), math.sin(self.h)
        x0, y0 = self.x, self.y
        for i in range(1, n + 1):
            u = total * i / n
            if u < blend:
                d = offset * 0.5 * (1 - math.cos(math.pi * u / blend))
            elif u <= blend + hold:
                d = offset
            else:
                d = offset * 0.5 * (1 + math.cos(math.pi * (u - blend - hold) / blend))
            self._push(x0 + u * c - d * s, y0 + u * s + d * c, label)
        self.x, self.y = x0 + total * c, y0 + total * s
        return self

    def build(self) -> tuple[np.ndarray, list[str]]:
        """Resample to uniform ``ds`` arc-length spacing; labels by nearest vertex."""
        pts = np.asarray(self._pts, dtype=float)
        seg = np.hypot(*np.diff(pts, axis=0).T)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        n = int(round(s[-1] / self.ds))
        su = np.linspace(0.0, s[-1], n + 1)
        xy = np.column_stack([np.interp(su, s, pts[:, 0]), np.interp(su, s, pts[:, 1])])
        idx = np.clip(np.searchsorted(s, su, side="left"), 0, len(s) - 1)
        labels = [self._labels[i] for i in idx]
        return xy, labels


class ReferencePath:
    """Polyline with arc length, heading, signed curvature, labels and projection."""

    def __init__(self, xy: np.ndarray, labels: Sequence[str] | None = None, max_seg: float = 0.5):
        xy = np.asarray(xy, dtype=float)
        if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 2:
            raise ValueError("path_xy must be (N>=2, 2)")
        lab = list(labels) if labels is not None else ["path"] * len(xy)
        if len(lab) != len(xy):
            raise ValueError("segment_labels must have one label per path point")
        keep = np.concatenate([[True], np.hypot(*np.diff(xy, axis=0).T) > 1e-6])
        xy = xy[keep]
        lab = [l for l, k in zip(lab, keep) if k]
        seg = np.hypot(*np.diff(xy, axis=0).T)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        input_seg = float(seg.max())
        if input_seg > max_seg:  # densify coarse input
            n = int(math.ceil(s[-1] / 0.25))
            su = np.linspace(0, s[-1], n + 1)
            idx = np.clip(np.searchsorted(s, su, side="right") - 1, 0, len(s) - 1)
            lab = [lab[i] for i in idx]
            xy = np.column_stack([np.interp(su, s, xy[:, 0]), np.interp(su, s, xy[:, 1])])
            seg = np.hypot(*np.diff(xy, axis=0).T)
            s = np.concatenate([[0.0], np.cumsum(seg)])
        self.xy = xy
        self.s = s
        self.labels = np.asarray(lab, dtype=object)
        self.length = float(s[-1])
        d = np.diff(xy, axis=0)
        self.seg_dir = d / seg[:, None]
        self.seg_len = seg
        th = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
        self.seg_heading = th
        k = np.zeros(len(xy))
        k[1:-1] = np.diff(th) / (0.5 * (seg[:-1] + seg[1:]))
        k[0], k[-1] = k[1], k[-2]
        if input_seg > max_seg:  # spread vertex curvature spikes of a coarse polyline
            from scipy.ndimage import uniform_filter1d

            k = uniform_filter1d(k, size=max(1, int(round(input_seg / np.median(seg)))), mode="nearest")
        self.kappa = k
        self.heading = np.concatenate([th, th[-1:]])
        # arc length since the last non-lane -> lane boundary (for settling masks)
        since = np.full(len(xy), np.inf)
        boundary = None
        for i in range(len(xy)):
            if self.labels[i] != "lane":
                boundary = None
                since[i] = np.nan
            else:
                if boundary is None and i > 0 and self.labels[i - 1] != "lane":
                    boundary = s[i]
                if boundary is not None:
                    since[i] = s[i] - boundary
        self.s_since_turn = since

    # -- geometry queries -------------------------------------------------
    def point_at(self, s: float) -> np.ndarray:
        if s >= self.length:  # straight extrapolation past the end
            return self.xy[-1] + (s - self.length) * self.seg_dir[-1]
        if s <= 0:
            return self.xy[0] + s * self.seg_dir[0]
        return np.array([np.interp(s, self.s, self.xy[:, 0]), np.interp(s, self.s, self.xy[:, 1])])

    def kappa_at(self, s: float) -> float:
        return float(np.interp(s, self.s, self.kappa))

    def label_at(self, s: float) -> str:
        i = int(np.clip(np.searchsorted(self.s, s, side="right") - 1, 0, len(self.s) - 1))
        return str(self.labels[i])

    def project(self, p: np.ndarray, s_hint: float | None = None, back: float = 3.0,
                fwd: float = 8.0) -> tuple[float, float]:
        """Nearest point (windowed around ``s_hint``) -> (s, signed cross-track)."""
        if s_hint is None:
            i0, i1 = 0, len(self.seg_len)
        else:
            i0 = max(0, int(np.searchsorted(self.s, s_hint - back)) - 1)
            i1 = min(len(self.seg_len), int(np.searchsorted(self.s, s_hint + fwd)) + 1)
        a = self.xy[i0:i1]
        dvec = self.seg_dir[i0:i1]
        L = self.seg_len[i0:i1]
        rel = p[None, :] - a
        u = np.clip(np.einsum("ij,ij->i", rel, dvec), 0.0, L)
        proj = a + u[:, None] * dvec
        dist2 = np.sum((p[None, :] - proj) ** 2, axis=1)
        j = int(np.argmin(dist2))
        r = p - proj[j]
        cte = float(dvec[j, 0] * r[1] - dvec[j, 1] * r[0])
        s = float(self.s[i0 + j] + u[j])
        # past the end: keep the along-track distance on the extrapolated line
        if i0 + j == len(self.seg_len) - 1:
            s = float(self.s[i0 + j] + np.dot(rel[j], dvec[j]))
        return s, cte


def speed_profile(path: ReferencePath, v_max: float, a_lat_max: float, a_dec: float,
                  kappa_window_m: float = 2.0, plateau_m: float = 6.0) -> np.ndarray:
    """Curvature-limited speed, backward pass for decelerations, zero at the end.

    ``kappa_window_m`` max-filters |kappa| (slow down before a curvature step);
    ``plateau_m`` min-filters the speed so short straights inside a manoeuvre
    (e.g. between S-curve blends) do not trigger brief speed-ups.
    """
    from scipy.ndimage import maximum_filter1d, minimum_filter1d

    ds = max(float(np.median(path.seg_len)), 1e-3)
    k = np.abs(path.kappa)
    if kappa_window_m > 0:
        k = maximum_filter1d(k, size=2 * int(math.ceil(kappa_window_m / ds)) + 1, mode="nearest")
    v = np.minimum(v_max, np.sqrt(a_lat_max / np.maximum(k, 1e-9)))
    if plateau_m > 0:
        v = minimum_filter1d(v, size=2 * int(math.ceil(plateau_m / ds)) + 1, mode="nearest")
    v[-1] = 0.0
    for i in range(len(v) - 2, -1, -1):
        v[i] = min(v[i], math.sqrt(v[i + 1] ** 2 + 2 * a_dec * path.seg_len[i]))
    return v


# ---------------------------------------------------------------------------
# MJCF
# ---------------------------------------------------------------------------


@dataclass
class GroundSpec:
    kind: str  # plane | hfield
    center: tuple[float, float]
    half: tuple[float, float]
    z0: float = 0.0  # plane height or hfield base elevation (geom z)
    span: float = 1.0  # hfield vertical scale
    heights01: np.ndarray | None = None  # (nrow, ncol) normalised, row 0 = south
    res: float = 0.5

    def height_at(self, x, y):
        if self.kind == "plane" or self.heights01 is None:
            return np.zeros_like(np.asarray(x, dtype=float)) + self.z0
        from scipy.ndimage import map_coordinates

        xa, ya = np.asarray(x, float), np.asarray(y, float)
        nr, nc = self.heights01.shape
        cx = np.clip((xa - (self.center[0] - self.half[0])) / self.res, 0, nc - 1.000001)
        cy = np.clip((ya - (self.center[1] - self.half[1])) / self.res, 0, nr - 1.000001)
        h = map_coordinates(self.heights01, [np.atleast_1d(cy), np.atleast_1d(cx)], order=1, mode="nearest")
        out = self.z0 + self.span * h
        return float(out[0]) if np.ndim(x) == 0 else out.reshape(xa.shape)


def _f(v: float) -> str:
    return f"{v:.6g}"


def _vec(*v: float) -> str:
    return " ".join(_f(x) for x in v)


def build_rover_mjcf(p: RoverParams, ground: GroundSpec,
                     obstacles: Sequence[dict] | None = None) -> str:
    """MJCF string: ground (plane / hfield), static obstacles, Ackermann rover."""
    L, T, r, hw = p.wheelbase, p.track, p.wheel_radius, p.wheel_half_width
    lx, ly, lz = p.chassis_size
    cz = p.chassis_z
    m = p.chassis_mass
    ixx = m / 12 * (ly**2 + (lz + 0.3) ** 2)
    iyy = m / 12 * (lx**2 + (lz + 0.3) ** 2)
    izz = m / 12 * (lx**2 + ly**2)
    mw = p.wheel_mass
    iw_axle = 0.5 * mw * r**2
    iw_lat = mw * (3 * r**2 + (2 * hw) ** 2) / 12
    fr = _vec(p.mu_slide, p.mu_torsion, p.mu_roll)
    rr_torque = p.c_rr * p.total_mass * 9.81 / 4 * r
    solref = _vec(*p.contact_solref)
    solimp = _vec(*p.contact_solimp)

    if p.wheel_geom == "cylinder":
        wcol = f'<geom class="wheel" type="cylinder" size="{_vec(r, hw)}" euler="1.5708 0 0"/>'
    elif p.wheel_geom == "sphere":
        wcol = f'<geom class="wheel" type="sphere" size="{_f(r)}"/>'
    else:
        wcol = f'<geom class="wheel" type="ellipsoid" size="{_vec(r, hw, r)}"/>'

    def wheel_body(name: str) -> str:
        return f"""
          <body name="{name}_wheel">
            <joint name="{name}_spin" type="hinge" axis="0 1 0" damping="1" armature="{_f(p.wheel_armature)}"
                   frictionloss="{_f(rr_torque)}"/>
            <inertial pos="0 0 0" mass="{_f(mw)}" diaginertia="{_vec(iw_lat, iw_axle, iw_lat)}"/>
            {wcol}
            <geom class="visual" type="cylinder" size="{_vec(r, hw)}" euler="1.5708 0 0" material="tyre"/>
            <geom class="visual" type="cylinder" size="{_vec(0.25, hw + 0.01)}" euler="1.5708 0 0" material="hub"/>
            <geom class="visual" type="box" size="{_vec(0.06, hw + 0.015, r * 0.92)}" material="spoke"/>
          </body>"""

    def knuckle(name: str, y: float) -> str:
        return f"""
        <body name="{name}_knuckle" pos="0 {_f(y)} 0">
          <joint name="{name}_steer" type="hinge" axis="0 0 1" range="{_vec(-p.knuckle_limit, p.knuckle_limit)}"
                 damping="50" armature="{_f(p.steer_armature)}"/>
          <inertial pos="0 0 0.2" mass="{_f(p.knuckle_mass)}" diaginertia="0.6 0.6 0.3"/>
          <geom class="visual" type="cylinder" size="0.06 0.12" pos="0 0 {_f(r + 0.08)}" material="frame"/>
          {wheel_body(name)}
        </body>"""

    if ground.kind == "hfield":
        nr, nc = ground.heights01.shape
        hasset = (f'<hfield name="terrain" nrow="{nr}" ncol="{nc}" '
                  f'size="{_vec(ground.half[0], ground.half[1], ground.span, 1.0)}"/>')
        gxml = (f'<geom name="ground" type="hfield" hfield="terrain" '
                f'pos="{_vec(ground.center[0], ground.center[1], ground.z0)}" material="soil" '
                f'condim="3" friction="{fr}" solref="{solref}" solimp="{solimp}"/>')
    else:
        hasset = ""
        gxml = (f'<geom name="ground" type="plane" size="{_vec(ground.half[0], ground.half[1], 0.1)}" '
                f'pos="{_vec(ground.center[0], ground.center[1], ground.z0)}" material="soil" '
                f'condim="3" friction="{fr}" solref="{solref}" solimp="{solimp}"/>')

    obs_xml = []
    for i, o in enumerate(obstacles or []):
        z = float(o.get("z", ground.height_at(o["x"], o["y"])))
        h = float(o.get("h", 1.5))
        obs_xml.append(
            f'<geom name="obstacle_{i}" type="box" pos="{_vec(o["x"], o["y"], z + h / 2)}" '
            f'euler="0 0 {_f(o.get("yaw", 0.0))}" size="{_vec(o["lx"] / 2, o["ly"] / 2, h / 2)}" '
            f'material="obstacle" condim="3"/>')

    deck_bottom = cz - lz / 2
    return f"""<mujoco model="mothership_ugv">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{_f(p.timestep)}" integrator="implicitfast" cone="elliptic" impratio="{_f(p.impratio)}"
          solver="Newton" iterations="50" tolerance="1e-9"/>
  <visual>
    <global offwidth="1920" offheight="1088"/>
    <quality shadowsize="4096" offsamples="4"/>
    <map haze="0.25"/>
    <headlight ambient="0.38 0.38 0.38" diffuse="0.55 0.55 0.55" specular="0.1 0.1 0.1"/>
  </visual>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.55 0.68 0.85" rgb2="0.12 0.14 0.18" width="512" height="512"/>
    <texture name="soil" type="2d" builtin="checker" rgb1="0.46 0.34 0.22" rgb2="0.36 0.26 0.17"
             mark="edge" markrgb="0.28 0.20 0.13" width="512" height="512"/>
    <material name="soil" texture="soil" texrepeat="0.5 0.5" texuniform="true" reflectance="0"/>
    <material name="body" rgba="0.93 0.55 0.12 1"/>
    <material name="hopper" rgba="0.20 0.42 0.25 1"/>
    <material name="frame" rgba="0.22 0.22 0.24 1"/>
    <material name="tyre" rgba="0.07 0.07 0.07 1"/>
    <material name="hub" rgba="0.75 0.75 0.72 1"/>
    <material name="spoke" rgba="0.95 0.85 0.15 1"/>
    <material name="front" rgba="0.95 0.95 0.95 1"/>
    <material name="obstacle" rgba="0.75 0.12 0.10 1"/>
    {hasset}
  </asset>
  <default>
    <default class="vehicle">
      <geom contype="2" conaffinity="1" condim="3" friction="{fr}" solref="{solref}" solimp="{solimp}"/>
    </default>
    <default class="wheel">
      <geom contype="2" conaffinity="1" condim="{p.wheel_condim}" group="3" friction="{fr}" solref="{solref}" solimp="{solimp}"/>
    </default>
    <default class="visual">
      <geom contype="0" conaffinity="0" group="1" density="0"/>
    </default>
  </default>
  <worldbody>
    {gxml}
    {"".join(obs_xml)}
    <body name="chassis" pos="0 0 {_f(r + 0.05)}">
      <freejoint name="root"/>
      <light name="sun" mode="trackcom" directional="true" pos="10 -7 30" dir="-0.35 0.25 -1"
             diffuse="0.75 0.72 0.65" specular="0.2 0.2 0.2" castshadow="true"/>
      <inertial pos="{_vec(L / 2, 0, cz + 0.05)}" mass="{_f(m)}" diaginertia="{_vec(ixx, iyy, izz)}"/>
      <geom name="deck" class="vehicle" type="box" size="{_vec(lx / 2, ly / 2, lz / 2)}" pos="{_vec(L / 2, 0, cz)}" material="body"/>
      <geom class="visual" type="box" size="1.0 0.85 0.32" pos="{_vec(L / 2 - 0.5, 0, cz + lz / 2 + 0.32)}" material="hopper"/>
      <geom class="visual" type="box" size="0.35 0.9 0.32" pos="{_vec(L / 2 + lx / 2 - 0.4, 0, cz + lz / 2 + 0.32)}" material="frame"/>
      <geom class="visual" type="box" size="0.04 0.98 0.08" pos="{_vec(L / 2 + lx / 2, 0, cz)}" material="front"/>
      <geom class="visual" type="cylinder" size="0.05 0.45" pos="{_vec(L / 2 + lx / 2 - 0.4, 0, cz + lz / 2 + 1.05)}" material="frame"/>
      <geom class="visual" type="cylinder" size="0.14 0.08" pos="{_vec(L / 2 + lx / 2 - 0.4, 0, cz + lz / 2 + 1.55)}" material="front"/>
      <geom class="visual" type="box" size="0.12 0.12 {_f((deck_bottom - r) / 2 + 0.05)}" pos="{_vec(0, T / 2, (deck_bottom + r) / 2)}" material="frame"/>
      <geom class="visual" type="box" size="0.12 0.12 {_f((deck_bottom - r) / 2 + 0.05)}" pos="{_vec(0, -T / 2, (deck_bottom + r) / 2)}" material="frame"/>
      <site name="rear_axle" pos="0 0 0" size="0.05"/>
      <body name="rl_mount" pos="0 {_f(T / 2)} 0">{wheel_body("rl")}
      </body>
      <body name="rr_mount" pos="0 {_f(-T / 2)} 0">{wheel_body("rr")}
      </body>
      <body name="front_axle" pos="{_f(L)} 0 0">
        <joint name="axle_roll" type="hinge" axis="1 0 0" range="{_vec(-p.axle_roll_limit, p.axle_roll_limit)}" damping="3000"/>
        <inertial pos="0 0 0" mass="{_f(p.front_axle_mass)}" diaginertia="17 0.5 17"/>
        <geom class="visual" type="box" size="{_vec(0.1, T / 2 - 0.25, 0.1)}" pos="0 0 {_f(r + 0.05)}" material="frame"/>
        {knuckle("fl", T / 2)}
        {knuckle("fr", -T / 2)}
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="steer_fl" joint="fl_steer" kp="{_f(p.steer_kp)}" kv="{_f(p.steer_kv)}"
              ctrlrange="{_vec(-p.knuckle_limit, p.knuckle_limit)}" forcerange="{_vec(-p.steer_torque_max, p.steer_torque_max)}"/>
    <position name="steer_fr" joint="fr_steer" kp="{_f(p.steer_kp)}" kv="{_f(p.steer_kv)}"
              ctrlrange="{_vec(-p.knuckle_limit, p.knuckle_limit)}" forcerange="{_vec(-p.steer_torque_max, p.steer_torque_max)}"/>
    <velocity name="drive_fl" joint="fl_spin" kv="{_f(p.drive_kv)}" ctrlrange="-10 10" forcerange="{_vec(-p.drive_torque_max, p.drive_torque_max)}"/>
    <velocity name="drive_fr" joint="fr_spin" kv="{_f(p.drive_kv)}" ctrlrange="-10 10" forcerange="{_vec(-p.drive_torque_max, p.drive_torque_max)}"/>
    <velocity name="drive_rl" joint="rl_spin" kv="{_f(p.drive_kv)}" ctrlrange="-10 10" forcerange="{_vec(-p.drive_torque_max, p.drive_torque_max)}"/>
    <velocity name="drive_rr" joint="rr_spin" kv="{_f(p.drive_kv)}" ctrlrange="-10 10" forcerange="{_vec(-p.drive_torque_max, p.drive_torque_max)}"/>
  </actuator>
</mujoco>
"""


def make_ground(path_xy: np.ndarray, terrain: str = "dem", margin: float = 25.0,
                res: float = 0.5) -> GroundSpec:
    """Plane or DEM heightfield patch covering the path bounding box + margin."""
    xy = np.asarray(path_xy, float)
    x0, y0 = xy.min(axis=0) - margin
    x1, y1 = xy.max(axis=0) + margin
    nc = int(math.ceil((x1 - x0) / res)) + 1
    nr = int(math.ceil((y1 - y0) / res)) + 1
    hx, hy = (nc - 1) * res / 2, (nr - 1) * res / 2
    cx, cy = x0 + hx, y0 + hy
    if terrain == "flat":
        return GroundSpec("plane", (cx, cy), (hx, hy), 0.0, res=res)
    if terrain != "dem":
        raise ValueError("terrain must be 'flat' or 'dem'")
    from prunemaxxing.geo.site import load_site
    from prunemaxxing.sim.world_frame import frame_for_site, load_terrain

    frame = frame_for_site(load_site())
    half_x = max(abs(x0), abs(x0 + 2 * hx)) + 2 * res
    half_y = max(abs(y0), abs(y0 + 2 * hy)) + 2 * res
    terr = load_terrain(frame, half_x, half_y, res_m=res)
    xs = x0 + np.arange(nc) * res
    ys = y0 + np.arange(nr) * res
    gx, gy = np.meshgrid(xs, ys)
    h = terr.height_at(gx, gy)
    zmin = float(h.min())
    span = max(float(h.max()) - zmin, 0.05)
    return GroundSpec("hfield", (cx, cy), (hx, hy), zmin, span, (h - zmin) / span, res)


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------


class PathTracker:
    """Pure pursuit (adaptive lookahead, delay-compensated) + speed profile + PI."""

    def __init__(self, path: ReferencePath, rover: RoverParams, ctrl: ControllerParams,
                 v_max: float, dt: float):
        self.path, self.p, self.c, self.dt = path, rover, ctrl, dt
        self.v_prof = speed_profile(path, v_max, rover.a_lat_max, ctrl.a_dec_profile, ctrl.kappa_window_m,
                                    ctrl.plateau_m)
        self.s_hint: float | None = None
        self.s_pred_hint: float | None = None
        self.steer_cmd = 0.0
        self.v_ref = 0.0
        self.i_v = 0.0
        self.i_cte = 0.0
        self.finished = False

    def step(self, x: float, y: float, yaw: float, v: float, steer_meas: float) -> dict:
        p, c, L = self.p, self.c, self.p.wheelbase
        pos = np.array([x, y])
        s, cte = self.path.project(pos, self.s_hint)
        self.s_hint = s
        # pose prediction over the actuator lag (exact arc of the kinematic bicycle)
        d = v * c.t_pred
        km = math.tan(steer_meas) / L
        if abs(km) < 1e-6:
            xp, yp, yawp = x + d * math.cos(yaw), y + d * math.sin(yaw), yaw
        else:
            yawp = yaw + km * d
            xp = x + (math.sin(yawp) - math.sin(yaw)) / km
            yp = y - (math.cos(yawp) - math.cos(yaw)) / km
        sp, ctep = self.path.project(np.array([xp, yp]), self.s_pred_hint if self.s_pred_hint is not None else s)
        self.s_pred_hint = sp
        ld = float(np.clip(c.k_ld * abs(v) + c.ld0, c.ld_min, c.ld_max))
        tgt = self.path.point_at(sp + ld)
        dx, dy = tgt[0] - xp, tgt[1] - yp
        lx = math.cos(yawp) * dx + math.sin(yawp) * dy
        ly = -math.sin(yawp) * dx + math.cos(yawp) * dy
        dist = max(math.hypot(lx, ly), 1e-3)
        alpha = math.atan2(ly, lx)
        kappa = 2.0 * math.sin(alpha) / dist
        if c.k_ff:
            kappa += c.k_ff * (self.path.kappa_at(sp + 0.5 * ld) - kappa)
        if c.ki_cte and abs(self.path.kappa_at(s)) < 1e-3 and s > 0:
            self.i_cte = float(np.clip(self.i_cte + ctep * abs(v) * self.dt, -0.5, 0.5))
            kappa -= c.ki_cte * self.i_cte
        delta_raw = math.atan(L * kappa)
        saturated = abs(delta_raw) >= p.steer_limit
        delta = float(np.clip(delta_raw, -p.steer_limit, p.steer_limit))
        dmax = p.steer_rate_max * self.dt
        self.steer_cmd = float(np.clip(delta, self.steer_cmd - dmax, self.steer_cmd + dmax))

        # longitudinal
        if s >= self.path.length - 0.05:
            self.finished = True
        if self.finished:
            v_target = 0.0
        else:
            v_target = max(float(np.interp(s, self.path.s, self.v_prof)), c.v_creep)
        self.v_ref = float(np.clip(v_target, self.v_ref - p.a_dec_max * self.dt,
                                   self.v_ref + p.a_lon_max * self.dt))
        if self.finished and self.v_ref <= 0.0:
            u = 0.0
            self.i_v = 0.0
        else:
            e = self.v_ref - v
            self.i_v = float(np.clip(self.i_v + c.ki_v * e * self.dt, -c.i_v_max, c.i_v_max))
            u = self.v_ref + c.kp_v * e + self.i_v
        return {"steer_cmd": self.steer_cmd, "v_cmd": self.v_ref, "u_wheel": u, "s": s, "cte": cte,
                "saturated": saturated, "ld": ld}


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass
class RoverRunResult:
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    yaw: np.ndarray
    v: np.ndarray
    v_cmd: np.ndarray
    steer: np.ndarray
    steer_cmd: np.ndarray
    wheel_omega: np.ndarray  # (n, 4) fl, fr, rl, rr  [rad/s]
    cross_track: np.ndarray
    s_along: np.ndarray
    label: np.ndarray
    summary: dict
    cross_track_front: np.ndarray = field(default_factory=lambda: np.zeros(0))
    s_front: np.ndarray = field(default_factory=lambda: np.zeros(0))
    roll: np.ndarray = field(default_factory=lambda: np.zeros(0))
    pitch: np.ndarray = field(default_factory=lambda: np.zeros(0))
    lane_settled: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))
    path_xy: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    path_s: np.ndarray = field(default_factory=lambda: np.zeros(0))
    path_labels: np.ndarray = field(default_factory=lambda: np.zeros(0, object))
    v_profile: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def to_csv(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["t", "x", "y", "z", "yaw", "v", "v_cmd", "steer", "steer_cmd", "cross_track", "s", "label"])
            for i in range(len(self.t)):
                w.writerow([f"{self.t[i]:.3f}", f"{self.x[i]:.4f}", f"{self.y[i]:.4f}", f"{self.z[i]:.4f}",
                            f"{self.yaw[i]:.5f}", f"{self.v[i]:.4f}", f"{self.v_cmd[i]:.4f}",
                            f"{self.steer[i]:.5f}", f"{self.steer_cmd[i]:.5f}", f"{self.cross_track[i]:.4f}",
                            f"{self.s_along[i]:.3f}", self.label[i]])
        return path


def _rms(a: np.ndarray) -> float:
    return float(np.sqrt(np.mean(a**2))) if a.size else float("nan")


def _amax(a: np.ndarray) -> float:
    return float(np.max(np.abs(a))) if a.size else float("nan")


# ---------------------------------------------------------------------------
# Closed loop
# ---------------------------------------------------------------------------


def _quat_from_mat(R: np.ndarray) -> np.ndarray:
    import mujoco

    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.ravel())
    return q


def _start_quat(ground: GroundSpec, x: float, y: float, yaw: float) -> tuple[np.ndarray, float]:
    h = 1.0
    if ground.kind == "hfield":
        dzdx = (ground.height_at(x + h, y) - ground.height_at(x - h, y)) / (2 * h)
        dzdy = (ground.height_at(x, y + h) - ground.height_at(x, y - h)) / (2 * h)
        n = np.array([-dzdx, -dzdy, 1.0])
    else:
        n = np.array([0.0, 0.0, 1.0])
    n /= np.linalg.norm(n)
    fwd = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    fwd -= n * fwd.dot(n)
    fwd /= np.linalg.norm(fwd)
    left = np.cross(n, fwd)
    R = np.column_stack([fwd, left, n])
    return _quat_from_mat(R), float(n[2])


def run_closed_loop(
    path_xy: np.ndarray,
    v_max: float = 1.4,
    terrain: str = "dem",
    start_pose: tuple[float, float, float] | None = None,
    dt_ctrl: float = 0.05,
    max_time: float | None = None,
    record_video_path: str | Path | None = None,
    segment_labels: Sequence[str] | None = None,
    *,
    rover: RoverParams | None = None,
    ctrl: ControllerParams | None = None,
    obstacles: Sequence[dict] | None = None,
    pose_noise: tuple[float, float] = (0.0, 0.0),
    settle_m: float = 4.0,
    video_speedup: float = 6.0,
    video_size: tuple[int, int] = (960, 544),
    seed: int = 0,
    ground: GroundSpec | None = None,
) -> RoverRunResult:
    """Drive the MuJoCo rover along ``path_xy`` (world metres) and log what happens.

    ``segment_labels``: one label per path point (e.g. 'lane' / 'headland' / 'detour').
    ``obstacles``: dicts ``{x, y, lx, ly, h, yaw}`` added as static collidable boxes.
    ``pose_noise``: (sigma_xy [m], sigma_yaw [rad]) white noise on the controller's
    pose measurement (RTK-GNSS-like); physics and logs use the true state.
    """
    import mujoco

    rover = rover or RoverParams()
    ctrl = ctrl or ControllerParams()
    wall0 = time.perf_counter()
    path = ReferencePath(path_xy, segment_labels)
    if ground is None:
        ground = make_ground(path.xy, terrain)
    xml = build_rover_mjcf(rover, ground, obstacles)
    model = mujoco.MjModel.from_xml_string(xml)
    if ground.kind == "hfield":
        model.hfield_data[:] = ground.heights01.astype(np.float32).ravel()
    data = mujoco.MjData(model)

    def jid(name: str) -> int:
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)

    root = jid("root")
    qa_root, va_root = model.jnt_qposadr[root], model.jnt_dofadr[root]
    qa_steer = [model.jnt_qposadr[jid(n)] for n in ("fl_steer", "fr_steer")]
    va_spin = [model.jnt_dofadr[jid(n)] for n in ("fl_spin", "fr_spin", "rl_spin", "rr_spin")]
    act = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i for i in range(model.nu)}
    a_steer = [act["steer_fl"], act["steer_fr"]]
    a_drive = [act["drive_fl"], act["drive_fr"], act["drive_rl"], act["drive_rr"]]
    wheel_bodies = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{n}_wheel") for n in ("fl", "fr", "rl", "rr")}
    wheel_geoms = np.array([g for g in range(model.ngeom)
                            if model.geom_bodyid[g] in wheel_bodies and model.geom_contype[g] != 0])
    deck_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "deck")
    ground_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
    obstacle_geoms = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"obstacle_{i}")
                               for i in range(len(obstacles or []))], dtype=int)

    L, T, r = rover.wheelbase, rover.track, rover.wheel_radius
    if start_pose is None:
        x0, y0 = path.xy[0]
        yaw0 = float(path.heading[0])
    else:
        x0, y0, yaw0 = start_pose
    quat, nz = _start_quat(ground, x0, y0, yaw0)
    data.qpos[qa_root:qa_root + 3] = [x0, y0, ground.height_at(x0, y0) + r / nz + 0.01]
    data.qpos[qa_root + 3:qa_root + 7] = quat
    mujoco.mj_forward(model, data)

    nsub = max(1, int(round(dt_ctrl / model.opt.timestep)))
    dt_ctrl = nsub * model.opt.timestep
    # settle onto the wheel contacts with brakes held
    data.ctrl[:] = 0.0
    mujoco.mj_step(model, data, int(round(1.0 / model.opt.timestep)))

    tracker = PathTracker(path, rover, ctrl, v_max, dt_ctrl)
    if max_time is None:
        mean_v = max(float(np.mean(np.maximum(tracker.v_prof, ctrl.v_creep))), 0.1)
        max_time = 1.6 * path.length / mean_v + 30.0
    n_max = int(math.ceil(max_time / dt_ctrl)) + 1
    rng = np.random.default_rng(seed)

    cols = ("t", "x", "y", "z", "yaw", "v", "v_cmd", "steer", "steer_cmd", "cte", "cte_f", "s", "s_f",
            "roll", "pitch", "wroll", "wpitch", "vz", "sat", "ncontact", "pen", "tau_drive", "tau_steer")
    log = {k: np.zeros(n_max) for k in cols}
    log_w = np.zeros((n_max, 4))
    labels: list[str] = []
    frames_q: list[np.ndarray] = []
    frame_dt = video_speedup / 24.0
    next_frame_t = 0.0
    reason = "timeout"
    s_front_hint = None
    deck_hits = 0
    obstacle_hits = 0
    k = 0
    t = 0.0
    stop_counter = 0
    for k in range(n_max):
        t = k * dt_ctrl
        qp = data.qpos
        pos = qp[qa_root:qa_root + 3].copy()
        R = data.xmat[model.jnt_bodyid[root]].reshape(3, 3)
        yaw = math.atan2(R[1, 0], R[0, 0])
        vel = data.qvel[va_root:va_root + 3]
        v = float(vel[0] * math.cos(yaw) + vel[1] * math.sin(yaw))
        wloc = data.qvel[va_root + 3:va_root + 6]
        dL, dR = float(qp[qa_steer[0]]), float(qp[qa_steer[1]])
        kL, kR = math.tan(dL) / L, math.tan(dR) / L
        k_meas = 0.5 * (kL / (1 + kL * T / 2) + kR / (1 - kR * T / 2))
        steer = math.atan(L * k_meas)

        xm, ym, yawm = pos[0], pos[1], yaw
        if pose_noise[0] > 0 or pose_noise[1] > 0:
            xm += rng.normal(0, pose_noise[0])
            ym += rng.normal(0, pose_noise[0])
            yawm += rng.normal(0, pose_noise[1])
        out = tracker.step(xm, ym, yawm, v, steer)
        s_true, cte_true = path.project(pos[:2], log["s"][k - 1] if k else None)
        front = pos[:2] + L * np.array([math.cos(yaw), math.sin(yaw)])
        s_front_hint, cte_f = path.project(front, s_front_hint if s_front_hint is not None else s_true)

        # contacts
        ncon = data.ncon
        nwheel, pen = 0, 0.0
        if ncon:
            g1 = data.contact.geom1[:ncon]
            g2 = data.contact.geom2[:ncon]
            dist = data.contact.dist[:ncon]
            touching = set(g1[np.isin(g1, wheel_geoms)]) | set(g2[np.isin(g2, wheel_geoms)])
            nwheel = len(touching)
            pen = float(max(0.0, -dist.min()))
            deck_hits += int(np.any(((g1 == deck_gid) & (g2 == ground_gid)) | ((g2 == deck_gid) & (g1 == ground_gid))))
            obstacle_hits += int(np.any(np.isin(g1, obstacle_geoms) | np.isin(g2, obstacle_geoms)))

        row = (t, pos[0], pos[1], pos[2], yaw, v, out["v_cmd"], steer, out["steer_cmd"], cte_true, cte_f,
               s_true, s_front_hint, math.atan2(R[2, 1], R[2, 2]), math.asin(float(np.clip(-R[2, 0], -1, 1))),
               wloc[0], wloc[1], vel[2], float(out["saturated"]), nwheel, pen,
               float(np.max(np.abs(data.actuator_force[a_drive]))), float(np.max(np.abs(data.actuator_force[a_steer]))))
        for key, val in zip(cols, row):
            log[key][k] = val
        log_w[k] = data.qvel[va_spin]
        labels.append(path.label_at(s_true))

        if record_video_path is not None and t >= next_frame_t - 1e-9:
            frames_q.append(data.qpos.copy())
            next_frame_t += frame_dt

        if R[2, 2] < 0.5:
            reason = "flipped"
            break
        if tracker.finished and abs(v) < 0.02 and out["v_cmd"] <= 0.0:
            stop_counter += 1
            if stop_counter >= 10:
                reason = "reached_end"
                break

        # actuators: Ackermann knuckle angles + electronic differential
        kc = math.tan(out["steer_cmd"]) / L
        data.ctrl[a_steer[0]] = math.atan(L * kc / (1 - kc * T / 2))
        data.ctrl[a_steer[1]] = math.atan(L * kc / (1 + kc * T / 2))
        u = out["u_wheel"]
        fl = u * math.hypot(1 - k_meas * T / 2, k_meas * L)
        fr = u * math.hypot(1 + k_meas * T / 2, k_meas * L)
        rl = u * (1 - k_meas * T / 2)
        rr = u * (1 + k_meas * T / 2)
        for a, vw in zip(a_drive, (fl, fr, rl, rr)):
            data.ctrl[a] = vw / r
        mujoco.mj_step(model, data, nsub)

    n = k + 1
    for key in log:
        log[key] = log[key][:n]
    log_w = log_w[:n]
    lab = np.asarray(labels[:n], dtype=object)
    wall_sim = time.perf_counter() - wall0

    # ---- metrics ----------------------------------------------------------------
    cte = log["cte"]

    def vertex(s_arr):
        return np.clip(np.searchsorted(path.s, s_arr, side="right") - 1, 0, len(path.s) - 1)

    with np.errstate(invalid="ignore"):
        settled = (lab == "lane") & (path.s_since_turn[vertex(log["s"])] >= settle_m)
        lab_f = path.labels[vertex(log["s_f"])]
        settled_f = (lab_f == "lane") & (path.s_since_turn[vertex(log["s_f"])] >= settle_m)
    moving = log["t"] >= 0
    final_s = float(log["s"][-1])
    completed = reason == "reached_end" and final_s >= path.length - 0.3
    xy = np.column_stack([log["x"], log["y"]])
    travelled = float(np.sum(np.hypot(*np.diff(xy, axis=0).T))) if n > 1 else 0.0
    cte_f = log["cte_f"]
    by_label = {}
    for lbl in sorted(set(lab.tolist()) | set(lab_f.tolist())):
        msk, mskf = lab == lbl, lab_f == lbl
        by_label[lbl] = {"rms_cte": _rms(cte[msk]), "max_cte": _amax(cte[msk]),
                         "rms_cte_front": _rms(cte_f[mskf]), "max_cte_front": _amax(cte_f[mskf]),
                         "n": int(msk.sum())}
    by_label["lane_settled"] = {"rms_cte": _rms(cte[settled]), "max_cte": _amax(cte[settled]),
                                "rms_cte_front": _rms(cte_f[settled_f]), "max_cte_front": _amax(cte_f[settled_f]),
                                "n": int(settled.sum()), "settle_m": settle_m}
    pen = log["pen"]
    summary = {
        "completed": bool(completed),
        "end_reason": reason,
        "terrain": "flat" if ground.kind == "plane" else "dem",
        "ground_geom": ground.kind,
        "duration_s": float(log["t"][-1]),
        "path_length_m": path.length,
        "final_s_m": final_s,
        "distance_travelled_m": travelled,
        "rms_cte": _rms(cte[moving]),
        "max_cte": _amax(cte[moving]),
        "rms_cte_by_label": {k_: v_["rms_cte"] for k_, v_ in by_label.items()},
        "max_cte_by_label": {k_: v_["max_cte"] for k_, v_ in by_label.items()},
        "by_label": by_label,
        "pct_time_steer_saturated": 100.0 * float(np.mean(log["sat"])),
        "mean_speed": float(np.mean(log["v"])),
        "max_speed": float(np.max(log["v"])),
        "physics_health": {
            "max_roll_deg": float(np.degrees(np.max(np.abs(log["roll"])))),
            "max_pitch_deg": float(np.degrees(np.max(np.abs(log["pitch"])))),
            "rms_roll_rate_dps": float(np.degrees(_rms(log["wroll"]))),
            "rms_pitch_rate_dps": float(np.degrees(_rms(log["wpitch"]))),
            "rms_vertical_speed_mps": _rms(log["vz"]),
            "min_wheels_in_contact": int(log["ncontact"].min()),
            "pct_ticks_all_4_wheels_in_contact": 100.0 * float(np.mean(log["ncontact"] >= 4)),
            "max_penetration_m": float(pen.max()),
            "median_penetration_m": float(np.median(pen)),
            "deck_ground_contact_ticks": int(deck_hits),
            "obstacle_contact_ticks": int(obstacle_hits),
        },
        "actuators": {
            "max_drive_torque_nm": float(log["tau_drive"].max()),
            "pct_ticks_drive_torque_saturated": 100.0 * float(np.mean(log["tau_drive"] >= 0.98 * rover.drive_torque_max)),
            "max_steer_torque_nm": float(log["tau_steer"].max()),
            "pct_ticks_steer_torque_saturated": 100.0 * float(np.mean(log["tau_steer"] >= 0.98 * rover.steer_torque_max)),
            "steer_cmd_vs_measured_rms_rad": _rms(log["steer_cmd"] - log["steer"]),
            "v_cmd_vs_measured_rms_mps": _rms(log["v_cmd"] - log["v"]),
        },
        "model": {
            "total_mass_kg": float(model.body_subtreemass[model.jnt_bodyid[root]]),
            "timestep_s": float(model.opt.timestep),
            "dt_ctrl_s": float(dt_ctrl),
            "integrator": "implicitfast",
            "wheel_collision": rover.wheel_geom,
            "steer_kp": rover.steer_kp,
            "steer_kv": rover.steer_kv,
            "steer_rate_max": rover.steer_rate_max,
            "steer_limit_bicycle_rad": rover.steer_limit,
            "drive_kv": rover.drive_kv,
            "drive_torque_max": rover.drive_torque_max,
            "nq": int(model.nq),
            "nv": int(model.nv),
        },
        "rover_params": {k_: (list(v_) if isinstance(v_, tuple) else v_) for k_, v_ in asdict(rover).items()},
        "controller_params": asdict(ctrl),
        "v_max": v_max,
        "pose_noise": list(pose_noise),
        "wall_time_sim_s": wall_sim,
        "realtime_factor": float(log["t"][-1] / max(wall_sim, 1e-9)),
    }

    result = RoverRunResult(
        t=log["t"], x=log["x"], y=log["y"], z=log["z"], yaw=log["yaw"], v=log["v"], v_cmd=log["v_cmd"],
        steer=log["steer"], steer_cmd=log["steer_cmd"], wheel_omega=log_w, cross_track=cte, s_along=log["s"],
        label=lab, summary=summary, cross_track_front=cte_f, s_front=log["s_f"], roll=log["roll"],
        pitch=log["pitch"],
        lane_settled=settled, path_xy=path.xy, path_s=path.s, path_labels=path.labels, v_profile=tracker.v_prof,
    )

    if record_video_path is not None and frames_q:
        w0 = time.perf_counter()
        render_chase_video(model, frames_q, result, ground, record_video_path, obstacles=obstacles,
                           frame_dt=frame_dt, size=video_size, root_qadr=qa_root)
        summary["wall_time_video_s"] = time.perf_counter() - w0
        summary["video_speedup"] = video_speedup
    summary["wall_time_total_s"] = time.perf_counter() - wall0
    return result


# ---------------------------------------------------------------------------
# Chase-cam video
# ---------------------------------------------------------------------------


def _font(size: int):
    from PIL import ImageFont

    try:
        from matplotlib import font_manager

        return ImageFont.truetype(font_manager.findfont("DejaVu Sans"), size)
    except Exception:  # pragma: no cover - font fallback
        return ImageFont.load_default()


def render_chase_video(model, frames_q: list[np.ndarray], res: RoverRunResult, ground: GroundSpec,
                       out_path: str | Path, obstacles: Sequence[dict] | None = None,
                       frame_dt: float = 0.25, size: tuple[int, int] = (960, 544),
                       root_qadr: int = 0) -> Path:
    """Replay recorded qpos snapshots through a smoothed chase camera -> H.264 mp4."""
    import imageio.v2 as imageio
    import mujoco
    from PIL import Image, ImageDraw

    W, H = (int(size[0]) // 16) * 16, (int(size[1]) // 16) * 16
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # map.znear / zfar / shadowclip are relative to the model extent (hundreds of m here)
    ext = float(model.stat.extent)
    model.vis.map.znear = 0.1 / ext
    model.vis.map.zfar = 4000.0 / ext
    model.vis.map.shadowclip = 30.0 / ext
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=H, width=W, max_geom=4000)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = 14.0
    cam.elevation = -24.0
    opt = mujoco.MjvOption()

    pz = ground.height_at(res.path_xy[:, 0], res.path_xy[:, 1]) + 0.06
    path3 = np.column_stack([res.path_xy, pz])
    tz = ground.height_at(res.x, res.y) + 0.07
    trail3 = np.column_stack([res.x, res.y, tz])
    font, font_s = _font(20), _font(15)

    # minimap (top-down, from logged physics positions)
    mm = 190
    allxy = np.vstack([res.path_xy, np.column_stack([res.x, res.y])])
    lo, hi = allxy.min(0) - 6, allxy.max(0) + 6
    scale = (mm - 12) / float(np.max(hi - lo))

    def to_mm(xy):
        q = (np.asarray(xy) - lo) * scale
        return np.column_stack([6 + q[:, 0], mm - 6 - q[:, 1]])

    path_mm = [tuple(p) for p in to_mm(res.path_xy[::4])]

    eye9 = np.eye(3).ravel()

    def add_line(scene, pts, rgba, width):
        for a, b in zip(pts[:-1], pts[1:]):
            if scene.ngeom >= scene.maxgeom:
                return
            g = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), eye9,
                                np.asarray(rgba, np.float32))
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width, a, b)
            scene.ngeom += 1

    writer = imageio.get_writer(str(out_path), fps=24, codec="libx264", quality=8, macro_block_size=16,
                                pixelformat="yuv420p", ffmpeg_params=["-preset", "fast"])
    yaw_s = None
    try:
        for i, q in enumerate(frames_q):
            data.qpos[:] = q
            mujoco.mj_forward(model, data)
            t = i * frame_dt
            j = int(np.clip(np.searchsorted(res.t, t), 0, len(res.t) - 1))
            pos = q[root_qadr:root_qadr + 3]
            yaw = float(res.yaw[j])
            if yaw_s is None:
                yaw_s = yaw
            dy = math.atan2(math.sin(yaw - yaw_s), math.cos(yaw - yaw_s))
            yaw_s += dy * (1 - math.exp(-frame_dt / 1.2))
            cam.lookat[:] = pos + np.array([1.8 * math.cos(yaw), 1.8 * math.sin(yaw), 0.6])
            cam.azimuth = math.degrees(yaw_s)
            renderer.update_scene(data, camera=cam, scene_option=opt)
            sc = renderer.scene
            s_now = res.s_along[j]
            near = np.abs(res.path_s - s_now) < 45.0
            idx = np.nonzero(near)[0][::3]
            add_line(sc, path3[idx], (0.15, 0.85, 1.0, 0.9), 0.05)
            j0 = max(0, j - 1200)
            add_line(sc, trail3[j0:j + 1:6], (1.0, 0.45, 0.05, 1.0), 0.07)
            img = Image.fromarray(renderer.render())
            dr = ImageDraw.Draw(img, "RGBA")
            dr.rectangle([10, 10, 470, 98], fill=(10, 12, 16, 170))
            dr.text((20, 16), f"MuJoCo physics  t = {res.t[j]:6.1f} s   [{res.label[j]}]", font=font,
                    fill=(235, 235, 235, 255))
            dr.text((20, 44), f"v = {res.v[j]:4.2f} m/s (cmd {res.v_cmd[j]:4.2f})   "
                              f"steer = {math.degrees(res.steer[j]):5.1f} deg (cmd {math.degrees(res.steer_cmd[j]):5.1f})",
                    font=font_s, fill=(200, 220, 255, 255))
            dr.text((20, 68), f"cross-track = {100 * res.cross_track[j]:+5.1f} cm   s = {s_now:6.1f} m"
                              f"   x{frame_dt * 24:.0f} speed", font=font_s, fill=(255, 200, 140, 255))
            ox, oy = W - mm - 10, H - mm - 10
            dr.rectangle([ox, oy, ox + mm, oy + mm], fill=(10, 12, 16, 190))
            dr.line([(ox + a, oy + b) for a, b in path_mm], fill=(40, 200, 255, 255), width=2)
            tr = to_mm(np.column_stack([res.x[:j + 1:5], res.y[:j + 1:5]]))
            if len(tr) > 1:
                dr.line([(ox + a, oy + b) for a, b in tr], fill=(255, 120, 20, 255), width=2)
            cxy = to_mm(np.array([[res.x[j], res.y[j]]]))[0]
            dr.ellipse([ox + cxy[0] - 4, oy + cxy[1] - 4, ox + cxy[0] + 4, oy + cxy[1] + 4], fill=(255, 255, 255, 255))
            writer.append_data(np.asarray(img.convert("RGB")))
    finally:
        writer.close()
        renderer.close()
    return out_path
