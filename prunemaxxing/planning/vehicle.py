"""Mothership vehicle model (kinematic bicycle + actuator dynamics) and controller.

The controller only sees the reference path and the *measured* state; it outputs
``steer_cmd`` and ``v_cmd``. The plant applies steering-rate limits, first-order
actuator lag, acceleration limits and small slip noise. Nothing snaps the state
back onto the path, so the logged trajectory is what the vehicle really did.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class VehicleParams:
    wheelbase_m: float = 2.4
    track_m: float = 1.7  # wheels ride in furrows two beds apart (2 x 0.85 m)
    wheel_radius_m: float = 0.45
    max_steer_rad: float = 0.38  # R_min = L / tan(0.38) ~ 6.0 m
    steer_rate_rad_s: float = 0.6
    steer_tau_s: float = 0.18
    accel_mps2: float = 0.4
    decel_mps2: float = 0.8
    speed_tau_s: float = 0.45
    v_lane: float = 1.1
    v_headland: float = 0.8
    v_transit: float = 1.9
    v_detour: float = 0.7
    a_lat_max: float = 0.3
    stop_decel_mps2: float = 0.25
    slip_std_m_per_sqrt_s: float = 0.006
    yaw_noise_std_rad_per_sqrt_s: float = 0.0015


@dataclass
class VehicleState:
    x: float  # rear-axle centre (UTM)
    y: float
    yaw: float
    v: float = 0.0
    steer: float = 0.0
    odo: float = 0.0  # travelled distance (wheel spin)

    def centre(self, wheelbase: float) -> tuple[float, float]:
        h = 0.5 * wheelbase
        return self.x + h * math.cos(self.yaw), self.y + h * math.sin(self.yaw)


SPEED_BY_LABEL = {
    "lane": "v_lane",
    "headland": "v_headland",
    "transit": "v_transit",
    "weed_detour": "v_detour",
    "obstacle_detour": "v_detour",
}


class PurePursuitController:
    """Adaptive-lookahead pure pursuit with curvature/stop-aware speed profile."""

    def __init__(self, params: VehicleParams, l0: float = 1.6, kv: float = 1.1, lmin: float = 2.2, lmax: float = 6.0):
        self.p = params
        self.l0, self.kv, self.lmin, self.lmax = l0, kv, lmin, lmax
        self.idx = 0

    def reset(self, idx: int = 0) -> None:
        self.idx = idx

    def nearest(self, path: np.ndarray, x: float, y: float, window: int = 160) -> int:
        lo = max(0, self.idx - 4)
        hi = min(len(path), self.idx + window)
        seg = path[lo:hi]
        d2 = (seg[:, 0] - x) ** 2 + (seg[:, 1] - y) ** 2
        self.idx = lo + int(np.argmin(d2))
        return self.idx

    def cross_track(self, path: np.ndarray, i: int, x: float, y: float) -> float:
        j = min(i + 1, len(path) - 1)
        k = max(j - 1, 0)
        a, b = path[k], path[j]
        t = b - a
        n = math.hypot(t[0], t[1])
        if n < 1e-9:
            return 0.0
        return float(((x - a[0]) * t[1] - (y - a[1]) * t[0]) / n) * -1.0  # +left

    def command(
        self,
        st: VehicleState,
        path: np.ndarray,
        s: np.ndarray,
        labels: np.ndarray,
        kappa: np.ndarray,
        s_stop: float,
    ) -> tuple[float, float, int, float]:
        i = self.nearest(path, st.x, st.y)
        cte = self.cross_track(path, i, st.x, st.y)
        ld = float(np.clip(self.l0 + self.kv * abs(st.v), self.lmin, self.lmax))
        s_target = min(s[i] + ld, s[-1])
        j = int(np.searchsorted(s, s_target))
        j = min(j, len(path) - 1)
        tx, ty = path[j]
        dx, dy = tx - st.x, ty - st.y
        alpha = math.atan2(dy, dx) - st.yaw
        alpha = (alpha + math.pi) % (2 * math.pi) - math.pi
        dist = max(math.hypot(dx, dy), 0.5)
        steer_cmd = math.atan2(2.0 * self.p.wheelbase_m * math.sin(alpha), dist)
        steer_cmd = float(np.clip(steer_cmd, -self.p.max_steer_rad, self.p.max_steer_rad))

        # speed: label limit + curvature preview + stopping profile
        prev = min(len(path) - 1, int(np.searchsorted(s, s[i] + 8.0)))
        lab_v = min(getattr(self.p, SPEED_BY_LABEL.get(labels[k], "v_lane")) for k in range(i, prev + 1, 4))
        kmax = float(np.max(np.abs(kappa[i : prev + 1]))) if prev > i else abs(float(kappa[i]))
        v_curv = math.sqrt(self.p.a_lat_max / kmax) if kmax > 1e-4 else 1e9
        remaining = max(0.0, s_stop - s[i])
        v_stop = math.sqrt(2.0 * self.p.stop_decel_mps2 * remaining)
        v_cmd = max(0.0, min(lab_v, v_curv, v_stop))
        # do not command a speed the steering cannot follow (heading error large)
        if abs(alpha) > 0.6:
            v_cmd = min(v_cmd, 0.4)
        return steer_cmd, v_cmd, i, cte


def step_vehicle(st: VehicleState, steer_cmd: float, v_cmd: float, p: VehicleParams, dt: float, rng) -> None:
    # steering actuator: first-order lag + rate limit
    d_steer = (steer_cmd - st.steer) / p.steer_tau_s
    d_steer = max(-p.steer_rate_rad_s, min(p.steer_rate_rad_s, d_steer))
    st.steer = max(-p.max_steer_rad, min(p.max_steer_rad, st.steer + d_steer * dt))
    # drive: lag + accel/decel limits
    dv = (v_cmd - st.v) / p.speed_tau_s
    dv = max(-p.decel_mps2, min(p.accel_mps2, dv))
    st.v = max(0.0, st.v + dv * dt)
    # kinematic bicycle (rear axle) + slip noise
    st.x += st.v * math.cos(st.yaw) * dt
    st.y += st.v * math.sin(st.yaw) * dt
    st.yaw += st.v / p.wheelbase_m * math.tan(st.steer) * dt
    if st.v > 0.05:
        lat = rng.normal(0.0, p.slip_std_m_per_sqrt_s * math.sqrt(dt))
        st.x += -math.sin(st.yaw) * lat
        st.y += math.cos(st.yaw) * lat
        st.yaw += rng.normal(0.0, p.yaw_noise_std_rad_per_sqrt_s * math.sqrt(dt))
    st.odo += st.v * dt
