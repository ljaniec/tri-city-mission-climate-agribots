"""Legacy v1 tracker (snaps to furrows). Superseded by :mod:`prunemaxxing.planning.vehicle`."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from prunemaxxing.config import MOTHERSHIP_TURN_RADIUS_M
from prunemaxxing.planning.bed_geometry import BedLayout, snap_easting_to_furrow


@dataclass
class UGVState:
    easting: float
    northing: float
    yaw_rad: float
    speed_mps: float = 1.2


@dataclass
class TrackerConfig:
    lookahead_m: float = 8.0
    max_speed_mps: float = 1.6
    min_speed_mps: float = 0.4
    wheelbase_m: float = 2.4
    max_steer_rad: float = 0.55
    dt: float = 0.2
    cross_track_gain: float = 0.35
    bed_lock: bool = True


@dataclass
class TrackSample:
    t: float
    easting: float
    northing: float
    yaw_rad: float
    speed_mps: float
    steer_rad: float
    cross_track_m: float
    planned_easting: float
    planned_northing: float


@dataclass
class PathTracker:
    planned: np.ndarray  # Nx2 easting, northing (bed-snapped)
    layout: BedLayout | None = None
    cfg: TrackerConfig = field(default_factory=TrackerConfig)
    state: UGVState | None = None
    history: list[TrackSample] = field(default_factory=list)
    _wp_idx: int = 0

    def __post_init__(self) -> None:
        if self.state is None and len(self.planned):
            p0 = self.planned[0]
            yaw = 0.0
            if len(self.planned) > 1:
                d = self.planned[1] - self.planned[0]
                yaw = float(np.arctan2(d[1], d[0]))
            self.state = UGVState(float(p0[0]), float(p0[1]), yaw)

    def _nearest_index(self, e: float, n: float) -> int:
        d2 = (self.planned[:, 0] - e) ** 2 + (self.planned[:, 1] - n) ** 2
        return int(np.argmin(d2))

    def _lookahead_point(self, e: float, n: float) -> tuple[np.ndarray, float]:
        i0 = self._nearest_index(e, n)
        self._wp_idx = max(self._wp_idx, i0)
        target = self.planned[min(self._wp_idx, len(self.planned) - 1)]
        acc = 0.0
        i = self._wp_idx
        while i < len(self.planned) - 1 and acc < self.cfg.lookahead_m:
            seg = self.planned[i + 1] - self.planned[i]
            acc += float(np.linalg.norm(seg))
            i += 1
            target = self.planned[i]
        # cross-track to nearest segment
        nearest = self.planned[self._nearest_index(e, n)]
        ct = float(np.hypot(e - nearest[0], n - nearest[1]))
        return target, ct

    def step(self, t: float) -> TrackSample:
        assert self.state is not None
        e, n, yaw = self.state.easting, self.state.northing, self.state.yaw_rad
        target, ct = self._lookahead_point(e, n)

        # Pure pursuit curvature
        dx = target[0] - e
        dy = target[1] - n
        local_x = np.cos(yaw) * dx + np.sin(yaw) * dy
        local_y = -np.sin(yaw) * dx + np.cos(yaw) * dy
        ld = max(np.hypot(local_x, local_y), 1e-3)
        curvature = 2.0 * local_y / (ld**2)
        steer = float(
            np.clip(
                np.arctan(self.cfg.wheelbase_m * curvature)
                + self.cfg.cross_track_gain * np.tanh(ct) * np.sign(local_y + 1e-9),
                -self.cfg.max_steer_rad,
                self.cfg.max_steer_rad,
            )
        )

        # Slow near sharp steer / high cross-track (protect beds)
        speed = self.cfg.max_speed_mps * (1.0 - 0.55 * abs(steer) / self.cfg.max_steer_rad)
        speed = float(np.clip(speed, self.cfg.min_speed_mps, self.cfg.max_speed_mps))
        if ct > 1.5:
            speed *= 0.6

        # Bicycle model integration
        yaw += (speed / self.cfg.wheelbase_m) * np.tan(steer) * self.cfg.dt
        e += speed * np.cos(yaw) * self.cfg.dt
        n += speed * np.sin(yaw) * self.cfg.dt

        # Bed lock: snap toward the *lookahead furrow* (allows headland lane changes
        # without mounting bed crowns once settled on a lane).
        if self.cfg.bed_lock and self.layout is not None:
            target_furrow = snap_easting_to_furrow(float(target[0]), self.layout)
            # Soft attract to target furrow; hard clamp if drifting onto a crown
            e = 0.65 * e + 0.35 * target_furrow
            e = snap_easting_to_furrow(e, self.layout)

        self.state = UGVState(e, n, float(yaw), speed)
        sample = TrackSample(
            t=t,
            easting=e,
            northing=n,
            yaw_rad=float(yaw),
            speed_mps=speed,
            steer_rad=steer,
            cross_track_m=ct,
            planned_easting=float(target[0]),
            planned_northing=float(target[1]),
        )
        self.history.append(sample)
        return sample

    def run_until_end(self, max_steps: int = 5000) -> list[TrackSample]:
        if self.planned.size == 0:
            return []
        goal = self.planned[-1]
        t = 0.0
        for _ in range(max_steps):
            s = self.step(t)
            t += self.cfg.dt
            # Only finish after the lookahead index has reached the final waypoint.
            if self._wp_idx >= len(self.planned) - 1:
                if np.hypot(s.easting - goal[0], s.northing - goal[1]) < 2.5:
                    break
        return self.history

    @property
    def actual_polyline(self) -> np.ndarray:
        if not self.history:
            return np.zeros((0, 2))
        return np.array([[h.easting, h.northing] for h in self.history], dtype=float)


def min_turn_radius_from_steer(wheelbase: float, max_steer: float) -> float:
    return wheelbase / max(np.tan(max_steer), 1e-6)


# Keep constant referenced for docs / tests
_ = MOTHERSHIP_TURN_RADIUS_M
