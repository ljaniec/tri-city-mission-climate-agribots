"""Drone sortie model: timed legs, spread patterns, wind-disturbed actual tracks.

A sortie = load on the mothership deck -> vertical climb in the pad column to the
drone's cruise layer -> transit to the patch (descending to work altitude) ->
application pattern -> transit back -> vertical descent onto the pad.  Port pads
cruise on the low layer, starboard pads on the high one, so crossing transits
are vertically separated; near the pad the RTK / visual lock holds the column.

* DUST: 2.5 m x 2.5 m zig-zag over a sub-patch of the cell (spinner spreader).
* CARTRIDGE: a straight line along one bed ridge of the cell, dropping 16 pucks.

Timing is analytic (trapezoidal velocity profiles) so a 100 h mission simulates
in seconds; full trajectories are regenerated deterministically on demand
(seeded by sortie id) for maps, dashboards and MuJoCo replay.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import lfilter
from shapely.geometry import LineString, Point, Polygon

from prunemaxxing.planning.payload_modes import PayloadMode, PayloadSpec

PHASE_DOCKED, PHASE_CLIMB, PHASE_OUT, PHASE_APPLY, PHASE_BACK, PHASE_LAND = range(6)
PHASE_NAMES = ("docked", "climb", "outbound", "apply", "return", "land")


@dataclass
class DroneParams:
    cruise_agl_m: float = 8.0  # low (port) layer
    layer_step_m: float = 2.0  # starboard pads cruise this much higher
    pad_lock_m: float = 6.0  # gust error fades out within this radius of the pad (precision landing lock)
    climb_mps: float = 2.5
    descend_mps: float = 1.8
    v_loaded_mps: float = 7.0
    v_empty_mps: float = 9.0
    accel_mps2: float = 2.5
    battery_kwh: float = 1.6
    swap_threshold: float = 0.25
    swap_time_s: float = 25.0
    turnaround_s: float = 1.5
    p_empty_kw: float = 4.2
    p_per_kg_kw: float = 0.17
    deck_height_m: float = 2.80  # drone origin above ground when docked (pad top 2.36 + skid 0.44)
    wind_gust_std_mps: float = 1.6
    wind_tau_s: float = 4.0
    pos_ctrl_tau_s: float = 0.9
    gust_err_gain: float = 0.5  # RTK position loop rejects half of the gust-driven drift
    sample_dt_s: float = 0.25


# Dock pads relative to mothership origin (x fwd, y left), metres.  The 2.9 m x 2.5 m pad
# pitch leaves >= 0.5 m between parked rotor discs (2.2 m tip-to-tip).
DOCK_OFFSETS = np.array([[1.45, 1.25], [1.45, -1.25], [-1.45, 1.25], [-1.45, -1.25]])  # = sim.world_robots.DOCKS
DOCK_LAYER = (0, 1, 1, 0)  # cruise layer per pad: diagonal pads share one, adjacent pads are 2 m apart


def cruise_agl(p: DroneParams, k: int) -> float:
    return p.cruise_agl_m + p.layer_step_m * DOCK_LAYER[k % len(DOCK_LAYER)]


def dock_xy(ms_x: float, ms_y: float, ms_yaw: float, k: int) -> np.ndarray:
    off = DOCK_OFFSETS[k % len(DOCK_OFFSETS)]
    c, s = math.cos(ms_yaw), math.sin(ms_yaw)
    return np.array([ms_x + c * off[0] - s * off[1], ms_y + s * off[0] + c * off[1]])


GLIDE_SLOPE = 1.0  # transits stay on the cruise layer and descend / climb at 45 deg next to the patch
LEVEL_NEAR_PAD_M = 3.0  # ... and are level for at least this far out from the pad column


def _glide_run(h: float, drop: float) -> float:
    """Horizontal length of the glide between the cruise layer and the patch."""
    g = drop / GLIDE_SLOPE
    return min(h, g, max(h - LEVEL_NEAR_PAD_M, 0.5 * g))


def _transit_len(h: float, drop: float) -> float:
    g = _glide_run(h, drop)
    return (h - g) + math.hypot(g, drop)


def _transit_poly(top: np.ndarray, end: np.ndarray) -> np.ndarray:
    """Level at ``top`` height above the pad, then a glide onto ``end`` (both (3,) points)."""
    dxy = end[:2] - top[:2]
    h = float(np.hypot(*dxy))
    g = _glide_run(h, max(top[2] - end[2], 0.0))
    if h - g < 0.05:
        return np.vstack([top, end])
    q = np.array([*(top[:2] + dxy * (1.0 - g / h)), top[2]])
    return np.vstack([top, q, end])


def _leg_time(d: float, v: float, a: float) -> float:
    if d <= 1e-9:
        return 0.0
    if d >= v * v / a:
        return d / v + v / a
    return 2.0 * math.sqrt(d / a)


def _leg_profile(d: float, v: float, a: float, t: np.ndarray) -> np.ndarray:
    """Distance travelled at times t along a trapezoidal profile of length d."""
    if d <= 1e-9:
        return np.zeros_like(t)
    if d >= v * v / a:
        ta = v / a
        T = d / v + ta
        s = np.where(
            t < ta,
            0.5 * a * t**2,
            np.where(t < T - ta, 0.5 * a * ta**2 + v * (t - ta), d - 0.5 * a * np.maximum(T - t, 0) ** 2),
        )
    else:
        ta = math.sqrt(d / a)
        T = 2 * ta
        s = np.where(t < ta, 0.5 * a * t**2, d - 0.5 * a * np.maximum(T - t, 0) ** 2)
    return np.clip(s, 0.0, d)


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------


def cell_patches(cell_poly: Polygon, cx: float, cy: float, mode: PayloadMode, bed_x0: float,
                 bed_spacing: float, n_needed: int) -> list[np.ndarray]:
    """Ordered list of application patterns (each (k,2) polyline) covering the cell."""
    minx, miny, maxx, maxy = cell_poly.bounds
    out: list[np.ndarray] = []
    if mode is PayloadMode.CARTRIDGE:
        k0 = math.ceil((minx - bed_x0) / bed_spacing)
        k1 = math.floor((maxx - bed_x0) / bed_spacing)
        ridges = [bed_x0 + k * bed_spacing for k in range(k0, k1 + 1)]
        # interleave so successive sorties spread across the cell
        order = ridges[::2] + ridges[1::2]
        for x in order:
            ln = LineString([(x, miny + 0.4), (x, maxy - 0.4)]).intersection(cell_poly)
            if ln.is_empty or ln.length < 1.0:
                continue
            if ln.geom_type != "LineString":
                ln = max(ln.geoms, key=lambda g: g.length)
            out.append(np.asarray(ln.coords))
    else:
        n = 4
        w = (maxx - minx) / n
        h = (maxy - miny) / n
        centres = []
        for j in range(n):
            cols = range(n) if j % 2 == 0 else range(n - 1, -1, -1)
            for i in cols:
                centres.append((minx + (i + 0.5) * w, miny + (j + 0.5) * h))
        # visit patches in a spread-out order
        order = centres[::3] + centres[1::3] + centres[2::3]
        for px, py in order:
            if not cell_poly.contains(Point(px, py)):
                continue
            hw, hh = 0.42 * w, 0.42 * h
            xs = [px - hw, px - hw, px, px, px + hw, px + hw]
            ys = [py - hh, py + hh, py + hh, py - hh, py - hh, py + hh]
            out.append(np.column_stack([xs, ys]))
    if not out:
        out = [np.array([[cx, cy - 1.0], [cx, cy + 1.0]])]
    return out


# ---------------------------------------------------------------------------
# Sortie timing + trajectory synthesis
# ---------------------------------------------------------------------------


@dataclass
class SortieTiming:
    t_climb: float
    t_out: float
    t_apply: float
    t_back: float
    t_land: float
    out_dist: float
    back_dist: float
    apply_dist: float

    @property
    def flight_time(self) -> float:
        return self.t_climb + self.t_out + self.t_apply + self.t_back + self.t_land


def _apply_length(pattern: np.ndarray) -> float:
    return float(np.sum(np.hypot(*np.diff(pattern, axis=0).T))) if len(pattern) > 1 else 0.0


def sortie_timing(dock: np.ndarray, pattern: np.ndarray, spec: PayloadSpec, p: DroneParams,
                  cruise_m: float | None = None) -> SortieTiming:
    cruise = p.cruise_agl_m if cruise_m is None else cruise_m
    climb_h = cruise - p.deck_height_m
    out_d = _transit_len(math.hypot(*(pattern[0] - dock)), cruise - spec.apply_alt_agl_m)
    back_d = _transit_len(math.hypot(*(pattern[-1] - dock)), cruise - spec.apply_alt_agl_m)
    ap = _apply_length(pattern)
    return SortieTiming(
        t_climb=_leg_time(climb_h, p.climb_mps, p.accel_mps2),
        t_out=_leg_time(out_d, p.v_loaded_mps, p.accel_mps2),
        t_apply=_leg_time(ap, spec.apply_speed_mps, 1.5) + (0.0 if spec.units_per_sortie > 1 else 2.0),
        t_back=_leg_time(back_d, p.v_empty_mps, p.accel_mps2),
        t_land=_leg_time(climb_h, p.descend_mps, 1.2),
        out_dist=out_d,
        back_dist=back_d,
        apply_dist=ap,
    )


def sortie_energy_kwh(tm: SortieTiming, payload_kg: float, p: DroneParams) -> float:
    p_load = p.p_empty_kw + p.p_per_kg_kw * payload_kg
    loaded = tm.t_climb + tm.t_out + 0.5 * tm.t_apply
    empty = 0.5 * tm.t_apply + tm.t_back + tm.t_land
    return (p_load * loaded + p.p_empty_kw * empty) / 3600.0


@dataclass
class SortieTrajectory:
    t: np.ndarray  # absolute times
    planned: np.ndarray  # (k,3)
    actual: np.ndarray  # (k,3)
    phase: np.ndarray  # (k,) int
    payload_frac: np.ndarray
    drops: np.ndarray  # (m,3) cartridge drop points (actual), (0,3) for dust
    drop_t: np.ndarray
    yaw: np.ndarray | None = None  # unwrapped heading, set by :func:`smooth_heading`


def smooth_heading(tr: SortieTrajectory, dock_yaw: float, sigma_s: float = 1.2) -> np.ndarray:
    """Heading that follows the transit legs, holds still while applying, and
    takes off / lands aligned with the mothership."""
    t = tr.t - tr.t[0]
    v = np.gradient(tr.actual[:, :2], t, axis=0)
    hdg = np.arctan2(v[:, 1], v[:, 0])
    transit = np.isin(tr.phase, (PHASE_OUT, PHASE_BACK)) & (np.hypot(v[:, 0], v[:, 1]) > 1.0)
    raw = np.full(len(t), np.nan)
    raw[transit] = hdg[transit]
    raw[np.isin(tr.phase, (PHASE_CLIMB, PHASE_LAND))] = dock_yaw
    raw[0] = raw[-1] = dock_yaw
    for i in range(1, len(raw)):  # hold through apply / slow segments
        if np.isnan(raw[i]):
            raw[i] = raw[i - 1]
    raw = np.unwrap(raw)
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.25
    out = gaussian_filter1d(raw, sigma_s / max(dt, 1e-3), mode="nearest")
    return out - (out[0] - dock_yaw)


def synthesize_trajectory(
    sortie_id: int,
    t_launch: float,
    dock: np.ndarray,
    pattern: np.ndarray,
    spec: PayloadSpec,
    p: DroneParams,
    ground_z,  # callable (x, y) -> z, vectorised
    wind_mean: np.ndarray,
    cruise_m: float | None = None,
) -> SortieTrajectory:
    cruise = p.cruise_agl_m if cruise_m is None else cruise_m
    dt = p.sample_dt_s
    z_dock = float(ground_z(dock[0], dock[1])) + p.deck_height_m
    gz_pat = np.asarray(ground_z(pattern[:, 0], pattern[:, 1]), dtype=float)
    cruise_dock = float(ground_z(dock[0], dock[1])) + cruise
    work = gz_pat + spec.apply_alt_agl_m
    p0 = np.array([dock[0], dock[1], z_dock])
    p1 = np.array([dock[0], dock[1], cruise_dock])
    pat3 = np.column_stack([pattern, work])
    legs = [
        (np.vstack([p0, p1]), p.climb_mps, p.accel_mps2, PHASE_CLIMB),
        (_transit_poly(p1, pat3[0]), p.v_loaded_mps, p.accel_mps2, PHASE_OUT),
        (pat3, spec.apply_speed_mps, 1.5, PHASE_APPLY),
        (_transit_poly(p1, pat3[-1])[::-1], p.v_empty_mps, p.accel_mps2, PHASE_BACK),
        (np.vstack([p1, p0]), p.descend_mps, 1.2, PHASE_LAND),
    ]
    ts, pts, phs = [], [], []
    t_cursor = 0.0
    for poly, v, a, ph in legs:
        seg = np.diff(poly, axis=0)
        seg_len = np.linalg.norm(seg, axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg_len)])
        d = float(cum[-1])
        T = _leg_time(d, v, a)
        if ph == PHASE_APPLY and spec.units_per_sortie <= 1:
            T += 2.0  # hover while the spinner empties the bin
        n = max(2, int(math.ceil(T / dt)) + 1)
        tt = np.linspace(0.0, T, n)
        sd = _leg_profile(d, v, a, np.minimum(tt, T - (2.0 if (ph == PHASE_APPLY and spec.units_per_sortie <= 1) else 0.0)))
        xyz = np.column_stack([np.interp(sd, cum, poly[:, k]) for k in range(3)])
        if ts:
            tt, xyz = tt[1:], xyz[1:]
        ts.append(t_cursor + tt)
        pts.append(xyz)
        phs.append(np.full(len(tt), ph))
        t_cursor += T
    t_rel = np.concatenate(ts)
    planned = np.vstack(pts)
    phase = np.concatenate(phs)

    # actual: closed-loop lag rounds corners + gust-driven position error
    rng = np.random.default_rng(1_000_003 + sortie_id)
    sig = p.pos_ctrl_tau_s / dt
    actual = np.column_stack([gaussian_filter1d(planned[:, k], sig, mode="nearest") for k in range(3)])
    actual[0], actual[-1] = planned[0], planned[-1]
    a_ou = math.exp(-dt / p.wind_tau_s)
    white = rng.normal(0.0, p.wind_gust_std_mps * math.sqrt(1 - a_ou**2), size=(len(t_rel), 2))
    gust = np.column_stack([lfilter([1.0], [1.0, -a_ou], white[:, k]) for k in range(2)])
    gust += 0.15 * wind_mean  # imperfect wind feed-forward
    a_c = math.exp(-dt / p.pos_ctrl_tau_s)
    err = np.column_stack([lfilter([(1 - a_c) * p.pos_ctrl_tau_s], [1.0, -a_c], gust[:, k]) for k in range(2)])
    airborne = (phase != PHASE_CLIMB) | (t_rel > 1.0)
    w = np.clip(np.minimum(t_rel, t_rel[-1] - t_rel) / 2.0, 0.0, 1.0) * airborne
    pad_d = np.hypot(planned[:, 0] - dock[0], planned[:, 1] - dock[1])
    w = w * np.clip(pad_d / p.pad_lock_m, 0.08, 1.0) * p.gust_err_gain
    actual[:, :2] += err * w[:, None]
    actual[:, 2] += 0.12 * np.sin(t_rel * 1.7 + rng.uniform(0, 6)) * w

    # payload + cartridge drops
    pf = np.ones(len(t_rel))
    ap = np.where(phase == PHASE_APPLY)[0]
    drops = np.zeros((0, 3))
    drop_t = np.zeros(0)
    if len(ap):
        a0, a1 = t_rel[ap[0]], t_rel[ap[-1]]
        pf = np.where(t_rel < a0, 1.0, np.where(t_rel > a1, 0.0, 1.0 - (t_rel - a0) / max(a1 - a0, 1e-6)))
        if spec.units_per_sortie > 1:
            k = spec.units_per_sortie
            frac = (np.arange(k) + 0.5) / k
            drop_t = a0 + frac * (a1 - a0)
            drops = np.column_stack([np.interp(drop_t, t_rel, actual[:, j]) for j in range(3)])
            pf = 1.0 - np.searchsorted(drop_t, t_rel, side="right") / k
    return SortieTrajectory(t_launch + t_rel, planned, actual, phase, pf, drops, t_launch + drop_t)
