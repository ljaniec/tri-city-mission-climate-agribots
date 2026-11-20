"""Airspace separation: rules shared by the scheduler and an audit over a finished mission.

Two flying drones are separated when they are at least ``H_SEP_M`` apart
horizontally *or* ``V_SEP_M`` apart vertically (a cylinder rule, like manned
traffic).  A drone parked on its pad (rotors stopped) only has to be cleared by
drones outside their own pad column -- the pad pitch itself is designed for
simultaneous parking.  Mothership clearance is measured from the drone's rotor
disc / skid envelope to the hull box, excluding the drone's own pad column.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from prunemaxxing.planning.swarm import DOCK_OFFSETS, PHASE_DOCKED

H_SEP_M = 4.0  # 2.2 m discs -> 1.8 m tip-to-tip gap
V_SEP_M = 1.5
PAD_H_SEP_M = 3.0  # adjacent pads (2.5 / 2.9 m pitch) ripple their take-offs and landings ...
PAD_V_SEP_M = 1.0  # ... by this height offset; diagonal pads (3.8 m) may work together
DRONE_RADIUS_M = 1.1  # 2.2 m tip-to-tip hexacopter
DRONE_BELOW_M = 0.8  # skids + winch cradle (paid out 0.35 m in transit) below the drone origin
# mothership keep-out hull in body frame (x fwd, y left, z up from ground), metres
MS_HULL = (-2.25, 2.35, -1.3, 1.3, 4.05)  # x0, x1, y0, y1, z_top (chute deflector lip)
MS_ENVELOPE = (-2.55, 2.55, -2.35, 2.35, 4.05)  # hull + rotor discs of drones parked on the pads
OWN_PAD_COLUMN_M = 0.6  # horizontal tolerance around a drone's own pad during climb / land


@dataclass
class SeparationReport:
    min_drone_drone_m: float = math.inf  # horizontal separation while vertically closer than V_SEP_M
    min_drone_ms_m: float = math.inf  # envelope-to-hull clearance outside the own pad column
    worst_pair: dict = field(default_factory=dict)
    worst_ms: dict = field(default_factory=dict)
    seconds_below: dict = field(default_factory=dict)  # threshold -> pair-seconds under it
    seconds_ms_below: dict = field(default_factory=dict)
    pad_conflict_s: float = 0.0  # pair-seconds breaking the pad-column ripple rule
    sampled_s: float = 0.0


def hull_clearance(px, py, pz, ms_x, ms_y, ms_yaw, ground_z) -> np.ndarray:
    """Gap (m) between the drone envelope and the mothership hull (negative = overlap)."""
    c, s = math.cos(ms_yaw), math.sin(ms_yaw)
    dx, dy = np.asarray(px) - ms_x, np.asarray(py) - ms_y
    bx, by = c * dx + s * dy, -s * dx + c * dy
    bz = np.asarray(pz) - ground_z
    x0, x1, y0, y1, zt = MS_HULL
    ex = np.maximum(np.maximum(x0 - bx, bx - x1), 0.0)
    ey = np.maximum(np.maximum(y0 - by, by - y1), 0.0)
    h_gap = np.hypot(ex, ey) - DRONE_RADIUS_M
    v_gap = bz - DRONE_BELOW_M - zt
    return np.where((h_gap > 0) & (v_gap > 0), np.hypot(h_gap, v_gap), np.maximum(h_gap, v_gap))


def conflict(pa: np.ndarray, pb: np.ndarray, both_pad=False, slack_h=0.0, slack_v=0.0) -> np.ndarray:
    """Element-wise loss of separation between (..., 3) positions.

    ``both_pad`` marks samples where both drones are inside their own pad
    columns (precision lock, slow vertical motion): neighbouring pads closer
    than ``PAD_H_SEP_M`` then only need a ``PAD_V_SEP_M`` height offset.
    ``slack_h`` / ``slack_v`` widen the minima (planning between samples)."""
    h_min = np.where(both_pad, PAD_H_SEP_M, H_SEP_M) + slack_h
    v_min = np.where(both_pad, PAD_V_SEP_M, V_SEP_M) + slack_v
    h = np.hypot(pa[..., 0] - pb[..., 0], pa[..., 1] - pb[..., 1])
    return (h < h_min) & (np.abs(pa[..., 2] - pb[..., 2]) < v_min)


def track_velocity(t: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    return np.gradient(xyz, t, axis=0) if len(t) > 1 else np.zeros_like(xyz)


def dwell_tracks(res, visit: dict, dt: float = 0.25):
    """(t, pos (n_drones, len(t), 3), airborne, own-pad-column mask, ground z at the mothership)."""
    S = res.sorties
    t = np.arange(visit["t_arrive"], visit["t_depart"] + dt, dt)
    nd = res.n_drones
    gz = float(res.ground_z(visit["x"], visit["y"]))
    pos = np.zeros((nd, len(t), 3))
    air = np.zeros((nd, len(t)), dtype=bool)
    col = np.zeros((nd, len(t)), dtype=bool)
    c, s = math.cos(visit["yaw"]), math.sin(visit["yaw"])
    for d in range(nd):
        off = DOCK_OFFSETS[d % len(DOCK_OFFSETS)]
        dk = np.array([visit["x"] + c * off[0] - s * off[1], visit["y"] + s * off[0] + c * off[1]])
        pos[d, :, 0], pos[d, :, 1] = dk
        pos[d, :, 2] = float(res.ground_z(*dk)) + res.cfg.drone.deck_height_m
    for k in np.flatnonzero(S["station"] == visit["sid"]):
        d = int(S["drone"][k])
        tr = res.trajectory(int(k))
        m = (t >= tr.t[0]) & (t <= tr.t[-1])
        if not m.any():
            continue
        for j in range(3):
            pos[d, m, j] = np.interp(t[m], tr.t, tr.actual[:, j])
        ph = tr.phase[np.clip(np.searchsorted(tr.t, t[m]), 0, len(tr.t) - 1)]
        air[d, m] = ph != PHASE_DOCKED
        col[d, m] = np.hypot(pos[d, m, 0] - S["dock_x"][k], pos[d, m, 1] - S["dock_y"][k]) < OWN_PAD_COLUMN_M
    return t, pos, air, col, gz


def separation_report(res, stations=None, dt: float = 0.25, thresholds=(2.5, 3.0, 4.0),
                      ms_thresholds=(0.5, 1.0, 1.5)) -> SeparationReport:
    rep = SeparationReport(seconds_below={th: 0.0 for th in thresholds},
                           seconds_ms_below={th: 0.0 for th in ms_thresholds})
    visits = [v for v in res.visits if v.get("sorties", 0) > 0]
    if stations is not None:
        keep = set(stations)
        visits = [v for v in visits if v["sid"] in keep]
    nd = res.n_drones
    for v in visits:
        t, pos, air, col, gz = dwell_tracks(res, v, dt)
        rep.sampled_s += t[-1] - t[0]
        for a in range(nd):
            for b in range(a + 1, nd):
                both = air[a] & air[b]
                pad = both & col[a] & col[b]
                rep.pad_conflict_s += float(np.count_nonzero(pad & conflict(pos[a], pos[b], both_pad=True))) * dt
                act = (both & ~pad) | (air[a] & ~air[b] & ~col[a]) | (air[b] & ~air[a] & ~col[b])
                act &= np.abs(pos[a, :, 2] - pos[b, :, 2]) < V_SEP_M
                if not act.any():
                    continue
                h = np.hypot(pos[a, :, 0] - pos[b, :, 0], pos[a, :, 1] - pos[b, :, 1])
                h = np.where(act, h, np.inf)
                i = int(np.argmin(h))
                for th in thresholds:
                    rep.seconds_below[th] += float(np.count_nonzero(h < th)) * dt
                if h[i] < rep.min_drone_drone_m:
                    rep.min_drone_drone_m = float(h[i])
                    rep.worst_pair = {"sid": v["sid"], "t": float(t[i]), "drones": (a, b),
                                      "pa": pos[a, i].round(2).tolist(), "pb": pos[b, i].round(2).tolist()}
        for d in range(nd):
            m = air[d] & ~col[d]
            if not m.any():
                continue
            clr = hull_clearance(pos[d, :, 0], pos[d, :, 1], pos[d, :, 2], v["x"], v["y"], v["yaw"], gz)
            clr = np.where(m, clr, np.inf)
            i = int(np.argmin(clr))
            for th in ms_thresholds:
                rep.seconds_ms_below[th] += float(np.count_nonzero(clr < th)) * dt
            if clr[i] < rep.min_drone_ms_m:
                rep.min_drone_ms_m = float(clr[i])
                rep.worst_ms = {"sid": v["sid"], "t": float(t[i]), "drone": d, "p": pos[d, i].round(2).tolist()}
    return rep
