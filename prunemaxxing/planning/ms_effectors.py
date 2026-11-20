"""Mothership near-field effectors: snow-blower chip thrower + pneumatic air cannons.

Both reach up to 40 ft (12.19 m) from the mothership.  While the swarm works a
station, every application patch of the station's cells that lies entirely
inside the reach annulus is served by the mothership itself:

* DUST: an impeller (snow-blower style) at the hopper outlet throws chips up a
  slewing discharge chute with an adjustable deflector; the aim point sweeps
  the patch zig-zag while a burst is thrown.
* CARTRIDGE: two pneumatic air cannons (port / starboard turrets) fire the
  pressed seed cartridges one by one onto the bed ridge, alternating barrels
  while the other recharges.

Geometry constants (body frame of the mothership: origin at ground level
between the axles, +x forward, +y left) are shared with the MuJoCo model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

G = 9.81
FT = 0.3048
REACH_M = 40.0 * FT  # 12.19 m

KIND_BLOWER, KIND_CANNON = 0, 1
KIND_NAMES = ("blower", "air_cannon")

CHUTE_PIVOT = (0.0, 0.0, 2.95)  # slewing ring on top of the discharge column
CHUTE_TIP = (0.0, 0.0, 3.62)  # nominal throw origin (deflector lip ~0.4 m out, rotates with the chute)
CANNON_PIVOTS = ((0.0, 1.16, 2.05), (0.0, -1.16, 2.05))  # port, starboard elevation trunnions
CANNON_BARREL_M = 1.05  # trunnion -> muzzle


@dataclass
class EffectorParams:
    enabled: bool = True
    range_max_m: float = REACH_M
    range_min_m: float = 3.0  # cannot throw onto its own wheels / furrows
    # chip blower
    blower_rate_kg_s: float = 1.2  # 4.3 t/h impeller throughput
    blower_burst_kg: float = 50.0
    blower_spool_s: float = 2.5  # impeller spin-up after idling
    blower_elev_deg: float = 35.0
    chute_slew_dps: float = 60.0
    # air cannons
    cannon_cycle_s: float = 1.6  # per barrel: valve + receiver recharge
    cannon_elev_deg: float = 30.0
    cannon_slew_dps: float = 90.0
    cannon_sigma_m: float = 0.15  # landing scatter = sigma + per_m * range
    cannon_sigma_per_m: float = 0.012
    cannon_arc_deg: tuple[float, float] = (45.0, 135.0)  # off the bow; shots pass above / outside parked rotors
    unit_kg: float = 0.5  # one cartridge


def mount_world(ms_x: float, ms_y: float, ms_yaw: float, local: tuple[float, float, float]) -> np.ndarray:
    c, s = math.cos(ms_yaw), math.sin(ms_yaw)
    return np.array([ms_x + c * local[0] - s * local[1], ms_y + s * local[0] + c * local[1]])


def reachable(pattern: np.ndarray, origin_xy: np.ndarray, p: EffectorParams) -> bool:
    seg = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(pattern, axis=0).T))])
    u = np.linspace(0.0, seg[-1], max(2, int(seg[-1] / 0.25) + 1))
    xs, ys = np.interp(u, seg, pattern[:, 0]), np.interp(u, seg, pattern[:, 1])
    d = np.hypot(xs - origin_xy[0], ys - origin_xy[1])
    return bool(d.max() <= p.range_max_m and d.min() >= p.range_min_m)


def in_cannon_arc(pattern: np.ndarray, pivot_xy: np.ndarray, ms_yaw: float, side: int, p: EffectorParams) -> bool:
    """Every vertex of the pattern lies inside the turret's lateral firing arc."""
    rel = np.arctan2(pattern[:, 1] - pivot_xy[1], pattern[:, 0] - pivot_xy[0]) - ms_yaw
    rel = np.degrees((rel + math.pi) % (2 * math.pi) - math.pi) * (1.0 if side == 0 else -1.0)
    lo, hi = p.cannon_arc_deg
    return bool(np.all((rel >= lo) & (rel <= hi)))


def drop_points(pattern: np.ndarray, n: int) -> np.ndarray:
    seg = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(pattern, axis=0).T))])
    pos = (np.arange(n) + 0.5) / n * seg[-1]
    return np.column_stack([np.interp(pos, seg, pattern[:, 0]), np.interp(pos, seg, pattern[:, 1])])


def ang_diff(a: float, b: float) -> float:
    return (b - a + math.pi) % (2 * math.pi) - math.pi


def launch_speed(rng_m: float, dz: float, elev_rad: float) -> float:
    """Muzzle speed hitting a point rng_m away and dz below the muzzle (drag-free)."""
    c, t = math.cos(elev_rad), math.tan(elev_rad)
    den = 2 * c * c * (dz + rng_m * t)
    return math.sqrt(G * rng_m * rng_m / max(den, 1e-6))


def ballistic(p0: np.ndarray, p1: np.ndarray, elev_rad: float, n: int = 24) -> tuple[np.ndarray, float]:
    """Drag-free arc from muzzle p0 (x,y,z) to impact p1; returns (n,3) points and flight time."""
    d = np.asarray(p1[:2]) - np.asarray(p0[:2])
    R = float(np.hypot(*d))
    dz = float(p0[2] - p1[2])
    v = launch_speed(max(R, 0.3), dz, elev_rad)
    vh, vz = v * math.cos(elev_rad), v * math.sin(elev_rad)
    T = max(R, 0.3) / vh
    tt = np.linspace(0.0, T, n)
    u = tt / T
    xy = np.asarray(p0[:2])[None, :] + u[:, None] * d[None, :]
    z = p0[2] + vz * tt - 0.5 * G * tt**2
    z[-1] = p1[2]
    return np.column_stack([xy, z]), T


@dataclass
class BurstPlan:
    kind: int
    t0: float  # start slewing
    t_fire0: float
    t1: float  # last material lands
    az0: float
    az1: float
    kg: float
    units: int
    aim_xy: np.ndarray
    range_m: float


def plan_blower_burst(t: float, ms_xy: np.ndarray, ms_yaw: float, pattern: np.ndarray, kg: float,
                      chute_az: float, idle_s: float, p: EffectorParams) -> BurstPlan:
    tip = mount_world(ms_xy[0], ms_xy[1], ms_yaw, CHUTE_TIP)
    aim = pattern.mean(axis=0)
    az1 = math.atan2(aim[1] - tip[1], aim[0] - tip[0])
    slew = abs(ang_diff(chute_az, az1)) / math.radians(p.chute_slew_dps)
    spool = p.blower_spool_s if idle_s > 5.0 else 0.0
    t_fire0 = t + max(slew, spool)
    throw = kg / p.blower_rate_kg_s
    rng_m = float(np.hypot(*(aim - tip)))
    t_fly = rng_m / max(launch_speed(rng_m, CHUTE_TIP[2], math.radians(p.blower_elev_deg))
                        * math.cos(math.radians(p.blower_elev_deg)), 1e-3)
    return BurstPlan(KIND_BLOWER, t, t_fire0, t_fire0 + throw + t_fly, chute_az, az1, kg, 1, aim, rng_m)


def plan_cannon_volley(t: float, ms_xy: np.ndarray, ms_yaw: float, pattern: np.ndarray, kg: float,
                       turret_az: float, side: int, p: EffectorParams) -> BurstPlan:
    n = max(1, int(math.ceil(kg / p.unit_kg - 1e-6)))
    pts = drop_points(pattern, n)
    aim = pattern.mean(axis=0)
    piv = mount_world(ms_xy[0], ms_xy[1], ms_yaw, CANNON_PIVOTS[side])
    az1 = math.atan2(pts[0, 1] - piv[1], pts[0, 0] - piv[0])
    slew = abs(ang_diff(turret_az, az1)) / math.radians(p.cannon_slew_dps)
    t_fire0 = t + slew
    rng_m = float(np.hypot(*(aim - piv)))
    el = math.radians(p.cannon_elev_deg)
    t_fly = rng_m / max(launch_speed(rng_m, CANNON_PIVOTS[side][2], el) * math.cos(el), 1e-3)
    return BurstPlan(KIND_CANNON, t, t_fire0, t_fire0 + (n - 1) * shot_gap(p) + t_fly, turret_az, az1, kg, n, aim,
                     rng_m)


def shot_gap(p: EffectorParams) -> float:
    """Twin barrels per turret alternate, each recharging for ``cannon_cycle_s``."""
    return p.cannon_cycle_s / 2.0


def side_of(ms_xy: np.ndarray, ms_yaw: float, pt: np.ndarray) -> int:
    """0 = port (left of heading), 1 = starboard."""
    lat = -math.sin(ms_yaw) * (pt[0] - ms_xy[0]) + math.cos(ms_yaw) * (pt[1] - ms_xy[1])
    return 0 if lat > 0 else 1


def volley_shots(k: int, row: dict, pattern: np.ndarray, ground_z, p: EffectorParams) -> dict[str, np.ndarray]:
    """Per-pellet schedule of a cannon volley (deterministic, seeded by row id)."""
    n = int(row["units"])
    side = int(row["side"])
    tgt = drop_points(pattern, n)
    yaw = float(row["ms_yaw"])
    piv = mount_world(row["ms_x"], row["ms_y"], yaw, CANNON_PIVOTS[side])
    ms_gz = float(ground_z(row["ms_x"], row["ms_y"]))
    rng = np.random.default_rng(2_000_029 + k)
    rngm = np.hypot(tgt[:, 0] - piv[0], tgt[:, 1] - piv[1])
    sig = p.cannon_sigma_m + p.cannon_sigma_per_m * rngm
    land = tgt + rng.normal(0.0, 1.0, size=(n, 2)) * sig[:, None]
    gz = np.asarray(ground_z(land[:, 0], land[:, 1]), dtype=float) + 0.05
    az = np.arctan2(land[:, 1] - piv[1], land[:, 0] - piv[0])
    el = math.radians(p.cannon_elev_deg)
    reach = CANNON_BARREL_M * math.cos(el)
    muzzle = np.column_stack([piv[0] + reach * np.cos(az), piv[1] + reach * np.sin(az),
                              np.full(n, ms_gz + CANNON_PIVOTS[side][2] + CANNON_BARREL_M * math.sin(el))])
    rr = np.hypot(land[:, 0] - muzzle[:, 0], land[:, 1] - muzzle[:, 1])
    t_fly = np.array([rr[i] / (launch_speed(rr[i], muzzle[i, 2] - gz[i], el) * math.cos(el)) for i in range(n)])
    return {"t_fire": row["t_fire0"] + np.arange(n) * shot_gap(p), "barrel": np.arange(n) % 2, "side": side,
            "muzzle": muzzle, "target": tgt, "land": np.column_stack([land, gz]), "az": az, "t_fly": t_fly}
