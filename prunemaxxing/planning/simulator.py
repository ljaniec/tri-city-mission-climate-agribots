"""Legacy v1 waypoint-stepped coverage simulation. Superseded by :mod:`prunemaxxing.planning.mission`."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from prunemaxxing.config import (
    DEFAULT_SWARM_SIZE,
    DRONE_TETHER_REACH_M,
    MOTHERSHIP_HOPPER_CAPACITY_KG,
)
from prunemaxxing.geo.grid import FieldTwin
from prunemaxxing.kpis.metrics import KPIReport, compute_kpis
from prunemaxxing.planning.bed_geometry import build_bed_layout, smooth_bed_path
from prunemaxxing.planning.drone_allocator import DroneSwarmAllocator
from prunemaxxing.planning.mothership_cpp import (
    MothershipPath,
    PathStyle,
    plan_mothership_coverage,
)
from prunemaxxing.planning.path_tracker import PathTracker, TrackSample, TrackerConfig
from prunemaxxing.planning.payload_modes import PayloadMode


@dataclass
class SimFrame:
    step: int
    mothership_e: float
    mothership_n: float
    mothership_yaw: float
    hopper_kg: float
    drones: list[tuple[float, float, float]]  # e, n, alt
    completion: np.ndarray
    biomass_t: float
    deposited_step_kg: float


@dataclass
class CoverageSimResult:
    path: MothershipPath
    twin: FieldTwin
    allocator: DroneSwarmAllocator
    steps: int
    hopper_refills: int
    road_stockpile_drops_kg: float
    offroad_compaction_m2: float
    kpis: KPIReport
    planned_polyline: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    actual_polyline: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    drone_planned: dict[int, np.ndarray] = field(default_factory=dict)
    drone_actual: dict[int, np.ndarray] = field(default_factory=dict)
    track_samples: list[TrackSample] = field(default_factory=list)
    frames: list[SimFrame] = field(default_factory=list)
    bed_spacing_m: float = 0.85


def _completion_grid(twin: FieldTwin) -> np.ndarray:
    return np.array(
        [
            [twin.get(r, c).completion_ratio for c in range(twin.ncols)]
            for r in range(twin.nrows)
        ],
        dtype=float,
    )


def _live_drone_poses(
    allocator: DroneSwarmAllocator,
) -> list[tuple[float, float, float]]:
    return [(d.easting, d.northing, d.alt_m) for d in allocator.drones]


def _yaw_along(polyline: np.ndarray, idx: int) -> float:
    if len(polyline) < 2:
        return 0.0
    i = min(max(idx, 0), len(polyline) - 2)
    d = polyline[i + 1] - polyline[i]
    if float(np.hypot(d[0], d[1])) < 1e-9:
        return 0.0
    return float(np.arctan2(d[1], d[0]))


def _rebuild_track(
    planned: np.ndarray, layout
) -> tuple[list[TrackSample], np.ndarray]:
    """Run a slow pure-pursuit pass so the actual trail matches the full plan."""
    if len(planned) < 2:
        return [], planned.copy()
    tracker = PathTracker(
        planned=planned,
        layout=layout,
        cfg=TrackerConfig(
            bed_lock=True,
            lookahead_m=max(5.0, getattr(layout, "spacing_m", 0.85) * 6),
            max_speed_mps=0.95,  # calmer pacing for demo video
            min_speed_mps=0.35,
            dt=0.25,
        ),
    )
    samples = tracker.run_until_end(max_steps=min(20000, max(1500, len(planned) * 50)))
    return samples, tracker.actual_polyline


def _pose_at_waypoint(
    planned: np.ndarray,
    wp_index: int,
) -> tuple[float, float, float]:
    """
    Drive mothership pose directly from the planned waypoint index.

    Do NOT index into dense track samples — those cluster on early lanes and make
    the video race then freeze mid-field. Waypoint-synced pose = steady pacing.
    """
    if len(planned) == 0:
        return 0.0, 0.0, 0.0
    pi = int(np.clip(wp_index, 0, len(planned) - 1))
    # If waypoint list was smoothed shorter than raw path, map proportionally
    return float(planned[pi, 0]), float(planned[pi, 1]), _yaw_along(planned, pi)


def run_coverage_simulation(
    twin: FieldTwin,
    swarm_size: int = DEFAULT_SWARM_SIZE,
    wireless_power: bool = True,
    ops_radius_m: float = DRONE_TETHER_REACH_M,
    hopper_capacity_kg: float = MOTHERSHIP_HOPPER_CAPACITY_KG,
    max_passes: int = 3,
    payload_mode: PayloadMode | str = PayloadMode.DUST,
    record_frames: bool = True,
    frame_stride: int = 1,
    path_style: PathStyle = "complex",
    tethered: bool | None = None,
) -> CoverageSimResult:
    """
    Advance mothership along bed-constrained CPP; drones deposit via wireless ops bubble.

    Mothership pose is synced to waypoint progress along the planned/tracked path
    (steady pacing for the full route — no mid-field freeze).
    """
    if tethered is not None:
        wireless_power = not tethered

    layout = build_bed_layout(twin)
    path = plan_mothership_coverage(
        twin, tether_reach_m=ops_radius_m, path_style=path_style
    )
    planned = np.array([[w.easting, w.northing] for w in path.waypoints], dtype=float)
    planned = smooth_bed_path(planned, layout)

    allocator = DroneSwarmAllocator.create(
        swarm_size,
        wireless_power=wireless_power,
        payload_mode=payload_mode,
        ops_radius_m=ops_radius_m,
    )
    # Seed drones near the path start so early frames aren't chaotic
    if len(planned):
        for i, d in enumerate(allocator.drones):
            ang = 2 * np.pi * i / max(1, len(allocator.drones))
            d.easting = float(planned[0, 0] + 8.0 * np.cos(ang))
            d.northing = float(planned[0, 1] + 8.0 * np.sin(ang))
            d.alt_m = 5.0 + 0.4 * i

    track_samples, actual = _rebuild_track(planned, layout)

    hopper = 0.0
    refills = 0
    stockpile_kg = 0.0
    steps = 0
    frames: list[SimFrame] = []
    all_actual_parts: list[np.ndarray] = [actual] if len(actual) else []

    perimeter = {
        (r, c)
        for r in range(twin.nrows)
        for c in range(twin.ncols)
        if r in (0, twin.nrows - 1) or c in (0, twin.ncols - 1)
    }

    wp_list = path.waypoints
    final_planned = planned

    for _pass in range(max_passes):
        if not twin.work_cells() or not wp_list:
            break

        n_wp = len(wp_list)
        for wp_i, wp in enumerate(wp_list):
            steps += 1
            on_perimeter = (wp.row, wp.col) in perimeter
            if on_perimeter and hopper < 0.20 * hopper_capacity_kg:
                need = hopper_capacity_kg - hopper
                hopper = hopper_capacity_kg
                refills += 1
                stockpile_kg += need
            elif hopper < 1e-6:
                hopper = hopper_capacity_kg
                refills += 1
                stockpile_kg += hopper_capacity_kg

            deposited, hopper = allocator.allocate_at_waypoint(
                twin, wp, hopper, step=steps
            )

            # Steady pose: one waypoint per step along the smoothed plan
            # (map wp_i → planned index in case smooth_bed_path shortened the path)
            plan_i = int(round(wp_i / max(1, n_wp - 1) * (len(planned) - 1))) if n_wp > 1 else 0
            me, mn, myaw = _pose_at_waypoint(planned, plan_i)

            if record_frames and (steps % frame_stride == 0):
                frames.append(
                    SimFrame(
                        step=steps,
                        mothership_e=me,
                        mothership_n=mn,
                        mothership_yaw=myaw,
                        hopper_kg=hopper,
                        drones=_live_drone_poses(allocator),
                        completion=_completion_grid(twin),
                        biomass_t=twin.total_deposited_t(),
                        deposited_step_kg=deposited,
                    )
                )

        if twin.work_cells():
            path = plan_mothership_coverage(
                twin, tether_reach_m=ops_radius_m, path_style=path_style
            )
            raw_wps = list(path.waypoints)
            raw_planned = np.array(
                [[w.easting, w.northing] for w in raw_wps], dtype=float
            )
            raw_planned = smooth_bed_path(raw_planned, layout)
            # Continue from nearest point to current pose — never teleport to path start
            if frames and len(raw_planned):
                le, ln = frames[-1].mothership_e, frames[-1].mothership_n
                d2 = (raw_planned[:, 0] - le) ** 2 + (raw_planned[:, 1] - ln) ** 2
                start_i = int(np.argmin(d2))
                planned = raw_planned[start_i:]
                # Approximate matching waypoints (same order as pre-smooth path)
                wp_start = min(start_i, max(0, len(raw_wps) - 1))
                wp_list = raw_wps[wp_start:]
            else:
                planned = raw_planned
                wp_list = raw_wps
            if len(planned) < 2:
                break
            track_samples, actual = _rebuild_track(planned, layout)
            if len(actual):
                all_actual_parts.append(actual)
            final_planned = (
                np.vstack([final_planned, planned])
                if len(final_planned)
                else planned
            )
        else:
            final_planned = planned

    twin.lock_no_till_beds()
    kpis = compute_kpis(
        twin=twin,
        path=path,
        allocator=allocator,
        hopper_refills=refills,
        stockpile_kg=stockpile_kg,
        offroad_compaction_m2=0.0,
        steps=steps,
    )
    drone_planned, drone_actual = allocator.finalized_paths()
    # Prefer frame poses for the displayed "actual" trail so the map matches the
    # video (steady waypoint pacing) instead of a jagged offline tracker dump.
    if frames:
        actual_poly = np.array(
            [[f.mothership_e, f.mothership_n] for f in frames], dtype=float
        )
    elif all_actual_parts:
        actual_poly = np.vstack(all_actual_parts)
    else:
        actual_poly = np.zeros((0, 2), dtype=float)

    return CoverageSimResult(
        path=path,
        twin=twin,
        allocator=allocator,
        steps=steps,
        hopper_refills=refills,
        road_stockpile_drops_kg=stockpile_kg,
        offroad_compaction_m2=0.0,
        kpis=kpis,
        planned_polyline=final_planned,
        actual_polyline=actual_poly,
        drone_planned=drone_planned,
        drone_actual=drone_actual,
        track_samples=track_samples,
        frames=frames,
        bed_spacing_m=layout.spacing_m,
    )
