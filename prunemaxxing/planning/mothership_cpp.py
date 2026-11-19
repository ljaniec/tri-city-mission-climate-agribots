"""Adaptive coverage path planning for the mothership UGV."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from prunemaxxing.config import MOTHERSHIP_TURN_RADIUS_M
from prunemaxxing.geo.grid import CellState, FieldTwin


@dataclass
class Waypoint:
    row: int
    col: int
    easting: float
    northing: float
    lane_id: int


@dataclass
class MothershipPath:
    waypoints: list[Waypoint]
    lane_spacing_cells: int
    total_length_m: float
    excluded_cells: int

    @property
    def n_waypoints(self) -> int:
        return len(self.waypoints)


def _lane_spacing_for_tether(reach_m: float, cell_size_m: float) -> int:
    """Space lanes so tethered drones can reach mid-lane cells from the path."""
    # Path down lane centers; half-spacing should be <= tether reach
    spacing_m = max(cell_size_m, min(2.0 * reach_m, 4.0 * cell_size_m))
    return max(1, int(round(spacing_m / cell_size_m)))


def _cell_ok(cell: CellState) -> bool:
    return cell.traversable and not cell.weed_exclusion_flag and not cell.no_till_bed_status


def plan_mothership_coverage(
    twin: FieldTwin,
    tether_reach_m: float = 40.0,
    serpentine: bool = True,
) -> MothershipPath:
    """
    Boustrophedon (serpentine) CPP over traversable, non-weed cells.

    Lanes run north–south (along rows) with spacing derived from drone tether reach
    so the mothership remains a valid mobile base for the swarm.
    """
    cell_size = float(twin.dem.pixel_size_x)
    lane_spacing = _lane_spacing_for_tether(tether_reach_m, cell_size)

    # Columns that still have work / are traversable somewhere
    active_cols = []
    for c in range(twin.ncols):
        if any(_cell_ok(twin.get(r, c)) for r in range(twin.nrows)):
            active_cols.append(c)

    if not active_cols:
        return MothershipPath([], lane_spacing, 0.0, twin.nrows * twin.ncols)

    # Pick lane columns: every lane_spacing among active columns, plus edges
    lane_cols = active_cols[::lane_spacing]
    if active_cols[-1] not in lane_cols:
        lane_cols.append(active_cols[-1])

    waypoints: list[Waypoint] = []
    excluded = sum(1 for cell in twin.iter_cells() if not _cell_ok(cell))

    for lane_id, col in enumerate(lane_cols):
        rows = [r for r in range(twin.nrows) if _cell_ok(twin.get(r, col))]
        if not rows:
            # fallback: nearest traversable in adjacent cols
            continue
        if serpentine and lane_id % 2 == 1:
            rows = list(reversed(rows))
        for r in rows:
            cell = twin.get(r, col)
            waypoints.append(
                Waypoint(
                    row=r,
                    col=col,
                    easting=cell.easting,
                    northing=cell.northing,
                    lane_id=lane_id,
                )
            )

    length = 0.0
    for a, b in zip(waypoints, waypoints[1:]):
        length += float(np.hypot(b.easting - a.easting, b.northing - a.northing))

    # Soft kinematic note: length inflated slightly for turn radius at lane ends
    n_lane_changes = max(0, len(lane_cols) - 1)
    length += n_lane_changes * math_turn_penalty(MOTHERSHIP_TURN_RADIUS_M)

    return MothershipPath(
        waypoints=waypoints,
        lane_spacing_cells=lane_spacing,
        total_length_m=length,
        excluded_cells=excluded,
    )


def math_turn_penalty(turn_radius_m: float) -> float:
    # Approximate half-circle lane change
    return float(np.pi * turn_radius_m)


def path_polyline(path: MothershipPath) -> np.ndarray:
    """Nx2 array of (easting, northing)."""
    if not path.waypoints:
        return np.zeros((0, 2), dtype=float)
    return np.array([[w.easting, w.northing] for w in path.waypoints], dtype=float)
