"""Legacy cell-grid coverage planner (v1). Superseded by :mod:`prunemaxxing.planning.route_planner`."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from prunemaxxing.config import MOTHERSHIP_TURN_RADIUS_M
from prunemaxxing.geo.grid import CellState, FieldTwin
from prunemaxxing.planning.bed_geometry import (
    BedLayout,
    build_bed_layout,
    snap_easting_to_furrow,
    smooth_bed_path,
)

PathStyle = Literal["serpentine", "complex"]


@dataclass
class Waypoint:
    row: int
    col: int
    easting: float
    northing: float
    lane_id: int
    segment: str = "coverage"


@dataclass
class MothershipPath:
    waypoints: list[Waypoint]
    lane_spacing_cells: int
    total_length_m: float
    excluded_cells: int
    path_style: str = "serpentine"
    segment_labels: list[str] = field(default_factory=list)

    @property
    def n_waypoints(self) -> int:
        return len(self.waypoints)


def _lane_spacing_for_tether(reach_m: float, cell_size_m: float) -> int:
    """Space lanes so tethered drones can reach mid-lane cells from the path."""
    spacing_m = max(cell_size_m, min(2.0 * reach_m, 4.0 * cell_size_m))
    return max(1, int(round(spacing_m / cell_size_m)))


def _cell_ok(cell: CellState) -> bool:
    return cell.traversable and not cell.weed_exclusion_flag and not cell.no_till_bed_status


def _is_weed(twin: FieldTwin, row: int, col: int) -> bool:
    if not (0 <= row < twin.nrows and 0 <= col < twin.ncols):
        return True
    return bool(twin.get(row, col).weed_exclusion_flag)


def _en_to_rc(twin: FieldTwin, easting: float, northing: float) -> tuple[int, int]:
    dy = max(float(twin.dem.pixel_size_y), 1e-9)
    dx = max(float(twin.dem.pixel_size_x), 1e-9)
    r = int(np.clip(round((twin.dem.northing(0) - northing) / dy), 0, twin.nrows - 1))
    c = int(np.clip(round((easting - twin.dem.easting(0)) / dx), 0, twin.ncols - 1))
    return r, c


def _point_in_weed(twin: FieldTwin, easting: float, northing: float) -> bool:
    r, c = _en_to_rc(twin, easting, northing)
    return _is_weed(twin, r, c)


def _nearest_ok_cell(twin: FieldTwin, easting: float, northing: float) -> tuple[int, int]:
    best = (0, 0)
    best_d = float("inf")
    for cell in twin.iter_cells():
        if not _cell_ok(cell):
            continue
        d = (cell.easting - easting) ** 2 + (cell.northing - northing) ** 2
        if d < best_d:
            best_d = d
            best = (cell.row, cell.col)
    if best_d == float("inf"):
        return _en_to_rc(twin, easting, northing)
    return best


def _make_wp(
    twin: FieldTwin,
    easting: float,
    northing: float,
    lane_id: int,
    segment: str,
    *,
    snap_furrow: bool = False,
    layout: BedLayout | None = None,
) -> Waypoint | None:
    """Build a waypoint; None if the point cannot be placed outside weed cells."""
    e, n = float(easting), float(northing)
    if snap_furrow and layout is not None:
        e = snap_easting_to_furrow(e, layout)
    if _point_in_weed(twin, e, n):
        row, col = _nearest_ok_cell(twin, e, n)
        if not _cell_ok(twin.get(row, col)):
            return None
        cell = twin.get(row, col)
        e, n = cell.easting, cell.northing
        if snap_furrow and layout is not None:
            e = snap_easting_to_furrow(e, layout)
        if _point_in_weed(twin, e, n):
            return None
    row, col = _nearest_ok_cell(twin, e, n)
    return Waypoint(
        row=row,
        col=col,
        easting=e,
        northing=n,
        lane_id=lane_id,
        segment=segment,
    )


def _push(
    dest: list[Waypoint],
    twin: FieldTwin,
    easting: float,
    northing: float,
    lane_id: int,
    segment: str,
    *,
    snap_furrow: bool = False,
    layout: BedLayout | None = None,
) -> None:
    wp = _make_wp(
        twin,
        easting,
        northing,
        lane_id,
        segment,
        snap_furrow=snap_furrow,
        layout=layout,
    )
    if wp is not None:
        dest.append(wp)


def _path_length(waypoints: list[Waypoint]) -> float:
    length = 0.0
    for a, b in zip(waypoints, waypoints[1:]):
        length += float(np.hypot(b.easting - a.easting, b.northing - a.northing))
    return length


def math_turn_penalty(turn_radius_m: float) -> float:
    return float(np.pi * turn_radius_m)


def _active_lane_cols(twin: FieldTwin, lane_spacing: int) -> list[int]:
    active_cols = [
        c
        for c in range(twin.ncols)
        if any(_cell_ok(twin.get(r, c)) for r in range(twin.nrows))
    ]
    if not active_cols:
        return []
    lane_cols = active_cols[::lane_spacing]
    if active_cols[-1] not in lane_cols:
        lane_cols.append(active_cols[-1])
    return lane_cols


def _contiguous_row_runs(rows: list[int]) -> list[list[int]]:
    if not rows:
        return []
    rows = sorted(rows)
    runs: list[list[int]] = [[rows[0]]]
    for r in rows[1:]:
        if r == runs[-1][-1] + 1:
            runs[-1].append(r)
        else:
            runs.append([r])
    return runs


def _headland_arc(
    twin: FieldTwin,
    e0: float,
    n0: float,
    e1: float,
    n1: float,
    *,
    lane_id: int,
    turn_radius_m: float,
    n_points: int = 12,
    layout: BedLayout | None = None,
) -> list[Waypoint]:
    """Clothoid-like multi-segment headland turn (not a sharp U-turn)."""
    de = e1 - e0
    dn = n1 - n0
    outward = 1.0 if n1 >= n0 else -1.0
    if abs(dn) < 1e-3:
        outward = 1.0 if n0 > twin.dem.northing(twin.nrows // 2) else -1.0
    bulge = max(turn_radius_m, 0.55 * abs(de), abs(dn) * 0.35)

    pts: list[Waypoint] = []
    for i in range(1, n_points + 1):
        t = i / (n_points + 1)
        s = t * t * (3.0 - 2.0 * t)  # smoothstep ease-in/out
        lobe = np.sin(np.pi * t)
        e = e0 + s * de
        n = n0 + s * dn + outward * bulge * lobe
        for _ in range(8):
            if not _point_in_weed(twin, e, n):
                break
            n += outward * twin.dem.pixel_size_y * 0.6
        _push(pts, twin, e, n, lane_id, "headland_arc", layout=layout)
    return pts


def _weed_detour(
    twin: FieldTwin,
    col: int,
    row_a: int,
    row_b: int,
    *,
    lane_id: int,
    layout: BedLayout | None,
) -> list[Waypoint]:
    """Detour around weed cells between two ok cells on the same lane."""
    lo, hi = (row_a, row_b) if row_a < row_b else (row_b, row_a)
    candidates = [c for c in (col - 1, col + 1, col - 2, col + 2) if 0 <= c < twin.ncols]
    detour_col = None
    for c in candidates:
        if all(
            (not twin.get(r, c).weed_exclusion_flag) and twin.get(r, c).traversable
            for r in range(lo, hi + 1)
        ):
            detour_col = c
            break
    if detour_col is None:
        for c in range(twin.ncols):
            if c == col:
                continue
            if all(not twin.get(r, c).weed_exclusion_flag for r in range(lo, hi + 1)):
                detour_col = c
                break
    if detour_col is None:
        detour_col = int(np.clip(col + 1, 0, twin.ncols - 1))

    start = twin.get(row_a, col)
    end = twin.get(row_b, col)
    side = twin.get(row_a, detour_col)
    side_end = twin.get(row_b, detour_col)

    wps: list[Waypoint] = []
    _push(
        wps,
        twin,
        0.65 * start.easting + 0.35 * side.easting,
        start.northing,
        lane_id,
        "weed_detour",
        layout=layout,
    )
    _push(
        wps,
        twin,
        side.easting,
        start.northing,
        lane_id,
        "weed_detour",
        snap_furrow=True,
        layout=layout,
    )
    step = 1 if row_b > row_a else -1
    for r in range(row_a + step, row_b, step):
        cell = twin.get(r, detour_col)
        if cell.weed_exclusion_flag:
            continue
        _push(
            wps,
            twin,
            cell.easting,
            cell.northing,
            lane_id,
            "weed_detour",
            snap_furrow=True,
            layout=layout,
        )
    _push(
        wps,
        twin,
        side_end.easting,
        end.northing,
        lane_id,
        "weed_detour",
        snap_furrow=True,
        layout=layout,
    )
    _push(
        wps,
        twin,
        0.35 * side_end.easting + 0.65 * end.easting,
        end.northing,
        lane_id,
        "weed_detour",
        layout=layout,
    )
    return wps


def _service_loop_s_curve(
    twin: FieldTwin,
    *,
    lane_id: int,
    layout: BedLayout | None,
) -> list[Waypoint]:
    """Figure-8 / S-curve service loop near the northern (stockpile-edge) perimeter."""
    north_rows = [
        r
        for r in range(min(4, twin.nrows))
        if any(_cell_ok(twin.get(r, c)) for c in range(twin.ncols))
    ]
    if not north_rows:
        north_rows = [0]
    row = north_rows[0]
    ok_cols = [c for c in range(twin.ncols) if _cell_ok(twin.get(row, c))]
    if len(ok_cols) < 4:
        for r in range(twin.nrows):
            cols = [c for c in range(twin.ncols) if _cell_ok(twin.get(r, c))]
            if len(cols) >= 4:
                row, ok_cols = r, cols
                break
    if len(ok_cols) < 3:
        return []

    c_lo, c_hi = ok_cols[0], ok_cols[-1]
    e0 = twin.get(row, c_lo).easting
    e1 = twin.get(row, c_hi).easting
    n_base = twin.get(row, c_lo).northing
    amp = max(twin.dem.pixel_size_y * 1.2, MOTHERSHIP_TURN_RADIUS_M * 0.8)
    n_pts = max(16, (c_hi - c_lo) * 2)

    wps: list[Waypoint] = []
    for i in range(n_pts + 1):
        t = i / n_pts
        e = e0 + t * (e1 - e0)
        n = n_base + amp * np.sin(2.0 * np.pi * t)
        if _point_in_weed(twin, e, n):
            n = n_base
        _push(wps, twin, e, n, lane_id, "service_loop", layout=layout)
    for i in range(1, n_pts + 1):
        t = i / n_pts
        e = e1 - t * (e1 - e0)
        n = n_base - amp * np.sin(2.0 * np.pi * t)
        if _point_in_weed(twin, e, n):
            n = n_base
        _push(wps, twin, e, n, lane_id, "service_loop", layout=layout)
    return wps


def _filter_weed_waypoints(twin: FieldTwin, waypoints: list[Waypoint]) -> list[Waypoint]:
    """Drop waypoints that land on weed cells (cell index or continuous EN)."""
    out: list[Waypoint] = []
    for w in waypoints:
        if _point_in_weed(twin, w.easting, w.northing):
            continue
        if 0 <= w.row < twin.nrows and 0 <= w.col < twin.ncols:
            if twin.get(w.row, w.col).weed_exclusion_flag:
                continue
        out.append(w)
    return out


def _cover_lane_with_detours(
    twin: FieldTwin,
    col: int,
    rows: list[int],
    *,
    lane_id: int,
    layout: BedLayout,
    reverse: bool,
) -> list[Waypoint]:
    """Travel a lane as contiguous runs, inserting weed detours across gaps."""
    runs = _contiguous_row_runs(rows)
    if reverse:
        runs = list(reversed([list(reversed(run)) for run in runs]))
    wps: list[Waypoint] = []
    for ri, run in enumerate(runs):
        for r in run:
            cell = twin.get(r, col)
            wps.append(
                Waypoint(
                    row=r,
                    col=col,
                    easting=snap_easting_to_furrow(cell.easting, layout),
                    northing=cell.northing,
                    lane_id=lane_id,
                    segment="coverage",
                )
            )
        if ri + 1 < len(runs):
            wps.extend(
                _weed_detour(
                    twin,
                    col,
                    run[-1],
                    runs[ri + 1][0],
                    lane_id=lane_id,
                    layout=layout,
                )
            )
    return wps


def plan_mothership_coverage(
    twin: FieldTwin,
    tether_reach_m: float = 40.0,
    serpentine: bool = True,
    path_style: PathStyle = "serpentine",
) -> MothershipPath:
    """
    Coverage path planner.

    ``path_style="serpentine"`` — classic boustrophedon (default for API compat).
    ``path_style="complex"`` — curved headlands, weed detours + service loop.
    """
    if path_style == "complex":
        return plan_mothership_coverage_complex(twin, tether_reach_m=tether_reach_m)

    cell_size = float(twin.dem.pixel_size_x)
    lane_spacing = _lane_spacing_for_tether(tether_reach_m, cell_size)
    lane_cols = _active_lane_cols(twin, lane_spacing)

    if not lane_cols:
        return MothershipPath(
            [], lane_spacing, 0.0, twin.nrows * twin.ncols, path_style="serpentine"
        )

    waypoints: list[Waypoint] = []
    excluded = sum(1 for cell in twin.iter_cells() if not _cell_ok(cell))

    for lane_id, col in enumerate(lane_cols):
        rows = [r for r in range(twin.nrows) if _cell_ok(twin.get(r, col))]
        if not rows:
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
                    segment="coverage",
                )
            )

    length = _path_length(waypoints)
    n_lane_changes = max(0, len(lane_cols) - 1)
    length += n_lane_changes * math_turn_penalty(MOTHERSHIP_TURN_RADIUS_M)

    return MothershipPath(
        waypoints=waypoints,
        lane_spacing_cells=lane_spacing,
        total_length_m=length,
        excluded_cells=excluded,
        path_style="serpentine",
        segment_labels=sorted({w.segment for w in waypoints}),
    )


def plan_mothership_coverage_complex(
    twin: FieldTwin,
    tether_reach_m: float = 40.0,
    turn_radius_m: float = MOTHERSHIP_TURN_RADIUS_M,
) -> MothershipPath:
    """
    Demo-oriented coverage with:

    - Curved multi-segment headland turns between lanes
    - Explicit weed-exclusion detours (never cut through weed cells)
    - Spiral inward on a southern sub-block + S-curve stockpile-edge service loop
    - Furrow snapping inside bedded parcels via ``bed_geometry``
    """
    cell_size = float(twin.dem.pixel_size_x)
    lane_spacing = _lane_spacing_for_tether(tether_reach_m, cell_size)
    layout = build_bed_layout(twin)
    excluded = sum(1 for cell in twin.iter_cells() if not _cell_ok(cell))

    lane_cols = _active_lane_cols(twin, lane_spacing)
    if not lane_cols:
        return MothershipPath(
            [], lane_spacing, 0.0, twin.nrows * twin.ncols, path_style="complex"
        )

    waypoints: list[Waypoint] = []

    # 1) Figure-8 / S-curve service loop near stockpile edge
    waypoints.extend(_service_loop_s_curve(twin, lane_id=0, layout=layout))

    # 2) Lanes with arc headlands + weed detours
    main_cols = lane_cols

    prev_end: Waypoint | None = waypoints[-1] if waypoints else None

    for lane_id, col in enumerate(main_cols):
        rows_all = [r for r in range(twin.nrows) if _cell_ok(twin.get(r, col))]
        if not rows_all:
            continue
        reverse = lane_id % 2 == 1
        lane_wps = _cover_lane_with_detours(
            twin, col, rows_all, lane_id=lane_id, layout=layout, reverse=reverse
        )
        if not lane_wps:
            continue
        if prev_end is not None:
            waypoints.extend(
                _headland_arc(
                    twin,
                    prev_end.easting,
                    prev_end.northing,
                    lane_wps[0].easting,
                    lane_wps[0].northing,
                    lane_id=lane_id,
                    turn_radius_m=turn_radius_m,
                    n_points=14,
                    layout=layout,
                )
            )
        waypoints.extend(lane_wps)
        prev_end = waypoints[-1]

    waypoints = _filter_weed_waypoints(twin, waypoints)

    # Bed-path smoothing; reattach segment labels from nearest original
    if len(waypoints) >= 3:
        en = np.array([[w.easting, w.northing] for w in waypoints], dtype=float)
        smoothed = smooth_bed_path(en, layout, turn_radius=turn_radius_m)
        rebuilt: list[Waypoint] = []
        for e, n in smoothed:
            dists = [(w.easting - e) ** 2 + (w.northing - n) ** 2 for w in waypoints]
            src = waypoints[int(np.argmin(dists))]
            snap = src.segment in ("coverage", "weed_detour")
            _push(
                rebuilt,
                twin,
                float(e),
                float(n),
                src.lane_id,
                src.segment,
                snap_furrow=snap,
                layout=layout,
            )
        waypoints = _filter_weed_waypoints(twin, rebuilt)

    labels = sorted({w.segment for w in waypoints})
    return MothershipPath(
        waypoints=waypoints,
        lane_spacing_cells=lane_spacing,
        total_length_m=_path_length(waypoints),
        excluded_cells=excluded,
        path_style="complex",
        segment_labels=labels,
    )


def path_polyline(path: MothershipPath) -> np.ndarray:
    """Nx2 array of (easting, northing)."""
    if not path.waypoints:
        return np.zeros((0, 2), dtype=float)
    return np.array([[w.easting, w.northing] for w in path.waypoints], dtype=float)


def path_turning_score(path: MothershipPath) -> float:
    """Sum of absolute heading changes (radians) — higher means more turning."""
    if len(path.waypoints) < 3:
        return 0.0
    total = 0.0
    for i in range(1, len(path.waypoints) - 1):
        a, b, c = path.waypoints[i - 1], path.waypoints[i], path.waypoints[i + 1]
        v1 = np.array([b.easting - a.easting, b.northing - a.northing])
        v2 = np.array([c.easting - b.easting, c.northing - b.northing])
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-9 or n2 < 1e-9:
            continue
        cosang = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
        total += float(np.arccos(cosang))
    return total
