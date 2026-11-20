"""Permanent raised-bed / furrow geometry constraints for the UGV."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from prunemaxxing.config import BED_ROW_SPACING_M, MOTHERSHIP_TURN_RADIUS_M
from prunemaxxing.geo.grid import FieldTwin


@dataclass
class BedLayout:
    """North–south beds (rows along northing), spaced across easting."""

    origin_easting: float
    origin_northing: float
    spacing_m: float
    n_beds: int
    bed_width_m: float = 0.45  # plantable crown
    furrow_width_m: float = 0.40  # trafficable furrow for UGV

    @property
    def lane_centers_easting(self) -> np.ndarray:
        # UGV travels in furrows between beds
        return self.origin_easting + (np.arange(self.n_beds) + 0.5) * self.spacing_m


def build_bed_layout(
    twin: FieldTwin,
    spacing_m: float | None = None,
) -> BedLayout:
    spacing = spacing_m or float(np.mean(BED_ROW_SPACING_M))
    # Align beds to DEM west edge
    e0 = twin.dem.easting(0)
    e1 = twin.dem.easting(twin.ncols)
    width = e1 - e0
    n_beds = max(1, int(width / spacing))
    return BedLayout(
        origin_easting=e0,
        origin_northing=twin.dem.northing(twin.nrows),
        spacing_m=spacing,
        n_beds=n_beds,
        furrow_width_m=max(0.35, spacing - 0.45),
    )


def snap_easting_to_furrow(easting: float, layout: BedLayout) -> float:
    centers = layout.lane_centers_easting
    return float(centers[int(np.argmin(np.abs(centers - easting)))])


def project_path_to_beds(
    waypoints_en: np.ndarray,
    layout: BedLayout,
) -> np.ndarray:
    """
    Snap planned polyline eastings onto furrow centers.

    Preserves northing; enforces that the mothership never mounts bed crowns
    once no-till beds are formed (spacing 0.70–1.0 m).
    """
    if waypoints_en.size == 0:
        return waypoints_en
    out = waypoints_en.copy()
    out[:, 0] = [snap_easting_to_furrow(e, layout) for e in out[:, 0]]
    return out


def curvature_ok(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, turn_radius: float) -> bool:
    """Reject instantaneous turns tighter than mothership turn radius."""
    v1 = p1 - p0
    v2 = p2 - p1
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return True
    cosang = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    ang = np.arccos(cosang)
    # chord length vs radius: require path step not imply r < turn_radius
    if ang < 1e-3:
        return True
    implied_r = 0.5 * min(n1, n2) / max(np.sin(ang / 2.0), 1e-6)
    return implied_r >= 0.85 * turn_radius


def smooth_bed_path(
    waypoints_en: np.ndarray,
    layout: BedLayout,
    turn_radius: float = MOTHERSHIP_TURN_RADIUS_M,
) -> np.ndarray:
    snapped = project_path_to_beds(waypoints_en, layout)
    if len(snapped) < 3:
        return snapped
    kept = [snapped[0]]
    for i in range(1, len(snapped) - 1):
        if curvature_ok(kept[-1], snapped[i], snapped[i + 1], turn_radius):
            kept.append(snapped[i])
    kept.append(snapped[-1])
    return np.asarray(kept, dtype=float)
