from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from prunemaxxing.config import TARGET_FIELD_WGS84
from prunemaxxing.geo.crs import utm31n_to_wgs84, wgs84_to_utm31n
from prunemaxxing.geo.dem import load_dem
from prunemaxxing.geo.grid import (
    build_field_twin_from_dem,
    crop_target_field,
    select_pilot_window,
)
from prunemaxxing.planning.mothership_cpp import plan_mothership_coverage
from prunemaxxing.planning.simulator import run_coverage_simulation

ROOT = Path(__file__).resolve().parents[1]
DEM = ROOT / "viladecans_full_dem_10m.TIF"


@pytest.mark.skipif(not DEM.exists(), reason="DEM not present")
def test_dem_load_and_target_crop():
    dem = load_dem(DEM)
    assert dem.width == 340 and dem.height == 338
    assert dem.epsg == 32631
    target = crop_target_field(dem)
    assert target.width > 50 and target.height > 50
    # Target field elevations are near sea-level horticultural plain
    assert float(np.nanmean(target.elevation)) < 5.0


def test_crs_roundtrip():
    lon, lat = 2.03, 41.286
    e, n = wgs84_to_utm31n(lon, lat)
    lon2, lat2 = utm31n_to_wgs84(e, n)
    assert abs(lon - lon2) < 1e-5
    assert abs(lat - lat2) < 1e-5


@pytest.mark.skipif(not DEM.exists(), reason="DEM not present")
def test_pilot_twin_and_cpp():
    dem = load_dem(DEM)
    pilot = select_pilot_window(crop_target_field(dem), area_ha=6.0)
    twin = build_field_twin_from_dem(pilot, target_kg_per_m2=15.0)
    assert twin.eligible_area_m2() > 40_000
    path = plan_mothership_coverage(twin)
    assert path.n_waypoints > 10
    assert path.total_length_m > 0


@pytest.mark.skipif(not DEM.exists(), reason="DEM not present")
def test_coverage_sim_makes_progress():
    dem = load_dem(DEM)
    pilot = select_pilot_window(crop_target_field(dem), area_ha=6.0)
    twin = build_field_twin_from_dem(pilot, target_kg_per_m2=15.0)
    cell = twin.get(twin.nrows // 2, twin.ncols // 2)
    twin.apply_weed_patches(
        [(cell.easting, cell.northing)], radius_m=10.0, species="Cynodon_dactylon"
    )
    result = run_coverage_simulation(twin, swarm_size=4, max_passes=2)
    assert result.kpis.biomass_applied_t > 0
    assert result.kpis.offroad_compaction_m2 == 0.0
    assert result.kpis.weed_excluded_cells >= 1
    assert 0.0 <= result.kpis.coverage_uniformity <= 1.0


def test_target_bounds_constant():
    assert TARGET_FIELD_WGS84["lon_min"] < TARGET_FIELD_WGS84["lon_max"]
