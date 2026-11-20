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
from prunemaxxing.geo.polygons import apply_map_layers, load_map_layers
from prunemaxxing.planning.bed_geometry import build_bed_layout, smooth_bed_path
from prunemaxxing.planning.mothership_cpp import (
    plan_mothership_coverage,
    plan_mothership_coverage_complex,
    path_turning_score,
)
from prunemaxxing.planning.path_tracker import PathTracker
from prunemaxxing.planning.payload_modes import PayloadMode
from prunemaxxing.planning.simulator import run_coverage_simulation

ROOT = Path(__file__).resolve().parents[1]
DEM = ROOT / "viladecans_full_dem_10m.TIF"
GEOJSON = ROOT / "data" / "parcels_weeds.geojson"


@pytest.mark.skipif(not DEM.exists(), reason="DEM not present")
def test_dem_load_and_target_crop():
    dem = load_dem(DEM)
    assert dem.width == 340 and dem.height == 338
    assert dem.epsg == 32631
    target = crop_target_field(dem)
    assert target.width > 50 and target.height > 50
    assert float(np.nanmean(target.elevation)) < 5.0


def test_crs_roundtrip():
    lon, lat = 2.03, 41.286
    e, n = wgs84_to_utm31n(lon, lat)
    lon2, lat2 = utm31n_to_wgs84(e, n)
    assert abs(lon - lon2) < 1e-5
    assert abs(lat - lat2) < 1e-5


@pytest.mark.skipif(not GEOJSON.exists(), reason="GeoJSON missing")
def test_geojson_weeds_load():
    layers = load_map_layers(GEOJSON)
    assert len(layers.parcels) >= 3
    assert len(layers.weeds) == 3
    assert {w.props["species"] for w in layers.weeds} >= {
        "Sorghum_halepense",
        "Cyperus_rotundus",
        "Cynodon_dactylon",
    }


@pytest.mark.skipif(not DEM.exists() or not GEOJSON.exists(), reason="assets missing")
def test_apply_polygons_and_cpp():
    dem = load_dem(DEM)
    pilot = select_pilot_window(crop_target_field(dem), area_ha=6.0)
    twin = build_field_twin_from_dem(pilot, target_kg_per_m2=15.0)
    layers = load_map_layers(GEOJSON)
    counts = apply_map_layers(twin, layers)
    assert counts["weed_cells"] >= 1
    path = plan_mothership_coverage(twin)
    assert path.n_waypoints > 10
    layout = build_bed_layout(twin)
    planned = np.array([[w.easting, w.northing] for w in path.waypoints])
    snapped = smooth_bed_path(planned, layout)
    assert len(snapped) >= 2


@pytest.mark.skipif(not DEM.exists() or not GEOJSON.exists(), reason="assets missing")
def test_complex_cpp_longer_more_turning_avoids_weeds():
    dem = load_dem(DEM)
    pilot = select_pilot_window(crop_target_field(dem), area_ha=6.0)
    twin = build_field_twin_from_dem(pilot, target_kg_per_m2=15.0)
    apply_map_layers(twin, load_map_layers(GEOJSON))

    serp = plan_mothership_coverage(twin, path_style="serpentine")
    complex_path = plan_mothership_coverage_complex(twin)

    assert complex_path.n_waypoints > 0
    assert complex_path.path_style == "complex"
    assert complex_path.total_length_m > serp.total_length_m
    assert path_turning_score(complex_path) > path_turning_score(serp)

    # Required complex segments present
    labels = set(complex_path.segment_labels)
    assert "headland_arc" in labels
    assert "service_loop" in labels or "spiral" in labels
    assert "coverage" in labels

    # Never visit weed-excluded cells (by index or continuous EN)
    dy = max(float(twin.dem.pixel_size_y), 1e-9)
    dx = max(float(twin.dem.pixel_size_x), 1e-9)
    for w in complex_path.waypoints:
        assert not twin.get(w.row, w.col).weed_exclusion_flag
        r = int(
            np.clip(round((twin.dem.northing(0) - w.northing) / dy), 0, twin.nrows - 1)
        )
        c = int(
            np.clip(round((w.easting - twin.dem.easting(0)) / dx), 0, twin.ncols - 1)
        )
        assert not twin.get(r, c).weed_exclusion_flag

    # Dispatch via path_style flag matches dedicated complex planner
    via_flag = plan_mothership_coverage(twin, path_style="complex")
    assert via_flag.n_waypoints == complex_path.n_waypoints
    assert via_flag.total_length_m == complex_path.total_length_m


@pytest.mark.skipif(not DEM.exists(), reason="DEM not present")
def test_path_tracker_bed_lock():
    dem = load_dem(DEM)
    pilot = select_pilot_window(crop_target_field(dem), area_ha=6.0)
    twin = build_field_twin_from_dem(pilot)
    layout = build_bed_layout(twin)
    planned = np.array(
        [[layout.lane_centers_easting[0], twin.dem.northing(r)] for r in range(0, twin.nrows, 2)],
        dtype=float,
    )
    tracker = PathTracker(planned=planned, layout=layout)
    hist = tracker.run_until_end(max_steps=400)
    assert len(hist) > 10
    # Stays near a furrow center
    errs = [abs(s.easting - layout.lane_centers_easting[0]) for s in hist]
    assert max(errs) < layout.spacing_m * 0.6


@pytest.mark.skipif(not DEM.exists() or not GEOJSON.exists(), reason="assets missing")
def test_coverage_sim_wireless_makes_progress():
    dem = load_dem(DEM)
    pilot = select_pilot_window(crop_target_field(dem), area_ha=6.0)
    twin = build_field_twin_from_dem(pilot, target_kg_per_m2=15.0)
    apply_map_layers(twin, load_map_layers(GEOJSON))
    result = run_coverage_simulation(
        twin,
        swarm_size=4,
        wireless_power=True,
        max_passes=2,
        payload_mode=PayloadMode.DUST,
        record_frames=True,
    )
    assert result.kpis.biomass_applied_t > 0
    assert result.kpis.offroad_compaction_m2 == 0.0
    assert result.kpis.weed_excluded_cells >= 1
    assert len(result.frames) > 5
    assert len(result.actual_polyline) > 5


@pytest.mark.skipif(not DEM.exists() or not GEOJSON.exists(), reason="assets missing")
def test_drone_planned_and_actual_paths():
    """Each of 4 drones logs non-empty planned + actual sortie polylines."""
    dem = load_dem(DEM)
    pilot = select_pilot_window(crop_target_field(dem), area_ha=6.0)
    twin = build_field_twin_from_dem(pilot, target_kg_per_m2=15.0)
    apply_map_layers(twin, load_map_layers(GEOJSON))
    result = run_coverage_simulation(
        twin,
        swarm_size=4,
        wireless_power=True,
        max_passes=1,
        payload_mode=PayloadMode.DUST,
        record_frames=True,
    )
    assert len(result.allocator.drones) == 4
    assert set(result.drone_planned) == {0, 1, 2, 3}
    assert set(result.drone_actual) == {0, 1, 2, 3}
    for di in range(4):
        planned = result.drone_planned[di]
        actual = result.drone_actual[di]
        assert planned.ndim == 2 and planned.shape[1] == 2
        assert actual.ndim == 2 and actual.shape[1] == 2
        assert len(planned) >= 2, f"drone {di} planned empty"
        assert len(actual) >= 2, f"drone {di} actual empty"
        # Actual should not be identical to planned (noise / lag).
        # Use absolute metres — UTM magnitudes make default rtol useless.
        assert float(np.mean(np.abs(planned - actual))) > 0.05


def test_target_bounds_constant():
    assert TARGET_FIELD_WGS84["lon_min"] < TARGET_FIELD_WGS84["lon_max"]
