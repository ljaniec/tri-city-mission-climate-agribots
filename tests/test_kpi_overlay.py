"""KPI cards burned into the cinematic / dashboard frames."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from prunemaxxing.geo.dem import load_dem
from prunemaxxing.geo.grid import build_field_twin_from_dem, crop_target_field, select_pilot_window
from prunemaxxing.geo.site import load_site
from prunemaxxing.planning.mission import MissionConfig, run_mission
from prunemaxxing.viz.kpi_overlay import KpiOverlay

DEM = Path(__file__).resolve().parents[1] / "viladecans_full_dem_10m.TIF"


@pytest.fixture(scope="module")
def mission():
    if not DEM.exists():
        pytest.skip("DEM missing")
    twin = build_field_twin_from_dem(select_pilot_window(crop_target_field(load_dem(DEM)), 12.0))
    return run_mission(load_site(), twin, MissionConfig(max_stations=6))


def test_cards_track_mission_progress(mission):
    ov = KpiOverlay(mission)
    early = {c.label: c for c in ov.cards(0.0)}
    late = {c.label: c for c in ov.cards(mission.t_end)}
    assert len(late) == 7 and len(ov.cards(mission.t_end, compact=True)) == 6
    assert early["APPLIED"].frac == 0.0
    assert late["APPLIED"].frac > early["APPLIED"].frac
    assert late["STATIONS"].value.startswith(f"{len(mission.visits)} /")
    assert late["DRONE SWARM"].value == f"{len(mission.sorties['t_launch']):,} sorties"


def test_station_milestone_flashes_then_decays(mission):
    ov = KpiOverlay(mission)
    v = mission.visits[1]
    ov.cards(v["t_arrive"] - 1.0)
    flash = [next(c for c in ov.cards(v["t_arrive"] + k) if c.label == "STATIONS").flash for k in range(30)]
    assert flash[0] == pytest.approx(1.0)
    assert flash[-1] < 0.1


def test_draw_only_touches_the_card_band(mission):
    ov = KpiOverlay(mission)
    img = np.full((540, 960, 3), 90, np.uint8)
    out = ov.draw(img, 0.5 * mission.t_end, compact=True)
    assert out.shape == img.shape and out.dtype == np.uint8
    changed = np.any(out != img, axis=2)
    rows = np.where(changed.any(axis=1))[0]
    assert rows.size and rows.min() > 540 * 0.85
    assert np.array_equal(ov.draw(img, 0.0, alpha=0.0), img)
