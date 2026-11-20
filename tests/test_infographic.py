"""Motion-graphics infographic: a short preview render of a small mission."""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest

from prunemaxxing.geo.dem import load_dem
from prunemaxxing.geo.grid import build_field_twin_from_dem, crop_target_field, select_pilot_window
from prunemaxxing.geo.site import load_site
from prunemaxxing.planning.mission import MissionConfig, run_mission
from prunemaxxing.planning.route_planner import plan_global_route
from prunemaxxing.viz.infographic import render_infographic

ROOT = Path(__file__).resolve().parents[1]
DEM = ROOT / "viladecans_full_dem_10m.TIF"


@pytest.fixture(scope="module")
def small_mission():
    if not DEM.exists():
        pytest.skip("DEM missing")
    site = load_site()
    twin = build_field_twin_from_dem(select_pilot_window(crop_target_field(load_dem(DEM)), 12.0))
    plan = plan_global_route(site, twin)
    return run_mission(site, twin, MissionConfig(max_stations=6), plan=plan)


def test_preview_render_writes_a_playable_video(small_mission, tmp_path):
    fps = 12
    out = render_infographic(small_mission, tmp_path / "preview.mp4", fps=fps, preview=True)
    assert out.exists()
    assert out.stat().st_size > 20_000

    reader = imageio.get_reader(out)
    try:
        n = reader.count_frames()
        first = reader.get_data(0)
        mid = reader.get_data(n // 2)
        last = reader.get_data(n - 1)
    finally:
        reader.close()
    assert abs(n - 6 * fps) <= 1
    assert first.shape[1] == 960 and first.shape[0] in (540, 544)
    assert not np.array_equal(first, mid) and not np.array_equal(mid, last)
    assert mid.std() > 10
