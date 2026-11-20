"""Offline single-file HTML mission replay (prunemaxxing.viz.html_replay)."""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path

import numpy as np
import pytest

from prunemaxxing.geo.dem import load_dem
from prunemaxxing.geo.grid import build_field_twin_from_dem, crop_target_field, select_pilot_window
from prunemaxxing.geo.site import load_site
from prunemaxxing.planning.mission import MissionConfig, run_mission
from prunemaxxing.planning.route_planner import plan_global_route
from prunemaxxing.viz.basemap import build_basemap
from prunemaxxing.viz.html_replay import export_html_replay

ROOT = Path(__file__).resolve().parents[1]
DEM = ROOT / "viladecans_full_dem_10m.TIF"
MAX_BYTES = 8_000_000


@pytest.fixture(scope="module")
def site_twin_plan():
    if not DEM.exists():
        pytest.skip("DEM missing")
    site = load_site()
    twin = build_field_twin_from_dem(select_pilot_window(crop_target_field(load_dem(DEM)), 12.0))
    return site, twin, plan_global_route(site, twin)


@pytest.fixture(scope="module")
def basemap(site_twin_plan):
    return build_basemap(site_twin_plan[0])


def _payload(html: str) -> dict:
    m = re.search(r'<script type="application/json" id="replay-data">(.*?)</script>', html, re.S)
    assert m, "embedded replay data missing"
    return json.loads(m.group(1))


def _decode(entry) -> np.ndarray:
    dtype, b64 = entry
    return np.frombuffer(base64.b64decode(b64), dtype={"f4": "<f4", "f8": "<f8", "i1": "i1", "u1": "u1", "i2": "<i2",
                                                      "u2": "<u2", "i4": "<i4", "u4": "<u4"}[dtype])


@pytest.mark.parametrize("mode", ["dust", "cartridge"])
def test_export_is_self_contained_and_embeds_the_mission(site_twin_plan, basemap, tmp_path, mode):
    site, twin, plan = site_twin_plan
    res = run_mission(site, twin, MissionConfig(payload_mode=mode, max_stations=6), plan=plan)
    out = export_html_replay(res, tmp_path / f"replay_{mode}.html", basemap=basemap)

    assert out.exists() and out.suffix == ".html"
    assert 200_000 < out.stat().st_size < MAX_BYTES
    html = out.read_text(encoding="utf-8")

    # no network access of any kind
    low = html.lower()
    for needle in ('src="http', "src='http", 'href="http', "href='http", "@import url(http", "url(http", "url('http",
                   'url("http', "fetch(", "xmlhttprequest", "<link "):
        assert needle not in low, needle
    assert re.search(r"<script[^>]*\bsrc=", html, re.I) is None
    assert 'src="data:image/' in html  # basemap inlined

    # embedded data decodes and matches the mission
    d = _payload(html)
    assert {"meta", "kpis", "impact", "factors", "timeline", "parcels", "obstacles", "stations", "cells", "events",
            "bin"} <= d.keys()
    assert d["meta"]["mode"] == mode
    assert d["meta"]["t_end"] == pytest.approx(res.t_end)
    assert len(d["stations"]) == len(res.visits)
    assert len(d["events"]) >= len(res.visits)
    assert len(res.sorties["kg"]) > 0
    assert len(_decode(d["bin"]["s_drone"])) == len(res.sorties["kg"])
    assert len(_decode(d["bin"]["m_kind"])) == len(res.ms_shots.get("kg", []))
    assert len(_decode(d["bin"]["ms_t"])) == len(res.ms_log["t"])
    x = _decode(d["bin"]["ms_x"]).astype(float) / 10.0
    assert 0.0 <= x.min() and x.max() <= d["meta"]["W_m"]
    assert np.isclose(_decode(d["bin"]["d_kg"]).sum(), res.deposits["kg"].sum(), rtol=1e-3)
    assert "</script" not in html[html.index('id="replay-data"'):html.index("</script>", html.index('id="replay-data"'))]
