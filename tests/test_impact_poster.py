"""Climate-impact poster: impact arithmetic + a fast low-dpi render."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from prunemaxxing.config import DEFAULT_DEM_PATH, SHOWCASE_AREA_HA
from prunemaxxing.kpis.impact import FACTORS, factor_table, impact_from_material, impact_summary
from prunemaxxing.viz.impact_poster import EQUIVALENCES, factor_lines, render_impact_poster

ROOT = Path(__file__).resolve().parents[1]
DEM = ROOT / DEFAULT_DEM_PATH
PICKLES = ROOT / "outputs" / "dev"


def _small_missions() -> dict:
    from prunemaxxing.geo.dem import load_dem
    from prunemaxxing.geo.grid import build_field_twin_from_dem, crop_target_field, select_pilot_window
    from prunemaxxing.geo.site import load_site
    from prunemaxxing.planning.mission import MissionConfig, run_mission
    from prunemaxxing.planning.route_planner import plan_global_route

    site = load_site()
    twin = build_field_twin_from_dem(select_pilot_window(crop_target_field(load_dem(DEM)), SHOWCASE_AREA_HA))
    plan = plan_global_route(site, twin)
    return {m: run_mission(site, twin, MissionConfig(payload_mode=m, max_stations=6), plan=plan)
            for m in ("dust", "cartridge")}


@pytest.fixture(scope="module")
def missions() -> dict:
    if DEM.exists():
        return _small_missions()
    paths = {m: PICKLES / f"mission_{m}.pkl" for m in ("dust", "cartridge")}
    if all(p.exists() for p in paths.values()):
        return {m: pickle.loads(p.read_bytes()) for m, p in paths.items()}
    pytest.skip("neither the DEM nor outputs/dev mission pickles are available")


def test_impact_from_material_arithmetic():
    F = {k: f.value for k, f in FACTORS.items()}
    out = impact_from_material(10.0, 2.0, 15.0)
    burned = 10.0 * F["dry_matter_frac"] * F["combustion_frac"]
    assert out["burn_avoided_dm_t"] == pytest.approx(burned)
    assert out["pm25_avoided_kg"] == pytest.approx(burned * 3.9)
    assert out["burn_ghg_avoided_co2e_t"] == pytest.approx(burned / 1000 * (2.7 * 27 + 0.07 * 273))
    assert out["soil_co2e_t"] == pytest.approx(2.0 * F["soil_co2e"])
    low = impact_from_material(10.0, 2.0, 1.2)
    assert low["soil_co2e_t"] == pytest.approx(out["soil_co2e_t"] * 1.2 / 15.0)
    assert low["burn_avoided_dm_t"] == pytest.approx(burned)


def test_impact_summary_consistency(missions):
    for mode, res in missions.items():
        s, k = impact_summary(res), res.kpis
        assert s["payload_mode"] == mode
        assert s["total_co2e_t"] == pytest.approx(s["soil_co2e_t"] + s["burn_ghg_avoided_co2e_t"])
        assert s["burn_avoided_dm_t"] == pytest.approx(k["biomass_applied_t"] * 0.60 * 0.85)
        assert s["pm25_avoided_kg"] == pytest.approx(s["burn_avoided_dm_t"] * 3.9)
        ghg = (s["burn_ch4_avoided_kg"] * FACTORS["gwp_ch4"].value + s["burn_n2o_avoided_kg"] * FACTORS["gwp_n2o"].value)
        assert s["burn_ghg_avoided_co2e_t"] == pytest.approx(ghg / 1000.0)
        assert s["co2e_t_per_ha"] == pytest.approx(s["total_co2e_t"] / s["area_ha"])
        assert s["soil_co2e_t"] == pytest.approx(k["co2e_t"], abs=0.06)
        assert s["biogenic_co2_not_released_t"] > s["total_co2e_t"] - s["soil_co2e_t"]
    rows = factor_table()
    assert {r["key"] for r in rows} >= {"ef_ch4", "ef_n2o", "ef_pm25", "gwp_ch4", "gwp_n2o", "soil_co2e"}
    assert all(r["source"] for r in rows)


def test_footnotes_cover_every_factor():
    text = "\n".join(factor_lines())
    for r in factor_table():
        assert r["source"] in text, r["key"]
    assert text.count("†") >= sum(r["assumed"] for r in factor_table())
    eq = EQUIVALENCES["spain_ghg_per_capita"]
    assert eq["source"] in text and f"{eq['value']:g}" in text


def _check_png(path: Path, dpi: int) -> None:
    im = np.asarray(Image.open(path).convert("RGB"))
    assert im.shape == (round(10.8 * dpi), round(19.2 * dpi), 3)
    assert im.std() > 10


def test_render_single_mode_poster(missions, tmp_path):
    out = render_impact_poster(missions["dust"], tmp_path / "impact_dust.png", dpi=40)
    assert out.exists()
    _check_png(out, 40)


def test_render_combined_poster_without_map(missions, tmp_path):
    out = render_impact_poster([missions["cartridge"], missions["dust"]], tmp_path / "impact.png", dpi=40,
                               with_map=False)
    _check_png(out, 40)


def test_render_rejects_bad_input(tmp_path):
    with pytest.raises(ValueError):
        render_impact_poster([], tmp_path / "x.png")
