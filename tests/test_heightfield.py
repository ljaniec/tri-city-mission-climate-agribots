"""Smoke tests for DEM → MuJoCo heightfield."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
DEM = ROOT / "viladecans_full_dem_10m.TIF"

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault(
    "__EGL_VENDOR_LIBRARY_FILENAMES",
    "/usr/share/glvnd/egl_vendor.d/10_nvidia.json",
)


@pytest.mark.skipif(not DEM.exists(), reason="DEM not present")
def test_generate_heightfield_grid_shape(tmp_path):
    from prunemaxxing.sim.heightfield import generate_heightfield

    png = tmp_path / "h.png"
    meta = tmp_path / "h.json"
    spec = generate_heightfield(
        dem_path=DEM,
        out_png=png,
        out_meta=meta,
        upsample=4.0,
        elevation_z=2.0,
    )
    assert png.exists() and meta.exists()
    assert spec.nrow >= 80 and spec.ncol >= 80
    assert spec.elev_max_m > spec.elev_min_m
    assert abs(spec.z_offset) <= spec.elevation_z + 1e-6


@pytest.mark.skipif(not DEM.exists(), reason="DEM not present")
def test_mujoco_load_and_hero_not_black():
    from prunemaxxing.sim.mujoco_runner import load_model, render_still_hero

    model, data = load_model(regenerate_heightfield=True)
    assert model.nhfield >= 1
    gid = __import__("mujoco").mj_name2id(
        model, __import__("mujoco").mjtObj.mjOBJ_GEOM, "terrain"
    )
    assert gid >= 0
    assert int(model.geom_type[gid]) == int(__import__("mujoco").mjtGeom.mjGEOM_HFIELD)

    img = render_still_hero(camera="hero_orbit", width=320, height=180)
    assert img.shape == (180, 320, 3)
    assert float(img.mean()) > 15.0  # not a black frame
    assert float(img.std()) > 5.0
