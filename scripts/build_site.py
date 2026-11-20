#!/usr/bin/env python3
"""Regenerate data/site_known.geojson, unknown_obstacles.geojson and scenery.geojson."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prunemaxxing.config import DEFAULT_DEM_PATH, SHOWCASE_AREA_HA  # noqa: E402
from prunemaxxing.geo.dem import load_dem  # noqa: E402
from prunemaxxing.geo.grid import crop_target_field, select_pilot_window  # noqa: E402
from prunemaxxing.geo.site import design_pilot_site, write_site_geojson  # noqa: E402


def main() -> int:
    dem = select_pilot_window(crop_target_field(load_dem(ROOT / DEFAULT_DEM_PATH)), SHOWCASE_AREA_HA)
    site = design_pilot_site(
        origin_e=dem.easting(0),
        origin_n=dem.northing(dem.height),
        width_m=dem.width * dem.pixel_size_x,
        height_m=dem.height * dem.pixel_size_y,
    )
    paths = write_site_geojson(site)
    area = sum(p.polygon.area for p in site.parcels) / 1e4
    print(f"Site {site.width_m:.0f} x {site.height_m:.0f} m, parcels {area:.2f} ha")
    for k, v in paths.items():
        print(f"  {k}: {v.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
