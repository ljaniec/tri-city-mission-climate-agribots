#!/usr/bin/env python3
"""Run 6 ha pilot coverage on the Viladecans target field DEM window."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prunemaxxing.config import (
    DEFAULT_DEM_PATH,
    DEFAULT_GEOJSON_PATH,
    DEFAULT_SWARM_SIZE,
    PilotSpec,
)
from prunemaxxing.geo.dem import load_dem
from prunemaxxing.geo.grid import (
    build_field_twin_from_dem,
    crop_target_field,
    select_pilot_window,
)
from prunemaxxing.geo.polygons import apply_map_layers, load_map_layers
from prunemaxxing.logistics.macro_router import plan_macro_logistics
from prunemaxxing.planning.simulator import run_coverage_simulation


def main() -> int:
    parser = argparse.ArgumentParser(description="PruneMaxxing coverage demo")
    parser.add_argument("--dem", default=str(ROOT / DEFAULT_DEM_PATH))
    parser.add_argument("--area-ha", type=float, default=6.0)
    parser.add_argument("--kg-per-m2", type=float, default=15.0)
    parser.add_argument("--swarm", type=int, default=DEFAULT_SWARM_SIZE)
    parser.add_argument("--passes", type=int, default=4)
    parser.add_argument("--geojson", default=str(ROOT / DEFAULT_GEOJSON_PATH))
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    pilot = PilotSpec(area_ha=args.area_ha, target_kg_per_m2=args.kg_per_m2)
    macro = plan_macro_logistics(pilot)

    dem = load_dem(args.dem)
    target = crop_target_field(dem)
    pilot_dem = select_pilot_window(target, area_ha=pilot.area_ha, anchor="sw")
    twin = build_field_twin_from_dem(pilot_dem, target_kg_per_m2=pilot.target_kg_per_m2)

    layers = load_map_layers(args.geojson)
    apply_map_layers(twin, layers)

    result = run_coverage_simulation(
        twin,
        swarm_size=args.swarm,
        wireless_power=True,
        max_passes=args.passes,
    )

    print("PruneMaxxing — pilot coverage simulation")
    print(
        f"  DEM crop: {pilot_dem.width}x{pilot_dem.height} cells "
        f"({pilot_dem.width * pilot_dem.pixel_size_x:.0f}m x "
        f"{pilot_dem.height * pilot_dem.pixel_size_y:.0f}m)"
    )
    print(f"  Eligible area: {twin.eligible_area_m2()/10000:.3f} ha")
    print(f"  Macro demand/deficit: {macro.demand_t:.0f} t / {macro.deficit_t:.0f} t")
    print(f"  Mothership waypoints: {result.path.n_waypoints}")
    print(f"  Steps / refills: {result.steps} / {result.hopper_refills}")
    print()
    print(result.kpis.format_text())

    if args.json_out:
        payload = {
            "macro": macro.as_dict(),
            "kpis": result.kpis.as_dict(),
            "swarm": result.allocator.summary(),
            "path": {
                "n_waypoints": result.path.n_waypoints,
                "length_m": result.path.total_length_m,
                "lane_spacing_cells": result.path.lane_spacing_cells,
                "excluded_cells": result.path.excluded_cells,
            },
        }
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2))
        print(f"\nWrote {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
