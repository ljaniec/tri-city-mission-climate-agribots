#!/usr/bin/env python3
"""Headless mission run on the Viladecans pilot site: KPIs + logs, no rendering (~3 s)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prunemaxxing.config import DEFAULT_DEM_PATH, SHOWCASE_AREA_HA, PilotSpec  # noqa: E402
from prunemaxxing.geo.dem import load_dem  # noqa: E402
from prunemaxxing.geo.grid import build_field_twin_from_dem, crop_target_field, select_pilot_window  # noqa: E402
from prunemaxxing.geo.site import load_site  # noqa: E402
from prunemaxxing.logistics.macro_router import plan_macro_logistics  # noqa: E402
from prunemaxxing.planning.mission import MissionConfig, run_mission  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--payload", choices=["dust", "cartridge"], default="dust")
    ap.add_argument("--swarm", type=int, default=4)
    ap.add_argument("--out", default="", help="directory for control_log.csv, sorties.csv, kpis.json")
    args = ap.parse_args()

    site = load_site()
    dem = select_pilot_window(crop_target_field(load_dem(ROOT / DEFAULT_DEM_PATH)), SHOWCASE_AREA_HA)
    twin = build_field_twin_from_dem(dem)
    res = run_mission(site, twin, MissionConfig(payload_mode=args.payload, swarm_size=args.swarm))
    k = res.kpis
    macro = plan_macro_logistics(PilotSpec(area_ha=k["area_treated_ha"], target_kg_per_m2=k["dose_kg_per_m2"]))

    print(f"PruneMaxxing mission — {args.payload} mode, {args.swarm} drones")
    print(f"  route: {res.plan.length_m:.0f} m planned, {len(res.plan.lanes)} lanes, {len(res.plan.stations)} stations")
    print(f"  macro demand / local deficit: {macro.demand_t:.0f} t / {macro.deficit_t:.0f} t")
    width = max(len(key) for key in k)
    for key, val in k.items():
        if key == "obstacles_detected":
            for ob in val:
                print(f"  {'obstacle':{width}s}  {ob}")
            continue
        print(f"  {key:{width}s}  {val}")
    if args.out:
        from prunemaxxing.viz.mission_viz import export_mission_data

        paths = export_mission_data(res, Path(args.out))
        (Path(args.out) / "macro.json").write_text(json.dumps(macro.as_dict(), indent=1))
        print("\n".join(f"  wrote {p}" for p in paths.values()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
