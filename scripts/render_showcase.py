#!/usr/bin/env python3
"""End-to-end PruneMaxxing wow showcase: polygons, tracking, MuJoCo cams, videos."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# NVIDIA EGL before mujoco import chain
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault(
    "__EGL_VENDOR_LIBRARY_FILENAMES",
    "/usr/share/glvnd/egl_vendor.d/10_nvidia.json",
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dem", default=str(ROOT / "viladecans_full_dem_10m.TIF"))
    parser.add_argument(
        "--geojson", default=str(ROOT / "data" / "parcels_weeds.geojson")
    )
    parser.add_argument("--out", default=str(ROOT / "outputs" / "showcase"))
    parser.add_argument("--area-ha", type=float, default=6.0)
    parser.add_argument("--kg-per-m2", type=float, default=15.0)
    parser.add_argument("--swarm", type=int, default=4)
    parser.add_argument("--passes", type=int, default=5)
    parser.add_argument(
        "--path-style",
        choices=["complex", "serpentine"],
        default="complex",
        help="complex=curved headlands/weed detours/spiral; serpentine=classic lawnmower",
    )
    parser.add_argument(
        "--payload",
        choices=["dust", "cartridge"],
        default="dust",
        help="dust=wood shavings; cartridge=seed+mycelium wood-chip units",
    )
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--skip-mujoco", action="store_true")
    args = parser.parse_args()

    from prunemaxxing.geo.dem import load_dem
    from prunemaxxing.geo.grid import (
        build_field_twin_from_dem,
        crop_target_field,
        select_pilot_window,
    )
    from prunemaxxing.geo.polygons import apply_map_layers, load_map_layers
    from prunemaxxing.planning.payload_modes import PayloadMode
    from prunemaxxing.planning.simulator import run_coverage_simulation
    from prunemaxxing.viz.showcase import render_showcase

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("== Loading DEM + GeoJSON parcels/weeds ==")
    dem = load_dem(args.dem)
    pilot_dem = select_pilot_window(
        crop_target_field(dem), area_ha=args.area_ha, anchor="sw"
    )
    twin = build_field_twin_from_dem(pilot_dem, target_kg_per_m2=args.kg_per_m2)
    layers = load_map_layers(args.geojson)
    counts = apply_map_layers(twin, layers)
    print(f"  Weed cells flagged: {counts['weed_cells']}")

    print("== Coverage sim + bed-locked path tracking ==")
    result = run_coverage_simulation(
        twin,
        swarm_size=max(3, args.swarm),
        wireless_power=True,
        max_passes=args.passes,
        payload_mode=PayloadMode(args.payload),
        record_frames=True,
        frame_stride=max(1, args.passes),  # fewer twin frames for snappy video
        path_style=args.path_style,
    )
    print(
        f"  path_style={args.path_style}  waypoints={result.path.n_waypoints}  "
        f"drones={len(result.drone_planned)}  "
        f"drone_path_pts={[len(v) for v in result.drone_planned.values()]}"
    )
    print(result.kpis.format_text())

    mujoco_cams = None
    if not args.skip_mujoco:
        print("== MuJoCo robot-camera renders ==")
        try:
            from prunemaxxing.sim.mujoco_runner import (
                animate_and_render,
                render_still_hero,
                write_effector_showcase_stills,
            )
            import imageio.v2 as imageio

            hero = render_still_hero(PayloadMode(args.payload), camera="hero_orbit")
            imageio.imwrite(out_dir / "mujoco_hero.png", hero)
            imageio.imwrite(out_dir / "mujoco_hero_dem.png", hero)
            fx_stills = write_effector_showcase_stills(out_dir)
            print(f"  Effector stills: {list(fx_stills.keys())}")
            mujoco_cams = animate_and_render(
                result,
                max_frames=min(args.max_frames, 80),
                payload_mode=PayloadMode(args.payload),
                width=640,
                height=368,
            )
            print(f"  Cameras: {list(mujoco_cams.keys())}")
        except Exception as e:
            print(f"  MuJoCo render skipped: {type(e).__name__}: {e}")

    print("== Compositing showcase dashboard + videos ==")
    outputs = render_showcase(
        result,
        out_dir=out_dir,
        layers=layers,
        mujoco_cams=mujoco_cams,
        max_frames=args.max_frames,
        fps=8,
    )

    meta = {
        "kpis": result.kpis.as_dict(),
        "weed_cells": counts["weed_cells"],
        "payload_mode": args.payload,
        "n_frames": len(result.frames),
        "planned_points": int(len(result.planned_polyline)),
        "tracked_points": int(len(result.actual_polyline)),
        "outputs": {k: str(v) for k, v in outputs.items()},
    }
    (out_dir / "showcase_meta.json").write_text(json.dumps(meta, indent=2))
    print("\nOutputs:")
    for k, v in outputs.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
