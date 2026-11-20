#!/usr/bin/env python3
"""End-to-end PruneMaxxing showcase.

mission sim (mothership control + lidar re-planning + drone swarm) ->
data exports (control log CSV, sorties, events, KPIs) -> map poster ->
tracking report -> MuJoCo 1:1 world replay (stills, cinematic video) ->
live dashboard video with MuJoCo camera panes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", "/usr/share/glvnd/egl_vendor.d/10_nvidia.json")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--payload", choices=["dust", "cartridge"], default="dust")
    ap.add_argument("--swarm", type=int, default=4)
    ap.add_argument("--out", default=None, help="default outputs/showcase_<payload>")
    ap.add_argument("--video-seconds", type=float, default=72.0, help="dashboard length")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--preview", action="store_true", help="short low-cost run (few frames)")
    ap.add_argument("--skip-mujoco", action="store_true")
    ap.add_argument("--skip-dashboard", action="store_true")
    ap.add_argument("--skip-cinematic", action="store_true")
    ap.add_argument("--poster-dpi", type=int, default=200)
    args = ap.parse_args()

    from prunemaxxing.config import DEFAULT_DEM_PATH, SHOWCASE_AREA_HA
    from prunemaxxing.geo.dem import load_dem
    from prunemaxxing.geo.grid import build_field_twin_from_dem, crop_target_field, select_pilot_window
    from prunemaxxing.geo.site import load_site
    from prunemaxxing.planning.mission import MissionConfig, run_mission
    from prunemaxxing.viz.basemap import build_basemap
    from prunemaxxing.viz.mission_viz import (
        export_mission_data,
        render_dashboard,
        render_map_poster,
        render_tracking_report,
    )

    out = Path(args.out) if args.out else ROOT / "outputs" / f"showcase_{args.payload}"
    out.mkdir(parents=True, exist_ok=True)

    log("site + digital twin")
    site = load_site()
    dem = select_pilot_window(crop_target_field(load_dem(ROOT / DEFAULT_DEM_PATH)), SHOWCASE_AREA_HA)
    twin = build_field_twin_from_dem(dem)

    log(f"mission simulation ({args.payload}, {args.swarm} drones)")
    t0 = time.time()
    res = run_mission(site, twin, MissionConfig(payload_mode=args.payload, swarm_size=args.swarm))
    k = res.kpis
    log(f"  {k['mission_time_h']:.1f} h mission, {k['sorties']:,} sorties, {k['biomass_applied_t']:.0f} t, "
        f"lane x-track RMS {100 * k['cte_rms_lane_m']:.1f} cm, {len(k['obstacles_detected'])} hidden obstacles "
        f"found ({time.time() - t0:.1f} s wall)")
    paths = {k2: str(v) for k2, v in export_mission_data(res, out).items()}

    log("basemap + poster + tracking report")
    bm = build_basemap(site)
    paths["map_poster"] = str(render_map_poster(res, out / "map_poster.png", bm, dpi=args.poster_dpi))
    phys = None
    phys_path = ROOT / "outputs" / "validation" / "rover_mujoco_summary.json"
    if phys_path.exists():
        pj = json.loads(phys_path.read_text())
        rb = pj.get("rms_cte_by_label", {})
        phys = {"completed": pj.get("completed"),
                "lane_rms_cm": round(100 * rb.get("lane_settled", rb.get("lane", float("nan"))), 1),
                "headland_rms_cm": round(100 * rb.get("headland", float("nan")), 1),
                "detour_rms_cm": round(100 * rb.get("detour", float("nan")), 1),
                "max_cm": round(100 * pj.get("max_cte", float("nan")), 1)}
    paths["tracking_report"] = str(render_tracking_report(res, out / "tracking_report.png", phys))

    cam_provider = None
    scene = None
    if not args.skip_mujoco:
        from prunemaxxing.sim.replay import ReplayScene, render_cinematic, render_hero_stills

        log("MuJoCo world + replay scene")
        scene = ReplayScene(res)
        for name, p in render_hero_stills(scene, out).items():
            paths[f"still_{name}"] = str(p)
        if not args.skip_cinematic:
            log("cinematic MuJoCo video")
            paths["cinematic"] = str(render_cinematic(scene, out / "cinematic.mp4", fps=args.fps,
                                                      preview=args.preview))
        cam_provider = scene.dashboard_cams(main_size=(1280, 720), second_size=(640, 360))

    if not args.skip_dashboard:
        log("dashboard video")
        paths["dashboard"] = str(render_dashboard(
            res, out / "dashboard.mp4", fps=args.fps, target_s=args.video_seconds, cam_provider=cam_provider,
            basemap=bm, max_frames=120 if args.preview else None, still_every=0 if args.preview else 400))

    meta = {"payload": args.payload, "swarm": args.swarm, "kpis": res.kpis, "outputs": paths}
    (out / "showcase_meta.json").write_text(json.dumps(meta, indent=1, default=float))
    log(f"done -> {out}")
    for k2, v in paths.items():
        print(f"  {k2:18s} {Path(v).relative_to(ROOT) if Path(v).is_absolute() else v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
