#!/usr/bin/env python
"""Render the PruneMaxxing climate-impact posters (combined + one per payload mode).

    python scripts/render_impact_poster.py --from-pickles        # reuse outputs/dev/mission_*.pkl
    python scripts/render_impact_poster.py                       # re-run both full missions (~minutes)

Writes to outputs/impact/: impact_poster.png (both modes), impact_dust.png, impact_cartridge.png and
impact_summary.json (impact summaries, factor table, equivalence factors, sources).
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODES = ("dust", "cartridge")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def _load_pickles(pickle_dir: Path) -> dict:
    out = {}
    for mode in MODES:
        p = pickle_dir / f"mission_{mode}.pkl"
        if not p.exists():
            raise SystemExit(f"missing {p}; run without --from-pickles to simulate the missions")
        with p.open("rb") as fh:
            out[mode] = pickle.load(fh)
        log(f"  loaded {p.relative_to(ROOT) if p.is_relative_to(ROOT) else p}")
    return out


def _run_missions() -> dict:
    from prunemaxxing.config import DEFAULT_DEM_PATH, SHOWCASE_AREA_HA
    from prunemaxxing.geo.dem import load_dem
    from prunemaxxing.geo.grid import build_field_twin_from_dem, crop_target_field, select_pilot_window
    from prunemaxxing.geo.site import load_site
    from prunemaxxing.planning.mission import MissionConfig, run_mission

    site = load_site()
    dem = select_pilot_window(crop_target_field(load_dem(ROOT / DEFAULT_DEM_PATH)), SHOWCASE_AREA_HA)
    twin = build_field_twin_from_dem(dem)
    out = {}
    for mode in MODES:
        t0 = time.time()
        log(f"  mission simulation ({mode})")
        out[mode] = run_mission(site, twin, MissionConfig(payload_mode=mode))
        log(f"    {out[mode].kpis['mission_time_h']:.1f} h mission ({time.time() - t0:.1f} s wall)")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-pickles", action="store_true", help="load mission_{dust,cartridge}.pkl instead of simulating")
    ap.add_argument("--pickle-dir", default=str(ROOT / "outputs" / "dev"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "impact"))
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--no-map", action="store_true", help="skip the coverage-map thumbnails")
    args = ap.parse_args()

    from prunemaxxing.kpis.impact import factor_table, impact_summary
    from prunemaxxing.viz.basemap import build_basemap
    from prunemaxxing.viz.impact_poster import EQUIVALENCES, factor_lines, render_impact_poster

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    log("missions")
    results = _load_pickles(Path(args.pickle_dir)) if args.from_pickles else _run_missions()

    bm = None
    if not args.no_map:
        log("basemap")
        bm = build_basemap(results["dust"].site)

    jobs = [("impact_poster", [results["dust"], results["cartridge"]]),
            ("impact_dust", results["dust"]),
            ("impact_cartridge", results["cartridge"])]
    paths, timings = {}, {}
    for name, res in jobs:
        t0 = time.time()
        p = render_impact_poster(res, out / f"{name}.png", dpi=args.dpi, basemap=bm, with_map=not args.no_map)
        timings[name] = round(time.time() - t0, 2)
        paths[name] = str(p)
        log(f"  {name}.png ({timings[name]} s)")

    summary = {
        "summaries": {mode: impact_summary(res) for mode, res in results.items()},
        "factor_table": factor_table(),
        "equivalences": EQUIVALENCES,
        "footnotes": factor_lines(),
        "notes": ["Biogenic CO2 from pile burning is context only and never added to total_co2e_t.",
                  "The two modes are alternative missions on the same parcels; their figures are not summed."],
        "source": "pickles" if args.from_pickles else "simulation",
        "outputs": paths,
        "render_s": timings,
        "dpi": args.dpi,
    }
    sp = out / "impact_summary.json"
    sp.write_text(json.dumps(summary, indent=1, default=_json_default))
    paths["impact_summary"] = str(sp)

    log(f"done -> {out}")
    for k, v in paths.items():
        rel = Path(v).relative_to(ROOT) if Path(v).is_relative_to(ROOT) else v
        print(f"  {k:18s} {rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
