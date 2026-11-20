#!/usr/bin/env python3
"""Regenerate MuJoCo pilot DEM heightfield PNG + JSON sidecar."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dem",
        default=str(ROOT / "viladecans_full_dem_10m.TIF"),
        help="Path to viladecans_full_dem_10m.TIF",
    )
    parser.add_argument("--area-ha", type=float, default=6.0)
    parser.add_argument("--elevation-z", type=float, default=2.4, help="MuJoCo max relief (m)")
    parser.add_argument("--half-x", type=float, default=28.0, help="Arena half-extent X (m)")
    parser.add_argument("--upsample", type=float, default=8.0)
    parser.add_argument("--no-beds", action="store_true", help="Skip baked bed/furrow ripples")
    args = parser.parse_args()

    from prunemaxxing.sim.heightfield import generate_heightfield

    spec = generate_heightfield(
        dem_path=args.dem,
        area_ha=args.area_ha,
        elevation_z=args.elevation_z,
        half_x=args.half_x,
        upsample=args.upsample,
        bake_beds=not args.no_beds,
    )
    print(f"Wrote {spec.png_path}")
    print(f"  grid {spec.nrow}×{spec.ncol}  size=\"{spec.size_attr}\"  z_offset={spec.z_offset:.4f}")
    print(f"  DEM elev [{spec.elev_min_m:.2f}, {spec.elev_max_m:.2f}] m over "
          f"{spec.dem_width_m:.0f}×{spec.dem_height_m:.0f} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
