#!/usr/bin/env python
"""Build (or reuse cached) the 1:1 MuJoCo world: textures, meshes, hfield, MJCF.

    MUJOCO_GL=egl python scripts/build_world.py --payload dust
    python scripts/build_world.py --payload cartridge --force
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prunemaxxing.sim import world_builder as wb  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--payload", choices=["dust", "cartridge"], default="dust")
    ap.add_argument("--swarm", type=int, default=4)
    ap.add_argument("--texture-px", type=int, default=4096)
    ap.add_argument("--out", type=Path, default=wb.DEFAULT_OUT)
    ap.add_argument("--force", action="store_true", help="regenerate all assets even if cached")
    ap.add_argument("--no-compile", action="store_true", help="skip the MuJoCo compile check")
    args = ap.parse_args()

    spec = wb.build_world(payload_mode=args.payload, swarm_size=args.swarm, out_dir=args.out,
                          texture_px=args.texture_px, force=args.force)
    print(f"xml: {spec.xml_path}")
    print(f"terrain z: {spec.h_min:.2f} .. {spec.h_max:.2f} m, hfield {spec.half_x * 2:.0f} x {spec.half_y * 2:.0f} m")
    print(f"cells: {spec.cells.nrows} x {spec.cells.ncols} (work cells {int(spec.cells.work.sum())})")
    if not args.no_compile:
        t = time.time()
        m = wb.load_world_model(spec)
        print(f"compiled in {time.time() - t:.2f}s: ngeom={m.ngeom} nbody={m.nbody} nmesh={m.nmesh} "
              f"ntex={m.ntex} nmocap={m.nmocap}")


if __name__ == "__main__":
    main()
