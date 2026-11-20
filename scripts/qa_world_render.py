#!/usr/bin/env python
"""Render QA stills of the 1:1 MuJoCo world into outputs/world_qa/.

    MUJOCO_GL=egl __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json \
        python scripts/qa_world_render.py [--only overview,ms_front] [--width 1920 --height 1080]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", "/usr/share/glvnd/egl_vendor.d/10_nvidia.json")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from prunemaxxing.sim import world_builder as wb  # noqa: E402
from prunemaxxing.sim import world_robots as wr  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "world_qa"


# ---------------------------------------------------------------------------
# Scene staging
# ---------------------------------------------------------------------------


def demo_completion(spec: wb.WorldSpec, ms_xy) -> np.ndarray:
    """Parcel A: done west of the robot's lane and behind it; parcel D: a gradient; rest untouched."""
    cg = spec.cells
    C = cg.centers()
    out = np.full((cg.nrows, cg.ncols), np.nan, np.float32)
    out[cg.work] = 0.0
    a = spec.beds[0]
    inA = cg.work & (C[..., 0] > a["bx0"]) & (C[..., 0] < a["bx1"]) & (C[..., 1] > a["by0"]) & (C[..., 1] < a["by1"])
    lane_col = int(np.floor((ms_xy[0] - cg.x0) / cg.dx))
    cols = np.arange(cg.ncols)[None, :].repeat(cg.nrows, 0)
    out[inA & (cols < lane_col)] = 1.0
    behind = inA & (cols == lane_col)
    prog = np.clip((ms_xy[1] - C[..., 1]) / cg.dy + 0.5, 0, 1)
    out[behind] = prog[behind]
    near = inA & (cols == lane_col + 1)
    out[near] = 0.35 * prog[near]
    d = spec.beds[3] if len(spec.beds) > 3 else None
    if d is not None:
        inD = cg.work & (C[..., 0] > d["bx0"]) & (C[..., 0] < d["bx1"]) & (C[..., 1] > d["by0"]) & (C[..., 1] < d["by1"])
        g = np.clip((d["bx1"] - 40 - C[..., 0]) / 60.0, 0, 1) * np.clip((d["by1"] - C[..., 1]) / 80.0, 0, 1)
        out[inD] = g[inD]
    return out


def _fx_curtain(data, ids, ks, src, ground_z, rng, spread=0.35, kind="dust"):
    """Falling particles from a cradle at `src` down to the ground (dust: every 4th is a puff)."""
    for mid, k in zip(ids, ks):
        puff = kind == "dust" and k % 4 == 3
        t = rng.uniform(0.7, 1.0) if puff else rng.random() ** 0.8
        z = src[2] - t * (src[2] - ground_z - (0.12 if puff else 0.05))
        r = spread * (0.35 + 1.4 * t) * np.sqrt(rng.random()) * (1.6 if puff else 1.0)
        a = rng.uniform(0, 2 * np.pi)
        data.mocap_pos[mid] = [src[0] + r * np.cos(a), src[1] + r * np.sin(a), z]
        data.mocap_quat[mid] = wr.quat_axis(rng.normal(size=3), rng.uniform(0, np.pi))


def stage(spec, model, data, spot: str = "laneA", mode: str = "dust", seed: int = 3) -> dict:
    """Place mothership + drones + FX for a QA scene; returns key positions."""
    rng = np.random.default_rng(seed)
    mujoco.mj_resetData(model, data)
    wb.park_fx(model, data)
    a = spec.beds[0]
    if spot == "laneA":
        x = spec.furrow_x(a["minx"] + 29.3, 0)
        y = -62.0
    elif spot == "tractor":
        x = spec.furrow_x(-29.2, 0)
        y = -121.5
    else:
        raise ValueError(spot)
    yaw = np.pi / 2
    wb.place_mothership(model, data, spec, x, y, yaw, wheel_angle=1.3, steer=0.06)
    mujoco.mj_forward(model, data)
    gz = lambda px, py: float(spec.ground_z(px, py))  # noqa: E731
    p = spec.beds[0]["pitch"]
    hover = [
        (x - 2 * p, y + 9.0, 5.2, 0.15, -1.25),
        (x + 3 * p, y + 4.5, 6.8, -0.4, -0.95),
        (x - 5 * p, y - 6.5, 4.4, 0.6, -1.45),
    ]
    out = {"ms": (x, y), "drones": []}
    for i in range(spec.swarm_size):
        if i < len(hover):
            hx, hy, hz, hyaw, lift = hover[i]
            z = gz(hx, hy) + hz
            wb.place_drone(model, data, i, hx, hy, z, yaw=yaw + hyaw, lift=lift,
                           rotor_phase=rng.uniform(0, 6), tilt=(0.04, -0.02))
            out["drones"].append((hx, hy, z))
        else:
            dk = wb.dock_world_pose(model, data, i % 4)
            wb.place_drone(model, data, i, dk[0], dk[1], dk[2] + wr.DRONE_SKID_Z, yaw=yaw, lift=0.0,
                           rotor_phase=0.0)
            out["drones"].append(tuple(dk))
    mujoco.mj_forward(model, data)
    # FX from the cradles of hovering drones
    kind = "dust" if mode == "dust" else "cart"
    ids = wb.fx_mocap_ids(model, kind)
    per = len(ids) // 3
    for j in range(min(3, spec.swarm_size)):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"drone{j}_cradle")
        src = data.xpos[bid] + np.array([0, 0, -0.25])
        ks = range(j * per, (j + 1) * per)
        _fx_curtain(data, ids[j * per:(j + 1) * per], ks, src, gz(src[0], src[1]), rng,
                    spread=0.45 if kind == "dust" else 0.3, kind=kind)
    mujoco.mj_forward(model, data)
    return out


def free_cam(lookat, distance, azimuth, elevation) -> mujoco.MjvCamera:
    c = mujoco.MjvCamera()
    c.type = mujoco.mjtCamera.mjCAMERA_FREE
    c.lookat[:] = lookat
    c.distance = distance
    c.azimuth = azimuth
    c.elevation = elevation
    return c


def save(img: np.ndarray, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{name}.png"
    Image.fromarray(img).save(p)
    return p


# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--only", type=str, default="")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    only = set(s for s in args.only.split(",") if s)

    t0 = time.time()
    spec = wb.build_world(payload_mode="dust", force=args.force)
    t1 = time.time()
    model = wb.load_world_model(spec)
    data = mujoco.MjData(model)
    t2 = time.time()
    r = wb.make_renderer(model, args.height, args.width)
    painter = wb.MulchPainter(spec, model, "dust")
    t3 = time.time()
    print(f"build {t1 - t0:.2f}s, compile {t2 - t1:.2f}s, renderer+painter {t3 - t2:.2f}s, ngeom {model.ngeom}")

    want = lambda n: not only or n in only  # noqa: E731
    shots = []

    def shot(name, cam, focus=None, half=45.0):
        if not want(name):
            return
        wb.focus_shadows(r, model, data, focus, half)
        ts = time.perf_counter()
        r.update_scene(data, camera=cam)
        img = r.render()
        dt = (time.perf_counter() - ts) * 1000
        path = save(img, name)
        shots.append((name, dt, path))
        print(f"  {name:28s} {dt:6.1f} ms -> {path.relative_to(ROOT)}")

    # ---- scene 1: lane in parcel A (dust) ----
    pos = stage(spec, model, data, "laneA", "dust")
    comp = demo_completion(spec, pos["ms"])
    ms_full = painter.apply(model, r, comp, force=True)
    comp2 = comp.copy()
    lane_col = int(np.floor((pos["ms"][0] - spec.cells.x0) / spec.cells.dx))
    comp2[:, lane_col + 1] = np.where(np.isfinite(comp2[:, lane_col + 1]), comp2[:, lane_col + 1] + 0.05,
                                      comp2[:, lane_col + 1])
    ms_inc = painter.apply(model, r, comp2)
    ms_inc2 = painter.apply(model, r, comp)
    print(f"MulchPainter: full {ms_full:.1f} ms, one-column update {ms_inc:.1f} / {ms_inc2:.1f} ms")
    wb.set_obstacle_detected(model, "unk_tractor_A", True)
    msx, msy = pos["ms"]
    mz = float(spec.terrain.height_at(msx, msy))
    focus = np.array([msx, msy + 3, mz])

    shot("cam_overview", "cam_overview")
    shot("cam_top", "cam_top")
    shot("cam_low_field", "cam_low_field")
    shot("ms_mast", "ms_mast", focus, 40)
    shot("ms_front", "ms_front", focus, 40)
    shot("ms_side", "ms_side", focus, 30)
    shot("drone0_cam", "drone0_cam", focus, 40)
    shot("cine_hero", free_cam(focus + [0, 1.5, 2.8], 19.0, -128.0, -14.0), focus, 40)
    shot("cine_drone_cradle", free_cam(np.array(pos["drones"][0]) + [0, 0, -1.6], 7.5, -62.0, -6.0), focus, 30)
    shot("cine_low_wheels", free_cam(focus + [0.8, -0.5, 0.9], 8.5, -18.0, -6.0), focus, 25)
    shot("cine_swarm_high", free_cam(focus + [0, 2, 2], 42.0, -150.0, -32.0), focus, 60)
    shot("wide_poplars_ditch", "cam_ditch")
    shot("wide_greenhouses_masia", "cam_farmstead")
    hat = spec.terrain.height_at
    rx, ry = -168.0, -48.0
    shot("cine_reeds_ditch", free_cam([rx, ry, float(hat(rx, ry)) + 2.2], 16.0, 150.0, -6.0), [rx, ry, 0], 40)
    lm = spec.meta.get("landmarks", {})
    if lm.get("houses"):
        hx, hy = lm["houses"][0]
        shot("cine_masia", free_cam([hx, hy, float(hat(hx, hy)) + 4.0], 34.0, -60.0, -12.0), [hx, hy, 0], 45)
    tr = lm.get("obstacles", {}).get("unk_log_D")
    if tr:
        shot("cine_log_obstacle", free_cam([tr[0], tr[1], float(hat(*tr)) + 0.8], 14.0, -70.0, -20.0), [tr[0], tr[1], 0], 30)

    # ---- scene 2: approaching the hidden tractor ----
    if want("near_tractor") or want("near_tractor_front"):
        pos2 = stage(spec, model, data, "tractor", "dust", seed=8)
        tx, ty = pos2["ms"]
        f2 = np.array([tx, ty + 6, float(spec.terrain.height_at(tx, ty))])
        shot("near_tractor", free_cam(f2 + [0, 2, 1.5], 26.0, -115.0, -18.0), f2, 45)
        shot("near_tractor_front", "ms_front", f2, 40)

    # ---- cartridge mode ----
    if want("cartridge_hero") or want("cartridge_front"):
        wb.set_payload_mode(model, "cartridge")
        painter_c = wb.MulchPainter(spec, model, "cartridge")
        pos = stage(spec, model, data, "laneA", "cartridge")
        painter_c.apply(model, r, demo_completion(spec, pos["ms"]), force=True)
        shot("cartridge_hero", free_cam(focus + [0, 1.5, 2.6], 17.0, -135.0, -16.0), focus, 40)
        shot("cartridge_cradle", free_cam(np.array(pos["drones"][0]) + [0, 0, -1.6], 6.5, -70.0, -4.0), focus, 30)
        shot("cartridge_front", "ms_front", focus, 40)
        wb.set_payload_mode(model, "dust")
        painter.apply(model, r, comp, force=True)

    # ---- timing ----
    if not only or "timing" in only:
        cam = "ms_mast"
        for (h, w) in ((1080, 1920), (720, 1280)):
            rr = r if (h, w) == (args.height, args.width) else wb.make_renderer(model, h, w)
            wb.focus_shadows(rr, model, data, focus, 40)
            for _ in range(3):
                rr.update_scene(data, camera=cam)
                rr.render()
            ts = time.perf_counter()
            for _ in range(10):
                rr.update_scene(data, camera=cam)
                rr.render()
            print(f"render {w}x{h} ({cam}): {(time.perf_counter() - ts) * 100:.1f} ms/frame (update_scene+render+readback)")
            wb.focus_shadows(rr, model, data, None)
            ts = time.perf_counter()
            for _ in range(10):
                rr.update_scene(data, camera="cam_overview")
                rr.render()
            print(f"render {w}x{h} (cam_overview): {(time.perf_counter() - ts) * 100:.1f} ms/frame")
    print(f"done: {len(shots)} stills in {OUT}")


if __name__ == "__main__":
    main()
