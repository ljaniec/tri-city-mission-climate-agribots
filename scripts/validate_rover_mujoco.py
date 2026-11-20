#!/usr/bin/env python3
"""MuJoCo physics closed-loop path-tracking validation of the mothership UGV.

Reference (real site frame, parcel A): furrow-centred lane north 110 m, headland
U-turn (2 x R6.5 quarter arcs + straight, lane spacing 59.5 m), lane south 60 m,
S-curve detour (+5 m, 14 m cosine blends) around a 6 m obstacle, 30 m lane.
The rover is driven by pure pursuit at 20 Hz on the DEM heightfield.

Outputs (outputs/validation/):
  rover_mujoco_log.csv, rover_mujoco_tracking.png, rover_mujoco_summary.json,
  rover_mujoco_chase.mp4
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", "/usr/share/glvnd/egl_vendor.d/10_nvidia.json")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from prunemaxxing.geo.site import load_site  # noqa: E402
from prunemaxxing.sim.rover_physics import PathBuilder, RoverRunResult, run_closed_loop  # noqa: E402
from prunemaxxing.sim.world_frame import frame_for_site  # noqa: E402

OUT = ROOT / "outputs" / "validation"
FURROW = 0.85
LANE_OFFSET = 29.325  # 34.5 furrow pitches -> lane centre on a furrow
LANE_SPACING = 59.5
TURN_R = 6.5
START_MARGIN = 18.0  # from parcel south edge (keeps headland and the path end inside A)
DETOUR_OFFSET = 5.0  # left of travel (east on the southbound lane, away from lane 1)
DETOUR_BLEND = 14.0
DETOUR_HOLD = 8.0  # 6 m obstacle + 1 m each side
OBSTACLE = {"lx": 3.0, "ly": 6.0, "h": 1.6}

TARGETS = {"lane_settled_rms_cte_m": 0.08, "lane_settled_max_cte_m": 0.20, "headland_max_cte_m": 0.60,
           "wall_time_s": 120.0}


def build_reference():
    site = load_site()
    frame = frame_for_site(site)
    e0, n0, e1, n1 = site.parcels[0].polygon.bounds
    (ax0, ax1), (ay0, ay1) = frame.to_world(np.array([e0, e1]), np.array([n0, n1]))
    ax0, ax1, ay0, ay1 = float(ax0), float(ax1), float(ay0), float(ay1)
    lane_x = ax0 + LANE_OFFSET
    y0 = ay0 + START_MARGIN
    pb = PathBuilder(lane_x, y0, math.pi / 2, ds=0.25)
    pb.straight(110.0, "lane")
    pb.arc(TURN_R, -math.pi / 2, "headland")
    pb.straight(LANE_SPACING - 2 * TURN_R, "headland")
    pb.arc(TURN_R, -math.pi / 2, "headland")
    pb.straight(60.0, "lane")
    det_start_y = pb.y
    pb.detour(DETOUR_OFFSET, DETOUR_BLEND, DETOUR_HOLD, "detour")
    det_end_y = pb.y
    pb.straight(30.0, "lane")
    xy, labels = pb.build()
    lane2_x = lane_x + LANE_SPACING
    obstacle = dict(OBSTACLE, x=lane2_x, y=det_start_y - DETOUR_BLEND - DETOUR_HOLD / 2, yaw=0.0)
    geo = {
        "parcel_world": (ax0, ay0, ax1, ay1),
        "lane1_x": lane_x,
        "lane2_x": lane2_x,
        "lane_top_y": y0 + 110.0,
        "detour_start_y": det_start_y,
        "detour_end_y": det_end_y,
        "frame": {"center_e": frame.center_e, "center_n": frame.center_n},
        "furrow_x0": ax0 + 0.5 * FURROW,
    }
    return xy, labels, obstacle, geo


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

BG = "#0d1117"
PANEL = "#151b23"
GRID = "#2a323d"
TXT = "#d8dee9"
C_PLAN = {"lane": "#39c5ff", "headland": "#b48cff", "detour": "#ffd166"}
C_ACT = "#ff7a1a"
C_FRONT = "#ff4fa0"


def _wheel_tracks(res: RoverRunResult, L=2.4, T=1.7):
    c, s = np.cos(res.yaw), np.sin(res.yaw)
    p = np.column_stack([res.x, res.y])
    fwd = np.column_stack([c, s])
    left = np.column_stack([-s, c])
    return {"rl": p + T / 2 * left, "rr": p - T / 2 * left,
            "fl": p + L * fwd + T / 2 * left, "fr": p + L * fwd - T / 2 * left}


def _style(ax):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.tick_params(colors=TXT, labelsize=9)
    ax.xaxis.label.set_color(TXT)
    ax.yaxis.label.set_color(TXT)
    ax.title.set_color(TXT)
    ax.grid(True, color=GRID, lw=0.6, alpha=0.8)


def _plot_plan(ax, res, lw=1.6, zorder=3):
    lab = res.path_labels
    xy = res.path_xy
    start = 0
    for i in range(1, len(lab) + 1):
        if i == len(lab) or lab[i] != lab[start]:
            seg = xy[start:min(i + 1, len(xy))]
            ax.plot(seg[:, 0], seg[:, 1], color=C_PLAN.get(lab[start], "#8ecae6"), lw=lw, zorder=zorder)
            start = i


def make_figure(res: RoverRunResult, geo: dict, obstacle: dict, out_png: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    s = res.summary
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor=BG)
    gs = fig.add_gridspec(3, 3, width_ratios=[0.92, 0.42, 1.55], left=0.035, right=0.99, top=0.895, bottom=0.085,
                          wspace=0.16, hspace=0.36)
    ax_map = fig.add_subplot(gs[:, 0])
    gz = gs[:, 1].subgridspec(2, 1, height_ratios=[0.42, 0.58], hspace=0.16)
    ax_za = fig.add_subplot(gz[0])
    ax_zb = fig.add_subplot(gz[1])
    ax_cte = fig.add_subplot(gs[0, 2])
    ax_st = fig.add_subplot(gs[1, 2])
    ax_v = fig.add_subplot(gs[2, 2])
    for ax in (ax_map, ax_za, ax_zb, ax_cte, ax_st, ax_v):
        _style(ax)

    ok = s["completed"]
    fig.suptitle("MuJoCo physics closed loop: 4 t Ackermann mothership UGV on the Viladecans DEM "
                 "(parcel A, pure pursuit @ 20 Hz, actuator-in-the-loop)",
                 color=TXT, fontsize=17, fontweight="bold", x=0.5, y=0.975)
    b = s["by_label"]
    sub = (f"completed: {'YES' if ok else 'NO'} ({s['end_reason']})   |   {s['duration_s']:.0f} s, "
           f"{s['distance_travelled_m']:.0f} m driven, mean {s['mean_speed']:.2f} m/s   |   "
           f"lane (settled {b['lane_settled']['settle_m']:.0f} m) RMS {100 * b['lane_settled']['rms_cte']:.1f} cm / "
           f"max {100 * b['lane_settled']['max_cte']:.1f} cm   |   headland max {100 * b['headland']['max_cte']:.1f} cm"
           f"   |   detour max {100 * b['detour']['max_cte']:.1f} cm   |   steer saturated "
           f"{s['pct_time_steer_saturated']:.1f} %")
    fig.text(0.5, 0.935, sub, color="#a3e635" if ok else "#ff6b6b", ha="center", fontsize=12)

    # ---- top-down overview ---------------------------------------------------
    ax0, ay0, ax1, ay1 = geo["parcel_world"]
    ax_map.add_patch(Rectangle((ax0, ay0), ax1 - ax0, ay1 - ay0, fill=False, ec="#6b7280", lw=1.2, ls="--",
                               zorder=1))
    ax_map.text(geo["lane1_x"] - 8, ay0 + 1.0, "parcel A boundary", color="#9ca3af", fontsize=9)
    _plot_plan(ax_map, res, lw=2.2)
    ax_map.plot(res.x, res.y, color=C_ACT, lw=1.2, zorder=4)
    ob = Rectangle((obstacle["x"] - obstacle["lx"] / 2, obstacle["y"] - obstacle["ly"] / 2), obstacle["lx"],
                   obstacle["ly"], fc="#c0392b", ec="#ff8a80", lw=1.0, zorder=5)
    ax_map.add_patch(ob)
    ax_map.plot(res.path_xy[0, 0], res.path_xy[0, 1], "o", color="#a3e635", ms=8, zorder=6)
    ax_map.plot(res.path_xy[-1, 0], res.path_xy[-1, 1], "s", color="#ff6b6b", ms=8, zorder=6)
    ax_map.text(res.path_xy[0, 0] - 2, res.path_xy[0, 1] - 4, "start", color="#a3e635", fontsize=10, ha="right")
    ax_map.text(res.path_xy[-1, 0] + 2, res.path_xy[-1, 1] - 2, "end", color="#ff6b6b", fontsize=10)
    l1, l2 = geo["lane1_x"], geo["lane2_x"]
    ytop = geo["lane_top_y"]
    ax_map.set_xlim(l1 - 9, l2 + 12)
    ax_map.set_ylim(res.path_xy[:, 1].min() - 5, res.path_xy[:, 1].max() + 5)
    ax_map.set_aspect("equal")
    ax_map.set_xlabel("world x = easting - %.1f  [m]" % geo["frame"]["center_e"])
    ax_map.set_ylabel("world y = northing - %.1f  [m]" % geo["frame"]["center_n"])
    ax_map.set_title("top-down: planned (by segment) vs MuJoCo rear-axle trajectory", fontsize=11)
    handles = [Line2D([], [], color=C_PLAN["lane"], lw=2.2, label="plan: lane"),
               Line2D([], [], color=C_PLAN["headland"], lw=2.2, label="plan: headland"),
               Line2D([], [], color=C_PLAN["detour"], lw=2.2, label="plan: detour"),
               Line2D([], [], color=C_ACT, lw=1.6, label="actual (rear axle)"),
               Line2D([], [], color="#e5e7eb", lw=0.8, ls=":", label="actual wheel tracks"),
               Line2D([], [], color="#7c5a3a", lw=0.8, label="furrow centres (0.85 m)")]
    ax_map.legend(handles=handles, loc="center", fontsize=9, facecolor=PANEL, edgecolor=GRID,
                  labelcolor=TXT, framealpha=0.95)

    tracks = _wheel_tracks(res)

    def inset(ins, xlim, ylim, title, y_furrow_max=None):
        # widen the requested window so equal aspect fills the panel box
        bb = ins.get_position()
        wpx, hpx = bb.width * fig.get_figwidth(), bb.height * fig.get_figheight()
        dx, dy = xlim[1] - xlim[0], ylim[1] - ylim[0]
        sc = min(wpx / dx, hpx / dy)
        cx, cy = 0.5 * (xlim[0] + xlim[1]), 0.5 * (ylim[0] + ylim[1])
        xlim = (cx - wpx / sc / 2, cx + wpx / sc / 2)
        ylim = (cy - hpx / sc / 2, cy + hpx / sc / 2)
        ins.tick_params(labelsize=8)
        fx0 = geo["furrow_x0"]
        k0, k1 = math.floor((xlim[0] - fx0) / FURROW), math.ceil((xlim[1] - fx0) / FURROW)
        ytop_f = ylim[1] if y_furrow_max is None else min(ylim[1], y_furrow_max)
        for kk in range(k0, k1 + 1):
            xf = fx0 + kk * FURROW
            ins.plot([xf, xf], [ylim[0], ytop_f], color="#7c5a3a", lw=0.7, zorder=1)
        _plot_plan(ins, res, lw=2.0, zorder=2)
        for name, tr in tracks.items():
            ins.plot(tr[:, 0], tr[:, 1], color="#e5e7eb" if name[0] == "r" else "#f9a8d4", lw=0.7, ls=":",
                     zorder=3)
        ins.plot(res.x, res.y, color=C_ACT, lw=1.4, zorder=4)
        ins.add_patch(Rectangle((obstacle["x"] - obstacle["lx"] / 2, obstacle["y"] - obstacle["ly"] / 2),
                                obstacle["lx"], obstacle["ly"], fc="#c0392b", ec="#ff8a80", lw=1.0, zorder=5))
        ins.set_xlim(*xlim)
        ins.set_ylim(*ylim)
        ins.set_aspect("equal", adjustable="box")
        ins.set_title(title, fontsize=10, color=TXT, pad=4)
        ax_map.indicate_inset_zoom(ins, edgecolor="#e5e7eb", alpha=0.7, lw=0.9)
        return ins

    inset(ax_za, (l2 - 8.0, l2 + 3.0), (ytop - 10, ytop + 7.5),
          "zoom A: headland -> lane 2 entry", y_furrow_max=ytop)
    inset(ax_zb, (l2 - 3.0, l2 + 8.5), (geo["detour_end_y"] - 3, geo["detour_start_y"] + 3),
          "zoom B: S-curve detour (+5 m)")

    # ---- cross-track vs s ------------------------------------------------------
    sl = res.path_s
    lab = res.path_labels
    start = 0
    for i in range(1, len(lab) + 1):
        if i == len(lab) or lab[i] != lab[start]:
            l_ = lab[start]
            if l_ != "lane":
                ax_cte.axvspan(sl[start], sl[min(i, len(sl) - 1)], color=C_PLAN.get(l_, "#888"), alpha=0.12, lw=0)
                ax_cte.text(0.5 * (sl[start] + sl[min(i, len(sl) - 1)]), 0.92, l_, transform=ax_cte.get_xaxis_transform(),
                            color=C_PLAN.get(l_, TXT), ha="center", fontsize=9)
                ax_cte.axvspan(sl[min(i, len(sl) - 1)], sl[min(i, len(sl) - 1)] + b["lane_settled"]["settle_m"],
                               color="#9ca3af", alpha=0.18, lw=0, hatch="//")
            start = i
    ax_cte.axhspan(-0.2, 0.2, color="#22c55e", alpha=0.10, lw=0)
    ax_cte.axhline(0.2, color="#22c55e", lw=1.0, ls="--")
    ax_cte.axhline(-0.2, color="#22c55e", lw=1.0, ls="--")
    ax_cte.axhline(0, color="#6b7280", lw=0.8)
    ax_cte.plot(res.s_front, res.cross_track_front, color=C_FRONT, lw=0.8, alpha=0.75,
                label="front axle (incl. geometric off-tracking, ~0.43 m on R6.5 arcs)")
    ax_cte.plot(res.s_along, res.cross_track, color=C_ACT, lw=1.5, label="rear axle (control point, logged cross_track)")
    peak = max(float(np.max(np.abs(res.cross_track))), float(np.max(np.abs(res.cross_track_front))))
    lim = float(np.clip(1.1 * peak, 0.3, 0.65))
    ax_cte.set_ylim(-lim, lim)
    ax_cte.set_xlim(0, res.path_s[-1])
    ax_cte.set_xlabel("arc length along reference s [m]")
    ax_cte.set_ylabel("cross-track error [m]  (+ = left)")
    ax_cte.set_title("cross-track error vs s  (green band = +/-0.2 m furrow tolerance; hatched = 4 m settling "
                     "excluded from lane stats)", fontsize=11)
    ax_cte.legend(loc="lower left", fontsize=9, facecolor=PANEL, edgecolor=GRID, labelcolor=TXT, ncol=1)

    # ---- steering --------------------------------------------------------------
    lim_deg = math.degrees(s["rover_params"]["steer_limit"])
    ax_st.axhline(lim_deg, color="#ef4444", lw=0.9, ls="--")
    ax_st.axhline(-lim_deg, color="#ef4444", lw=0.9, ls="--")
    ax_st.text(1.0, lim_deg, "steer limit 0.38 rad (R_min 6.0 m)", color="#ef4444", fontsize=8, va="bottom",
               ha="left")
    ax_st.plot(res.t, np.degrees(res.steer_cmd), color="#60a5fa", lw=1.3, label="steer_cmd (bicycle, rate-limited)")
    ax_st.plot(res.t, np.degrees(res.steer), color="#fbbf24", lw=1.0, label="steer measured (knuckles -> bicycle)")
    ax_st.set_xlim(0, res.t[-1])
    ax_st.set_ylabel("steering angle [deg]")
    ax_st.set_title("steering: command vs MuJoCo knuckle servo response", fontsize=11)
    ax_st.legend(loc="lower left", fontsize=9, facecolor=PANEL, edgecolor=GRID, labelcolor=TXT, ncol=1)

    # ---- speed -----------------------------------------------------------------
    ax_v.plot(res.t, res.v_cmd, color="#60a5fa", lw=1.3, label="v_cmd (profile, 0.4 m/s^2 ramps)")
    ax_v.plot(res.t, res.v, color="#34d399", lw=1.0, label="v measured (rear axle, physics)")
    ax_v.set_xlim(0, res.t[-1])
    ax_v.set_ylim(-0.05, max(1.6, float(res.v.max()) + 0.1))
    ax_v.set_xlabel("time [s]")
    ax_v.set_ylabel("speed [m/s]")
    ax_v.set_title("longitudinal: speed command vs measured", fontsize=11)
    ax_v.legend(loc="lower center", fontsize=9, facecolor=PANEL, edgecolor=GRID, labelcolor=TXT, ncol=2)
    ph = s["physics_health"]
    m = s["model"]
    rp = s["rover_params"]
    fig.text(0.5, 0.008,
             f"model: {m['total_mass_kg']:.0f} kg, wheelbase {rp['wheelbase']} m, track {rp['track']} m, "
             f"wheel r {rp['wheel_radius']} m ({m['wheel_collision']} contact), c_rr {rp['c_rr']} | physics dt "
             f"{1000 * m['timestep_s']:.1f} ms ({m['integrator']}), control {1000 * m['dt_ctrl_s']:.0f} ms ZOH | "
             f"steer servo kp {m['steer_kp']:.0f} N m/rad, kv {m['steer_kv']:.0f} N m s/rad, slew "
             f"{m['steer_rate_max']} rad/s | drive kv {m['drive_kv']:.0f} N m s/rad, max {m['drive_torque_max']:.0f} "
             f"N m/wheel\nhealth: all 4 wheels in contact {ph['pct_ticks_all_4_wheels_in_contact']:.2f}% of ticks, "
             f"max roll {ph['max_roll_deg']:.1f} deg, max pitch {ph['max_pitch_deg']:.1f} deg (DEM grade up to ~9%), "
             f"max contact penetration {1000 * ph['max_penetration_m']:.2f} mm, chassis-ground contacts "
             f"{ph['deck_ground_contact_ticks']}",
             ha="center", va="bottom", fontsize=9, color="#9ca3af", linespacing=1.5)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=100, facecolor=BG)
    plt.close(fig)
    return out_png


def _jsonable(o):
    """Strict JSON: numpy scalars -> Python, NaN / inf -> None."""
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if math.isfinite(o) else None
    return o


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--terrain", default="dem", choices=["dem", "flat"])
    ap.add_argument("--v-max", type=float, default=1.4)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--pose-noise", type=float, nargs=2, default=(0.0, 0.0), metavar=("SIGMA_XY", "SIGMA_YAW"),
                    help="white noise on the controller pose measurement [m, rad]")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    xy, labels, obstacle, geo = build_reference()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    video = None if args.no_video else out / "rover_mujoco_chase.mp4"
    res = run_closed_loop(xy, v_max=args.v_max, terrain=args.terrain, segment_labels=labels,
                          record_video_path=video, obstacles=[obstacle], pose_noise=tuple(args.pose_noise))
    res.to_csv(out / "rover_mujoco_log.csv")
    png = make_figure(res, geo, obstacle, out / "rover_mujoco_tracking.png")
    wall = time.perf_counter() - t0

    s = res.summary
    b = s["by_label"]
    checks = {
        "completed": bool(s["completed"]),
        "lane_settled_rms_cte": b["lane_settled"]["rms_cte"] < TARGETS["lane_settled_rms_cte_m"],
        "lane_settled_max_cte": b["lane_settled"]["max_cte"] < TARGETS["lane_settled_max_cte_m"],
        "headland_max_cte": b["headland"]["max_cte"] < TARGETS["headland_max_cte_m"],
        "no_flip": s["end_reason"] != "flipped",
        "no_deck_ground_contact": s["physics_health"]["deck_ground_contact_ticks"] == 0,
        "no_obstacle_contact": s["physics_health"]["obstacle_contact_ticks"] == 0,
        "wall_time": wall < TARGETS["wall_time_s"],
    }
    s["validation"] = {"targets": TARGETS, "checks": checks, "all_passed": all(checks.values()),
                       "wall_time_script_s": wall}
    s["reference"] = {k: v for k, v in geo.items()}
    s["reference"]["obstacle"] = obstacle
    s["reference"]["path_points"] = int(len(res.path_xy))
    s["outputs"] = {"csv": str(out / "rover_mujoco_log.csv"), "figure": str(png),
                    "video": str(video) if video else None}
    (out / "rover_mujoco_summary.json").write_text(json.dumps(_jsonable(s), indent=2) + "\n")

    print(f"completed={s['completed']} ({s['end_reason']})  duration={s['duration_s']:.1f}s  "
          f"travelled={s['distance_travelled_m']:.1f}m / path {s['path_length_m']:.1f}m  mean v={s['mean_speed']:.2f}")
    for k in ("lane_settled", "lane", "headland", "detour"):
        if k in b:
            print(f"  {k:13s} rms={100 * b[k]['rms_cte']:5.1f} cm  max={100 * b[k]['max_cte']:5.1f} cm  "
                  f"(front axle rms={100 * b[k]['rms_cte_front']:5.1f} max={100 * b[k]['max_cte_front']:5.1f})")
    print(f"  overall rms={100 * s['rms_cte']:.1f} cm max={100 * s['max_cte']:.1f} cm  "
          f"steer saturated {s['pct_time_steer_saturated']:.1f}%")
    print("  physics:", json.dumps(s["physics_health"]))
    print("  checks:", checks)
    print(f"  wall: sim {s['wall_time_sim_s']:.1f}s, video {s.get('wall_time_video_s', 0):.1f}s, total {wall:.1f}s")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
