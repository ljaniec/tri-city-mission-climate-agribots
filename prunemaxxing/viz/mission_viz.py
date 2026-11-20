"""Mission visuals: map poster, tracking report, live dashboard video, exports."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Callable

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch, Patch, Rectangle  # noqa: E402
from matplotlib.patches import Polygon as MplPolygon  # noqa: E402

from prunemaxxing.planning import ms_effectors as fx  # noqa: E402
from prunemaxxing.planning.mission import LABEL_CODES, LABEL_NAMES, MissionResult  # noqa: E402
from prunemaxxing.planning.swarm import PHASE_NAMES  # noqa: E402
from prunemaxxing.viz.basemap import Basemap, CoveragePainter, build_basemap  # noqa: E402
from prunemaxxing.viz.style import COLORS, DRONE_PATH_COLORS  # noqa: E402

MONO = "DejaVu Sans Mono"
SANS = "DejaVu Sans"
INK = COLORS["ink"]
MUTED = COLORS["muted"]
BG = "#0b100c"
PANEL = "#121a13"
PLANNED = "#f3dc8a"
ACTUAL = "#38d6e6"
BYPASS = "#ff8c2a"
WEED = "#ff4d3d"
OBST = "#ff3df2"
FXC = "#ffe066"  # mothership blower / air-cannon
CHIP = "#e0b97a"
SEG_COLORS = {"lane": "#7bdc6a", "headland": "#f6c445", "transit": "#8fb8ff", "weed_detour": "#ff6b5a",
              "obstacle_detour": "#ff8c2a", "station": "#cccccc"}
MS_LEN, MS_WID = 4.2, 2.0


def _style_axes(ax, title: str | None = None) -> None:
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_color("#2c3a2d")
    ax.tick_params(colors=MUTED, labelsize=8)
    if title:
        ax.set_title(title, color=INK, fontsize=10, loc="left", pad=6, fontweight="bold")


def _ms_polygon(x: float, y: float, yaw: float, scale: float = 1.0) -> np.ndarray:
    L, W = MS_LEN * scale / 2, MS_WID * scale / 2
    pts = np.array([[L, 0.0], [L * 0.62, W], [-L, W], [-L, -W], [L * 0.62, -W]])
    c, s = math.cos(yaw), math.sin(yaw)
    return pts @ np.array([[c, s], [-s, c]]) + [x, y]


def _effector_overlay(res: MissionResult, ms: dict, t: float, rng: np.random.Generator):
    """Top-view geometry of the mothership effectors: (lines, spray/pellet points, telemetry rows)."""
    lines, pts, rows = [], [], []
    for st in res.ms_effector_state(t):
        if st["kind"] == fx.KIND_BLOWER:
            o = fx.mount_world(ms["x"], ms["y"], ms["yaw"], fx.CHUTE_TIP)
            if st["firing"] and st["aim"] is not None:
                a = np.asarray(st["aim"])
                lines.append(np.vstack([o, a]))
                u = rng.uniform(0.15, 1.0, 26)
                stream = o[None, :] + u[:, None] * (a - o)[None, :] + rng.normal(0, 0.25, (26, 2)) * u[:, None]
                pts.append(np.vstack([stream, a[None, :] + rng.normal(0, 0.9, (30, 2))]))
                rows.append(f" blower az {math.degrees(st['rel_az']) % 360:5.1f}° {st['range_m']:4.1f} m THROWING")
            else:
                rows.append(f" blower az {math.degrees(st['rel_az']) % 360:5.1f}°        idle")
        else:
            o = fx.mount_world(ms["x"], ms["y"], ms["yaw"], fx.CANNON_PIVOTS[st["side"]])
            if len(st["pellets"]):
                pts.append(st["pellets"][:, :2])
            if st["firing"] and st["aim"] is not None:
                lines.append(np.vstack([o, st["aim"]]))
            name = "port" if st["side"] == 0 else "stbd"
            state = "FIRING" if st["firing"] else "idle"
            rows.append(f" cannon {name} az {math.degrees(st['rel_az']) % 360:5.1f}° {state}")
    return lines, (np.vstack(pts) if pts else np.zeros((0, 2))), rows


def _fmt_clock(t: float) -> str:
    start_h = 7.0  # mission starts 07:00 day 1
    tot = start_h * 3600 + t
    day = int(tot // 86400) + 1
    hh = int((tot % 86400) // 3600)
    mm = int((tot % 3600) // 60)
    ss = int(tot % 60)
    return f"Day {day} {hh:02d}:{mm:02d}:{ss:02d}"


def _sample_sorties(res: MissionResult, per_drone: int) -> list[int]:
    S = res.sorties
    out: list[int] = []
    for d in range(res.n_drones):
        idx = np.where(S["drone"] == d)[0]
        if len(idx) == 0:
            continue
        pick = np.linspace(0, len(idx) - 1, min(per_drone, len(idx))).astype(int)
        out.extend(int(idx[p]) for p in pick)
    return out


def _draw_static_layers(ax, res: MissionResult, show_hidden: bool = True) -> dict:
    site = res.site
    arts: dict = {}
    for w in site.weeds:
        ax.add_patch(MplPolygon(np.asarray(w.polygon.exterior.coords), closed=True, facecolor=(1, 0.3, 0.24, 0.22),
                                edgecolor=WEED, lw=1.4, hatch="////", zorder=4))
    for inf in res.plan.inflated_weeds:
        ax.add_patch(MplPolygon(np.asarray(inf.exterior.coords), closed=True, fill=False, edgecolor=WEED, lw=0.6,
                                ls=(0, (3, 3)), alpha=0.6, zorder=4))
    arts["obstacles"] = {}
    for ob in site.obstacles:
        p = MplPolygon(np.asarray(ob.polygon.exterior.coords), closed=True, facecolor=(1, 1, 1, 0.0),
                       edgecolor=(1, 1, 1, 0.35 if show_hidden else 0.0), lw=0.8, ls=":", zorder=5)
        ax.add_patch(p)
        arts["obstacles"][ob.oid] = p
    return arts


# ---------------------------------------------------------------------------
# Poster
# ---------------------------------------------------------------------------


def render_map_poster(res: MissionResult, out_path: Path, basemap: Basemap | None = None,
                      sorties_per_drone: int = 40, dpi: int = 200) -> Path:
    bm = basemap or build_basemap(res.site)
    painter = CoveragePainter(bm, res)
    painter.advance(res.t_end)
    fig = plt.figure(figsize=(19.2, 10.8), dpi=dpi, facecolor=BG)
    mode = res.spec.mode.value
    fig.text(0.012, 0.965, "PRUNEMAXXING", color=COLORS["amber"], fontsize=22, fontweight="bold", family=SANS)
    fig.text(0.185, 0.966, f"Mission map — {mode.upper()} mode · {res.kpis['area_treated_ha']:.2f} ha · "
             f"{res.kpis['biomass_applied_t']:.0f} t applied · {res.kpis['mission_time_h']:.0f} h · "
             f"{res.kpis['sorties']:,} drone sorties", color=INK, fontsize=14, family=SANS)
    fig.text(0.012, 0.938, "Parc Agrari del Baix Llobregat (Viladecans) · UTM 31N · DEM 10 m · planned route from the "
             "known map vs executed trajectory with lidar-discovered obstacles", color=MUTED, fontsize=10)

    ax = fig.add_axes([0.012, 0.03, 0.50, 0.89])
    _style_axes(ax)
    e0, e1, n0, n1 = bm.extent
    ax.imshow(bm.rgb, extent=bm.extent, origin="upper", zorder=0, interpolation="antialiased")
    ov = painter.rgba()
    ov[..., 3] *= 0.75
    ax.imshow(ov, extent=bm.extent, origin="upper", zorder=1, interpolation="antialiased")
    _draw_static_layers(ax, res, show_hidden=False)
    for ob in res.site.obstacles:
        ax.add_patch(MplPolygon(np.asarray(ob.polygon.exterior.coords), closed=True, facecolor=OBST, edgecolor="white",
                                lw=0.8, alpha=0.95, zorder=9))

    S = res.sorties
    for k in _sample_sorties(res, sorties_per_drone):
        d = int(S["drone"][k])
        cp, ca = DRONE_PATH_COLORS[d % len(DRONE_PATH_COLORS)]
        tr = res.trajectory(k)
        ax.plot(tr.planned[:, 0], tr.planned[:, 1], color=cp, lw=0.6, ls=(0, (2, 2)), alpha=0.75, zorder=6)
        ax.plot(tr.actual[:, 0], tr.actual[:, 1], color=ca, lw=0.9, alpha=0.95, zorder=7)
        ax.scatter(tr.actual[-1, 0], tr.actual[-1, 1], s=2, color=ca, zorder=7)

    P = res.plan.path
    ax.plot(P[:, 0], P[:, 1], color=PLANNED, lw=1.3, ls=(0, (5, 3)), alpha=0.95, zorder=8)
    L = res.ms_log
    ax.plot(L["x"], L["y"], color=ACTUAL, lw=4.0, alpha=0.25, zorder=8)
    ax.plot(L["x"], L["y"], color=ACTUAL, lw=1.5, zorder=8)
    for det in res.detours:
        ax.plot(det["pts"][:, 0], det["pts"][:, 1], color=BYPASS, lw=2.2, ls=(0, (2, 1.5)), zorder=9)
    st = np.array([s.xy for s in res.plan.stations])
    for v in res.visits:
        if v.get("ms_bursts", 0):
            ax.add_patch(Circle((v["x"], v["y"]), res.cfg.effectors.range_max_m, facecolor=(1.0, 0.88, 0.4, 0.10),
                                edgecolor=FXC, lw=0.6, ls=(0, (1.5, 1.5)), zorder=9))
    ax.scatter(st[:, 0], st[:, 1], marker="s", s=10, color="white", edgecolor="black", lw=0.4, zorder=10)
    ax.scatter(*res.plan.depot_xy, marker="*", s=180, color=COLORS["amber"], edgecolor="black", zorder=11)
    for e in res.events:
        if e["kind"] == "obstacle_detected":
            ax.scatter(*e["xy"], marker="o", s=28, color=BYPASS, edgecolor="white", lw=0.6, zorder=11)
            ax.add_patch(Circle(e["xy"], res.cfg.lidar_range_m if "waterlog" not in e["oid"] else res.cfg.camera_range_m,
                                fill=False, edgecolor=BYPASS, lw=0.7, ls=":", alpha=0.8, zorder=10))
    for p in res.site.parcels:
        c = p.polygon.centroid
        ax.text(p.polygon.bounds[0] + 3, p.polygon.bounds[3] - 3, p.pid.replace("_", " "), color="white",
                fontsize=7.5, va="top", zorder=12, bbox=dict(boxstyle="round,pad=0.2", fc=(0, 0, 0, 0.45), ec="none"))
        _ = c
    _scale_bar(ax, bm, 50.0)
    ax.set_xlim(e0 + 10, e1 - 10)
    ax.set_ylim(n0 + 10, n1 - 10)
    ax.set_xticks([])
    ax.set_yticks([])
    handles = [
        Line2D([], [], color=PLANNED, ls=(0, (5, 3)), lw=1.5, label="Mothership planned (known map)"),
        Line2D([], [], color=ACTUAL, lw=2, label="Mothership actual (10 Hz log)"),
        Line2D([], [], color=BYPASS, ls=(0, (2, 1.5)), lw=2, label="Online bypass (lidar re-plan)"),
        Patch(facecolor=(1, 0.3, 0.24, 0.3), edgecolor=WEED, hatch="////", label="Weed exclusion (mapped)"),
        Patch(facecolor=OBST, edgecolor="white", label="Unknown obstacle (not in satellite map)"),
        Line2D([], [], color="#c4b5fd", ls=(0, (2, 2)), label="Drone planned sortie"),
        Line2D([], [], color="#a78bfa", lw=1.2, label="Drone actual (wind, controller lag)"),
        Line2D([], [], marker="s", color="white", ls="", markeredgecolor="k", label="Drone service station"),
        Patch(facecolor=(1.0, 0.88, 0.4, 0.2), edgecolor=FXC, ls=(0, (1.5, 1.5)),
              label=("Air-cannon" if mode == "cartridge" else "Chip-blower") + " reach from the mothership (40 ft)"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=7.5, facecolor=(0.04, 0.06, 0.04, 0.85),
              edgecolor="#2c3a2d", labelcolor=INK, framealpha=0.85)

    # zoom insets on detours
    zooms = [d for d in res.detours][:3]
    for i, det in enumerate(zooms):
        axz = fig.add_axes([0.525 + i * 0.158, 0.50, 0.150, 0.40])
        _detour_inset(axz, res, bm, det, painter)
    # station close-up with sorties
    axs = fig.add_axes([0.525, 0.03, 0.235, 0.43])
    _station_inset(axs, res, bm, painter)
    axk = fig.add_axes([0.770, 0.03, 0.222, 0.43])
    _kpi_card(axk, res)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, facecolor=BG)
    plt.close(fig)
    return out_path


def _scale_bar(ax, bm: Basemap, length: float) -> None:
    e0, e1, n0, n1 = bm.extent
    x0, y0 = e0 + 18, n0 + 16
    ax.add_patch(Rectangle((x0, y0), length, 2.2, color="white", zorder=12))
    ax.add_patch(Rectangle((x0 + length / 2, y0), length / 2, 2.2, color="black", zorder=12))
    ax.text(x0 + length / 2, y0 + 4, f"{length:.0f} m", color="white", ha="center", fontsize=8, zorder=12)
    ax.annotate("N", xy=(x0 + length + 18, y0 + 16), xytext=(x0 + length + 18, y0 + 2), color="white",
                ha="center", fontsize=9, fontweight="bold", zorder=12,
                arrowprops=dict(arrowstyle="-|>", color="white", lw=1.4))


def _detour_inset(ax, res: MissionResult, bm: Basemap, det: dict, painter: CoveragePainter) -> None:
    ob = next(o for o in res.site.obstacles if o.oid == det["oid"])
    c = ob.polygon.centroid
    r = 26.0
    _style_axes(ax)
    ax.imshow(bm.rgb, extent=bm.extent, origin="upper", zorder=0, interpolation="antialiased")
    ax.add_patch(MplPolygon(np.asarray(ob.polygon.exterior.coords), closed=True, facecolor=OBST, edgecolor="white",
                            lw=1.0, zorder=5))
    P = res.plan.path
    ax.plot(P[:, 0], P[:, 1], color=PLANNED, lw=1.6, ls=(0, (5, 3)), zorder=6)
    ax.plot(det["pts"][:, 0], det["pts"][:, 1], color=BYPASS, lw=2.0, ls=(0, (2, 1.5)), zorder=7)
    L = res.ms_log
    ax.plot(L["x"], L["y"], color=ACTUAL, lw=2.0, zorder=8)
    ev = next(e for e in res.events if e["kind"] == "obstacle_detected" and e["oid"] == det["oid"])
    ax.scatter(*ev["xy"], s=40, color=BYPASS, edgecolor="white", zorder=9)
    ax.annotate(f"lidar contact\n{ev['range_m']:.1f} m", xy=ev["xy"], xytext=(ev["xy"][0] + 6, ev["xy"][1] - 8),
                color="white", fontsize=7, zorder=10, arrowprops=dict(arrowstyle="-", color="white", lw=0.6))
    ax.set_xlim(c.x - r, c.x + r)
    ax.set_ylim(c.y - r * 1.3, c.y + r * 1.3)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{ob.kind.upper()} · {ev.get('action', '')}", color=INK, fontsize=9, loc="left")
    ax.text(0.02, 0.02, f"discovered T+{ev['t'] / 3600:.1f} h\nbypass {det['length_m']:.0f} m",
            transform=ax.transAxes, color="white", fontsize=7,
            bbox=dict(boxstyle="round,pad=0.3", fc=(0, 0, 0, 0.55), ec="none"))


def _station_inset(ax, res: MissionResult, bm: Basemap, painter: CoveragePainter) -> None:
    S = res.sorties
    sid = int(np.bincount(S["station"]).argmax()) if len(S["station"]) else 0
    v = next(v for v in res.visits if v["sid"] == sid)
    cx, cy = v["x"], v["y"]
    r = 34.0
    _style_axes(ax)
    ax.imshow(bm.rgb, extent=bm.extent, origin="upper", zorder=0, interpolation="antialiased")
    idx = np.where(S["station"] == sid)[0]
    pick = idx[np.linspace(0, len(idx) - 1, min(36, len(idx))).astype(int)]
    for k in pick:
        d = int(S["drone"][k])
        cp, ca = DRONE_PATH_COLORS[d % len(DRONE_PATH_COLORS)]
        tr = res.trajectory(int(k))
        ax.plot(tr.planned[:, 0], tr.planned[:, 1], color=cp, lw=0.8, ls=(0, (2, 2)), zorder=5)
        ax.plot(tr.actual[:, 0], tr.actual[:, 1], color=ca, lw=1.2, zorder=6)
        if len(tr.drops):
            ax.scatter(tr.drops[:, 0], tr.drops[:, 1], s=3, color="#e8d9a0", zorder=7)
    for c in v["cells"]:
        poly = res.plan.cells[c].polygon
        if poly is not None and poly.geom_type == "Polygon":
            ax.add_patch(MplPolygon(np.asarray(poly.exterior.coords), closed=True, fill=False, edgecolor="white",
                                    lw=0.4, alpha=0.5, zorder=4))
    ax.add_patch(MplPolygon(_ms_polygon(cx, cy, v["yaw"], 1.0), closed=True, facecolor=COLORS["amber"],
                            edgecolor="black", lw=0.8, zorder=8))
    M = res.ms_shots
    mine = np.flatnonzero(M["station"] == sid) if len(M.get("kg", [])) else np.zeros(0, dtype=int)
    ax.add_patch(Circle((cx, cy), res.cfg.effectors.range_max_m, fill=False, edgecolor=FXC, lw=1.4,
                        ls=(0, (2, 2)), zorder=9))
    for k in mine[np.linspace(0, len(mine) - 1, min(24, len(mine))).astype(int)] if len(mine) else []:
        if M["kind"][k] == fx.KIND_CANNON:
            vv = res.ms_volley(int(k))
            ax.scatter(vv["land"][:, 0], vv["land"][:, 1], s=6, color=FXC, edgecolor="black", lw=0.2, zorder=9)
            o = vv["muzzle"][0]
        else:
            o = fx.mount_world(cx, cy, v["yaw"], fx.CHUTE_TIP)
            pat = res.patterns[int(M["cell"][k])][int(M["pattern"][k])]
            ax.plot(pat[:, 0], pat[:, 1], color=FXC, lw=1.1, zorder=9)
        ax.plot([o[0], M["aim_x"][k]], [o[1], M["aim_y"][k]], color=FXC, lw=0.8, alpha=0.8, zorder=9)
    ax.set_xlim(cx - r, cx + r)
    ax.set_ylim(cy - r * 0.9, cy + r * 0.9)
    ax.set_xticks([])
    ax.set_yticks([])
    dur = (v["t_depart"] - v["t_arrive"]) / 3600
    eff = "cannon volleys" if res.spec.mode.value == "cartridge" else "blower bursts"
    ax.set_title(f"Station {sid}: {v['sorties']} sorties + {v.get('ms_bursts', 0)} {eff} · {dur:.1f} h · "
                 f"ring = 40 ft reach", color=INK, fontsize=9, loc="left")


def _kpi_card(ax, res: MissionResult) -> None:
    k = res.kpis
    ax.set_facecolor(PANEL)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#2c3a2d")
    lines = [
        ("AGRONOMY / CLIMATE", None),
        ("Area treated", f"{k['area_treated_ha']:.2f} ha"),
        ("Dose", f"{k['dose_kg_per_m2']:.1f} kg/m²  (CV {100 * k['dose_cv']:.1f} %)"),
        ("Biomass applied", f"{k['biomass_applied_t']:,.0f} t"),
        ("Cells at target", f"{k['cells_complete_pct']:.1f} %"),
        ("CO₂e / N retained", f"{k['co2e_t']:.0f} t / {k['n_retained_kg']:,.0f} kg"),
        ("LOGISTICS / ROBOTICS", None),
        ("Mission time", f"{k['mission_time_h']:.1f} h  ({k['mission_days_10h_shifts']:.0f} × 10 h shifts)"),
        ("Mothership", f"{k['ms_distance_km']:.2f} km · {k['stations']} stations"),
        ("Lane x-track RMS / max", f"{100 * k.get('cte_rms_lane_m', 0):.1f} / {100 * k.get('cte_max_lane_m', 0):.1f} cm"),
        ("Within furrow ±20 cm", f"{k.get('lane_within_furrow_tol_pct', 0):.1f} %"),
        ("Bed compaction", f"{k['bed_compaction_m2']:.0f} m² ({k['bed_compaction_pct_of_area']:.2f} %)"),
        ("Unknown obstacles", f"{len(k['obstacles_detected'])} found · {k['detour_m']:.0f} m bypass"),
        ("Drone sorties", f"{k['sorties']:,} · {k['drone_flight_km']:,.0f} km · {k['battery_swaps']} swaps"),
        ("MS " + ("air cannons" if "cannon" in k.get("ms_effector", "") else "chip blower") + " (≤12.2 m)",
         f"{k.get('ms_effector_t', 0):,.0f} t ({k.get('ms_effector_share_pct', 0):.0f} %) · "
         + (f"{k.get('ms_effector_units', 0):,} shots" if "cannon" in k.get("ms_effector", "")
            else f"{k.get('ms_effector_bursts', 0):,} bursts")),
        ("Fleet throughput", f"{k['swarm_throughput_t_per_h']:.1f} t/h · drone util {k['drone_utilisation_pct']:.0f} %"),
        ("Tender trucks", f"{k['tender_trips']} trips · {k['tender_km']:.0f} km"),
        ("ECONOMICS (€/1000 m²)", None),
    ]
    if "regensol_baseline_eur_per_1000m2" in k:
        lines += [
            ("REGENSOL manual baseline", f"{k['regensol_baseline_eur_per_1000m2']:,.2f}"),
            ("PruneMaxxing (robot ops)", f"{k['cost_eur_per_1000m2']:,.2f}  (ops {k['robot_ops_eur_per_1000m2']:.0f})"),
            ("Saving / overhead recovered", f"{k['saving_vs_regensol_eur_per_1000m2']:,.0f} € · {k['overhead_recovered_pct']:.0f} %"),
        ]
    else:
        lines += [
            ("Cartridges / seeds", f"{k['cartridges_placed']:,} / {k['seeds_placed']:,}"),
            ("Robot ops", f"{k['robot_ops_eur_per_1000m2']:,.0f}"),
            ("Cartridge material", f"{k['cartridge_material_eur_per_1000m2']:,.0f}"),
        ]
    y = 0.97
    for label, val in lines:
        if val is None:
            ax.text(0.04, y, label, color=COLORS["amber"], fontsize=9, fontweight="bold", transform=ax.transAxes)
            y -= 0.05
            continue
        ax.text(0.04, y, label, color=MUTED, fontsize=8, transform=ax.transAxes)
        ax.text(0.97, y, val, color=INK, fontsize=8, ha="right", transform=ax.transAxes, family=MONO)
        y -= 0.0435


# ---------------------------------------------------------------------------
# Tracking report
# ---------------------------------------------------------------------------


def render_tracking_report(res: MissionResult, out_path: Path, physics_summary: dict | None = None) -> Path:
    L = res.ms_log
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor=BG)
    fig.text(0.012, 0.962, "Mothership path tracking — controller commands vs vehicle response",
             color=INK, fontsize=18, fontweight="bold")
    fig.text(0.012, 0.935, "Pure pursuit (adaptive look-ahead) → steering actuator (0.6 rad/s, τ 0.18 s) + drive "
             "(0.4 m/s², τ 0.45 s) → kinematic bicycle with slip noise. Nothing snaps the state to the path.",
             color=MUTED, fontsize=10)
    moving = L["v"] > 0.02
    odo = L["odo"]
    ax1 = fig.add_axes([0.04, 0.60, 0.62, 0.30])
    _style_axes(ax1, "Cross-track error along the executed route")
    for name, code in LABEL_CODES.items():
        if name == "station":
            continue
        m = (L["mode"] == code) & moving
        if m.any():
            ax1.scatter(odo[m], 100 * L["cte"][m], s=1.5, color=SEG_COLORS[name], label=name.replace("_", " "))
    ax1.axhspan(-20, 20, color="#7bdc6a", alpha=0.08)
    ax1.axhline(20, color="#7bdc6a", lw=0.6, ls="--")
    ax1.axhline(-20, color="#7bdc6a", lw=0.6, ls="--")
    ax1.text(odo.max() * 0.995, 21, "furrow tolerance ±20 cm", color="#7bdc6a", fontsize=8, ha="right")
    ax1.set_xlabel("distance travelled [m]", color=MUTED)
    ax1.set_ylabel("cross-track [cm]", color=MUTED)
    ax1.set_ylim(-30, 30)
    ax1.legend(loc="lower left", fontsize=8, markerscale=6, ncol=5, facecolor=PANEL, labelcolor=INK, edgecolor="#2c3a2d")

    # time window around the first obstacle bypass (or a headland)
    # driving clock: collapse station dwells so the manoeuvre is continuous
    st_code = LABEL_CODES["station"]
    dts = np.diff(L["t"], prepend=L["t"][0])
    parked = (L["mode"] == st_code) & (np.roll(L["mode"], 1) == st_code)
    tdrive = np.cumsum(np.where(parked, 0.0, dts))
    if res.detours:
        i_c = int(np.searchsorted(L["t"], res.detours[0]["t"]))
    else:
        i_c = int(np.argmax(np.abs(L["steer_cmd"])))
    t_c = tdrive[i_c]
    win = (tdrive > t_c - 8) & (tdrive < t_c + 55) & ~parked
    tt = tdrive[win] - t_c
    ax2 = fig.add_axes([0.04, 0.33, 0.30, 0.22])
    _style_axes(ax2, "Steering: command vs actual (around first bypass)")
    ax2.plot(tt, np.degrees(L["steer_cmd"][win]), color=PLANNED, lw=1.2, label="steer_cmd")
    ax2.plot(tt, np.degrees(L["steer"][win]), color=ACTUAL, lw=1.6, label="steer (actuator)")
    ax2.axhline(np.degrees(res.cfg.vehicle.max_steer_rad), color=WEED, lw=0.6, ls=":")
    ax2.axhline(-np.degrees(res.cfg.vehicle.max_steer_rad), color=WEED, lw=0.6, ls=":")
    ax2.set_xlabel("driving time since lidar contact [s]", color=MUTED)
    ax2.set_ylabel("deg", color=MUTED)
    ax2.legend(fontsize=8, facecolor=PANEL, labelcolor=INK, edgecolor="#2c3a2d")
    ax3 = fig.add_axes([0.37, 0.33, 0.29, 0.22])
    _style_axes(ax3, "Speed: command vs actual")
    ax3.plot(tt, L["v_cmd"][win], color=PLANNED, lw=1.2, label="v_cmd")
    ax3.plot(tt, L["v"][win], color=ACTUAL, lw=1.6, label="v")
    stop = np.flatnonzero(L["mode"][win] == st_code)
    if stop.size:
        ax3.axvline(tt[stop[0]], color=MUTED, lw=0.8, ls=":")
        ax3.text(tt[stop[0]] + 0.8, 0.12, "drone station stop", color=MUTED, fontsize=8)
    for ax in (ax2, ax3):
        ax.axvline(0.0, color=OBST, lw=0.8, ls=":")
    ax3.text(0.8, 1.02, "obstacle seen → bypass speed", color=OBST, fontsize=8)
    ax3.set_xlabel("driving time since lidar contact [s]", color=MUTED)
    ax3.set_ylabel("m/s", color=MUTED)
    ax3.legend(fontsize=8, facecolor=PANEL, labelcolor=INK, edgecolor="#2c3a2d")

    # xy zoom of the bypass
    ax4 = fig.add_axes([0.69, 0.33, 0.29, 0.57])
    _style_axes(ax4, "Planned vs re-planned vs actual (first bypass)")
    P = res.plan.path
    if res.detours:
        det = res.detours[0]
        ob = next(o for o in res.site.obstacles if o.oid == det["oid"])
        c = ob.polygon.centroid
        ax4.add_patch(MplPolygon(np.asarray(ob.polygon.exterior.coords), closed=True, facecolor=OBST, edgecolor="w"))
        ax4.add_patch(MplPolygon(np.asarray(ob.polygon.buffer(res.cfg.obstacle_clearance_m).exterior.coords),
                                 closed=True, fill=False, edgecolor=OBST, ls=":", lw=0.8))
        ax4.plot(det["pts"][:, 0], det["pts"][:, 1], color=BYPASS, lw=2, ls=(0, (2, 1.5)), label="re-planned bypass")
        cx, cy, r = c.x, c.y, 24
    else:
        cx, cy, r = L["x"][0], L["y"][0], 30
    ax4.plot(P[:, 0], P[:, 1], color=PLANNED, lw=1.5, ls=(0, (5, 3)), label="global plan")
    ax4.plot(L["x"], L["y"], color=ACTUAL, lw=1.8, label="actual (rear-axle ctrl)")
    ax4.set_xlim(cx - r, cx + r)
    ax4.set_ylim(cy - r * 1.6, cy + r * 1.6)
    ax4.set_aspect("equal")
    ax4.legend(fontsize=8, facecolor=PANEL, labelcolor=INK, edgecolor="#2c3a2d", loc="lower right")
    ax4.set_xticks([])
    ax4.set_yticks([])

    ax5 = fig.add_axes([0.04, 0.05, 0.30, 0.22])
    _style_axes(ax5, "|cross-track| on bed lanes (after 4 m settling)")
    lane = (L["mode"] == LABEL_CODES["lane"]) & moving
    ax5.hist(100 * np.abs(L["cte"][lane]), bins=60, color="#7bdc6a", alpha=0.85)
    ax5.axvline(20, color=WEED, lw=0.8, ls="--")
    ax5.set_xlabel("cm", color=MUTED)
    ax5.set_yscale("log")

    ax6 = fig.add_axes([0.37, 0.05, 0.61, 0.22])
    ax6.set_facecolor(PANEL)
    ax6.set_xticks([])
    ax6.set_yticks([])
    for sp in ax6.spines.values():
        sp.set_color("#2c3a2d")
    k = res.kpis
    rows = [("segment", "RMS [cm]", "max [cm]")]
    for name in ("lane", "headland", "transit", "weed_detour", "obstacle_detour"):
        if f"cte_rms_{name}_m" in k:
            rows.append((name.replace("_", " "), f"{100 * k[f'cte_rms_{name}_m']:.1f}", f"{100 * k[f'cte_max_{name}_m']:.1f}"))
    for i, row in enumerate(rows):
        for j, cell in enumerate(row):
            ax6.text(0.02 + j * 0.13, 0.88 - i * 0.13, cell, transform=ax6.transAxes, fontsize=10,
                     color=COLORS["amber"] if i == 0 else INK, family=MONO)
    notes = [f"Within furrow tolerance on lanes: {k.get('lane_within_furrow_tol_pct', 0):.1f} %",
             f"Steering saturated: {k.get('steer_saturated_pct', 0):.2f} % of driving time",
             f"Executed {k['ms_distance_km']:.2f} km vs planned {k['planned_distance_km']:.2f} km "
             f"(+{1000 * (k['ms_distance_km'] - k['planned_distance_km']):.0f} m bypasses)"]
    if physics_summary:
        ps = physics_summary
        rl = ps.get("rms_cte_by_label", {})
        ph = ps.get("physics_health", {})
        notes.append(f"MuJoCo multibody check · {ps.get('model', {}).get('total_mass_kg', 4000) / 1000:.0f} t Ackermann "
                     f"rover on the DEM heightfield, same controller, {ps.get('path_length_m', 0):.0f} m test route:")
        notes.append(f"   settled lane RMS {100 * rl.get('lane_settled', 0):.1f} cm · headland "
                     f"{100 * rl.get('headland', 0):.1f} cm · detour {100 * rl.get('detour', 0):.1f} cm · "
                     f"max {100 * ps.get('max_cte', 0):.1f} cm")
        if ph:
            notes.append(f"   4 wheels in contact {ph.get('pct_ticks_all_4_wheels_in_contact', 0):.2f} % · max pitch "
                         f"{ph.get('max_pitch_deg', 0):.1f}° · roll {ph.get('max_roll_deg', 0):.1f}° · "
                         f"{'completed' if ps.get('completed') else 'NOT completed'}")
    for i, ntext in enumerate(notes):
        ax6.text(0.43, 0.88 - i * 0.14, ntext, transform=ax6.transAxes, fontsize=9, color=INK)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100, facecolor=BG)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Timeline (activity-weighted time warp)
# ---------------------------------------------------------------------------


def mission_timeline(res: MissionResult, target_s: float = 72.0, fps: int = 24) -> tuple[np.ndarray, np.ndarray]:
    """Frame sim-times + per-frame speed factor (sim s per video s)."""
    t_end = res.t_end
    bps = {0.0, t_end}
    dwell = [(v["t_arrive"], v["t_depart"]) for v in res.visits]
    for a, b in dwell:
        bps.update((a, b))
    boosts: list[tuple[float, float, float]] = [(0.0, 30.0, 1 / 8.0)]
    if dwell:
        boosts.append((dwell[0][0], dwell[0][0] + 90.0, 1 / 8.0))
    for e in res.events:
        if e["kind"] == "obstacle_detected":
            boosts.append((e["t"] - 6.0, e["t"] + 34.0, 1 / 10.0))
    for a, b, _ in boosts:
        bps.update((max(0.0, a), min(t_end, b)))
    edges = np.array(sorted(bps))

    def base_rate(tm: float, k_drive: float, k_dwell: float) -> float:
        for a, b in dwell:
            if a <= tm < b:
                return k_dwell / max(b - a, 1.0)
        return k_drive

    def rates(k_drive, k_dwell):
        out = []
        for a, b in zip(edges[:-1], edges[1:]):
            m = 0.5 * (a + b)
            r = base_rate(m, k_drive, k_dwell)
            for ba, bb, br in boosts:
                if ba <= m < bb:
                    r = max(r, br)
            out.append(r)
        return np.array(out)

    k_drive, k_dwell = 1 / 75.0, 0.32
    for _ in range(30):
        r = rates(k_drive, k_dwell)
        total = float(np.sum(r * np.diff(edges)))
        if abs(total - target_s) < 0.5:
            break
        boost_part = float(sum(min(b, t_end) - max(a, 0) for a, b, _ in boosts) / 6.0)
        f = max(0.2, (target_s - min(boost_part, target_s * 0.5)) / max(total - min(boost_part, total * 0.5), 1e-6))
        k_drive *= f
        k_dwell *= f
    r = rates(k_drive, k_dwell)
    V = np.concatenate([[0.0], np.cumsum(r * np.diff(edges))])
    n = int(V[-1] * fps)
    vt = np.arange(n) / fps
    times = np.interp(vt, V, edges)
    seg = np.clip(np.searchsorted(edges, times, side="right") - 1, 0, len(r) - 1)
    speed = 1.0 / r[seg]
    return times, speed


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------


class Dashboard:
    W, H = 1920, 1080
    CARD_STRIP = (0.008, 0.092)  # figure-y band reserved for the KPI cards
    CAM_SIZES = {"main": (1280, 608), "second": (640, 320)}  # match the camera axes aspect

    def __init__(self, res: MissionResult, basemap: Basemap | None = None,
                 cam_provider: Callable[[float], dict[str, np.ndarray]] | None = None,
                 cam_titles: tuple[str, str] = ("MuJoCo · mothership chase cam", "MuJoCo · drone D0 belly cam"),
                 kpi_cards: bool = True):
        from prunemaxxing.viz.kpi_overlay import KpiOverlay

        self.res = res
        self.bm = basemap or build_basemap(res.site)
        self.painter = CoveragePainter(self.bm, res)
        self.cams = cam_provider
        self.kpi = KpiOverlay(res) if kpi_cards else None
        self.fig = plt.figure(figsize=(self.W / 100, self.H / 100), dpi=100, facecolor=BG)
        self._build(cam_titles)
        self._last_t = -1.0
        S = res.sorties
        order = np.argsort(res.deposits["t"])
        self._dep_t = res.deposits["t"][order]
        self._dep_cum = np.cumsum(res.deposits["kg"][order]) / 1000.0
        self._target_t = float(res.cell_target_kg.sum() / 1000.0)
        M = res.ms_shots
        if len(M.get("kg", [])):
            o = np.argsort(M["t1"])
            self._ms_t, self._ms_cum = M["t1"][o], np.cumsum(M["kg"][o]) / 1000.0
        else:
            self._ms_t, self._ms_cum = np.zeros(0), np.zeros(0)
        self._sortie_t = np.sort(S["t_launch"])
        self._swap_t = np.sort([e["t"] for e in res.events if e["kind"] == "battery_swap"])
        self._det = sorted([e for e in res.events if e["kind"] == "obstacle_detected"], key=lambda e: e["t"])

    # ------------------------------------------------------------- layout
    def _rect(self, x: float, y: float, w: float, h: float) -> list[float]:
        """Squeeze the original 0.015..0.92 layout band above the KPI card strip."""
        if self.kpi is None:
            return [x, y, w, h]
        y0 = self.CARD_STRIP[1] + 0.008
        k = (0.92 - y0) / (0.92 - 0.015)
        return [x, y0 + (y - 0.015) * k, w, h * k]

    def _build(self, cam_titles) -> None:
        f = self.fig
        res = self.res
        bm = self.bm
        R = self._rect
        f.text(0.010, 0.962, "PRUNEMAXXING", color=COLORS["amber"], fontsize=21, fontweight="bold")
        f.text(0.150, 0.963, f"autonomous prunings-to-mulch logistics · mothership UGV + {res.n_drones} wireless "
               f"heavy-lift drones · {res.spec.mode.value.upper()} mode", color=INK, fontsize=13)
        self.txt_clock = f.text(0.990, 0.962, "", color=INK, fontsize=15, ha="right", family=MONO, fontweight="bold")
        self.txt_speed = f.text(0.990, 0.938, "", color=COLORS["amber"], fontsize=10, ha="right", family=MONO)

        ax = f.add_axes(R(0.008, 0.015, 0.525, 0.905))
        self.ax_map = ax
        _style_axes(ax)
        ax.imshow(bm.rgb, extent=bm.extent, origin="upper", zorder=0, interpolation="antialiased")
        self.im_cov = ax.imshow(np.zeros((2, 2, 4)), extent=bm.extent, origin="upper", zorder=1,
                                interpolation="antialiased")
        self.static = _draw_static_layers(ax, res, show_hidden=True)
        P = res.plan.path
        ax.plot(P[:, 0], P[:, 1], color=PLANNED, lw=1.1, ls=(0, (5, 3)), alpha=0.85, zorder=6)
        self.ln_bypass = [ax.plot([], [], color=BYPASS, lw=2.0, ls=(0, (2, 1.5)), zorder=7)[0] for _ in res.detours]
        self.ln_glow, = ax.plot([], [], color=ACTUAL, lw=5.0, alpha=0.22, zorder=8)
        self.ln_act, = ax.plot([], [], color=ACTUAL, lw=1.8, zorder=8)
        st = np.array([s.xy for s in res.plan.stations])
        self.sc_st = ax.scatter(st[:, 0], st[:, 1], marker="s", s=14, c=["#777777"] * len(st), edgecolor="black",
                                lw=0.4, zorder=9)
        self.ring = Circle((0, 0), res.cfg.lidar_range_m, fill=False, edgecolor=ACTUAL, lw=0.8, ls=(0, (3, 3)),
                           alpha=0.6, zorder=9)
        ax.add_patch(self.ring)
        reach = res.cfg.effectors.range_max_m
        self.fx_ring = Circle((0, 0), reach, fill=False, edgecolor=FXC, lw=1.0, ls=(0, (1.5, 1.5)), zorder=9)
        ax.add_patch(self.fx_ring)
        self.ln_fx = [ax.plot([], [], color=FXC, lw=1.3, zorder=12)[0] for _ in range(2)]
        self.sc_fx = ax.scatter([], [], s=3, color=CHIP, zorder=12)
        self._fx_rng = np.random.default_rng(3)
        self.ms_patch = MplPolygon(_ms_polygon(0, 0, 0, 2.2), closed=True, facecolor=COLORS["amber"],
                                   edgecolor="black", lw=1.0, zorder=12)
        ax.add_patch(self.ms_patch)
        self.dr_planned, self.dr_actual, self.dr_prev = [], [], []
        for d in range(res.n_drones):
            cp, ca = DRONE_PATH_COLORS[d % len(DRONE_PATH_COLORS)]
            self.dr_prev.append(ax.plot([], [], color=ca, lw=0.8, alpha=0.45, zorder=10)[0])
            self.dr_planned.append(ax.plot([], [], color=cp, lw=0.9, ls=(0, (2, 2)), zorder=10)[0])
            self.dr_actual.append(ax.plot([], [], color=ca, lw=1.6, zorder=11)[0])
        cols = [DRONE_PATH_COLORS[d % 4][1] for d in range(res.n_drones)]
        self.sc_dr = ax.scatter(np.zeros(res.n_drones), np.zeros(res.n_drones), s=42, c=cols, edgecolor="white",
                                lw=1.0, zorder=13)
        self.txt_alert = ax.text(0.5, 0.955, "", transform=ax.transAxes, ha="center", va="top", fontsize=12,
                                 color="white", fontweight="bold", zorder=20,
                                 bbox=dict(boxstyle="round,pad=0.4", fc=(0.9, 0.35, 0.05, 0.85), ec="none"))
        self.txt_alert.set_visible(False)
        self.txt_phase = ax.text(0.012, 0.985, "", transform=ax.transAxes, ha="left", va="top", fontsize=10.5,
                                 color="white", family=MONO, zorder=20,
                                 bbox=dict(boxstyle="round,pad=0.4", fc=(0, 0, 0, 0.6), ec="none"))
        _scale_bar(ax, bm, 50.0)
        e0, e1, n0, n1 = bm.extent
        ax.set_xlim(e0 + 12, e1 - 12)
        ax.set_ylim(n0 + 12, n1 - 12)
        ax.set_xticks([])
        ax.set_yticks([])
        handles = [
            Line2D([], [], color=PLANNED, ls=(0, (5, 3)), lw=1.5, label="planned route"),
            Line2D([], [], color=ACTUAL, lw=2, label="actual route"),
            Line2D([], [], color=BYPASS, ls=(0, (2, 1.5)), lw=2, label="lidar re-plan"),
            Patch(facecolor=(1, 0.3, 0.24, 0.3), edgecolor=WEED, hatch="////", label="weed exclusion"),
            Patch(facecolor="none", edgecolor="white", ls=":", label="hidden obstacle"),
            Line2D([], [], color="#c4b5fd", ls=(0, (2, 2)), label="drone planned"),
            Line2D([], [], color="#a78bfa", lw=1.4, label="drone actual"),
            Line2D([], [], color=FXC, ls=(0, (1.5, 1.5)), lw=1.2,
                   label="MS " + ("air cannons" if res.spec.mode.value == "cartridge" else "chip blower")
                   + f" reach {res.cfg.effectors.range_max_m:.1f} m (40 ft)"),
        ]
        ax.legend(handles=handles, loc="lower right", fontsize=7.5, facecolor=(0.03, 0.05, 0.03, 0.8),
                  edgecolor="#2c3a2d", labelcolor=INK, ncol=1)

        # right column
        x0 = 0.545
        wcol = 0.447
        self.ax_cam = f.add_axes(R(x0, 0.505, wcol, 0.415))
        _style_axes(self.ax_cam, cam_titles[0])
        self.im_cam = self.ax_cam.imshow(np.zeros((9, 16, 3), dtype=np.uint8), aspect="auto")
        self.ax_cam.set_xticks([])
        self.ax_cam.set_yticks([])
        self.ax_cam2 = f.add_axes(R(x0, 0.255, wcol * 0.49, 0.215))
        _style_axes(self.ax_cam2, cam_titles[1])
        self.im_cam2 = self.ax_cam2.imshow(np.zeros((9, 16, 3), dtype=np.uint8), aspect="auto")
        self.ax_cam2.set_xticks([])
        self.ax_cam2.set_yticks([])
        if self.cams is None:
            for a in (self.ax_cam, self.ax_cam2):
                a.text(0.5, 0.5, "MuJoCo cameras disabled (--skip-mujoco)", transform=a.transAxes, ha="center",
                       color=MUTED)

        # minimap zoom
        self.ax_mini = f.add_axes(R(x0 + wcol * 0.51, 0.255, wcol * 0.49, 0.215))
        _style_axes(self.ax_mini, "Live minimap · 70 m around the mothership")
        a = self.ax_mini
        a.imshow(bm.rgb, extent=bm.extent, origin="upper", zorder=0, interpolation="antialiased")
        self.im_cov2 = a.imshow(np.zeros((2, 2, 4)), extent=bm.extent, origin="upper", zorder=1)
        _draw_static_layers(a, res, show_hidden=True)
        a.plot(P[:, 0], P[:, 1], color=PLANNED, lw=1.0, ls=(0, (5, 3)), zorder=6)
        self.ln_act2, = a.plot([], [], color=ACTUAL, lw=2.0, zorder=8)
        self.ms_patch2 = MplPolygon(_ms_polygon(0, 0, 0, 1.0), closed=True, facecolor=COLORS["amber"],
                                    edgecolor="black", lw=0.8, zorder=12)
        a.add_patch(self.ms_patch2)
        self.dr_planned2, self.dr_actual2 = [], []
        for d in range(res.n_drones):
            cp, ca = DRONE_PATH_COLORS[d % 4]
            self.dr_planned2.append(a.plot([], [], color=cp, lw=0.9, ls=(0, (2, 2)), zorder=10)[0])
            self.dr_actual2.append(a.plot([], [], color=ca, lw=1.6, zorder=11)[0])
        self.sc_dr2 = a.scatter(np.zeros(res.n_drones), np.zeros(res.n_drones), s=30, c=cols, edgecolor="white",
                                lw=0.8, zorder=13)
        self.sc_drops = a.scatter([], [], s=4, color="#f2e2a0", zorder=12)
        self.fx_ring2 = Circle((0, 0), reach, fill=False, edgecolor=FXC, lw=1.0, ls=(0, (2, 2)), zorder=9)
        a.add_patch(self.fx_ring2)
        self.ln_fx2 = [a.plot([], [], color=FXC, lw=1.6, alpha=0.9, zorder=12)[0] for _ in range(2)]
        self.sc_fx2 = a.scatter([], [], s=6, color=CHIP, edgecolor="#5a4020", lw=0.2, zorder=13)
        a.set_xticks([])
        a.set_yticks([])

        # chart
        self.ax_ch = f.add_axes(R(x0 + 0.03, 0.06, wcol * 0.42, 0.15) if self.kpi is not None
                                else [x0 + 0.03, 0.04, wcol * 0.42, 0.17])
        _style_axes(self.ax_ch, "Biomass applied & coverage")
        self.ax_ch.set_xlim(0, res.t_end / 3600.0)
        tgt = float(res.cell_target_kg.sum() / 1000.0)
        self.ax_ch.set_ylim(0, tgt * 1.05)
        self.ax_ch.axhline(tgt, color=MUTED, lw=0.6, ls="--")
        self.ln_bio, = self.ax_ch.plot([], [], color=COLORS["amber"], lw=2, label="total")
        self.ln_ms, = self.ax_ch.plot([], [], color=FXC, lw=1.2, ls=(0, (3, 1.5)), label="by mothership effector")
        self.ax_ch.legend(loc="upper left", fontsize=7, facecolor=PANEL, labelcolor=INK, edgecolor="#2c3a2d")
        self.fill_bio = None
        self.ax_ch.set_xlabel("mission time [h]", color=MUTED, fontsize=8)
        self.ax_ch.set_ylabel("t", color=MUTED, fontsize=8)
        self.ax_ch2 = self.ax_ch.twinx()
        self.ax_ch2.set_ylim(0, 105)
        self.ax_ch2.tick_params(colors=MUTED, labelsize=8)
        self.ln_cov, = self.ax_ch2.plot([], [], color="#7bdc6a", lw=1.4)
        self.ax_ch2.set_ylabel("cells at dose [%]", color=MUTED, fontsize=8)
        self.vl = self.ax_ch.axvline(0, color="white", lw=0.6, alpha=0.6)

        # telemetry
        self.ax_tel = f.add_axes(R(x0 + wcol * 0.51, 0.015, wcol * 0.49, 0.215))
        self.ax_tel.set_facecolor(PANEL)
        self.ax_tel.set_xticks([])
        self.ax_tel.set_yticks([])
        for sp in self.ax_tel.spines.values():
            sp.set_color("#2c3a2d")
        self.txt_tel = self.ax_tel.text(0.03, 0.96, "", transform=self.ax_tel.transAxes, va="top", ha="left",
                                        fontsize=7.9, color=INK, family=MONO, linespacing=1.22)
        # precompute coverage curve
        times = np.linspace(0, res.t_end, 400)
        comp = res.completion_at(times)
        valid = res.cell_target_kg > 0
        self._cov_t = times / 3600.0
        self._cov_pct = 100.0 * np.mean(comp[:, valid] >= 0.999, axis=1)

    # ------------------------------------------------------------- per frame
    def frame(self, t: float, speed: float) -> np.ndarray:
        res = self.res
        if t < self._last_t:
            self.painter.reset()
        self.painter.advance(t)
        self._last_t = t
        ov = self.painter.rgba()
        self.im_cov.set_data(ov)
        self.im_cov2.set_data(ov)

        ms = res.ms_state_at(t)
        L = res.ms_log
        k = int(np.searchsorted(L["t"], t, side="right"))
        xs, ys = L["x"][:k], L["y"][:k]
        xs = np.append(xs, ms["x"])
        ys = np.append(ys, ms["y"])
        self.ln_act.set_data(xs, ys)
        self.ln_glow.set_data(xs, ys)
        self.ln_act2.set_data(xs, ys)
        self.ms_patch.set_xy(_ms_polygon(ms["x"], ms["y"], ms["yaw"], 2.2))
        self.ms_patch2.set_xy(_ms_polygon(ms["x"], ms["y"], ms["yaw"], 1.0))
        self.ring.center = (ms["x"], ms["y"])

        # detections + bypasses
        alert = None
        for i, det in enumerate(res.detours):
            if t >= det["t"]:
                self.ln_bypass[i].set_data(det["pts"][:, 0], det["pts"][:, 1])
            else:
                self.ln_bypass[i].set_data([], [])
        for e in self._det:
            p = self.static["obstacles"][e["oid"]]
            if t >= e["t"]:
                p.set_facecolor((1.0, 0.24, 0.95, 0.85))
                p.set_edgecolor("white")
                p.set_linestyle("-")
                if t - e["t"] < 60.0:
                    alert = f"LIDAR CONTACT · unknown {e['obstacle_type']} at {e['range_m']:.1f} m · {e.get('action', '')}"
        if alert:
            self.txt_alert.set_text(alert)
            self.txt_alert.set_visible(True)
            self.ring.set_edgecolor(BYPASS)
            self.ring.set_linewidth(1.8)
        else:
            self.txt_alert.set_visible(False)
            self.ring.set_edgecolor(ACTUAL)
            self.ring.set_linewidth(0.8)

        # stations colours
        cols = []
        cur_sid = -1
        for v in res.visits:
            if v["t_arrive"] <= t < v["t_depart"]:
                cur_sid = v["sid"]
        done = {v["sid"] for v in res.visits if v["t_depart"] <= t}
        for s in res.plan.stations:
            cols.append("#7bdc6a" if s.sid in done else (COLORS["amber"] if s.sid == cur_sid else "#5a5a5a"))
        self.sc_st.set_facecolors(cols)

        # drones
        pos = []
        drops = []
        phases = []
        for d in range(res.n_drones):
            stt = res.drone_state_at(d, t)
            pos.append((stt["x"], stt["y"]))
            phases.append(stt)
            k_s = stt["sortie"]
            if k_s >= 0 and stt["phase"] != 0:
                tr = res.trajectory(k_s)
                m = tr.t <= t
                self.dr_planned[d].set_data(tr.planned[:, 0], tr.planned[:, 1])
                self.dr_actual[d].set_data(tr.actual[m, 0], tr.actual[m, 1])
                self.dr_planned2[d].set_data(tr.planned[:, 0], tr.planned[:, 1])
                self.dr_actual2[d].set_data(tr.actual[m, 0], tr.actual[m, 1])
                if len(tr.drops):
                    dm = tr.drop_t <= t
                    drops.extend(tr.drops[dm, :2].tolist())
            else:
                for ln in (self.dr_planned[d], self.dr_actual[d], self.dr_planned2[d], self.dr_actual2[d]):
                    ln.set_data([], [])
            # previous sortie (faint) for context during time-lapse
            order, t_load, t_land = res._index_for(d)
            j = int(np.searchsorted(t_land, t)) - 1
            if j >= 0:
                trp = res.trajectory(int(order[j]))
                self.dr_prev[d].set_data(trp.actual[:, 0], trp.actual[:, 1])
        parked = cur_sid >= 0
        self.fx_ring.center = self.fx_ring2.center = (ms["x"], ms["y"])
        self.fx_ring.set_visible(parked)
        self.fx_ring2.set_visible(parked)
        fl, fp, fx_rows = _effector_overlay(res, ms, t, self._fx_rng)
        for i in range(2):
            seg = fl[i] if i < len(fl) else np.zeros((0, 2))
            self.ln_fx[i].set_data(seg[:, 0], seg[:, 1])
            self.ln_fx2[i].set_data(seg[:, 0], seg[:, 1])
        self.sc_fx.set_offsets(fp)
        self.sc_fx2.set_offsets(fp)
        pos = np.array(pos)
        self.sc_dr.set_offsets(pos)
        self.sc_dr2.set_offsets(pos)
        self.sc_drops.set_offsets(np.array(drops) if drops else np.zeros((0, 2)))
        r = 35.0
        self.ax_mini.set_xlim(ms["x"] - r * 1.15, ms["x"] + r * 1.15)
        self.ax_mini.set_ylim(ms["y"] - r * 0.65, ms["y"] + r * 0.65)

        # charts
        th = t / 3600.0
        kk = int(np.searchsorted(self._dep_t, t, side="right"))
        if kk > 0:
            step = max(1, kk // 600)
            self.ln_bio.set_data(np.append(self._dep_t[:kk:step] / 3600, th), np.append(self._dep_cum[:kk:step],
                                                                                      self._dep_cum[kk - 1]))
        m = self._cov_t <= th
        self.ln_cov.set_data(np.append(self._cov_t[m], th), np.append(self._cov_pct[m], self._cov_pct[m][-1] if m.any() else 0))
        km = int(np.searchsorted(self._ms_t, t, side="right"))
        if km > 0:
            step = max(1, km // 400)
            self.ln_ms.set_data(np.append(self._ms_t[:km:step] / 3600, th), np.append(self._ms_cum[:km:step],
                                                                                   self._ms_cum[km - 1]))
        else:
            self.ln_ms.set_data([], [])
        self.vl.set_xdata([th, th])

        # text
        applied = float(self._dep_cum[kk - 1]) if kk > 0 else 0.0
        mode_name = LABEL_NAMES.get(ms["mode"], "?")
        if cur_sid >= 0:
            eff = "air cannons" if res.spec.mode.value == "cartridge" else "chip blower"
            phase_line = f"DRONE STATION {cur_sid + 1}/{len(res.plan.stations)} · parked · swarm + {eff} working"
        elif t >= res.t_end - 1:
            phase_line = "MISSION COMPLETE · back at depot"
        else:
            phase_line = {"lane": "DRIVING · bed lane (wheels locked in furrows)",
                          "headland": "DRIVING · headland U-turn",
                          "transit": "DRIVING · transfer on farm tracks",
                          "weed_detour": "DRIVING · bypassing mapped weed patch",
                          "obstacle_detour": "DRIVING · lidar bypass of unknown obstacle"}.get(mode_name, "DRIVING")
        self.txt_phase.set_text(f"T+{t / 3600:6.2f} h   {phase_line}")
        self.txt_clock.set_text(_fmt_clock(t))
        if speed < 1.5:
            sp_txt = "real time"
        elif speed < 100:
            sp_txt = f"x{speed:.0f} real time"
        else:
            sp_txt = f"time-lapse x{speed:,.0f}"
        self.txt_speed.set_text(sp_txt)

        n_sort = int(np.searchsorted(self._sortie_t, t, side="right"))
        n_swap = int(np.searchsorted(self._swap_t, t, side="right"))
        n_det = sum(1 for e in self._det if e["t"] <= t)
        hop = ms["hopper_kg"]
        bar = "█" * int(round(10 * hop / res.cfg.hopper_capacity_kg)) + "░" * (10 - int(round(10 * hop / res.cfg.hopper_capacity_kg)))
        lines = [
            "MOTHERSHIP",
            f" speed  {ms['v']:4.2f} / cmd {ms['v_cmd']:4.2f} m/s",
            f" steer  {math.degrees(ms['steer']):+5.1f} / cmd {math.degrees(ms['steer_cmd']):+5.1f} deg",
            f" x-trk  {100 * ms['cte']:+5.1f} cm  (furrow ±20)",
            f" hopper {bar} {hop:5.0f} kg",
        ]
        lines += [f"EFFECTOR (reach {res.cfg.effectors.range_max_m:.1f} m / 40 ft)"] + fx_rows
        lines += [
            "SWARM  phase      cell  batt",
        ]
        for d, stt in enumerate(phases):
            ph = PHASE_NAMES[stt["phase"]] if stt["phase"] else ("loading" if stt["sortie"] >= 0 else "docked")
            cell = f"{stt['cell']:4d}" if stt["cell"] >= 0 else "   -"
            lines.append(f" D{d}    {ph:<9s} {cell}  {100 * res.battery_at(d, t):3.0f}%")
        if self.kpi is None:
            lines += [
                f"applied {applied:7.1f} / {self._target_t:,.0f} t  (MS {float(self._ms_cum[km - 1]) if km else 0.0:5.1f})",
                f"sorties {n_sort:7,d}   swaps {n_swap:5d}",
                f"unknown obstacles found: {n_det}",
            ]
        self.txt_tel.set_text("\n".join(lines))

        if self.cams is not None:
            imgs = self.cams(t)
            if "main" in imgs:
                self.im_cam.set_data(imgs["main"])
                self.im_cam.set_extent((0, imgs["main"].shape[1], imgs["main"].shape[0], 0))
            if "second" in imgs:
                self.im_cam2.set_data(imgs["second"])
                self.im_cam2.set_extent((0, imgs["second"].shape[1], imgs["second"].shape[0], 0))
        self.fig.canvas.draw()
        buf = np.asarray(self.fig.canvas.buffer_rgba())[..., :3].copy()
        if self.kpi is not None:
            y0, y1 = (int(round(self.H * (1 - v))) for v in self.CARD_STRIP[::-1])
            buf = self.kpi.draw(buf, t, box=(int(0.008 * self.W), y0, int(0.992 * self.W), y1), bg_alpha=0.92)
        return buf

    def close(self) -> None:
        plt.close(self.fig)


def render_dashboard(res: MissionResult, out_path: Path, fps: int = 24, target_s: float = 72.0,
                     cam_provider=None, basemap: Basemap | None = None, max_frames: int | None = None,
                     still_every: int = 0) -> Path:
    times, speed = mission_timeline(res, target_s=target_s, fps=fps)
    if max_frames is not None and len(times) > max_frames:
        sel = np.linspace(0, len(times) - 1, max_frames).astype(int)
        times, speed = times[sel], speed[sel]
    from prunemaxxing.viz.video_io import video_writer

    dash = Dashboard(res, basemap=basemap, cam_provider=cam_provider)
    try:
        with video_writer(out_path, fps) as writer:
            for i, (t, sp) in enumerate(zip(times, speed)):
                img = dash.frame(float(t), float(sp))
                writer.append_data(img)
                if still_every and i % still_every == 0:
                    imageio.imwrite(out_path.with_name(f"dashboard_still_{i:05d}.png"), img)
    finally:
        dash.close()
    return out_path


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------


def export_mission_data(res: MissionResult, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    p = out_dir / "control_log.csv"
    res.write_control_log(p)
    paths["control_log"] = p
    p = out_dir / "sorties.csv"
    S = res.sorties
    cols = list(S.keys())
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for i in range(len(S["kg"])):
            w.writerow([f"{S[c][i]:.3f}" if S[c].dtype.kind == "f" else int(S[c][i]) for c in cols])
    paths["sorties"] = p
    p = out_dir / "events.json"
    ev = [e for e in res.events if e["kind"] not in ("battery_swap", "tender_topup")]
    p.write_text(json.dumps(ev, indent=1, default=float))
    paths["events"] = p
    p = out_dir / "kpis.json"
    p.write_text(json.dumps(res.kpis, indent=1, default=float))
    paths["kpis"] = p
    p = out_dir / "detours.json"
    p.write_text(json.dumps([{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in d.items()}
                             for d in res.detours], default=float))
    paths["detours"] = p
    return paths
