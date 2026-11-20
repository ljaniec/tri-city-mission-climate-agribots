#!/usr/bin/env python3
"""One-slide project overview for a 3-minute pitch (16:9 PNG + PDF).

Numbers come from the rendered showcases (``outputs/showcase_<payload>/kpis.json``)
and the climate impact summary (``outputs/impact/impact_summary.json``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prunemaxxing.viz.style import COLORS  # noqa: E402

BG, PANEL, INK, MUTED = COLORS["bg"], "#18211a", COLORS["ink"], COLORS["muted"]
AMBER, CYAN, RED, GREEN, VIOLET = COLORS["amber"], "#38d6e6", "#e2574c", "#7bdc6a", "#a78bfa"
EDGE = "#2c3a2d"


def _load(out: Path) -> dict:
    k = {m: json.loads((out / f"showcase_{m}" / "kpis.json").read_text()) for m in ("dust", "cartridge")}
    imp = json.loads((out / "impact" / "impact_summary.json").read_text())["summaries"]
    return {"k": k, "imp": imp}


def _box(fig, x, y, w, h, fc=PANEL, ec=EDGE, lw=1.2, r=0.008, z=1, alpha=1.0):
    fig.patches.append(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                      transform=fig.transFigure, fc=fc, ec=ec, lw=lw, zorder=z, alpha=alpha))


def _arrow(fig, path, color=MUTED, lw=2.2, style="-|>", ms=16, z=5, ls="-"):
    """Polyline arrow in figure coords; the head is drawn on the last segment."""
    for (x0, y0), (x1, y1) in zip(path[:-2], path[1:-1]):
        fig.lines.append(plt.Line2D([x0, x1], [y0, y1], transform=fig.transFigure, color=color, lw=lw, zorder=z,
                                    ls=ls, solid_capstyle="round"))
    fig.patches.append(FancyArrowPatch(path[-2], path[-1], transform=fig.transFigure, arrowstyle=style,
                                       mutation_scale=ms, color=color, lw=lw, zorder=z, ls=ls))


def _step(fig, x, y, w, h, n, title, lines, color, title_size=14.5, line_size=11.2):
    _box(fig, x, y, w, h)
    _box(fig, x, y, 0.0045, h, fc=color, ec="none", r=0.002, z=2)
    cx, top = x + 0.016, y + h - 0.022
    fig.text(cx, top, str(n), fontsize=12, fontweight="bold", color=BG, ha="center", va="center", zorder=4,
             bbox=dict(boxstyle="circle,pad=0.28", fc=color, ec="none"))
    fig.text(cx + 0.014, top, title, fontsize=title_size, fontweight="bold", color=INK, va="center", zorder=4)
    for i, ln in enumerate(lines):
        fig.text(x + 0.012, top - 0.04 - i * 0.03, ln, fontsize=line_size, color=MUTED if ln.startswith("·") else INK,
                 va="center", zorder=4)


def _bullets(fig, x, y, head, color, items, size=13.8, gap=0.035, wrap_gap=0.0268):
    fig.patches.append(FancyBboxPatch((x, y - 0.012), 0.004, 0.026, boxstyle="round,pad=0,rounding_size=0.002",
                                      transform=fig.transFigure, fc=color, ec="none"))
    fig.text(x + 0.011, y, head, fontsize=13, fontweight="bold", color=color, va="center")
    y -= 0.039
    for it in items:
        rows = it.split("\n")
        fig.text(x + 0.006, y, "•", fontsize=size, color=color, va="center")
        for j, row in enumerate(rows):
            fig.text(x + 0.02, y - j * wrap_gap, row, fontsize=size, color=INK, va="center")
        y -= gap + wrap_gap * (len(rows) - 1)
    return y - 0.006


def _after(fig, txt, pad=0.012) -> float:
    """Figure-x just right of an already placed text artist."""
    bb = txt.get_window_extent(renderer=fig.canvas.get_renderer())
    return fig.transFigure.inverted().transform((bb.x1, bb.y0))[0] + pad


def _hero(fig, path: Path, rect, crop=(0.08, 0.14, 0.92, 0.92)):
    im = Image.open(path).convert("RGB")
    W, H = im.size
    im = im.crop((int(crop[0] * W), int(crop[1] * H), int(crop[2] * W), int(crop[3] * H)))
    ax = fig.add_axes(rect)
    ax.imshow(im, aspect="auto", interpolation="lanczos")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(EDGE)
        sp.set_linewidth(1.2)


def render(out_dir: Path, outputs: Path, dpi: int = 200) -> list[Path]:
    d = _load(outputs)
    kd, kc, imp = d["k"]["dust"], d["k"]["cartridge"], d["imp"]
    dust, cart = imp["dust"], imp["cartridge"]
    save_pct = 100.0 * kd["saving_vs_regensol_eur_per_1000m2"] / kd["regensol_baseline_eur_per_1000m2"]
    n_obst = len(kd["obstacles_detected"])

    fig = plt.figure(figsize=(19.2, 10.8), dpi=dpi, facecolor=BG)
    plt.rcParams["font.family"] = "DejaVu Sans"

    # header
    t = fig.text(0.025, 0.945, "PRUNEMAXXING", fontsize=40, fontweight="bold", color=AMBER, va="center")
    fig.text(_after(fig, t, 0.016), 0.945, "prunings into living soil, spread by an autonomous robot fleet",
             fontsize=24, color=INK, va="center")
    fig.text(0.025, 0.897, "AGR-E2  ·  Parc Agrari del Baix Llobregat, Viladecans  ·  Unite! Tri-City Hackathon 2026",
             fontsize=13, color=MUTED, va="center")
    fig.text(0.975, 0.897, "mothership UGV + 4 wireless heavy-lift drones  ·  1:1 MuJoCo digital twin",
             fontsize=13, color=MUTED, va="center", ha="right")
    fig.lines.append(plt.Line2D([0.025, 0.975], [0.875, 0.875], transform=fig.transFigure, color=EDGE, lw=1.4))

    # left column: bullets
    x0, y = 0.025, 0.845
    y = _bullets(fig, x0, y, "PROBLEM", RED, [
        "Urban and forest prunings are pile-burned or\nlandfilled: smoke, CH\u2084 / N\u2082O, lost soil carbon",
        f"Manual mulching (REGENSOL) costs "
        f"\u20ac{kd['regensol_baseline_eur_per_1000m2']:,.0f} / 1000 m\u00b2",
    ])
    y = _bullets(fig, x0, y, "SOLUTION", AMBER, [
        f"Mothership carries 2.5 t and parks at {kd['stations']} stations;\n"
        f"{kd['swarm_size']} drones fly 25 kg sorties to every bed",
        "40 ft chip blower or twin air cannons on board\ntreat everything within 12 m of the robot",
        "Two payloads: wood-chip mulch (15 kg/m\u00b2) or\nseed + mycorrhiza cartridges (1.2 kg/m\u00b2)",
    ])
    y = _bullets(fig, x0, y, f"RESULTS  ·  {kd['area_treated_ha']:.2f} ha simulated", GREEN, [
        f"{kd['biomass_applied_t']:,.0f} t in {kd['mission_time_h']:.0f} h, {kd['cells_complete_pct']:.0f} % of cells "
        f"at dose (CV {100 * kd['dose_cv']:.2f} %)",
        f"{n_obst}/{n_obst} unmapped obstacles found by lidar and\nbypassed online",
        f"Cartridges: {cart['seeds_placed']:,.0f} seeds in {kc['mission_time_h']:.0f} h",
    ])
    _bullets(fig, x0, y, "NEXT", CYAN, [
        "Field pilot with Parc Agrari farmers and REGENSOL;\nreal-drone payload and swap tests",
    ])

    # right: system flow
    fx0, fx1 = 0.395, 0.975
    _box(fig, fx0, 0.19, fx1 - fx0, 0.675, fc="#121912", ec=EDGE, r=0.01, z=0)
    t = fig.text(fx0 + 0.015, 0.84, "SYSTEM FLOW", fontsize=15, fontweight="bold", color=AMBER, va="center")
    fig.text(_after(fig, t), 0.84, "tasks and actions, from satellite map to climate report", fontsize=13,
             color=MUTED, va="center")

    bw, bh, ry = 0.128, 0.155, 0.645
    xs = [fx0 + 0.015 + i * (bw + 0.026) for i in range(3)]
    _step(fig, xs[0], ry, bw, bh, 1, "MAP", ["Satellite + 10 m DEM", "\u2192 digital twin", "· parcels, furrows, weeds"],
          CYAN)
    _step(fig, xs[1], ry, bw, bh, 2, "PLAN", ["Furrow-centred route", f"+ {kd['stations']} drone stations",
                                              "· dose per 10 m cell"], CYAN)
    _step(fig, xs[2], ry, bw, bh, 3, "LOAD", ["Depot \u2192 2.5 t hopper", "mulch or cartridges",
                                              "· supply-truck top-ups"], AMBER)
    for a, b in ((xs[0], xs[1]), (xs[1], xs[2])):
        _arrow(fig, [(a + bw + 0.003, ry + bh / 2), (b - 0.003, ry + bh / 2)], color=INK)
    _hero(fig, ROOT / "outputs" / "showcase_dust" / "hero_swarm.png",
          [xs[2] + bw + 0.02, ry, fx1 - 0.015 - (xs[2] + bw + 0.02), bh])

    # mission loop
    ly0, ly1 = 0.335, 0.605
    _box(fig, fx0 + 0.012, ly0, fx1 - fx0 - 0.024, ly1 - ly0, fc="none", ec=AMBER, lw=1.6, r=0.01, z=0)
    fig.text(fx1 - 0.03, ly1 - 0.002, f"  MISSION LOOP  ·  repeated at {kd['stations']} stations  ", fontsize=12,
             fontweight="bold", color=AMBER, va="center", ha="right", zorder=6,
             bbox=dict(fc="#121912", ec="none", pad=1.5))
    sy, sh = ly0 + 0.045, 0.19
    d4x, d4w = fx0 + 0.028, 0.142
    d5x, d5w = d4x + d4w + 0.03, 0.218
    d6x = d5x + d5w + 0.03
    d6w = fx1 - 0.028 - d6x
    _step(fig, d4x, sy, d4w, sh, 4, "DRIVE", ["Pure pursuit in furrows", f"{100 * kd['cte_rms_lane_m']:.1f} cm lane RMS",
                                               "Lidar 25 m: unknown", "obstacle \u2192 re-plan"], AMBER)
    _box(fig, d5x, sy, d5w, sh)
    _box(fig, d5x, sy, 0.0045, sh, fc=GREEN, ec="none", r=0.002, z=2)
    top = sy + sh - 0.022
    fig.text(d5x + 0.016, top, "5", fontsize=12, fontweight="bold", color=BG, ha="center", va="center", zorder=4,
             bbox=dict(boxstyle="circle,pad=0.28", fc=GREEN, ec="none"))
    fig.text(d5x + 0.03, top, "PARK & SPREAD", fontsize=14.5, fontweight="bold", color=INK, va="center", zorder=4)
    fig.text(d5x + 0.155, top, "in parallel", fontsize=11, color=MUTED, va="center", zorder=4)
    lanes = [(VIOLET, "Drones", "load \u2192 fly \u2192 spread \u2192 land", "battery swap, deck charging"),
             (AMBER, "Mothership", "chip blower / air cannons", "everything within 12 m")]
    for i, (c, who, a, b) in enumerate(lanes):
        yy = top - 0.05 - i * 0.068
        fig.patches.append(FancyBboxPatch((d5x + 0.012, yy - 0.026), d5w - 0.022, 0.056,
                                          boxstyle="round,pad=0,rounding_size=0.005", transform=fig.transFigure,
                                          fc="#202b22", ec=c, lw=1.1, zorder=3))
        fig.text(d5x + 0.02, yy + 0.012, who, fontsize=11.5, fontweight="bold", color=c, va="center", zorder=4)
        fig.text(d5x + 0.088, yy + 0.012, a, fontsize=11, color=INK, va="center", zorder=4)
        fig.text(d5x + 0.088, yy - 0.013, b, fontsize=10.5, color=MUTED, va="center", zorder=4)
    _step(fig, d6x, sy, d6w, sh, 6, "VERIFY", ["Per-cell dose log", "Coverage check", "Gaps \u2192 re-task",
                                                "· next station"], GREEN)
    _arrow(fig, [(d4x + d4w + 0.003, sy + sh / 2), (d5x - 0.003, sy + sh / 2)], color=INK)
    _arrow(fig, [(d5x + d5w + 0.003, sy + sh / 2), (d6x - 0.003, sy + sh / 2)], color=INK)
    yb = ly0 + 0.018
    _arrow(fig, [(d6x + d6w / 2, sy - 0.002), (d6x + d6w / 2, yb), (d4x + d4w / 2, yb), (d4x + d4w / 2, sy - 0.003)],
           color=AMBER, lw=2.0)
    fig.text((d4x + d6x + d6w) / 2, yb, "  drive to the next station  ", fontsize=10.5, color=AMBER, ha="center",
             va="center", zorder=6, bbox=dict(fc="#121912", ec="none", pad=1.0))
    # LOAD -> loop entry, loop exit -> REPORT
    lx = xs[2] + bw / 2
    _arrow(fig, [(lx, ry - 0.003), (lx, 0.625), (d4x + d4w / 2, 0.625), (d4x + d4w / 2, sy + sh + 0.003)], color=INK)

    rx, ry2, rw, rh = fx0 + 0.012, 0.205, fx1 - fx0 - 0.024, 0.105
    _box(fig, rx, ry2, rw, rh)
    _box(fig, rx, ry2, 0.0045, rh, fc=CYAN, ec="none", r=0.002, z=2)
    fig.text(rx + 0.016, ry2 + rh / 2, "7", fontsize=12, fontweight="bold", color=BG, ha="center", va="center",
             zorder=4, bbox=dict(boxstyle="circle,pad=0.28", fc=CYAN, ec="none"))
    fig.text(rx + 0.03, ry2 + rh / 2 + 0.02, "REPORT", fontsize=14.5, fontweight="bold", color=INK, va="center")
    fig.text(rx + 0.03, ry2 + rh / 2 - 0.02, "after the last station", fontsize=10.5, color=MUTED, va="center")
    fig.text(rx + 0.135, ry2 + rh / 2 + 0.02, "Dose map + KPIs  ·  CO\u2082e, PM2.5 and \u20ac per field",
             fontsize=12, color=INK, va="center")
    fig.text(rx + 0.135, ry2 + rh / 2 - 0.02, "Live dashboard  ·  animated infographic  ·  interactive HTML replay",
             fontsize=12, color=MUTED, va="center")
    _arrow(fig, [(d6x + d6w - 0.02, ly0 - 0.002), (d6x + d6w - 0.02, ry2 + rh + 0.003)], color=INK)

    # KPI strip
    tiles = [
        (f"{kd['biomass_applied_t']:,.0f} t", f"prunings spread in {kd['mission_time_h']:.0f} h", AMBER),
        (f"{kd['cells_complete_pct']:.0f} %", f"of cells at dose (CV {100 * kd['dose_cv']:.2f} %)", GREEN),
        (f"{kd['sorties']:,}", f"drone sorties by {kd['swarm_size']} drones", VIOLET),
        (f"\u2212{save_pct:.0f} %", f"cost vs REGENSOL (\u20ac{dust['saving_eur_total'] / 1000:,.0f}k on the site)",
         AMBER),
        (f"{dust['total_co2e_t']:,.0f} t CO\u2082e", "stored in soil + burning avoided", GREEN),
        (f"{dust['pm25_avoided_kg'] / 1000:.1f} t", "PM2.5 smoke avoided", CYAN),
    ]
    n, gx = len(tiles), 0.012
    tw = (0.95 - gx * (n - 1)) / n
    for i, (v, lab, c) in enumerate(tiles):
        tx = 0.025 + i * (tw + gx)
        _box(fig, tx, 0.04, tw, 0.115)
        _box(fig, tx, 0.04, tw, 0.006, fc=c, ec="none", r=0.002, z=2)
        fig.text(tx + 0.014, 0.115, v, fontsize=29, fontweight="bold", color=c, va="center")
        fig.text(tx + 0.014, 0.068, lab, fontsize=11.5, color=INK, va="center")
    fig.text(0.025, 0.016, "Wood-chip mulch mission on the 9.08 ha Viladecans digital twin (DEM 10 m, MuJoCo 1:1)  ·  "
             "Impact factors: IPCC 2006 V4 Tab. 2.5, IPCC AR6 GWP100, Andreae & Merlet 2001, REGENSOL trials "
             "(AGR-E2 brief)", fontsize=9.5, color=MUTED, va="center")

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / "pitch_slide.png", out_dir / "pitch_slide.pdf"]
    for p in paths:
        fig.savefig(p, dpi=dpi, facecolor=BG)
    plt.close(fig)
    return paths


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "pitch"))
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()
    for p in render(Path(args.out), ROOT / "outputs", dpi=args.dpi):
        print(p.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
