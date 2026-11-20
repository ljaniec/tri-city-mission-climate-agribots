"""One-page climate impact poster (16:9 slide and print handout).

``render_impact_poster(results, out_path)`` draws a static English infographic for one
or two finished missions (DUST and/or CARTRIDGE). Every number is read from
:func:`prunemaxxing.kpis.impact.impact_summary`, the mission KPIs or the payload spec;
the footnotes are generated from :func:`prunemaxxing.kpis.impact.factor_table`.
Icons are matplotlib patches, no external images.

The page is a single axes whose data units are inches (19.2 x 10.8), so text can be
measured for layout and icons stay round.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any, Callable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402
from matplotlib.patches import (  # noqa: E402
    Circle,
    Ellipse,
    FancyBboxPatch,
    PathPatch,
    Polygon,
    Rectangle,
    RegularPolygon,
    Wedge,
)
from matplotlib.path import Path as MPath  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.ndimage import maximum_filter  # noqa: E402

from prunemaxxing.config import REGENSOL_OVERHEAD_EUR_PER_1000M2, TRANSPORT_SPREAD_SHARE  # noqa: E402
from prunemaxxing.kpis.impact import FACTORS, factor_table, impact_summary  # noqa: E402
from prunemaxxing.viz.basemap import Basemap, CoveragePainter, build_basemap  # noqa: E402
from prunemaxxing.viz.mission_stats import MissionTimeline  # noqa: E402
from prunemaxxing.viz.mission_viz import ACTUAL, BG, CHIP, FXC, OBST, PANEL, WEED  # noqa: E402
from prunemaxxing.viz.style import COLORS, FONT_FAMILY  # noqa: E402

__all__ = ["EQUIVALENCES", "factor_lines", "render_impact_poster"]

SANS = FONT_FAMILY
W, H = 19.2, 10.8
M = 0.32  # outer margin [in]
GAP = 0.16
EDGE = "#2c3a2d"
INK, MUTED, AMBER = COLORS["ink"], COLORS["muted"], COLORS["amber"]
MS = ACTUAL
GREEN = "#7CFC00"
CANOPY = COLORS["canopy"]
SOIL_C = "#5fae45"
FIRE = "#ff6a3d"
FIRE_LT = "#ffab85"
NITRO = "#c3ec6a"
SMOKE = "#c4ccc0"
CART = "#9be15d"
MODE_COLOR = {"dust": CHIP, "cartridge": CART}
MODE_TITLE = {"dust": "wood-chip mulch", "cartridge": "seed cartridges"}
SPECIES = {
    "Quercus_ilex": ("holm oak", "oak"),
    "Pinus_halepensis": ("Aleppo pine", "pine"),
    "Ceratonia_siliqua": ("carob tree", "carob"),
}

# Everyday equivalence (context only, not a model factor).
EQUIVALENCES: dict[str, dict[str, Any]] = {
    "spain_ghg_per_capita": {
        "value": 5.5,
        "unit": "t CO2e / person / yr",
        "label": "greenhouse-gas emissions per resident of Spain, 2024",
        "source": "Eurostat sdg_13_10, total excl. LULUCF and memo items (EEA inventory data, updated 22 Apr 2026)",
    },
}

HEADER_WORD_Y, HEADLINE_Y, SUB_Y, RULE_Y = 10.36, 9.70, 9.34, 9.15
HERO_Y, HERO_H = 6.86, 2.14
MID_Y, MID_H = 2.52, 4.18
FLEET_Y, FLEET_H = 1.30, 1.08
FOOT_Y, FOOT_H = 0.10, 1.06


# ---------------------------------------------------------------------------
# formatting
# ---------------------------------------------------------------------------


def _n(x: float, d: int = 0) -> str:
    """Half-up rounding with thousands separators (434.5 -> '435')."""
    q = Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP)
    return f"{q:,.{d}f}"


def _auto(x: float) -> str:
    a = abs(x)
    return _n(x, 0 if a >= 100 else 1 if a >= 1 else 2)


def _mass_row(kgs: Sequence[float]) -> list[str]:
    """Same unit for every value of a row (t when the largest is >= 10 t)."""
    if max(kgs) >= 10_000:
        return [f"{_auto(v / 1000.0)} t" for v in kgs]
    return [f"{_auto(v)} kg" for v in kgs]


def _mass(kg: float) -> str:
    return _mass_row([kg])[0]


_SUBS = [(r"CO2e", "CO₂e"), (r"CO2", "CO₂"), (r"CH4", "CH₄"), (r"N2O", "N₂O"), (r"\bm2\b", "m²")]


def _pretty(s: str) -> str:
    for a, b in _SUBS:
        s = re.sub(a, b, s)
    return s


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:]


def _ft(m: float) -> str:
    return f"{m / 0.3048:.0f} ft"


# ---------------------------------------------------------------------------
# page
# ---------------------------------------------------------------------------


class _Page:
    def __init__(self, dpi: int):
        self.dpi = dpi
        self.fig = plt.figure(figsize=(W, H), dpi=dpi, facecolor=BG)
        ax = self.fig.add_axes((0, 0, 1, 1))
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.set_autoscale_on(False)
        ax.axis("off")
        self.ax = ax
        self.renderer = self.fig.canvas.get_renderer()
        self._fp: dict[tuple, FontProperties] = {}

    def tw(self, s: str, size: float, weight: str = "normal", style: str = "normal") -> float:
        """Rendered width of ``s`` in inches."""
        key = (size, weight, style)
        fp = self._fp.get(key)
        if fp is None:
            fp = self._fp[key] = FontProperties(family=SANS, size=size, weight=weight, style=style)
        w, _, _ = self.renderer.get_text_width_height_descent(s, fp, ismath=False)
        return w / self.dpi

    def text(self, x, y, s, size=12.0, color=INK, weight="normal", ha="left", va="baseline", style="normal",
             z=6.0, **kw):
        return self.ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, fontstyle=style, ha=ha, va=va,
                            family=SANS, zorder=z, **kw)

    def fit(self, s: str, max_w: float, size: float, weight: str = "normal", min_size: float = 6.0,
            style: str = "normal") -> float:
        w = self.tw(s, size, weight, style)
        return size if w <= max_w else max(min_size, size * max_w / w * 0.98)

    def runs(self, x: float, y: float, parts: Sequence[tuple], ha: str = "left") -> float:
        """Draw ``(text, size, color, weight, gap_after)`` runs on one baseline; returns the end x."""
        ws = [self.tw(p[0], p[1], p[3]) for p in parts]
        total = sum(ws) + sum(p[4] for p in parts[:-1])
        x0 = x - total if ha == "right" else x - total / 2 if ha == "center" else x
        for p, w in zip(parts, ws):
            self.text(x0, y, p[0], p[1], p[2], p[3])
            x0 += w + p[4]
        return x0 - parts[-1][4]

    def runs_width(self, parts: Sequence[tuple]) -> float:
        return sum(self.tw(p[0], p[1], p[3]) for p in parts) + sum(p[4] for p in parts[:-1])

    def wrap(self, s: str, max_w: float, size: float, weight: str = "normal") -> list[str]:
        lines: list[str] = []
        cur = ""
        for word in s.split(" "):
            cand = word if not cur else f"{cur} {word}"
            if not cur or self.tw(cand, size, weight) <= max_w:
                cur = cand
            else:
                lines.append(cur)
                cur = word
        if cur:
            lines.append(cur)
        return lines

    def para(self, x, y, s, max_w, size, color=MUTED, lh=1.30, weight="normal") -> float:
        lines = self.wrap(s, max_w, size, weight)
        step = size / 72.0 * lh
        for i, ln in enumerate(lines):
            self.text(x, y - i * step, ln, size, color, weight)
        return y - len(lines) * step

    def box(self, x, y, w, h, fc=PANEL, ec=EDGE, r=0.10, lw=1.0, z=1.0, **kw) -> FancyBboxPatch:
        p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, zorder=z,
                           **kw)
        self.ax.add_patch(p)
        return p

    def rect(self, x, y, w, h, fc, z=4.0, **kw) -> Rectangle:
        p = Rectangle((x, y), w, h, fc=fc, zorder=z, **{"ec": "none", **kw})
        self.ax.add_patch(p)
        return p

    def line(self, xs, ys, color=EDGE, lw=1.0, z=3.0, **kw):
        return self.ax.plot(xs, ys, color=color, lw=lw, zorder=z, solid_capstyle="butt", **kw)[0]

    def chip(self, x, y, s, fc, tc=BG, size=10.5) -> float:
        w = self.tw(s, size, "bold")
        pad = 0.075
        cap = 0.73 * size / 72.0
        self.box(x, y - 0.07, w + 2 * pad, cap + 0.14, fc=fc, ec="none", r=0.05, z=5)
        self.text(x + pad, y, s, size, tc, "bold", z=6)
        return w + 2 * pad

    def legend(self, x_right, y, items: Sequence[tuple[str, str]], size=10.0) -> float:
        """Right-aligned swatch legend; returns its left x."""
        x = x_right
        for label, color in reversed(items):
            x -= self.tw(label, size)
            self.text(x, y, label, size, MUTED)
            x -= 0.19
            self.rect(x, y - 0.005, 0.13, 0.13, color, z=5)
            x -= 0.22
        return x

    def panel(self, x, y, w, h, title=None, sub=None, legend=None) -> None:
        self.box(x, y, w, h)
        if not title:
            return
        ty = y + h - 0.40
        self.text(x + 0.24, ty, title, 14, AMBER, "bold")
        right = x + w - 0.24
        if legend:
            right = self.legend(right, ty, legend) - 0.1
        if sub:
            sx = x + 0.24 + self.tw(title, 14, "bold") + 0.16
            self.text(sx, ty, sub, self.fit(sub, right - sx, 11.5), MUTED)


# ---------------------------------------------------------------------------
# icons (inch coordinates; r = half size)
# ---------------------------------------------------------------------------


def _curve(cx: float, cy: float, r: float, segs) -> MPath:
    start, *rest = segs
    verts, codes = [start], [MPath.MOVETO]
    for c in rest:
        verts += list(c)
        codes += [MPath.CURVE4] * 3
    verts.append(start)
    codes.append(MPath.CLOSEPOLY)
    return MPath(np.asarray(verts, dtype=float) * r + [cx, cy], codes)


_FLAME = [(0, -1), ((-0.80, -1.0), (-0.92, -0.20), (-0.48, 0.32)), ((-0.30, 0.55), (-0.06, 0.72), (0.04, 1.05)),
          ((0.30, 0.70), (0.82, 0.32), (0.72, -0.32)), ((0.64, -0.80), (0.34, -1.0), (0, -1))]
_LEAF = [(-0.80, -0.80), ((-0.95, 0.30), (-0.05, 0.95), (0.88, 0.88)), ((0.95, -0.05), (0.30, -0.95), (-0.80, -0.80))]
_DROP = [(0, 1.0), ((0.22, 0.58), (0.68, 0.12), (0.68, -0.32)), ((0.68, -0.76), (0.36, -1.0), (0, -1.0)),
         ((-0.36, -1.0), (-0.68, -0.76), (-0.68, -0.32)), ((-0.68, 0.12), (-0.22, 0.58), (0, 1.0))]


def _lw(r: float, k: float) -> float:
    return max(0.8, r * 72.0 * k)


def icon_flame(ax, cx, cy, r, color=FIRE, z=7.0):
    ax.add_patch(PathPatch(_curve(cx, cy, r, _FLAME), fc=color, ec="none", zorder=z))
    ax.add_patch(PathPatch(_curve(cx + 0.02 * r, cy - 0.38 * r, 0.55 * r, _FLAME), fc=FXC, ec="none", zorder=z + 0.1))


def icon_leaf(ax, cx, cy, r, color=SOIL_C, z=7.0):
    ax.add_patch(PathPatch(_curve(cx, cy, r, _LEAF), fc=color, ec="none", zorder=z))
    ax.plot([cx - 0.62 * r, cx + 0.52 * r], [cy - 0.62 * r, cy + 0.52 * r], color=PANEL, lw=_lw(r, 0.07),
            solid_capstyle="round", zorder=z + 0.1)
    ax.plot([cx - 0.80 * r, cx - 1.0 * r], [cy - 0.80 * r, cy - 1.0 * r], color=color, lw=_lw(r, 0.10),
            solid_capstyle="round", zorder=z)


def icon_drop(ax, cx, cy, r, color="#5cc8ff", z=7.0):
    ax.add_patch(PathPatch(_curve(cx, cy, r, _DROP), fc=color, ec="none", zorder=z))
    ax.add_patch(Ellipse((cx - 0.28 * r, cy - 0.35 * r), 0.22 * r, 0.42 * r, angle=20, fc="white", alpha=0.55,
                         ec="none", zorder=z + 0.1))


def icon_smoke(ax, cx, cy, r, color=SMOKE, z=7.0):
    for dx, dy, rr in ((-0.45, -0.05, 0.40), (0.02, 0.22, 0.52), (0.48, -0.02, 0.40), (0.0, -0.20, 0.40)):
        ax.add_patch(Circle((cx + dx * r, cy + dy * r), rr * r, fc=color, ec="none", zorder=z))
    for dx, dy in ((-0.55, -0.78), (-0.15, -0.88), (0.25, -0.76), (0.62, -0.90)):
        ax.add_patch(Circle((cx + dx * r, cy + dy * r), 0.09 * r, fc=color, ec="none", zorder=z))


def icon_nitrogen(ax, cx, cy, r, color=NITRO, z=7.0):
    ax.add_patch(RegularPolygon((cx, cy), 6, radius=r, orientation=0.0, fc="none", ec=color, lw=_lw(r, 0.10),
                                zorder=z))
    ax.text(cx, cy, "N", ha="center", va="center", fontsize=r * 72 * 0.95, color=color, fontweight="bold",
            family=SANS, zorder=z)


def icon_coin(ax, cx, cy, r, color=AMBER, z=7.0):
    ax.add_patch(Circle((cx, cy), r, fc=color, ec="none", zorder=z))
    ax.add_patch(Circle((cx, cy), 0.78 * r, fc="none", ec=PANEL, lw=_lw(r, 0.05), alpha=0.45, zorder=z + 0.1))
    ax.text(cx, cy, "€", ha="center", va="center", fontsize=r * 72 * 1.0, color=PANEL, fontweight="bold",
            family=SANS, zorder=z + 0.2)


def icon_seedling(ax, cx, cy, r, color=CART, z=7.0):
    ax.add_patch(Ellipse((cx, cy - 0.80 * r), 1.6 * r, 0.36 * r, fc="#7a5a3a", ec="none", zorder=z))
    ax.plot([cx, cx], [cy - 0.78 * r, cy + 0.20 * r], color=color, lw=_lw(r, 0.11), solid_capstyle="round",
            zorder=z + 0.1)
    ax.add_patch(Ellipse((cx - 0.42 * r, cy + 0.22 * r), 0.95 * r, 0.42 * r, angle=-28, fc=color, ec="none",
                         zorder=z + 0.2))
    ax.add_patch(Ellipse((cx + 0.45 * r, cy + 0.48 * r), 1.05 * r, 0.46 * r, angle=30, fc=color, ec="none",
                         zorder=z + 0.2))


def icon_drone(ax, cx, cy, r, color=INK, z=7.0):
    lw = _lw(r, 0.07)
    for sx in (-1, 1):
        for sy in (-1, 1):
            px, py = cx + sx * 0.6 * r, cy + sy * 0.6 * r
            ax.plot([cx, px], [cy, py], color=color, lw=lw, zorder=z)
            ax.add_patch(Circle((px, py), 0.36 * r, fc="none", ec=color, lw=lw * 0.8, zorder=z))
            ax.add_patch(Circle((px, py), 0.07 * r, fc=color, ec="none", zorder=z))
    ax.add_patch(FancyBboxPatch((cx - 0.22 * r, cy - 0.22 * r), 0.44 * r, 0.44 * r,
                                boxstyle=f"round,pad=0,rounding_size={0.1 * r}", fc=color, ec="none", zorder=z + 0.1))


def icon_tree(ax, cx, cy, r, kind="oak", z=7.0):
    trunk = "#7a5534"
    if kind == "pine":
        ax.add_patch(Polygon([(cx - 0.11 * r, cy - r), (cx + 0.11 * r, cy - r), (cx + 0.10 * r, cy + 0.45 * r),
                              (cx - 0.02 * r, cy + 0.45 * r)], fc=trunk, ec="none", zorder=z))
        ax.plot([cx + 0.03 * r, cx - 0.45 * r], [cy + 0.05 * r, cy + 0.38 * r], color=trunk, lw=_lw(r, 0.07),
                zorder=z)
        ax.plot([cx + 0.06 * r, cx + 0.48 * r], [cy + 0.18 * r, cy + 0.42 * r], color=trunk, lw=_lw(r, 0.06),
                zorder=z)
        for dx, dy, ww, hh, c in ((-0.52, 0.42, 0.85, 0.40, "#466f38"), (0.50, 0.46, 0.85, 0.40, "#466f38"),
                                  (-0.12, 0.62, 1.05, 0.44, "#537f41"), (0.30, 0.78, 0.80, 0.36, "#5f8f4a"),
                                  (-0.34, 0.84, 0.66, 0.30, "#6a9c52")):
            ax.add_patch(Ellipse((cx + dx * r, cy + dy * r), ww * r, hh * r, fc=c, ec="none", zorder=z + 0.1))
    elif kind == "carob":
        ax.add_patch(Rectangle((cx - 0.11 * r, cy - r), 0.22 * r, 0.75 * r, fc=trunk, ec="none", zorder=z))
        ax.add_patch(Ellipse((cx, cy + 0.12 * r), 1.95 * r, 1.25 * r, fc="#2e5a2a", ec="none", zorder=z + 0.1))
        ax.add_patch(Ellipse((cx - 0.25 * r, cy + 0.32 * r), 1.0 * r, 0.62 * r, fc="#3c7035", ec="none",
                             zorder=z + 0.2))
        for dx, dy, a in ((-0.55, -0.22, 70), (0.10, -0.32, 100), (0.55, -0.12, 80)):
            ax.add_patch(Ellipse((cx + dx * r, cy + dy * r), 0.32 * r, 0.09 * r, angle=a, fc="#6b3f1f", ec="none",
                                 zorder=z + 0.3))
    else:
        ax.add_patch(Rectangle((cx - 0.09 * r, cy - r), 0.18 * r, 0.8 * r, fc=trunk, ec="none", zorder=z))
        for dx, dy, rr, c in ((-0.38, 0.05, 0.45, "#2f5a2c"), (0.38, 0.05, 0.45, "#2f5a2c"),
                              (0.0, 0.38, 0.55, "#3d6b3a"), (0.0, 0.0, 0.48, "#35632f")):
            ax.add_patch(Circle((cx + dx * r, cy + dy * r), rr * r, fc=c, ec="none", zorder=z + 0.1))


ICONS: dict[str, Callable] = {
    "flame": icon_flame, "leaf": icon_leaf, "drop": icon_drop, "smoke": icon_smoke, "nitrogen": icon_nitrogen,
    "coin": icon_coin, "seedling": icon_seedling, "drone": icon_drone,
}


def _cartridge_art(ax, cx: float, cy: float, s: float, z: float = 6.0) -> None:
    """Cut-away seed cartridge: pressed chips, three seeds, mycelium threads (s = half width)."""
    rng = np.random.default_rng(4)
    w, hgt, e = 2 * s, 1.15 * s, 0.62 * s
    x0, y0 = cx - s, cy - hgt / 2
    ax.add_patch(Ellipse((cx, y0), w, e, fc="#8a6538", ec="none", zorder=z))
    ax.add_patch(Rectangle((x0, y0), w, hgt, fc="#a77d4a", ec="none", zorder=z))
    for k, a in ((0.0, 0.22), (0.86, 0.18)):
        ax.add_patch(Rectangle((x0 + k * w, y0), 0.14 * w, hgt, fc="black", alpha=a, ec="none", zorder=z + 0.05))
    ax.add_patch(Ellipse((cx, y0 + hgt), w, e, fc="#d8b27a", ec="none", zorder=z + 0.1))
    tones = ["#e9c58e", "#c9995e", "#b8874d", "#f0d3a2", "#9c7040"]
    for _ in range(90):
        t, rr = rng.uniform(0, 2 * np.pi), np.sqrt(rng.uniform(0, 0.85))
        px, py = cx + rr * s * np.cos(t), y0 + hgt + rr * e / 2 * np.sin(t)
        L, Wd, a = rng.uniform(0.07, 0.14) * s, rng.uniform(0.025, 0.045) * s, rng.uniform(0, np.pi)
        d = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
        pts = np.array([[-L, -Wd], [L, -Wd], [L, Wd], [-L, Wd]]) @ d * [1, 0.55] + [px, py]
        ax.add_patch(Polygon(pts, fc=tones[rng.integers(len(tones))], ec="none", zorder=z + 0.15))
    for _ in range(55):
        px, py = rng.uniform(x0 + 0.05 * w, x0 + 0.95 * w), rng.uniform(y0 + 0.05 * hgt, y0 + 0.92 * hgt)
        ax.add_patch(Ellipse((px, py), rng.uniform(0.06, 0.13) * s, rng.uniform(0.02, 0.04) * s,
                             angle=rng.uniform(-30, 30), fc=tones[rng.integers(len(tones))], alpha=0.8, ec="none",
                             zorder=z + 0.06))
    xs = np.linspace(x0 + 0.04 * w, x0 + 0.96 * w, 60)
    for base, ph in ((0.30, 0.0), (0.52, 1.7), (0.74, 3.1)):
        ys = y0 + base * hgt + 0.05 * s * np.sin(xs / s * 9 + ph) + 0.02 * s * np.sin(xs / s * 23 + ph)
        ax.plot(xs, ys, color="#f4f1e6", lw=_lw(s, 0.012), alpha=0.75, zorder=z + 0.2)
    for fx_, fy_, ang in ((0.28, 0.40, 25), (0.52, 0.62, -15), (0.74, 0.36, 40)):
        px, py = x0 + fx_ * w, y0 + fy_ * hgt
        ax.add_patch(Ellipse((px, py), 0.24 * s, 0.15 * s, angle=ang, fc="#4a2a12", ec="#f4f1e6", lw=_lw(s, 0.01),
                             zorder=z + 0.3))
        ax.add_patch(Ellipse((px - 0.03 * s, py + 0.02 * s), 0.08 * s, 0.04 * s, angle=ang, fc="#a0703f",
                             ec="none", zorder=z + 0.35))


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------


@dataclass
class _Mode:
    res: Any
    s: dict[str, Any]
    k: dict[str, Any]
    mode: str
    color: str
    tl: MissionTimeline | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.mode.upper()

    @property
    def effector(self) -> str:
        return "air cannons" if self.mode == "cartridge" else "chip chute"


def _modes(results) -> list[_Mode]:
    if hasattr(results, "kpis"):
        results = [results]
    results = list(results)
    if not 1 <= len(results) <= 2:
        raise ValueError("render_impact_poster takes one or two MissionResults")
    order = {"dust": 0, "cartridge": 1}
    out = []
    for r in results:
        mode = r.spec.mode.value
        out.append(_Mode(r, impact_summary(r), r.kpis, mode, MODE_COLOR.get(mode, AMBER), MissionTimeline(r)))
    out.sort(key=lambda m: order.get(m.mode, 2))
    return out


def _find(modes: list[_Mode], mode: str) -> _Mode | None:
    return next((m for m in modes if m.mode == mode), None)


def _people(co2e_t: float) -> float:
    return co2e_t / EQUIVALENCES["spain_ghg_per_capita"]["value"]


def _cartridges_per_m2(m: _Mode) -> float:
    """Design density from the dose and the cartridge mass (independent of how far the mission got)."""
    spec = m.res.spec
    return float(m.k["dose_kg_per_m2"]) / (spec.unit_mass_kg / max(spec.units_per_sortie, 1))


def _species(m: _Mode) -> list[tuple[str, str, str]]:
    out = []
    for sp in m.s.get("species") or m.res.spec.local_tree_species:
        common, kind = SPECIES.get(sp, (sp.replace("_", " ").lower(), "oak"))
        out.append((common, sp.replace("_", " "), kind))
    return out


# ---------------------------------------------------------------------------
# header + hero
# ---------------------------------------------------------------------------


def _header(P: _Page, modes: list[_Mode]) -> None:
    res = modes[0].res
    s = modes[0].s
    x = P.runs(M, HEADER_WORD_Y, [("PRUNEMAXXING", 26, AMBER, "bold", 0.22), ("CLIMATE IMPACT", 15, INK, "bold", 0)])
    meta = "Parc Agrari del Baix Llobregat, Viladecans  ·  AGR-E2  ·  Unite! Tri-City Hackathon 2026"
    P.text(W - M, HEADER_WORD_Y, meta, P.fit(meta, W - M - x - 0.4, 12.5), MUTED, ha="right")
    n_par = len(res.site.parcels)
    fleet = f"a mothership UGV and {modes[0].k['swarm_size']} wireless heavy-lift drones"
    where = f"{n_par} parcels ({_n(s['area_ha'], 2)} ha)"
    if len(modes) == 2:
        head = "One robot fleet, two ways to keep prunings out of the fire"
        sub = (f"Viladecans prunings are usually pile-burned or landfilled. Instead, {fleet} spread them on {where}: "
               f"as soil mulch or as seeded tree cartridges.")
    else:
        m = modes[0]
        head = f"{_n(s['material_t'])} t of prunings kept out of the fire"
        how = ("as wood-chip mulch" if m.mode == "dust" else "pressed into seeded wood-chip cartridges")
        sub = (f"Viladecans prunings are usually pile-burned or landfilled. Instead, {fleet} spread them on {where}, "
               f"{how} at {m.k['dose_kg_per_m2']:g} kg/m².")
    P.text(M, HEADLINE_Y, head, P.fit(head, W - 2 * M, 36, "bold"), INK, "bold")
    P.text(M, SUB_Y, sub, P.fit(sub, W - 2 * M, 14), MUTED)
    P.line([M, W - M], [RULE_Y, RULE_Y], EDGE, 1.0)


def _hero_metrics(m: _Mode, single: bool) -> list[dict[str, Any]]:
    s, k = m.s, m.k
    co2 = dict(icon="leaf", icolor=SOIL_C, color=GREEN, label="CLIMATE BENEFIT", value=_auto(s["total_co2e_t"]),
               unit="t CO₂e",
               details=[f"stored + avoided · {_n(s['co2e_t_per_ha'], 1)} t/ha",
                        f"≈ yearly GHG of {_n(_people(s['total_co2e_t']))} Spaniards¹"])
    pm = dict(icon="smoke", icolor=SMOKE, color=INK, label="PM2.5 SMOKE AVOIDED", value=_n(s["pm25_avoided_kg"]),
              unit="kg",
              details=["from open pile burning", f"+ {_mass(s['co_avoided_kg'])} CO · "
                                                 f"{_mass(s['nox_avoided_kg'])} NOx"])
    nit = dict(icon="nitrogen", icolor=NITRO, color=NITRO, label="NITROGEN RETAINED",
               value=_n(s["n_retained_kg"]), unit="kg N",
               details=["kept in the soil", f"organic carbon +{_auto(s['soc_increase_pp'])} pp"])
    out = [co2, pm]
    if m.mode == "cartridge":
        sp = _species(m)
        seeds = dict(icon="seedling", icolor=CART, color=CART, label="TREE SEEDS PLACED",
                     value=_n(s["seeds_placed"] or 0), unit="seeds",
                     details=[f"in {_n(s['cartridges_placed'] or 0)} cartridges",
                              f"{len(sp)} local tree species"])
        out += [nit, seeds] if single else [seeds]
    else:
        out.append(nit)
        if single and s.get("saving_eur_total") is not None:
            out.append(dict(icon="coin", icolor=AMBER, color=AMBER, label="SAVED vs REGENSOL",
                            value=_n(s["saving_eur_total"]), unit="€",
                            details=[f"{_n(s['saving_eur_per_1000m2'], 2)} €/1000 m² × {_n(s['area_ha'], 2)} ha",
                                     f"{_n(k.get('overhead_recovered_pct', 0))} % of the regenerative overhead"]))
    return out


def _map_thumb(P: _Page, m: _Mode, x: float, y: float, size: float, bm: Basemap) -> None:
    res = m.res
    e0, n0, e1, n1 = res.site.bounds
    half = max(e1 - e0, n1 - n0) / 2 + 6.0
    ce, cn = (e0 + e1) / 2, (n0 + n1) / 2
    c0, r0 = bm.px(ce - half, cn + half)
    c1, r1 = bm.px(ce + half, cn - half)
    Hh, Ww = bm.rgb.shape[:2]
    c0, r0 = int(max(0, math.floor(c0))), int(max(0, math.floor(r0)))
    c1, r1 = int(min(Ww, math.ceil(c1))), int(min(Hh, math.ceil(r1)))
    E0, E1 = bm.extent[0] + c0 * bm.res, bm.extent[0] + c1 * bm.res
    N1, N0 = bm.extent[3] - r0 * bm.res, bm.extent[3] - r1 * bm.res
    painter = CoveragePainter(bm, res)
    painter.advance(res.t_end)
    ov = painter.rgba()[r0:r1, c0:c1]
    a = ov[..., 3:4].astype(np.float32)
    if m.mode == "cartridge":
        a = maximum_filter(a[..., 0], size=5)[..., None] * 0.95
    else:
        a = np.clip(a * 1.1, 0, 0.92)
    img = bm.rgb[r0:r1, c0:c1].astype(np.float32) / 255.0
    img = img * (1 - a) + ov[..., :3] * a
    img *= 0.92
    px = max(16, int(size * P.dpi * 1.5))
    im = Image.fromarray(np.clip(img * 255, 0, 255).astype(np.uint8)).resize((px, px), Image.LANCZOS)
    frame = P.box(x, y, size, size, fc="none", ec=m.color, r=0.08, lw=1.2, z=5)
    art = P.ax.imshow(np.asarray(im), extent=(x, x + size, y, y + size), origin="upper", zorder=4, aspect="auto",
                      interpolation="antialiased")
    art.set_clip_path(frame)

    def X(e):
        return x + (np.asarray(e) - E0) / (E1 - E0) * size

    def Y(n):
        return y + (np.asarray(n) - N0) / (N1 - N0) * size

    L = res.ms_log
    ln = P.ax.plot(X(L["x"]), Y(L["y"]), color=MS, lw=0.9, alpha=0.9, zorder=4.5)[0]
    ln.set_clip_path(frame)
    found = {e["oid"] for e in res.events if e["kind"] == "obstacle_detected"}
    for ob in res.site.obstacles:
        if ob.oid in found:
            c = ob.polygon.centroid
            P.ax.scatter([X(c.x)], [Y(c.y)], s=14, color=OBST, edgecolor="white", lw=0.5, zorder=4.6)
    P.text(x + 0.08, y + 0.09, f"{_n(m.s['area_ha'], 2)} ha · {len(res.site.parcels)} parcels", 8.5, "white", z=6,
           bbox=dict(boxstyle="round,pad=0.25", fc=(0, 0, 0, 0.55), ec="none"))


DETAIL_SIZE = 10.5


def _unit_size(big: float) -> float:
    return round(big * 0.42, 1)


METRIC_GAP = 0.26


def _value_need(P: _Page, met: dict[str, Any], big: float) -> float:
    vw = P.tw(met["value"], big, "bold")
    if met["unit"]:
        vw += 0.08 + P.tw(met["unit"], _unit_size(big), "bold")
    return max(vw, 0.44 + P.tw(met["label"], 10.5, "bold"))


def _detail_need(P: _Page, met: dict[str, Any], size: float = DETAIL_SIZE) -> float:
    return max(P.tw(d, size) for d in met["details"])


def _metric_widths(P: _Page, metrics: list[dict[str, Any]], area_w: float, big: float,
                   dsize: float | None = None) -> tuple[list[float], float]:
    """Column widths and the largest common detail size (pt) that fits every column."""
    avail = area_w - METRIC_GAP * (len(metrics) - 1)
    vneed = [_value_need(P, mt, big) for mt in metrics]
    size = DETAIL_SIZE if dsize is None else dsize
    while True:
        full = [max(v, _detail_need(P, mt, size)) for v, mt in zip(vneed, metrics)]
        if sum(full) <= avail or size <= 8.5 or dsize is not None:
            break
        size -= 0.25
    scale = min(1.0, avail / sum(full))
    full = [f * scale for f in full]
    extra = (avail - sum(full)) / len(full)
    return [f + extra for f in full], size


def _metric_area(x: float, w: float, h: float, with_map: bool) -> tuple[float, float]:
    """(left x, width) of the metric row inside a hero card."""
    cx0 = x + 0.14 + (h - 0.28) + 0.30 if with_map else x + 0.26
    return cx0, x + w - 0.20 - cx0


def _fit_big(P: _Page, metrics: list[dict[str, Any]], area_w: float, big: float) -> float:
    avail = area_w - METRIC_GAP * (len(metrics) - 1)
    while big > 24 and sum(_value_need(P, mt, big) for mt in metrics) > avail:
        big -= 0.5
    return big


def _metric(P: _Page, x: float, y0: float, w: float, h: float, met: dict[str, Any], big: float,
            dsize: float = DETAIL_SIZE) -> None:
    top = y0 + h - 0.60
    r = 0.16
    ICONS[met["icon"]](P.ax, x + r, top - 0.19, r, met["icolor"])
    lx = x + 2 * r + 0.12
    P.text(lx, top - 0.25, met["label"], P.fit(met["label"], x + w - lx, 10.5, "bold"), MUTED, "bold")
    vy = y0 + 0.71
    P.text(x, vy, met["value"], big, met["color"], "bold")
    if met["unit"]:
        P.text(x + P.tw(met["value"], big, "bold") + 0.08, vy, met["unit"], _unit_size(big), INK, "bold")
    for i, d in enumerate(met["details"][:2]):
        P.text(x, y0 + 0.42 - i * 0.235, d, P.fit(d, w, dsize), INK if i == 0 else MUTED)


def _hero_card(P: _Page, m: _Mode, x: float, y: float, w: float, h: float, metrics: list[dict[str, Any]],
               bm: Basemap | None, big: float, dsize: float | None = None) -> None:
    P.box(x, y, w, h)
    if bm is not None:
        _map_thumb(P, m, x + 0.14, y + 0.14, h - 0.28, bm)
    cx0, area_w = _metric_area(x, w, h, bm is not None)
    ty = y + h - 0.42
    cw = P.chip(cx0, ty, f"{m.name} MODE", m.color)
    title = f"{MODE_TITLE.get(m.mode, m.mode)} · {m.k['dose_kg_per_m2']:g} kg/m²"
    tx = cx0 + cw + 0.14
    P.text(tx, ty, title, 14, INK, "bold")
    facts = f"{_n(m.s['material_t'])} t · {_n(m.k['mission_time_h'])} h · {_n(m.k['sorties'])} sorties"
    avail = x + w - 0.24 - (tx + P.tw(title, 14, "bold") + 0.3)
    P.text(x + w - 0.24, ty, facts, P.fit(facts, avail, 12), MUTED, ha="right")
    widths, dsize = _metric_widths(P, metrics, area_w, big, dsize)
    xx = cx0
    for mt, mw in zip(metrics, widths):
        _metric(P, xx, y, mw, h, mt, big, dsize)
        xx += mw + METRIC_GAP


def _hero(P: _Page, modes: list[_Mode], bm: Basemap | None) -> None:
    if len(modes) == 2:
        w = (W - 2 * M - GAP) / 2
        area_w = _metric_area(M, w, HERO_H, bm is not None)[1]
        cards = [(m, M + i * (w + GAP), _hero_metrics(m, False)) for i, m in enumerate(modes)]
        big = min(_fit_big(P, mets, area_w, 40) for _, _, mets in cards)
        dsize = min(_metric_widths(P, mets, area_w, big)[1] for _, _, mets in cards)
        for m, cx, mets in cards:
            _hero_card(P, m, cx, HERO_Y, w, HERO_H, mets, bm, big, dsize)
    else:
        w = W - 2 * M
        mets = _hero_metrics(modes[0], True)
        big = _fit_big(P, mets, _metric_area(M, w, HERO_H, bm is not None)[1], 46)
        _hero_card(P, modes[0], M, HERO_Y, w, HERO_H, mets, bm, big)


# ---------------------------------------------------------------------------
# middle panels
# ---------------------------------------------------------------------------


def _fire_vs_field(P: _Page, modes: list[_Mode], x: float, y: float, w: float, h: float) -> None:
    two = len(modes) == 2
    P.panel(x, y, w, h, "FIRE VS FIELD", "the same prunings, two fates",
            [(m.name, m.color) for m in modes] if two else None)
    S = [m.s for m in modes]
    gut = 0.40
    cw = (w - 0.48 - gut) / 2
    lx, rx = x + 0.24, x + 0.24 + cw + gut
    hy = y + h - 0.84
    icon_flame(P.ax, lx + 0.14, hy + 0.08, 0.15, FIRE)
    P.text(lx + 0.40, hy, "IF PILE-BURNED", 13, FIRE, "bold")
    dm = " / ".join(_n(s["burn_avoided_dm_t"]) for s in S)
    sub = f"{dm} t of dry matter would combust"
    P.text(lx + 0.40, hy - 0.22, sub, P.fit(sub, cw - 0.40, 10), MUTED)
    icon_leaf(P.ax, rx + 0.15, hy + 0.08, 0.15, SOIL_C)
    P.text(rx + 0.40, hy, "SPREAD ON THE BEDS", 13, GREEN, "bold")
    doses = " / ".join(f"{m.k['dose_kg_per_m2']:g}" for m in modes)
    sub = f"{_n(S[0]['area_ha'], 2)} ha at {doses} kg/m² · SOC in the top 20 cm"
    P.text(rx + 0.40, hy - 0.22, sub, P.fit(sub, cw - 0.40, 10), MUTED)
    rule_y = y + 1.30
    P.line([x + 0.24 + cw + gut / 2] * 2, [rule_y + 0.16, hy + 0.22], EDGE, 1.0)

    fire = [
        ("CH₄", "methane", _mass_row([s["burn_ch4_avoided_kg"] for s in S])),
        ("N₂O", "nitrous oxide", _mass_row([s["burn_n2o_avoided_kg"] for s in S])),
        ("CO", "carbon monoxide", _mass_row([s["co_avoided_kg"] for s in S])),
        ("NOx", "nitrogen oxides", _mass_row([s["nox_avoided_kg"] for s in S])),
        ("PM2.5", "fine smoke", _mass_row([s["pm25_avoided_kg"] for s in S])),
    ]
    field_rows = [
        ("CO₂e", "soil carbon", [f"{_n(s['soil_co2e_t'], 1)} t" for s in S]),
        ("N", "nitrogen", [f"{_n(s['n_retained_kg'])} kg" for s in S]),
        ("SOC", "organic carbon", [f"+{_auto(s['soc_increase_pp'])} pp" for s in S]),
        ("H₂O", "available water", [f"+{_auto(s['awc_gain_mm'])} mm" for s in S]),
    ]
    vsize, nsize = (11.0, 9.8) if two else (12.0, 10.5)
    pitch = 0.272
    ry = hy - 0.56
    chem_w = max(P.tw(r[0], vsize + 0.5, "bold") for r in fire + field_rows) + 0.14
    for cx, rows, col in ((lx, fire, FIRE_LT), (rx, field_rows, GREEN)):
        vals = [v for r in rows for v in r[2]]
        vw = max(P.tw(v, vsize, "bold") for v in vals) + 0.14
        rights = [cx + cw - vw * (len(modes) - 1 - i) for i in range(len(modes))]
        name_x = cx + chem_w
        name_w = rights[0] - vw + 0.10 - name_x
        for i, (chem, name, vs) in enumerate(rows):
            yy = ry - i * pitch
            P.text(cx, yy, chem, vsize + 0.5, col, "bold")
            P.text(name_x, yy, name, P.fit(name, name_w, nsize), MUTED)
            for v, xr, m in zip(vs, rights, modes):
                P.text(xr, yy, v, vsize, m.color if two else col, "bold", ha="right")
    bio = " / ".join(_mass_row([s["biogenic_co2_not_released_t"] * 1000 for s in S]))
    note = f"+ {bio} of biogenic CO₂: context only, not counted"
    P.text(lx, ry - 5 * pitch, note, P.fit(note, cw, 9.5, style="italic"), MUTED, style="italic")
    water = " / ".join(f"{_auto(s['awc_gain_m3'])} m³" for s in S)
    note = f"water: small but real, ≈ {water} more held on {_n(S[0]['area_ha'], 2)} ha"
    P.text(rx, ry - 4 * pitch, note, P.fit(note, cw, 9.5, style="italic"), MUTED, style="italic")

    P.line([x + 0.24, x + w - 0.24], [rule_y, rule_y], EDGE, 0.8)
    by = rule_y - 0.28
    P.text(x + 0.24, by, "COUNTED CLIMATE BENEFIT, t CO₂e", 10.5, INK, "bold")
    P.legend(x + w - 0.24, by, [("soil carbon", SOIL_C), ("avoided CH₄ + N₂O", AMBER)], size=9.5)
    tots = [s["total_co2e_t"] for s in S]
    bx0 = x + (1.45 if two else 0.24)
    label_room = 2.30 if two else 2.75
    sc = (x + w - 0.24 - label_room - bx0) / max(max(tots), 1e-9)
    bh = 0.24 if two else 0.36
    for i, (m, s) in enumerate(zip(modes, S)):
        yy = by - 0.52 - i * 0.36 if two else by - 0.62
        if two:
            P.text(x + 0.24, yy + bh / 2, m.name, 10.5, m.color, "bold", va="center")
        soil, ghg = s["soil_co2e_t"] * sc, s["burn_ghg_avoided_co2e_t"] * sc
        P.rect(bx0, yy, soil, bh, SOIL_C)
        P.rect(bx0 + soil, yy, ghg, bh, AMBER)
        end = bx0 + soil + ghg + 0.10
        if two:
            P.runs(end, yy + 0.05, [(f"{_n(s['total_co2e_t'], 1)} t", 12.5, INK, "bold", 0.10),
                                    (f"= {_n(s['soil_co2e_t'], 1)} + {_n(s['burn_ghg_avoided_co2e_t'], 1)}", 10,
                                     MUTED, "normal", 0)])
            continue
        lbl = f"soil carbon {_n(s['soil_co2e_t'], 1)}"
        if P.tw(lbl, 10, "bold") < soil - 0.2:
            P.text(bx0 + 0.1, yy + bh / 2, lbl, 10, PANEL, "bold", va="center")
        v = _n(s["burn_ghg_avoided_co2e_t"], 1)
        if P.tw(v, 9.5, "bold") < ghg - 0.08:
            P.text(bx0 + soil + ghg / 2, yy + bh / 2, v, 9.5, PANEL, "bold", ha="center", va="center")
        P.runs(end, yy + bh / 2 + 0.02, [(f"{_n(s['total_co2e_t'], 1)} t", 15, INK, "bold", 0.10),
                                          (f"{_n(s['co2e_t_per_ha'], 1)} t per ha", 11, MUTED, "normal", 0)])
        ch4 = s["burn_ch4_avoided_kg"] / 1000 * FACTORS["gwp_ch4"].value
        n2o = s["burn_n2o_avoided_kg"] / 1000 * FACTORS["gwp_n2o"].value
        gwp = f"avoided = CH₄ {_n(ch4, 1)} + N₂O {_n(n2o, 1)} (GWP100)"
        P.text(end, yy + bh / 2 - 0.22, gwp, P.fit(gwp, x + w - 0.24 - end, 9.5), MUTED)


def _cost(P: _Page, modes: list[_Mode], x: float, y: float, w: float, h: float) -> None:
    dust, cart = _find(modes, "dust"), _find(modes, "cartridge")
    P.panel(x, y, w, h, "COST", "€ per 1000 m² · REGENSOL scope")
    if dust is None and cart is not None:
        _cost_donut(P, cart, x, y, w, h)
        return
    s, k = dust.s, dust.k
    reg, conv = s["regensol_eur_per_1000m2"], s["conventional_eur_per_1000m2"]
    repl = TRANSPORT_SPREAD_SHARE * reg
    ops, tot = float(k["robot_ops_eur_per_1000m2"]), s["cost_eur_per_1000m2"]
    both = cart is not None
    label_w = 1.72
    bx0 = x + 0.24 + label_w
    sc = (x + w - 0.24 - 0.82 - bx0) / reg

    def X(v: float) -> float:
        return bx0 + v * sc

    pitch = 0.47 if both else 0.54
    bh = 0.26 if both else 0.30
    top = y + h - (1.06 if both else 1.16)
    rows = [
        ("REGENSOL", "regenerative baseline", [(0, conv, "#55654f", None), (conv, reg, "#7a8c70", None)],
         _n(reg, 2), INK),
        ("− transport + spreading", f"{TRANSPORT_SPREAD_SHARE * 100:.1f} % replaced by the fleet",
         [(reg - repl, reg, (1, 0.30, 0.24, 0.16), "///")], f"−{_n(repl, 2)}", WEED),
        ("+ robot operations", "mothership, drones, supervisor", [(reg - repl, reg - repl + ops, MS, None)],
         f"+{_n(ops, 2)}", MS),
        ("PruneMaxxing", f"{dust.name} mode", [(0, tot, CHIP, None)], _n(tot, 2), CHIP),
    ]
    for i, (l1, l2, segs, val, vcol) in enumerate(rows):
        yy = top - i * pitch
        P.text(x + 0.24, yy + bh - 0.09, l1, P.fit(l1, label_w - 0.08, 11, "bold"), INK, "bold")
        P.text(x + 0.24, yy - 0.06, l2, P.fit(l2, label_w - 0.08, 8.8), MUTED)
        for a, b, c, hatch in segs:
            kw = dict(hatch=hatch, ec=WEED, lw=0.8, hatchcolor=WEED) if hatch else {}
            P.rect(X(a), yy, (b - a) * sc, bh, c, **kw)
        P.text(X(max(sg[1] for sg in segs)) + 0.07, yy + bh / 2, val, 10.5, vcol, "bold", va="center")
    y_reg = top
    if P.tw("conventional", 8.5) < conv * sc - 0.15:
        P.text(X(0) + 0.07, y_reg + bh / 2, "conventional", 8.5, INK, va="center", z=6)
    over = REGENSOL_OVERHEAD_EUR_PER_1000M2
    if P.tw(f"+{_n(over)}", 8.5) < over * sc - 0.06:
        P.text(X(conv + over / 2), y_reg + bh / 2, f"+{_n(over)}", 8.5, INK, ha="center", va="center", z=6)
    for i, xv in ((0, reg), (1, reg - repl), (2, tot)):
        y_a, y_b = top - i * pitch, top - (i + 1) * pitch + bh
        P.line([X(xv)] * 2, [y_a, y_b], MUTED, 0.7, ls=(0, (1.5, 1.5)))
    P.line([X(conv)] * 2, [top - 3 * pitch - 0.05, top + bh + 0.05], INK, 0.7, ls=(0, (4, 2)), alpha=0.6, z=5)
    P.text(X(conv), top - 3 * pitch - 0.19, f"conventional {_n(conv, 2)}", 8.5, MUTED, ha="center")

    sy = top - 3 * pitch - (0.50 if both else 0.62)
    pct_conv = 100.0 * (tot - conv) / conv
    saving = s.get("saving_eur_per_1000m2") or 0.0
    total = s.get("saving_eur_total") or 0.0
    if both:
        P.runs(x + 0.24, sy, [("saves", 11, INK, "normal", 0.08), (f"{_n(saving, 2)} €/1000 m²", 11, GREEN, "bold", 0.08),
                              (f"→ €{_n(total)} on {_n(s['area_ha'], 2)} ha", 11, GREEN, "bold", 0)])
        info = (f"{_n(k.get('overhead_recovered_pct', 0))} % of the regenerative overhead recovered · "
                f"{pct_conv:+.1f} % vs conventional")
        P.text(x + 0.24, sy - 0.22, info, P.fit(info, w - 0.48, 9.5), MUTED)
        cy = y + 0.36
        P.line([x + 0.24, x + w - 0.24], [y + 0.80, y + 0.80], EDGE, 0.8)
        cs, ck = cart.s, cart.k
        mat = float(ck.get("cartridge_material_eur_per_1000m2", 0.0))
        ctot = cs["cost_eur_per_1000m2"]
        P.text(x + 0.24, cy + bh - 0.09, cart.name, 11, CART, "bold")
        P.text(x + 0.24, cy - 0.06, "no REGENSOL baseline", 8.8, MUTED)
        P.rect(X(0), cy, mat * sc, bh, CART)
        P.rect(X(mat), cy, (ctot - mat) * sc, bh, MS)
        P.text(X(ctot) + 0.07, cy + bh / 2, _n(ctot, 2), 10.5, CART, "bold", va="center")
        note = f"material {_n(mat, 2)} + robot ops {_n(float(ck['robot_ops_eur_per_1000m2']), 2)}"
        P.text(X(0), cy - 0.17, note, P.fit(note, x + w - 0.24 - X(0), 8.8), MUTED)
    else:
        P.runs(x + 0.24, sy, [(f"€{_n(total)}", 24, GREEN, "bold", 0.12),
                              (f"saved on {_n(s['area_ha'], 2)} ha", 13, INK, "normal", 0)])
        P.text(x + 0.24, sy - 0.32, f"{_n(saving, 2)} €/1000 m² below the REGENSOL regenerative baseline", 10.5, INK)
        info = (f"{_n(k.get('overhead_recovered_pct', 0))} % of the regenerative overhead recovered · "
                f"only {pct_conv:+.1f} % vs conventional")
        P.text(x + 0.24, sy - 0.56, info, P.fit(info, w - 0.48, 10.5), MUTED)


def _cost_donut(P: _Page, m: _Mode, x: float, y: float, w: float, h: float) -> None:
    s, k = m.s, m.k
    tot = s["cost_eur_per_1000m2"]
    mat = float(k.get("cartridge_material_eur_per_1000m2", 0.0))
    ops = float(k["robot_ops_eur_per_1000m2"])
    R = min(1.0, (h - 1.3) / 2)
    cx, cy = x + 0.30 + R, y + h - 0.85 - R
    a = 360.0 * mat / max(tot, 1e-9)
    P.ax.add_patch(Wedge((cx, cy), R, 90 - a, 90, width=0.30, fc=CART, ec=PANEL, lw=1.5, zorder=4))
    P.ax.add_patch(Wedge((cx, cy), R, 90 - 360, 90 - a, width=0.30, fc=MS, ec=PANEL, lw=1.5, zorder=4))
    P.text(cx, cy + 0.02, _n(tot, 2), 19, INK, "bold", ha="center")
    P.text(cx, cy - 0.24, "€ / 1000 m²", 9.5, MUTED, ha="center")
    lx = cx + R + 0.32
    per = mat / max(_cartridges_per_m2(m) * 1000.0, 1e-9)
    rows = [(CART, "cartridge material", mat, ["pressed chips + seed + mycorrhizal", f"inoculum · ≈ €{per:.2f} each"]),
            (MS, "robot operations", ops, ["mothership, drones, air cannons,", "supervisor, tender trucks"])]
    yy = y + h - 1.05
    rw = x + w - 0.24 - lx
    for col, name, val, sub in rows:
        P.rect(lx, yy - 0.01, 0.15, 0.15, col, z=5)
        P.text(lx + 0.24, yy, name, P.fit(name, rw - 0.95, 11, "bold"), INK, "bold")
        P.text(x + w - 0.24, yy, _n(val, 2), 11, col, "bold", ha="right")
        for j, ln in enumerate(sub):
            P.text(lx + 0.24, yy - 0.22 - j * 0.19, ln, P.fit(ln, rw - 0.24, 9), MUTED)
        yy -= 0.90
    P.runs(x + 0.24, y + 0.62, [(f"≈ €{_n(tot * s['area_ha'] * 10.0)}", 18, CART, "bold", 0.12),
                                (f"for {_n(s['area_ha'], 2)} ha", 12, INK, "normal", 0)])
    P.text(x + 0.24, y + 0.30, "No REGENSOL baseline: a different job (reforestation, not mulching).", 9,
           MUTED, style="italic")


def _reforestation(P: _Page, m: _Mode, x: float, y: float, w: float, h: float) -> None:
    s, k, spec = m.s, m.k, m.res.spec
    P.panel(x, y, w, h, "REFORESTATION", "seed + mycelium in pressed prunings")
    art_s = 0.60
    acx, acy = x + 0.24 + art_s + 0.05, y + h - 1.42
    _cartridge_art(P.ax, acx, acy, art_s)
    nx = acx + art_s + 0.34
    cart_n, seeds = s.get("cartridges_placed") or 0, s.get("seeds_placed") or 0
    nw = x + w - 0.24 - nx
    for yy, val, unit in ((y + h - 1.02, _n(cart_n), "cartridges"), (y + h - 1.52, _n(seeds), "tree seeds")):
        parts = [(val, 26, CART, "bold", 0.08), (unit, 12, INK, "normal", 0)]
        if P.runs_width(parts) > nw:
            f = nw / P.runs_width(parts)
            parts = [(val, 26 * f, CART, "bold", 0.08), (unit, 12 * f, INK, "normal", 0)]
        P.runs(nx, yy, parts)
    unit_kg = spec.unit_mass_kg / max(spec.units_per_sortie, 1)
    for i, ln in enumerate([f"{spec.seeds_per_unit} seeds + mycorrhizal mycelium in {unit_kg:g} kg of chips",
                            f"{_cartridges_per_m2(m):g} cartridges per m² · {k['dose_kg_per_m2']:g} kg/m²"]):
        P.text(nx, y + h - 1.82 - i * 0.21, ln, P.fit(ln, nw, 9.5), MUTED)

    sp = _species(m)
    hy = y + 1.98
    P.text(x + 0.24, hy, "LOCAL TREE SPECIES", 10, MUTED, "bold")
    n = max(len(sp), 1)
    colw = (w - 0.48) / n
    for i, (common, latin, kind) in enumerate(sp):
        cxi = x + 0.24 + (i + 0.5) * colw
        icon_tree(P.ax, cxi, hy - 0.50, 0.29, kind)
        P.text(cxi, hy - 0.98, common, P.fit(common, colw - 0.08, 10.5, "bold"), INK, "bold", ha="center")
        P.text(cxi, hy - 1.17, latin, P.fit(latin, colw - 0.08, 9, style="italic"), MUTED, ha="center",
               style="italic")

    share = float(k.get("ms_effector_share_pct", 0.0))
    by = y + 0.22
    bw = w - 0.48
    P.rect(x + 0.24, by, bw * (1 - share / 100), 0.16, CART)
    P.rect(x + 0.24 + bw * (1 - share / 100), by, bw * share / 100, 0.16, FXC)
    P.text(x + 0.24, by + 0.25, f"{k['swarm_size']} drones · {_n(100 - share, 1)} %", 9.5, INK)
    right = f"twin air cannons on the mothership · {_n(share, 1)} % · {_ft(k['ms_effector_reach_m'])} reach"
    P.text(x + w - 0.24, by + 0.25, right, P.fit(right, bw * 0.66, 9.5), FXC, ha="right")


def _timeline(P: _Page, m: _Mode, x: float, y: float, w: float, h: float) -> None:
    k, tl = m.k, m.tl
    P.panel(x, y, w, h, f"{_n(k['mission_time_h'])} HOURS IN THE FIELD", "tonnes applied over mission time")
    total = tl.applied_kg[-1] / 1000.0
    ms_t = tl.ms_kg[-1] / 1000.0
    ly = y + h - 0.78
    P.legend(x + w - 0.24, ly, [(f"mothership {m.effector} {_n(ms_t)} t ({_n(k['ms_effector_share_pct'], 1)} %)",
                                 FXC)], size=9.5)
    P.rect(x + 0.24, ly - 0.005, 0.13, 0.13, m.color, z=5)
    P.text(x + 0.43, ly, f"all material {_n(total)} t", 9.5, MUTED)
    ax = P.fig.add_axes(((x + 0.78) / W, (y + 0.70) / H, (w - 1.06) / W, (h - 1.70) / H))
    ax.set_facecolor(PANEL)
    th = tl.t / 3600.0
    ax.fill_between(th, 0, tl.applied_kg / 1000.0, color=m.color, alpha=0.22, lw=0)
    ax.plot(th, tl.applied_kg / 1000.0, color=m.color, lw=2.0)
    ax.fill_between(th, 0, tl.ms_kg / 1000.0, color=FXC, alpha=0.35, lw=0)
    ax.plot(th, tl.ms_kg / 1000.0, color=FXC, lw=1.3)
    ax.set_xlim(0, th[-1])
    ax.set_ylim(0, max(total, 1e-6) * 1.16)
    for ob in k.get("obstacles_detected", []):
        t_h = float(ob["t_h"])
        ax.axvline(t_h, color=OBST, lw=0.8, ls=(0, (2, 2)), alpha=0.9)
        ax.text(t_h, total * 1.12, ob.get("obstacle_type", "obstacle"), color=OBST, fontsize=8.5, ha="center",
                va="top", family=SANS, bbox=dict(boxstyle="round,pad=0.15", fc=PANEL, ec="none"))
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(EDGE)
    ax.tick_params(colors=MUTED, labelsize=9.5, length=3)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g} h"))
    ax.set_ylabel("t applied", color=MUTED, fontsize=9.5, family=SANS)
    for lb in ax.get_xticklabels() + ax.get_yticklabels():
        lb.set_family(SANS)
    P.text(x + 0.24, y + 0.18, "dotted: unknown obstacle found by lidar and bypassed", 8.8, MUTED, style="italic")


def _middle(P: _Page, modes: list[_Mode]) -> None:
    dust, cart = _find(modes, "dust"), _find(modes, "cartridge")
    third: Callable = ((lambda P_, x, y, w, h: _reforestation(P_, cart, x, y, w, h)) if cart is not None
                       else (lambda P_, x, y, w, h: _timeline(P_, modes[0], x, y, w, h)))
    if len(modes) == 2:
        fr = [0.435, 0.295, 0.270]
    elif dust is not None:
        fr = [0.400, 0.305, 0.295]
    else:
        fr = [0.400, 0.280, 0.320]
    panels = [lambda P_, x, y, w, h: _fire_vs_field(P_, modes, x, y, w, h),
              lambda P_, x, y, w, h: _cost(P_, modes, x, y, w, h), third]
    avail = W - 2 * M - GAP * (len(panels) - 1)
    x = M
    for f, fn in zip(fr, panels):
        w = avail * f
        fn(P, x, MID_Y, w, MID_H)
        x += w + GAP


# ---------------------------------------------------------------------------
# fleet strip + footnotes
# ---------------------------------------------------------------------------


def _fleet_tiles(modes: list[_Mode]) -> list[dict[str, Any]]:
    two = len(modes) == 2

    def vals(fn: Callable[[_Mode], float], d: int = 0, shared: bool = False) -> list[tuple[str, str]]:
        v = [_n(fn(m), d) for m in modes]
        if shared and len(set(v)) == 1:
            return [(v[0], INK)]
        return [(vi, m.color if two else INK) for vi, m in zip(v, modes)]

    def join(fn: Callable[[_Mode], str]) -> str:
        return " / ".join(fn(m) for m in modes)

    ft = _ft(modes[0].k["ms_effector_reach_m"])
    n_obs = vals(lambda m: len(m.k.get("obstacles_detected", [])), shared=True)
    tiles = [
        dict(label="MISSION TIME", vals=vals(lambda m: m.k["mission_time_h"]), unit="h",
             sub=(f"per mission on {_n(modes[0].s['area_ha'], 2)} ha" if two
                  else f"{modes[0].k['mission_days_10h_shifts']:g} × 10 h shifts")),
        dict(label="DRONE SORTIES", vals=vals(lambda m: m.k["sorties"]), unit="",
             sub=(join(lambda m: f"{m.res.spec.unit_mass_kg:g} kg") + " per sortie" if two
                  else f"{_n(modes[0].k['drone_flight_km'])} km flown · {_n(modes[0].k['battery_swaps'])} swaps")),
        dict(label="DRONE ENERGY", vals=vals(lambda m: m.s["drone_energy_kwh"]), unit="kWh",
             sub=join(lambda m: _n(m.s["drone_energy_kwh_per_t"], 2)) + " kWh per tonne"),
        dict(label="MOTHERSHIP EFFECTOR" if two else f"MOTHERSHIP {modes[0].effector.upper()}",
             vals=vals(lambda m: m.k["ms_effector_share_pct"], 1), unit="%",
             sub=(join(lambda m: m.effector.split()[-1]) + f" · {ft} reach" if two
                  else (f"{_n(modes[0].k['ms_effector_units'])} cartridges" if modes[0].mode == "cartridge"
                        else f"{_n(modes[0].k['ms_effector_t'])} t") + f" within {ft}")),
        dict(label="BED COMPACTION", vals=vals(lambda m: m.s["bed_compaction_pct"], 2, shared=True), unit="%",
             sub="of area · wheels stay in the furrows"),
        dict(label="UNKNOWN OBSTACLES", vals=n_obs, unit="", sub="found by lidar and bypassed"),
    ]
    if not two and modes[0].mode == "dust":
        k = modes[0].k
        tiles.append(dict(label="FLEET THROUGHPUT", vals=[(_n(k["swarm_throughput_t_per_h"], 1), INK)], unit="t/h",
                          sub=f"drone utilisation {_n(k['drone_utilisation_pct'])} %"))
    return tiles


def _sparkline(P: _Page, modes: list[_Mode], x: float, y: float, w: float, h: float) -> None:
    two = len(modes) == 2
    P.text(x, y + h - 0.02, "APPLIED OVER TIME" + (" · % of target" if two else ", t"), 9.5, MUTED, "bold")
    gx, gy, gw, gh = x, y + 0.04, w - 0.55, h - 0.36
    t_max = max(m.tl.t[-1] for m in modes)
    P.line([gx, gx + gw], [gy, gy], EDGE, 0.8)
    for m in modes:
        tl = m.tl
        xs = gx + tl.t / max(t_max, 1e-9) * gw
        if two:
            ys = gy + tl.applied_kg / max(tl.applied_kg[-1], 1e-9) * gh
        else:
            top = max(tl.applied_kg[-1], 1e-9)
            ys = gy + tl.applied_kg / top * gh
            P.ax.fill_between(xs, gy, gy + tl.ms_kg / top * gh, color=FXC, alpha=0.4, lw=0, zorder=4)
        P.ax.fill_between(xs, gy, ys, color=m.color, alpha=0.14, lw=0, zorder=3.5)
        P.line(xs, ys, m.color, 1.6, z=4.5)
        if two:
            P.text(xs[-1] + 0.06, ys[-1] - 0.04, f"{_n(m.k['mission_time_h'])} h", 9.5, m.color, "bold")
        else:
            P.text(xs[-1] + 0.06, ys[-1] - 0.04, f"{_n(tl.applied_kg[-1] / 1000)} t", 9.5, m.color, "bold",
                   va="top")
            P.text(gx + gw, gy - 0.05, f"{_n(m.k['mission_time_h'])} h", 8.5, MUTED, ha="right", va="top")


def _fleet(P: _Page, modes: list[_Mode]) -> None:
    x, y, w, h = M, FLEET_Y, W - 2 * M, FLEET_H
    P.box(x, y, w, h)
    icon_drone(P.ax, x + 0.44, y + h / 2, 0.26, INK)
    P.text(x + 0.86, y + h - 0.40, "ONE FLEET", 14, AMBER, "bold")
    fleet = f"1 mothership UGV + {modes[0].k['swarm_size']} wireless heavy-lift drones, electric"
    P.para(x + 0.86, y + h - 0.64, fleet, 2.05, 9.5, MUTED)
    tiles = _fleet_tiles(modes)
    spark = not (len(modes) == 1 and modes[0].mode == "dust")
    spark_w = 2.75 if spark else 0.0
    x_t = x + 3.15
    tile_w = (x + w - 0.22 - spark_w - x_t) / len(tiles)
    for i, t in enumerate(tiles):
        tx = x_t + i * tile_w
        P.line([tx - 0.14] * 2, [y + 0.18, y + h - 0.18], EDGE, 0.8)
        mw = tile_w - 0.24
        P.text(tx, y + h - 0.32, t["label"], P.fit(t["label"], mw, 9.5, "bold"), MUTED, "bold")
        parts: list[tuple] = []
        for j, (v, c) in enumerate(t["vals"]):
            if j:
                parts.append(("/", 14, MUTED, "normal", 0.07))
            parts.append((v, 19, c, "bold", 0.07))
        if t["unit"]:
            parts.append((t["unit"], 12, INK, "bold", 0))
        pw = P.runs_width(parts)
        if pw > mw:
            parts = [(p[0], p[1] * mw / pw, p[2], p[3], p[4]) for p in parts]
        P.runs(tx, y + h - 0.70, parts)
        P.text(tx, y + 0.18, t["sub"], P.fit(t["sub"], mw, 9.5), MUTED)
    if spark:
        sx = x + w - 0.22 - spark_w + 0.06
        P.line([sx - 0.20] * 2, [y + 0.18, y + h - 0.18], EDGE, 0.8)
        _sparkline(P, modes, sx, y + 0.18, spark_w - 0.06, h - 0.48)


def _common_words(seqs: list[list[str]], suffix: bool) -> list[str]:
    out: list[str] = []
    if not seqs:
        return out
    for i in range(min(len(s) for s in seqs) - 1):
        idx = -1 - i if suffix else i
        word = seqs[0][idx]
        if all(s[idx] == word for s in seqs):
            out.append(word)
        else:
            break
    return out[::-1] if suffix else out


def _fv(v: float) -> str:
    return f"{v:,g}"


def _fu(unit: str) -> str:
    return "" if unit.strip() in ("", "-") else f" {_pretty(unit)}"


def factor_lines() -> list[str]:
    """Footnote lines generated from :func:`factor_table` (grouped by source, † = assumption)."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in factor_table():
        groups.setdefault(r["source"], []).append(r)
    out = []
    for src, rs in groups.items():
        dag = " †" if any(r["assumed"] for r in rs) else ""
        if len(rs) == 1:
            r = rs[0]
            out.append(f"{_cap(_pretty(r['label']))}: {_fv(r['value'])}{_fu(r['unit'])}{dag} — {src}")
            continue
        labels = [_pretty(r["label"]) for r in rs]
        core = [re.sub(r"\s*\(.*?\)", "", lb).split() for lb in labels]
        paren = [re.findall(r"\(.*?\)", lb) for lb in labels]
        pre, suf = _common_words(core, False), _common_words(core, True)
        units = {r["unit"] for r in rs}
        items = []
        for r, words, par in zip(rs, core, paren):
            short = " ".join(words[len(pre): len(words) - len(suf)])
            item = f"{short} {_fv(r['value'])}" + ("" if len(units) == 1 else _fu(r["unit"]))
            items.append(item + (f" {par[0]}" if par else ""))
        head = re.sub(r"\s+(of|at|from)$", "", " ".join(pre)) if pre else " ".join(suf)
        unit = f" ({_fu(next(iter(units))).strip()})" if len(units) == 1 and _fu(next(iter(units))) else ""
        lead = f"{_cap(head)}{unit}: " if head else ""
        out.append(f"{lead}{' · '.join(items)}{dag} — {src}")
    eq = EQUIVALENCES["spain_ghg_per_capita"]
    out.append(f"¹ Everyday equivalence: {_fv(eq['value'])}{_fu(eq['unit'])} ({eq['label']}) — {eq['source']}")
    return out


def _footnotes(P: _Page, modes: list[_Mode]) -> None:
    x, y, w, h = M, FOOT_Y, W - 2 * M, FOOT_H
    top = y + h - 0.12
    size, lh = 7.9, 0.128
    col1 = 5.55
    P.text(x, top, "METHOD", 8.8, AMBER, "bold")
    method = ("Counted climate benefit = soil carbon stored (REGENSOL trial rate, scaled by dose) + the CH₄ and N₂O "
              "that pile burning would have emitted (IPCC Tier 1 factors × AR6 GWP100). Burned dry matter = fresh "
              "tonnes × dry-matter fraction × combustion fraction. Biogenic CO₂ from burning is shown for context "
              "only and never added to the CO₂e total (no double counting); tree growth from cartridges is not "
              "counted. Costs are € per 1000 m² in the REGENSOL scope; robot operations come from the simulated "
              "mission log.")
    if len(modes) == 2:
        method += (f" The two modes are alternative missions on the same {_n(modes[0].s['area_ha'], 2)} ha: "
                   f"figures are per mission, not summed.")
    lines = P.wrap(method, col1, size)
    for i, ln in enumerate(lines):
        P.text(x, top - 0.19 - i * lh, ln, size, MUTED)
    fx0 = x + col1 + 0.35
    P.text(fx0, top, "FACTORS & SOURCES", 8.8, AMBER, "bold")
    P.text(fx0 + P.tw("FACTORS & SOURCES", 8.8, "bold") + 0.14, top, "generated from factor_table() · † assumed value",
           8.3, MUTED)
    colw = (x + w - fx0 - 0.30) / 2
    wrapped = [P.wrap(fl, colw, size) for fl in factor_lines()]
    total = sum(len(wl) for wl in wrapped)
    col, row = 0, 0
    for wl in wrapped:
        if col == 0 and row > 0 and row + len(wl) > math.ceil(total / 2):
            col, row = 1, 0
        for j, ln in enumerate(wl):
            P.text(fx0 + col * (colw + 0.30) + (0.12 if j else 0.0), top - 0.19 - row * lh, ln, size, MUTED)
            row += 1


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def render_impact_poster(results, out_path, dpi: int = 200, basemap: Basemap | None = None, *,
                         with_map: bool = True) -> Path:
    """Render the climate impact poster for one MissionResult or a sequence of up to two (dust / cartridge).

    Two results give the combined "two modes, one fleet" page, one result a single-mode page.
    ``basemap`` is reused for the coverage thumbnails (built from the site when None);
    ``with_map=False`` skips the thumbnails.
    """
    modes = _modes(results)
    out_path = Path(out_path)
    bm = (basemap if basemap is not None else build_basemap(modes[0].res.site)) if with_map else None
    P = _Page(dpi)
    try:
        _header(P, modes)
        _hero(P, modes, bm)
        _middle(P, modes)
        _fleet(P, modes)
        _footnotes(P, modes)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        P.fig.savefig(out_path, dpi=dpi, facecolor=BG)
    finally:
        plt.close(P.fig)
    return out_path
