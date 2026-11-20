"""Animated motion-graphics infographic of a finished mission (16:9 MP4, English).

``render_infographic(res, out_path)`` tells the mission as a short story: an intro
card with the target, the whole mission compressed against a mission clock
(counters, material flow, work split, parcels, drone swimlanes, event log) and an
outro with the impact totals and their sources. Every number comes from
:class:`~prunemaxxing.viz.mission_stats.MissionTimeline` and
:mod:`prunemaxxing.kpis.impact`.

All artists live on one full-canvas axes in 1920 x 1080 design units (y down) and
are created once; frames only update data, text and alpha in place.
"""

from __future__ import annotations

import math
import textwrap
from pathlib import Path
from typing import Any, Callable

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import to_rgb, to_rgba  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle, Wedge  # noqa: E402
from matplotlib.transforms import Affine2D  # noqa: E402

from prunemaxxing.kpis.impact import factor_table, impact_from_material  # noqa: E402
from prunemaxxing.viz.mission_stats import DRONE_STATE_COLORS, DRONE_STATES, MissionTimeline  # noqa: E402
from prunemaxxing.viz.mission_viz import ACTUAL, BG, BYPASS, CHIP, FXC, OBST, PANEL, WEED  # noqa: E402
from prunemaxxing.viz.style import COLORS, DRONE_PATH_COLORS  # noqa: E402
from prunemaxxing.viz.video_io import video_writer  # noqa: E402

__all__ = ["render_infographic"]

W, H = 1920.0, 1080.0
M = 56.0
SANS, MONO = "DejaVu Sans", "DejaVu Sans Mono"
INK, MUTED, AMBER = COLORS["ink"], COLORS["muted"], COLORS["amber"]
EDGE = "#2c3a2d"
TRACK = "#1c261d"
ROW = "#172118"
DIM = "#4e5c4b"
GREEN = "#7bdc6a"
SKY = "#8fb8ff"
DRONES = DRONE_PATH_COLORS[0][1]
START_HOUR = 7.0  # same clock convention as the dashboard: mission starts 07:00 on day 1
SWIM_WINDOW_S = 2 * 3600.0

CROP_NAMES = {
    "sweet_potato_beauregard": "Sweet potato",
    "carboniato_ohenry": "Carboniato",
    "cover_then_sweet_potato": "Cover crop → sweet potato",
    "horticulture_mix": "Horticulture mix",
}
STATE_NAMES = {"idle": "idle", "loading": "loading", "outbound": "outbound", "applying": "applying",
               "return": "return", "swap": "battery swap"}
FACTOR_SHORT = {
    "dry_matter_frac": "dry matter", "combustion_frac": "share combusted", "ef_ch4": "CH4", "ef_n2o": "N2O",
    "ef_co": "CO", "ef_nox": "NOx", "ef_co2": "biogenic CO2 (context only)", "ef_pm25": "PM2.5",
    "gwp_ch4": "GWP100 CH4", "gwp_n2o": "N2O", "soil_co2e": "soil carbon", "n_retained": "N retained",
    "soc_increase": "SOC", "awc_per_soc": "plant-available water", "depth_cm": "incorporation depth",
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _clip01(x: float) -> float:
    return min(max(float(x), 0.0), 1.0)


def _ease_out(x: float) -> float:
    return 1.0 - (1.0 - _clip01(x)) ** 3


def _smooth(x: float) -> float:
    x = _clip01(x)
    return x * x * (3.0 - 2.0 * x)


def _smooth_arr(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _lerp(a: float, b: float, u: float) -> float:
    return a + (b - a) * u


def _mix(c0: str, c1: str, u: float) -> tuple[float, float, float]:
    a, b = np.array(to_rgb(c0)), np.array(to_rgb(c1))
    return tuple(a + (b - a) * _clip01(u))


def _pretty(s: str) -> str:
    for a, b in (("CO2", "CO₂"), ("CH4", "CH₄"), ("N2O", "N₂O"), ("m2", "m²")):
        s = s.replace(a, b)
    return s


def _dec(final: float, big: float = 100.0) -> int:
    return 0 if abs(final) >= big else 1


def _sig2(x: float) -> int:
    if x <= 0:
        return 0
    p = 10 ** max(int(math.floor(math.log10(x))) - 1, 0)
    return int(round(x / p) * p)


def _clock_text(t: float) -> str:
    tot = START_HOUR * 3600.0 + t
    day = int(tot // 86400) + 1
    return f"DAY {day} · {int(tot % 86400 // 3600):02d}:{int(tot % 3600 // 60):02d}"


def _band(x0: float, y0a: float, y0b: float, x1: float, y1a: float, y1b: float, n: int = 26) -> np.ndarray:
    """Sankey ribbon polygon between two vertical segments (smoothstep S-curve)."""
    u = np.linspace(0.0, 1.0, n)
    s = u * u * (3.0 - 2.0 * u)
    xs = x0 + (x1 - x0) * u
    top = y0a + (y1a - y0a) * s
    bot = y0b + (y1b - y0b) * s
    return np.column_stack([np.concatenate([xs, xs[::-1]]), np.concatenate([top, bot[::-1]])])


def _parcel_label(pid: str, crop: str) -> str:
    letter, _, rest = pid.partition("_")
    return f"{letter}  {CROP_NAMES.get(crop, rest.replace('_', ' ').capitalize())}"


def _effector_names(cart: bool) -> tuple[str, str, str]:
    """(name, short detail, unit word for ms_units)."""
    if cart:
        return "Twin air cannons", "pneumatic, port + starboard", "cartridges"
    return "Chip blower", "snow-blower chip chute", "bursts"


def _source_short(src: str) -> str:
    s = src.split(",")[0]
    return s.split(" range")[0].strip()


def _sources_text() -> str:
    """'items (source)' sentences for every impact factor, straight from ``factor_table()``."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in factor_table():
        groups.setdefault(r["source"], []).append(r)
    parts = []
    for src, rows in groups.items():
        chunks, run, unit = [], [], None
        for r in rows:
            if run and r["unit"] != unit:
                chunks.append(", ".join(run) + ("" if unit == "-" else f" {unit}"))
                run = []
            unit = r["unit"]
            name = FACTOR_SHORT.get(r["key"], r["label"]) + ("*" if r["assumed"] else "")
            run.append(f"{name} {r['value']:g}")
        chunks.append(", ".join(run) + ("" if unit == "-" else f" {unit}"))
        parts.append(f"{'; '.join(chunks)} [{src}]")
    txt = " · ".join(parts) + " · * assumption. Biogenic CO2 of burning is never added to the soil-carbon figure."
    return _pretty(txt)


def _sources_short() -> str:
    seen: list[str] = []
    for r in factor_table():
        if r["source"].startswith("assumption"):
            continue
        s = _source_short(r["source"])
        if s not in seen:
            seen.append(s)
    return "Impact factors: " + " · ".join(seen) + " · counterfactual: open pile burning"


# ---------------------------------------------------------------------------
# timing
# ---------------------------------------------------------------------------


class _Clock:
    """Video time -> mission time: intro, slowed start, linear body, eased landing, outro.

    Nominal 60 s layout (scaled with ``seconds``): 3 s intro, 48 s mission, 9 s outro.
    The first ~1.5 mission hours run slowly so the start is readable.
    """

    def __init__(self, t_end: float, seconds: float, fps: int, slow_h: float = 1.5):
        self.k = seconds / 60.0
        self.fps = fps
        self.seconds = seconds
        self.n = max(int(round(seconds * fps)), 2)
        self.intro = 3.0 * self.k
        self.outro = seconds - 9.0 * self.k
        Lm = self.outro - self.intro
        s = np.linspace(0.0, Lm, 8001)
        h0, ein = 0.025 * Lm, 0.0125 * Lm
        a, b, e, h1 = 0.11 * Lm, 0.06 * Lm, 0.06 * Lm, 0.03 * Lm
        env = _smooth_arr((s - h0) / ein) * (1.0 - 0.8 * _smooth_arr((s - (Lm - h1 - e)) / e)) * (s < Lm - h1)
        sw = _smooth_arr((s - (h0 + a)) / b)
        g0, g1 = env * (1.0 - sw), env * sw

        def cum(g):
            return np.concatenate([[0.0], np.cumsum(0.5 * (g[1:] + g[:-1]) * np.diff(s))])

        G0, G1 = cum(g0), cum(g1)
        i_slow = int(np.searchsorted(s, h0 + a))
        r0 = min(slow_h * 3600.0, 0.05 * t_end) / max(G0[i_slow], 1e-9)
        r1 = (t_end - r0 * G0[-1]) / max(G1[-1], 1e-9)
        if r1 < r0:
            r0 = r1 = t_end / max(G0[-1] + G1[-1], 1e-9)
        T = r0 * G0 + r1 * G1
        scale = t_end / max(T[-1], 1e-9)
        T *= scale
        rate = (r0 * g0 + r1 * g1) * scale
        self._s, self._T = s, T
        self.v = np.arange(self.n) / fps
        sm = self.v - self.intro
        self.t = np.maximum.accumulate(np.interp(sm, s, T))
        self.t[self.v >= self.outro] = t_end
        self.rate = np.interp(sm, s, rate, left=0.0, right=0.0)
        moving = np.concatenate([[False], np.diff(T) > 0])
        self._Tm, self._sm = T[moving], s[moving]
        self.t_end = t_end

    def v_of(self, t: float) -> float:
        """Video time at which mission time t is first shown."""
        if len(self._Tm) == 0:
            return self.intro
        return self.intro + float(np.interp(t, self._Tm, self._sm))


# ---------------------------------------------------------------------------
# animated groups
# ---------------------------------------------------------------------------


class _Group:
    """Artists that enter / leave together: ease-out fade + slide via a shared offset transform."""

    def __init__(self, ax, t_in: float, dur: float, dy: float, t_out: float | None, dur_out: float, dy_out: float):
        self.off = Affine2D()
        self.tr = self.off + ax.transData
        self.t_in, self.dur, self.dy = t_in, max(dur, 1e-6), dy
        self.t_out, self.dur_out, self.dy_out = t_out, max(dur_out, 1e-6), dy_out
        self.items: dict[Any, list] = {}
        self.a = 0.0
        self._key: tuple | None = None

    def add(self, art, alpha: float | None = 1.0):
        """alpha=None: colours/alpha are managed by the caller, the group only toggles visibility."""
        self.items[art] = [alpha, 1.0]
        self._apply(art)
        return art

    def _apply(self, art) -> None:
        base, own = self.items[art]
        if base is None:
            art.set_visible(self.a > 0.003 and own > 0.003)
            return
        al = base * own * self.a
        art.set_visible(al > 0.003)
        art.set_alpha(min(al, 1.0))

    def own(self, art, value: float) -> None:
        rec = self.items[art]
        if rec[1] != value:
            rec[1] = value
            self._apply(art)

    def update(self, v: float) -> None:
        ain = _ease_out((v - self.t_in) / self.dur)
        aout = 1.0 if self.t_out is None else 1.0 - _smooth((v - self.t_out) / self.dur_out)
        a = ain * aout
        dy = self.dy * (1.0 - ain) + self.dy_out * (1.0 - aout)
        key = (round(a, 4), round(dy, 2))
        if key == self._key:
            return
        self._key = key
        self.a = a
        self.off.clear().translate(0.0, dy)
        for art in self.items:
            self._apply(art)


# ---------------------------------------------------------------------------
# renderer
# ---------------------------------------------------------------------------


class _Infographic:
    def __init__(self, res, tl: MissionTimeline, clock: _Clock, size: tuple[int, int]):
        w, h = int(size[0]), int(size[1])
        if abs(h / w - 9.0 / 16.0) > 0.01:
            raise ValueError(f"size must be 16:9, got {size}")
        self.size = (w, h)
        self.res, self.tl, self.ck = res, tl, clock
        self.fig = plt.figure(figsize=(19.2, 10.8), dpi=w / 19.2, facecolor=BG)
        self.ax = ax = self.fig.add_axes((0.0, 0.0, 1.0, 1.0))
        ax.set_axis_off()
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        self.fig.canvas.draw()
        self._r = self.fig.canvas.get_renderer()
        self._px = W / self.fig.bbox.width
        self.cart = res.spec.mode.value == "cartridge"
        self.kp = res.kpis
        self.imp = tl.impact
        self.fin = tl.at(tl.t_end)
        self.target_t = tl.target_kg / 1000.0
        self.area_ha = float(self.kp["area_treated_ha"])
        self.dose = float(self.kp["dose_kg_per_m2"])
        self.scope = _clip01(self.fin["frac"])
        self.partial = self.scope < 0.98
        if self.partial:
            # impact_summary() credits the whole bedded area; a partial run only mulched part of it, and its
            # cost per 1000 m2 mixes partial time with full area, so the € saving is withheld
            imp = dict(tl.impact)
            imp.update(impact_from_material(self.fin["applied_t"], self.area_ha * self.scope, self.dose))
            imp["total_co2e_t"] = imp["soil_co2e_t"] + imp["burn_ghg_avoided_co2e_t"]
            imp["saving_eur_total"] = imp["saving_eur_per_1000m2"] = None
            self.imp = imp
            self.fin = self._rescale(self.fin)
        self.tdec = _dec(self.target_t, 300.0)
        self.eff_name, self.eff_detail, self.eff_unit = _effector_names(self.cart)
        self.groups: list[_Group] = []
        self.visits = list(res.visits)
        self.v_depart = np.array([v["t_depart"] for v in self.visits]) if self.visits else np.zeros(0)
        self.v_arrive = np.array([v["t_arrive"] for v in self.visits]) if self.visits else np.zeros(0)
        arr_depot = [e["t"] for e in res.events if e["kind"] == "arrive_depot"]
        self.t_home = float(arr_depot[-1]) if arr_depot else math.inf
        self._prev_flow: dict[int, dict[str, float]] = {}
        self.dyn: list = []
        self.main_groups: list[_Group] = []
        self.outro_groups: list[_Group] = []

        self._build_intro()
        self._build_header()
        self._build_ribbon()
        self._build_ring()
        self._build_tiles()
        self._build_sankey()
        self._build_donut()
        self._build_swim()
        self._build_parcels()
        self._build_chart()
        self._build_feed()
        self._build_footer()
        self._build_outro()
        ax.set_aspect("auto")
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        self._base = self._background_region()
        self.dyn.sort(key=lambda a: a.get_zorder())
        # main sequence is "steady" once every panel has entered: blit static layer, redraw dynamic artists only
        self.v_steady = (max(g.t_in + g.dur for g in self.main_groups), self.ck.outro)
        self.v_frozen = max([g.t_in + g.dur for g in self.outro_groups] + [self.v_countup_end]) + 1.0 / clock.fps
        self._steady = False
        self._bg = None
        self._frozen = False
        w8, h8 = -(-w // 8) * 8, -(-h // 8) * 8
        self._out = np.empty((h8, w8, 3), dtype=np.uint8)
        self._out[:] = np.array([int(round(255 * c)) for c in to_rgb(BG)], dtype=np.uint8)

    # ------------------------------------------------------------- primitives
    def _group(self, t_in: float, dur: float = 0.55, dy: float = 22.0, t_out: float | None = None,
               dur_out: float = 0.5, dy_out: float = -12.0) -> _Group:
        k = self.ck.k
        g = _Group(self.ax, t_in, dur * k, dy, t_out, dur_out * k, dy_out)
        self.groups.append(g)
        return g

    def _main_group(self, delay: float) -> _Group:
        g = self._group(self.ck.intro + delay * self.ck.k, t_out=self.ck.outro, dur_out=0.6)
        self.main_groups.append(g)
        return g

    def _outro_group(self, delay: float, dy: float = 24.0) -> _Group:
        g = self._group(self.ck.outro + delay * self.ck.k, dur=0.6, dy=dy)
        self.outro_groups.append(g)
        return g

    def _dyn(self, art):
        """Register an artist that changes during the steady main sequence (redrawn every frame)."""
        self.dyn.append(art)
        return art

    def patch(self, g: _Group, p, alpha: float | None = 1.0, dyn: bool = False):
        self.ax.add_patch(p)
        g.add(p, alpha)
        return self._dyn(p) if dyn else p

    def text(self, g: _Group, x: float, y: float, s: str, size: float = 11, color=INK, weight: str = "normal",
             family: str = SANS, ha: str = "left", va: str = "center", alpha: float | None = 1.0, z: float = 4,
             dyn: bool = False, **kw):
        t = self.ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, family=family, ha=ha, va=va,
                         transform=g.tr, zorder=z, **kw)
        g.add(t, alpha)
        return self._dyn(t) if dyn else t

    def rect(self, g: _Group, x: float, y: float, w: float, h: float, color, alpha: float | None = 1.0,
             z: float = 2, dyn: bool = False):
        p = Rectangle((x, y), w, h, facecolor=color, edgecolor="none", transform=g.tr, zorder=z)
        return self.patch(g, p, alpha, dyn)

    def box(self, g: _Group, x0: float, y0: float, x1: float, y1: float, fc=PANEL, ec=EDGE, r: float = 10,
            lw: float = 1.0, alpha: float | None = 1.0, z: float = 1, dyn: bool = False):
        p = FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle=f"round,pad=0,rounding_size={r}", facecolor=fc,
                           edgecolor=ec, lw=lw, transform=g.tr, zorder=z)
        return self.patch(g, p, alpha, dyn)

    def panel(self, g: _Group, x0: float, y0: float, x1: float, y1: float, title: str, sub: str | None = None):
        self.box(g, x0, y0, x1, y1)
        self.rect(g, x0 + 20, y0, 34, 3, AMBER, z=3)
        t = self.text(g, x0 + 20, y0 + 28, title, size=11.5, weight="bold", color=INK)
        if sub:
            self.text(g, x0 + 28 + self.tw(title, 11.5, "bold"), y0 + 28, sub, size=10, color=MUTED)
        return t

    def tw(self, s: str, size: float, weight: str = "normal", family: str = SANS) -> float:
        """Rendered text width in design units."""
        t = self.ax.text(0, 0, s, fontsize=size, fontweight=weight, family=family)
        w = t.get_window_extent(self._r).width * self._px
        t.remove()
        return w

    def _rescale(self, snap: dict) -> dict:
        """Live impact counters from ``self.imp`` (same pro-rata by applied mass as MissionTimeline.at)."""
        f = float(self.tl.applied_kg[-1])
        f = snap["applied_t"] * 1000.0 / f if f > 0 else 0.0
        out = dict(snap)
        for key in ("soil_co2e_t", "burn_ghg_avoided_co2e_t", "total_co2e_t", "n_retained_kg", "pm25_avoided_kg"):
            out[key] = f * self.imp[key]
        out["saving_eur"] = None
        return out

    # ------------------------------------------------------------- background
    def _background_region(self):
        """Soft vignette rendered once and kept as a raster region (an image artist costs ~50 ms per draw)."""
        w, h = self.fig.canvas.get_width_height()
        yy, xx = np.mgrid[0:h, 0:w]
        d = np.hypot((xx - 0.42 * w) / w, (yy - 0.35 * h) / h * 0.75)
        u = np.clip(1.0 - d / 0.9, 0.0, 1.0) ** 1.6
        c0, c1 = np.array(to_rgb(BG)), np.array(to_rgb("#132015"))
        img = (c0[None, None, :] + (c1 - c0)[None, None, :] * u[..., None]) * 255.0
        fimg = self.fig.figimage(np.round(img).astype(np.uint8), 0, 0, zorder=-5, origin="upper")
        self.ax.set_visible(False)
        self.fig.canvas.draw()
        base = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        fimg.remove()
        self.ax.set_visible(True)
        return base

    # ------------------------------------------------------------- intro
    def _build_intro(self) -> None:
        k, I = self.ck.k, self.ck.intro
        ax = self.ax
        mode_title = "Seed-cartridge reforestation mission" if self.cart else "Wood-chip mulch mission"
        self.wm_w64 = self.tw("PRUNEMAXXING", 64, "bold")
        self.wm_w26 = self.tw("PRUNEMAXXING", 26, "bold")
        self.mt_w32 = self.tw(mode_title, 32)
        self.wordmark = ax.text(0, 0, "PRUNEMAXXING", color=AMBER, fontsize=64, fontweight="bold", family=SANS,
                                va="center", ha="left", zorder=6)
        self.mode_title = ax.text(0, 0, mode_title, color=INK, fontsize=32, family=SANS, va="center", ha="left",
                                  zorder=6)
        g = self._group(0.3 * k, dur=0.6, dy=18.0, t_out=I - 0.6 * k, dur_out=0.45, dy_out=-16.0)
        self.g_intro = g
        self.intro_line = self.rect(g, 960, 458, 0, 3, AMBER, z=5)
        self.text(g, 960, 556, "Parc Agrari del Baix Llobregat · Viladecans · AGR-E2 · Unite! Tri-City Hackathon",
                  size=15, color=MUTED, ha="center")
        area = float(self.kp["area_treated_ha"])
        dose = float(self.kp["dose_kg_per_m2"])
        tiles = [(f"{area:.2f} ha", f"{len(self.res.site.parcels)} parcels to treat"),
                 (f"{dose:g} kg/m²", "seed + mycorrhiza cartridges" if self.cart else "wood-chip mulch dose"),
                 (f"{self.target_t:,.{self.tdec}f} t", (f"{int(self.kp['cartridges_placed']):,} cartridges"
                                                        if self.cart else "shredded prunings to spread")),
                 (f"1 + {self.res.n_drones}", "mothership UGV + heavy-lift drones")]
        tw, gap = 330.0, 24.0
        x = 960 - (len(tiles) * tw + (len(tiles) - 1) * gap) / 2
        for j, (val, lab) in enumerate(tiles):
            gt = self._group((0.85 + 0.12 * j) * k, dur=0.5, dy=26.0, t_out=I - 0.6 * k, dur_out=0.45, dy_out=-16.0)
            self.box(gt, x, 616, x + tw, 736, fc=PANEL, ec=EDGE)
            self.rect(gt, x + 20, 616, 34, 3, AMBER, z=3)
            self.text(gt, x + tw / 2, 664, val, size=30, weight="bold", color=AMBER, ha="center")
            self.text(gt, x + tw / 2, 708, lab, size=12.5, color=MUTED, ha="center")
            x += tw + gap
        gc = self._group(1.5 * k, dur=0.5, dy=12.0, t_out=I - 0.6 * k, dur_out=0.45, dy_out=-16.0)
        self.text(gc, 960, 800, "Counterfactual: these prunings would otherwise be burned in open piles.",
                  size=14, color=INK, ha="center")

    def _update_intro(self, v: float) -> None:
        k, I = self.ck.k, self.ck.intro
        a = _ease_out(v / (0.6 * k))
        p = _smooth((v - (I - 0.55 * k)) / (0.7 * k))
        fs = _lerp(64, 26, p)
        dy = 30.0 * (1.0 - a)
        self.wordmark.set_fontsize(fs)
        self.wordmark.set_position((_lerp(960 - self.wm_w64 / 2, M, p), _lerp(406, 52, p) + dy))
        self.wordmark.set_alpha(a)
        a2 = _ease_out((v - 0.35 * k) / (0.55 * k))
        self.mode_title.set_fontsize(_lerp(32, 19, p))
        self.mode_title.set_position((_lerp(960 - self.mt_w32 / 2, M + self.wm_w26 + 24, p),
                                      _lerp(506, 53, p) + 20.0 * (1.0 - a2)))
        self.mode_title.set_alpha(a2)
        self.intro_line.set_width(560 * _ease_out((v - 0.35 * k) / (0.8 * k)))
        self.intro_line.set_x(960 - self.intro_line.get_width() / 2)

    # ------------------------------------------------------------- header
    def _build_header(self) -> None:
        g = self._group(self.ck.intro - 0.05 * self.ck.k)
        k = self.kp
        dose = float(k["dose_kg_per_m2"])
        what = "seed cartridges" if self.cart else "shredded prunings"
        self.text(g, M, 90, f"Parc Agrari del Baix Llobregat · Viladecans · {len(self.res.site.parcels)} parcels · "
                  f"{k['area_treated_ha']:.2f} ha · {dose:g} kg/m² · {self.target_t:,.{self.tdec}f} t of {what} · "
                  f"1 mothership + {self.res.n_drones} drones", size=11, color=MUTED)
        self.t_clock = self.text(g, W - M, 50, "", size=22, weight="bold", family=MONO, ha="right", dyn=True)
        self.t_rate = self.text(g, W - M, 88, "", size=11, family=MONO, color=AMBER, ha="right", dyn=True)

    def _update_header(self, i: int, snap: dict) -> None:
        t = snap["t"]
        self.t_clock.set_text(_clock_text(t))
        r = self.ck.rate[i]
        if self.ck.v[i] >= self.ck.outro - 1e-9 or t >= self.tl.t_end - 1e-6:
            head = "partial run complete" if self.partial else "mission complete"
        elif r < 1.0:
            head = "mission clock"
        else:
            head = f"time-lapse ×{_sig2(r):,}"
        self.t_rate.set_text(f"{head} · T+ {t / 3600.0:.1f} h")

    # ------------------------------------------------------------- ribbon
    def _build_ribbon(self) -> None:
        g = self.g_rib = self._main_group(0.08)
        tl = self.tl
        self.box(g, M, 112, W - M, 176)
        self.text(g, M + 20, 136, "MISSION", size=9.5, weight="bold", color=MUTED)
        self.text(g, M + 20, 154, "TIMELINE", size=9.5, weight="bold", color=MUTED)
        x0, x1 = 196.0, 1572.0
        self.rx0, self.rx1 = x0, x1
        self.rect(g, x0, 147, x1 - x0, 6, TRACK, z=2)
        self.rib_fill = self.rect(g, x0, 147, 0, 6, ACTUAL, z=3, dyn=True)
        X = lambda t: x0 + (x1 - x0) * t / max(tl.t_end, 1e-9)  # noqa: E731
        self.rX = X
        self.text(g, x0 + 5, 126, "Day 1", size=9, color=MUTED)
        self.rect(g, x0 - 0.5, 120, 1.0, 40, DIM, z=2)
        dh = 24.0 - START_HOUR
        while dh * 3600 < tl.t_end - 1800:
            xd = X(dh * 3600)
            self.rect(g, xd - 0.5, 120, 1.0, 40, DIM, z=2)
            self.text(g, xd + 5, 126, f"Day {int((START_HOUR + dh) // 24) + 1}", size=9, color=MUTED)
            dh += 24.0
        segs = [[(X(t), 143), (X(t), 157)] for t in self.v_arrive]
        self.rib_ticks = LineCollection(segs, linewidths=1.6, transform=g.tr, zorder=3)
        self.ax.add_collection(self.rib_ticks, autolim=False)
        g.add(self._dyn(self.rib_ticks), None)
        self.rib_head_line = self.rect(g, x0 - 1, 137, 2, 24, INK, z=5, dyn=True)
        self.rib_head = self.patch(g, Polygon([[0, 0]] * 3, closed=True, facecolor=INK, edgecolor="none",
                                              transform=g.tr, zorder=5), dyn=True)
        # obstacles appear only once discovered
        self.obst = []
        heights = {o.oid: o.height_m for o in self.res.site.obstacles}
        for e in self.res.events:
            if e["kind"] != "obstacle_detected":
                continue
            xo = X(e["t"])
            dia = self.patch(g, Polygon([[xo, 141], [xo + 8, 150], [xo, 159], [xo - 8, 150]], closed=True,
                                        facecolor=OBST, edgecolor="white", lw=0.8, transform=g.tr, zorder=6), dyn=True)
            ring = self.patch(g, Circle((xo, 150), 8, fill=False, edgecolor=OBST, lw=1.6, transform=g.tr, zorder=6),
                              alpha=None, dyn=True)
            name = "waterlogged patch" if e["obstacle_type"].startswith("water") else e["obstacle_type"]
            lab = self.text(g, xo, 168, name, size=9, color=OBST, ha="center", dyn=True)
            self.obst.append(dict(t=float(e["t"]), x=xo, dia=dia, ring=ring, lab=lab,
                                  sensor="Lidar" if heights.get(e["oid"], 1.0) > 0.3 else "Camera", ev=e))
        self.text(g, 1600, 134, "STATIONS DONE", size=9, weight="bold", color=MUTED)
        self.rib_count = self.text(g, 1600, 158, "", size=17, weight="bold", family=MONO, dyn=True)
        self.pill = self.box(g, 1736, 136, 1852, 164, fc=GREEN, ec="none", r=14, z=3, dyn=True)
        self.pill_t = self.text(g, 1794, 150, "", size=10, weight="bold", color=BG, ha="center", z=4, dyn=True)

    def _phase_label(self, snap: dict) -> tuple[str, str]:
        t = snap["t"]
        if t >= self.t_home - 1.0:
            return "AT DEPOT", AMBER
        if t <= 1.0:
            return "DEPARTING", ACTUAL
        ph = snap["phase"]
        if ph == "parked":
            return "PARKED", GREEN
        return {"lane": ("DRIVING", ACTUAL), "headland": ("U-TURN", ACTUAL), "transit": ("TRANSFER", ACTUAL),
                "weed_detour": ("WEED DETOUR", WEED), "obstacle_detour": ("BYPASS", BYPASS)}.get(ph, ("DRIVING", ACTUAL))

    def _update_ribbon(self, i: int, v: float, snap: dict) -> None:
        g = self.g_rib
        t = snap["t"]
        x = self.rX(t)
        self.rib_fill.set_width(x - self.rx0)
        self.rib_head_line.set_x(x - 1)
        self.rib_head.set_xy([[x - 6, 131], [x + 6, 131], [x, 139]])
        dep = np.sort(self.v_depart)
        n_done = int(np.searchsorted(dep, t, side="right"))
        cols = [to_rgba(GREEN if td <= t else (AMBER if ta <= t else DIM), g.a)
                for ta, td in zip(self.v_arrive, self.v_depart)]
        self.rib_ticks.set_colors(cols)
        k = self.ck.k
        flash = 0.0
        if n_done:
            flash = 1.0 - _clip01((v - self.ck.v_of(float(dep[n_done - 1]))) / (0.5 * k))
        self.rib_count.set_text(f"{n_done:>2d} / {len(self.visits)}")
        self.rib_count.set_color(_mix(INK, GREEN, flash))
        lab, col = self._phase_label(snap)
        self.pill_t.set_text(lab)
        self.pill.set_facecolor(col)
        for o in self.obst:
            on = t >= o["t"]
            vo = self.ck.v_of(o["t"])
            pop = _ease_out((v - vo) / (0.35 * k)) if on else 0.0
            g.own(o["dia"], pop)
            g.own(o["lab"], pop)
            s = 1.0 + 0.35 * math.sin(math.pi * _clip01((v - vo) / (0.35 * k))) if on else 1.0
            o["dia"].set_xy([[o["x"], 150 - 9 * s], [o["x"] + 8 * s, 150], [o["x"], 150 + 9 * s],
                             [o["x"] - 8 * s, 150]])
            pr = _clip01((v - vo) / (1.2 * k)) if on else 1.0
            ring_a = (1.0 - pr) * g.a
            o["ring"].set_radius(8 + 34 * _ease_out(pr))
            o["ring"].set_edgecolor(to_rgba(OBST, ring_a))
            g.own(o["ring"], 1.0 if (on and pr < 1.0) else 0.0)

    # ------------------------------------------------------------- ring (tonnes applied)
    def _build_ring(self) -> None:
        g = self.g_ring = self._main_group(0.18)
        k = self.kp
        self.panel(g, M, 192, 596, 576, "MATERIAL APPLIED")
        self.text(g, 576, 220, f"{k['dose_kg_per_m2']:g} kg/m² on {k['area_treated_ha']:.2f} ha", size=10,
                  color=MUTED, ha="right")
        cx, cy, R, w = 326.0, 394.0, 138.0, 24.0
        self.ring_c = (cx, cy, R, w)
        tr = Wedge((cx, cy), R, 0, 360, width=w, facecolor=TRACK, edgecolor="none", transform=g.tr, zorder=2)
        self.ax.add_patch(tr)
        g.add(tr)
        for q in range(4):
            a = math.radians(-90 + 90 * q)
            seg = LineCollection([[(cx + (R + 4) * math.cos(a), cy + (R + 4) * math.sin(a)),
                                   (cx + (R + 11) * math.cos(a), cy + (R + 11) * math.sin(a))]],
                                 colors=DIM, linewidths=1.5, transform=g.tr, zorder=2)
            self.ax.add_collection(seg, autolim=False)
            g.add(seg)
        self.ring_glow = self.patch(g, Wedge((cx, cy), R + 7, -90, -90, width=w + 14, facecolor=AMBER,
                                             edgecolor="none", transform=g.tr, zorder=2), alpha=0.13, dyn=True)
        self.ring_w = self.patch(g, Wedge((cx, cy), R, -90, -90, width=w, facecolor=AMBER, edgecolor="none",
                                          transform=g.tr, zorder=3), dyn=True)
        self.ring_heads = []
        for rad, al in ((22, 0.10), (16, 0.22), (11, 0.9)):
            c = Circle((cx, cy - R + w / 2), rad, facecolor=AMBER if rad > 11 else "#ffd27a", edgecolor="none",
                       transform=g.tr, zorder=4)
            self.ring_heads.append(self.patch(g, c, alpha=al, dyn=True))
        widest = f"{max(self.target_t, self.fin['applied_t']):,.{self.tdec}f} t"
        fs = 40.0
        while fs > 26 and self.tw(widest, fs, "bold") > 1.6 * (R - w):
            fs -= 1.0
        self.ring_val = self.text(g, cx, cy - 18, "", size=fs, weight="bold", ha="center", dyn=True)
        self.text(g, cx, cy + 26, f"of {self.target_t:,.{self.tdec}f} t target", size=13, color=MUTED, ha="center")
        self.ring_pct = self.text(g, cx, cy + 60, "", size=17, weight="bold", color=AMBER, ha="center", dyn=True)
        self.ring_cells = self.text(g, cx, 556, "", size=10.5, color=MUTED, ha="center", dyn=True)

    def _update_ring(self, snap: dict) -> None:
        cx, cy, R, w = self.ring_c
        f = _clip01(snap["frac"])
        th = -90 + 360 * f
        self.ring_w.set_theta2(th)
        self.ring_glow.set_theta2(th)
        a = math.radians(th)
        for c in self.ring_heads:
            c.center = (cx + (R - w / 2) * math.cos(a), cy + (R - w / 2) * math.sin(a))
            self.g_ring.own(c, 1.0 if f > 0.002 else 0.0)
        self.ring_val.set_text(f"{snap['applied_t']:,.{self.tdec}f} t")
        self.ring_pct.set_text(f"{100 * snap['frac']:.1f} %")
        self.ring_cells.set_text(f"cells at full dose  {snap['cells_done']:,} / {snap['n_cells']:,}")

    # ------------------------------------------------------------- counter tiles
    def _tile_specs(self) -> list[dict[str, Any]]:
        imp = self.imp
        cd = _dec(imp["total_co2e_t"])
        specs = [
            dict(label="CO₂e stored + avoided", unit="t", color=GREEN, final=imp["total_co2e_t"], dec=cd,
                 val=lambda s: s["total_co2e_t"],
                 sub=lambda s: f"soil {s['soil_co2e_t']:,.{cd}f} t + no burning {s['burn_ghg_avoided_co2e_t']:,.{cd}f} t"),
            dict(label="PM2.5 smoke avoided", unit="kg", color=SKY, final=imp["pm25_avoided_kg"],
                 dec=_dec(imp["pm25_avoided_kg"]), val=lambda s: s["pm25_avoided_kg"],
                 sub=lambda s: "vs open pile burning"),
        ]
        if self.cart:
            specs += [
                dict(label="Seed cartridges placed", unit="", color=FXC, final=float(self.fin.get("cartridges", 0)),
                     dec=0, val=lambda s: s.get("cartridges", 0.0), sub=lambda s: "0.5 kg · seeds + mycorrhiza"),
                dict(label="Seeds sown", unit="", color="#b8f26d", final=float(self.fin.get("seeds", 0)), dec=0,
                     val=lambda s: s.get("seeds", 0.0),
                     sub=lambda s: f"3 per cartridge · {len(self.imp['species'])} native species"),
            ]
        else:
            if self.partial:
                ha_per_t = 1.0 / (self.dose * 10.0)
                specs.append(dict(label="Area mulched", unit="ha", color=AMBER, final=self.fin["applied_t"] * ha_per_t,
                                  dec=2, val=lambda s: s["applied_t"] * ha_per_t,
                                  sub=lambda s: f"of {self.area_ha:.2f} ha · partial run, no € figure"))
            else:
                per = float(imp["saving_eur_per_1000m2"] or 0.0)
                specs.append(dict(label="Saved vs REGENSOL", unit="", prefix="€", color=AMBER,
                                  final=float(imp["saving_eur_total"] or 0), dec=0,
                                  val=lambda s: s["saving_eur"] or 0.0,
                                  sub=lambda s: f"{per:,.0f} € per 1000 m² treated"))
            specs.append(dict(label="Nitrogen retained", unit="kg N", color=CHIP, final=imp["n_retained_kg"],
                              dec=_dec(imp["n_retained_kg"]), val=lambda s: s["n_retained_kg"],
                              sub=lambda s: "REGENSOL trial factor"))
        specs += [
            dict(label="Drone sorties", unit="", color=DRONES, final=float(self.fin["sorties"]), dec=0,
                 val=lambda s: s["sorties"], sub=lambda s: f"{s['drone_km']:,.0f} km flown"),
            dict(label="Battery swaps", unit="", color=DRONE_STATE_COLORS["swap"], final=float(self.fin["swaps"]),
                 dec=0, val=lambda s: s["swaps"], sub=lambda s: f"{s['drone_kwh']:,.0f} kWh drone energy"),
        ]
        return specs

    def _build_tiles(self) -> None:
        self.tiles = []
        tw, th = 262.0, 136.0
        # DejaVu digits are tabular, so a number's width follows from its characters (no per-frame layout)
        w0 = self.tw("0", 26, "bold")
        self.glyph_w = {c: (self.tw("0" + c * 10, 26, "bold") - w0) / 10 for c in "0,.€"}
        for j, sp in enumerate(self._tile_specs()):
            col, row = j % 2, j // 2
            x0 = M + col * (tw + 16)
            y0 = 592 + row * (th + 16)
            g = self._main_group(0.28 + 0.05 * j)
            self.box(g, x0, y0, x0 + tw, y0 + th)
            self.rect(g, x0, y0 + 16, 4, th - 32, sp["color"], z=3)
            self.text(g, x0 + 20, y0 + 28, sp["label"], size=10.5, color=MUTED)
            pre = sp.get("prefix", "")
            val = self.text(g, x0 + 20, y0 + 86, "", size=26, weight="bold", color=sp["color"], va="baseline",
                            dyn=True)
            unit = None
            if sp["unit"]:
                unit = self.text(g, x0 + 20, y0 + 86, sp["unit"], size=13, color=INK, va="baseline", dyn=True)
            sub = self.text(g, x0 + 20, y0 + 112, "", size=9.5, color=MUTED, dyn=True)
            self.tiles.append(dict(sp=sp, val=val, unit=unit, sub=sub, pre=pre, x=x0 + 20))

    def _num_w(self, s: str) -> float:
        gw = self.glyph_w
        return sum(gw["0"] if c.isdigit() else gw.get(c, gw["0"]) for c in s)

    def _update_tiles(self, snap: dict) -> None:
        for t in self.tiles:
            sp = t["sp"]
            s = f"{t['pre']}{float(sp['val'](snap)):,.{sp['dec']}f}"
            t["val"].set_text(s)
            if t["unit"] is not None:
                t["unit"].set_x(t["x"] + self._num_w(s) + 8)
            t["sub"].set_text(sp["sub"](snap))

    # ------------------------------------------------------------- sankey
    XD, XT, XH, XS, XB, NW = 668.0, 812.0, 960.0, 1116.0, 1290.0, 14.0
    YC, YD, YE = 398.0, 348.0, 486.0

    def _build_sankey(self) -> None:
        g = self.g_sk = self._main_group(0.34)
        self.panel(g, 628, 192, 1352, 548, "MATERIAL FLOW", "cumulative tonnes · band width ∝ mass")
        tl = self.tl
        F = max(self.fin["tender_t"] + tl.hopper0_kg / 1000.0, self.fin["applied_t"], 1e-6)
        self.sk_scale = 116.0 / F
        self.sk_dec = _dec(F, 300.0)
        cols = dict(dt=CHIP, th=CHIP, hd=DRONES, he=FXC, db=DRONES, eb=FXC)
        alph = dict(dt=0.55, th=0.55, hd=0.55, he=0.7, db=0.55, eb=0.7)
        self.sk_bands = {}
        for key, c in cols.items():
            self.sk_bands[key] = self.patch(g, Polygon(np.zeros((4, 2)), closed=True, facecolor=c, edgecolor="none",
                                                       transform=g.tr, zorder=2), alpha=alph[key], dyn=True)
        ncol = dict(depot=CHIP, trucks=CHIP, hopper=ACTUAL, drones=DRONES, eff=FXC, beds=GREEN)
        self.sk_nodes = {}
        for key, c in ncol.items():
            self.sk_nodes[key] = self.rect(g, 0, 0, self.NW, 0, c, z=3, dyn=True)
        n_tr = int(self.res.cfg.tender_trucks)
        X = self
        y1, y2, y3 = 246.0, 263.0, 279.0
        lab = [("depot", X.XD - 20, "left", "Depot", "seed cartridges" if self.cart else "shredded prunings", False),
               ("trucks", X.XT + 7, "center", f"{n_tr} supply trucks", "", True),
               ("hopper", X.XH + 7, "center", "Hopper", "on the mothership", False),
               ("drones", X.XS + 7, "center", f"{self.res.n_drones} drones", "", True),
               ("beds", 1332, "right", "Beds", f"{self.kp['area_treated_ha']:.2f} ha", False)]
        self.sk_vals, self.sk_extra = {}, {}
        for key, x, ha, name, extra, dyn_extra in lab:
            self.text(g, x, y1, name, size=10.5, weight="bold", ha=ha)
            self.sk_vals[key] = self.text(g, x, y2, "", size=10, family=MONO, color=INK, ha=ha, dyn=True)
            self.sk_extra[key] = self.text(g, x, y3, extra, size=9, color=MUTED, ha=ha, dyn=dyn_extra)
        fe = self.tl.flow_at(self.tl.t_end)["effector_to_beds"]
        w_eff = max(self.tw(f"{self.eff_name} · 40 ft", 10.5, "bold"),
                    self.tw(f"{fe:,.{self.sk_dec}f} t · {self.fin['ms_units']:,.0f} {self.eff_unit}", 10, family=MONO))
        xe = min(X.XS, 1336 - w_eff)
        self.text(g, xe, 514, f"{self.eff_name} · 40 ft", size=10.5, weight="bold", color=FXC)
        self.sk_vals["eff"] = self.text(g, xe, 531, "", size=10, family=MONO, color=INK, dyn=True)
        cap = float(self.res.cfg.hopper_capacity_kg)
        self.hop_cap = cap
        gx = X.XH + 7
        self.text(g, gx - 62, 499, "hopper stock", size=9, color=MUTED)
        self.rect(g, gx - 62, 509, 124, 6, TRACK, z=2)
        self.hop_bar = self.rect(g, gx - 62, 509, 0, 6, CHIP, z=3, dyn=True)
        self.hop_txt = self.text(g, gx - 62, 528, "", size=9, family=MONO, color=INK, dyn=True)
        # flow particles
        self.sk_rng = np.random.default_rng(5)
        self.sk_np = 5
        self.sk_jit = {key: self.sk_rng.uniform(-0.32, 0.32, self.sk_np) for key in cols}
        self.sk_phase = {key: self.sk_rng.uniform(0, 1, self.sk_np) for key in cols}
        self.sk_pcol = cols
        self.sk_dots = self.ax.scatter(np.zeros(1), np.zeros(1), s=9, transform=g.tr, zorder=4, linewidths=0)
        g.add(self._dyn(self.sk_dots), None)

    def _sankey_geom(self, fl: dict) -> dict:
        s = self.sk_scale
        X = self
        h_t = fl["tender_to_hopper"] * s
        hd_l = fl["hopper_to_drones"] * s
        he = fl["hopper_to_effector"] * s
        hd_b = fl["drones_to_beds"] * s
        yc, yd, ye = X.YC, X.YD, X.YE
        hh = max(h_t, hd_l + he)
        ytop = yc - (hd_l + he) / 2
        hb = hd_b + he
        nodes = dict(depot=(X.XD, yc - h_t / 2, h_t), trucks=(X.XT, yc - h_t / 2, h_t), hopper=(X.XH, yc - hh / 2, hh),
                     drones=(X.XS, yd - hd_l / 2, hd_l), eff=(X.XS, ye - he / 2, he), beds=(X.XB, yc - hb / 2, hb))
        bands = dict(
            dt=(X.XD + X.NW, yc - h_t / 2, yc + h_t / 2, X.XT, yc - h_t / 2, yc + h_t / 2),
            th=(X.XT + X.NW, yc - h_t / 2, yc + h_t / 2, X.XH, yc - h_t / 2, yc + h_t / 2),
            hd=(X.XH + X.NW, ytop, ytop + hd_l, X.XS, yd - hd_l / 2, yd + hd_l / 2),
            he=(X.XH + X.NW, ytop + hd_l, ytop + hd_l + he, X.XS, ye - he / 2, ye + he / 2),
            db=(X.XS + X.NW, yd - hd_l / 2, yd - hd_l / 2 + hd_b, X.XB, yc - hb / 2, yc - hb / 2 + hd_b),
            eb=(X.XS + X.NW, ye - he / 2, ye + he / 2, X.XB, yc - hb / 2 + hd_b, yc + hb / 2),
        )
        return dict(nodes=nodes, bands=bands)

    def _flow(self, i: int) -> dict:
        if i not in self._prev_flow:
            self._prev_flow[i] = self.tl.flow_at(float(self.ck.t[i]))
            if len(self._prev_flow) > 16:
                self._prev_flow.pop(next(iter(self._prev_flow)))
        return self._prev_flow[i]

    def _update_sankey(self, i: int, v: float, snap: dict) -> None:
        g = self.g_sk
        fl = self._flow(i)
        prev = self._flow(max(i - 3, 0))
        geo = self._sankey_geom(fl)
        for key, (x, y, h) in geo["nodes"].items():
            n = self.sk_nodes[key]
            n.set_xy((x, y))
            n.set_height(max(h, 0.0))
            g.own(n, 1.0 if h > 0.05 else 0.0)
        for key, b in geo["bands"].items():
            self.sk_bands[key].set_xy(_band(*b))
            g.own(self.sk_bands[key], 1.0 if abs(b[2] - b[1]) > 0.05 else 0.0)
        d = self.sk_dec
        self.sk_vals["depot"].set_text(f"{fl['depot_to_tender']:,.{d}f} t")
        self.sk_vals["trucks"].set_text(f"{fl['tender_to_hopper']:,.{d}f} t")
        self.sk_extra["trucks"].set_text(f"{snap['tender_deliveries']:,} top-ups")
        self.sk_vals["hopper"].set_text(f"{fl['tender_to_hopper'] + self.tl.hopper0_kg / 1000:,.{d}f} t in")
        self.sk_vals["drones"].set_text(f"{fl['drones_to_beds']:,.{d}f} t")
        self.sk_extra["drones"].set_text(f"{snap['sorties']:,} sorties")
        self.sk_vals["beds"].set_text(f"{snap['applied_t']:,.{d}f} t")
        self.sk_vals["eff"].set_text(f"{fl['effector_to_beds']:,.{d}f} t · {snap['ms_units']:,.0f} {self.eff_unit}")
        hk = snap["hopper_kg"]
        self.hop_txt.set_text(f"{hk:,.0f} / {self.hop_cap:,.0f} kg")
        self.hop_bar.set_width(124 * _clip01(hk / self.hop_cap))
        # particles on active bands
        act = dict(dt=fl["depot_to_tender"] - prev["depot_to_tender"], th=fl["tender_to_hopper"] - prev["tender_to_hopper"],
                   hd=fl["hopper_to_drones"] - prev["hopper_to_drones"],
                   he=fl["hopper_to_effector"] - prev["hopper_to_effector"],
                   db=fl["drones_to_beds"] - prev["drones_to_beds"], eb=fl["effector_to_beds"] - prev["effector_to_beds"])
        pts, cols = [], []
        speed = 0.55 / max(self.ck.k, 0.1)
        for key, b in geo["bands"].items():
            if act[key] <= 1e-6 or abs(b[2] - b[1]) < 2.0:
                continue
            x0, y0a, y0b, x1, y1a, y1b = b
            u = (self.sk_phase[key] + v * speed) % 1.0
            sm = u * u * (3 - 2 * u)
            ya = y0a + (y1a - y0a) * sm
            yb = y0b + (y1b - y0b) * sm
            y = 0.5 * (ya + yb) + self.sk_jit[key] * (yb - ya)
            pts.append(np.column_stack([x0 + (x1 - x0) * u, y]))
            fade = np.sin(np.pi * u) ** 0.5
            c = np.array(to_rgba("#fff6d8" if key in ("dt", "th") else self.sk_pcol[key]))
            cc = np.tile(c, (len(u), 1))
            cc[:, 3] = 0.95 * fade * g.a
            cols.append(cc)
        if pts:
            self.sk_dots.set_offsets(np.vstack(pts))
            self.sk_dots.set_facecolors(np.vstack(cols))
            g.own(self.sk_dots, 1.0)
        else:
            g.own(self.sk_dots, 0.0)

    # ------------------------------------------------------------- donut
    def _build_donut(self) -> None:
        g = self.g_dn = self._main_group(0.44)
        self.panel(g, 1384, 192, W - M, 548, "WORK SPLIT", "who put it on the beds")
        cx, cy, R, w = 1504.0, 384.0, 100.0, 30.0
        self.dn_c = (cx, cy, R, w)
        tr = Wedge((cx, cy), R, 0, 360, width=w, facecolor=TRACK, edgecolor="none", transform=g.tr, zorder=2)
        self.ax.add_patch(tr)
        g.add(tr)
        self.dn_d = self.patch(g, Wedge((cx, cy), R, -90, -90, width=w, facecolor=DRONES, edgecolor=PANEL, lw=2.0,
                                        transform=g.tr, zorder=3), dyn=True)
        self.dn_e = self.patch(g, Wedge((cx, cy), R + 6, -90, -90, width=w + 12, facecolor=FXC, edgecolor=PANEL,
                                        lw=2.0, transform=g.tr, zorder=3), dyn=True)
        self.dn_pct = self.text(g, cx, cy - 8, "", size=24, weight="bold", color=FXC, ha="center", dyn=True)
        self.text(g, cx, cy + 24, "by the mothership", size=9.5, color=MUTED, ha="center")
        lx = 1634.0
        self.rect(g, lx, 282, 14, 14, DRONES, z=3)
        self.text(g, lx + 24, 289, f"{self.res.n_drones} drones", size=12, weight="bold")
        self.dn_dv = self.text(g, lx + 24, 313, "", size=11, family=MONO, dyn=True)
        self.dn_dx = self.text(g, lx + 24, 334, "", size=9.5, color=MUTED, dyn=True)
        self.rect(g, lx, 382, 14, 14, FXC, z=3)
        self.text(g, lx + 24, 389, self.eff_name, size=12, weight="bold")
        self.dn_ev = self.text(g, lx + 24, 413, "", size=11, family=MONO, dyn=True)
        self.text(g, lx + 24, 434, self.eff_detail, size=9.5, color=MUTED)
        reach = float(self.res.cfg.effectors.range_max_m)
        self.text(g, 1404, 522, f"The mothership throws to cells within {reach:.1f} m (40 ft) of each station.",
                  size=9.5, color=MUTED)

    def _update_donut(self, snap: dict) -> None:
        d, e = snap["drone_t"], snap["ms_t"]
        tot = d + e
        fe = e / tot if tot > 1e-9 else 0.0
        on = tot > 1e-6
        self.dn_e.set_theta1(-90)
        self.dn_e.set_theta2(-90 + 360 * fe)
        self.dn_d.set_theta1(-90 + 360 * fe)
        self.dn_d.set_theta2(270)
        self.g_dn.own(self.dn_d, 1.0 if on else 0.0)
        self.g_dn.own(self.dn_e, 1.0 if on and fe > 1e-4 else 0.0)
        dd = self.sk_dec
        self.dn_pct.set_text(f"{100 * fe:.0f} %" if on else "–")
        self.dn_dv.set_text(f"{d:,.{dd}f} t · {100 * (1 - fe) if on else 0:.0f} %")
        self.dn_dx.set_text(f"{snap['sorties']:,} sorties")
        self.dn_ev.set_text(f"{e:,.{dd}f} t · {100 * fe:.0f} %")

    # ------------------------------------------------------------- swimlanes
    def _build_swim(self) -> None:
        g = self.g_sw = self._main_group(0.54)
        self.panel(g, 628, 564, 1352, 800, "DRONE ACTIVITY", "last 2 mission hours")
        res, tl = self.res, self.tl
        nd = res.n_drones
        self.sw_x0, self.sw_x1 = 696.0, 1196.0
        self.sw_nb = 500
        top, span = 610.0, 130.0
        gap = 8.0
        lh = (span - gap * (nd - 1)) / max(nd, 1)
        self.sw_dt = 2.0
        N = int(math.ceil(tl.t_end / self.sw_dt)) + 2
        codes = np.zeros((nd, N), dtype=np.int8)
        sidx = {s: j for j, s in enumerate(DRONE_STATES)}
        dt = self.sw_dt
        for sp in tl.spans:
            a, b = int(sp.t0 / dt + 0.5), int(sp.t1 / dt + 0.5)
            if b > a:
                codes[sp.drone, a:b] = sidx[sp.state]
        pal = np.array([to_rgb(DRONE_STATE_COLORS[s]) for s in DRONE_STATES])
        self.sw_C = np.zeros((nd, N + 1, 3))
        self.sw_C[:, 1:] = np.cumsum(pal[codes], axis=1)
        self.sw_N = N
        self.sw_empty = np.array(to_rgb(TRACK))
        self.sw_imgs, self.sw_state, self.sw_dot = [], [], []
        for d in range(nd):
            y = top + d * (lh + gap)
            col = DRONE_PATH_COLORS[d % len(DRONE_PATH_COLORS)][1]
            self.text(g, 648, y + lh / 2, f"D{d + 1}", size=11, weight="bold", color=col)
            im = self.ax.imshow(np.zeros((1, self.sw_nb, 3)), extent=(self.sw_x0, self.sw_x1, y + lh, y),
                                interpolation="nearest", aspect="auto", transform=g.tr, zorder=3)
            g.add(self._dyn(im))
            self.sw_imgs.append(im)
            dot = Circle((1218, y + lh / 2), 5, facecolor=DIM, edgecolor="none", transform=g.tr, zorder=4)
            self.sw_dot.append(self.patch(g, dot, dyn=True))
            self.sw_state.append(self.text(g, 1230, y + lh / 2, "", size=10, color=INK, dyn=True))
        yb = top + span
        self.rect(g, self.sw_x1 - 1, top - 6, 2, span + 12, INK, z=5, dyn=True)
        xm = 0.5 * (self.sw_x0 + self.sw_x1)
        self.rect(g, xm - 0.5, top, 1, span, BG, z=4, alpha=0.6, dyn=True)
        for x, s, ha in ((self.sw_x0, "−2 h", "left"), (xm, "−1 h", "center"), (self.sw_x1, "now", "center")):
            self.text(g, x, yb + 14, s, size=9, color=MUTED, ha=ha)
        x = 696.0
        for s in DRONE_STATES:
            self.rect(g, x, 776, 12, 12, DRONE_STATE_COLORS[s], z=3)
            self.text(g, x + 18, 782, STATE_NAMES[s], size=9.5, color=MUTED)
            x += 18 + self.tw(STATE_NAMES[s], 9.5) + 22
        self.sw_air = self.text(g, 1332, 592, "", size=13, weight="bold", ha="right", dyn=True)
        self.sw_air_w = self.tw("0 / 4", 13, "bold")
        self.text(g, 1332 - self.sw_air_w - 10, 592, "airborne now", size=10, color=MUTED, ha="right")

    def _update_swim(self, snap: dict) -> None:
        t = snap["t"]
        dt, nb = self.sw_dt, self.sw_nb
        e = (t - SWIM_WINDOW_S + np.arange(nb + 1) * (SWIM_WINDOW_S / nb)) / dt
        tot = np.diff(e)
        ec = np.clip(e, 0.0, self.sw_N)
        i0 = np.minimum(np.floor(ec).astype(int), self.sw_N - 1)
        fr = (ec - i0)[:, None]
        valid = np.diff(ec)
        for d, im in enumerate(self.sw_imgs):
            C = self.sw_C[d]
            Ce = C[i0] + fr * (C[i0 + 1] - C[i0])
            s = np.diff(Ce, axis=0) + self.sw_empty[None, :] * (tot - valid)[:, None]
            im.set_data((s / tot[:, None])[None, :, :].clip(0, 1))
        airborne = 0
        for d, st in enumerate(snap["drone_states"]):
            air = st in ("outbound", "applying", "return")
            airborne += air
            c = DRONE_STATE_COLORS[st]
            self.sw_dot[d].set_facecolor(c)
            self.sw_state[d].set_text(STATE_NAMES[st])
            self.sw_state[d].set_color(_mix(c, "#ffffff", 0.35))
        self.sw_air.set_text(f"{airborne} / {len(self.sw_imgs)}")
        self.sw_air.set_color(GREEN if airborne else MUTED)

    # ------------------------------------------------------------- parcels
    def _build_parcels(self) -> None:
        g = self.g_pc = self._main_group(0.62)
        self.panel(g, 1384, 564, W - M, 800, "PARCELS", "share of target dose applied")
        tl = self.tl
        dose = float(self.kp["dose_kg_per_m2"])
        crops = {p.pid: p.crop for p in self.res.site.parcels}
        self.pc = []
        for j, pid in enumerate(tl.parcels):
            y = 622 + j * 46
            name = _parcel_label(pid, crops.get(pid, ""))
            self.text(g, 1404, y, name, size=11.5, color=INK)
            ha = tl.parcel_target_kg[j] / dose / 1e4
            self.text(g, 1404 + self.tw(name, 11.5) + 10, y, f"{ha:.2f} ha", size=9.5, color=MUTED)
            pct = self.text(g, 1844, y, "", size=11, weight="bold", family=MONO, ha="right", dyn=True)
            self.rect(g, 1404, y + 14, 440, 9, TRACK, z=2)
            bar = self.rect(g, 1404, y + 14, 0, 9, CHIP, z=3, dyn=True)
            self.pc.append((pct, bar))

    def _update_parcels(self, snap: dict) -> None:
        for (pct, bar), f in zip(self.pc, snap["parcel_frac"]):
            f = _clip01(f)
            done = f >= 0.999
            bar.set_width(440 * f)
            bar.set_facecolor(GREEN if done else CHIP)
            pct.set_text(f"{100 * f:5.1f} %")
            pct.set_color(GREEN if done else INK)

    # ------------------------------------------------------------- area chart
    def _build_chart(self) -> None:
        g = self.g_ch = self._main_group(0.72)
        self.panel(g, 628, 816, 1352, 1032, "APPLIED OVER TIME", "tonnes, stacked")
        tl = self.tl
        self.cx0, self.cx1, self.cy0, self.cy1 = 700.0, 1330.0, 866.0, 990.0
        self.cymax = 1.08 * max(tl.target_kg, float(tl.applied_kg[-1]), 1e-6)
        lx = 1332.0
        for lab, c in ((self.eff_name.lower(), FXC), ("drones", DRONES)):
            self.text(g, lx, 844, lab, size=9.5, color=MUTED, ha="right")
            lx -= self.tw(lab, 9.5) + 18
            self.rect(g, lx, 838, 12, 12, c, z=3)
            lx -= 16
        self.rect(g, self.cx0, self.cy1, self.cx1 - self.cx0, 1, DIM, z=2)
        yt = self._cy(tl.target_kg)
        segs = [[(x, yt), (min(x + 6, self.cx1), yt)] for x in np.arange(self.cx0, self.cx1, 10)]
        lc = LineCollection(segs, colors=MUTED, linewidths=1.0, transform=g.tr, zorder=2)
        self.ax.add_collection(lc, autolim=False)
        g.add(lc)
        self.text(g, self.cx0 - 8, yt, f"{self.target_t:,.{self.tdec}f} t", size=9, color=MUTED, ha="right")
        self.text(g, self.cx0 - 8, self.cy1, "0", size=9, color=MUTED, ha="right")
        th = tl.t_end / 3600.0
        step = next((s for s in (1, 2, 3, 6, 12, 24, 48) if th / s <= 6), 96)
        for hv in np.arange(0, th + 1e-9, step):
            x = self._cx(hv * 3600)
            self.rect(g, x - 0.5, self.cy1, 1, 5, DIM, z=2)
            self.text(g, x, self.cy1 + 16, f"{hv:.0f} h", size=9, color=MUTED, ha="center")
        self.ch_ms = self.patch(g, Polygon(np.zeros((3, 2)), closed=True, facecolor=FXC, edgecolor="none",
                                           transform=g.tr, zorder=3), alpha=0.85, dyn=True)
        self.ch_dr = self.patch(g, Polygon(np.zeros((3, 2)), closed=True, facecolor=DRONES, edgecolor="none",
                                           transform=g.tr, zorder=3), alpha=0.75, dyn=True)
        self.ch_line = self.rect(g, self.cx0, self.cy0 - 4, 1.5, self.cy1 - self.cy0 + 4, INK, z=4, alpha=0.7,
                                 dyn=True)
        self.ch_dot = self.patch(g, Circle((self.cx0, self.cy1), 4.5, facecolor=INK, edgecolor="none", transform=g.tr,
                                           zorder=5), dyn=True)

    def _cx(self, t: float) -> float:
        return self.cx0 + (self.cx1 - self.cx0) * t / max(self.tl.t_end, 1e-9)

    def _cy(self, kg) -> Any:
        return self.cy1 - (self.cy1 - self.cy0) * np.asarray(kg) / self.cymax

    def _update_chart(self, snap: dict) -> None:
        tl = self.tl
        t = snap["t"]
        k = int(np.searchsorted(tl.t, t, side="right"))
        ts = np.append(tl.t[:k], t)
        ms = np.append(tl.ms_kg[:k], snap["ms_t"] * 1000.0)
        tot = ms + np.append(tl.drone_kg[:k], snap["drone_t"] * 1000.0)
        xs = self._cx(ts)
        base = np.full_like(xs, self.cy1)
        self.ch_ms.set_xy(np.column_stack([np.concatenate([xs, xs[::-1]]),
                                           np.concatenate([self._cy(ms), base[::-1]])]))
        self.ch_dr.set_xy(np.column_stack([np.concatenate([xs, xs[::-1]]),
                                           np.concatenate([self._cy(tot), self._cy(ms)[::-1]])]))
        x = self._cx(t)
        self.ch_line.set_x(x - 0.75)
        self.ch_dot.center = (x, float(self._cy(tot[-1])))

    # ------------------------------------------------------------- event log
    def _story_events(self) -> list[dict[str, Any]]:
        res, tl, kp = self.res, self.tl, self.kp
        ev: list[dict[str, Any]] = [dict(t=0.0, title="Mothership leaves the depot",
                                         detail=f"hopper pre-loaded with {tl.hopper0_kg / 1000:.1f} t", color=ACTUAL)]
        if self.visits:
            ev.append(dict(t=float(self.v_arrive[0]), title=f"Parked at station 1 of {len(self.visits)}",
                           detail=f"{res.n_drones} drones start sorties · {self.eff_name.lower()} reach 40 ft",
                           color=GREEN))
        top = sorted((e for e in res.events if e["kind"] == "tender_topup"), key=lambda e: e["t"])
        if top:
            ev.append(dict(t=float(top[0]["t"]), title="Supply truck tops up the hopper",
                           detail=f"+{float(top[0]['kg']):,.0f} kg · {res.cfg.tender_trucks} trucks shuttle from the depot",
                           color=CHIP))
        sw = sorted((e for e in res.events if e["kind"] == "battery_swap"), key=lambda e: e["t"])
        if sw:
            e = sw[0]
            ev.append(dict(t=float(e["t"]), title="First battery swap",
                           detail=f"D{int(e['drone']) + 1} at {100 * float(e.get('soc', 0)):.0f} % charge · "
                                  f"{res.cfg.drone.swap_time_s:.0f} s hot-swap on the mothership",
                           color=DRONE_STATE_COLORS["swap"]))
        dets = {d["oid"]: d for d in res.detours}
        for o in self.obst:
            e = o["ev"]
            name = "waterlogged patch" if e["obstacle_type"].startswith("water") else e["obstacle_type"]
            d = dets.get(e["oid"])
            detail = (f"re-planned online · bypass {d['side']} · {d['length_m']:.0f} m detour" if d
                      else e.get("action", "monitored"))
            ev.append(dict(t=o["t"], title=f"{o['sensor']}: unknown {name}",
                           detail=f"{e['range_m']:.0f} m ahead · {detail}", color=OBST, alert=True))
        dep_t, dep_cum = tl._dep_t, tl._dep_cum
        for p in (0.25, 0.5, 0.75):
            j = int(np.searchsorted(dep_cum, p * tl.target_kg))
            if j < len(dep_t):
                tp = float(dep_t[j])
                done = int(np.sum(self.v_depart <= tp))
                ev.append(dict(t=tp, title=f"{int(p * 100)} % of the target on the beds",
                               detail=f"{dep_cum[j] / 1000:,.{self.tdec}f} t after {tp / 3600:.1f} h · "
                                      f"{done} of {len(self.visits)} stations done", color=AMBER))
        if math.isfinite(self.t_home):
            ev.append(dict(t=self.t_home, title="Mission complete · back at the depot",
                           detail=f"{self.fin['cells_done']:,} of {self.fin['n_cells']:,} cells at full dose · "
                                  f"{kp['mission_time_h']:.1f} h", color=AMBER))
        ev.sort(key=lambda e: e["t"])
        gap = 0.9 * self.ck.k
        last = -1e9
        out = []
        for e in ev:
            v = max(self.ck.v_of(e["t"]), last + gap)
            if v > self.ck.outro - 0.3 * self.ck.k:
                break
            e["v"] = last = v
            out.append(e)
        return out

    def _build_feed(self) -> None:
        g = self.g_fd = self._main_group(0.82)
        self.panel(g, 1384, 816, W - M, 1032, "EVENT LOG")
        self.events = self._story_events()
        for e in self.events:
            e["fs"] = 10.5
            while e["fs"] > 8 and self.tw(e["title"], e["fs"], "bold") > 330:
                e["fs"] -= 0.5
            e["dfs"] = 9.5
            while e["dfs"] > 7.5 and self.tw(e["detail"], e["dfs"]) > 412:
                e["dfs"] -= 0.5
        clip = Rectangle((1396, 856), 456, 172, transform=g.tr)
        self.fd_rows = []
        for _ in range(4):
            bx = self.box(g, 1404, 0, 1844, 50, fc=ROW, ec=ROW, r=6, lw=1.4, z=2, dyn=True)
            acc = self.rect(g, 1404, 0, 4, 50, AMBER, z=3, dyn=True)
            tit = self.text(g, 1420, 0, "", size=10.5, weight="bold", dyn=True)
            tim = self.text(g, 1832, 0, "", size=9, family=MONO, color=MUTED, ha="right", dyn=True)
            det = self.text(g, 1420, 0, "", size=9.5, color=MUTED, dyn=True)
            for a in (bx, acc, tit, tim, det):
                a.set_clip_path(clip)
            self.fd_rows.append((bx, acc, tit, tim, det))

    def _update_feed(self, v: float) -> None:
        g = self.g_fd
        k = self.ck.k
        shown = [e for e in self.events if e["v"] <= v]
        p = _ease_out((v - shown[-1]["v"]) / (0.4 * k)) if shown else 1.0
        for r, (bx, acc, tit, tim, det) in enumerate(self.fd_rows):
            j = len(shown) - 1 - r
            if j < 0 or (r == 3 and p >= 1.0):
                for a in (bx, acc, tit, tim, det):
                    g.own(a, 0.0)
                continue
            e = shown[j]
            y = 862 + (r - 1 + p) * 56
            al = p if r == 0 else (1.0 - p if r == 3 else 1.0)
            bx.set_y(y)
            acc.set_y(y)
            acc.set_facecolor(e["color"])
            tit.set_position((1420, y + 16))
            tim.set_position((1832, y + 16))
            det.set_position((1420, y + 35))
            tit.set_text(e["title"])
            tit.set_fontsize(e["fs"])
            tit.set_color(e["color"] if e.get("alert") else INK)
            tim.set_text(f"T+{e['t'] / 3600:.1f} h")
            det.set_text(e["detail"])
            det.set_fontsize(e["dfs"])
            fl = 1.0 - _clip01((v - e["v"]) / (1.8 * k))
            bx.set_facecolor(_mix(ROW, e["color"], (0.30 if e.get("alert") else 0.16) * fl))
            bx.set_edgecolor(_mix(ROW, e["color"], (0.9 if e.get("alert") else 0.3) * fl))
            for a in (bx, acc, tit, tim, det):
                g.own(a, al)

    # ------------------------------------------------------------- footer
    def _build_footer(self) -> None:
        g = self._main_group(0.9)
        self.text(g, M, 1058, _sources_short(), size=9, color=MUTED)

    # ------------------------------------------------------------- outro
    def _build_outro(self) -> None:
        k = self.ck.k
        imp, kp, fin = self.imp, self.kp, self.fin
        F = {r["key"]: r["value"] for r in factor_table()}
        g = self._outro_group(0.35, dy=20.0)
        title = (f"PARTIAL RUN  ·  {100 * self.scope:.0f} % OF THE TARGET APPLIED" if self.partial
                 else "MISSION COMPLETE")
        self.text(g, 960, 150, title, size=13, weight="bold", color=AMBER, ha="center")
        hw = self.tw(title, 13, "bold") / 2 + 22
        self.rect(g, 960 - hw - 170, 149, 170, 1.5, AMBER, z=3)
        self.rect(g, 960 + hw, 149, 170, 1.5, AMBER, z=3)
        h = kp["mission_time_h"]
        where = (f"{self.area_ha * self.scope:.2f} of {self.area_ha:.2f} ha" if self.partial
                 else f"{self.area_ha:.2f} ha")
        if self.cart:
            head = f"{int(kp['cartridges_placed']):,} seed cartridges placed on {where} in {h:.0f} h"
        else:
            head = f"{fin['applied_t']:,.0f} t of shredded prunings spread on {where} in {h:.0f} h"
        self.text(g, 960, 200, head, size=30, weight="bold", ha="center")
        why = ("pressed from prunings that would otherwise be burned in open piles" if self.cart
               else "instead of burning them in open piles")
        self.text(g, 960, 246, f"{why} · 1 mothership + {self.res.n_drones} drones · {len(self.visits)} stations · "
                  f"{fin['cells_done']:,} of {fin['n_cells']:,} cells at {kp['dose_kg_per_m2']:g} kg/m²",
                  size=14, color=MUTED, ha="center")
        cd = _dec(imp["total_co2e_t"])
        tiles = [
            (imp["total_co2e_t"], cd, "", "t CO₂e stored + avoided", GREEN,
             f"{imp['soil_co2e_t']:,.{cd}f} t soil carbon ({F['soil_co2e']:g} t CO₂e/ha at 15 kg/m², scaled by dose) "
             f"+ {imp['burn_ghg_avoided_co2e_t']:,.1f} t of CH₄ and N₂O that open burning would emit"),
            (imp["pm25_avoided_kg"], _dec(imp["pm25_avoided_kg"]), "", "kg PM2.5 smoke avoided", SKY,
             f"{imp['burn_avoided_dm_t']:,.0f} t of dry matter not burned in the open × {F['ef_pm25']:g} g PM2.5 "
             f"per kg (Andreae & Merlet 2001)"),
            (imp["n_retained_kg"], _dec(imp["n_retained_kg"]), "", "kg nitrogen retained", CHIP,
             f"{F['n_retained']:g} kg N/ha at 15 kg/m², scaled by dose · soil organic carbon "
             f"+{imp['soc_increase_pp']:.2f} pp (REGENSOL trials)"),
        ]
        if self.cart:
            spp = ", ".join(s.replace("_", " ") for s in imp["species"])
            tiles.append((float(kp["cartridges_placed"]), 0, "", "seed cartridges placed", FXC,
                          f"{int(kp['seeds_placed']):,} seeds of {len(imp['species'])} native species: {spp} · "
                          f"wood chips + mycorrhiza"))
        elif self.partial:
            tiles.append((self.area_ha * self.scope, 2, "", "ha mulched", AMBER,
                          f"{100 * self.scope:.0f} % of the {self.area_ha:.2f} ha target. Partial run: impacts are "
                          f"scaled to the mulched area and no € comparison is made"))
        else:
            reg, cost = imp["regensol_eur_per_1000m2"], imp["cost_eur_per_1000m2"]
            tiles.append((float(imp["saving_eur_total"] or 0.0), 0, "€", "saved vs REGENSOL manual", AMBER,
                          f"{cost:,.0f} vs {reg:,.0f} € per 1000 m² ({100 * (cost - reg) / reg:+.0f} %): the fleet "
                          f"replaces tractor transport + spreading"))
        self.out_tiles = []
        tw = (W - 2 * M - 3 * 16) / 4
        for j, (val, dec, pre, unit, col, detail) in enumerate(tiles):
            x0 = M + j * (tw + 16)
            gt = self._outro_group(0.6 + 0.12 * j, dy=30.0)
            self.box(gt, x0, 290, x0 + tw, 560)
            self.rect(gt, x0, 290, tw, 4, col, z=3)
            num = self.text(gt, x0 + 28, 362, "", size=40, weight="bold", color=col, va="baseline")
            self.text(gt, x0 + 28, 400, unit, size=15, color=INK)
            lines = textwrap.wrap(_pretty(detail), 50)
            for li, line in enumerate(lines[:4]):
                self.text(gt, x0 + 28, 446 + li * 22, line, size=10.5, color=MUTED)
            self.out_tiles.append((num, val, dec, pre, gt))
        self.v_countup_end = max(gt.t_in for *_, gt in self.out_tiles) + 1.3 * k
        # avoided burning emissions
        gb = self._outro_group(1.25)
        self.box(gb, M, 584, 952, 790)
        self.rect(gb, M + 20, 584, 34, 3, OBST, z=3)
        self.text(gb, M + 20, 614, "IF THE PILES HAD BEEN BURNED  ·  emissions avoided", size=11.5, weight="bold")
        burn = [(f"{imp['burn_ch4_avoided_kg']:,.0f}", "kg CH₄"), (f"{imp['burn_n2o_avoided_kg']:,.1f}", "kg N₂O"),
                (f"{imp['co_avoided_kg'] / 1000:,.1f}", "t CO"), (f"{imp['nox_avoided_kg']:,.0f}", "kg NOx"),
                (f"{imp['pm25_avoided_kg']:,.0f}", "kg PM2.5")]
        cw = (952 - M - 40) / len(burn)
        for j, (val, lab) in enumerate(burn):
            x = M + 20 + j * cw
            self.text(gb, x, 676, val, size=21, weight="bold", color=INK)
            self.text(gb, x, 708, lab, size=10.5, color=MUTED)
        self.text(gb, M + 20, 758, _pretty(f"Plus {imp['biogenic_co2_not_released_t']:,.0f} t biogenic CO2 not "
                                           f"released (context only, not counted). IPCC 2006 V4 Table 2.5 factors."),
                  size=9.5, color=MUTED)
        go = self._outro_group(1.4)
        self.box(go, 968, 584, W - M, 790)
        self.rect(go, 988, 584, 34, 3, ACTUAL, z=3)
        self.text(go, 988, 614, "THE MISSION", size=11.5, weight="bold")
        n_obs = len(kp.get("obstacles_detected", []))
        shifts = max(1, round(float(kp["mission_days_10h_shifts"])))
        ops = [(f"{h:.1f} h", f"mission time (≈ {shifts} shift{'s' if shifts > 1 else ''} of 10 h)"),
               (f"{len(self.visits)}", "drone stations"),
               (f"{int(kp['sorties']):,}", "drone sorties"),
               (f"{int(kp['battery_swaps']):,}", "battery swaps"),
               (f"{int(kp['tender_trips']):,}", "supply-truck trips"),
               (f"{n_obs}", f"unknown obstacles bypassed ({kp['detour_m']:.0f} m)")]
        cw2 = (W - M - 988 - 20) / 3
        for j, (val, lab) in enumerate(ops):
            x = 988 + (j % 3) * cw2
            y = 660 + (j // 3) * 68
            self.text(go, x, y, val, size=19, weight="bold", color=INK)
            self.text(go, x, y + 28, lab, size=10, color=MUTED)
        gs = self._outro_group(1.6, dy=16.0)
        self.text(gs, M, 826, "ASSUMPTIONS & SOURCES", size=10.5, weight="bold", color=AMBER)
        lines = textwrap.wrap(_sources_text(), 190)[:7]
        lines.append("Mission figures: simulated PruneMaxxing mission on the Viladecans digital twin (DEM 10 m); the "
                     "mission clock assumes continuous operation from 07:00 on day 1.")
        for li, line in enumerate(lines):
            self.text(gs, M, 856 + li * 24, line, size=10.5, color=MUTED)
        gc = self._outro_group(1.9, dy=10.0)
        self.text(gc, 960, 1046, "PRUNEMAXXING  ·  AGR-E2  ·  Unite! Tri-City Hackathon", size=11, weight="bold",
                  color=AMBER, ha="center")

    def _update_outro(self, v: float) -> None:
        k = self.ck.k
        for num, val, dec, pre, gt in self.out_tiles:
            u = _ease_out((v - gt.t_in) / (1.3 * k))
            num.set_text(f"{pre}{val * u:,.{dec}f}")

    # ------------------------------------------------------------- frame
    def frame(self, i: int) -> np.ndarray:
        """RGB frame i (the returned buffer is reused by the next call)."""
        v = float(self.ck.v[i])
        if v >= self.v_frozen and self._frozen:
            return self._out
        t = float(self.ck.t[i])
        snap = self.tl.at(t)
        if self.partial:
            snap = self._rescale(snap)
        for g in self.groups:
            g.update(v)
        self._update_intro(v)
        if self.ck.intro - 0.2 * self.ck.k <= v < self.ck.outro + 0.7 * self.ck.k:
            self._update_header(i, snap)
            self._update_ribbon(i, v, snap)
            self._update_ring(snap)
            self._update_tiles(snap)
            self._update_sankey(i, v, snap)
            self._update_donut(snap)
            self._update_swim(snap)
            self._update_parcels(snap)
            self._update_chart(snap)
            self._update_feed(v)
        if v >= self.ck.outro:
            self._update_header(i, snap)
            self._update_outro(v)
        self._draw(self.v_steady[0] <= v < self.v_steady[1])
        self._frozen = v >= self.v_frozen
        return self._fit(np.asarray(self.fig.canvas.buffer_rgba()))

    def _draw(self, steady: bool) -> None:
        canvas = self.fig.canvas
        rend = canvas.get_renderer()
        if steady != self._steady:
            for a in self.dyn:
                a.set_animated(steady)
            self._steady = steady
            self._bg = None
        if steady and self._bg is None:
            canvas.restore_region(self._base)
            self.ax.draw(rend)
            self._bg = canvas.copy_from_bbox(self.fig.bbox)
        if steady:
            canvas.restore_region(self._bg)
            for a in self.dyn:
                a.draw(rend)
        else:
            canvas.restore_region(self._base)
            self.ax.draw(rend)

    def _fit(self, rgba: np.ndarray) -> np.ndarray:
        """Copy into the output buffer, centred and padded to multiples of 8 (libx264 macro blocks)."""
        out = self._out
        w, h = self.size
        hh, ww = min(h, rgba.shape[0]), min(w, rgba.shape[1])
        oy, ox = (out.shape[0] - h) // 2, (out.shape[1] - w) // 2
        out[oy:oy + hh, ox:ox + ww] = rgba[:hh, :ww, :3]
        return out

    def close(self) -> None:
        plt.close(self.fig)


def render_infographic(res, out_path, fps: int = 24, seconds: float = 60.0, size: tuple[int, int] = (1920, 1080),
                       preview: bool = False, progress: Callable[[int, int], None] | None = None) -> Path:
    """Render the animated mission infographic to an MP4 and return its path.

    ``preview=True`` renders a short (<= 6 s) 960x540 version for quick checks.
    """
    if preview:
        seconds, size = min(float(seconds), 6.0), (960, 540)
    tl = MissionTimeline(res)
    clock = _Clock(tl.t_end, float(seconds), int(fps))
    art = _Infographic(res, tl, clock, size)
    out_path = Path(out_path)
    try:
        with video_writer(out_path, fps) as writer:
            for i in range(clock.n):
                writer.append_data(art.frame(i))
                if progress is not None:
                    progress(i + 1, clock.n)
    finally:
        art.close()
    return out_path
