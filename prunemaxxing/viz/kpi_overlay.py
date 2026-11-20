"""Animated KPI cards burned into video frames (cinematic lower-third + dashboard strip).

Values come from :class:`MissionTimeline` at the frame's mission time, so every
video shows the same numbers as the poster / infographic / HTML replay. Discrete
milestones (station finished, obstacle found) make their card flash briefly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

from prunemaxxing.viz.mission_stats import MissionTimeline
from prunemaxxing.viz.style import DRONE_PATH_COLORS

INK = (232, 239, 228)
MUTED = (154, 171, 148)
PANEL = (8, 12, 8)
TRACK = (44, 56, 44)
AMBER = (224, 151, 47)
GREEN = (123, 220, 106)
CYAN = (56, 214, 230)
FXC = (255, 224, 102)
LEAF = (124, 252, 0)


def _hex(c: str) -> tuple[int, int, int]:
    c = c.lstrip("#")
    return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))


@lru_cache(maxsize=64)
def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold" if bold else "normal"))
    return ImageFont.truetype(path, max(6, int(size)))


def _fmt_t(x: float) -> str:
    return f"{x:,.1f}" if x < 100 else f"{x:,.0f}"


def _fmt_big(x: float) -> str:
    if x >= 1e6:
        return f"{x / 1e6:.2f} M"
    if x >= 1e5:
        return f"{x / 1e3:,.0f} k"
    return f"{x:,.0f}"


@dataclass
class Card:
    label: str
    value: str
    sub: str = ""
    frac: float | None = None
    color: tuple[int, int, int] = AMBER
    split: float | None = None  # second colour share of the bar (drones vs effector)
    split_color: tuple[int, int, int] = FXC
    dots: list[tuple[int, int, int] | None] = field(default_factory=list)
    flash: float = 0.0


class KpiOverlay:
    """Stateful renderer: call :meth:`draw` once per frame in increasing time."""

    def __init__(self, res, timeline: MissionTimeline | None = None):
        self.res = res
        self.tl = timeline or MissionTimeline(res)
        self.cart = res.spec.mode.value == "cartridge"
        self.eff_name = "air cannons" if self.cart else "chip blower"
        self.det_t = sorted(e["t"] for e in res.events if e["kind"] == "obstacle_detected")
        self.n_obst = len(res.site.obstacles)
        self._last = {}
        self._flash = {}

    # ------------------------------------------------------------------ content
    def _milestones(self, key: str, value: int) -> float:
        prev = self._last.get(key)
        self._last[key] = value
        f = self._flash.get(key, 0.0) * 0.9
        if prev is not None and value > prev:
            f = 1.0
        self._flash[key] = f
        return f

    def cards(self, t: float, compact: bool = False) -> list[Card]:
        s = self.tl.at(t)
        obst = int(np.searchsorted(self.det_t, t, side="right"))
        f_st = self._milestones("stations", s["stations"])
        f_ob = self._milestones("obstacles", obst)
        applied = Card("APPLIED", f"{_fmt_t(s['applied_t'])} t",
                       f"of {s['target_t']:,.0f} t  ·  {100 * s['frac']:.1f} %", min(s["frac"], 1.0), AMBER)
        cells = Card("CELLS AT DOSE", f"{s['cells_done']:,} / {s['n_cells']:,}",
                     f"{100 * s['cells_done'] / max(s['n_cells'], 1):.0f} % of the field", s["cells_done"] / max(s["n_cells"], 1),
                     GREEN)
        stations = Card("STATIONS", f"{s['stations']} / {s['n_stations']}",
                        f"unknown obstacles found {obst} / {self.n_obst}", s["stations"] / max(s["n_stations"], 1), CYAN,
                        flash=max(f_st, f_ob))
        dots = []
        for d, st in enumerate(s["drone_states"]):
            dots.append(_hex(DRONE_PATH_COLORS[d % len(DRONE_PATH_COLORS)][1]) if st in ("outbound", "applying", "return")
                        else None)
        air = sum(1 for x in dots if x is not None)
        swarm = Card("DRONE SWARM", f"{s['sorties']:,} sorties", f"{s['swaps']:,} battery swaps  ·  {air} airborne",
                     None, (196, 181, 253), dots=dots)
        share = s["ms_t"] / max(s["applied_t"], 1e-9)
        if self.cart:
            eff = Card("MOTHERSHIP AIR CANNONS", f"{_fmt_big(s['ms_units'])} shots",
                       f"40 ft reach  ·  {100 * share:.1f} % of cartridges", 1.0, CYAN, split=share)
        else:
            eff = Card("MOTHERSHIP CHIP BLOWER", f"{_fmt_t(s['ms_t'])} t thrown",
                       f"40 ft reach  ·  {100 * share:.1f} % of the mulch", 1.0, AMBER, split=share)
        climate = Card("CLIMATE", f"{_fmt_t(s['total_co2e_t'])} t CO\u2082e",
                       f"stored + avoided  ·  PM2.5 \u2212{s['pm25_avoided_kg']:,.0f} kg", None, LEAF)
        if self.cart:
            last = Card("SEEDS PLANTED", _fmt_big(s.get("seeds", 0.0)),
                        f"{s.get('cartridges', 0.0):,.0f} cartridges with mycorrhiza", None, LEAF)
        else:
            last = Card("SAVED VS REGENSOL", f"\u20ac{(s['saving_eur'] or 0.0):,.0f}",
                        f"transport + spreading by robots", None, AMBER)
        if compact:
            return [applied, cells, stations, swarm, eff, climate]
        return [applied, cells, stations, swarm, eff, climate, last]

    # ------------------------------------------------------------------ drawing
    def draw(self, img: np.ndarray, t: float, box: tuple[int, int, int, int] | None = None, compact: bool = False,
             alpha: float = 1.0, bg_alpha: float = 0.74) -> np.ndarray:
        H, W = img.shape[:2]
        if box is None:
            s = H / 1080.0
            box = (int(40 * s), int(H - 104 * s), int(W - 40 * s), int(H - 26 * s))
        cards = self.cards(t, compact=compact)
        if alpha <= 0.01:
            return img
        im = Image.fromarray(img).convert("RGBA")
        ov = Image.new("RGBA", im.size, (0, 0, 0, 0))
        dr = ImageDraw.Draw(ov)
        (self._draw_bar if compact else self._draw_cards)(dr, cards, box, alpha, bg_alpha)
        return np.asarray(Image.alpha_composite(im, ov).convert("RGB"))

    @staticmethod
    def _fit(dr: ImageDraw.ImageDraw, text: str, size: int, width: float, bold: bool) -> ImageFont.FreeTypeFont:
        f = _font(size, bold)
        while size > 8 and dr.textlength(text, font=f) > width:
            size -= 1
            f = _font(size, bold)
        return f

    def _progress(self, dr, c: Card, x0, x1, y, h, a):
        dr.rounded_rectangle([x0, y, x1, y + h], radius=h / 2, fill=TRACK + (int(220 * a),))
        if c.frac is None:
            return
        xf = x0 + (x1 - x0) * float(np.clip(c.frac, 0, 1))
        if xf - x0 > h:
            dr.rounded_rectangle([x0, y, xf, y + h], radius=h / 2, fill=c.color + (int(255 * a),))
        if c.split:
            xs = x0 + (x1 - x0) * float(np.clip(1 - c.split, 0, 1))
            dr.rounded_rectangle([xs, y, x1, y + h], radius=h / 2, fill=c.split_color + (int(255 * a),))

    def _draw_cards(self, dr, cards: list[Card], box, a: float, bg_a: float) -> None:
        x0, y0, x1, y1 = box
        n = len(cards)
        gap = max(6, int((y1 - y0) * 0.09))
        w = (x1 - x0 - gap * (n - 1)) / n
        h = y1 - y0
        s = h / 100.0
        for i, c in enumerate(cards):
            cx0 = x0 + i * (w + gap)
            cx1 = cx0 + w
            fill = PANEL + (int(255 * bg_a * a),)
            dr.rounded_rectangle([cx0, y0, cx1, y1], radius=int(10 * s), fill=fill,
                                 outline=(44, 58, 45, int(255 * a)), width=1)
            if c.flash > 0.02:
                dr.rounded_rectangle([cx0 - 1, y0 - 1, cx1 + 1, y1 + 1], radius=int(11 * s),
                                     outline=c.color + (int(255 * a * c.flash),), width=max(2, int(3 * s)))
            dr.rounded_rectangle([cx0, y0, cx0 + 5 * s, y1], radius=int(3 * s), fill=c.color + (int(255 * a),))
            px = cx0 + 16 * s
            inner = w - 26 * s
            dr.text((px, y0 + 9 * s), c.label, font=_font(int(12.5 * s), True), fill=MUTED + (int(255 * a),))
            fv = self._fit(dr, c.value, int(30 * s), inner, True)
            dr.text((px, y0 + 26 * s), c.value, font=fv, fill=INK + (int(255 * a),))
            fs = self._fit(dr, c.sub, int(12.5 * s), inner, False)
            dr.text((px, y0 + 64 * s), c.sub, font=fs, fill=MUTED + (int(255 * a),))
            if c.dots:
                r = 6 * s
                for j, col in enumerate(c.dots):
                    cxd = cx1 - 14 * s - (len(c.dots) - 1 - j) * 3.2 * r
                    cyd = y0 + 16 * s
                    if col is None:
                        dr.ellipse([cxd - r, cyd - r, cxd + r, cyd + r], outline=MUTED + (int(200 * a),), width=1)
                    else:
                        dr.ellipse([cxd - r, cyd - r, cxd + r, cyd + r], fill=col + (int(255 * a),))
            if c.frac is not None:
                self._progress(dr, c, px, cx1 - 12 * s, y1 - 13 * s, 5 * s, a)

    def _draw_bar(self, dr, cards: list[Card], box, a: float, bg_a: float) -> None:
        x0, y0, x1, y1 = box
        h = y1 - y0
        s = h / 78.0
        dr.rounded_rectangle([x0, y0, x1, y1], radius=int(12 * s), fill=PANEL + (int(255 * bg_a * a),),
                             outline=(44, 58, 45, int(200 * a)), width=1)
        n = len(cards)
        w = (x1 - x0) / n
        for i, c in enumerate(cards):
            cx0 = x0 + i * w
            if i:
                dr.line([cx0, y0 + 12 * s, cx0, y1 - 12 * s], fill=(60, 76, 60, int(255 * a)), width=1)
            if c.flash > 0.02:
                dr.rounded_rectangle([cx0 + 4 * s, y0 + 4 * s, cx0 + w - 4 * s, y1 - 4 * s], radius=int(9 * s),
                                     outline=c.color + (int(255 * a * c.flash),), width=max(2, int(2.5 * s)))
            px = cx0 + 18 * s
            inner = w - 32 * s
            dr.ellipse([px, y0 + 14 * s, px + 8 * s, y0 + 22 * s], fill=c.color + (int(255 * a),))
            dr.text((px + 14 * s, y0 + 10 * s), c.label, font=_font(int(12 * s), True), fill=MUTED + (int(255 * a),))
            fv = self._fit(dr, c.value, int(27 * s), inner, True)
            dr.text((px, y0 + 30 * s), c.value, font=fv, fill=INK + (int(255 * a),))
            if c.frac is not None:
                self._progress(dr, c, px, cx0 + w - 16 * s, y1 - 12 * s, 4 * s, a)
            elif c.dots:
                r = 4.5 * s
                for j, col in enumerate(c.dots):
                    cxd = px + r + j * 3.0 * r
                    cyd = y1 - 10 * s
                    if col is None:
                        dr.ellipse([cxd - r, cyd - r, cxd + r, cyd + r], outline=MUTED + (int(200 * a),), width=1)
                    else:
                        dr.ellipse([cxd - r, cyd - r, cxd + r, cyd + r], fill=col + (int(255 * a),))
