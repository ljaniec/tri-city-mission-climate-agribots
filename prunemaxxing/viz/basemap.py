"""Raster basemap of the pilot site (orthophoto-like) for maps, posters, dashboards."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter, zoom
from shapely.geometry import Polygon

from prunemaxxing.geo.site import Site
from prunemaxxing.sim.world_frame import Terrain, frame_for_site, load_terrain

CROP_SOIL = {
    "sweet_potato_beauregard": ((112, 82, 56), (84, 118, 52)),
    "carboniato_ohenry": ((96, 70, 48), None),
    "cover_then_sweet_potato": ((104, 84, 58), (110, 132, 64)),
    "horticulture_mix": ((108, 80, 54), (92, 140, 70)),
}


@dataclass
class Basemap:
    rgb: np.ndarray  # (H, W, 3) uint8, row 0 = north
    extent: tuple[float, float, float, float]  # e0, e1, n0, n1 (UTM)
    res: float
    parcel_mask: np.ndarray  # (H, W) bool

    def px(self, e, n):
        e0, _, _, n1 = self.extent
        return (np.asarray(e) - e0) / self.res, (n1 - np.asarray(n)) / self.res

    def crop(self, e0: float, e1: float, n0: float, n1: float):
        c0, r0 = self.px(e0, n1)
        c1, r1 = self.px(e1, n0)
        c0, c1 = int(max(0, np.floor(c0))), int(min(self.rgb.shape[1], np.ceil(c1)))
        r0, r1 = int(max(0, np.floor(r0))), int(min(self.rgb.shape[0], np.ceil(r1)))
        ext = (self.extent[0] + c0 * self.res, self.extent[0] + c1 * self.res,
               self.extent[3] - r1 * self.res, self.extent[3] - r0 * self.res)
        return self.rgb[r0:r1, c0:c1], ext


def _noise(h: int, w: int, scales: tuple[float, ...], rng, amps=None) -> np.ndarray:
    out = np.zeros((h, w))
    amps = amps or [1.0 / (i + 1) for i in range(len(scales))]
    for s, a in zip(scales, amps):
        out += a * gaussian_filter(rng.standard_normal((h, w)), s, mode="wrap")
    out /= np.abs(out).max() + 1e-9
    return out


def build_basemap(site: Site, terrain: Terrain | None = None, res: float = 0.25, margin: float = 45.0,
                  seed: int = 3) -> Basemap:
    rng = np.random.default_rng(seed)
    e0, n0, e1, n1 = site.bounds
    e0, n0, e1, n1 = e0 - margin, n0 - margin, e1 + margin, n1 + margin
    W = int(round((e1 - e0) / res))
    H = int(round((n1 - n0) / res))
    fr = frame_for_site(site)
    if terrain is None:
        terrain = load_terrain(fr, site.width_m / 2 + margin + 10, site.height_m / 2 + margin + 10, 1.0)

    def P(geom_coords):
        return [((x - e0) / res, (n1 - y) / res) for x, y in geom_coords]

    # base: grassy farmland with noise
    base = np.zeros((H, W, 3))
    g = _noise(H, W, (3, 12, 40), rng)
    base[..., 0] = 92 + 14 * g
    base[..., 1] = 104 + 18 * g
    base[..., 2] = 62 + 8 * g
    img = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
    dr = ImageDraw.Draw(img, "RGBA")

    # greenhouses / houses beneath trees
    for it in site.scenery:
        if it.kind == "greenhouse":
            dr.polygon(P(it.geometry.exterior.coords), fill=(214, 220, 222, 255), outline=(170, 176, 180, 255))
            minx, miny, maxx, maxy = it.geometry.bounds
            for x in np.arange(minx + 2, maxx, 4.0):
                dr.line(P([(x, miny), (x, maxy)]), fill=(186, 194, 198, 255), width=1)
        elif it.kind == "house":
            dr.polygon(P(it.geometry.exterior.coords), fill=(176, 92, 58, 255), outline=(120, 60, 40, 255))
            c = it.geometry.centroid
            minx, miny, maxx, maxy = it.geometry.bounds
            dr.line(P([(minx, c.y), (maxx, c.y)]), fill=(140, 70, 44, 255), width=2)

    # parcels: soil + beds
    parcel_mask_img = Image.new("L", (W, H), 0)
    pm = ImageDraw.Draw(parcel_mask_img)
    for p in site.parcels:
        soil, leaf = CROP_SOIL.get(p.crop, ((104, 78, 52), None))
        dr.polygon(P(p.polygon.exterior.coords), fill=soil + (255,))
        pm.polygon(P(p.polygon.exterior.coords), fill=255)
    arr = np.asarray(img).astype(float)
    mask = np.asarray(parcel_mask_img) > 0
    # bed stripes (N-S) inside bedded area
    xs = e0 + (np.arange(W) + 0.5) * res
    for p in site.parcels:
        bx0, by0, bx1, by1 = p.polygon.buffer(-12.0, join_style=2).bounds
        soil, leaf = CROP_SOIL.get(p.crop, ((104, 78, 52), None))
        phase = ((xs - p.polygon.bounds[0]) / p.bed_spacing_m) % 1.0
        ridge = 0.5 + 0.5 * np.cos(2 * np.pi * (phase - 0.0))  # ridges at k*0.85, furrows between
        c0, r0 = int((bx0 - e0) / res), int((n1 - by1) / res)
        c1, r1 = int((bx1 - e0) / res), int((n1 - by0) / res)
        sub = arr[r0:r1, c0:c1]
        shade = (0.78 + 0.34 * ridge[c0:c1])[None, :, None]
        sub *= shade
        if leaf is not None:
            dots = (rng.random(sub.shape[:2]) < 0.10) & (ridge[c0:c1][None, :] > 0.75)
            sub[dots] = np.array(leaf) * (0.85 + 0.3 * rng.random((int(dots.sum()), 1)))
        arr[r0:r1, c0:c1] = sub
    tex = _noise(H, W, (1.5, 6, 25), rng)
    arr[mask] *= (1.0 + 0.10 * tex[mask])[:, None]
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    dr = ImageDraw.Draw(img, "RGBA")

    # tracks, road, depot
    for t in site.tracks:
        poly = t.line.buffer(t.width_m / 2, cap_style=2)
        if t.surface == "asphalt":
            dr.polygon(P(poly.exterior.coords), fill=(64, 66, 70, 255))
            edge = t.line.parallel_offset(t.width_m / 2 - 0.4)
            for side in (edge, t.line.parallel_offset(-(t.width_m / 2 - 0.4))):
                if not side.is_empty:
                    dr.line(P(side.coords), fill=(220, 220, 210, 200), width=max(1, int(0.15 / res)))
            L = t.line.length
            for d in np.arange(0, L, 9.0):
                a, b = t.line.interpolate(d), t.line.interpolate(min(L, d + 4.5))
                dr.line(P([(a.x, a.y), (b.x, b.y)]), fill=(236, 230, 200, 230), width=max(1, int(0.15 / res)))
        else:
            col = (176, 150, 108, 255) if t.surface == "dirt" else (168, 166, 158, 255)
            dr.polygon(P(poly.exterior.coords), fill=col)
            if t.surface == "dirt" and t.width_m >= 4.0:
                for off in (-0.85, 0.85):
                    rut = t.line.parallel_offset(off)
                    if not rut.is_empty:
                        dr.line(P(rut.coords), fill=(150, 124, 86, 255), width=max(1, int(0.35 / res)))
                dr.line(P(t.line.coords), fill=(128, 140, 84, 160), width=max(1, int(0.3 / res)))
    dr.polygon(P(site.depot.polygon.exterior.coords), fill=(176, 174, 166, 255), outline=(130, 128, 120, 255))
    for sp in site.stockpiles:
        r = 5.0
        dr.ellipse([((sp.x - r - e0) / res, (n1 - sp.y - r) / res), ((sp.x + r - e0) / res, (n1 - sp.y + r) / res)],
                   fill=(122, 82, 46, 255), outline=(90, 60, 34, 255))
    for d in site.ditches:
        poly = d.line.buffer(d.width_m / 2 + 0.8, cap_style=2)
        dr.polygon(P(poly.exterior.coords), fill=(86, 92, 58, 255))
        poly = d.line.buffer(d.width_m / 2, cap_style=2)
        dr.polygon(P(poly.exterior.coords), fill=(46, 82, 92, 255))

    # vegetation (drop shadows towards NE: sun from SW afternoon)
    def blob(x, y, r, col, shadow=True):
        if shadow:
            sx, sy = x + 0.45 * r, y + 0.35 * r
            dr.ellipse([((sx - r - e0) / res, (n1 - sy - r) / res), ((sx + r - e0) / res, (n1 - sy + r) / res)],
                       fill=(20, 26, 16, 90))
        dr.ellipse([((x - r - e0) / res, (n1 - y - r) / res), ((x + r - e0) / res, (n1 - y + r) / res)], fill=col)
        hr = 0.55 * r
        dr.ellipse([((x - 0.25 * r - hr - e0) / res, (n1 - y - 0.2 * r - hr) / res),
                    ((x - 0.25 * r + hr - e0) / res, (n1 - y - 0.2 * r + hr) / res)],
                   fill=tuple(min(255, int(c * 1.18)) for c in col[:3]) + (120,))

    for it in site.scenery:
        if it.geometry.geom_type != "Point":
            continue
        x, y = it.geometry.x, it.geometry.y
        if it.kind == "poplar":
            blob(x, y, 2.4, (58, 96, 46, 255))
        elif it.kind == "olive":
            blob(x, y, 3.0, (112, 128, 92, 255))
        elif it.kind == "carob":
            blob(x, y, 3.2, (50, 84, 42, 255))
        elif it.kind == "pine":
            blob(x, y, 4.2, (60, 92, 54, 255))
        elif it.kind == "citrus":
            blob(x, y, 1.6, (44, 88, 40, 255))
        elif it.kind == "reed":
            blob(x, y, 0.9, (150, 156, 96, 230), shadow=False)
        elif it.kind == "pole":
            dr.ellipse([((x - 0.3 - e0) / res, (n1 - y - 0.3) / res), ((x + 0.3 - e0) / res, (n1 - y + 0.3) / res)],
                       fill=(60, 50, 40, 255))

    arr = np.asarray(img).astype(float)
    # hillshade from terrain (exaggerated: the delta is very flat)
    gx = np.linspace(e0, e1, W) - fr.center_e
    gy = np.linspace(n1, n0, H) - fr.center_n
    step = 4
    zz = terrain.height_at(*np.meshgrid(gx[::step], gy[::step]))
    zz = zoom(zz, (H / zz.shape[0], W / zz.shape[1]), order=1)[:H, :W]
    dzdx = np.gradient(zz, axis=1) / res
    dzdy = -np.gradient(zz, axis=0) / res
    ex = 25.0
    az, alt = np.radians(225.0), np.radians(40.0)
    slope = np.arctan(ex * np.hypot(dzdx, dzdy))
    aspect = np.arctan2(-dzdx, dzdy)
    hs = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    hs = gaussian_filter((hs - hs.mean()) / (hs.std() + 1e-9), 6.0)
    arr *= (1.0 + 0.03 * np.clip(hs, -3, 3))[..., None]
    rgb = np.clip(arr, 0, 255).astype(np.uint8)
    return Basemap(rgb=rgb, extent=(e0, e1, n0, n1), res=res, parcel_mask=mask)


class CoveragePainter:
    """Accumulates per-sortie application footprints into a dose raster.

    DUST: soft square stamp over the zig-zag patch (kg spread over its area).
    CARTRIDGE: a dot per dropped puck along the bed ridge.
    Time must advance monotonically between ``advance`` calls (or ``reset``).
    """

    def __init__(self, basemap: Basemap, result, res: float = 0.25):
        self.bm = basemap
        self.res_ = result
        self.mode = result.spec.mode.value
        H, W = basemap.rgb.shape[:2]
        self.dose = np.zeros((H, W), dtype=np.float32)
        S = result.sorties
        M = getattr(result, "ms_shots", None) or {}
        t_all = [S["apply_t1"]]
        kind = [np.zeros(len(S["apply_t1"]), dtype=int)]
        idx = [np.arange(len(S["apply_t1"]))]
        if len(M.get("kg", [])):
            t_all.append(M["t1"])
            kind.append(np.ones(len(M["t1"]), dtype=int))
            idx.append(np.arange(len(M["t1"])))
        t_all, kind, idx = np.concatenate(t_all), np.concatenate(kind), np.concatenate(idx)
        o = np.argsort(t_all, kind="stable")
        self.t_done, self.job_kind, self.job_idx = t_all[o], kind[o], idx[o]
        self.cursor = 0
        self.full = float(result.spec.dose_kg_per_m2)
        rng = np.random.default_rng(5)
        self.grain = gaussian_filter(rng.standard_normal((H, W)).astype(np.float32), 0.7)

    def reset(self) -> None:
        self.dose[:] = 0
        self.cursor = 0

    def _stamp(self, k: int, ms: bool = False) -> None:
        S = self.res_.ms_shots if ms else self.res_.sorties
        cell = int(S["cell"][k])
        pat = self.res_.patterns[cell][int(S["pattern"][k])]
        kg = float(S["kg"][k])
        res = self.bm.res
        if self.mode == "cartridge":
            if ms:
                land = self.res_.ms_volley(k)["land"]
                n, xs, ys = len(land), land[:, 0], land[:, 1]
            else:
                n = self.res_.spec.units_per_sortie
                seg = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(pat, axis=0).T))])
                dpos = (np.arange(n) + 0.5) / n * seg[-1]
                xs = np.interp(dpos, seg, pat[:, 0])
                ys = np.interp(dpos, seg, pat[:, 1])
            cs, rs = self.bm.px(xs, ys)
            per = kg / n / (res * res)
            for c, r in zip(cs.astype(int), rs.astype(int)):
                if 1 <= r < self.dose.shape[0] - 1 and 1 <= c < self.dose.shape[1] - 1:
                    self.dose[r, c] += per
                    self.dose[r - 1, c] += 0.25 * per
                    self.dose[r + 1, c] += 0.25 * per
            return
        minx, miny = pat.min(axis=0) - 0.55
        maxx, maxy = pat.max(axis=0) + 0.55
        c0, r0 = self.bm.px(minx, maxy)
        c1, r1 = self.bm.px(maxx, miny)
        c0, r0, c1, r1 = int(c0), int(r0), int(np.ceil(c1)), int(np.ceil(r1))
        c0, r0 = max(c0, 0), max(r0, 0)
        c1, r1 = min(c1, self.dose.shape[1]), min(r1, self.dose.shape[0])
        if c1 <= c0 or r1 <= r0:
            return
        area = (maxx - minx) * (maxy - miny)
        self.dose[r0:r1, c0:c1] += kg / area

    def advance(self, t: float) -> None:
        k = int(np.searchsorted(self.t_done, t, side="right"))
        for j in range(self.cursor, k):
            self._stamp(int(self.job_idx[j]), bool(self.job_kind[j]))
        self.cursor = max(self.cursor, k)

    def rgba(self) -> np.ndarray:
        H, W = self.dose.shape
        out = np.zeros((H, W, 4), dtype=np.float32)
        if self.mode == "cartridge":
            v = np.clip(self.dose / (0.5 / (self.bm.res**2)), 0, 1)
            out[..., 0] = 0.70 + 0.10 * self.grain
            out[..., 1] = 0.62 + 0.10 * self.grain
            out[..., 2] = 0.34
            out[..., 3] = np.clip(v * 1.2, 0, 0.95)
        else:
            v = np.clip(self.dose / self.full, 0, 1.3)
            out[..., 0] = 0.70 + 0.08 * self.grain - 0.06 * (v > 1.05)
            out[..., 1] = 0.55 + 0.07 * self.grain - 0.05 * (v > 1.05)
            out[..., 2] = 0.36 + 0.05 * self.grain
            out[..., 3] = np.clip(v, 0, 1) ** 0.75 * 0.80
        return out


def completion_rgba(basemap: Basemap, res_cells, completion: np.ndarray, mode: str = "dust") -> np.ndarray:
    """RGBA overlay (basemap resolution) tinting mulched cells.

    ``res_cells`` = list of shapely polygons per cell (same order as completion).
    Cached rasterisation of the cell index map is stored on the basemap object.
    """
    if not hasattr(basemap, "_cell_index"):
        H, W = basemap.rgb.shape[:2]
        e0, _, _, n1 = basemap.extent
        im = Image.new("I", (W, H), -1)
        dr = ImageDraw.Draw(im)
        for i, poly in enumerate(res_cells):
            if poly is None or poly.is_empty:
                continue
            geoms = poly.geoms if poly.geom_type == "MultiPolygon" else [poly]
            for g in geoms:
                if isinstance(g, Polygon) and not g.is_empty:
                    dr.polygon([((x - e0) / basemap.res, (n1 - y) / basemap.res) for x, y in g.exterior.coords],
                               fill=i)
        basemap._cell_index = np.asarray(im, dtype=np.int32)  # type: ignore[attr-defined]
        rng = np.random.default_rng(11)
        basemap._grain = gaussian_filter(rng.standard_normal((H, W)), 0.8)  # type: ignore[attr-defined]
    idx = basemap._cell_index  # type: ignore[attr-defined]
    comp = np.nan_to_num(np.asarray(completion, dtype=float), nan=0.0)
    vals = np.where(idx >= 0, comp[np.clip(idx, 0, None)], 0.0)
    H, W = idx.shape
    out = np.zeros((H, W, 4), dtype=np.float32)
    if mode == "cartridge":
        col = np.array([0.42, 0.36, 0.20])
    else:
        col = np.array([0.24, 0.15, 0.08])
    grain = basemap._grain  # type: ignore[attr-defined]
    out[..., :3] = col * (1.0 + 0.25 * grain[..., None])
    out[..., 3] = np.clip(vals, 0, 1) ** 0.8 * 0.78
    return out
