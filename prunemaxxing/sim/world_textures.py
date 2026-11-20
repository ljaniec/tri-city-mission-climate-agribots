"""Procedural, seeded texture baking for the 1:1 PruneMaxxing MuJoCo world.

All ground painting is done in *world coordinates* (x east, y north, metres) so
that textures baked at different resolutions/windows (the 580 m ground texture,
the 8 cm/px field texture, the far "patchwork" apron) agree exactly at seams.

Noise is hash-based value noise on a world-anchored lattice, sampled with a
separable cubic B-spline (sparse interpolation matrices), so any rectangular
pixel grid can be evaluated cheaply and deterministically.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field

import numpy as np
from scipy import sparse
from shapely.geometry import Polygon

# ---------------------------------------------------------------------------
# Noise
# ---------------------------------------------------------------------------

_M1 = np.uint64(0x9E3779B97F4A7C15)
_M2 = np.uint64(0xC2B2AE3D27D4EB4F)
_M3 = np.uint64(0x165667B19E3779F9)
_F1 = np.uint64(0xFF51AFD7ED558CCD)
_F2 = np.uint64(0xC4CEB9FE1A85EC53)
_VALUE_NOISE_STD = 0.1384  # std of 2D cubic B-spline value noise with U(-.5,.5) lattice


def hash_u01(ix, iy, seed: int) -> np.ndarray:
    """Deterministic float64 in [0, 1) from integer lattice coords."""
    with np.errstate(over="ignore"):
        a = np.asarray(ix, dtype=np.int64).astype(np.uint64)
        b = np.asarray(iy, dtype=np.int64).astype(np.uint64)
        h = (a * _M1) ^ (b * _M2) ^ (np.uint64(seed & 0xFFFFFFFF) * _M3 + np.uint64(0x2545F4914F6CDD1D))
        h ^= h >> np.uint64(33)
        h *= _F1
        h ^= h >> np.uint64(33)
        h *= _F2
        h ^= h >> np.uint64(33)
    return (h >> np.uint64(11)).astype(np.float64) / float(2**53)


def _bspline_matrix(coords: np.ndarray, spacing: float, period: int | None = None):
    """Sparse (n x L) cubic B-spline interpolation matrix + lattice index origin."""
    p = np.asarray(coords, dtype=np.float64) / spacing
    i0 = np.floor(p).astype(np.int64)
    t = p - i0
    t2, t3 = t * t, t * t * t
    w = np.stack(
        [
            (1 - t) ** 3 / 6.0,
            (3 * t3 - 6 * t2 + 4) / 6.0,
            (-3 * t3 + 3 * t2 + 3 * t + 1) / 6.0,
            t3 / 6.0,
        ],
        axis=1,
    )
    idx = i0[:, None] + np.arange(-1, 3)[None, :]
    if period is not None:
        idx = np.mod(idx, period)
        lo, L = 0, period
    else:
        lo = int(idx.min())
        L = int(idx.max()) - lo + 1
    rows = np.repeat(np.arange(len(p)), 4)
    mat = sparse.csr_matrix((w.ravel().astype(np.float32), (rows, (idx - lo).ravel())), shape=(len(p), L))
    return mat, lo, L


def value_noise(xs, ys, spacing: float, seed: int, period: tuple[int, int] | None = None) -> np.ndarray:
    """Unit-std value noise on the grid ys x xs (rows = ys). Shape (len(ys), len(xs)), float32."""
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    wx, lox, lx = _bspline_matrix(xs, spacing, None if period is None else period[0])
    wy, loy, ly = _bspline_matrix(ys, spacing, None if period is None else period[1])
    gi, gj = np.meshgrid(np.arange(lox, lox + lx), np.arange(loy, loy + ly))
    lat = (hash_u01(gi, gj, seed) - 0.5).astype(np.float32)
    tmp = wy @ lat  # (ny, lx)
    out = (wx @ tmp.T).T  # (ny, nx)
    return np.ascontiguousarray(out, dtype=np.float32) / _VALUE_NOISE_STD


def fbm(xs, ys, spacing: float, seed: int, octaves: int = 4, gain: float = 0.5, lac: float = 2.0,
        min_spacing: float = 0.0, period: tuple[int, int] | None = None) -> np.ndarray:
    """Fractal sum of value noise, normalised to ~unit std."""
    out = None
    amp, tot, sp = 1.0, 0.0, spacing
    per = period
    for k in range(octaves):
        if sp < min_spacing:
            break
        n = value_noise(xs, ys, sp, seed * 131 + k * 7919, per)
        out = n * amp if out is None else out + n * amp
        tot += amp * amp
        amp *= gain
        sp /= lac
        if per is not None:
            per = (int(per[0] * lac), int(per[1] * lac))
    return out / np.float32(np.sqrt(tot))


def _skey(name: str) -> int:
    """Stable small integer key from a string (Python's hash() is salted per process)."""
    return zlib.crc32(name.encode()) % 997


def sstep(e0, e1, x):
    """Smoothstep from e0 to e1 (works for e0 > e1 too)."""
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def lerp_rgb(img: np.ndarray, col, mask: np.ndarray) -> None:
    """In-place img = img*(1-m) + col*m. col: rgb tuple or (H,W,3) array."""
    m = mask[..., None].astype(np.float32)
    col = np.asarray(col, dtype=np.float32)
    img *= 1.0 - m
    img += col * m


def to_u8(img: np.ndarray) -> np.ndarray:
    return np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)


def srgb(hexstr: str) -> tuple[float, float, float]:
    h = hexstr.lstrip("#")
    return tuple(int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


# ---------------------------------------------------------------------------
# Raster window helper
# ---------------------------------------------------------------------------


@dataclass
class Raster:
    """Pixel grid in world coords. Row 0 = north (image top)."""

    x0: float
    y0: float
    x1: float
    y1: float
    w: int
    h: int

    @property
    def resx(self) -> float:
        return (self.x1 - self.x0) / self.w

    @property
    def resy(self) -> float:
        return (self.y1 - self.y0) / self.h

    @property
    def xs(self) -> np.ndarray:
        return (self.x0 + (np.arange(self.w) + 0.5) * self.resx).astype(np.float64)

    @property
    def ys(self) -> np.ndarray:
        return (self.y1 - (np.arange(self.h) + 0.5) * self.resy).astype(np.float64)

    def window(self, bx0, by0, bx1, by1):
        """Slices (rows, cols) + local coordinate vectors for a world bbox, or None."""
        c0 = int(np.floor((bx0 - self.x0) / self.resx))
        c1 = int(np.ceil((bx1 - self.x0) / self.resx))
        r0 = int(np.floor((self.y1 - by1) / self.resy))
        r1 = int(np.ceil((self.y1 - by0) / self.resy))
        c0, c1 = max(c0, 0), min(c1, self.w)
        r0, r1 = max(r0, 0), min(r1, self.h)
        if c1 <= c0 or r1 <= r0:
            return None
        xs = self.x0 + (np.arange(c0, c1) + 0.5) * self.resx
        ys = self.y1 - (np.arange(r0, r1) + 0.5) * self.resy
        return (slice(r0, r1), slice(c0, c1)), xs, ys


def seg_dist(X, Y, a, b):
    """Distance + along-segment param (metres) from points to segment a-b."""
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    t = np.clip(((X - ax) * dx + (Y - ay) * dy) / max(L2, 1e-12), 0.0, 1.0)
    px = ax + t * dx - X
    py = ay + t * dy - Y
    return np.sqrt(px * px + py * py), t * np.sqrt(L2)


def rect_sdist(X, Y, x0, y0, x1, y1):
    """Signed distance to an axis-aligned rectangle (negative inside)."""
    dx = np.maximum(x0 - X, X - x1)
    dy = np.maximum(y0 - Y, Y - y1)
    out = np.sqrt(np.maximum(dx, 0) ** 2 + np.maximum(dy, 0) ** 2)
    inside = np.minimum(np.maximum(dx, dy), 0)
    return out + inside


def poly_sdist(X, Y, poly: Polygon):
    """Signed distance to a convex polygon (exact for convex, ok for our rectangles)."""
    pts = np.asarray(poly.exterior.coords)[:-1]
    if Polygon(pts).exterior.is_ccw is False:
        pts = pts[::-1]
    d = None
    sd = None
    for i in range(len(pts)):
        a, b = pts[i], pts[(i + 1) % len(pts)]
        di, _ = seg_dist(X, Y, a, b)
        d = di if d is None else np.minimum(d, di)
        ex, ey = b[0] - a[0], b[1] - a[1]
        cross = ex * (Y - a[1]) - ey * (X - a[0])
        s = cross >= 0
        sd = s if sd is None else (sd & s)
    return np.where(sd, -d, d)


# ---------------------------------------------------------------------------
# Scene description consumed by the painter (all in world coords)
# ---------------------------------------------------------------------------


@dataclass
class PaintParcel:
    pid: str
    crop: str
    x0: float
    y0: float
    x1: float
    y1: float
    bx0: float
    by0: float
    bx1: float
    by1: float
    pitch: float
    seed: int


@dataclass
class PaintScene:
    site_box: tuple[float, float, float, float]
    parcels: list[PaintParcel]
    tracks: list[tuple[str, list[tuple[float, float]], float, str]]  # id, pts, width, surface
    ditches: list[tuple[str, list[tuple[float, float]], float]]
    weeds: list[tuple[str, Polygon, str]]
    depot: Polygon
    stockpiles: list[tuple[float, float]]
    waterlogged: list[tuple[float, float, float]]  # x, y, r
    obstacle_footprints: list[tuple[str, Polygon]]
    trees: list[tuple[str, float, float, float, float]]  # kind, x, y, crown_r, height
    houses: list[Polygon]
    greenhouses: list[Polygon]
    orchard_box: tuple[float, float, float, float] | None
    greenhouse_box: tuple[float, float, float, float] | None
    yard_box: tuple[float, float, float, float] | None
    pine_box: tuple[float, float, float, float] | None
    poles: list[tuple[float, float]] = field(default_factory=list)
    seed: int = 1234


# Palette ---------------------------------------------------------------------
C_DRY = srgb("#a39a62")
C_GREEN = srgb("#5f7434")
C_LUSH = srgb("#4a6a26")
C_SOIL = srgb("#8a6f52")
C_SOIL_DARK = srgb("#5e4936")
C_DIRT = srgb("#b59d76")
C_RUT = srgb("#94805f")
C_ASPHALT = srgb("#3d3e40")
C_GRAVEL = srgb("#aca79c")
C_WATER = srgb("#2e3f3c")
C_MUD = srgb("#3f3226")


# ---------------------------------------------------------------------------
# Patchwork farmland (outside the site) — world function, also used for apron
# ---------------------------------------------------------------------------

_PATCH_TYPES = [
    # name, base colour, stripe colour, stripe pitch (m), stripe duty, noise amp
    ("stubble", srgb("#c2ad72"), srgb("#a8925a"), 1.6, 0.35, 0.05),
    ("plough", srgb("#806248"), srgb("#5f4834"), 0.9, 0.5, 0.06),
    ("artichoke", srgb("#6f7d5e"), srgb("#57503f"), 1.4, 0.45, 0.06),
    ("lettuce", srgb("#6f9a3c"), srgb("#6b5640"), 0.75, 0.5, 0.05),
    ("fallow", srgb("#9c9358"), srgb("#8a8250"), 3.0, 0.5, 0.08),
    ("alfalfa", srgb("#5d7d32"), srgb("#55722d"), 2.2, 0.5, 0.05),
    ("plastic", srgb("#2f2f30"), srgb("#7b644a"), 1.5, 0.55, 0.03),
    ("melon", srgb("#7c7445"), srgb("#58703a"), 2.0, 0.4, 0.07),
]


def patchwork(X: np.ndarray, Y: np.ndarray, seed: int, res: float) -> np.ndarray:
    """Agricultural patchwork of jittered rectangles (world fn). Returns float32 (H,W,3)."""
    cell = 95.0
    ci = np.floor(X / cell).astype(np.int64)
    cj = np.floor(Y / cell).astype(np.int64)
    # merge neighbours: field id from hash of (ci//sx, cj//sy) for elongated fields
    sx = 1 + (hash_u01(cj, 0, seed + 3) * 2.0).astype(np.int64)
    fi = np.floor_divide(ci, sx)
    fid_u = hash_u01(fi, cj, seed + 5)
    kind = (fid_u * len(_PATCH_TYPES)).astype(np.int64) % len(_PATCH_TYPES)
    ori = hash_u01(fi, cj, seed + 7) > 0.5
    img = np.zeros(X.shape + (3,), np.float32)
    lx = X - fi * sx * cell
    ly = Y - cj * cell
    for k, (_, base, stripe, pitch, duty, amp) in enumerate(_PATCH_TYPES):
        m = kind == k
        if not m.any():
            continue
        coord = np.where(ori, lx, ly)
        if pitch > 2.5 * res:
            ph = np.mod(coord / pitch, 1.0)
            s = sstep(duty - 0.12, duty + 0.12, ph) * sstep(1.0, 0.88, ph)
        else:
            s = np.full(X.shape, duty, np.float32)
        col = np.asarray(base, np.float32)[None, None] * (1 - s[..., None]) + np.asarray(stripe, np.float32)[None, None] * s[..., None]
        img[m] = col[m]
    # field tint jitter
    tint = (hash_u01(fi, cj, seed + 11) - 0.5) * 0.12
    img *= (1.0 + tint)[..., None].astype(np.float32)
    # field boundaries: tracks / hedges (dark-green lines)
    edge_x = np.minimum(np.mod(X, cell * 1.0), cell - np.mod(X, cell))
    edge_y = np.minimum(np.mod(Y, cell), cell - np.mod(Y, cell))
    same_left = np.floor_divide(np.floor((X - 1.5) / cell).astype(np.int64), sx) == fi
    same_right = np.floor_divide(np.floor((X + 1.5) / cell).astype(np.int64), sx) == fi
    vx = (~(same_left & same_right)).astype(np.float32) * sstep(2.2, 0.8, edge_x)
    vy = sstep(2.2, 0.8, edge_y)
    border = np.maximum(vx, vy)
    hedge = hash_u01(ci, cj, seed + 13) > 0.6
    bcol = np.where(hedge[..., None], np.asarray(srgb("#41522b"), np.float32), np.asarray(srgb("#b8a27a"), np.float32))
    img = img * (1 - border[..., None]) + bcol * border[..., None]
    return img


# ---------------------------------------------------------------------------
# Main ground painter
# ---------------------------------------------------------------------------


WEED_MARGIN = 8.0


def weed_sdf_on_grid(zid: str, poly: Polygon, xs, ys, seed: int, min_spacing: float = 0.0) -> np.ndarray:
    """Organic signed distance (negative inside) of a weed patch, world-coordinate noise."""
    x0, y0, x1, y1 = poly.bounds
    key = _skey(zid)
    X = np.asarray(xs, np.float32)[None, :]
    Y = np.asarray(ys, np.float32)[:, None]
    n_big = fbm(xs, ys, 9.0, seed + 930 + key, 3, min_spacing=min_spacing)
    n1 = fbm(xs, ys, 3.0, seed + 900 + key, 3, min_spacing=min_spacing)
    return rect_sdist(X, Y, x0, y0, x1, y1) + 3.0 * n_big + 1.0 * n1


def weed_sdf(zid: str, poly: Polygon, seed: int, res: float = 0.25):
    """Callable sdf(x, y) (bilinear lookup) matching the painted weed outline."""
    from scipy.ndimage import map_coordinates

    x0, y0, x1, y1 = poly.bounds
    xs = np.arange(x0 - WEED_MARGIN, x1 + WEED_MARGIN + res, res)
    ys = np.arange(y0 - WEED_MARGIN, y1 + WEED_MARGIN + res, res)
    grid = weed_sdf_on_grid(zid, poly, xs, ys, seed)

    def f(x, y):
        return float(map_coordinates(grid, [[(y - ys[0]) / res], [(x - xs[0]) / res]], order=1, mode="nearest")[0])

    return f


class GroundPainter:
    """Paints albedo for any world-aligned raster (deterministic, seam-free)."""

    def __init__(self, scene: PaintScene):
        self.s = scene
        self.seed = scene.seed

    # -- shared noise helpers ------------------------------------------------
    def n(self, xs, ys, spacing, key, octaves=4, min_px=1.5, res=0.1):
        return fbm(xs, ys, spacing, self.seed * 1000 + key, octaves=octaves, min_spacing=min_px * res)

    def paint(self, R: Raster, mulch: str | None = None):
        """Return dict with 'rgb' float32 (H,W,3); for mulch variants also masks."""
        s = self.s
        xs, ys = R.xs, R.ys
        res = min(R.resx, R.resy)
        H, W = R.h, R.w
        X = xs[None, :].astype(np.float32)
        Y = ys[:, None].astype(np.float32)
        out = {}

        # --- base meadow -----------------------------------------------------
        n_big = self.n(xs, ys, 60.0, 1, 3, res=res)
        n_mid = self.n(xs, ys, 9.0, 2, 4, res=res)
        n_fine = self.n(xs, ys, 1.2, 3, 3, res=res)
        g = sstep(-0.9, 0.9, 0.6 * n_big + 0.5 * n_mid)
        img = np.empty((H, W, 3), np.float32)
        img[:] = np.asarray(C_DRY, np.float32)
        lerp_rgb(img, C_GREEN, g * 0.85)
        soil_patch = sstep(1.1, 1.6, n_mid * 0.7 + n_fine * 0.5)
        lerp_rgb(img, srgb("#8d7a5a"), soil_patch * 0.6)
        speck = np.random.default_rng(self.seed + 17 + R.w).standard_normal((H, W)).astype(np.float32)
        img *= (1.0 + 0.07 * n_fine + 0.035 * speck)[..., None]

        # --- outside-site land use: patchwork fields ------------------------
        sx0, sy0, sx1, sy1 = s.site_box
        outside = rect_sdist(X, Y, sx0 - 14, sy0 - 34, sx1 + 18, sy1 + 22)
        pw_mask = sstep(4.0, 12.0, outside + 4.0 * n_mid)
        # keep orchard/greenhouse/yard/pines areas out of the patchwork
        for box in (s.orchard_box, s.greenhouse_box, s.yard_box, s.pine_box):
            if box is None:
                continue
            d = rect_sdist(X, Y, box[0] - 8, box[1] - 8, box[2] + 8, box[3] + 8)
            pw_mask = pw_mask * sstep(-2.0, 6.0, d)
        if pw_mask.max() > 0:
            Xf = np.broadcast_to(X, (H, W))
            Yf = np.broadcast_to(Y, (H, W))
            pw = patchwork(Xf, Yf, self.seed + 99, res)
            pw *= (1.0 + 0.06 * n_mid + 0.04 * speck)[..., None]
            lerp_rgb(img, pw, pw_mask)
            del pw

        self._paint_outside_features(R, img, X, Y, res, speck)

        # --- site verge: greener mown grass inside the site --------------
        inside = sstep(6.0, -2.0, rect_sdist(X, Y, sx0 - 8, sy0 - 24, sx1 + 10, sy1 + 10) + 3 * n_mid)
        verge = np.empty_like(img)
        verge[:] = np.asarray(srgb("#6c7a3d"), np.float32)
        lerp_rgb(verge, srgb("#8e8c55"), sstep(-0.5, 1.4, n_mid * 0.8 + n_fine * 0.4))
        verge *= (1.0 + 0.09 * n_fine + 0.04 * speck)[..., None]
        lerp_rgb(img, verge, inside * 0.9)
        del verge

        # --- parcels -----------------------------------------------------------
        mulch_layers = None
        if mulch is not None:
            mulch_layers = {
                "dust": img.copy(),
                "cart": img.copy(),
                "threshold": np.full((H, W), 255, np.uint8),
                "weight": np.zeros((H, W), np.uint8),
            }
        for p in s.parcels:
            self._paint_parcel(R, img, p, res, mulch_layers)

        # --- ditches, tracks, depot ---------------------------------------
        layers = [img] + ([mulch_layers["dust"], mulch_layers["cart"]] if mulch_layers else [])
        for L in layers:
            self._paint_ditches(R, L, res)
            self._paint_tracks(R, L, res)
            self._paint_depot(R, L, res)
            self._paint_weeds(R, L, res)
            self._paint_waterlogged(R, L, res)
            self._paint_ao(R, L, res)
        # gentle large-scale grading so nothing looks flat
        grade = 1.0 + 0.05 * n_big
        for L in layers:
            L *= grade[..., None]
        out["rgb"] = img
        if mulch_layers:
            out.update(mulch_layers)
        return out

    # -----------------------------------------------------------------------
    def _paint_outside_features(self, R, img, X, Y, res, speck):
        s = self.s
        # citrus orchard floor: tree rows N-S every 8 m, drip strip + mown alleys
        if s.orchard_box is not None:
            b = s.orchard_box
            win = R.window(b[0] - 10, b[1] - 10, b[2] + 10, b[3] + 10)
            if win is not None:
                (rs, cs), xs, ys = win
                Xl, Yl = xs[None, :], ys[:, None]
                inside = sstep(4.0, -2.0, rect_sdist(Xl, Yl, *b))
                rowd = np.abs(np.mod(Xl - b[0] + 4.0, 8.0) - 4.0)
                strip = sstep(1.6, 0.9, rowd)
                n = fbm(xs, ys, 3.0, self.seed + 501, 3)
                floor = np.empty(inside.shape + (3,), np.float32)
                floor[:] = np.asarray(srgb("#79803f"), np.float32)
                lerp_rgb(floor, srgb("#8a6c4f"), strip * (0.85 + 0.1 * n))
                floor *= (1 + 0.08 * n)[..., None]
                sub = img[rs, cs]
                lerp_rgb(sub, floor, inside)
        # greenhouse complex floor: compacted pale soil; crop rows under tunnels
        if s.greenhouse_box is not None:
            b = s.greenhouse_box
            win = R.window(b[0] - 8, b[1] - 8, b[2] + 8, b[3] + 8)
            if win is not None:
                (rs, cs), xs, ys = win
                Xl, Yl = xs[None, :], ys[:, None]
                inside = sstep(5.0, -1.0, rect_sdist(Xl, Yl, *b))
                n = fbm(xs, ys, 4.0, self.seed + 511, 3)
                sub = img[rs, cs]
                base = np.empty(inside.shape + (3,), np.float32)
                base[:] = np.asarray(srgb("#b4a588"), np.float32)
                base *= (1 + 0.06 * n)[..., None]
                lerp_rgb(sub, base, inside)
                for gpoly in s.greenhouses:
                    gx0, gy0, gx1, gy1 = gpoly.bounds
                    d = rect_sdist(Xl, Yl, gx0 + 0.2, gy0 + 0.2, gx1 - 0.2, gy1 - 0.2)
                    m = sstep(0.3, -0.3, d)
                    if not m.any():
                        continue
                    rows = np.abs(np.mod(Yl - gy0, 1.0) - 0.5)
                    crop = sstep(0.32, 0.18, rows)
                    col = np.empty(m.shape + (3,), np.float32)
                    col[:] = np.asarray(srgb("#6e5a44"), np.float32)
                    lerp_rgb(col, srgb("#5f8f34"), crop * (0.8 + 0.2 * n))
                    lerp_rgb(sub, col, m)
        # masia yard + pines duff
        for box, c1, c2, key in ((s.yard_box, "#b9a37f", "#9c8a6b", 521), (s.pine_box, "#7d7046", "#6d6038", 531)):
            if box is None:
                continue
            win = R.window(box[0] - 10, box[1] - 10, box[2] + 10, box[3] + 10)
            if win is None:
                continue
            (rs, cs), xs, ys = win
            Xl, Yl = xs[None, :], ys[:, None]
            n = fbm(xs, ys, 5.0, self.seed + key, 4, min_spacing=1.5 * res)
            inside = sstep(6.0, -2.0, rect_sdist(Xl, Yl, *box) + 3.0 * n)
            col = np.empty(inside.shape + (3,), np.float32)
            col[:] = np.asarray(srgb(c1), np.float32)
            lerp_rgb(col, srgb(c2), sstep(-0.6, 0.8, n))
            col *= (1 + 0.06 * n)[..., None]
            lerp_rgb(img[rs, cs], col, inside)

    # -----------------------------------------------------------------------
    def _paint_parcel(self, R, img, p: PaintParcel, res, mulch_layers):
        win = R.window(p.x0 - 1.5, p.y0 - 1.5, p.x1 + 1.5, p.y1 + 1.5)
        if win is None:
            return
        (rs, cs), xs, ys = win
        X = xs[None, :].astype(np.float32)
        Y = ys[:, None].astype(np.float32)
        h, w = len(ys), len(xs)
        seed = self.seed + p.seed * 37
        rng = np.random.default_rng(seed + R.w)
        speck = rng.standard_normal((h, w)).astype(np.float32)
        n_mid = fbm(xs, ys, 7.0, seed + 1, 4, min_spacing=1.5 * res)
        n_fine = fbm(xs, ys, 0.9, seed + 2, 3, min_spacing=1.5 * res)
        # anisotropic along-bed variation (beds run N-S)
        n_bed = value_noise(xs / 0.5, ys / 5.0, 1.0, seed + 3) * 0.6 + value_noise(xs / 0.25, ys / 1.5, 1.0, seed + 4) * 0.4

        parcel_m = sstep(0.6, -0.4, rect_sdist(X, Y, p.x0, p.y0, p.x1, p.y1) + 0.35 * n_fine)
        bed_m = sstep(0.25, -0.35, rect_sdist(X, Y, p.bx0, p.by0, p.bx1, p.by1))

        crop = p.crop
        soil = {
            "sweet_potato_beauregard": ("#8f7254", "#a3865f", "#634c37"),
            "carboniato_ohenry": ("#6d5540", "#7f6449", "#47362a"),
            "cover_then_sweet_potato": ("#836a4d", "#94795a", "#5c4734"),
            "horticulture_mix": ("#86694c", "#9b7d5a", "#5a4532"),
        }.get(crop, ("#86694c", "#9b7d5a", "#5a4532"))
        flank_c, crown_c, furrow_c = (np.asarray(srgb(c), np.float32) for c in soil)

        # ---- headland: compacted soil + tyre ruts ----
        head = np.empty((h, w, 3), np.float32)
        head[:] = np.asarray(srgb("#9b8466"), np.float32)
        lerp_rgb(head, srgb("#8a7459"), sstep(-0.8, 0.8, n_mid))
        head *= (1.0 + 0.07 * n_fine + 0.05 * speck)[..., None]
        # loop track around the bed area (centre of headland), two tyre pairs
        cx0, cy0 = p.x0 + 6.0, p.y0 + 6.0
        cx1, cy1 = p.x1 - 6.0, p.y1 - 6.0
        dloop = np.abs(rect_sdist(X, Y, cx0, cy0, cx1, cy1))
        rut = np.zeros((h, w), np.float32)
        for off in (-0.85, 0.85, -2.6, 2.6):
            rw = 0.19 if abs(off) < 1 else 0.15
            rut = np.maximum(rut, sstep(rw + 0.08, rw - 0.06, np.abs(dloop - abs(off)) + 0.03 * n_fine) * (1.0 if abs(off) < 1 else 0.55))
        # U-turn arcs (turn radius 6 m) at both ends of the bed area
        prng = np.random.default_rng(seed + 9)
        for _ in range(9):
            lane = p.bx0 + prng.uniform(4, p.bx1 - p.bx0 - 16)
            lane = p.x0 + (np.round((lane - p.x0) / p.pitch - 0.5) + 0.5) * p.pitch
            for end_y, sgn in ((p.by1, 1.0), (p.by0, -1.0)):
                if prng.random() < 0.5:
                    continue
                cxa, cya = lane + 6.0, end_y + sgn * 0.5
                rr = np.sqrt((X - cxa) ** 2 + (Y - cya) ** 2)
                half = (sgn * (Y - cya)) > 0
                for off in (-0.85, 0.85):
                    rut = np.maximum(rut, half * sstep(0.27, 0.12, np.abs(rr - (6.0 + off))) * 0.8)
        tread = 0.85 + 0.15 * np.sign(np.sin(Y * 40.0 + X * 40.0))
        head *= (1.0 - 0.16 * rut * tread)[..., None]
        lerp_rgb(head, srgb("#7c6a52"), rut * 0.35)
        # sparse weedy tufts on headland edges
        edge_d = -rect_sdist(X, Y, p.x0, p.y0, p.x1, p.y1)
        tufts = sstep(1.3, 2.2, n_fine * 1.2 + speck * 0.35) * sstep(2.5, 0.3, edge_d)
        lerp_rgb(head, srgb("#6f7a3a"), tufts * 0.8)

        # ---- beds: ridge/furrow stripes ----
        u = np.mod((X - p.x0) / p.pitch, 1.0)
        d = (np.where(u < 0.5, u, u - 1.0) * p.pitch).astype(np.float32)  # offset from ridge centre
        ad = np.abs(d)
        crown = sstep(0.19, 0.12, ad)
        furrow = sstep(0.24, 0.33, ad)
        bed = flank_c[None, None] * np.ones((h, w, 1), np.float32)
        lerp_rgb(bed, crown_c, crown)
        lerp_rgb(bed, furrow_c, furrow)
        # sun-side flank highlight (sun from WSW) adds depth even at distance
        west_flank = sstep(0.12, 0.2, -d) * sstep(0.3, 0.22, -d)
        bed *= (1.0 + 0.06 * west_flank - 0.05 * sstep(0.12, 0.2, d) * sstep(0.3, 0.22, d))[..., None]
        clods = 0.10 * n_fine + 0.07 * speck + 0.06 * n_bed
        bed *= (1.0 + clods)[..., None]
        bed *= (1.0 + 0.07 * n_mid)[..., None]
        # moisture darkening in furrows, patchy
        wet = sstep(0.2, 1.2, n_mid + 0.4 * n_bed) * furrow
        bed *= (1.0 - 0.18 * wet)[..., None]

        k_ridge = np.round((X - p.x0) / p.pitch).astype(np.int64)  # ridge index
        plants = None
        if crop in ("sweet_potato_beauregard", "horticulture_mix"):
            sp = 0.30 if crop == "sweet_potato_beauregard" else 0.36
            ky = (Y - p.by0) / sp
            jitter = hash_u01(k_ridge, 0, seed + 21).astype(np.float32)
            jj = np.floor(ky + jitter).astype(np.int64)
            fy = (ky + jitter - jj - 0.5) * sp
            pres = hash_u01(k_ridge, jj, seed + 22) > 0.08
            size = 0.7 + 0.6 * hash_u01(k_ridge, jj, seed + 23).astype(np.float32)
            ox = (hash_u01(k_ridge, jj, seed + 24).astype(np.float32) - 0.5) * 0.08
            if crop == "sweet_potato_beauregard":
                rad = 0.085 * size
                pcol = np.asarray(srgb("#4f7f22"), np.float32)
                block = None
            else:
                rad = 0.13 * size
                block = np.floor_divide(k_ridge - int(np.round((p.bx0 - p.x0) / p.pitch)), 14) % 4
                pcol = None
            r = np.sqrt((d - ox) ** 2 + fy**2)
            plants = sstep(rad + max(0.6 * res, 0.02), rad * 0.5, r) * pres
            if crop == "horticulture_mix":
                cols = np.stack(
                    [
                        np.asarray(srgb(c), np.float32)
                        for c in ("#5f8a30", "#6e3a38", "#647a4c", "#7a9443")
                    ]
                )
                pc = cols[block]
                # onions: thin continuous green lines instead of heads
                onion = block == 3
                line = sstep(0.06 + 0.5 * res, 0.02, np.abs(d)) * (0.6 + 0.4 * sstep(-1, 1, n_fine))
                plants = np.where(onion, line, plants)
                pc = pc * (1.0 + 0.15 * n_fine)[..., None]
                lerp_rgb(bed, pc, plants * bed_m)
            else:
                pc = pcol[None, None] * (1.0 + 0.18 * n_fine + 0.1 * speck)[..., None]
                lerp_rgb(bed, pc, plants * bed_m)
        elif crop == "cover_then_sweet_potato":
            cover = sstep(-0.9, 0.7, 0.8 * n_fine + 0.5 * speck + 0.6 * crown - 0.3 * furrow + 0.3 * n_mid)
            cc = np.empty((h, w, 3), np.float32)
            cc[:] = np.asarray(srgb("#5b7a2b"), np.float32)
            lerp_rgb(cc, srgb("#8b8a45"), sstep(-0.5, 1.2, n_mid + 0.5 * speck))
            cc *= (1.0 + 0.15 * speck)[..., None]
            lerp_rgb(bed, cc, cover * 0.85)

        # compose parcel: headland, then beds
        sub = img[rs, cs]
        lerp_rgb(sub, head, parcel_m)
        lerp_rgb(sub, bed, bed_m * parcel_m)

        if mulch_layers is None:
            return
        # ---- mulched variants (same pixels, chips over beds) ----
        n_chip = fbm(xs, ys, 0.35, seed + 41, 2, min_spacing=1.2 * res)
        for mode in ("dust", "cart"):
            L = mulch_layers[mode][rs, cs]
            lerp_rgb(L, head, parcel_m)
            if mode == "dust":
                mc = np.empty((h, w, 3), np.float32)
                mc[:] = np.asarray(srgb("#3a281c"), np.float32)
                lerp_rgb(mc, srgb("#5a412c"), sstep(-0.6, 1.6, 0.8 * n_chip + 0.6 * speck))
                mc *= (1.0 + 0.10 * speck + 0.06 * n_mid)[..., None]
                mc *= (1.0 - 0.25 * furrow + 0.05 * crown)[..., None]
                if plants is not None and crop == "sweet_potato_beauregard":
                    lerp_rgb(mc, srgb("#4f7f22"), plants * 0.4)
            else:
                mc = np.empty((h, w, 3), np.float32)
                mc[:] = np.asarray(srgb("#9a7650"), np.float32)
                lerp_rgb(mc, srgb("#c09a68"), sstep(-0.4, 1.4, 0.7 * n_chip + 0.7 * speck))
                lerp_rgb(mc, srgb("#6e5136"), sstep(1.0, 2.0, -speck * 0.8 - n_chip * 0.6))
                mc *= (1.0 + 0.06 * n_mid)[..., None]
                mc *= (1.0 - 0.2 * furrow)[..., None]
                # sparse seed+mycelium pucks on crowns (green / ochre dots)
                spk = 0.42
                ky = (Y - p.by0) / spk
                jit = hash_u01(k_ridge, 5, seed + 31).astype(np.float32)
                jj = np.floor(ky + jit).astype(np.int64)
                fy = (ky + jit - jj - 0.5) * spk
                pres = hash_u01(k_ridge, jj, seed + 32) > 0.45
                rpk = np.sqrt(d**2 + fy**2)
                puck = sstep(0.075 + 0.5 * res, 0.04, rpk) * pres
                green = hash_u01(k_ridge, jj, seed + 33) > 0.45
                pc = np.where(green[..., None], np.asarray(srgb("#5d7f2e"), np.float32), np.asarray(srgb("#c8962e"), np.float32))
                lerp_rgb(mc, pc, puck)
            lerp_rgb(L, mc, bed_m * parcel_m)
        # coverage threshold: beds fill one after another (random order per ridge), each with
        # an irregular along-bed front; the ridge index uses floor so furrows join their bed.
        k_bed = np.floor((X - p.x0) / p.pitch + 0.5).astype(np.int64)
        r_bed = hash_u01(k_bed, 0, seed + 61).astype(np.float32)
        along = value_noise(xs / 0.6, ys / 4.0, 1.0, seed + 62) * 0.7 + value_noise(xs, ys / 0.8, 1.0, seed + 63) * 0.3
        thr = 0.05 + 0.86 * r_bed + 0.06 * along + 0.04 * n_fine + 0.015 * speck
        thr = np.clip(thr, 0.03, 0.97)
        T = mulch_layers["threshold"][rs, cs]
        Wt = mulch_layers["weight"][rs, cs]
        wloc = bed_m * parcel_m
        upd = wloc > (Wt / 255.0)
        T[upd] = to_u8(thr)[upd]
        Wt[upd] = to_u8(wloc)[upd]

    # -----------------------------------------------------------------------
    def _paint_tracks(self, R, img, res):
        for tid, pts, width, surface in self.s.tracks:
            for a, b in zip(pts[:-1], pts[1:]):
                pad = width / 2 + 3.0
                win = R.window(min(a[0], b[0]) - pad, min(a[1], b[1]) - pad, max(a[0], b[0]) + pad, max(a[1], b[1]) + pad)
                if win is None:
                    continue
                (rs, cs), xs, ys = win
                X = xs[None, :]
                Y = ys[:, None]
                dist, along = seg_dist(X, Y, a, b)
                h, w = dist.shape
                key = _skey(tid)
                n1 = fbm(xs, ys, 2.0, self.seed + 600 + key, 3, min_spacing=1.5 * res)
                n2 = fbm(xs, ys, 0.6, self.seed + 610 + key, 2, min_spacing=1.2 * res)
                speck = np.random.default_rng(self.seed + key + R.w).standard_normal((h, w)).astype(np.float32)
                sub = img[rs, cs]
                half = width / 2.0
                if surface == "asphalt":
                    shoulder = sstep(half + 1.2, half + 0.2, dist + 0.25 * n1)
                    col = np.empty((h, w, 3), np.float32)
                    col[:] = np.asarray(srgb("#a29a86"), np.float32)
                    col *= (1 + 0.08 * speck)[..., None]
                    lerp_rgb(sub, col, shoulder)
                    m = sstep(half + 0.05, half - 0.1, dist + 0.04 * n2)
                    col[:] = np.asarray(C_ASPHALT, np.float32)
                    patch = sstep(0.9, 1.3, n1)
                    lerp_rgb(col, srgb("#2c2d2f"), patch * 0.8)
                    crack = sstep(0.06, 0.0, np.abs(n2)) * sstep(0.2, 0.9, n1)
                    col *= (1.0 + 0.10 * speck + 0.05 * n1 - 0.35 * crack)[..., None]
                    # wheel-path polish (lighter worn lanes)
                    for off in (-1.6, 1.6):
                        lane = sstep(0.7, 0.2, np.abs(np.abs(dist) - abs(off)))
                        col *= (1.0 + 0.05 * lane)[..., None]
                    wear = sstep(-0.4, 0.8, n2 + 0.6 * speck)
                    edge = sstep(0.09 + res * 0.5, 0.05, np.abs(dist - (half - 0.32))) * wear
                    lerp_rgb(col, srgb("#d8d6cc"), edge * 0.85)
                    dash = (np.mod(along, 12.0) < 4.5).astype(np.float32)
                    cen = sstep(0.07 + res * 0.5, 0.04, dist) * dash * wear
                    lerp_rgb(col, srgb("#d8d6cc"), cen * 0.8)
                    lerp_rgb(sub, col, m)
                elif surface == "gravel":
                    m = sstep(half + 0.4, half - 0.3, dist + 0.3 * n1)
                    col = np.empty((h, w, 3), np.float32)
                    col[:] = np.asarray(C_GRAVEL, np.float32)
                    col *= (1.0 + 0.16 * speck + 0.06 * n1)[..., None]
                    for off in (-0.85, 0.85):
                        rut = sstep(0.35, 0.15, np.abs(dist - abs(off)) if off > 0 else np.abs(dist - abs(off)))
                        col *= (1.0 - 0.08 * rut)[..., None]
                    lerp_rgb(sub, col, m)
                else:  # dirt farm track
                    m = sstep(half + 0.45, half - 0.35, dist + 0.35 * n1)
                    col = np.empty((h, w, 3), np.float32)
                    col[:] = np.asarray(C_DIRT, np.float32)
                    col *= (1.0 + 0.09 * speck + 0.06 * n1 + 0.05 * n2)[..., None]
                    for off in (0.85,):
                        rut = sstep(0.32, 0.12, np.abs(dist - off) + 0.05 * n2)
                        lerp_rgb(col, C_RUT, rut * 0.75)
                        col *= (1.0 - 0.07 * rut)[..., None]
                    if width >= 3.9:
                        med = sstep(0.42, 0.18, dist + 0.12 * n2) * sstep(-0.6, 0.4, n1 + 0.6 * speck)
                        mc = np.empty((h, w, 3), np.float32)
                        mc[:] = np.asarray(srgb("#77803f"), np.float32)
                        mc *= (1.0 + 0.15 * speck)[..., None]
                        lerp_rgb(col, mc, med * 0.85)
                    puddle = sstep(1.5, 1.9, n1) * sstep(half, half - 1.0, dist)
                    col *= (1.0 - 0.15 * puddle)[..., None]
                    lerp_rgb(sub, col, m)

    def _paint_ditches(self, R, img, res):
        for did, pts, width in self.s.ditches:
            for a, b in zip(pts[:-1], pts[1:]):
                pad = width / 2 + 4.0
                win = R.window(min(a[0], b[0]) - pad, min(a[1], b[1]) - pad, max(a[0], b[0]) + pad, max(a[1], b[1]) + pad)
                if win is None:
                    continue
                (rs, cs), xs, ys = win
                X = xs[None, :]
                Y = ys[:, None]
                dist, along = seg_dist(X, Y, a, b)
                h, w = dist.shape
                key = _skey(did)
                n1 = fbm(xs, ys, 1.5, self.seed + 700 + key, 3, min_spacing=1.5 * res)
                speck = np.random.default_rng(self.seed + key + R.w).standard_normal((h, w)).astype(np.float32)
                half = width / 2.0
                sub = img[rs, cs]
                lush = sstep(half + 3.4, half + 1.0, dist + 0.6 * n1)
                lc = np.empty((h, w, 3), np.float32)
                lc[:] = np.asarray(C_LUSH, np.float32)
                lerp_rgb(lc, srgb("#6d7a35"), sstep(-0.5, 1.0, n1))
                lc *= (1.0 + 0.14 * speck)[..., None]
                lerp_rgb(sub, lc, lush)
                bank = sstep(half + 0.7, half + 0.1, dist + 0.2 * n1)
                bc = np.empty((h, w, 3), np.float32)
                bc[:] = np.asarray(srgb("#4b3f2c"), np.float32)
                bc *= (1.0 + 0.1 * speck)[..., None]
                lerp_rgb(sub, bc, bank)
                water = sstep(half - 0.05, half - 0.3, dist + 0.08 * n1)
                wc = np.empty((h, w, 3), np.float32)
                wc[:] = np.asarray(C_WATER, np.float32)
                vertical = abs(b[1] - a[1]) > abs(b[0] - a[0])
                if vertical:
                    streak = value_noise(xs / 0.4, ys / 5.0, 1.0, self.seed + 710 + key)
                else:
                    streak = value_noise(xs / 5.0, ys / 0.4, 1.0, self.seed + 710 + key)
                wc *= (1.0 + 0.10 * streak + 0.03 * speck)[..., None]
                lerp_rgb(wc, srgb("#6f8a93"), sstep(0.8, 1.8, streak) * 0.35)
                lerp_rgb(sub, wc, water)

    def _paint_depot(self, R, img, res):
        s = self.s
        x0, y0, x1, y1 = s.depot.bounds
        win = R.window(x0 - 8, y0 - 8, x1 + 8, y1 + 8)
        if win is None:
            return
        (rs, cs), xs, ys = win
        X = xs[None, :]
        Y = ys[:, None]
        h, w = len(ys), len(xs)
        n1 = fbm(xs, ys, 2.5, self.seed + 801, 3, min_spacing=1.5 * res)
        n2 = fbm(xs, ys, 0.5, self.seed + 802, 2, min_spacing=1.2 * res)
        speck = np.random.default_rng(self.seed + 803 + R.w).standard_normal((h, w)).astype(np.float32)
        sub = img[rs, cs]
        m = sstep(0.8, -0.4, rect_sdist(X, Y, x0, y0, x1, y1) + 0.5 * n1)
        col = np.empty((h, w, 3), np.float32)
        col[:] = np.asarray(C_GRAVEL, np.float32)
        lerp_rgb(col, srgb("#8f8b82"), sstep(0.3, 1.4, n1) * 0.6)
        col *= (1.0 + 0.18 * speck + 0.05 * n2)[..., None]
        lerp_rgb(sub, col, m)
        # wood-chip spill around stockpiles
        for (sx, sy) in s.stockpiles:
            r = np.sqrt((X - sx) ** 2 + (Y - sy) ** 2)
            spill = sstep(7.8, 4.0, r + 1.4 * n1 + 0.4 * n2) * sstep(-1.5, 0.6, n2 + 0.5 * speck + (6.5 - r) * 0.5)
            cc = np.empty((h, w, 3), np.float32)
            cc[:] = np.asarray(srgb("#7a5634"), np.float32)
            lerp_rgb(cc, srgb("#a47a4b"), sstep(-0.5, 1.2, speck))
            lerp_rgb(sub, cc, spill)

    def _paint_weeds(self, R, img, res):
        for zid, poly, species in self.s.weeds:
            x0, y0, x1, y1 = poly.bounds
            win = R.window(x0 - WEED_MARGIN, y0 - WEED_MARGIN, x1 + WEED_MARGIN, y1 + WEED_MARGIN)
            if win is None:
                continue
            (rs, cs), xs, ys = win
            h, w = len(ys), len(xs)
            key = _skey(zid)
            n1 = fbm(xs, ys, 3.0, self.seed + 900 + key, 3, min_spacing=1.5 * res)
            n2 = fbm(xs, ys, 0.45, self.seed + 910 + key, 2, min_spacing=1.2 * res)
            speck = np.random.default_rng(self.seed + key + R.w).standard_normal((h, w)).astype(np.float32)
            m = sstep(1.2, -1.0, weed_sdf_on_grid(zid, poly, xs, ys, self.seed, min_spacing=1.5 * res))
            sub = img[rs, cs]
            col = np.empty((h, w, 3), np.float32)
            if species.startswith("Sorghum"):
                col[:] = np.asarray(srgb("#7b8644"), np.float32)
                lerp_rgb(col, srgb("#a19b64"), sstep(-0.2, 1.2, n2 + 0.5 * speck))
                lerp_rgb(col, srgb("#4f5a26"), sstep(0.8, 1.8, -n2 - 0.6 * speck) * 0.8)
            elif species.startswith("Cyperus"):
                col[:] = np.asarray(srgb("#5c6f34"), np.float32)
                lerp_rgb(col, srgb("#838d4c"), sstep(-0.2, 1.4, n2 + 0.6 * speck))
                lerp_rgb(col, srgb("#7b6347"), sstep(1.0, 2.0, -n2 - speck) * 0.7)
            else:  # Cynodon creeping mat
                col[:] = np.asarray(srgb("#7f8c4a"), np.float32)
                runners = sstep(0.08, 0.0, np.abs(n2)) * 0.6
                lerp_rgb(col, srgb("#9fa461"), runners)
                lerp_rgb(col, srgb("#6b7d3a"), sstep(-0.3, 1.2, n1 + 0.4 * speck) * 0.6)
            col *= (1.0 + 0.12 * speck)[..., None]
            lerp_rgb(sub, col, m)

    def _paint_waterlogged(self, R, img, res):
        for i, (cx, cy, rad) in enumerate(self.s.waterlogged):
            win = R.window(cx - rad - 4, cy - rad - 4, cx + rad + 4, cy + rad + 4)
            if win is None:
                continue
            (rs, cs), xs, ys = win
            X = xs[None, :]
            Y = ys[:, None]
            h, w = len(ys), len(xs)
            n1 = fbm(xs, ys, 2.0, self.seed + 950 + i, 4, min_spacing=1.5 * res)
            speck = np.random.default_rng(self.seed + 960 + i + R.w).standard_normal((h, w)).astype(np.float32)
            r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
            sub = img[rs, cs]
            damp = sstep(rad + 3.0, rad - 0.5, r + 1.2 * n1)
            sub *= (1.0 - 0.28 * damp)[..., None]
            mud = sstep(rad + 0.6, rad - 0.8, r + 1.0 * n1)
            mc = np.empty((h, w, 3), np.float32)
            mc[:] = np.asarray(C_MUD, np.float32)
            mc *= (1.0 + 0.12 * speck + 0.08 * n1)[..., None]
            lerp_rgb(sub, mc, mud)
            water = sstep(rad - 0.6, rad - 1.8, r + 1.3 * n1)
            wc = np.empty((h, w, 3), np.float32)
            wc[:] = np.asarray(srgb("#7f9aa6"), np.float32)
            lerp_rgb(wc, srgb("#4d5f63"), sstep(-0.5, 1.0, n1) * 0.6)
            lerp_rgb(sub, wc, water)

    def _paint_ao(self, R, img, res):
        """Baked contact shadow / litter under trees, houses, greenhouses, obstacles."""
        s = self.s
        for kind, x, y, cr, ht in s.trees:
            rad = cr * 1.25 + 0.5
            win = R.window(x - rad, y - rad, x + rad, y + rad)
            if win is None:
                continue
            (rs, cs), xs, ys = win
            r = np.sqrt((xs[None, :] - x) ** 2 + (ys[:, None] - y) ** 2)
            ao = sstep(rad, 0.15 * rad, r) * (0.35 if kind not in ("reed",) else 0.25)
            img[rs, cs] *= (1.0 - ao)[..., None]
            if kind in ("olive", "carob", "pine", "citrus"):
                lit = sstep(rad * 0.8, rad * 0.2, r) * 0.25
                lerp_rgb(img[rs, cs], srgb("#5d4c33"), lit)
        for poly in list(s.houses) + [fp for _, fp in s.obstacle_footprints]:
            x0, y0, x1, y1 = poly.bounds
            win = R.window(x0 - 2.5, y0 - 2.5, x1 + 2.5, y1 + 2.5)
            if win is None:
                continue
            (rs, cs), xs, ys = win
            sd = poly_sdist(xs[None, :], ys[:, None], poly)
            ao = sstep(2.2, -0.3, sd) * 0.35
            img[rs, cs] *= (1.0 - ao)[..., None]
        for poly in s.greenhouses:
            x0, y0, x1, y1 = poly.bounds
            win = R.window(x0 - 1.2, y0 - 1.2, x1 + 1.2, y1 + 1.2)
            if win is None:
                continue
            (rs, cs), xs, ys = win
            sd = rect_sdist(xs[None, :], ys[:, None], x0, y0, x1, y1)
            ao = sstep(1.0, -0.2, sd) * sstep(-0.6, 0.0, sd) * 0.3
            img[rs, cs] *= (1.0 - ao)[..., None]


# ---------------------------------------------------------------------------
# Sky
# ---------------------------------------------------------------------------


def _sky_color(d: np.ndarray, sun: np.ndarray, seed: int) -> np.ndarray:
    """d: (...,3) unit directions (x east, y north, z up)."""
    z = d[..., 2]
    el = np.arcsin(np.clip(z, -1, 1))
    zen = np.asarray(srgb("#3f74b8"), np.float32)
    mid = np.asarray(srgb("#86aedb"), np.float32)
    hor = np.asarray(srgb("#e3dccb"), np.float32)
    t = np.clip(el / (np.pi / 2), 0, 1)
    a = sstep(0.0, 0.22, t)[..., None]
    b = sstep(0.12, 0.85, t)[..., None]
    col = hor * (1 - a) + mid * a
    col = col * (1 - b) + zen * b
    below = z < 0
    ground = np.asarray(srgb("#d5d0c0"), np.float32)
    col = np.where(below[..., None], ground, col)
    # sun glow
    cosang = np.clip((d * sun).sum(-1), -1, 1)
    ang = np.arccos(cosang)
    glow = 0.45 * np.exp(-((ang / 0.35) ** 2)) + 0.25 * np.exp(-((ang / 0.9) ** 2))
    disk = sstep(0.022, 0.012, ang)
    col = col + glow[..., None] * np.asarray([1.0, 0.86, 0.62], np.float32)
    col = col * (1 - disk[..., None]) + disk[..., None] * np.asarray([1.0, 0.98, 0.92], np.float32)
    # clouds: project on a plane at altitude, fbm
    zc = np.maximum(z, 0.02)
    px = d[..., 0] / zc * 1.0
    py = d[..., 1] / zc * 1.0
    flat_x = px.ravel()
    flat_y = py.ravel()
    # evaluate fbm on scattered points via a fine grid lookup
    g = 512
    lim = 6.0
    gx = np.linspace(-lim, lim, g)
    nz = fbm(gx, gx, 1.6, seed, 5)
    nz2 = fbm(gx, gx, 0.35, seed + 5, 3)
    from scipy.ndimage import map_coordinates

    ix = (np.clip(flat_x, -lim, lim) + lim) / (2 * lim) * (g - 1)
    iy = (np.clip(flat_y, -lim, lim) + lim) / (2 * lim) * (g - 1)
    c1 = map_coordinates(nz, [iy, ix], order=1).reshape(z.shape)
    c2 = map_coordinates(nz2, [iy, ix], order=1).reshape(z.shape)
    cover = sstep(0.55, 1.5, c1 + 0.35 * c2)
    fade = sstep(0.03, 0.25, z) * (1 - sstep(0.85, 1.0, z) * 0.3)
    cover = cover * fade
    shade = sstep(-1.0, 1.5, c2) * 0.18
    cloud = np.asarray([1.0, 0.98, 0.95], np.float32) * (1.0 - shade[..., None]) + glow[..., None] * 0.3
    col = col * (1 - cover[..., None] * 0.85) + cloud * cover[..., None] * 0.85
    return np.clip(col, 0, 1)


def bake_skybox(face: int, sun_dir: tuple[float, float, float], seed: int = 5) -> np.ndarray:
    """MuJoCo skybox cross (gridsize 3x4, layout '.U..LFRB.D..') as uint8 image.

    Face orientation measured against MuJoCo's renderer (centre-pixel probes):
    L = -x, R = +x, F = -y, B = +y, U = +z, D = -z, with the per-face (u, v)
    mapping below (a = 2u - 1, b = 2v - 1; v = 0 is the image top row).
    """
    S = face
    sun = np.asarray(sun_dir, np.float32)
    sun /= np.linalg.norm(sun)
    out = np.zeros((3 * S, 4 * S, 3), np.uint8)
    uu, vv = np.meshgrid((np.arange(S) + 0.5) / S, (np.arange(S) + 0.5) / S)
    a = (2 * uu - 1).astype(np.float32)
    b = (2 * vv - 1).astype(np.float32)
    one = np.ones_like(a)
    faces = {
        "L": (-one, -a, -b),
        "R": (one, a, -b),
        "F": (a, -one, -b),
        "B": (-a, one, -b),
        "U": (a, -b, one),
        "D": (a, b, -one),
    }
    layout = ".U..LFRB.D.."
    for i, ch in enumerate(layout):
        if ch == ".":
            continue
        r, c = divmod(i, 4)
        x, y, z = faces[ch]
        d = np.stack([x, y, z], -1)
        d /= np.linalg.norm(d, axis=-1, keepdims=True)
        out[r * S : (r + 1) * S, c * S : (c + 1) * S] = to_u8(_sky_color(d, sun, seed))
    return out


# ---------------------------------------------------------------------------
# Small material textures + palette atlas
# ---------------------------------------------------------------------------


def _tileable(size: int, cells: int, seed: int, octaves: int = 4) -> np.ndarray:
    xs = (np.arange(size) + 0.5) * cells / size
    return fbm(xs, xs, 1.0, seed, octaves, period=(cells, cells))


def bake_material_textures(seed: int = 3) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    out: dict[str, np.ndarray] = {}
    S = 512
    # wood chips (stockpiles, hopper fill)
    n1 = _tileable(S, 16, seed + 1)
    n2 = _tileable(S, 64, seed + 2, 2)
    sp = rng.standard_normal((S, S)).astype(np.float32)
    chip = np.empty((S, S, 3), np.float32)
    chip[:] = srgb("#8a5f38")
    lerp_rgb(chip, srgb("#b98a55"), sstep(-0.3, 1.3, 0.7 * n2 + 0.6 * sp))
    lerp_rgb(chip, srgb("#4d321e"), sstep(0.9, 2.0, -0.8 * n2 - 0.6 * sp))
    chip *= (1 + 0.08 * n1)[..., None]
    out["chips"] = to_u8(chip)
    # dark fine dust mulch
    dust = np.empty((S, S, 3), np.float32)
    dust[:] = srgb("#4f3726")
    lerp_rgb(dust, srgb("#76553a"), sstep(-0.5, 1.5, 0.6 * n2 + 0.7 * sp))
    dust *= (1 + 0.07 * n1)[..., None]
    out["dust"] = to_u8(dust)
    # plaster (masia walls)
    n3 = _tileable(S, 8, seed + 3)
    pl = np.empty((S, S, 3), np.float32)
    pl[:] = srgb("#ece4d2")
    lerp_rgb(pl, srgb("#d9c7a3"), sstep(0.2, 1.6, n3) * 0.7)
    pl *= (1 + 0.03 * sp + 0.03 * n2)[..., None]
    out["plaster"] = to_u8(pl)
    ochre = pl * np.asarray(srgb("#e3b878"), np.float32) / np.asarray(srgb("#ece4d2"), np.float32)
    out["ochre"] = to_u8(ochre)
    # terracotta roof tiles (rows along v, ripples along u)
    yy, xx = np.mgrid[:S, :S].astype(np.float32) / S
    ripple = 0.5 + 0.5 * np.cos(xx * 2 * np.pi * 16)
    rows = np.mod(yy * 12, 1.0)
    tile = np.empty((S, S, 3), np.float32)
    tile[:] = srgb("#b4593a")
    lerp_rgb(tile, srgb("#8e4128"), (1 - ripple) * 0.7)
    tile *= (1.0 - 0.35 * sstep(0.88, 1.0, rows))[..., None]
    jitter = hash_u01(np.floor(xx * 16).astype(int), np.floor(yy * 12).astype(int), seed).astype(np.float32)
    tile *= (0.88 + 0.22 * jitter)[..., None]
    tile *= (1 + 0.05 * n3 + 0.04 * sp)[..., None]
    out["roof"] = to_u8(tile)
    # corrugated container (green), steel
    corr = np.empty((S, S, 3), np.float32)
    corr[:] = srgb("#3e6b4a")
    corr *= (0.82 + 0.18 * (0.5 + 0.5 * np.sin(xx * 2 * np.pi * 24)))[..., None]
    corr *= (1 + 0.04 * n2)[..., None]
    lerp_rgb(corr, srgb("#7a5a3a"), sstep(1.3, 2.0, n3) * 0.5)
    out["container"] = to_u8(corr)
    # solar panel cells
    cell = np.empty((S, S, 3), np.float32)
    cell[:] = srgb("#1d2a4a")
    gx = np.mod(xx * 8, 1.0)
    gy = np.mod(yy * 6, 1.0)
    grid = np.maximum(sstep(0.04, 0.0, np.minimum(gx, 1 - gx)), sstep(0.04, 0.0, np.minimum(gy, 1 - gy)))
    lerp_rgb(cell, srgb("#c9cdd4"), grid * 0.8)
    fine = np.maximum(sstep(0.01, 0.0, np.abs(np.mod(xx * 48, 1.0) - 0.5) - 0.48), 0)
    cell *= (1 - 0.15 * fine)[..., None]
    cell *= (1 + 0.05 * n1)[..., None]
    out["solar"] = to_u8(cell)
    # tyre tread (dark rubber with blocks), for wheel bands
    tread = np.empty((S, S, 3), np.float32)
    tread[:] = srgb("#1c1c1d")
    blocks = (np.mod(xx * 24 + 0.5 * (np.mod(yy * 4, 1.0) > 0.5), 1.0) < 0.55).astype(np.float32)
    tread *= (0.8 + 0.4 * blocks)[..., None]
    out["tread"] = to_u8(tread)
    # asphalt-ish dark metal / painted panel subtle noise
    panel = np.empty((S, S, 3), np.float32)
    panel[:] = 1.0
    panel *= (1 + 0.03 * n2 + 0.02 * sp)[..., None]
    out["panel"] = to_u8(np.clip(panel * 0.97, 0, 1))
    # bark
    bark = np.empty((S, S, 3), np.float32)
    stri = value_noise(np.arange(S) / S * 64, np.arange(S) / S * 4, 1.0, seed + 9, period=(64, 4))
    bark[:] = srgb("#8f8a7c")
    lerp_rgb(bark, srgb("#4f4a40"), sstep(0.2, 1.4, stri + 0.4 * n2))
    out["bark"] = to_u8(bark)
    return out


# Atlas bands: (name, base colour, dark colour, light colour, leafy?)
ATLAS_BANDS: list[tuple[str, str, str, str, bool]] = [
    ("poplar0", "#587a33", "#26391a", "#93ad5e", True),
    ("poplar1", "#62833a", "#2c421d", "#9fb868", True),
    ("poplar2", "#52733a", "#223620", "#8aa462", True),
    ("poplar3", "#6b8840", "#344a22", "#a8bd6e", True),
    ("olive0", "#7f8b66", "#434c35", "#b3bc98", True),
    ("olive1", "#86906e", "#474f39", "#bcc4a2", True),
    ("olive2", "#76825f", "#3d4530", "#a9b38e", True),
    ("carob0", "#3f5a2a", "#1d2d13", "#68854a", True),
    ("carob1", "#46612e", "#213216", "#718e50", True),
    ("citrus0", "#365a25", "#16290f", "#5f8a3f", True),
    ("citrus1", "#3d6229", "#192e11", "#679246", True),
    ("citrus2", "#33552a", "#152611", "#5a8442", True),
    ("orange", "#f08a1c", "#b45a0c", "#ffb547", False),
    ("pine0", "#4a6430", "#233318", "#76904f", True),
    ("pine1", "#536c34", "#283a1a", "#80995a", True),
    ("pine2", "#45602e", "#203016", "#6f8a4b", True),
    ("reed_green", "#7b8f45", "#45552a", "#a7b56a", False),
    ("reed_tan", "#b5a36c", "#7c6c45", "#d6c892", False),
    ("reed_leaf", "#6f8a3d", "#3e5223", "#9db45f", False),
    ("plume", "#c9b997", "#8f7f62", "#e6dcc4", False),
    ("sorghum0", "#69803c", "#384a1f", "#99a564", False),
    ("sorghum1", "#8a9150", "#4f5629", "#b8b47c", False),
    ("panicle", "#8a5a3a", "#5a3722", "#b7845a", False),
    ("cyperus", "#5e7f2c", "#33491a", "#8eae49", False),
    ("cynodon", "#7d8c4a", "#4c5a2c", "#a3b06a", False),
    ("bark_pale", "#a5a094", "#5f5b52", "#c9c5b8", False),
    ("bark_dark", "#5a4d3e", "#2d251d", "#7d6d5a", False),
    ("bark_pine", "#7a5640", "#45301f", "#9e7658", False),
    ("log_bark", "#6e6356", "#352f28", "#988c7a", True),
    ("log_wood", "#d2b98c", "#9d8460", "#ead6ad", False),
    ("dead_branch", "#7a6a55", "#463a2d", "#a08d74", False),
    ("dry_grass", "#b8a66a", "#7d6f44", "#d8c98f", False),
    ("pole_wood", "#6d5a45", "#3f3226", "#8c765c", False),
    ("steel", "#9aa0a6", "#5d6267", "#cfd4d8", False),
    ("galv", "#b9bdbf", "#7d8184", "#e2e5e6", False),
    ("ceramic", "#e9e6dc", "#a9a69c", "#ffffff", False),
    ("wire", "#2a2a2a", "#151515", "#444444", False),
    ("plastic_film", "#eef1f0", "#c8cdcc", "#ffffff", False),
    ("pipe_blue", "#2f6fb5", "#1b4170", "#5d97d6", False),
    ("pipe_alu", "#c3c7cb", "#80858a", "#e7eaec", False),
    ("chip_brown", "#8a5f38", "#5a3b21", "#b48650", False),
    ("mud", "#4a3a2b", "#2a2018", "#6a5541", False),
    ("leaf_dead", "#9b7b46", "#5f4a28", "#c09a5c", False),
    ("root_soil", "#5f4a35", "#33281c", "#7d6449", False),
]
ATLAS_INDEX = {b[0]: i for i, b in enumerate(ATLAS_BANDS)}
ATLAS_BAND_PX = 32
ATLAS_W = 512


def atlas_v(name: str, t) -> np.ndarray:
    """Texture v for band `name`, t in [0,1] = dark..light (kept inside band)."""
    i = ATLAS_INDEX[name]
    nb = len(ATLAS_BANDS)
    t = np.clip(np.asarray(t, np.float32), 0.0, 1.0)
    # v=0 is the image top; band i occupies rows [i*px, (i+1)*px); light at band top
    return ((i + 0.82 - 0.64 * t) / nb).astype(np.float32)


def bake_atlas(seed: int = 11) -> np.ndarray:
    nb = len(ATLAS_BANDS)
    P = ATLAS_BAND_PX
    W = ATLAS_W
    img = np.zeros((nb * P, W, 3), np.float32)
    xs = (np.arange(W) + 0.5) / W * 32
    for i, (name, base, dark, light, leafy) in enumerate(ATLAS_BANDS):
        ys = (np.arange(P) + 0.5) / P * 2
        if leafy:
            n = fbm(xs, ys + i * 7.3, 1.0, seed + i, 4, period=(32, 10_000))
            n2 = fbm(xs * 4, ys * 4 + i * 3.1, 1.0, seed + 100 + i, 2, period=(128, 10_000))
        else:
            n = fbm(xs / 4, ys + i * 7.3, 1.0, seed + i, 3, period=(8, 10_000))
            n2 = fbm(xs, ys + i * 3.1, 1.0, seed + 100 + i, 2, period=(32, 10_000))
        t = np.linspace(1.0, 0.0, P, dtype=np.float32)[:, None]  # top light -> bottom dark
        b, d, l = (np.asarray(srgb(c), np.float32) for c in (base, dark, light))
        tt = np.clip(t + (0.16 if leafy else 0.06) * n + (0.10 if leafy else 0.04) * n2, 0, 1)
        lo = d[None, None] * (1 - tt[..., None] * 2) + b[None, None] * (tt[..., None] * 2)
        hi = b[None, None] * (2 - tt[..., None] * 2) + l[None, None] * (tt[..., None] * 2 - 1)
        band = np.where((tt < 0.5)[..., None], lo, hi)
        img[i * P : (i + 1) * P] = band
    return to_u8(img)


def bake_patchwork_texture(x0, y0, x1, y1, px: int, seed: int, haze_rgb=None, haze_r=(330.0, 1100.0)) -> np.ndarray:
    """Low-res farmland for the far apron (world fn, continues the ground).

    With ``haze_rgb`` the farmland fades to that colour between radii haze_r
    (baked aerial perspective; MuJoCo fog would also grey out the skybox).
    """
    R = Raster(x0, y0, x1, y1, px, px)
    X = np.broadcast_to(R.xs[None, :], (px, px))
    Y = np.broadcast_to(R.ys[:, None], (px, px))
    img = patchwork(X, Y, seed, R.resx)
    n = fbm(R.xs, R.ys, 40.0, seed + 3, 4)
    img *= (1.0 + 0.06 * n)[..., None]
    img = img * 0.9 + np.asarray(srgb("#a9a586"), np.float32) * 0.1
    if haze_rgb is not None:
        d = np.sqrt(X**2 + Y**2)
        hz = sstep(haze_r[0], haze_r[1], d) ** 0.8
        lerp_rgb(img, haze_rgb, np.maximum(hz, sstep(haze_r[1] * 0.9, haze_r[1], np.maximum(np.abs(X), np.abs(Y)))))
    return to_u8(img)


def downsample(img: np.ndarray, f: int) -> np.ndarray:
    h, w = img.shape[0] // f * f, img.shape[1] // f * f
    a = img[:h, :w].reshape(h // f, f, w // f, f, -1).mean(axis=(1, 3))
    return a


__all__ = [
    "Raster",
    "PaintParcel",
    "PaintScene",
    "GroundPainter",
    "bake_skybox",
    "bake_atlas",
    "bake_material_textures",
    "bake_patchwork_texture",
    "atlas_v",
    "ATLAS_BANDS",
    "fbm",
    "value_noise",
    "hash_u01",
    "sstep",
    "to_u8",
    "srgb",
    "seg_dist",
    "rect_sdist",
]
