"""Procedural mesh generation (terrain, raised beds, vegetation, structures).

Meshes are written in MuJoCo's legacy binary ``.msh`` format (vertices, normals,
texcoords, faces).  Texcoord v = 0 is the *top* row of the texture image
(verified against MuJoCo 4.0.1).  Vegetation and props share one palette atlas
(``world_textures.ATLAS_BANDS``): u carries fine leafy detail (wraps), v picks
the colour band and a dark..light shade, which gives per-vertex colour and fake
ambient occlusion with a single texture/material per mesh.
"""

from __future__ import annotations

import struct
from functools import lru_cache
from pathlib import Path

import numpy as np

from prunemaxxing.sim.world_textures import atlas_v, hash_u01

UP = np.array([0.0, 0.0, 1.0])


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def smooth_normals(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    N = np.zeros_like(V)
    for k in range(3):
        np.add.at(N, F[:, k], fn)
    n = np.linalg.norm(N, axis=1, keepdims=True)
    N = np.where(n > 1e-12, N / np.maximum(n, 1e-12), UP)
    return N


class MeshBuilder:
    def __init__(self) -> None:
        self.V: list[np.ndarray] = []
        self.N: list[np.ndarray] = []
        self.T: list[np.ndarray] = []
        self.F: list[np.ndarray] = []
        self.n = 0

    def add(self, V, F, N=None, T=None, up_bias: float = 0.0) -> None:
        V = np.asarray(V, np.float64)
        F = np.asarray(F, np.int64)
        if N is None:
            N = smooth_normals(V, F)
        if up_bias > 0:
            N = N * (1 - up_bias) + UP * up_bias
            N /= np.linalg.norm(N, axis=1, keepdims=True)
        if T is None:
            T = np.zeros((len(V), 2))
        self.V.append(V)
        self.N.append(np.asarray(N, np.float64))
        self.T.append(np.asarray(T, np.float64))
        self.F.append(F + self.n)
        self.n += len(V)

    @property
    def nvert(self) -> int:
        return self.n

    def arrays(self):
        V = np.concatenate(self.V).astype(np.float32)
        N = np.concatenate(self.N).astype(np.float32)
        T = np.concatenate(self.T).astype(np.float32)
        F = np.concatenate(self.F).astype(np.int32)
        return V, N, T, F

    def write(self, path: Path) -> dict:
        V, N, T, F = self.arrays()
        # drop degenerate faces (MuJoCo rejects area < 1e-14)
        a = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
        keep = np.linalg.norm(a.astype(np.float64), axis=1) > 1e-9
        F = F[keep]
        write_msh(path, V, N, T, F)
        lo, hi = V.min(0), V.max(0)
        return {"nvert": int(len(V)), "nface": int(len(F)), "bbox": [lo.tolist(), hi.tolist()]}


def write_msh(path: Path, V, N, T, F) -> None:
    with open(path, "wb") as f:
        f.write(struct.pack("4i", len(V), len(N), len(T), len(F)))
        f.write(np.ascontiguousarray(V, np.float32).tobytes())
        f.write(np.ascontiguousarray(N, np.float32).tobytes())
        f.write(np.ascontiguousarray(T, np.float32).tobytes())
        f.write(np.ascontiguousarray(F, np.int32).tobytes())


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


@lru_cache(maxsize=8)
def icosphere(subdiv: int):
    t = (1.0 + 5**0.5) / 2.0
    V = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
         (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    F = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6),
         (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10),
         (8, 6, 7), (9, 8, 1)]
    V = [np.array(v, float) / np.linalg.norm(v) for v in V]
    for _ in range(subdiv):
        cache = {}
        nF = []

        def mid(a, b):
            key = (min(a, b), max(a, b))
            if key not in cache:
                m = V[a] + V[b]
                V.append(m / np.linalg.norm(m))
                cache[key] = len(V) - 1
            return cache[key]

        for a, b, c in F:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            nF += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        F = nF
    return np.array(V), np.array(F, np.int64)


def _lumps(dirs: np.ndarray, rng: np.random.Generator, k: int = 9, fmin=2.0, fmax=5.5) -> np.ndarray:
    w = rng.normal(size=(k, 3))
    w /= np.linalg.norm(w, axis=1, keepdims=True)
    f = rng.uniform(fmin, fmax, k)
    ph = rng.uniform(0, 2 * np.pi, k)
    a = rng.uniform(0.5, 1.0, k)
    s = (np.sin(dirs @ w.T * f + ph) * a).sum(1) / np.sqrt((a**2).sum() / 2)
    return s


def blob(builder: MeshBuilder, c, radii, rng, band: str, subdiv=2, lump=0.18, profile=None,
         light_bias=0.0, flat_bottom=None, uv_scale=3.0, t_range=(0.25, 1.0), lump_freq=(2.0, 5.5)) -> np.ndarray:
    """Lumpy displaced ellipsoid (tree crowns, shrubs). Returns world vertices."""
    D, F = icosphere(subdiv)
    r = 1.0 + lump * _lumps(D, rng, fmin=lump_freq[0], fmax=lump_freq[1])
    P = D * r[:, None]
    if profile is not None:
        P[:, :2] *= profile(D[:, 2])[:, None]
    P = P * np.asarray(radii)[None, :]
    if flat_bottom is not None:
        P[:, 2] = np.maximum(P[:, 2], -flat_bottom * radii[2])
    V = P + np.asarray(c)[None, :]
    N = smooth_normals(V, F)
    hz = (D[:, 2] + 1) / 2
    t = 0.30 + 0.40 * hz + 0.30 * np.clip(N[:, 2], -0.3, 1) + 0.9 * (r - 1.0) + light_bias
    t = t_range[0] + (t_range[1] - t_range[0]) * np.clip(t, 0, 1)
    u = (P[:, 0] * 0.9 + P[:, 1] * 0.55 + P[:, 2] * 0.35) / uv_scale + rng.uniform(0, 1)
    T = np.stack([u, atlas_v(band, t)], 1)
    builder.add(V, F, N, T)
    return V


def tube(builder: MeshBuilder, pts, radii, band: str, sides=6, t0=0.2, t1=0.8, uv_len=1.0, cap=False):
    """Generalised cylinder along a polyline."""
    pts = np.asarray(pts, float)
    radii = np.broadcast_to(np.asarray(radii, float), (len(pts),))
    n = len(pts)
    tang = np.gradient(pts, axis=0)
    tang /= np.maximum(np.linalg.norm(tang, axis=1, keepdims=True), 1e-9)
    ref = np.array([0.0, 0.0, 1.0]) if abs(tang[0, 2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    rings = []
    norms = []
    b1 = np.cross(tang[0], ref)
    b1 /= np.linalg.norm(b1)
    ang = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    for i in range(n):
        b1 = b1 - tang[i] * (b1 @ tang[i])
        nb = np.linalg.norm(b1)
        if nb < 1e-6:
            b1 = np.cross(tang[i], ref)
            nb = np.linalg.norm(b1)
        b1 /= nb
        b2 = np.cross(tang[i], b1)
        dirs = np.cos(ang)[:, None] * b1 + np.sin(ang)[:, None] * b2
        rings.append(pts[i] + radii[i] * dirs)
        norms.append(dirs)
    V = np.concatenate(rings)
    N = np.concatenate(norms)
    F = []
    for i in range(n - 1):
        for k in range(sides):
            a = i * sides + k
            b = i * sides + (k + 1) % sides
            c = (i + 1) * sides + (k + 1) % sides
            d = (i + 1) * sides + k
            F += [(a, b, c), (a, c, d)]
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    u = np.repeat(s / uv_len, sides)
    tt = np.tile(t0 + (t1 - t0) * (0.5 + 0.5 * np.cos(ang)), n)
    T = np.stack([u, atlas_v(band, tt)], 1)
    F = np.array(F, np.int64)
    if cap:
        ci = len(V)
        V = np.vstack([V, pts[-1][None]])
        N = np.vstack([N, tang[-1][None]])
        T = np.vstack([T, [[0.0, atlas_v(band, t1)]]])
        last = (n - 1) * sides
        F = np.vstack([F, [(last + k, last + (k + 1) % sides, ci) for k in range(sides)]])
    builder.add(V, F, N, T)


def ribbon(builder: MeshBuilder, pts, widths, side_dir, band: str, t0=0.2, t1=0.9, up_bias=0.55, u0=0.0):
    """Double-sided strip (grass blade / strap leaf) along pts, widened along side_dir."""
    pts = np.asarray(pts, float)
    widths = np.asarray(widths, float)
    side = np.asarray(side_dir, float)
    side = side / max(np.linalg.norm(side), 1e-9)
    L = pts - side * widths[:, None] / 2
    Rr = pts + side * widths[:, None] / 2
    n = len(pts)
    V = np.empty((2 * n, 3))
    V[0::2] = L
    V[1::2] = Rr
    F = []
    for i in range(n - 1):
        a, b, c, d = 2 * i, 2 * i + 1, 2 * i + 3, 2 * i + 2
        F += [(a, b, c), (a, c, d)]
    F = np.array(F, np.int64)
    tt = np.repeat(np.linspace(t0, t1, n), 2)
    T = np.stack([np.full(2 * n, u0) + np.tile([0.0, 0.02], n), atlas_v(band, tt)], 1)
    N = smooth_normals(V, F)
    Nb = N * (1 - up_bias) + UP * up_bias
    Nb /= np.linalg.norm(Nb, axis=1, keepdims=True)
    Nf = -N * (1 - up_bias) + UP * up_bias
    Nf /= np.linalg.norm(Nf, axis=1, keepdims=True)
    builder.add(V, F, Nb, T)
    builder.add(V, F[:, ::-1], Nf, T)


def box(builder: MeshBuilder, c, half, band: str, yaw=0.0, t=0.6, R=None):
    """Flat-shaded box (24 vertices)."""
    hx, hy, hz = half
    corners = np.array([[sx * hx, sy * hy, sz * hz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    faces = [((0, 2, 6, 4), (-1, 0, 0)), ((1, 5, 7, 3), (1, 0, 0)), ((0, 4, 5, 1), (0, -1, 0)),
             ((2, 3, 7, 6), (0, 1, 0)), ((0, 1, 3, 2), (0, 0, -1)), ((4, 6, 7, 5), (0, 0, 1))]
    # corner index = sx*4 + sy*2 + sz with sx,sy,sz in {0,1}
    if R is None:
        cy, sy_ = np.cos(yaw), np.sin(yaw)
        R = np.array([[cy, -sy_, 0], [sy_, cy, 0], [0, 0, 1]])
    V, N, F, T = [], [], [], []
    for quad, nrm in faces:
        base = len(V)
        for q in quad:
            V.append(R @ corners[q] + c)
            N.append(R @ np.array(nrm, float))
        F += [(base, base + 1, base + 2), (base, base + 2, base + 3)]
        shade = t * (1.0 if nrm[2] >= 0 else 0.7)
        T += [[0.0, atlas_v(band, shade)]] * 4
    builder.add(np.array(V), np.array(F), np.array(N), np.array(T))


# ---------------------------------------------------------------------------
# Terrain meshes
# ---------------------------------------------------------------------------


def grid_mesh(xs, ys, Z, uv, skip=None, normals=None):
    """Regular (non-uniform) grid. Z,uv: (ny,nx[,2]). skip: (ny-1,nx-1) bool cells to drop."""
    ny, nx = Z.shape
    X, Y = np.meshgrid(xs, ys)
    V = np.stack([X, Y, Z], -1).reshape(-1, 3)
    T = uv.reshape(-1, 2)
    if normals is None:
        dzdy, dzdx = np.gradient(Z, ys, xs)
        Nn = np.stack([-dzdx, -dzdy, np.ones_like(Z)], -1)
        Nn /= np.linalg.norm(Nn, axis=-1, keepdims=True)
        normals = Nn.reshape(-1, 3)
    idx = np.arange(ny * nx).reshape(ny, nx)
    a = idx[:-1, :-1]
    b = idx[:-1, 1:]
    c = idx[1:, 1:]
    d = idx[1:, :-1]
    keep = np.ones(a.shape, bool) if skip is None else ~skip
    F = np.concatenate([
        np.stack([a[keep], b[keep], c[keep]], 1),
        np.stack([a[keep], c[keep], d[keep]], 1),
    ])
    used = np.unique(F)
    remap = -np.ones(len(V), np.int64)
    remap[used] = np.arange(len(used))
    return V[used], F if len(used) == len(V) else remap[F], T[used], normals[used]


def bed_profile(dx_from_ridge: np.ndarray, ridge_h: float) -> np.ndarray:
    """Raised-bed cross-section height (m) vs offset from ridge centre (m)."""
    ad = np.abs(dx_from_ridge)
    crown = ridge_h - 0.6 * ad**2
    flank = np.interp(ad, [0.15, 0.30], [ridge_h - 0.6 * 0.15**2, 0.015])
    furrow = np.interp(ad, [0.30, 0.425], [0.015, 0.0])
    return np.where(ad <= 0.15, crown, np.where(ad <= 0.30, flank, furrow))


# ---------------------------------------------------------------------------
# Vegetation
# ---------------------------------------------------------------------------


def _cluster_crown(b: MeshBuilder, center, pts, sizes, band, rng, env_r, env_h, squash=1.0, subdiv=2, lump=0.32,
                   uv_scale=1.2, shade=0.42):
    """Foliage as many small crinkly leaf clusters; outer/upper clusters lighter (fake AO)."""
    center = np.asarray(center, float)
    for p, s in zip(pts, sizes):
        rel = (np.asarray(p) - center) / np.array([env_r, env_r, env_h])
        expo = float(np.clip(np.linalg.norm(rel), 0.0, 1.2))
        bias = shade * (expo - 0.6) + 0.12 * float(np.clip(rel[2], -1, 1)) + 0.05 + rng.normal(0, 0.06)
        blob(b, p, [s, s, s * squash], rng, band, subdiv=subdiv, lump=lump, light_bias=bias, uv_scale=uv_scale,
             lump_freq=(3.5, 8.0))


def _sample_envelope(rng, n, env_fn, z0, z1, x, y, shell=(0.55, 1.0)):
    """Points inside a solid of revolution r <= env_fn(t), t = (z - z0) / (z1 - z0)."""
    out = []
    while len(out) < n:
        t = rng.uniform(0.02, 0.98)
        rmax = env_fn(t)
        rr = rmax * np.sqrt(rng.uniform(shell[0] ** 2, shell[1] ** 2))
        a = rng.uniform(0, 2 * np.pi)
        out.append((np.array([x + rr * np.cos(a), y + rr * np.sin(a), z0 + t * (z1 - z0)]), t, rmax))
    return out


def poplar(b: MeshBuilder, x, y, z, H, rng):
    """Lombardy poplar: slim trunk, narrow columnar crown of up-swept leaf clusters."""
    lean = rng.normal(0, 0.010, 2)
    top = np.array([x + lean[0] * H, y + lean[1] * H, z + 0.9 * H])
    tube(b, [[x, y, z - 0.3], [x, y, z + 1.2], (np.array([x, y, z + 1.2]) + top) / 2, top],
         [0.27 * H / 17, 0.22 * H / 17, 0.13, 0.04], "bark_pale", sides=7, t0=0.35, t1=0.95, uv_len=2.0)
    band = f"poplar{rng.integers(0, 4)}"
    R = rng.uniform(1.45, 1.95) * H / 17
    z0, z1 = z + 0.13 * H, z + 1.0 * H
    env = lambda t: R * max(np.sin(np.pi * min(t * 1.05, 1.0)) ** 0.55 * (1.18 - 0.62 * t), 0.08)  # noqa: E731
    cx, cy = x + lean[0] * H * 0.5, y + lean[1] * H * 0.5
    blob(b, [cx, cy, (z0 + z1) / 2], [0.62 * R, 0.62 * R, 0.46 * (z1 - z0)], rng, band, subdiv=3, lump=0.12,
         profile=lambda zz: np.clip(np.sin(np.pi * (zz + 1) / 2) ** 0.6 * (1.15 - 0.3 * (zz + 1)), 0.05, None),
         light_bias=-0.12, uv_scale=1.5)
    n = int(rng.integers(22, 28) * H / 17)
    pts, sizes = [], []
    for p, t, rmax in _sample_envelope(rng, n, env, z0, z1, cx, cy, (0.45, 0.95)):
        pts.append(p)
        sizes.append(rng.uniform(0.48, 0.7) * R * (0.55 + 0.45 * np.sin(np.pi * t)) + 0.15)
    _cluster_crown(b, [cx, cy, (z0 + z1) / 2], pts, sizes, band, rng, R, (z1 - z0) / 2, squash=1.55, subdiv=2)


def olive(b: MeshBuilder, x, y, z, H, rng):
    """Olive: 2-3 gnarled leaning stems, open silvery canopy of separate clusters."""
    n_stem = int(rng.integers(2, 4))
    tops = []
    for i in range(n_stem):
        a = rng.uniform(0, 2 * np.pi)
        r = rng.uniform(0.5, 1.3) * H / 6.5
        top = np.array([x + r * np.cos(a), y + r * np.sin(a), z + 0.48 * H])
        mid = np.array([x + 0.4 * r * np.cos(a) + rng.normal(0, 0.15), y + 0.4 * r * np.sin(a) + rng.normal(0, 0.15),
                        z + 0.22 * H])
        tube(b, [[x, y, z - 0.2], [x + rng.normal(0, 0.06), y + rng.normal(0, 0.06), z + 0.45], mid, top],
             [0.30, 0.25, 0.15, 0.09], "bark_dark", sides=7, t0=0.2, t1=0.75, uv_len=1.2)
        tops.append(top)
    band = f"olive{rng.integers(0, 3)}"
    Rc = 0.44 * H
    cz = z + 0.66 * H
    blob(b, [x, y, cz], [0.62 * Rc, 0.62 * Rc, 0.2 * H], rng, band, subdiv=2, lump=0.25, light_bias=-0.25)
    pts, sizes = [], []
    for _ in range(int(rng.integers(12, 16))):
        a = rng.uniform(0, 2 * np.pi)
        rr = Rc * np.sqrt(rng.uniform(0.15, 1.0))
        hz = rng.uniform(-0.75, 0.9) * 0.22 * H * (1.0 - 0.45 * rr / Rc)
        pts.append(np.array([x + rr * np.cos(a), y + rr * np.sin(a), cz + hz]))
        sizes.append(rng.uniform(0.65, 1.05) * H / 6.5)
    _cluster_crown(b, [x, y, cz], pts, sizes, band, rng, Rc, 0.25 * H, squash=0.72, subdiv=2, lump=0.35)
    for t in tops:  # branches into the canopy
        for _ in range(2):
            c = pts[int(rng.integers(len(pts)))]
            tube(b, [t, (t + c) / 2 + rng.normal(0, 0.15, 3), c], [0.09, 0.06, 0.03], "bark_dark", sides=5)


def carob(b: MeshBuilder, x, y, z, H, rng):
    """Carob: short thick trunk, dense dark-green dome."""
    tube(b, [[x, y, z - 0.2], [x, y, z + 0.3 * H], [x + rng.normal(0, 0.2), y + rng.normal(0, 0.2), z + 0.45 * H]],
         [0.34, 0.27, 0.16], "bark_dark", sides=7, t0=0.25, t1=0.7)
    band = f"carob{rng.integers(0, 2)}"
    R = 0.44 * H
    cz = z + 0.6 * H
    blob(b, [x, y, cz], [0.82 * R, 0.82 * R, 0.3 * H], rng, band, subdiv=3, lump=0.14, flat_bottom=0.6,
         light_bias=-0.18)
    pts, sizes = [], []
    for _ in range(int(rng.integers(13, 17))):
        d = rng.normal(size=3)
        d[2] = abs(d[2]) * 0.8 - 0.15
        d /= np.linalg.norm(d)
        pts.append(np.array([x, y, cz]) + d * np.array([R, R, 0.34 * H]) * rng.uniform(0.72, 0.95))
        sizes.append(rng.uniform(0.18, 0.28) * H)
    _cluster_crown(b, [x, y, cz], pts, sizes, band, rng, R, 0.34 * H, squash=0.8, subdiv=2, lump=0.3)


def citrus(b: MeshBuilder, x, y, z, H, rng):
    """Orange tree: short trunk, dense rounded crown with fruit."""
    tube(b, [[x, y, z - 0.2], [x, y, z + 0.55]], [0.09, 0.07], "bark_dark", sides=5)
    band = f"citrus{rng.integers(0, 3)}"
    R = H / 2.2
    c = np.array([x, y, z + 0.55 * H])
    blob(b, c, [R * 0.92, R * rng.uniform(0.85, 0.95), 0.46 * H], rng, band, subdiv=2, lump=0.12, flat_bottom=0.75,
         light_bias=-0.08, uv_scale=1.2)
    pts, sizes = [], []
    for _ in range(int(rng.integers(6, 9))):
        d = rng.normal(size=3)
        d[2] = abs(d[2]) * 0.9 - 0.1
        d /= np.linalg.norm(d)
        pts.append(c + d * np.array([R, R, 0.5 * H]) * 0.78)
        sizes.append(rng.uniform(0.3, 0.42) * R)
    _cluster_crown(b, c, pts, sizes, band, rng, R, 0.5 * H, squash=0.85, subdiv=1, lump=0.25, shade=0.3)
    D, F = icosphere(0)
    for _ in range(int(rng.integers(8, 13))):
        d = rng.normal(size=3)
        d[2] = -abs(d[2]) * 0.6 + 0.2
        d /= np.linalg.norm(d)
        p = c + d * np.array([R, R, 0.5 * H]) * 0.98
        V = D * 0.05 + p
        T = np.stack([np.zeros(len(V)), atlas_v("orange", 0.45 + 0.45 * D[:, 2])], 1)
        b.add(V, F, D, T)


def pine(b: MeshBuilder, x, y, z, H, rng):
    """Stone pine (Pinus pinea): leaning bare trunk, flat umbrella canopy of needle clusters."""
    a = rng.uniform(0, 2 * np.pi)
    lean = rng.uniform(0.08, 0.2) * H
    pts = []
    for t in np.linspace(0, 1, 6):
        bend = lean * (t**1.6)
        wob = 0.15 * np.sin(t * 7 + a)
        pts.append([x + bend * np.cos(a) + wob, y + bend * np.sin(a), z - 0.3 + t * 0.76 * H])
    pts = np.array(pts)
    tube(b, pts, np.linspace(0.34, 0.12, len(pts)) * H / 12, "bark_pine", sides=8, t0=0.25, t1=0.8, uv_len=1.5)
    top = pts[-1]
    band = f"pine{rng.integers(0, 3)}"
    spread = rng.uniform(3.2, 4.6) * H / 12
    cz = top[2] + 0.9
    blob(b, [top[0], top[1], cz + 0.2], [spread * 0.95, spread * 0.95, spread * 0.22], rng, band, subdiv=3, lump=0.12,
         flat_bottom=0.5, light_bias=-0.2)
    cp, cs = [], []
    for _ in range(int(rng.integers(5, 8))):
        aa = rng.uniform(0, 2 * np.pi)
        rr = rng.uniform(0.25, 0.85) * spread
        c = top + [rr * np.cos(aa), rr * np.sin(aa), rng.uniform(0.5, 1.3)]
        tube(b, [top - [0, 0, 1.2], (top + c) / 2 + [0, 0, 0.3], c], [0.14, 0.09, 0.05], "bark_pine", sides=5)
        for _ in range(4):
            q = c + [rng.normal(0, 0.9), rng.normal(0, 0.9), rng.uniform(0.2, 0.7)]
            cp.append(q)
            cs.append(rng.uniform(0.9, 1.4) * H / 12)
    _cluster_crown(b, [top[0], top[1], cz], cp, cs, band, rng, spread, 0.8, squash=0.42, subdiv=2, lump=0.3)


def reed_clump(b: MeshBuilder, x, y, z, H, rng, lean_dir=None, plume_b: MeshBuilder | None = None):
    """Giant cane (Arundo donax) clump: stems with alternate strap leaves; plumes go to `plume_b`."""
    n = int(rng.integers(11, 16))
    pb = plume_b if plume_b is not None else b
    for _ in range(n):
        a = rng.uniform(0, 2 * np.pi)
        r0 = 0.5 * np.sqrt(rng.uniform())
        base = np.array([x + r0 * np.cos(a), y + r0 * np.sin(a), z - 0.1])
        h = H * rng.uniform(0.55, 1.0)
        tilt = rng.uniform(0.06, 0.25) * (0.5 + r0)
        ld = np.array([np.cos(a), np.sin(a), 0.0])
        if lean_dir is not None:
            ld = 0.55 * ld + 0.45 * np.asarray(lean_dir)
        pts = [base + ld * tilt * h * t**1.5 + UP * h * t for t in (0, 0.33, 0.66, 1.0)]
        band = "reed_green" if rng.random() < 0.65 else "reed_tan"
        tube(b, pts, [0.022, 0.017, 0.011, 0.005], band, sides=4, t0=0.35, t1=0.95, uv_len=1.0)
        ang = rng.uniform(0, 2 * np.pi)
        nl = int(rng.integers(4, 6))
        for k in range(nl):
            tn = 0.12 + 0.8 * (k + rng.uniform(0, 0.6)) / nl
            p0 = base + ld * tilt * h * tn**1.5 + UP * h * tn
            ang += np.pi + rng.normal(0, 0.5)  # alternate sides
            dd = np.array([np.cos(ang), np.sin(ang), 0.0])
            L = rng.uniform(0.55, 0.9) * (1.2 - 0.45 * tn)
            lp = [p0, p0 + dd * L * 0.45 + UP * L * 0.2, p0 + dd * L - UP * L * 0.22]
            ribbon(b, lp, [0.055, 0.06, 0.006], np.cross(dd, UP), "reed_leaf", 0.4, 0.95, u0=rng.uniform())
        if h > 0.75 * H and rng.random() < 0.6:
            tip = pts[-1]
            blob(pb, tip + UP * 0.26 + ld * 0.1, [0.065, 0.065, 0.3], rng, "plume", subdiv=1, lump=0.35,
                 light_bias=0.1)


def grass_tuft(b: MeshBuilder, x, y, z, h, rng, n_blades, band, width=0.03, spread=0.25, panicle=None):
    for _ in range(n_blades):
        a = rng.uniform(0, 2 * np.pi)
        r0 = rng.uniform(0, spread * 0.3)
        base = np.array([x + r0 * np.cos(a), y + r0 * np.sin(a), z - 0.02])
        hh = h * rng.uniform(0.55, 1.0)
        out = np.array([np.cos(a), np.sin(a), 0.0]) * spread * rng.uniform(0.4, 1.0)
        pts = [base, base + out * 0.35 + UP * hh * 0.6, base + out * 1.0 + UP * hh * (0.85 + rng.uniform(-0.1, 0.1))]
        ribbon(b, pts, [width, width * 0.8, width * 0.08], np.cross(out + 1e-6, UP), band, 0.15, 0.95,
               up_bias=0.65, u0=rng.uniform())
    if panicle is not None and rng.random() < 0.55:
        tip = np.array([x, y, z + h * rng.uniform(1.0, 1.25)])
        tube(b, [[x, y, z], tip], [0.006, 0.004], "reed_green", sides=3)
        blob(b, tip + UP * 0.12, [0.06, 0.06, 0.16], rng, panicle, subdiv=0, lump=0.3)


def weed_patch(b: MeshBuilder, poly, species: str, height_at, rng, sdf=None):
    """Tufts inside `poly`; `sdf(x, y)` (negative inside) gives an organic outline if provided."""
    from shapely.geometry import Point

    x0, y0, x1, y1 = poly.bounds
    area = poly.area
    if species.startswith("Sorghum"):
        dens, H, blades, band, width, spread, pan = 0.85, 1.5, 6, None, 0.04, 0.38, "panicle"
    elif species.startswith("Cyperus"):
        dens, H, blades, band, width, spread, pan = 1.1, 0.38, 5, "cyperus", 0.014, 0.18, None
    else:
        dens, H, blades, band, width, spread, pan = 0.8, 0.14, 4, "cynodon", 0.012, 0.2, None
    n = int(area * dens)
    m = 5.0 if sdf is not None else 1.0
    xs = rng.uniform(x0 - m, x1 + m, n * 3)
    ys = rng.uniform(y0 - m, y1 + m, n * 3)
    cnt = 0
    for x, y in zip(xs, ys):
        if sdf is not None:
            d = -float(sdf(x, y))
        else:
            d = poly.exterior.distance(Point(x, y)) * (1 if poly.contains(Point(x, y)) else -1)
        p_keep = np.clip((d + 1.2) / 2.5, 0, 1) * (0.75 + 0.25 * hash_u01(int(x * 3), int(y * 3), 77))
        if rng.random() > p_keep:
            continue
        hfac = 0.6 + 0.5 * np.clip((d + 1.0) / 3.0, 0, 1)
        bnd = band or ("sorghum0" if rng.random() < 0.55 else "sorghum1")
        grass_tuft(b, x, y, float(height_at(x, y)), H * hfac * rng.uniform(0.8, 1.1), rng, blades, bnd, width,
                   spread, pan)
        cnt += 1
        if cnt >= n:
            break
    return cnt


# ---------------------------------------------------------------------------
# Structures
# ---------------------------------------------------------------------------


def greenhouse_meshes(film: MeshBuilder, frame: MeshBuilder, poly, H, height_at):
    x0, y0, x1, y1 = poly.bounds
    W = y1 - y0
    a, bz = W / 2.0, H
    yc = (y0 + y1) / 2
    nseg = 18
    th = np.linspace(0, np.pi, nseg + 1)
    xr = np.linspace(x0, x1, max(2, int((x1 - x0) / 8) + 1))
    ny_ = len(th)
    V, Nn, T = [], [], []
    for xi in xr:
        for t in th:
            yy = yc + a * np.cos(t)
            zz = float(height_at(xi, yy)) - 0.05 + bz * np.sin(t) * 1.02
            V.append([xi, yy, zz])
            n = np.array([0, np.cos(t) / a, np.sin(t) / bz])
            Nn.append(n / np.linalg.norm(n))
            T.append([0.0, atlas_v("plastic_film", 0.5 + 0.5 * np.sin(t))])
    V, Nn, T = np.array(V), np.array(Nn), np.array(T)
    F = []
    for i in range(len(xr) - 1):
        for k in range(nseg):
            p, q = i * ny_ + k, i * ny_ + k + 1
            r, s = (i + 1) * ny_ + k + 1, (i + 1) * ny_ + k
            F += [(p, s, r), (p, r, q)]
    F = np.array(F)
    # F winds counter-clockwise seen from inside; outward copy uses reversed winding
    film.add(V, F[:, ::-1], Nn, T)
    film.add(V, F, -Nn, T)
    # end walls
    for xi, sgn in ((x0, -1), (x1, 1)):
        c = np.array([xi, yc, float(height_at(xi, yc)) - 0.05])
        ring = np.array([[xi, yc + a * np.cos(t), float(height_at(xi, yc + a * np.cos(t))) - 0.05 + bz * np.sin(t)]
                         for t in th])
        Vw = np.vstack([c, ring])
        Fw = np.array([(0, k + 1, k + 2) for k in range(nseg)])
        Nw = np.tile([sgn, 0, 0], (len(Vw), 1)).astype(float)
        Tw = np.tile([0.0, atlas_v("plastic_film", 0.7)], (len(Vw), 1))
        # Fw winds counter-clockwise seen from +x
        film.add(Vw, Fw[:, ::-1] if sgn < 0 else Fw, Nw, Tw)
        film.add(Vw, Fw if sgn < 0 else Fw[:, ::-1], -Nw, Tw)
    # steel hoops every ~4 m + ridge purlin
    for xi in np.arange(x0, x1 + 0.01, 4.0):
        g = float(height_at(xi, yc))
        pts = [[xi, yc + a * np.cos(t) * 1.005, g - 0.1 + bz * np.sin(t) * 1.03] for t in np.linspace(0, np.pi, 13)]
        tube(frame, pts, 0.03, "galv", sides=4, t0=0.4, t1=0.9)
    xs = np.arange(x0, x1 + 0.01, 8.0)
    tube(frame, [[xx, yc, float(height_at(xx, yc)) + bz * 1.03 + 0.03] for xx in xs], 0.035, "galv", sides=4)


def pole_and_wires(b: MeshBuilder, poles, height_at, H=9.0):
    tops = []
    for (x, y) in poles:
        z = float(height_at(x, y))
        tube(b, [[x, y, z - 0.5], [x, y, z + H]], [0.15, 0.11], "pole_wood", sides=8, t0=0.3, t1=0.8, uv_len=3.0,
             cap=True)
        box(b, np.array([x, y, z + H - 0.45]), (0.06, 1.1, 0.06), "pole_wood", t=0.55)
        tt = []
        for dy in (-0.9, 0.0, 0.9):
            zz = z + H - 0.39 + (0.18 if dy == 0 else 0.0)
            tube(b, [[x, y + dy, zz - 0.02], [x, y + dy, zz + 0.16]], [0.045, 0.03], "ceramic", sides=6, cap=True)
            tt.append(np.array([x, y + dy, zz + 0.16]))
        tops.append(tt)
    for p, q in zip(tops[:-1], tops[1:]):
        for a, c in zip(p, q):
            L = np.linalg.norm(c - a)
            sag = 0.012 * L
            pts = [a + (c - a) * t - UP * sag * 4 * t * (1 - t) for t in np.linspace(0, 1, 12)]
            tube(b, pts, 0.011, "wire", sides=3)


def stockpile_mesh(b: MeshBuilder, cx, cy, R, H, height_at, rng, uv_scale=2.5):
    nr, ns = 18, 72
    rr = np.linspace(0, 1, nr)
    th = np.linspace(0, 2 * np.pi, ns, endpoint=False)
    k = 7
    w = rng.uniform(0, 2 * np.pi, k)
    amp = rng.uniform(0.02, 0.06, k)
    rad_mod = 1 + sum(amp[i] * np.sin((i + 2) * th + w[i]) for i in range(k))
    V = []
    for r in rr:
        for j, t in enumerate(th):
            rad = r * R * rad_mod[j]
            x, y = cx + rad * np.cos(t), cy + rad * np.sin(t)
            prof = (1 - r**1.7) ** 0.9 if r < 1 else 0.0
            lump = 0.12 * np.sin(5 * t + 9 * r + w[0]) * np.sin(3 * r * 3.1) + 0.08 * np.sin(11 * t + w[1]) * r
            zz = float(height_at(x, y)) + H * prof + lump * (1 - r) - 0.12 * r**8
            V.append([x, y, zz])
    V = np.array(V)
    F = []
    for i in range(nr - 1):
        for j in range(ns):
            a, bq = i * ns + j, i * ns + (j + 1) % ns
            c, d = (i + 1) * ns + (j + 1) % ns, (i + 1) * ns + j
            if i == 0:
                F.append((a, c, d))
            else:
                F += [(a, bq, c), (a, c, d)]
    F = np.array(F)
    T = np.stack([(V[:, 0] - cx) / uv_scale, (V[:, 1] - cy) / uv_scale], 1)
    b.add(V, F, None, T)


def fallen_log(b: MeshBuilder, a, c, radius, ground_at, rng):
    """Fallen poplar: tapered bent trunk lying on the ground + root plate + broken branches."""
    a = np.asarray(a, float)
    c = np.asarray(c, float)
    L = np.linalg.norm(c - a)
    d = (c - a) / L
    side = np.array([-d[1], d[0]])
    pts, rads = [], []
    for t in np.linspace(0, 1, 14):
        xy = a + d * L * t + side * 0.18 * np.sin(t * 3.0 + 0.4)
        r = radius * (1.0 - 0.55 * t) + 0.03 * np.sin(17 * t)
        z = float(ground_at(*xy)) + r - 0.04
        pts.append([xy[0], xy[1], z])
        rads.append(r)
    tube(b, pts, rads, "log_bark", sides=12, t0=0.25, t1=0.9, uv_len=1.4, cap=False)
    for _ in range(9):  # bark knots / burls
        i = int(rng.integers(1, len(pts) - 1))
        ang = rng.uniform(0, 2 * np.pi)
        nrm = np.array([side[0] * np.cos(ang), side[1] * np.cos(ang), np.sin(ang)])
        kr = rng.uniform(0.07, 0.14)
        blob(b, np.array(pts[i]) + nrm * rads[i] * 0.92, [kr, kr, kr * 0.7], rng, "log_bark", subdiv=1, lump=0.3,
             light_bias=-0.1)
    # broken top end cap (pale wood)
    D, F = icosphere(1)
    top = np.array(pts[-1])
    V = top + D * np.array([rads[-1] * 0.25, rads[-1], rads[-1]])
    T = np.stack([np.zeros(len(V)), atlas_v("log_wood", 0.7 + 0.2 * D[:, 2])], 1)
    b.add(V, F, D, T)
    # root plate standing up at the base
    base = np.array(pts[0])
    rd = np.array([-d[0], -d[1], 0.0])
    blob(b, base + rd * 0.35 + UP * 0.55, [0.45, 1.6, 1.3], rng, "root_soil", subdiv=2, lump=0.3)
    for _ in range(9):
        ang = rng.uniform(0, 2 * np.pi)
        tip = base + rd * rng.uniform(0.3, 0.8) + np.array([side[0], side[1], 0]) * np.cos(ang) * rng.uniform(1.0, 2.0) \
            + UP * (0.6 + np.sin(ang) * rng.uniform(0.6, 1.4))
        tube(b, [base + UP * 0.5, (base + UP * 0.5 + tip) / 2 + rng.normal(0, 0.1, 3), tip], [0.12, 0.07, 0.02],
             "root_soil", sides=5)
    # broken branches
    for _ in range(int(rng.integers(6, 9))):
        t = rng.uniform(0.35, 0.95)
        i = int(t * (len(pts) - 1))
        p0 = np.array(pts[i])
        up = rng.uniform(0.2, 1.0)
        sd = rng.choice([-1, 1])
        dirv = (np.array([side[0] * sd, side[1] * sd, 0]) * rng.uniform(0.4, 1.0)
                + np.array([d[0], d[1], 0]) * rng.uniform(0.2, 0.8) + UP * up)
        dirv /= np.linalg.norm(dirv)
        Lb = rng.uniform(1.0, 2.6)
        rb = rng.uniform(0.06, 0.13)
        mid = p0 + dirv * Lb * 0.5 + rng.normal(0, 0.12, 3)
        end = p0 + dirv * Lb
        end[2] = max(end[2], float(ground_at(end[0], end[1])) + 0.05)
        tube(b, [p0, mid, end], [rb, rb * 0.75, rb * 0.5], "dead_branch", sides=5, cap=True)
        if rng.random() < 0.6:
            blob(b, end + UP * 0.1, [0.35, 0.35, 0.25], rng, "leaf_dead", subdiv=1, lump=0.35)
