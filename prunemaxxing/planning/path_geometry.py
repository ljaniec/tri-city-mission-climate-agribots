"""Polyline utilities: arc length, densify, corner fillets, obstacle detours."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.prepared import prep


def arclength(path: np.ndarray) -> np.ndarray:
    if len(path) == 0:
        return np.zeros(0)
    d = np.hypot(*np.diff(path, axis=0).T)
    return np.concatenate([[0.0], np.cumsum(d)])


def densify(path: np.ndarray, step: float = 0.5, labels: np.ndarray | None = None):
    """Resample a polyline at ~uniform spacing, keeping original vertices."""
    path = np.asarray(path, dtype=float)
    out = [path[0]]
    out_lab = [labels[0]] if labels is not None else None
    for i in range(1, len(path)):
        a, b = path[i - 1], path[i]
        seg = float(np.hypot(*(b - a)))
        if seg < 1e-9:
            continue
        n = max(1, int(math.ceil(seg / step)))
        for k in range(1, n + 1):
            out.append(a + (b - a) * (k / n))
            if out_lab is not None:
                out_lab.append(labels[i])
    res = np.asarray(out)
    if labels is None:
        return res
    return res, np.asarray(out_lab, dtype=object)


def headings(path: np.ndarray) -> np.ndarray:
    d = np.gradient(path, axis=0)
    return np.arctan2(d[:, 1], d[:, 0])


def curvature(path: np.ndarray) -> np.ndarray:
    if len(path) < 3:
        return np.zeros(len(path))
    s = arclength(path)
    th = np.unwrap(headings(path))
    ds = np.gradient(s)
    ds[ds < 1e-6] = 1e-6
    return np.gradient(th) / ds


@dataclass
class Piece:
    pts: np.ndarray  # (k, 2)
    label: str


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.hypot(*v))
    return v / n if n > 1e-12 else v * 0.0


def fillet_polyline(
    pts: np.ndarray,
    seg_labels: list[str],
    radius: float,
    arc_step: float = 0.4,
) -> list[Piece]:
    """Replace each interior corner by a tangent circular arc of ``radius``.

    ``seg_labels[i]`` labels segment pts[i] -> pts[i+1]. Interior segments shared
    by two fillets give each at most half their length; the first and last
    segments may be consumed entirely. Returns straight/arc pieces with labels.
    """
    pts = np.asarray(pts, dtype=float)
    # drop duplicate points
    keep = [0]
    labs: list[str] = []
    for i in range(1, len(pts)):
        if np.hypot(*(pts[i] - pts[keep[-1]])) > 1e-6:
            keep.append(i)
            labs.append(seg_labels[i - 1])
    pts = pts[keep]
    n = len(pts)
    if n < 3:
        return [Piece(pts.copy(), labs[0] if labs else "transit")]

    seg_len = np.hypot(*np.diff(pts, axis=0).T)
    want = np.zeros(n)
    geo: dict[int, tuple] = {}
    for i in range(1, n - 1):
        u1 = _unit(pts[i] - pts[i - 1])
        u2 = _unit(pts[i + 1] - pts[i])
        cross = u1[0] * u2[1] - u1[1] * u2[0]
        dot = float(np.clip(np.dot(u1, u2), -1.0, 1.0))
        theta = math.acos(dot)
        if theta < 1e-3 or theta > math.pi - 1e-3:
            continue
        geo[i] = (u1, u2, cross, theta)
        want[i] = radius * math.tan(theta / 2.0)
    # neighbouring fillets share a segment: scale both tangents to fit
    scale = np.ones(n)
    for j in range(n - 1):
        need = want[j] + want[j + 1]
        if need > seg_len[j] and need > 0:
            f = seg_len[j] / need
            scale[j] = min(scale[j], f)
            scale[j + 1] = min(scale[j + 1], f)
    corner_data: dict[int, tuple] = {}
    for i, (u1, u2, cross, theta) in geo.items():
        t = want[i] * scale[i]
        r_eff = t / math.tan(theta / 2.0)
        corner_data[i] = (u1, u2, cross, theta, t, r_eff)

    pieces: list[Piece] = []
    cursor = pts[0].copy()
    for i in range(1, n):
        lab = labs[i - 1]
        if i in corner_data:
            u1, u2, cross, theta, t, r_eff = corner_data[i]
            t1 = pts[i] - u1 * t
            pieces.append(Piece(np.vstack([cursor, t1]), lab))
            if r_eff > 1e-6:
                normal = np.array([-u1[1], u1[0]]) if cross > 0 else np.array([u1[1], -u1[0]])
                center = t1 + normal * r_eff
                a0 = math.atan2(t1[1] - center[1], t1[0] - center[0])
                sweep = theta if cross > 0 else -theta
                k = max(2, int(math.ceil(abs(sweep) * r_eff / arc_step)) + 1)
                ang = a0 + np.linspace(0.0, sweep, k)
                arc = np.column_stack([center[0] + r_eff * np.cos(ang), center[1] + r_eff * np.sin(ang)])
                arc_lab = labs[i] if lab == "lane" else lab
                pieces.append(Piece(arc, arc_lab))
                cursor = arc[-1].copy()
            else:
                cursor = t1
        else:
            pieces.append(Piece(np.vstack([cursor, pts[i]]), lab))
            cursor = pts[i].copy()
    return pieces


def pieces_to_path(pieces: list[Piece], step: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    pts: list[np.ndarray] = []
    labs: list[str] = []
    for pc in pieces:
        p = pc.pts
        if len(p) < 2:
            continue
        dp, dl = densify(p, step, np.array([pc.label] * len(p), dtype=object))
        if pts and np.hypot(*(pts[-1][-1] - dp[0])) < 1e-6:
            dp, dl = dp[1:], dl[1:]
        if len(dp) == 0:
            continue
        pts.append(dp)
        labs.extend(dl.tolist())
    path = np.vstack(pts)
    return dedupe(path, np.asarray(labs, dtype=object), tol=0.2 * step)


def dedupe(path: np.ndarray, labels: np.ndarray, tol: float = 0.05):
    keep = [0]
    for i in range(1, len(path)):
        if np.hypot(*(path[i] - path[keep[-1]])) >= tol:
            keep.append(i)
    if keep[-1] != len(path) - 1:
        keep[-1] = len(path) - 1
    return path[keep], labels[keep]


def min_turn_radius(path: np.ndarray) -> float:
    k = np.abs(curvature(path))
    k = k[np.isfinite(k)]
    if len(k) == 0 or k.max() < 1e-9:
        return float("inf")
    return float(1.0 / np.percentile(k, 99.5))


# ---------------------------------------------------------------------------
# Detours around polygons (planning-time for mapped weeds, run-time for lidar
# discovered obstacles)
# ---------------------------------------------------------------------------


@dataclass
class Detour:
    i0: int  # path index where detour leaves the original path
    i1: int  # path index where it rejoins
    pts: np.ndarray  # detour polyline (dense), from path[i0] to path[i1]
    side: str
    offset_m: float
    length_m: float


def plan_detour(
    path: np.ndarray,
    s: np.ndarray,
    obstacle: Polygon,
    clearance: float,
    turn_radius: float,
    s_min: float = 0.0,
    allowed: Polygon | None = None,
    avoid: list[Polygon] | None = None,
    step: float = 0.5,
    lead: float | None = None,
) -> Detour | None:
    """Lateral bypass of ``obstacle`` (inflated by ``clearance``) for a dense path.

    The bypass leaves the path ``lead`` metres before the first blocked point
    (never before ``s_min``), runs parallel to the local path direction beside the
    inflated obstacle and rejoins after it. Corners are filleted with
    ``turn_radius``. Both sides are tried; the shorter collision-free one wins.
    """
    inflated = obstacle.buffer(clearance, join_style=1)
    pinf = prep(inflated)
    inside = np.array([pinf.contains(Point(p)) for p in path])
    blocked = np.where(inside & (s >= s_min - 1e-6))[0]
    if len(blocked) == 0:
        return None
    b0, b1 = int(blocked[0]), int(blocked[-1])
    # consolidate: only the first contiguous blocked run (+ rejoin after it)
    runs = np.split(blocked, np.where(np.diff(blocked) > 1)[0] + 1)
    b0, b1 = int(runs[0][0]), int(runs[0][-1])
    for run in runs[1:]:
        if s[run[0]] - s[b1] < 3 * turn_radius:
            b1 = int(run[-1])

    lead = 1.6 * turn_radius if lead is None else lead
    s_a = max(s_min, s[b0] - lead)
    s_b = min(s[-1], s[b1] + lead)
    i0 = int(np.searchsorted(s, s_a))
    i1 = int(np.searchsorted(s, s_b))
    i1 = min(i1, len(path) - 1)
    if i1 <= i0 + 1:
        return None
    # anchors before/after so the departure and re-join corners are filleted too
    pre = 1.2 * turn_radius
    i_pre = int(np.searchsorted(s, max(s_min, s[i0] - pre)))
    i_post = min(len(path) - 1, int(np.searchsorted(s, s[i1] + pre)))
    a_pre, b_post = path[i_pre], path[i_post]
    a, b = path[i0], path[i1]
    u = _unit(b - a)
    if np.hypot(*(b - a)) < 1e-6:
        return None
    v = np.array([-u[1], u[0]])
    coords = np.asarray(inflated.exterior.coords)
    rel = coords - a
    cu = rel @ u
    cv = rel @ v
    umin, umax = float(cu.min()), float(cu.max())
    L = float((b - a) @ u)
    umin = max(umin, 0.5)
    umax = min(umax, L - 0.5)

    hard = obstacle.buffer(max(clearance - 0.6, 0.6))
    best: Detour | None = None
    for side, base in (("left", float(cv.max())), ("right", float(cv.min()))):
        sign = 1.0 if side == "left" else -1.0
        for extra in np.arange(0.4, 8.0, 0.6):
            off = base + sign * extra
            c1 = a + u * umin + v * off
            c2 = a + u * umax + v * off
            poly = [a_pre] if i_pre < i0 else []
            poly += [a, c1, c2, b]
            poly += [b_post] if i_post > i1 else []
            pieces = fillet_polyline(np.vstack(poly), ["detour"] * (len(poly) - 1), turn_radius)
            pts, _ = pieces_to_path(pieces, step)
            line = LineString(pts)
            if line.intersects(hard):
                continue
            if allowed is not None and not allowed.buffer(0.01).contains(line):
                break
            if avoid and any(line.intersects(p) for p in avoid):
                break
            det = Detour(i_pre, i_post, pts, side, abs(off), float(line.length))
            if best is None or det.length_m < best.length_m:
                best = det
            break
    return best


def splice(path: np.ndarray, labels: np.ndarray, det: Detour, label: str, step: float = 0.5):
    new_path = np.vstack([path[: det.i0], det.pts, path[det.i1 + 1 :]])
    new_lab = np.concatenate(
        [labels[: det.i0], np.array([label] * len(det.pts), dtype=object), labels[det.i1 + 1 :]]
    )
    return dedupe(new_path, new_lab, tol=0.2 * step)
