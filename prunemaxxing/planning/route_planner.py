"""Global mothership route planner on the known map.

Route = depot -> parcels (boustrophedon lanes on furrow centres, curved headland
U-turns, mapped-weed bypasses) -> back to depot, with inter-parcel transfers on
the farm-track network. Drone service stations are placed along the lanes and
every work cell is assigned to its nearest station.

Only *known* information is used here (site_known.geojson). Hidden obstacles are
handled online by :mod:`prunemaxxing.planning.mission`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import networkx as nx
import numpy as np
import shapely
from shapely.geometry import LineString, MultiLineString, Point, Polygon, box
from shapely.ops import nearest_points, unary_union
from shapely.prepared import prep

from prunemaxxing.config import (
    DEFAULT_TARGET_KG_PER_M2,
    DRONE_OPS_RADIUS_SHOWCASE_M,
    HEADLAND_BED_MARGIN_M,
    LANE_SPACING_M,
    STATION_SPACING_M,
)
from prunemaxxing.geo.grid import FieldTwin
from prunemaxxing.geo.site import Site
from prunemaxxing.planning.path_geometry import (
    arclength,
    fillet_polyline,
    pieces_to_path,
    plan_detour,
    splice,
)


@dataclass
class PlannerConfig:
    lane_spacing_m: float = LANE_SPACING_M
    station_spacing_m: float = STATION_SPACING_M
    headland_turn_radius_m: float = 6.5
    transfer_turn_radius_m: float = 7.0
    weed_clearance_m: float = 3.5  # vehicle half-width 1.0 + 2.5 m rhizome-spread buffer
    headland_bed_margin_m: float = HEADLAND_BED_MARGIN_M
    ops_radius_m: float = DRONE_OPS_RADIUS_SHOWCASE_M
    target_kg_per_m2: float = DEFAULT_TARGET_KG_PER_M2
    weed_cell_buffer_m: float = 1.0
    path_step_m: float = 0.5


@dataclass
class Lane:
    lid: int
    parcel_id: str
    x: float
    y_lo: float
    y_hi: float
    direction: int  # +1 northbound, -1 southbound
    s0: float = 0.0
    s1: float = 0.0


@dataclass
class Station:
    sid: int
    parcel_id: str
    lane_id: int
    xy: np.ndarray
    s: float
    cells: list[int] = field(default_factory=list)


@dataclass
class WorkCell:
    idx: int
    row: int
    col: int
    e: float
    n: float
    parcel_id: str
    area_m2: float
    target_kg: float
    station: int = -1
    dist_m: float = 0.0
    polygon: Polygon | None = None


@dataclass
class GlobalPlan:
    path: np.ndarray  # (N, 2) UTM, ~0.5 m spacing
    s: np.ndarray
    labels: np.ndarray  # transit | lane | headland | weed_detour
    lanes: list[Lane]
    stations: list[Station]
    cells: list[WorkCell]
    depot_xy: np.ndarray
    start_heading: float
    inflated_weeds: list[Polygon]
    bedded_areas: dict[str, Polygon]
    cfg: PlannerConfig
    network: nx.Graph | None = None
    unreachable_cells: list[int] = field(default_factory=list)

    @property
    def length_m(self) -> float:
        return float(self.s[-1])

    def label_length(self) -> dict[str, float]:
        out: dict[str, float] = {}
        ds = np.diff(self.s)
        for lab, d in zip(self.labels[1:], ds):
            out[lab] = out.get(lab, 0.0) + float(d)
        return out


# ---------------------------------------------------------------------------


def _bedded_area(parcel: Polygon, margin: float) -> Polygon:
    inner = parcel.buffer(-margin, join_style=2)
    return inner if not inner.is_empty else parcel


def _lane_positions(parcel: Polygon, spacing: float, bed_spacing: float) -> list[float]:
    minx, _, maxx, _ = parcel.bounds
    width = maxx - minx
    n = max(1, int(math.ceil(width / spacing)))
    xs = []
    for i in range(n):
        x = minx + width * (i + 0.5) / n
        k = round((x - minx) / bed_spacing - 0.5)
        xs.append(minx + (k + 0.5) * bed_spacing)
    return xs


def _lane_extent(parcel: Polygon, x: float, inset: float) -> tuple[float, float] | None:
    inner = parcel.buffer(-inset, join_style=2)
    if inner.is_empty:
        return None
    _, miny, _, maxy = parcel.bounds
    seg = inner.intersection(LineString([(x, miny - 1), (x, maxy + 1)]))
    if seg.is_empty:
        return None
    if isinstance(seg, MultiLineString):
        seg = max(seg.geoms, key=lambda g: g.length)
    ys = [c[1] for c in seg.coords]
    return min(ys), max(ys)


def build_track_network(site: Site) -> nx.Graph:
    depot_c = site.depot.polygon.centroid
    lines = [t.line for t in site.tracks]

    def link(target: LineString) -> LineString:
        q = np.array(nearest_points(target, depot_c)[0].coords[0])
        c = np.array(depot_c.coords[0])
        d = q - c
        n = float(np.linalg.norm(d))
        # overshoot so noding registers a proper crossing
        return LineString([c, q + d / n * 0.5]) if n > 1e-6 else LineString([c, c + 0.5])

    # depot links: west farm track and the gravel access to the road
    for tid in ("track_west", "depot_access"):
        target = next((t.line for t in site.tracks if t.tid == tid), None)
        if target is not None:
            lines.append(link(target))
    noded = unary_union([shapely.set_precision(ln, 0.001) for ln in lines])
    geoms = list(noded.geoms) if hasattr(noded, "geoms") else [noded]
    g = nx.Graph()
    for ln in geoms:
        cs = [tuple(np.round(c, 3)) for c in ln.coords]
        for a, b in zip(cs[:-1], cs[1:]):
            g.add_edge(a, b, weight=float(math.dist(a, b)))
    return g


def _snap_into(g: nx.Graph, p: np.ndarray) -> tuple:
    """Insert projection of p onto nearest edge; returns node key."""
    best = None
    pt = Point(p)
    for a, b in g.edges():
        seg = LineString([a, b])
        d = seg.distance(pt)
        if best is None or d < best[0]:
            best = (d, a, b, seg)
    _, a, b, seg = best
    q = seg.interpolate(seg.project(pt))
    key = (round(q.x, 3), round(q.y, 3))
    if key in (a, b):
        return key
    g.remove_edge(a, b)
    g.add_edge(a, key, weight=float(math.dist(a, key)))
    g.add_edge(key, b, weight=float(math.dist(key, b)))
    return key


def _network_route(g: nx.Graph, p: np.ndarray, q: np.ndarray) -> list[np.ndarray]:
    h = g.copy()
    kp = _snap_into(h, p)
    kq = _snap_into(h, q)
    nodes = nx.shortest_path(h, kp, kq, weight="weight")
    pts = [np.array(n, dtype=float) for n in nodes]
    # drop collinear interior nodes
    out = [pts[0]]
    for i in range(1, len(pts) - 1):
        a, b, c = out[-1], pts[i], pts[i + 1]
        u, v = b - a, c - b
        if abs(u[0] * v[1] - u[1] * v[0]) > 1e-6 * (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9):
            out.append(b)
    if len(pts) > 1:
        out.append(pts[-1])
    return out


def _segment_ok(a: np.ndarray, b: np.ndarray, no_go: list) -> bool:
    ln = LineString([a, b])
    return not any(ln.intersects(z) for z in no_go)


def _transfer(
    g: nx.Graph,
    p: np.ndarray,
    p_dir: np.ndarray | None,
    q: np.ndarray,
    q_dir: np.ndarray | None,
    no_go: list,
    lead: float,
) -> list[np.ndarray]:
    """Waypoints from p to q (exclusive of p, inclusive of q)."""
    if _segment_ok(p, q, no_go):
        d = q - p
        ok = True
        for u in (p_dir, q_dir):
            if u is not None and np.linalg.norm(d) > 1e-6:
                cosang = float(np.dot(u, d) / np.linalg.norm(d))
                ok &= cosang > math.cos(math.radians(35))
        if ok:
            return [q]
    p_out = p + p_dir * lead if p_dir is not None else p
    q_in = q - q_dir * lead if q_dir is not None else q
    route = _network_route(g, p_out, q_in)
    return route + [q]


@dataclass
class _ParcelPlan:
    lanes: list[Lane]
    entry: np.ndarray
    entry_dir: np.ndarray
    exit: np.ndarray
    exit_dir: np.ndarray


def _parcel_lanes(parcel, cfg: PlannerConfig, west_to_east: bool, start_north: bool, lid0: int) -> _ParcelPlan:
    xs = _lane_positions(parcel.polygon, cfg.lane_spacing_m, parcel.bed_spacing_m)
    if not west_to_east:
        xs = xs[::-1]
    lanes: list[Lane] = []
    direction = +1 if start_north else -1
    inset = cfg.headland_turn_radius_m + 1.0
    for i, x in enumerate(xs):
        ext = _lane_extent(parcel.polygon, x, inset)
        if ext is None:
            continue
        lanes.append(Lane(lid0 + len(lanes), parcel.pid, x, ext[0], ext[1], direction))
        direction = -direction
    first, last = lanes[0], lanes[-1]
    entry = np.array([first.x, first.y_lo if first.direction > 0 else first.y_hi])
    exit_ = np.array([last.x, last.y_hi if last.direction > 0 else last.y_lo])
    return _ParcelPlan(lanes, entry, np.array([0.0, first.direction]), exit_, np.array([0.0, last.direction]))


def plan_global_route(site: Site, twin: FieldTwin, cfg: PlannerConfig | None = None) -> GlobalPlan:
    cfg = cfg or PlannerConfig()
    g = build_track_network(site)
    bedded = {p.pid: _bedded_area(p.polygon, cfg.headland_bed_margin_m) for p in site.parcels}
    inflated_weeds = [w.polygon.buffer(cfg.weed_clearance_m, join_style=1) for w in site.weeds]
    no_go = list(bedded.values()) + inflated_weeds + [d.line.buffer(d.width_m / 2 + 0.5) for d in site.ditches]

    depot = np.array(site.depot.polygon.centroid.coords[0])
    coarse: list[np.ndarray] = [depot]
    seg_labels: list[str] = []
    lanes_all: list[Lane] = []
    remaining = list(site.parcels)
    pos, pos_dir = depot, None
    lead = cfg.transfer_turn_radius_m * 1.6

    while remaining:
        best = None
        for parcel in remaining:
            for we in (True, False):
                for sn in (True, False):
                    pp = _parcel_lanes(parcel, cfg, we, sn, len(lanes_all))
                    route = _transfer(g, pos, pos_dir, pp.entry, pp.entry_dir, no_go, lead)
                    length = float(sum(np.linalg.norm(b - a) for a, b in zip([pos] + route[:-1], route)))
                    if best is None or length < best[0]:
                        best = (length, parcel, pp, route)
        _, parcel, pp, route = best
        remaining.remove(parcel)
        for wp in route:
            coarse.append(wp)
            seg_labels.append("transit")
        R = cfg.headland_turn_radius_m
        for i, lane in enumerate(pp.lanes):
            start = np.array([lane.x, lane.y_lo if lane.direction > 0 else lane.y_hi])
            end = np.array([lane.x, lane.y_hi if lane.direction > 0 else lane.y_lo])
            if np.linalg.norm(coarse[-1] - start) > 1e-6:
                coarse.append(start)
                seg_labels.append("headland")
            coarse.append(end)
            seg_labels.append("lane")
            if i + 1 < len(pp.lanes):
                nxt = pp.lanes[i + 1]
                y_turn = end[1] + lane.direction * R
                coarse.append(np.array([lane.x, y_turn]))
                seg_labels.append("headland")
                coarse.append(np.array([nxt.x, y_turn]))
                seg_labels.append("headland")
        lanes_all.extend(pp.lanes)
        pos, pos_dir = pp.exit, pp.exit_dir

    for wp in _transfer(g, pos, pos_dir, depot, None, no_go, lead):
        coarse.append(wp)
        seg_labels.append("transit")

    # Fillet: headland corners with the headland radius, others with transfer radius
    pieces = fillet_polyline(np.vstack(coarse), seg_labels, cfg.headland_turn_radius_m)
    path, labels = pieces_to_path(pieces, cfg.path_step_m)
    s = arclength(path)

    # Mapped-weed bypasses on lanes (known map)
    for w_poly in [w.polygon for w in site.weeds]:
        parcel = site.parcel_at(*w_poly.centroid.coords[0])
        allowed = parcel.polygon.buffer(-0.5) if parcel else None
        det = plan_detour(path, s, w_poly, cfg.weed_clearance_m, cfg.headland_turn_radius_m,
                          allowed=allowed, step=cfg.path_step_m, lead=2.2 * cfg.headland_turn_radius_m)
        if det is not None:
            path, labels = splice(path, labels, det, "weed_detour", cfg.path_step_m)
            s = arclength(path)

    # lane s-ranges on the final path
    for lane in lanes_all:
        on_lane = np.where((labels == "lane") & (np.abs(path[:, 0] - lane.x) < 0.05)
                           & (path[:, 1] >= lane.y_lo - 0.5) & (path[:, 1] <= lane.y_hi + 0.5))[0]
        if len(on_lane):
            lane.s0, lane.s1 = float(s[on_lane[0]]), float(s[on_lane[-1]])

    stations = _place_stations(path, s, labels, lanes_all, bedded, cfg)
    cells, unreachable = _assign_cells(site, twin, stations, cfg)
    # drop stations that ended up serving no cell and re-index
    keep = [st for st in stations if st.cells]
    remap = {st.sid: k for k, st in enumerate(keep)}
    for k, st in enumerate(keep):
        st.sid = k
    for wc in cells:
        wc.station = remap.get(wc.station, -1)
    stations = keep
    yaw0 = math.atan2(path[min(4, len(path) - 1), 1] - path[0, 1], path[min(4, len(path) - 1), 0] - path[0, 0])
    return GlobalPlan(path, s, labels, lanes_all, stations, cells, depot, yaw0, inflated_weeds, bedded, cfg, g,
                      unreachable)


def _place_stations(path, s, labels, lanes, bedded, cfg: PlannerConfig) -> list[Station]:
    stations: list[Station] = []
    for lane in lanes:
        if lane.s1 <= lane.s0:
            continue
        bed = bedded[lane.parcel_id]
        _, by0, _, by1 = bed.bounds
        # stations evenly over the bedded stretch of the lane
        idx = np.where((s >= lane.s0) & (s <= lane.s1))[0]
        ys = path[idx, 1]
        inside = idx[(ys >= by0) & (ys <= by1)]
        if len(inside) == 0:
            continue
        sa, sb = float(s[inside[0]]), float(s[inside[-1]])
        n = max(1, int(math.ceil((sb - sa) / cfg.station_spacing_m)))
        for k in range(n):
            st = sa + (sb - sa) * (k + 0.5) / n
            i = int(np.searchsorted(s, st))
            # never park on a weed bypass: slide to the nearest plain lane point
            if labels[i] != "lane":
                cand = np.where((labels == "lane") & (s >= lane.s0) & (s <= lane.s1))[0]
                if len(cand) == 0:
                    continue
                i = int(cand[np.argmin(np.abs(s[cand] - st))])
            stations.append(Station(len(stations), lane.parcel_id, lane.lid, path[i].copy(), float(s[i])))
    stations.sort(key=lambda st: st.s)
    for k, st in enumerate(stations):
        st.sid = k
    return stations


def _assign_cells(site: Site, twin: FieldTwin, stations: list[Station], cfg: PlannerConfig):
    dem = twin.dem
    weeds = unary_union([w.polygon.buffer(cfg.weed_cell_buffer_m) for w in site.weeds]) if site.weeds else None
    cells: list[WorkCell] = []
    unreachable: list[int] = []
    st_xy = np.array([st.xy for st in stations]) if stations else np.zeros((0, 2))
    st_parcel = np.array([st.parcel_id for st in stations])
    for r in range(twin.nrows):
        for c in range(twin.ncols):
            e0, e1 = dem.easting(c), dem.easting(c + 1)
            n1, n0 = dem.northing(r), dem.northing(r + 1)
            sq = box(e0, n0, e1, n1)
            ce, cn = (e0 + e1) / 2, (n0 + n1) / 2
            parcel = site.parcel_at(ce, cn)
            if parcel is None:
                continue
            area_poly = sq.intersection(parcel.polygon)
            if weeds is not None:
                area_poly = area_poly.difference(weeds)
            area = float(area_poly.area)
            if area < 0.35 * sq.area:
                continue
            wc = WorkCell(len(cells), r, c, ce, cn, parcel.pid, area, area * cfg.target_kg_per_m2,
                          polygon=area_poly)
            mask = st_parcel == parcel.pid
            if mask.any():
                cand = np.where(mask)[0]
                d = np.hypot(st_xy[cand, 0] - ce, st_xy[cand, 1] - cn)
                k = int(cand[np.argmin(d)])
                wc.station, wc.dist_m = k, float(d.min())
                stations[k].cells.append(wc.idx)
                if wc.dist_m > cfg.ops_radius_m:
                    unreachable.append(wc.idx)
            cells.append(wc)
    return cells, unreachable
