"""Pilot site model: parcels, farm tracks, ditches, weeds, hidden obstacles, scenery.

Everything is stored in UTM 31N metres (shapely geometries). GeoJSON files keep
CRS84 lon/lat so they open in QGIS / geojson.io.

Three layers are kept deliberately separate:

* ``site_known.geojson``      – what the planner knows (survey / satellite):
  parcels, tracks, ditches, depot, stockpiles, mapped weed patches.
* ``unknown_obstacles.geojson`` – ground truth the planner cannot see; only the
  mothership lidar discovers these at run time.
* ``scenery.geojson``          – trees, reeds, buildings, greenhouses (rendering only).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import shapely
from shapely.geometry import LineString, Point, Polygon, mapping, shape

from prunemaxxing.geo.crs import utm31n_to_wgs84, wgs84_to_utm31n

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
SITE_KNOWN_PATH = DATA_DIR / "site_known.geojson"
UNKNOWN_OBSTACLES_PATH = DATA_DIR / "unknown_obstacles.geojson"
SCENERY_PATH = DATA_DIR / "scenery.geojson"


@dataclass
class Parcel:
    pid: str
    polygon: Polygon
    crop: str
    bed_spacing_m: float


@dataclass
class Track:
    tid: str
    line: LineString
    width_m: float
    surface: str  # asphalt | dirt | gravel


@dataclass
class Zone:
    zid: str
    polygon: Polygon
    kind: str  # weed_exclusion | depot | ...
    props: dict[str, Any] = field(default_factory=dict)


@dataclass
class Obstacle:
    oid: str
    kind: str  # tractor | log | pump | waterlogged
    polygon: Polygon
    height_m: float
    yaw_rad: float = 0.0
    props: dict[str, Any] = field(default_factory=dict)


@dataclass
class SceneryItem:
    kind: str  # poplar | olive | pine | citrus | reed | house | greenhouse | pole
    geometry: Any
    props: dict[str, Any] = field(default_factory=dict)


@dataclass
class Site:
    origin_e: float
    origin_n: float
    width_m: float
    height_m: float
    parcels: list[Parcel]
    tracks: list[Track]
    ditches: list[Track]
    weeds: list[Zone]
    depot: Zone
    stockpiles: list[Point]
    obstacles: list[Obstacle] = field(default_factory=list)  # hidden ground truth
    scenery: list[SceneryItem] = field(default_factory=list)

    def local(self, e: float, n: float) -> tuple[float, float]:
        return e - self.origin_e, n - self.origin_n

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return (
            self.origin_e,
            self.origin_n,
            self.origin_e + self.width_m,
            self.origin_n + self.height_m,
        )

    def parcel_at(self, e: float, n: float) -> Parcel | None:
        p = Point(e, n)
        for parcel in self.parcels:
            if parcel.polygon.contains(p):
                return parcel
        return None


# ---------------------------------------------------------------------------
# Site design (local metres from the SW corner of the 12 ha DEM window)
# ---------------------------------------------------------------------------

BED_SPACING_M = 0.85


def _rect(x0: float, y0: float, x1: float, y1: float) -> Polygon:
    return Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


def _rot_rect(cx: float, cy: float, w: float, h: float, yaw: float) -> Polygon:
    c, s = np.cos(yaw), np.sin(yaw)
    pts = []
    for dx, dy in ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)):
        pts.append((cx + c * dx - s * dy, cy + s * dx + c * dy))
    return Polygon(pts)


def design_pilot_site(origin_e: float, origin_n: float, width_m: float, height_m: float) -> Site:
    """Parc Agrari–style block: 4 parcels, cross tracks, ditches with reeds, depot."""

    def shift(geom):
        return shapely.transform(geom, lambda xy: xy + np.array([origin_e, origin_n]))

    W, H = width_m, height_m
    parcels = [
        Parcel("A_sweet_potato", shift(_rect(14, 24, 171, 171)), "sweet_potato_beauregard", BED_SPACING_M),
        Parcel("B_carboniato", shift(_rect(179, 14, 337, 171)), "carboniato_ohenry", BED_SPACING_M),
        Parcel("C_transition", shift(_rect(179, 179, 337, 329)), "cover_then_sweet_potato", BED_SPACING_M),
        Parcel("D_horticulture", shift(_rect(14, 179, 171, 329)), "horticulture_mix", BED_SPACING_M),
    ]
    tracks = [
        Track("road_south", shift(LineString([(-120, 6), (W + 140, 6)])), 6.5, "asphalt"),
        Track("track_west", shift(LineString([(10, 6), (10, 333)])), 4.0, "dirt"),
        Track("track_east", shift(LineString([(341, 6), (341, 333)])), 4.0, "dirt"),
        Track("track_north", shift(LineString([(10, 333), (341, 333)])), 4.0, "dirt"),
        Track("track_center_ns", shift(LineString([(175, 6), (175, 333)])), 5.0, "dirt"),
        Track("track_center_ew", shift(LineString([(10, 175), (341, 175)])), 5.0, "dirt"),
        Track("depot_access", shift(LineString([(37, 6), (37, 16)])), 6.0, "gravel"),
    ]
    ditches = [
        Track("ditch_west", shift(LineString([(4, -20), (4, H + 20)])), 2.4, "water"),
        Track("ditch_north", shift(LineString([(4, 338), (W + 60, 338)])), 2.0, "water"),
    ]
    weeds = [
        Zone("weed_sorghum_A", shift(_rect(80, 82, 106, 106)), "weed_exclusion",
             {"species": "Sorghum_halepense", "severity": "high"}),
        Zone("weed_cyperus_B", shift(_rect(245, 50, 270, 76)), "weed_exclusion",
             {"species": "Cyperus_rotundus", "severity": "high"}),
        Zone("weed_cynodon_C", shift(_rect(193, 240, 219, 258)), "weed_exclusion",
             {"species": "Cynodon_dactylon", "severity": "medium"}),
        Zone("weed_sorghum_D", shift(_rect(28, 248, 53, 276)), "weed_exclusion",
             {"species": "Sorghum_halepense", "severity": "medium"}),
    ]
    depot = Zone("depot_pad", shift(_rect(14, 10, 60, 22)), "depot", {"capacity_t": 120})
    stockpiles = [shift(Point(26, 16)), shift(Point(46, 16))]

    obstacles = [
        Obstacle("unk_tractor_A", "tractor", shift(_rot_rect(145.3, 62, 5.2, 2.6, 0.45)), 3.0, 0.45,
                 {"note": "broken-down tractor parked after satellite pass"}),
        Obstacle("unk_log_D", "log", shift(LineString([(80, 300), (106, 305)]).buffer(0.45, cap_style=1)), 0.9,
                 float(np.arctan2(5, 26)), {"note": "fallen poplar after storm"}),
        Obstacle("unk_pump_B", "pump", shift(_rot_rect(310.7, 130, 3.2, 2.4, 0.0)), 2.2, 0.0,
                 {"note": "temporary irrigation pump + pipe stack"}),
        Obstacle("unk_waterlog_C", "waterlogged", shift(Point(258.2, 214).buffer(5.0)), 0.05, 0.0,
                 {"note": "waterlogged soft ground (no-go for 4 t carrier)"}),
    ]

    scenery: list[SceneryItem] = []
    rng = np.random.default_rng(7)
    # Poplar row on the west bank of the irrigation ditch
    for y in np.arange(-15, H + 20, 9.0):
        scenery.append(SceneryItem("poplar", shift(Point(-2.5 + rng.normal(0, 0.4), y)),
                                   {"height_m": float(rng.uniform(14, 19))}))
    # Reeds (Arundo donax) on ditch banks
    for y in np.arange(-15, H + 15, 2.6):
        scenery.append(SceneryItem("reed", shift(Point(6.8 + rng.normal(0, 0.3), y)),
                                   {"height_m": float(rng.uniform(2.5, 4.0))}))
    for x in np.arange(8, W + 50, 2.8):
        scenery.append(SceneryItem("reed", shift(Point(x, 335.8 + rng.normal(0, 0.3))),
                                   {"height_m": float(rng.uniform(2.2, 3.6))}))
    # Olive / carob hedge beyond the north ditch
    for x in np.arange(0, W + 60, 11.0):
        kind = "olive" if rng.random() < 0.7 else "carob"
        scenery.append(SceneryItem(kind, shift(Point(x + rng.normal(0, 1.0), 345 + rng.normal(0, 1.5))),
                                   {"height_m": float(rng.uniform(5, 8))}))
    # Citrus orchard south of the road
    for x in np.arange(80, 330, 8.0):
        for y in np.arange(-14, -60, -7.0):
            scenery.append(SceneryItem("citrus", shift(Point(x + rng.normal(0, 0.3), y)),
                                       {"height_m": float(rng.uniform(2.8, 3.8))}))
    # Masia (farmhouse) + Aleppo pines NE
    scenery.append(SceneryItem("house", shift(_rect(360, 252, 386, 270)), {"height_m": 8.0}))
    scenery.append(SceneryItem("house", shift(_rect(390, 255, 402, 265)), {"height_m": 5.0}))
    for _ in range(14):
        scenery.append(SceneryItem("pine", shift(Point(rng.uniform(352, 420), rng.uniform(280, 345))),
                                   {"height_m": float(rng.uniform(9, 15))}))
    # Plastic greenhouse tunnels east of the block
    for i, y0 in enumerate(np.arange(20, 230, 11.0)):
        scenery.append(SceneryItem("greenhouse", shift(_rect(356, y0, 420, y0 + 8.0)), {"height_m": 3.4}))
    # Power poles along the road
    for x in np.arange(-100, W + 130, 38.0):
        scenery.append(SceneryItem("pole", shift(Point(x, 1.0)), {"height_m": 9.0}))

    return Site(
        origin_e=origin_e,
        origin_n=origin_n,
        width_m=width_m,
        height_m=height_m,
        parcels=parcels,
        tracks=tracks,
        ditches=ditches,
        weeds=weeds,
        depot=depot,
        stockpiles=stockpiles,
        obstacles=obstacles,
        scenery=scenery,
    )


# ---------------------------------------------------------------------------
# GeoJSON IO
# ---------------------------------------------------------------------------


def _map_coords(geom, fn):
    return shapely.transform(geom, lambda xy: np.array([fn(x, y) for x, y in xy]).reshape(-1, 2))


def _to_wgs(geom):
    return _map_coords(geom, utm31n_to_wgs84)


def _to_utm(geom):
    return _map_coords(geom, wgs84_to_utm31n)


def _feature(geom, props: dict) -> dict:
    return {"type": "Feature", "properties": props, "geometry": mapping(_to_wgs(geom))}


def _collection(name: str, feats: list[dict], meta: dict | None = None) -> dict:
    out = {
        "type": "FeatureCollection",
        "name": name,
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "features": feats,
    }
    if meta:
        out["prunemaxxing"] = meta
    return out


def write_site_geojson(site: Site, data_dir: Path = DATA_DIR) -> dict[str, Path]:
    data_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "origin_e": site.origin_e,
        "origin_n": site.origin_n,
        "width_m": site.width_m,
        "height_m": site.height_m,
        "epsg": 32631,
    }
    known: list[dict] = []
    for p in site.parcels:
        known.append(_feature(p.polygon, {"id": p.pid, "kind": "parcel", "crop": p.crop,
                                          "bed_spacing_m": p.bed_spacing_m, "bed_azimuth_deg": 0.0}))
    for t in site.tracks:
        known.append(_feature(t.line, {"id": t.tid, "kind": "track", "width_m": t.width_m,
                                       "surface": t.surface}))
    for d in site.ditches:
        known.append(_feature(d.line, {"id": d.tid, "kind": "ditch", "width_m": d.width_m}))
    for w in site.weeds:
        known.append(_feature(w.polygon, {"id": w.zid, "kind": "weed_exclusion", **w.props}))
    known.append(_feature(site.depot.polygon, {"id": site.depot.zid, "kind": "depot", **site.depot.props}))
    for i, sp in enumerate(site.stockpiles):
        known.append(_feature(sp, {"id": f"stockpile_{i}", "kind": "stockpile", "capacity_t": 60}))

    hidden = [
        _feature(o.polygon, {"id": o.oid, "kind": "unknown_obstacle", "obstacle_type": o.kind,
                             "height_m": o.height_m, "yaw_rad": o.yaw_rad, **o.props})
        for o in site.obstacles
    ]
    scen = [_feature(s.geometry, {"kind": s.kind, **s.props}) for s in site.scenery]

    paths = {
        "known": data_dir / SITE_KNOWN_PATH.name,
        "unknown": data_dir / UNKNOWN_OBSTACLES_PATH.name,
        "scenery": data_dir / SCENERY_PATH.name,
    }
    paths["known"].write_text(json.dumps(_collection("prunemaxxing_site_known", known, meta), indent=1))
    paths["unknown"].write_text(json.dumps(_collection("prunemaxxing_unknown_obstacles", hidden, meta), indent=1))
    paths["scenery"].write_text(json.dumps(_collection("prunemaxxing_scenery", scen, meta)))
    return paths


def load_site(
    known_path: Path = SITE_KNOWN_PATH,
    unknown_path: Path | None = UNKNOWN_OBSTACLES_PATH,
    scenery_path: Path | None = SCENERY_PATH,
) -> Site:
    data = json.loads(Path(known_path).read_text())
    meta = data.get("prunemaxxing", {})
    parcels, tracks, ditches, weeds, stockpiles = [], [], [], [], []
    depot = None
    for f in data["features"]:
        props = f["properties"]
        geom = _to_utm(shape(f["geometry"]))
        kind = props.get("kind")
        if kind == "parcel":
            parcels.append(Parcel(props["id"], geom, props.get("crop", ""), float(props.get("bed_spacing_m", BED_SPACING_M))))
        elif kind == "track":
            tracks.append(Track(props["id"], geom, float(props.get("width_m", 4.0)), props.get("surface", "dirt")))
        elif kind == "ditch":
            ditches.append(Track(props["id"], geom, float(props.get("width_m", 2.0)), "water"))
        elif kind == "weed_exclusion":
            extra = {k: v for k, v in props.items() if k not in ("id", "kind")}
            weeds.append(Zone(props["id"], geom, kind, extra))
        elif kind == "depot":
            depot = Zone(props["id"], geom, kind, {k: v for k, v in props.items() if k not in ("id", "kind")})
        elif kind == "stockpile":
            stockpiles.append(geom)

    obstacles: list[Obstacle] = []
    if unknown_path is not None and Path(unknown_path).exists():
        for f in json.loads(Path(unknown_path).read_text())["features"]:
            props = f["properties"]
            extra = {k: v for k, v in props.items() if k not in ("id", "kind", "obstacle_type", "height_m", "yaw_rad")}
            obstacles.append(Obstacle(props["id"], props["obstacle_type"], _to_utm(shape(f["geometry"])),
                                      float(props.get("height_m", 1.0)), float(props.get("yaw_rad", 0.0)), extra))

    scenery: list[SceneryItem] = []
    if scenery_path is not None and Path(scenery_path).exists():
        for f in json.loads(Path(scenery_path).read_text())["features"]:
            props = dict(f["properties"])
            kind = props.pop("kind")
            scenery.append(SceneryItem(kind, _to_utm(shape(f["geometry"])), props))

    if depot is None:
        raise ValueError("site_known.geojson has no depot feature")
    return Site(
        origin_e=float(meta["origin_e"]),
        origin_n=float(meta["origin_n"]),
        width_m=float(meta["width_m"]),
        height_m=float(meta["height_m"]),
        parcels=parcels,
        tracks=tracks,
        ditches=ditches,
        weeds=weeds,
        depot=depot,
        stockpiles=stockpiles,
        obstacles=obstacles,
        scenery=scenery,
    )
