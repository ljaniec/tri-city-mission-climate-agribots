"""Load parcel / weed / road GeoJSON and apply to the FieldTwin."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from prunemaxxing.config import WEED_EXCLUSION_SPECIES
from prunemaxxing.geo.crs import wgs84_to_utm31n
from prunemaxxing.geo.grid import FieldTwin


@dataclass
class PolygonFeature:
    feature_id: str
    kind: str
    props: dict[str, Any]
    rings_utm: list[list[tuple[float, float]]]  # exterior (+ holes) in UTM metres


@dataclass
class MapLayers:
    parcels: list[PolygonFeature] = field(default_factory=list)
    weeds: list[PolygonFeature] = field(default_factory=list)
    roads: list[PolygonFeature] = field(default_factory=list)
    stockpiles: list[tuple[float, float, dict]] = field(default_factory=list)

    @property
    def all_polygons(self) -> list[PolygonFeature]:
        return self.parcels + self.weeds


def _ring_wgs84_to_utm(coords: list[list[float]]) -> list[tuple[float, float]]:
    return [wgs84_to_utm31n(float(lon), float(lat)) for lon, lat in coords]


def load_map_layers(path: str | Path) -> MapLayers:
    data = json.loads(Path(path).read_text())
    layers = MapLayers()
    for feat in data.get("features", []):
        props = feat.get("properties") or {}
        geom = feat.get("geometry") or {}
        gtype = geom.get("type")
        fid = str(props.get("id", "unnamed"))
        kind = str(props.get("kind", "unknown"))

        if gtype == "Polygon":
            rings = [_ring_wgs84_to_utm(ring) for ring in geom["coordinates"]]
            poly = PolygonFeature(fid, kind, props, rings)
            if kind == "weed_exclusion":
                layers.weeds.append(poly)
            elif kind == "parcel":
                layers.parcels.append(poly)
            else:
                layers.parcels.append(poly)
        elif gtype == "LineString":
            ring = _ring_wgs84_to_utm(geom["coordinates"])
            layers.roads.append(PolygonFeature(fid, kind, props, [ring]))
        elif gtype == "Point":
            lon, lat = geom["coordinates"]
            e, n = wgs84_to_utm31n(float(lon), float(lat))
            layers.stockpiles.append((e, n, props))
    return layers


def _point_in_ring(x: float, y: float, ring: list[tuple[float, float]]) -> bool:
    """Ray casting; ring is closed or open."""
    n = len(ring)
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = ring[i]
        xj, yj = ring[j]
        intersect = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) + 1e-15) + xi
        )
        if intersect:
            inside = not inside
        j = i
    return inside


def point_in_polygon(x: float, y: float, rings: list[list[tuple[float, float]]]) -> bool:
    if not rings:
        return False
    if not _point_in_ring(x, y, rings[0]):
        return False
    for hole in rings[1:]:
        if _point_in_ring(x, y, hole):
            return False
    return True


def apply_map_layers(twin: FieldTwin, layers: MapLayers) -> dict[str, int]:
    """Flag weed cells and annotate parcels; returns counts."""
    weed_cells = 0
    parcel_cells = 0
    for cell in twin.iter_cells():
        for weed in layers.weeds:
            if point_in_polygon(cell.easting, cell.northing, weed.rings_utm):
                species = str(weed.props.get("species", "Sorghum_halepense"))
                if species not in WEED_EXCLUSION_SPECIES:
                    species = WEED_EXCLUSION_SPECIES[0]
                if not cell.weed_exclusion_flag:
                    weed_cells += 1
                cell.weed_exclusion_flag = True
                cell.traversable = False
                cell.weed_species = tuple(sorted(set(cell.weed_species) | {species}))
        for parcel in layers.parcels:
            if point_in_polygon(cell.easting, cell.northing, parcel.rings_utm):
                parcel_cells += 1
    twin.weed_mask = np.array(
        [
            [1 if twin.get(r, c).weed_exclusion_flag else 0 for c in range(twin.ncols)]
            for r in range(twin.nrows)
        ],
        dtype=np.uint8,
    )
    return {"weed_cells": weed_cells, "parcel_hits": parcel_cells}
