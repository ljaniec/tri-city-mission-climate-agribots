"""WGS84 ↔ UTM zone 31N (EPSG:32631) without pyproj dependency."""

from __future__ import annotations

import math

_A = 6378137.0
_F = 1 / 298.257223563
_K0 = 0.9996
_E2 = _F * (2 - _F)
_EP2 = _E2 / (1 - _E2)
_ZONE = 31
_LON0 = math.radians((_ZONE - 1) * 6 - 180 + 3)


def wgs84_to_utm31n(lon: float, lat: float) -> tuple[float, float]:
    """Return (easting, northing) metres in UTM 31N."""
    lat_r = math.radians(lat)
    lon_r = math.radians(lon)
    n = _A / math.sqrt(1 - _E2 * math.sin(lat_r) ** 2)
    t = math.tan(lat_r) ** 2
    c = _EP2 * math.cos(lat_r) ** 2
    a = (lon_r - _LON0) * math.cos(lat_r)
    m = _A * (
        (1 - _E2 / 4 - 3 * _E2**2 / 64 - 5 * _E2**3 / 256) * lat_r
        - (3 * _E2 / 8 + 3 * _E2**2 / 32 + 45 * _E2**3 / 1024) * math.sin(2 * lat_r)
        + (15 * _E2**2 / 256 + 45 * _E2**3 / 1024) * math.sin(4 * lat_r)
        - (35 * _E2**3 / 3072) * math.sin(6 * lat_r)
    )
    x = (
        _K0
        * n
        * (
            a
            + (1 - t + c) * a**3 / 6
            + (5 - 18 * t + t**2 + 72 * c - 58 * _EP2) * a**5 / 120
        )
        + 500_000.0
    )
    y = _K0 * (
        m
        + n
        * math.tan(lat_r)
        * (
            a**2 / 2
            + (5 - t + 9 * c + 4 * c**2) * a**4 / 24
            + (61 - 58 * t + t**2 + 600 * c - 330 * _EP2) * a**6 / 720
        )
    )
    return x, y


def utm31n_to_wgs84(easting: float, northing: float) -> tuple[float, float]:
    """Return (lon, lat) degrees from UTM 31N easting/northing."""
    x = easting - 500_000.0
    e1 = (1 - math.sqrt(1 - _E2)) / (1 + math.sqrt(1 - _E2))
    m = northing / _K0
    mu = m / (_A * (1 - _E2 / 4 - 3 * _E2**2 / 64 - 5 * _E2**3 / 256))
    phi1 = (
        mu
        + (3 * e1 / 2 - 27 * e1**3 / 32) * math.sin(2 * mu)
        + (21 * e1**2 / 16 - 55 * e1**4 / 32) * math.sin(4 * mu)
        + (151 * e1**3 / 96) * math.sin(6 * mu)
        + (1097 * e1**4 / 512) * math.sin(8 * mu)
    )
    n1 = _A / math.sqrt(1 - _E2 * math.sin(phi1) ** 2)
    t1 = math.tan(phi1) ** 2
    c1 = _EP2 * math.cos(phi1) ** 2
    r1 = _A * (1 - _E2) / (1 - _E2 * math.sin(phi1) ** 2) ** 1.5
    d = x / (n1 * _K0)
    lat = phi1 - (n1 * math.tan(phi1) / r1) * (
        d**2 / 2
        - (5 + 3 * t1 + 10 * c1 - 4 * c1**2 - 9 * _EP2) * d**4 / 24
        + (61 + 90 * t1 + 298 * c1 + 45 * t1**2 - 252 * _EP2 - 3 * c1**2) * d**6 / 720
    )
    lon = _LON0 + (
        d
        - (1 + 2 * t1 + c1) * d**3 / 6
        + (5 - 2 * c1 + 28 * t1 - 3 * c1**2 + 8 * _EP2 + 24 * t1**2) * d**5 / 120
    ) / math.cos(phi1)
    return math.degrees(lon), math.degrees(lat)


def wgs84_bbox_to_utm31n(
    lon_min: float, lat_min: float, lon_max: float, lat_max: float
) -> tuple[float, float, float, float]:
    """Axis-aligned UTM bbox covering the WGS84 rectangle corners."""
    corners = [
        wgs84_to_utm31n(lon_min, lat_min),
        wgs84_to_utm31n(lon_max, lat_min),
        wgs84_to_utm31n(lon_min, lat_max),
        wgs84_to_utm31n(lon_max, lat_max),
    ]
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    return min(xs), min(ys), max(xs), max(ys)
