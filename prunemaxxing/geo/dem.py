"""GeoTIFF DEM loader for viladecans_full_dem_10m.TIF (UTM 31N)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, TiffTags


@dataclass
class DEMGrid:
    """North-up DEM in UTM 31N metres."""

    elevation: np.ndarray  # (rows, cols) float32
    origin_easting: float  # upper-left pixel center/corner (tiepoint)
    origin_northing: float
    pixel_size_x: float
    pixel_size_y: float  # positive ground size; row increases south
    epsg: int = 32631

    @property
    def height(self) -> int:
        return int(self.elevation.shape[0])

    @property
    def width(self) -> int:
        return int(self.elevation.shape[1])

    def easting(self, col: float) -> float:
        return self.origin_easting + col * self.pixel_size_x

    def northing(self, row: float) -> float:
        return self.origin_northing - row * self.pixel_size_y

    def col(self, easting: float) -> float:
        return (easting - self.origin_easting) / self.pixel_size_x

    def row(self, northing: float) -> float:
        return (self.origin_northing - northing) / self.pixel_size_y

    def sample(self, easting: float, northing: float) -> float:
        c = int(round(self.col(easting)))
        r = int(round(self.row(northing)))
        if r < 0 or c < 0 or r >= self.height or c >= self.width:
            return float("nan")
        return float(self.elevation[r, c])

    def crop_utm(
        self, east_min: float, north_min: float, east_max: float, north_max: float
    ) -> "DEMGrid":
        c0 = max(0, int(math.floor(self.col(east_min))))
        c1 = min(self.width, int(math.ceil(self.col(east_max))))
        r0 = max(0, int(math.floor(self.row(north_max))))
        r1 = min(self.height, int(math.ceil(self.row(north_min))))
        if c1 <= c0 or r1 <= r0:
            raise ValueError("Crop window empty or outside DEM extent")
        elev = self.elevation[r0:r1, c0:c1].copy()
        return DEMGrid(
            elevation=elev,
            origin_easting=self.easting(c0),
            origin_northing=self.northing(r0),
            pixel_size_x=self.pixel_size_x,
            pixel_size_y=self.pixel_size_y,
            epsg=self.epsg,
        )

    def slope_deg(self) -> np.ndarray:
        """Finite-difference slope in degrees for each cell."""
        z = self.elevation.astype(np.float64)
        dzdx = np.zeros_like(z)
        dzdy = np.zeros_like(z)
        dzdx[:, 1:-1] = (z[:, 2:] - z[:, :-2]) / (2.0 * self.pixel_size_x)
        dzdy[1:-1, :] = (z[:-2, :] - z[2:, :]) / (2.0 * self.pixel_size_y)
        return np.degrees(np.arctan(np.hypot(dzdx, dzdy)))


def load_dem(path: str | Path) -> DEMGrid:
    """Load GeoTIFF DEM via Pillow (no GDAL required)."""
    path = Path(path)
    im = Image.open(path)
    if im.mode != "F":
        raise ValueError(f"Expected float DEM (mode F), got {im.mode}")

    tags = im.tag_v2
    scale = tags.get(33550)  # ModelPixelScaleTag
    tie = tags.get(33922)  # ModelTiepointTag
    if scale is None or tie is None:
        raise ValueError("DEM missing ModelPixelScale / ModelTiepoint GeoTIFF tags")

    pixel_size_x = float(scale[0])
    pixel_size_y = float(scale[1])
    origin_easting = float(tie[3])
    origin_northing = float(tie[4])

    elevation = np.array(im, dtype=np.float32)
    return DEMGrid(
        elevation=elevation,
        origin_easting=origin_easting,
        origin_northing=origin_northing,
        pixel_size_x=pixel_size_x,
        pixel_size_y=pixel_size_y,
    )


def describe_geotiff_tags(path: str | Path) -> dict:
    """Debug helper for GeoTIFF metadata."""
    im = Image.open(path)
    out = {}
    for k, v in im.tag_v2.items():
        out[TiffTags.TAGS.get(k, str(k))] = v
    return out
