"""Shared 1:1 world frame for MuJoCo scenes, replay and physics validation.

World frame (metres):
  x = easting  - center_e      (east)
  y = northing - center_n      (north)
  z = absolute DEM elevation   (metres above sea level, no exaggeration)

``center_e/center_n`` is the centre of the pilot DEM window (``Site`` bounds).
``Terrain`` is a smooth bicubic upsample of the 10 m DEM over the window plus a
margin, used both to bake the MuJoCo heightfield and to place vehicles on it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

from prunemaxxing.config import DEFAULT_DEM_PATH
from prunemaxxing.geo.dem import DEMGrid, load_dem

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class WorldFrame:
    center_e: float
    center_n: float

    def to_world(self, e, n):
        return np.asarray(e) - self.center_e, np.asarray(n) - self.center_n

    def to_utm(self, x, y):
        return np.asarray(x) + self.center_e, np.asarray(y) + self.center_n

    def xy(self, en: np.ndarray) -> np.ndarray:
        en = np.asarray(en, dtype=float)
        return en - np.array([self.center_e, self.center_n])


def frame_for_site(site) -> WorldFrame:
    return WorldFrame(site.origin_e + site.width_m / 2.0, site.origin_n + site.height_m / 2.0)


@dataclass
class Terrain:
    """Regular grid of absolute elevations in world coordinates."""

    x0: float  # world x of column 0
    y0: float  # world y of row 0 (south edge; rows increase northwards)
    res: float
    heights: np.ndarray  # (ny, nx) float64

    @property
    def nx(self) -> int:
        return int(self.heights.shape[1])

    @property
    def ny(self) -> int:
        return int(self.heights.shape[0])

    @property
    def extent(self) -> tuple[float, float, float, float]:
        return (self.x0, self.x0 + (self.nx - 1) * self.res, self.y0, self.y0 + (self.ny - 1) * self.res)

    def height_at(self, x, y):
        """Bilinear terrain height (scalar or array) in world coordinates."""
        xa = np.asarray(x, dtype=float)
        ya = np.asarray(y, dtype=float)
        cx = np.clip((xa - self.x0) / self.res, 0, self.nx - 1.000001)
        cy = np.clip((ya - self.y0) / self.res, 0, self.ny - 1.000001)
        out = map_coordinates(self.heights, [np.atleast_1d(cy), np.atleast_1d(cx)], order=1, mode="nearest")
        return float(out[0]) if np.ndim(x) == 0 else out.reshape(xa.shape)

    def normal_at(self, x: float, y: float, h: float = 1.0) -> np.ndarray:
        dzdx = (self.height_at(x + h, y) - self.height_at(x - h, y)) / (2 * h)
        dzdy = (self.height_at(x, y + h) - self.height_at(x, y - h)) / (2 * h)
        n = np.array([-dzdx, -dzdy, 1.0])
        return n / np.linalg.norm(n)


def load_terrain(
    frame: WorldFrame,
    half_x: float,
    half_y: float,
    res_m: float = 1.0,
    dem: DEMGrid | None = None,
    smooth_sigma_m: float = 6.0,
) -> Terrain:
    """Bicubic upsample of the full DEM covering [-half_x, half_x] x [-half_y, half_y]."""
    if dem is None:
        dem = load_dem(ROOT / DEFAULT_DEM_PATH)
    nx = int(round(2 * half_x / res_m)) + 1
    ny = int(round(2 * half_y / res_m)) + 1
    xs = -half_x + np.arange(nx) * res_m
    ys = -half_y + np.arange(ny) * res_m
    gx, gy = np.meshgrid(xs, ys)
    e = gx + frame.center_e
    n = gy + frame.center_n
    # pixel-centre fractional indices (PixelIsArea tiepoint)
    col = (e - dem.origin_easting) / dem.pixel_size_x - 0.5
    row = (dem.origin_northing - n) / dem.pixel_size_y - 0.5
    z = dem.elevation.astype(np.float64)
    z = np.where(np.isfinite(z), z, np.nanmedian(z))
    h = map_coordinates(z, [row, col], order=3, mode="nearest")
    if smooth_sigma_m > 0:
        h = gaussian_filter(h, smooth_sigma_m / res_m, mode="nearest")
    return Terrain(x0=-half_x, y0=-half_y, res=res_m, heights=h)
