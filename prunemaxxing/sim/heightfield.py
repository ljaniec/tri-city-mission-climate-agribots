"""Build a MuJoCo heightfield PNG from the 6 ha pilot DEM crop."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import zoom

from prunemaxxing.config import DEFAULT_DEM_PATH, PILOT_AREA_HA
from prunemaxxing.geo.dem import DEMGrid, load_dem
from prunemaxxing.geo.grid import crop_target_field, select_pilot_window

ASSETS_DIR = Path(__file__).with_name("assets")
HFIELD_PNG = ASSETS_DIR / "pilot_dem_hfield.png"
HFIELD_META = ASSETS_DIR / "pilot_dem_hfield.json"

# Scene half-extents (metres) — matches prior 30×30 plane arena scale.
DEFAULT_HALF_X = 28.0
DEFAULT_HALF_Y = 28.0
# MuJoCo scales normalized PNG elevations by elevation_z; exaggerate flat DEM.
DEFAULT_ELEVATION_Z = 2.4
DEFAULT_BASE_Z = 0.2
DEFAULT_UPSAMPLE = 8.0  # 25×24 @10 m → ~200×192 grid
DEFAULT_BED_AMPLITUDE = 0.06  # fraction of [0,1] after DEM normalize


@dataclass(frozen=True)
class HeightfieldSpec:
    """Parameters written beside the PNG for XML / load_model."""

    png_path: str
    nrow: int
    ncol: int
    half_x: float
    half_y: float
    elevation_z: float
    base_z: float
    z_offset: float  # place geom at (0,0,z_offset) so mean surface ≈ world z=0
    elev_min_m: float
    elev_max_m: float
    dem_width_m: float
    dem_height_m: float
    upsample: float
    bake_beds: bool

    @property
    def size_attr(self) -> str:
        return f"{self.half_x:.4g} {self.half_y:.4g} {self.elevation_z:.4g} {self.base_z:.4g}"


def _repo_dem_path(dem_path: str | Path | None = None) -> Path:
    if dem_path is not None:
        return Path(dem_path)
    root = Path(__file__).resolve().parents[2]
    return root / DEFAULT_DEM_PATH


def load_pilot_dem(
    dem_path: str | Path | None = None,
    area_ha: float = PILOT_AREA_HA,
    anchor: str = "sw",
) -> DEMGrid:
    dem = load_dem(_repo_dem_path(dem_path))
    return select_pilot_window(crop_target_field(dem), area_ha=area_ha, anchor=anchor)


def _normalize_elev(elev: np.ndarray) -> tuple[np.ndarray, float, float]:
    finite = elev[np.isfinite(elev)]
    if finite.size == 0:
        raise ValueError("DEM crop has no finite elevations")
    z0 = float(finite.min())
    z1 = float(finite.max())
    span = max(z1 - z0, 1e-6)
    out = np.nan_to_num((elev - z0) / span, nan=0.0).astype(np.float64)
    return np.clip(out, 0.0, 1.0), z0, z1


def _bake_bed_furrows(
    grid: np.ndarray,
    amplitude: float = DEFAULT_BED_AMPLITUDE,
    spacing_cols: float = 6.0,
) -> np.ndarray:
    """Subtle E–W bed ridges / furrow troughs (visual only, fraction of [0,1])."""
    nrows, ncols = grid.shape
    xs = np.linspace(0.0, ncols / spacing_cols * np.pi, ncols, dtype=np.float64)
    # Ridges high, furrows low — small amplitude so DEM shape still dominates.
    ripple = amplitude * (0.5 + 0.5 * np.sin(xs))
    return np.clip(grid + ripple[np.newaxis, :], 0.0, 1.0)


def build_height_grid(
    dem: DEMGrid,
    upsample: float = DEFAULT_UPSAMPLE,
    bake_beds: bool = True,
    bed_amplitude: float = DEFAULT_BED_AMPLITUDE,
) -> tuple[np.ndarray, float, float]:
    """
    Return (nrow, ncol) float grid in [0, 1], plus raw elev min/max metres.

    Row 0 is north (DEM north-up). MuJoCo PNG row 0 is also the +Y edge of the
    heightfield when the geom has identity orientation, matching a north=+Y scene.
    """
    elev = dem.elevation.astype(np.float64)
    norm, z0, z1 = _normalize_elev(elev)
    if upsample and upsample != 1.0:
        norm = zoom(norm, zoom=float(upsample), order=3)
        norm = np.clip(norm, 0.0, 1.0)
    if bake_beds:
        norm = _bake_bed_furrows(norm, amplitude=bed_amplitude)
    return norm.astype(np.float32), z0, z1


def write_heightfield_png(
    grid01: np.ndarray,
    path: Path,
) -> Path:
    """Write 8-bit grayscale PNG (white = high) for MuJoCo hfield assets."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    u8 = (np.clip(grid01, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(u8, mode="L").save(path)
    return path


def generate_heightfield(
    dem_path: str | Path | None = None,
    out_png: Path | None = None,
    out_meta: Path | None = None,
    area_ha: float = PILOT_AREA_HA,
    half_x: float = DEFAULT_HALF_X,
    half_y: float | None = None,
    elevation_z: float = DEFAULT_ELEVATION_Z,
    base_z: float = DEFAULT_BASE_Z,
    upsample: float = DEFAULT_UPSAMPLE,
    bake_beds: bool = True,
) -> HeightfieldSpec:
    """Crop pilot DEM → upsampled PNG + JSON sidecar for MuJoCo."""
    dem = load_pilot_dem(dem_path, area_ha=area_ha)
    grid, z0, z1 = build_height_grid(dem, upsample=upsample, bake_beds=bake_beds)
    dem_w = dem.width * dem.pixel_size_x
    dem_h = dem.height * dem.pixel_size_y
    if half_y is None:
        # Preserve DEM aspect inside the arena half-extents.
        aspect = dem_h / max(dem_w, 1e-9)
        half_y = float(half_x * aspect)

    # Mean surface at world z≈0 when geom pos z = -elevation_z * mean(grid).
    z_offset = -float(elevation_z) * float(grid.mean())

    png = Path(out_png) if out_png else HFIELD_PNG
    meta_path = Path(out_meta) if out_meta else HFIELD_META
    write_heightfield_png(grid, png)

    spec = HeightfieldSpec(
        png_path=str(png.relative_to(Path(__file__).parent))
        if png.is_relative_to(Path(__file__).parent)
        else str(png),
        nrow=int(grid.shape[0]),
        ncol=int(grid.shape[1]),
        half_x=float(half_x),
        half_y=float(half_y),
        elevation_z=float(elevation_z),
        base_z=float(base_z),
        z_offset=float(z_offset),
        elev_min_m=float(z0),
        elev_max_m=float(z1),
        dem_width_m=float(dem_w),
        dem_height_m=float(dem_h),
        upsample=float(upsample),
        bake_beds=bool(bake_beds),
    )
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(asdict(spec), indent=2) + "\n")
    return spec


def ensure_heightfield(
    force: bool = False,
    dem_path: str | Path | None = None,
    **kwargs,
) -> HeightfieldSpec:
    """Return existing sidecar+PNG, or regenerate when missing / forced."""
    if not force and HFIELD_PNG.exists() and HFIELD_META.exists():
        data = json.loads(HFIELD_META.read_text())
        return HeightfieldSpec(**data)
    return generate_heightfield(dem_path=dem_path, **kwargs)


def load_heightfield_spec() -> HeightfieldSpec:
    return ensure_heightfield(force=False)
