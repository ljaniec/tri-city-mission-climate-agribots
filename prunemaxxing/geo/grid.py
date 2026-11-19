"""Cell-level Digital Twin for Parc Agrari application parcels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from prunemaxxing.config import (
    C_SEQUESTERED_KG_PER_M2,
    CO2E_T_PER_HA,
    DEFAULT_TARGET_KG_PER_M2,
    MAX_MOTHERSHIP_SLOPE_DEG,
    N_RETAINED_KG_PER_HA,
    TARGET_FIELD_WGS84,
    WEED_EXCLUSION_SPECIES,
)
from prunemaxxing.geo.crs import wgs84_bbox_to_utm31n
from prunemaxxing.geo.dem import DEMGrid


@dataclass
class CellState:
    row: int
    col: int
    easting: float
    northing: float
    elevation_m: float
    slope_deg: float
    cell_area_m2: float
    target_biomass_kg: float
    deposited_biomass_kg: float = 0.0
    soil_organic_carbon_est: float = 0.0  # relative increase (fraction of potential)
    weed_exclusion_flag: bool = False
    weed_species: tuple[str, ...] = ()
    no_till_bed_status: bool = False  # True once beds formed / no-till locked
    traversable: bool = True

    @property
    def completion_ratio(self) -> float:
        if self.target_biomass_kg <= 0:
            return 1.0
        return min(1.0, self.deposited_biomass_kg / self.target_biomass_kg)

    @property
    def remaining_kg(self) -> float:
        return max(0.0, self.target_biomass_kg - self.deposited_biomass_kg)

    @property
    def is_complete(self) -> bool:
        return self.completion_ratio >= 0.99 or self.weed_exclusion_flag

    def deposit(self, kg: float) -> float:
        """Deposit biomass; returns actually accepted kg."""
        if self.weed_exclusion_flag or not self.traversable:
            return 0.0
        accepted = min(kg, self.remaining_kg)
        self.deposited_biomass_kg += accepted
        # Linear SOC proxy toward REGENSOL fine-earth sequestration
        self.soil_organic_carbon_est = self.completion_ratio * C_SEQUESTERED_KG_PER_M2
        return accepted


@dataclass
class FieldTwin:
    """Grid twin over a DEM crop; one cell per DEM pixel by default."""

    dem: DEMGrid
    cells: list[list[CellState]]
    target_kg_per_m2: float = DEFAULT_TARGET_KG_PER_M2
    max_slope_deg: float = MAX_MOTHERSHIP_SLOPE_DEG
    weed_mask: np.ndarray | None = None

    @property
    def nrows(self) -> int:
        return len(self.cells)

    @property
    def ncols(self) -> int:
        return len(self.cells[0]) if self.cells else 0

    def iter_cells(self) -> Iterable[CellState]:
        for row in self.cells:
            yield from row

    def get(self, row: int, col: int) -> CellState:
        return self.cells[row][col]

    def work_cells(self) -> list[CellState]:
        return [
            c
            for c in self.iter_cells()
            if c.traversable and not c.weed_exclusion_flag and not c.is_complete
        ]

    def apply_weed_patches(
        self, centers: list[tuple[float, float]], radius_m: float, species: str
    ) -> int:
        """Mark circular weed exclusion zones (easting, northing centers)."""
        if species not in WEED_EXCLUSION_SPECIES:
            raise ValueError(f"Unknown weed species {species}")
        flagged = 0
        for cell in self.iter_cells():
            for e, n in centers:
                if (cell.easting - e) ** 2 + (cell.northing - n) ** 2 <= radius_m**2:
                    if not cell.weed_exclusion_flag:
                        flagged += 1
                    cell.weed_exclusion_flag = True
                    cell.weed_species = tuple(sorted(set(cell.weed_species) | {species}))
                    cell.traversable = False
        return flagged

    def lock_no_till_beds(self) -> None:
        for cell in self.iter_cells():
            if cell.completion_ratio >= 0.99 and not cell.weed_exclusion_flag:
                cell.no_till_bed_status = True

    def coverage_uniformity(self) -> float:
        """1 - CV of completion ratios over eligible cells."""
        ratios = np.array(
            [
                c.completion_ratio
                for c in self.iter_cells()
                if not c.weed_exclusion_flag and c.traversable
            ],
            dtype=float,
        )
        if ratios.size == 0:
            return 1.0
        mean = float(ratios.mean())
        if mean < 1e-9:
            return 0.0
        cv = float(ratios.std() / mean)
        return float(max(0.0, 1.0 - cv))

    def total_deposited_t(self) -> float:
        return sum(c.deposited_biomass_kg for c in self.iter_cells()) / 1000.0

    def eligible_area_m2(self) -> float:
        return sum(
            c.cell_area_m2
            for c in self.iter_cells()
            if c.traversable and not c.weed_exclusion_flag
        )

    def estimated_co2e_t(self) -> float:
        area_ha = self.eligible_area_m2() / 10_000.0
        mean_completion = np.mean(
            [
                c.completion_ratio
                for c in self.iter_cells()
                if c.traversable and not c.weed_exclusion_flag
            ]
            or [0.0]
        )
        return float(mean_completion * CO2E_T_PER_HA * area_ha)

    def estimated_n_retained_kg(self) -> float:
        area_ha = self.eligible_area_m2() / 10_000.0
        mean_completion = np.mean(
            [
                c.completion_ratio
                for c in self.iter_cells()
                if c.traversable and not c.weed_exclusion_flag
            ]
            or [0.0]
        )
        return float(mean_completion * N_RETAINED_KG_PER_HA * area_ha)


def build_field_twin_from_dem(
    dem: DEMGrid,
    target_kg_per_m2: float = DEFAULT_TARGET_KG_PER_M2,
    max_slope_deg: float = MAX_MOTHERSHIP_SLOPE_DEG,
) -> FieldTwin:
    slopes = dem.slope_deg()
    cells: list[list[CellState]] = []
    area = dem.pixel_size_x * dem.pixel_size_y
    for r in range(dem.height):
        row_cells: list[CellState] = []
        for c in range(dem.width):
            elev = float(dem.elevation[r, c])
            slope = float(slopes[r, c])
            traversable = slope <= max_slope_deg and np.isfinite(elev)
            row_cells.append(
                CellState(
                    row=r,
                    col=c,
                    easting=dem.easting(c + 0.5),
                    northing=dem.northing(r + 0.5),
                    elevation_m=elev,
                    slope_deg=slope,
                    cell_area_m2=area,
                    target_biomass_kg=target_kg_per_m2 * area,
                    traversable=traversable,
                )
            )
        cells.append(row_cells)
    return FieldTwin(
        dem=dem,
        cells=cells,
        target_kg_per_m2=target_kg_per_m2,
        max_slope_deg=max_slope_deg,
    )


def crop_target_field(dem: DEMGrid) -> DEMGrid:
    """Crop DEM to the AGR-E2 specific target field WGS84 box."""
    e0, n0, e1, n1 = wgs84_bbox_to_utm31n(**TARGET_FIELD_WGS84)
    return dem.crop_utm(e0, n0, e1, n1)


def select_pilot_window(
    dem: DEMGrid,
    area_ha: float = 6.0,
    anchor: str = "sw",
) -> DEMGrid:
    """
    Select a rectangular ~area_ha window inside a DEM crop.

    Default: southwest corner of the target field (flattest orchard/horticulture edge).
    """
    cell_area = dem.pixel_size_x * dem.pixel_size_y
    n_cells = max(1, int(round(area_ha * 10_000.0 / cell_area)))
    # Prefer near-square footprint
    ncols = int(round(np.sqrt(n_cells * dem.pixel_size_y / dem.pixel_size_x)))
    nrows = int(round(n_cells / max(ncols, 1)))
    ncols = min(ncols, dem.width)
    nrows = min(nrows, dem.height)

    if anchor == "sw":
        r0, c0 = dem.height - nrows, 0
    elif anchor == "center":
        r0 = (dem.height - nrows) // 2
        c0 = (dem.width - ncols) // 2
    else:
        r0, c0 = 0, 0

    elev = dem.elevation[r0 : r0 + nrows, c0 : c0 + ncols].copy()
    return DEMGrid(
        elevation=elev,
        origin_easting=dem.easting(c0),
        origin_northing=dem.northing(r0),
        pixel_size_x=dem.pixel_size_x,
        pixel_size_y=dem.pixel_size_y,
        epsg=dem.epsg,
    )
