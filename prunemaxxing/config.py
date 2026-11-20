"""Domain constants from AGR-E2 / REGENSOL (enforce in twin + planners)."""

from __future__ import annotations

from dataclasses import dataclass


# --- Geospatial bounds (WGS84) ---
FULL_ZONE_WGS84 = dict(lon_min=2.015, lat_min=41.275, lon_max=2.055, lat_max=41.305)
TARGET_FIELD_WGS84 = dict(
    lon_min=2.0255, lat_min=41.2825, lon_max=2.0355, lat_max=41.2898
)

DEM_EPSG = 32631  # WGS 84 / UTM zone 31N
DEFAULT_DEM_PATH = "viladecans_full_dem_10m.TIF"

# --- Biomass / material ---
PILOT_AREA_HA = 6.0
TARGET_BIOMASS_T_PER_HA = (150.0, 200.0)  # AGR-E2 pilot
REGENSOL_BASELINE_T_PER_HA = (100.0, 150.0)  # trials
TARGET_KG_PER_M2 = (10.0, 20.0)  # 100–200 t/ha
DEFAULT_TARGET_KG_PER_M2 = 15.0  # mid of pilot band (150 t/ha)
BULK_DENSITY_KG_PER_M3 = 250.0
CHIP_MAX_LENGTH_CM = 5.0
CHIP_MAX_DIAMETER_CM = 3.0
CN_RATIO_TARGET = 40.0
CN_RATIO_RANGE = (36.0, 73.0)

# Local annual supply (t/year) — sourcing deficit for 900–1200 t pilot
MUNICIPAL_CHIP_SUPPLY_T_YEAR = 53.0
FOREST_BIOMASS_SUPPLY_T_YEAR = 220.0

# --- Field / soil ---
INCORPORATION_DEPTH_CM = 20.0
BED_ROW_SPACING_M = (0.70, 1.0)
PLANT_SPACING_M = (0.25, 0.40)
WEED_EXCLUSION_SPECIES = (
    "Sorghum_halepense",
    "Cyperus_rotundus",
    "Cynodon_dactylon",
)
COMPACTION_PENALTY_EUR_PER_1000M2 = (50.0, 100.0)
MAX_MOTHERSHIP_SLOPE_DEG = 8.0  # soft limit on flat Baix Llobregat plain

# --- Economic baselines (€ / 1000 m²) ---
CONVENTIONAL_COST_EUR_PER_1000M2 = 1895.56
REGENSOL_COST_EUR_PER_1000M2 = 2408.77
REGENSOL_OVERHEAD_EUR_PER_1000M2 = 513.21
TRANSPORT_SPREAD_SHARE = 0.277
MANUAL_WEEDING_HOURS_PER_1000M2 = 25.0

# --- Carbon / nutrient ---
SOC_INCREASE_PP = 0.45
C_SEQUESTERED_KG_PER_M2 = 1.2  # ~12 t C/ha → ~44 t CO2e/ha
CO2E_T_PER_HA = 44.0
N_RETAINED_KG_PER_HA = 700.0

# --- Robotics defaults ---
CELL_SIZE_M = 10.0
MOTHERSHIP_TURN_RADIUS_M = 6.0
MOTHERSHIP_HOPPER_CAPACITY_KG = 2500.0
DRONE_PAYLOAD_KG = 25.0
# Ops bubble around mothership (wireless power / battery-swap — no physical tether)
DRONE_TETHER_REACH_M = 40.0
DRONE_OPS_RADIUS_M = 40.0
DRONE_BATTERY_SORTIE_CELLS = 8
DEFAULT_SWARM_SIZE = 4
DEFAULT_BED_SPACING_M = 0.85
DEFAULT_GEOJSON_PATH = "data/parcels_weeds.geojson"

# --- Showcase mission (time-based simulation) ---
SHOWCASE_AREA_HA = 12.0  # DEM window holding the 4-parcel pilot block (~9.5 ha cropped)
DRONE_OPS_RADIUS_SHOWCASE_M = 50.0
LANE_SPACING_M = 59.5  # 70 beds x 0.85 m — drones cover +-30 m either side of a lane
STATION_SPACING_M = 30.0
HEADLAND_BED_MARGIN_M = 12.0  # beds stop 12 m from parcel edge (headland strip)


@dataclass(frozen=True)
class PilotSpec:
    """Resolved pilot application targets for a run."""

    area_ha: float = PILOT_AREA_HA
    target_kg_per_m2: float = DEFAULT_TARGET_KG_PER_M2

    @property
    def area_m2(self) -> float:
        return self.area_ha * 10_000.0

    @property
    def total_biomass_t(self) -> float:
        return self.area_ha * self.target_kg_per_m2 * 10.0  # kg/m² → t/ha

    @property
    def supply_deficit_t(self) -> float:
        local = MUNICIPAL_CHIP_SUPPLY_T_YEAR + FOREST_BIOMASS_SUPPLY_T_YEAR
        return max(0.0, self.total_biomass_t - local)
