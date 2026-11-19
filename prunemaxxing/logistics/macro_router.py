"""
Macro-logistics layer (scaffold).

Balances urban/forest supply → chipper hubs → field-edge stockpiles,
keeping heavy trucks on roads to avoid compaction.
"""

from __future__ import annotations

from dataclasses import dataclass

from prunemaxxing.config import (
    FOREST_BIOMASS_SUPPLY_T_YEAR,
    MUNICIPAL_CHIP_SUPPLY_T_YEAR,
    PilotSpec,
)


@dataclass
class SupplyNode:
    name: str
    annual_t: float
    kind: str  # municipal | forest | regional_plant


@dataclass
class MacroLogisticsPlan:
    demand_t: float
    local_supply_t: float
    deficit_t: float
    sources: list[SupplyNode]
    notes: list[str]

    def as_dict(self) -> dict:
        return {
            "demand_t": self.demand_t,
            "local_supply_t": self.local_supply_t,
            "deficit_t": self.deficit_t,
            "sources": [s.__dict__ for s in self.sources],
            "notes": self.notes,
        }


def plan_macro_logistics(pilot: PilotSpec | None = None) -> MacroLogisticsPlan:
    pilot = pilot or PilotSpec()
    sources = [
        SupplyNode("Viladecans municipal chips", MUNICIPAL_CHIP_SUPPLY_T_YEAR, "municipal"),
        SupplyNode("Local forest biomass", FOREST_BIOMASS_SUPPLY_T_YEAR, "forest"),
        SupplyNode("Sant Pere de Ribes plant", 0.0, "regional_plant"),
        SupplyNode("Baix Llobregat plant", 0.0, "regional_plant"),
    ]
    local = MUNICIPAL_CHIP_SUPPLY_T_YEAR + FOREST_BIOMASS_SUPPLY_T_YEAR
    deficit = max(0.0, pilot.total_biomass_t - local)
    notes = [
        "Heavy trucks stay on farm roads/tracks; drop at perimeter stockpile nodes only.",
        "Chip to <5 cm length / <3 cm diameter before field delivery.",
        f"Pilot demand ≈ {pilot.total_biomass_t:.0f} t for {pilot.area_ha:g} ha "
        f"@ {pilot.target_kg_per_m2:g} kg/m².",
        "Regional plants fill deficit via multi-source routing (OR-Tools TBD).",
    ]
    return MacroLogisticsPlan(
        demand_t=pilot.total_biomass_t,
        local_supply_t=local,
        deficit_t=deficit,
        sources=sources,
        notes=notes,
    )
