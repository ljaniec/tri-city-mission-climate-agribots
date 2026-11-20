"""AGR-E2 KPI aggregation from the Digital Twin + coverage run."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from prunemaxxing.config import (
    CONVENTIONAL_COST_EUR_PER_1000M2,
    REGENSOL_COST_EUR_PER_1000M2,
    TRANSPORT_SPREAD_SHARE,
)
from prunemaxxing.geo.grid import FieldTwin


@dataclass
class KPIReport:
    # Agronomic / climate
    biomass_applied_t: float
    coverage_uniformity: float
    co2e_sequestered_t: float
    n_retained_kg: float
    area_regenerated_m2: float
    area_regenerated_ha: float
    mean_completion: float
    weed_excluded_cells: int

    # Logistics / robotics
    mothership_path_km: float
    swarm_steps: int
    drone_flight_km: float
    drone_deposits: int
    tether_utilization: float
    hopper_refills: int
    road_stockpile_t: float
    offroad_compaction_m2: float

    # Economic
    cost_eur_per_1000m2_est: float
    cost_eur_per_tonne_est: float
    vs_regensol_delta_eur_per_1000m2: float
    vs_conventional_delta_eur_per_1000m2: float

    def as_dict(self) -> dict:
        return asdict(self)

    def format_text(self) -> str:
        d = self.as_dict()
        lines = ["=== PruneMaxxing KPI Report (AGR-E2) ==="]
        for k, v in d.items():
            if isinstance(v, float):
                lines.append(f"  {k}: {v:.4f}")
            else:
                lines.append(f"  {k}: {v}")
        return "\n".join(lines)


def compute_kpis(
    twin: FieldTwin,
    path: Any,
    allocator: Any,
    hopper_refills: int,
    stockpile_kg: float,
    offroad_compaction_m2: float,
    steps: int,
) -> KPIReport:
    eligible = [
        c for c in twin.iter_cells() if c.traversable and not c.weed_exclusion_flag
    ]
    mean_completion = (
        sum(c.completion_ratio for c in eligible) / len(eligible) if eligible else 0.0
    )
    area_m2 = twin.eligible_area_m2()
    area_ha = area_m2 / 10_000.0
    biomass_t = twin.total_deposited_t()

    # Cost model: REGENSOL baseline minus transport/spread automation savings.
    automation = 0.35 + 0.40 * mean_completion + (
        0.10 if offroad_compaction_m2 <= 0 else 0.0
    )
    transport_component = REGENSOL_COST_EUR_PER_1000M2 * TRANSPORT_SPREAD_SHARE
    est_cost = REGENSOL_COST_EUR_PER_1000M2 - automation * transport_component * 0.55
    est_cost -= 80.0 * mean_completion * twin.coverage_uniformity()

    cost_per_t = (
        (est_cost * (area_m2 / 1000.0) / biomass_t) if biomass_t > 0 else float("inf")
    )

    swarm = allocator.summary()
    return KPIReport(
        biomass_applied_t=biomass_t,
        coverage_uniformity=twin.coverage_uniformity(),
        co2e_sequestered_t=twin.estimated_co2e_t(),
        n_retained_kg=twin.estimated_n_retained_kg(),
        area_regenerated_m2=area_m2 * mean_completion,
        area_regenerated_ha=area_ha * mean_completion,
        mean_completion=mean_completion,
        weed_excluded_cells=sum(
            1 for c in twin.iter_cells() if c.weed_exclusion_flag
        ),
        mothership_path_km=path.total_length_m / 1000.0,
        swarm_steps=steps,
        drone_flight_km=swarm["total_flight_m"] / 1000.0,
        drone_deposits=swarm["total_deposits"],
        tether_utilization=swarm.get(
            "wireless_fraction", swarm.get("tether_fraction", 0.0)
        ),
        hopper_refills=hopper_refills,
        road_stockpile_t=stockpile_kg / 1000.0,
        offroad_compaction_m2=offroad_compaction_m2,
        cost_eur_per_1000m2_est=est_cost,
        cost_eur_per_tonne_est=cost_per_t,
        vs_regensol_delta_eur_per_1000m2=est_cost - REGENSOL_COST_EUR_PER_1000M2,
        vs_conventional_delta_eur_per_1000m2=est_cost - CONVENTIONAL_COST_EUR_PER_1000M2,
    )
