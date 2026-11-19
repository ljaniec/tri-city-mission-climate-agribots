"""Discrete-event mothership + drone coverage simulation (Digital Twin loop)."""

from __future__ import annotations

from dataclasses import dataclass

from prunemaxxing.config import (
    DEFAULT_SWARM_SIZE,
    DRONE_TETHER_REACH_M,
    MOTHERSHIP_HOPPER_CAPACITY_KG,
)
from prunemaxxing.geo.grid import FieldTwin
from prunemaxxing.kpis.metrics import KPIReport, compute_kpis
from prunemaxxing.planning.drone_allocator import DroneSwarmAllocator
from prunemaxxing.planning.mothership_cpp import MothershipPath, plan_mothership_coverage


@dataclass
class CoverageSimResult:
    path: MothershipPath
    twin: FieldTwin
    allocator: DroneSwarmAllocator
    steps: int
    hopper_refills: int
    road_stockpile_drops_kg: float
    offroad_compaction_m2: float
    kpis: KPIReport


def run_coverage_simulation(
    twin: FieldTwin,
    swarm_size: int = DEFAULT_SWARM_SIZE,
    tethered: bool = True,
    tether_reach_m: float = DRONE_TETHER_REACH_M,
    hopper_capacity_kg: float = MOTHERSHIP_HOPPER_CAPACITY_KG,
    max_passes: int = 3,
) -> CoverageSimResult:
    """
    Advance mothership along CPP path; at each waypoint drones deposit biomass.

    Heavy haulage is assumed to refill the mothership only at perimeter nodes
    (road edges) — off-road compaction footprint stays 0 by design.
    """
    path = plan_mothership_coverage(twin, tether_reach_m=tether_reach_m)
    allocator = DroneSwarmAllocator.create(swarm_size, tethered=tethered)

    hopper = 0.0
    refills = 0
    stockpile_kg = 0.0
    steps = 0

    # Perimeter / road-edge proxy: outer ring of the parcel.
    perimeter = {
        (r, c)
        for r in range(twin.nrows)
        for c in range(twin.ncols)
        if r in (0, twin.nrows - 1) or c in (0, twin.ncols - 1)
    }

    for _pass in range(max_passes):
        if not twin.work_cells():
            break
        if not path.waypoints:
            break
        for wp in path.waypoints:
            steps += 1
            # Heavy trucks never leave roads; mothership restocks only at perimeter nodes.
            on_perimeter = (wp.row, wp.col) in perimeter
            if on_perimeter and hopper < 0.20 * hopper_capacity_kg:
                need = hopper_capacity_kg - hopper
                hopper = hopper_capacity_kg
                refills += 1
                stockpile_kg += need
            elif hopper < 1e-6:
                # Abstracted return-to-edge refill (no off-road truck traffic).
                hopper = hopper_capacity_kg
                refills += 1
                stockpile_kg += hopper_capacity_kg

            deposited, hopper = allocator.allocate_at_waypoint(
                twin, wp, hopper, step=steps
            )
            _ = deposited

        # Replan lightly if incomplete cells remain (adaptive re-coverage)
        if twin.work_cells():
            path = plan_mothership_coverage(twin, tether_reach_m=tether_reach_m)

    twin.lock_no_till_beds()
    kpis = compute_kpis(
        twin=twin,
        path=path,
        allocator=allocator,
        hopper_refills=refills,
        stockpile_kg=stockpile_kg,
        offroad_compaction_m2=0.0,
        steps=steps,
    )
    return CoverageSimResult(
        path=path,
        twin=twin,
        allocator=allocator,
        steps=steps,
        hopper_refills=refills,
        road_stockpile_drops_kg=stockpile_kg,
        offroad_compaction_m2=0.0,
        kpis=kpis,
    )
