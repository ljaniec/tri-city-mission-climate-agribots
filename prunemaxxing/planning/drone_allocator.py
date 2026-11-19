"""Dynamic task allocation for tethered / battery-limited drone swarm."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from prunemaxxing.config import (
    DRONE_BATTERY_SORTIE_CELLS,
    DRONE_PAYLOAD_KG,
    DRONE_TETHER_REACH_M,
)
from prunemaxxing.geo.grid import CellState, FieldTwin
from prunemaxxing.planning.mothership_cpp import MothershipPath, Waypoint


@dataclass
class Drone:
    drone_id: int
    tethered: bool = True
    payload_kg: float = DRONE_PAYLOAD_KG
    tether_reach_m: float = DRONE_TETHER_REACH_M
    battery_cells_left: int = DRONE_BATTERY_SORTIE_CELLS
    hopper_load_kg: float = 0.0
    sorties: int = 0
    deposits: int = 0
    kg_delivered: float = 0.0
    flight_m: float = 0.0

    def can_reach(self, mothership: Waypoint, cell: CellState) -> bool:
        dist = float(
            np.hypot(cell.easting - mothership.easting, cell.northing - mothership.northing)
        )
        if self.tethered:
            return dist <= self.tether_reach_m
        return self.battery_cells_left > 0 and dist <= self.tether_reach_m * 1.5


@dataclass
class DepositEvent:
    drone_id: int
    cell_row: int
    cell_col: int
    kg: float
    mothership_lane: int
    step: int


@dataclass
class DroneSwarmAllocator:
    drones: list[Drone]
    events: list[DepositEvent] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        n: int,
        tethered: bool = True,
        payload_kg: float = DRONE_PAYLOAD_KG,
    ) -> "DroneSwarmAllocator":
        drones = [
            Drone(drone_id=i, tethered=tethered, payload_kg=payload_kg) for i in range(n)
        ]
        return cls(drones=drones)

    def allocate_at_waypoint(
        self,
        twin: FieldTwin,
        mothership: Waypoint,
        hopper_kg: float,
        step: int,
        max_cycles: int = 200,
    ) -> tuple[float, float]:
        """
        Assign reachable incomplete cells to drones at the current mothership pose.

        Loops deposit cycles until hopper is empty or no reachable demand remains.
        Returns (kg_deposited, hopper_remaining).
        """
        deposited = 0.0
        remaining_hopper = hopper_kg

        for _ in range(max_cycles):
            if remaining_hopper <= 1e-6:
                break

            candidates = [
                c
                for c in twin.iter_cells()
                if (not c.is_complete)
                and c.traversable
                and not c.weed_exclusion_flag
                and not c.no_till_bed_status
            ]
            candidates.sort(
                key=lambda c: (
                    -c.remaining_kg,
                    (c.easting - mothership.easting) ** 2
                    + (c.northing - mothership.northing) ** 2,
                )
            )

            cycle_kg = 0.0
            for drone in self.drones:
                if remaining_hopper <= 1e-6:
                    break
                if not drone.tethered and drone.battery_cells_left <= 0:
                    drone.battery_cells_left = DRONE_BATTERY_SORTIE_CELLS
                    drone.sorties += 1

                target = next(
                    (c for c in candidates if drone.can_reach(mothership, c)), None
                )
                if target is None:
                    continue

                load = min(drone.payload_kg, remaining_hopper, target.remaining_kg)
                if load <= 1e-6:
                    continue

                dist = float(
                    np.hypot(
                        target.easting - mothership.easting,
                        target.northing - mothership.northing,
                    )
                )
                accepted = target.deposit(load)
                remaining_hopper -= accepted
                deposited += accepted
                cycle_kg += accepted
                drone.kg_delivered += accepted
                drone.deposits += 1
                drone.flight_m += 2.0 * dist
                if not drone.tethered:
                    drone.battery_cells_left -= 1
                self.events.append(
                    DepositEvent(
                        drone_id=drone.drone_id,
                        cell_row=target.row,
                        cell_col=target.col,
                        kg=accepted,
                        mothership_lane=mothership.lane_id,
                        step=step,
                    )
                )
                if target.is_complete and target in candidates:
                    candidates.remove(target)

            if cycle_kg <= 1e-6:
                break

        return deposited, remaining_hopper

    def summary(self) -> dict:
        return {
            "n_drones": len(self.drones),
            "total_kg": sum(d.kg_delivered for d in self.drones),
            "total_deposits": sum(d.deposits for d in self.drones),
            "total_flight_m": sum(d.flight_m for d in self.drones),
            "sorties": sum(d.sorties for d in self.drones),
            "tether_fraction": (
                sum(1 for d in self.drones if d.tethered) / max(1, len(self.drones))
            ),
        }
