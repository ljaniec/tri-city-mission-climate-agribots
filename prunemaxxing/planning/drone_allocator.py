"""Legacy v1 instant-deposit swarm allocator. Superseded by :mod:`prunemaxxing.planning.swarm` (timed sorties)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from prunemaxxing.config import (
    DRONE_BATTERY_SORTIE_CELLS,
    DRONE_PAYLOAD_KG,
    DRONE_TETHER_REACH_M,
)
from prunemaxxing.geo.grid import CellState, FieldTwin
from prunemaxxing.planning.mothership_cpp import Waypoint
from prunemaxxing.planning.payload_modes import PayloadMode, get_payload_spec


def _leg_polyline(
    start: tuple[float, float],
    end: tuple[float, float],
    n_pts: int = 8,
) -> np.ndarray:
    return np.linspace(
        np.asarray(start, dtype=float),
        np.asarray(end, dtype=float),
        num=max(2, n_pts),
    )


def _noisy_leg(
    planned: np.ndarray,
    drone_id: int,
    rng: np.random.Generator,
    amp_m: float = 0.55,
) -> np.ndarray:
    """Gentle actual flight — small sway, no early chaos."""
    n = len(planned)
    if n < 2:
        return planned.copy()
    start, end = planned[0], planned[-1]
    de = float(end[0] - start[0])
    dn = float(end[1] - start[1])
    length = max(float(np.hypot(de, dn)), 1e-6)
    px, py = -dn / length, de / length
    t = np.linspace(0.0, 1.0, n)
    taper = np.sin(np.pi * t)
    amp = amp_m * (0.85 + 0.12 * drone_id)
    sway = amp * np.sin(np.pi * t + 0.4 * drone_id) * taper
    sway += rng.normal(0.0, 0.06, size=n) * taper
    # Tiny lateral offset so trails stay distinguishable without looking erratic
    lateral_bias = (drone_id - 1.5) * 0.45
    actual = np.empty_like(planned)
    actual[:, 0] = start[0] + de * t + px * (sway + lateral_bias * taper)
    actual[:, 1] = start[1] + dn * t + py * (sway + lateral_bias * taper)
    actual[0] = start
    actual[-1] = end
    return actual


def synthesize_sortie_paths(
    ms_e: float,
    ms_n: float,
    cell_e: float,
    cell_n: float,
    drone_id: int,
    rng: np.random.Generator,
    n_pts_leg: int = 8,
    include_return: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    out_p = _leg_polyline((ms_e, ms_n), (cell_e, cell_n), n_pts_leg)
    out_a = _noisy_leg(out_p, drone_id, rng)
    if not include_return:
        return out_p, out_a
    ret_p = _leg_polyline((cell_e, cell_n), (ms_e, ms_n), n_pts_leg)
    ret_a = _noisy_leg(ret_p, drone_id, rng, amp_m=1.05)
    return np.vstack([out_p, ret_p[1:]]), np.vstack([out_a, ret_a[1:]])


@dataclass
class Drone:
    drone_id: int
    wireless_power: bool = True
    payload_kg: float = DRONE_PAYLOAD_KG
    ops_radius_m: float = DRONE_TETHER_REACH_M
    battery_cells_left: int = DRONE_BATTERY_SORTIE_CELLS
    hopper_load_kg: float = 0.0
    sorties: int = 0
    battery_swaps: int = 0
    deposits: int = 0
    kg_delivered: float = 0.0
    flight_m: float = 0.0
    payload_mode: PayloadMode = PayloadMode.DUST
    easting: float = 0.0
    northing: float = 0.0
    alt_m: float = 8.0
    # Sticky exclusive cell assignment (row, col) until that cell is complete
    assigned_cell: tuple[int, int] | None = None

    def can_reach(self, mothership: Waypoint, cell: CellState) -> bool:
        dist = float(
            np.hypot(cell.easting - mothership.easting, cell.northing - mothership.northing)
        )
        reach = self.ops_radius_m * (1.35 if self.wireless_power else 1.0)
        return self.battery_cells_left > 0 and dist <= reach

    def dock_recharge_or_swap(self) -> None:
        self.battery_cells_left = DRONE_BATTERY_SORTIE_CELLS
        self.battery_swaps += 1
        self.sorties += 1


@dataclass
class DepositEvent:
    drone_id: int
    cell_row: int
    cell_col: int
    kg: float
    mothership_lane: int
    step: int
    payload_mode: str = PayloadMode.DUST.value
    drone_easting: float = 0.0
    drone_northing: float = 0.0


@dataclass
class DroneSwarmAllocator:
    drones: list[Drone]
    events: list[DepositEvent] = field(default_factory=list)
    payload_mode: PayloadMode = PayloadMode.DUST
    _planned_segs: dict[int, list[np.ndarray]] = field(default_factory=dict)
    _actual_segs: dict[int, list[np.ndarray]] = field(default_factory=dict)
    path_lens_by_step: list[tuple[int, dict[int, int]]] = field(default_factory=list)
    # Record throughout the run (rolling buffer) so all drones stay visible on the map
    path_record_stride: int = 10
    max_path_segments_per_drone: int = 100
    # Skip chaotic warm-up deposits near the start / mothership
    path_warmup_deposits: int = 24
    min_sortie_length_m: float = 14.0
    _rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(20241120)
    )

    @classmethod
    def create(
        cls,
        n: int,
        wireless_power: bool = True,
        payload_mode: PayloadMode | str = PayloadMode.DUST,
        ops_radius_m: float = DRONE_TETHER_REACH_M,
    ) -> "DroneSwarmAllocator":
        mode = PayloadMode(payload_mode)
        spec = get_payload_spec(mode)
        drones = [
            Drone(
                drone_id=i,
                wireless_power=wireless_power,
                payload_kg=spec.unit_mass_kg,
                ops_radius_m=ops_radius_m,
                payload_mode=mode,
            )
            for i in range(n)
        ]
        return cls(drones=drones, payload_mode=mode)

    def _trim_segs(self, drone_id: int) -> None:
        for store in (self._planned_segs, self._actual_segs):
            segs = store.get(drone_id, [])
            if len(segs) > self.max_path_segments_per_drone:
                # Keep recent history + evenly spaced older samples (full-run coverage)
                keep = self.max_path_segments_per_drone
                older = segs[:-keep // 2]
                recent = segs[-keep // 2 :]
                stride = max(1, len(older) // max(1, keep // 2))
                store[drone_id] = older[::stride][: keep // 2] + recent

    def _append_sortie(
        self,
        drone: Drone,
        mothership: Waypoint,
        cell_e: float,
        cell_n: float,
    ) -> None:
        if drone.deposits <= self.path_warmup_deposits:
            return
        if drone.deposits % self.path_record_stride != 1:
            return
        dist = float(
            np.hypot(cell_e - mothership.easting, cell_n - mothership.northing)
        )
        if dist < self.min_sortie_length_m:
            return
        # Outbound-only trails (MS → cell): clearer per-drone coverage on the map
        planned, actual = synthesize_sortie_paths(
            mothership.easting,
            mothership.northing,
            cell_e,
            cell_n,
            drone.drone_id,
            self._rng,
            n_pts_leg=6,
            include_return=False,
        )
        self._planned_segs.setdefault(drone.drone_id, []).append(planned)
        self._actual_segs.setdefault(drone.drone_id, []).append(actual)
        self._trim_segs(drone.drone_id)

    def path_lengths(self) -> dict[int, int]:
        out: dict[int, int] = {}
        for d in self.drones:
            segs = self._actual_segs.get(d.drone_id, [])
            out[d.drone_id] = int(sum(len(s) for s in segs))
        return out

    def finalized_paths(self) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
        planned: dict[int, np.ndarray] = {}
        actual: dict[int, np.ndarray] = {}
        for d in self.drones:
            p_segs = self._planned_segs.get(d.drone_id, [])
            a_segs = self._actual_segs.get(d.drone_id, [])
            planned[d.drone_id] = (
                np.vstack(p_segs) if p_segs else np.zeros((0, 2), dtype=float)
            )
            actual[d.drone_id] = (
                np.vstack(a_segs) if a_segs else np.zeros((0, 2), dtype=float)
            )
        return planned, actual

    def _pick_exclusive_target(
        self,
        drone: Drone,
        mothership: Waypoint,
        candidates: list[CellState],
        claimed: set[tuple[int, int]],
    ) -> CellState | None:
        """
        Sticky exclusive assignment: finish the assigned cell before switching.
        Among free cells, prefer nearest incomplete (minimize flight energy).
        """
        by_key = {(c.row, c.col): c for c in candidates}

        # Resume sticky assignment if still incomplete + reachable
        if drone.assigned_cell is not None:
            cur = by_key.get(drone.assigned_cell)
            if (
                cur is not None
                and not cur.is_complete
                and drone.can_reach(mothership, cur)
                and drone.assigned_cell not in claimed
            ):
                return cur
            drone.assigned_cell = None

        reachable = [
            c
            for c in candidates
            if drone.can_reach(mothership, c) and (c.row, c.col) not in claimed
        ]
        if not reachable:
            return None

        # Energy-aware nearest-first, with angular sectors so drones fan out
        n_d = max(1, len(self.drones))

        def _score(c: CellState) -> tuple[float, float]:
            de = c.easting - mothership.easting
            dn = c.northing - mothership.northing
            dist2 = de * de + dn * dn
            ang = float(np.arctan2(dn, de))
            sector = int(((ang + np.pi) / (2 * np.pi)) * n_d) % n_d
            # Prefer this drone's sector (0), else small penalty; then distance
            sector_pen = 0.0 if sector == drone.drone_id else 50.0
            return (sector_pen + dist2, -c.remaining_kg)

        reachable.sort(key=_score)
        choice = reachable[0]
        drone.assigned_cell = (choice.row, choice.col)
        return choice

    def allocate_at_waypoint(
        self,
        twin: FieldTwin,
        mothership: Waypoint,
        hopper_kg: float,
        step: int,
        max_cycles: int = 200,
    ) -> tuple[float, float]:
        deposited = 0.0
        remaining_hopper = hopper_kg
        spec = get_payload_spec(self.payload_mode)

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
            if not candidates:
                break

            claimed: set[tuple[int, int]] = set()
            cycle_kg = 0.0

            for drone in self.drones:
                if remaining_hopper <= 1e-6:
                    break
                if drone.battery_cells_left <= 0:
                    drone.dock_recharge_or_swap()
                    drone.easting = mothership.easting
                    drone.northing = mothership.northing
                    drone.alt_m = 2.0
                    drone.assigned_cell = None
                    continue

                target = self._pick_exclusive_target(
                    drone, mothership, candidates, claimed
                )
                if target is None:
                    # Hover offset near mothership (still spatially distinct)
                    ang = 0.5 * np.pi * drone.drone_id / max(1, len(self.drones))
                    drone.easting = mothership.easting + 6.0 * np.cos(ang)
                    drone.northing = mothership.northing + 6.0 * np.sin(ang)
                    drone.alt_m = 7.0 + drone.drone_id
                    continue

                claimed.add((target.row, target.col))

                load = min(drone.payload_kg, remaining_hopper, target.remaining_kg)
                if self.payload_mode is PayloadMode.CARTRIDGE:
                    units = max(1, int(load / spec.unit_mass_kg))
                    load = min(
                        units * spec.unit_mass_kg, remaining_hopper, target.remaining_kg
                    )
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
                drone.battery_cells_left -= 1
                self._append_sortie(drone, mothership, target.easting, target.northing)
                drone.easting = target.easting
                drone.northing = target.northing
                drone.alt_m = 4.5 + 0.5 * drone.drone_id
                self.events.append(
                    DepositEvent(
                        drone_id=drone.drone_id,
                        cell_row=target.row,
                        cell_col=target.col,
                        kg=accepted,
                        mothership_lane=mothership.lane_id,
                        step=step,
                        payload_mode=self.payload_mode.value,
                        drone_easting=target.easting,
                        drone_northing=target.northing,
                    )
                )
                if target.is_complete:
                    drone.assigned_cell = None
                    if target in candidates:
                        candidates.remove(target)

            if cycle_kg <= 1e-6:
                break

        self.path_lens_by_step.append((step, self.path_lengths()))
        return deposited, remaining_hopper

    def summary(self) -> dict:
        return {
            "n_drones": len(self.drones),
            "payload_mode": self.payload_mode.value,
            "total_kg": sum(d.kg_delivered for d in self.drones),
            "total_deposits": sum(d.deposits for d in self.drones),
            "total_flight_m": sum(d.flight_m for d in self.drones),
            "sorties": sum(d.sorties for d in self.drones),
            "battery_swaps": sum(d.battery_swaps for d in self.drones),
            "wireless_fraction": (
                sum(1 for d in self.drones if d.wireless_power) / max(1, len(self.drones))
            ),
            "tether_fraction": 0.0,
        }
