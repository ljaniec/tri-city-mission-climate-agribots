"""Mission time series shared by the infographic video, KPI overlays and HTML replay.

``MissionTimeline(res)`` samples a finished :class:`MissionResult` on a uniform
grid and answers "what does the mission look like at time t": material applied
(by drones vs the mothership effector), cells and parcels finished, sorties,
battery swaps, tender deliveries, hopper level, odometer, drone activity,
material flow and climate impact so far. Everything is cumulative and monotone
except stocks (hopper, in flight) and instantaneous states.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from prunemaxxing.kpis.impact import impact_summary
from prunemaxxing.planning.mission import LABEL_CODES

DRONE_STATES = ("idle", "loading", "outbound", "applying", "return", "swap")
DRONE_STATE_COLORS = {
    "idle": "#3a463a",
    "loading": "#e0b97a",
    "outbound": "#38d6e6",
    "applying": "#7CFC00",
    "return": "#2a8f9a",
    "swap": "#ff8c2a",
}
PHASE_NAMES = {v: k for k, v in LABEL_CODES.items()}


@dataclass
class DroneSpan:
    drone: int
    state: str
    t0: float
    t1: float


def _ramp_sum(t: np.ndarray, t0: np.ndarray, t1: np.ndarray, w: np.ndarray, chunk: int = 2048) -> np.ndarray:
    """sum_i w_i * clip((t - t0_i) / (t1_i - t0_i), 0, 1) for every t."""
    out = np.zeros(len(t))
    span = np.maximum(t1 - t0, 1e-6)
    for a in range(0, len(t0), chunk):
        s = slice(a, a + chunk)
        u = np.clip((t[:, None] - t0[None, s]) / span[None, s], 0.0, 1.0)
        out += u @ w[s]
    return out


class MissionTimeline:
    def __init__(self, res, n: int = 1500):
        self.res = res
        self.t_end = float(res.t_end)
        self.t = np.linspace(0.0, self.t_end, n)
        self.impact = impact_summary(res)
        t = self.t
        valid = res.cell_target_kg > 0
        self.target_kg = float(res.cell_target_kg[valid].sum())
        self.n_cells = int(valid.sum())

        # deposits -> total applied + per-cell completion time
        D = res.deposits
        cum = np.cumsum(D["kg"])
        self._dep_t, self._dep_cum = D["t"], cum
        self.applied_kg = self._step(D["t"], cum, t)
        done_t = np.full(len(res.cell_target_kg), np.inf)
        acc = np.zeros(len(res.cell_target_kg))
        for tt, c, kg in zip(D["t"], D["cell"], D["kg"]):
            acc[c] += kg
            if done_t[c] == np.inf and acc[c] >= 0.999 * res.cell_target_kg[c] and valid[c]:
                done_t[c] = tt
        self.cell_done_t = done_t
        self._ev = {"cells": np.sort(done_t[valid])}
        self.cells_done = np.searchsorted(self._ev["cells"], t, side="right")

        # mothership effector vs drones
        M = res.ms_shots
        if len(M.get("kg", [])):
            self.ms_kg = _ramp_sum(t, M["t_fire0"], M["t1"], M["kg"])
            self.ms_units = _ramp_sum(t, M["t_fire0"], M["t1"], M["units"].astype(float))
        else:
            self.ms_kg = np.zeros(len(t))
            self.ms_units = np.zeros(len(t))
        self.drone_kg = np.maximum(self.applied_kg - self.ms_kg, 0.0)

        # parcels
        self.parcels = [p.pid for p in res.site.parcels]
        pid_idx = {p: i for i, p in enumerate(self.parcels)}
        cell_p = np.array([pid_idx.get(c.parcel_id, -1) for c in res.plan.cells])
        self.parcel_target_kg = np.array([res.cell_target_kg[(cell_p == i) & valid].sum() for i in range(len(self.parcels))])
        self.parcel_kg = np.zeros((len(t), len(self.parcels)))
        for i in range(len(self.parcels)):
            m = cell_p[D["cell"]] == i
            self.parcel_kg[:, i] = self._step(D["t"][m], np.cumsum(D["kg"][m]), t)

        # swarm
        S = res.sorties
        self._ev["sorties"] = np.sort(S["t_land"])
        self.sorties_done = np.searchsorted(self._ev["sorties"], t, side="right")
        self.drone_km = self._step(np.sort(S["t_land"]), np.cumsum(S["dist_m"][np.argsort(S["t_land"])]) / 1000.0, t)
        self.drone_kwh = self._step(np.sort(S["t_land"]), np.cumsum(S["energy_kwh"][np.argsort(S["t_land"])]), t)
        ev = res.events
        swap_t = np.sort([e["t"] for e in ev if e["kind"] == "battery_swap"])
        self._ev["swaps"] = swap_t
        self.swaps = np.searchsorted(swap_t, t, side="right")
        top = sorted((e["t"], e["kg"]) for e in ev if e["kind"] == "tender_topup")
        top_t = np.array([a for a, _ in top])
        self._ev["tender_deliveries"] = top_t
        self.tender_deliveries = np.searchsorted(top_t, t, side="right")
        self.tender_kg = self._step(top_t, np.cumsum([b for _, b in top]), t) if top else np.zeros(len(t))
        arr_t = np.sort([e["t"] for e in ev if e["kind"] == "arrive_station"])
        self._ev["stations"] = arr_t
        self.stations_visited = np.searchsorted(arr_t, t, side="right")
        self.n_stations = len(res.visits)
        loaded = np.cumsum(S["kg"][np.argsort(S["t_load"])])
        self.drone_loaded_kg = self._step(np.sort(S["t_load"]), loaded, t)

        # mothership log
        L = res.ms_log
        self.hopper_kg = np.interp(t, L["t"], L["hopper_kg"])
        self.ms_km = np.interp(t, L["t"], L["odo"]) / 1000.0
        self.ms_speed = np.interp(t, L["t"], np.abs(L["v"]))
        self.mode = L["mode"][np.clip(np.searchsorted(L["t"], t, side="right") - 1, 0, len(L["t"]) - 1)]
        self.hopper0_kg = float(L["hopper_kg"][0])

        # drone activity
        self.spans = self._drone_spans()
        self._span_idx = {}
        for d in range(res.n_drones):
            mine = [s for s in self.spans if s.drone == d]
            self._span_idx[d] = (np.array([s.t0 for s in mine]), np.array([s.t1 for s in mine]),
                                 [s.state for s in mine])
        self.airborne = self._busy(S["t_launch"], S["t_land"], t)
        self._air = (np.sort(S["t_launch"]), np.sort(S["t_land"]))
        o = np.argsort(S["t_land"])
        self._land_cum = (S["t_land"][o], np.cumsum(S["dist_m"][o]) / 1000.0, np.cumsum(S["energy_kwh"][o]))
        self.events = sorted(
            [{"t": e["t"], "kind": e["kind"], "label": self._event_label(e)} for e in ev
             if e["kind"] in ("obstacle_detected", "detour_spliced", "arrive_station", "depart_depot", "arrive_depot")],
            key=lambda e: e["t"])

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _step(te: np.ndarray, cum: np.ndarray, t: np.ndarray) -> np.ndarray:
        if len(te) == 0:
            return np.zeros(len(t))
        i = np.searchsorted(te, t, side="right") - 1
        return np.where(i >= 0, np.asarray(cum)[np.clip(i, 0, None)], 0.0)

    @staticmethod
    def _busy(t0: np.ndarray, t1: np.ndarray, t: np.ndarray) -> np.ndarray:
        if len(t0) == 0:
            return np.zeros(len(t))
        a = np.searchsorted(np.sort(t0), t, side="right")
        b = np.searchsorted(np.sort(t1), t, side="right")
        return (a - b).astype(float)

    @staticmethod
    def _event_label(e: dict) -> str:
        k = e["kind"]
        if k == "obstacle_detected":
            return f"{e.get('obstacle_type', 'obstacle')} detected at {e.get('range_m', 0):.0f} m"
        if k == "detour_spliced":
            return f"bypass +{e.get('length_m', 0):.0f} m"
        if k == "arrive_station":
            return f"station {e.get('sid', 0) + 1}"
        return {"depart_depot": "leaves depot", "arrive_depot": "back at depot"}.get(k, k)

    def _drone_spans(self) -> list[DroneSpan]:
        res = self.res
        S = res.sorties
        out: list[DroneSpan] = []
        swap_s = float(res.cfg.drone.swap_time_s)
        for i in np.argsort(S["t_load"]):
            d = int(S["drone"][i])
            seq = (("loading", S["t_load"][i], S["t_launch"][i]), ("outbound", S["t_launch"][i], S["apply_t0"][i]),
                   ("applying", S["apply_t0"][i], S["apply_t1"][i]), ("return", S["apply_t1"][i], S["t_land"][i]))
            out.extend(DroneSpan(d, s, float(a), float(b)) for s, a, b in seq if b > a)
        for e in res.events:
            if e["kind"] == "battery_swap":
                out.append(DroneSpan(int(e["drone"]), "swap", float(e["t"]), float(e["t"]) + swap_s))
        out.sort(key=lambda s: (s.drone, s.t0))
        return out

    def _idx(self, t: float) -> tuple[int, float]:
        x = float(np.clip(t, 0.0, self.t_end)) / max(self.t_end, 1e-9) * (len(self.t) - 1)
        i = min(int(x), len(self.t) - 2)
        return i, x - i

    def _lerp(self, arr: np.ndarray, t: float):
        i, a = self._idx(t)
        return arr[i] * (1 - a) + arr[i + 1] * a

    # ------------------------------------------------------------------ queries
    def frac_applied(self, t: float) -> float:
        return self._exact(self._dep_t, self._dep_cum, t) / max(self.target_kg, 1e-9)

    def phase(self, t: float) -> str:
        L = self.res.ms_log
        i = int(np.clip(np.searchsorted(L["t"], t, side="right") - 1, 0, len(L["t"]) - 1))
        name = PHASE_NAMES.get(int(L["mode"][i]), "transit")
        return "parked" if name == "station" else name

    def _exact(self, te: np.ndarray, cum: np.ndarray, t: float) -> float:
        i = int(np.searchsorted(te, t, side="right")) - 1
        return float(cum[i]) if i >= 0 else 0.0

    def _ms_exact(self, t: float) -> tuple[float, float]:
        M = self.res.ms_shots
        if not len(M.get("kg", [])):
            return 0.0, 0.0
        tt = np.array([t])
        return (float(_ramp_sum(tt, M["t_fire0"], M["t1"], M["kg"])[0]),
                float(_ramp_sum(tt, M["t_fire0"], M["t1"], M["units"].astype(float))[0]))

    def drone_state(self, d: int, t: float) -> str:
        t0, t1, st = self._span_idx.get(d, (np.zeros(0), np.zeros(0), []))
        i = int(np.searchsorted(t0, t, side="right")) - 1
        for j in (i, i - 1):
            if 0 <= j < len(st) and t0[j] <= t < t1[j]:
                return st[j]
        return "idle"

    def count(self, key: str, t: float) -> int:
        return int(np.searchsorted(self._ev[key], t, side="right"))

    def at(self, t: float) -> dict[str, Any]:
        """Snapshot of every scalar at mission time t (seconds)."""
        t = float(np.clip(t, 0.0, self.t_end))
        applied = self._exact(self._dep_t, self._dep_cum, t)
        ms_kg, ms_units = self._ms_exact(t)
        ms_kg = min(ms_kg, applied)
        f = applied / max(float(self.applied_kg[-1]), 1e-9)
        imp = self.impact
        L = self.res.ms_log
        lt, lkm, lkwh = self._land_cum
        snap = {
            "t": t,
            "t_h": t / 3600.0,
            "phase": self.phase(t),
            "applied_t": applied / 1000.0,
            "target_t": self.target_kg / 1000.0,
            "frac": applied / max(self.target_kg, 1e-9),
            "drone_t": (applied - ms_kg) / 1000.0,
            "ms_t": ms_kg / 1000.0,
            "ms_units": ms_units,
            "cells_done": self.count("cells", t),
            "n_cells": self.n_cells,
            "sorties": self.count("sorties", t),
            "swaps": self.count("swaps", t),
            "tender_deliveries": self.count("tender_deliveries", t),
            "tender_t": float(self._lerp(self.tender_kg, t)) / 1000.0,
            "stations": self.count("stations", t),
            "n_stations": self.n_stations,
            "hopper_kg": float(np.interp(t, L["t"], L["hopper_kg"])),
            "ms_km": float(np.interp(t, L["t"], L["odo"])) / 1000.0,
            "ms_speed": float(np.interp(t, L["t"], np.abs(L["v"]))),
            "drone_km": self._exact(lt, lkm, t),
            "drone_kwh": self._exact(lt, lkwh, t),
            "airborne": float(np.searchsorted(self._air[0], t, side="right")
                              - np.searchsorted(self._air[1], t, side="right")),
            "parcel_frac": (self._lerp(self.parcel_kg, t) / np.maximum(self.parcel_target_kg, 1e-9)).tolist(),
            "soil_co2e_t": f * imp["soil_co2e_t"],
            "burn_ghg_avoided_co2e_t": f * imp["burn_ghg_avoided_co2e_t"],
            "total_co2e_t": f * imp["total_co2e_t"],
            "n_retained_kg": f * imp["n_retained_kg"],
            "pm25_avoided_kg": f * imp["pm25_avoided_kg"],
            "saving_eur": None if imp["saving_eur_total"] is None else f * imp["saving_eur_total"],
        }
        if imp.get("cartridges_placed"):
            snap["cartridges"] = f * imp["cartridges_placed"]
            snap["seeds"] = f * imp["seeds_placed"]
        snap["drone_states"] = [self.drone_state(d, t) for d in range(self.res.n_drones)]
        return snap

    def flow_at(self, t: float) -> dict[str, float]:
        """Cumulative material flow (t) along depot -> tender/hopper -> drones/effector -> beds."""
        s = self.at(t)
        drones_loaded = float(self._lerp(self.drone_loaded_kg, t)) / 1000.0
        return {
            "depot_to_hopper": self.hopper0_kg / 1000.0,
            "depot_to_tender": s["tender_t"],
            "tender_to_hopper": s["tender_t"],
            "hopper_to_drones": drones_loaded,
            "hopper_to_effector": s["ms_t"],
            "drones_to_beds": s["drone_t"],
            "effector_to_beds": s["ms_t"],
            "hopper_stock": s["hopper_kg"] / 1000.0,
            "in_flight": max(drones_loaded - s["drone_t"], 0.0),
        }

    def events_until(self, t: float) -> list[dict]:
        return [e for e in self.events if e["t"] <= t]

    def to_json(self, n: int = 600) -> dict[str, Any]:
        """Compact, JSON-serialisable time series (resampled to n points)."""
        idx = np.linspace(0, len(self.t) - 1, n).round().astype(int)
        r = lambda a, d=3: np.round(np.asarray(a)[idx], d).tolist()  # noqa: E731
        return {
            "t": r(self.t, 1),
            "applied_t": r(self.applied_kg / 1000.0),
            "drone_t": r(self.drone_kg / 1000.0),
            "ms_t": r(self.ms_kg / 1000.0),
            "cells_done": r(self.cells_done, 0),
            "sorties": r(self.sorties_done, 0),
            "swaps": r(self.swaps, 0),
            "tender_t": r(self.tender_kg / 1000.0),
            "hopper_kg": r(self.hopper_kg, 0),
            "ms_km": r(self.ms_km),
            "airborne": r(self.airborne, 2),
            "parcel_frac": np.round(self.parcel_kg[idx] / np.maximum(self.parcel_target_kg, 1e-9), 4).tolist(),
            "parcels": self.parcels,
            "target_t": round(self.target_kg / 1000.0, 3),
            "n_cells": self.n_cells,
            "events": [{"t": round(e["t"], 1), "kind": e["kind"], "label": e["label"]} for e in self.events],
            "impact": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.impact.items()},
        }
