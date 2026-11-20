"""Time-based mission simulation: mothership + drone swarm + hidden obstacles.

Mothership: pure-pursuit controller -> actuator-limited bicycle plant, stop-and-go
at drone service stations, logged at 10 Hz (commands *and* response). A 360°
lidar (25 m; camera 20 m for flat hazards) discovers the hidden obstacles from
``unknown_obstacles.geojson``; the local planner splices a bypass into the
active path, so the executed trajectory diverges from the global plan.

Swarm: at each station the drones run sorties (event-driven, analytic timing)
until every cell of the station reaches its dose; the mothership departs once
all drones are back on deck. Battery swaps, hopper level and tender top-ups
are tracked.

Near field: patches within 40 ft (12.19 m) of the parked mothership are served
by its own effectors in parallel with the swarm (:mod:`ms_effectors`): the
snow-blower chip chute in DUST mode, the two pneumatic cartridge cannons
(port / starboard, independent) in CARTRIDGE mode.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from shapely.geometry import Point, box
from shapely.ops import unary_union

from prunemaxxing.config import (
    CO2E_T_PER_HA,
    CONVENTIONAL_COST_EUR_PER_1000M2,
    N_RETAINED_KG_PER_HA,
    REGENSOL_COST_EUR_PER_1000M2,
    REGENSOL_OVERHEAD_EUR_PER_1000M2,
    TRANSPORT_SPREAD_SHARE,
)
from prunemaxxing.geo.grid import FieldTwin
from prunemaxxing.geo.site import Site
from prunemaxxing.planning import ms_effectors as fx
from prunemaxxing.planning.path_geometry import arclength, curvature, plan_detour, splice
from prunemaxxing.planning.payload_modes import PayloadMode, PayloadSpec, get_payload_spec
from prunemaxxing.planning.route_planner import GlobalPlan, PlannerConfig, plan_global_route
from prunemaxxing.planning.swarm import (
    PHASE_DOCKED,
    DroneParams,
    SortieTrajectory,
    cell_patches,
    dock_xy,
    smooth_heading,
    sortie_energy_kwh,
    sortie_timing,
    synthesize_trajectory,
)
from prunemaxxing.planning.vehicle import (
    PurePursuitController,
    VehicleParams,
    VehicleState,
    step_vehicle,
)
from prunemaxxing.sim.world_frame import Terrain, frame_for_site, load_terrain

LABEL_CODES = {"transit": 0, "lane": 1, "headland": 2, "weed_detour": 3, "obstacle_detour": 4, "station": 5}
LABEL_NAMES = {v: k for k, v in LABEL_CODES.items()}


@dataclass
class MissionConfig:
    vehicle: VehicleParams = field(default_factory=VehicleParams)
    drone: DroneParams = field(default_factory=DroneParams)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    swarm_size: int = 4
    payload_mode: str = "dust"
    dt_drive: float = 0.05
    log_hz: float = 10.0
    lidar_range_m: float = 25.0
    camera_range_m: float = 20.0
    sensor_period_s: float = 0.25
    obstacle_clearance_m: float = 2.2
    hopper_capacity_kg: float = 2500.0
    tender_capacity_kg: float = 12000.0
    tender_trucks: int = 2
    tender_speed_mps: float = 4.0
    tender_request_frac: float = 0.6
    tender_transfer_s: float = 150.0
    tender_depot_load_s: float = 300.0
    min_dwell_s: float = 15.0
    effectors: fx.EffectorParams = field(default_factory=fx.EffectorParams)
    wind_mean_mps: tuple[float, float] = (1.6, 1.9)  # garbí: SW sea breeze
    seed: int = 7
    max_stations: int | None = None  # truncate (tests / quick previews)


@dataclass
class MissionResult:
    site: Site
    plan: GlobalPlan
    cfg: MissionConfig
    spec: PayloadSpec
    terrain: Terrain
    ms_log: dict[str, np.ndarray]
    active_path: np.ndarray
    active_labels: np.ndarray
    detours: list[dict]
    events: list[dict]
    visits: list[dict]
    sorties: dict[str, np.ndarray]
    patterns: dict[int, list[np.ndarray]]
    deposits: dict[str, np.ndarray]
    cell_target_kg: np.ndarray
    cell_blocked: np.ndarray
    kpis: dict[str, Any]
    ms_shots: dict[str, np.ndarray] = field(default_factory=dict)
    _drone_index: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = field(default_factory=dict)

    # ------------------------------------------------------------------ queries
    @property
    def t_end(self) -> float:
        return float(self.ms_log["t"][-1])

    @property
    def n_drones(self) -> int:
        return self.cfg.swarm_size

    def ground_z(self, e, n):
        fr = frame_for_site(self.site)
        return self.terrain.height_at(np.asarray(e) - fr.center_e, np.asarray(n) - fr.center_n)

    def ms_state_at(self, t: float) -> dict[str, float]:
        L = self.ms_log
        tt = L["t"]
        t = float(np.clip(t, tt[0], tt[-1]))
        i = int(np.searchsorted(tt, t, side="right")) - 1
        i = max(0, min(i, len(tt) - 2))
        dt = tt[i + 1] - tt[i]
        a = 0.0 if dt <= 1e-9 else (t - tt[i]) / dt
        out = {}
        for k in ("x", "y", "z", "yaw", "v", "v_cmd", "steer", "steer_cmd", "cte", "s", "odo", "hopper_kg"):
            out[k] = float(L[k][i] + a * (L[k][i + 1] - L[k][i]))
        out["mode"] = int(L["mode"][i])
        out["t"] = t
        return out

    def _index_for(self, d: int):
        if d not in self._drone_index:
            m = np.where(self.sorties["drone"] == d)[0]
            order = m[np.argsort(self.sorties["t_launch"][m])]
            self._drone_index[d] = (order, self.sorties["t_load"][order], self.sorties["t_land"][order])
        return self._drone_index[d]

    def trajectory(self, k: int) -> SortieTrajectory:
        return _cached_traj(id(self), k, self)

    def drone_state_at(self, d: int, t: float) -> dict[str, Any]:
        order, t_load, t_land = self._index_for(d)
        j = int(np.searchsorted(t_load, t, side="right")) - 1
        ms = self.ms_state_at(t)
        dk = dock_xy(ms["x"], ms["y"], ms["yaw"], d)
        dock_z = float(self.ground_z(dk[0], dk[1])) + self.cfg.drone.deck_height_m
        state = {"x": dk[0], "y": dk[1], "z": dock_z, "yaw": ms["yaw"], "phase": PHASE_DOCKED,
                 "cell": -1, "payload": 0.0, "sortie": -1, "vx": 0.0, "vy": 0.0}
        if j >= 0 and t <= t_land[j]:
            k = int(order[j])
            t_launch = self.sorties["t_launch"][k]
            state["sortie"] = k
            state["cell"] = int(self.sorties["cell"][k])
            if t < t_launch:  # loading on deck
                state["payload"] = float((t - t_load[j]) / max(t_launch - t_load[j], 1e-6))
                return state
            tr = self.trajectory(k)
            x = float(np.interp(t, tr.t, tr.actual[:, 0]))
            y = float(np.interp(t, tr.t, tr.actual[:, 1]))
            z = float(np.interp(t, tr.t, tr.actual[:, 2]))
            i = int(np.clip(np.searchsorted(tr.t, t), 1, len(tr.t) - 1))
            vx = (tr.actual[i, 0] - tr.actual[i - 1, 0]) / max(tr.t[i] - tr.t[i - 1], 1e-6)
            vy = (tr.actual[i, 1] - tr.actual[i - 1, 1]) / max(tr.t[i] - tr.t[i - 1], 1e-6)
            yaw = float(np.interp(t, tr.t, tr.yaw)) if tr.yaw is not None else ms["yaw"]
            state.update(x=x, y=y, z=z, yaw=yaw, phase=int(tr.phase[i]), vx=vx, vy=vy,
                         payload=float(np.interp(t, tr.t, tr.payload_frac)))
        return state

    def battery_at(self, d: int, t: float) -> float:
        order, t_load, t_land = self._index_for(d)
        j = int(np.searchsorted(t_load, t, side="right")) - 1
        if j < 0:
            return 1.0
        k = int(order[j])
        S = self.sorties
        if t >= t_land[j]:
            return float(S["soc1"][k])
        a = (t - S["t_launch"][k]) / max(S["t_land"][k] - S["t_launch"][k], 1e-6)
        return float(S["soc0"][k] + np.clip(a, 0, 1) * (S["soc1"][k] - S["soc0"][k]))

    def recent_trail(self, d: int, t: float, window_s: float = 30.0) -> np.ndarray:
        """Actual xy samples of drone d over (t - window, t]."""
        order, t_load, t_land = self._index_for(d)
        lo = int(np.searchsorted(t_land, t - window_s))
        hi = int(np.searchsorted(t_load, t, side="right"))
        pts = []
        for j in range(lo, hi):
            tr = self.trajectory(int(order[j]))
            m = (tr.t > t - window_s) & (tr.t <= t)
            if m.any():
                pts.append(tr.actual[m])
        return np.vstack(pts) if pts else np.zeros((0, 3))

    def completion_at(self, times) -> np.ndarray:
        """(len(times), n_cells) completion ratio (NaN for blocked cells)."""
        times = np.atleast_1d(np.asarray(times, dtype=float))
        order = np.argsort(times)
        dep_t, dep_c, dep_kg = self.deposits["t"], self.deposits["cell"], self.deposits["kg"]
        n = len(self.cell_target_kg)
        acc = np.zeros(n)
        out = np.zeros((len(times), n))
        j = 0
        for oi in order:
            tq = times[oi]
            k = int(np.searchsorted(dep_t, tq, side="right"))
            if k > j:
                np.add.at(acc, dep_c[j:k], dep_kg[j:k])
                j = k
            out[oi] = acc
        tgt = np.where(self.cell_target_kg > 0, self.cell_target_kg, np.nan)
        return np.clip(out / tgt, 0.0, 1.0)

    def deposited_kg_at(self, t: float) -> float:
        k = int(np.searchsorted(self.deposits["t"], t, side="right"))
        return float(self.deposits["kg"][:k].sum())

    def detected_at(self, t: float) -> list[str]:
        return [e["oid"] for e in self.events if e["kind"] == "obstacle_detected" and e["t"] <= t]

    # ------------------------------------------------------------ near-field effectors
    def ms_row(self, k: int) -> dict[str, float]:
        return {c: v[k] for c, v in self.ms_shots.items()}

    def ms_volley(self, k: int) -> dict[str, np.ndarray]:
        cache = self.__dict__.setdefault("_volley_cache", {})
        v = cache.get(k)
        if v is None:
            if len(cache) > 4000:
                cache.clear()
            M = self.ms_shots
            pat = self.patterns[int(M["cell"][k])][int(M["pattern"][k])]
            v = fx.volley_shots(k, self.ms_row(k), pat, self.ground_z, self.cfg.effectors)
            cache[k] = v
        return v

    def _fx_index(self) -> dict[tuple[int, int], np.ndarray]:
        if "_fx_idx_cache" not in self.__dict__:
            M = self.ms_shots
            idx = {}
            if len(M.get("kg", [])):
                for key in {(int(a), int(b)) for a, b in zip(M["kind"], M["side"])}:
                    rows = np.flatnonzero((M["kind"] == key[0]) & (M["side"] == key[1]))
                    idx[key] = rows[np.argsort(M["t0"][rows], kind="stable")]
            self.__dict__["_fx_idx_cache"] = idx
        return self.__dict__["_fx_idx_cache"]

    def ms_effector_state(self, t: float) -> list[dict[str, Any]]:
        """Pose + activity of each mothership effector at time t.

        ``rel_az`` is relative to the mothership heading (what the slewing ring /
        turret joint sees); ``az`` is the world azimuth. Blower: ``aim`` (UTM xy)
        sweeps the patch while ``firing``. Cannon: ``pellets`` (n,3) in flight,
        ``shot`` fired at ``t_shot`` from ``barrel``.
        """
        M = self.ms_shots
        ms_yaw = self.ms_state_at(t)["yaw"]
        cart = self.spec.mode is PayloadMode.CARTRIDGE
        keys = [(fx.KIND_CANNON, 0), (fx.KIND_CANNON, 1)] if cart else [(fx.KIND_BLOWER, -1)]
        out = []
        for key in keys:
            park = math.pi if key[0] == fx.KIND_BLOWER else (0.5 - key[1]) * math.pi
            st = {"kind": key[0], "side": key[1], "rel_az": park, "firing": False, "row": -1, "aim": None,
                  "pellets": np.zeros((0, 3)), "trails": np.zeros((0, 8, 3)), "t_shot": -1e9, "barrel": 0, "elev": 0.0, "range_m": 0.0}
            rows = self._fx_index().get(key)
            if rows is not None and len(rows):
                j = int(np.searchsorted(M["t0"][rows], t, side="right")) - 1
                cand = [rows[j]] if j >= 0 else []
                if j >= 1:
                    cand.insert(0, rows[j - 1])  # pellets of the previous volley may still fly
                if j >= 0:
                    k = int(rows[j])
                    yaw_k = float(M["ms_yaw"][k])
                    rel0, rel1 = M["az0"][k] - yaw_k, M["az1"][k] - yaw_k
                    st["row"] = k
                    st["range_m"] = float(M["range_m"][k])
                    elev = math.radians(self.cfg.effectors.blower_elev_deg if key[0] == fx.KIND_BLOWER
                                        else self.cfg.effectors.cannon_elev_deg)
                    t0, tf0, tfree = M["t0"][k], M["t_fire0"][k], M["t_free"][k]
                    if t < tf0:
                        a = (t - t0) / max(tf0 - t0, 1e-6)
                        st["rel_az"] = rel0 + a * fx.ang_diff(rel0, rel1)
                        st["elev"] = a * elev
                    elif t <= tfree + 6.0:
                        st["elev"] = elev
                        if key[0] == fx.KIND_BLOWER:
                            pat = self.patterns[int(M["cell"][k])][int(M["pattern"][k])]
                            u = float(np.clip((t - tf0) / max(tfree - tf0, 1e-6), 0.0, 1.0))
                            seg = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(pat, axis=0).T))])
                            aim = np.array([np.interp(u * seg[-1], seg, pat[:, 0]), np.interp(u * seg[-1], seg, pat[:, 1])])
                            tip = fx.mount_world(M["ms_x"][k], M["ms_y"][k], yaw_k, fx.CHUTE_TIP)
                            st["rel_az"] = math.atan2(aim[1] - tip[1], aim[0] - tip[0]) - yaw_k
                            st["aim"] = aim
                            st["firing"] = t <= tfree
                        else:
                            v = self.ms_volley(k)
                            i = int(np.clip(np.searchsorted(v["t_fire"], t, side="right") - 1, 0, len(v["t_fire"]) - 1))
                            nxt = min(i + 1, len(v["az"]) - 1) if t > v["t_fire"][i] + 0.25 else i
                            st["rel_az"] = float(v["az"][nxt]) - yaw_k
                            st["firing"] = t <= tfree
                            st["t_shot"] = float(v["t_fire"][i]) if t >= v["t_fire"][0] else -1e9
                            st["barrel"] = int(v["barrel"][i])
                            st["aim"] = v["target"][i]
                    else:
                        a = float(np.clip((t - tfree - 6.0) / 4.0, 0.0, 1.0))
                        st["rel_az"] = rel1 + a * fx.ang_diff(rel1, park)
                        st["elev"] = (1 - a) * elev
                if key[0] == fx.KIND_CANNON:
                    fly, trails = [], []
                    lag = np.arange(9) * 0.028
                    for k in cand:
                        if t > M["t1"][k] + 0.5:
                            continue
                        v = self.ms_volley(int(k))
                        m = (v["t_fire"] <= t) & (t <= v["t_fire"] + v["t_fly"])
                        for i in np.flatnonzero(m):
                            p0, p1, T = v["muzzle"][i], v["land"][i], max(float(v["t_fly"][i]), 1e-6)
                            vz = (p1[2] - p0[2]) / T + 0.5 * fx.G * T
                            tt = np.maximum(t - v["t_fire"][i] - lag, 0.0)
                            u = tt / T
                            pts = np.column_stack([p0[0] + u * (p1[0] - p0[0]), p0[1] + u * (p1[1] - p0[1]),
                                                   p0[2] + vz * tt - 0.5 * fx.G * tt * tt])
                            fly.append(pts[0])
                            trails.append(pts[1:])
                    st["pellets"] = np.array(fly) if fly else np.zeros((0, 3))
                    st["trails"] = np.array(trails) if trails else np.zeros((0, 8, 3))
            st["az"] = st["rel_az"] + ms_yaw
            out.append(st)
        return out

    def ms_log_rows(self) -> list[dict]:
        L = self.ms_log
        return [{k: L[k][i] for k in L} for i in range(len(L["t"]))]

    def write_control_log(self, path) -> None:
        L = self.ms_log
        cols = ["t", "x", "y", "z", "yaw", "v", "v_cmd", "steer", "steer_cmd", "cte", "s", "odo", "hopper_kg", "mode"]
        with open(path, "w") as f:
            f.write(",".join(cols[:-1] + ["mode"]) + "\n")
            for i in range(len(L["t"])):
                vals = [f"{L[c][i]:.4f}" for c in cols[:-1]]
                vals.append(LABEL_NAMES.get(int(L["mode"][i]), "?"))
                f.write(",".join(vals) + "\n")


_TRAJ_CACHE: dict[tuple[int, int], SortieTrajectory] = {}


def _cached_traj(owner_id: int, k: int, res: MissionResult) -> SortieTrajectory:
    key = (owner_id, k)
    tr = _TRAJ_CACHE.get(key)
    if tr is None:
        if len(_TRAJ_CACHE) > 6000:
            _TRAJ_CACHE.clear()
        S = res.sorties
        cell = int(S["cell"][k])
        pattern = res.patterns[cell][int(S["pattern"][k])]
        tr = synthesize_trajectory(
            k, float(S["t_launch"][k]), np.array([S["dock_x"][k], S["dock_y"][k]]), pattern, res.spec,
            res.cfg.drone, res.ground_z, np.asarray(res.cfg.wind_mean_mps),
        )
        tr.yaw = smooth_heading(tr, res.ms_state_at(float(S["t_launch"][k]))["yaw"])
        _TRAJ_CACHE[key] = tr
    return tr


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------


class _Sim:
    def __init__(self, site: Site, twin: FieldTwin, cfg: MissionConfig, plan: GlobalPlan | None):
        self.site, self.twin, self.cfg = site, twin, cfg
        self.spec = get_payload_spec(cfg.payload_mode)
        pcfg = cfg.planner
        if abs(pcfg.target_kg_per_m2 - self.spec.dose_kg_per_m2) > 1e-9:
            pcfg = PlannerConfig(**{**pcfg.__dict__, "target_kg_per_m2": self.spec.dose_kg_per_m2})
        self.plan = plan if plan is not None else plan_global_route(site, twin, pcfg)
        if plan is not None:
            for c in self.plan.cells:
                c.target_kg = c.area_m2 * self.spec.dose_kg_per_m2
        self.rng = np.random.default_rng(cfg.seed)
        fr = frame_for_site(site)
        self.frame = fr
        self.terrain = load_terrain(fr, site.width_m / 2 + 70, site.height_m / 2 + 70, res_m=2.0)
        self.vp = cfg.vehicle
        self.dp = cfg.drone

        self.path = self.plan.path.copy()
        self.labels = self.plan.labels.copy()
        self._refresh_path()

        n = len(self.plan.cells)
        self.target = np.array([c.target_kg for c in self.plan.cells])
        self.deposited = np.zeros(n)
        self.blocked = np.zeros(n, dtype=bool)
        self.cell_sorties = np.zeros(n, dtype=int)
        self.patterns: dict[int, list[np.ndarray]] = {}

        self.station_s = np.array([st.s for st in self.plan.stations])
        self.events: list[dict] = []
        self.detours: list[dict] = []
        self.visits: list[dict] = []
        self.log: dict[str, list] = {k: [] for k in
                                     ("t", "x", "y", "z", "yaw", "v", "v_cmd", "steer", "steer_cmd", "cte", "s",
                                      "odo", "hopper_kg", "mode")}
        self.sortie_rows: list[tuple] = []
        self.ms_rows: list[tuple] = []
        self.fx_az: dict[int, float] = {}
        self.fx_t_idle = 0.0
        self.dep_rows: list[tuple] = []
        self.hopper = cfg.hopper_capacity_kg
        self.tender = {"pending": False, "arrive": 0.0, "serving": 0, "trips": 0, "km": 0.0,
                       "deliveries": 0, "wait_s": 0.0,
                       "fleet": [{"load": cfg.tender_capacity_kg, "ready": 0.0} for _ in range(cfg.tender_trucks)]}
        self.undetected = list(site.obstacles)
        self.detected_polys: list = []
        self.weed_no_go = [w.polygon.buffer(1.0) for w in site.weeds]

        yaw0 = self.plan.start_heading
        cx, cy = self.path[0]
        h = 0.5 * self.vp.wheelbase_m
        self.st = VehicleState(cx - h * math.cos(yaw0), cy - h * math.sin(yaw0), yaw0)
        self.ctrl = PurePursuitController(self.vp)
        self.t = 0.0

    # ------------------------------------------------------------------ helpers
    def _refresh_path(self) -> None:
        self.s = arclength(self.path)
        k = curvature(self.path)
        k[~np.isfinite(k)] = 0.0
        # light smoothing so a single fillet joint does not spike the speed limit
        self.kappa = np.convolve(k, np.ones(5) / 5.0, mode="same")

    def _z(self, e, n):
        return self.terrain.height_at(np.asarray(e) - self.frame.center_e, np.asarray(n) - self.frame.center_n)

    def _log(self, steer_cmd: float, v_cmd: float, cte: float, s_val: float, mode: int) -> None:
        cx, cy = self.st.centre(self.vp.wheelbase_m)
        L = self.log
        L["t"].append(self.t)
        L["x"].append(cx)
        L["y"].append(cy)
        L["z"].append(float(self._z(cx, cy)))
        yaw = self.st.yaw
        if L["yaw"]:
            prev = L["yaw"][-1]
            yaw = prev + ((yaw - prev + math.pi) % (2 * math.pi) - math.pi)
        L["yaw"].append(yaw)
        L["v"].append(self.st.v)
        L["v_cmd"].append(v_cmd)
        L["steer"].append(self.st.steer)
        L["steer_cmd"].append(steer_cmd)
        L["cte"].append(cte)
        L["s"].append(s_val)
        L["odo"].append(self.st.odo)
        L["hopper_kg"].append(self.hopper)
        L["mode"].append(mode)

    # ------------------------------------------------------------------ sensing
    def _sense(self, i_path: int) -> None:
        if not self.undetected:
            return
        yaw = self.st.yaw
        cx, cy = self.st.centre(self.vp.wheelbase_m)
        sensor = Point(cx + 1.6 * math.cos(yaw), cy + 1.6 * math.sin(yaw))
        for ob in list(self.undetected):
            rng_m = self.cfg.lidar_range_m if ob.height_m > 0.3 else self.cfg.camera_range_m
            d = ob.polygon.distance(sensor)
            if d > rng_m:
                continue
            self.undetected.remove(ob)
            self._on_detect(ob, d, i_path)

    def _on_detect(self, ob, dist: float, i_path: int) -> None:
        perceived = ob.polygon.buffer(0.25)
        ev = {"t": self.t, "kind": "obstacle_detected", "oid": ob.oid, "obstacle_type": ob.kind,
              "range_m": round(dist, 2), "xy": list(self.st.centre(self.vp.wheelbase_m))}
        self.events.append(ev)
        # cells under solid obstacles cannot be mulched
        if ob.height_m > 0.3:
            foot = ob.polygon.buffer(1.0)
            for c in self.plan.cells:
                if c.polygon is not None and c.polygon.intersects(foot):
                    lost = c.polygon.intersection(foot).area
                    left = max(0.0, c.area_m2 - lost)
                    if left < 0.35 * c.area_m2:
                        self.blocked[c.idx] = True
                        self.target[c.idx] = 0.0
                    else:
                        self.target[c.idx] = left * self.spec.dose_kg_per_m2
        s_cur = float(self.s[i_path])
        stop = self.st.v**2 / (2 * self.vp.decel_mps2) + 1.0
        parcel = self.site.parcel_at(*ob.polygon.centroid.coords[0])
        x0, y0, x1, y1 = self.site.bounds
        allowed = parcel.polygon.buffer(3.0) if parcel else box(x0, y0, x1, y1)
        avoid = self.weed_no_go + self.detected_polys
        det = plan_detour(self.path, self.s, perceived, self.cfg.obstacle_clearance_m,
                          self.cfg.planner.headland_turn_radius_m, s_min=s_cur + stop, allowed=allowed,
                          avoid=avoid, step=self.cfg.planner.path_step_m)
        self.detected_polys.append(perceived.buffer(1.0))
        if det is None:
            ev["action"] = "monitor (clear of path)"
            return
        old_path = self.path[det.i0 : det.i1 + 1].copy()
        self.path, self.labels = splice(self.path, self.labels, det, "obstacle_detour", self.cfg.planner.path_step_m)
        self._refresh_path()
        self.ctrl.reset(max(0, i_path - 2))
        ev["action"] = f"bypass {det.side} +{det.offset_m:.1f} m"
        self.detours.append({"t": self.t, "oid": ob.oid, "kind": ob.kind, "pts": det.pts.copy(),
                             "replaced": old_path, "side": det.side, "offset_m": det.offset_m,
                             "length_m": det.length_m, "detect_range_m": dist})
        self.events.append({"t": self.t, "kind": "detour_spliced", "oid": ob.oid, "length_m": round(det.length_m, 1)})
        self._reproject_stations(s_cur)

    def _reproject_stations(self, s_cur: float) -> None:
        lane_mask = self.labels == "lane"
        for k, st in enumerate(self.plan.stations):
            if self.station_s[k] < s_cur - 0.5:
                continue
            idx = np.where(self.s >= s_cur)[0]
            d = np.hypot(self.path[idx, 0] - st.xy[0], self.path[idx, 1] - st.xy[1])
            j = int(idx[np.argmin(d)])
            if d.min() > 1.0 or not lane_mask[j]:
                cand = idx[lane_mask[idx]]
                if len(cand):
                    dd = np.hypot(self.path[cand, 0] - st.xy[0], self.path[cand, 1] - st.xy[1])
                    j = int(cand[np.argmin(dd)])
            self.station_s[k] = float(self.s[j])

    # ------------------------------------------------------------------ driving
    def _drive_to(self, s_stop_of: Callable[[], float]) -> None:
        dt = self.cfg.dt_drive
        log_every = max(1, int(round(1.0 / (self.cfg.log_hz * dt))))
        sense_every = max(1, int(round(self.cfg.sensor_period_s / dt)))
        k = 0
        t_start = self.t
        while True:
            s_stop = s_stop_of()
            steer_cmd, v_cmd, i, cte = self.ctrl.command(self.st, self.path, self.s, self.labels, self.kappa, s_stop)
            remaining = s_stop - self.s[i]
            if remaining < 0.3 and self.st.v < 0.05:
                self._log(steer_cmd, 0.0, cte, float(self.s[i]), LABEL_CODES.get(self.labels[i], 0))
                return
            if remaining > 0.05:
                v_cmd = max(v_cmd, 0.08)
            if k % sense_every == 0:
                self._sense(i)
            if k % log_every == 0:
                self._log(steer_cmd, v_cmd, cte, float(self.s[i]), LABEL_CODES.get(self.labels[i], 0))
            step_vehicle(self.st, steer_cmd, v_cmd, self.vp, dt, self.rng)
            self.t += dt
            k += 1
            if self.t - t_start > 4 * 3600:
                raise RuntimeError("drive leg did not converge")

    # ------------------------------------------------------------------ swarm
    def _patterns_for(self, cid: int) -> list[np.ndarray]:
        if cid not in self.patterns:
            c = self.plan.cells[cid]
            parcel = next(p for p in self.site.parcels if p.pid == c.parcel_id)
            n_needed = int(math.ceil(c.target_kg / self.spec.unit_mass_kg))
            self.patterns[cid] = cell_patches(c.polygon, c.e, c.n, self.spec.mode, parcel.polygon.bounds[0],
                                              parcel.bed_spacing_m, n_needed)
        return self.patterns[cid]

    def _tender_update(self, t: float) -> float:
        """Advance the tender-truck fleet to time t; returns earliest time a load can start."""
        cap = self.cfg.hopper_capacity_kg
        T = self.tender
        if T["pending"] and t >= T["arrive"] + self.cfg.tender_transfer_s:
            truck = T["fleet"][T["serving"]]
            give = min(cap - self.hopper, truck["load"])
            self.hopper += give
            truck["load"] -= give
            T["pending"] = False
            T["deliveries"] += 1
            self.events.append({"t": T["arrive"], "kind": "tender_topup", "kg": round(give, 1), "truck": T["serving"]})
            if truck["load"] < 0.5 * cap:  # head back to the depot to reload
                ms = self.st.centre(self.vp.wheelbase_m)
                dist = 1.3 * math.hypot(ms[0] - self.plan.depot_xy[0], ms[1] - self.plan.depot_xy[1])
                truck["ready"] = t + 2 * dist / self.cfg.tender_speed_mps + self.cfg.tender_depot_load_s
                truck["load"] = self.cfg.tender_capacity_kg
                T["trips"] += 1
                T["km"] += 2 * dist / 1000.0
        if not T["pending"] and self.hopper < self.cfg.tender_request_frac * cap:
            ms = self.st.centre(self.vp.wheelbase_m)
            dist = 1.3 * math.hypot(ms[0] - self.plan.depot_xy[0], ms[1] - self.plan.depot_xy[1])
            k = min(range(len(T["fleet"])), key=lambda i: T["fleet"][i]["ready"])
            T["serving"] = k
            T["pending"] = True
            T["arrive"] = max(t + 0.5 * dist / self.cfg.tender_speed_mps, T["fleet"][k]["ready"])
        if self.hopper < self.spec.unit_mass_kg and T["pending"]:
            ready = T["arrive"] + self.cfg.tender_transfer_s
            if ready > t:
                T["wait_s"] += ready - t
            return self._tender_update(max(t, ready))
        return t

    def _effector_split(self, cells: list[int], cx: float, cy: float, yaw: float):
        """Assign each cell's in-reach patches to the mothership effector workers.

        Returns (workers, ms_pats, drone_pats): workers = [(kind, side)], ms_pats[c] =
        {worker: [pattern idx]}, drone_pats[c] = [pattern idx] left for the swarm.
        """
        ep = self.cfg.effectors
        cart = self.spec.mode is PayloadMode.CARTRIDGE
        workers = [(fx.KIND_CANNON, 0), (fx.KIND_CANNON, 1)] if cart else [(fx.KIND_BLOWER, -1)]
        ms_xy = np.array([cx, cy])
        tip = fx.mount_world(cx, cy, yaw, fx.CHUTE_TIP)
        pivots = [fx.mount_world(cx, cy, yaw, pv) for pv in fx.CANNON_PIVOTS]
        ms_pats: dict[int, dict[int, list[int]]] = {}
        drone_pats: dict[int, list[int]] = {}
        for c in cells:
            pats = self._patterns_for(c)
            mine: dict[int, list[int]] = {}
            rest: list[int] = []
            for pi, pat in enumerate(pats):
                w = -1
                if ep.enabled:
                    if cart:
                        side = fx.side_of(ms_xy, yaw, pat.mean(axis=0))
                        if fx.reachable(pat, pivots[side], ep):
                            w = side
                    elif fx.reachable(pat, tip, ep):
                        w = 0
                if w >= 0:
                    mine.setdefault(w, []).append(pi)
                else:
                    rest.append(pi)
            ms_pats[c] = mine
            drone_pats[c] = rest
        return workers, ms_pats, drone_pats

    def _dwell(self, sid: int, drones: list[dict]) -> None:
        st = self.plan.stations[sid]
        eps = 0.05  # kg
        cells = [c for c in st.cells if not self.blocked[c] and self.target[c] - self.deposited[c] > eps]
        cx, cy = self.st.centre(self.vp.wheelbase_m)
        yaw = self.st.yaw
        t_arr = self.t
        visit = {"sid": sid, "t_arrive": t_arr, "x": cx, "y": cy, "yaw": yaw, "cells": list(st.cells)}
        if not cells:
            self.t += self.cfg.min_dwell_s
            visit["t_depart"] = self.t
            visit["sorties"] = 0
            visit["ms_bursts"] = 0
            self.visits.append(visit)
            return
        workers, ms_pats, drone_pats = self._effector_split(cells, cx, cy, yaw)
        rem_d: dict[int, float] = {}
        rem_m: dict[int, float] = {}
        for c in cells:
            rem = self.target[c] - self.deposited[c]
            n_all = len(self._patterns_for(c))
            n_ms = sum(len(v) for v in ms_pats[c].values())
            share = rem if not drone_pats[c] else rem * n_ms / n_all
            rem_m[c], rem_d[c] = share, rem - share
        n_dr = len(drones)
        claims: dict[int, int] = {}
        heap = [(max(t_arr, d["t_free"]) + 0.5 * di, di) for di, d in enumerate(drones)]
        heap += [(t_arr + 1.0, n_dr + w) for w in range(len(workers))]
        heapq.heapify(heap)
        last_land = t_arr
        n_sorties = 0
        n_bursts = 0
        fx_count: dict[tuple[int, int], int] = {}
        ep = self.cfg.effectors
        while heap:
            t_free, wid = heapq.heappop(heap)
            if wid >= n_dr:  # ---- mothership effector worker
                w = wid - n_dr
                kind, side = workers[w]
                az_key = side if kind == fx.KIND_CANNON else -1
                az = self.fx_az.get(az_key, yaw + (math.pi if kind == fx.KIND_BLOWER else (0.5 - side) * math.pi))
                options = [(c, pi) for c in cells if rem_m[c] > eps for pi in ms_pats[c].get(w, [])]
                if not options:
                    continue
                tipxy = (fx.mount_world(cx, cy, yaw, fx.CHUTE_TIP) if kind == fx.KIND_BLOWER
                         else fx.mount_world(cx, cy, yaw, fx.CANNON_PIVOTS[side]))

                def turn(o):
                    m = self._patterns_for(o[0])[o[1]].mean(axis=0)
                    return abs(fx.ang_diff(az, math.atan2(m[1] - tipxy[1], m[0] - tipxy[0])))
                # stay on the current cell while it needs material, sweeping its patches
                cur = [o for o in options if o[0] == fx_count.get((w, -1), -1)]
                pool = cur if cur else options
                cid = min(pool, key=turn)[0]
                cand = ms_pats[cid][w]
                kcount = fx_count.get((w, cid), 0)
                pi = cand[kcount % len(cand)]
                fx_count[(w, cid)] = kcount + 1
                fx_count[(w, -1)] = cid
                t0 = self._tender_update(t_free)
                pat = self._patterns_for(cid)[pi]
                if kind == fx.KIND_BLOWER:
                    kg = min(ep.blower_burst_kg, rem_m[cid])
                    bp = fx.plan_blower_burst(t0, np.array([cx, cy]), yaw, pat, kg, az, t0 - self.fx_t_idle, ep)
                    busy_until = bp.t_fire0 + kg / ep.blower_rate_kg_s
                    self.fx_t_idle = busy_until
                else:
                    kg = min(self.spec.unit_mass_kg, rem_m[cid])
                    bp = fx.plan_cannon_volley(t0, np.array([cx, cy]), yaw, pat, kg, az, side, ep)
                    busy_until = bp.t_fire0 + bp.units * fx.shot_gap(ep)
                self.hopper -= kg
                rem_m[cid] -= kg
                self.deposited[cid] += kg
                self.dep_rows.append((0.5 * (bp.t_fire0 + bp.t1), cid, kg))
                self.ms_rows.append((sid, cid, pi, kind, side, bp.t0, bp.t_fire0, bp.t1, busy_until, kg, bp.units,
                                     bp.az0, bp.az1, bp.aim_xy[0], bp.aim_xy[1], bp.range_m, cx, cy, yaw))
                self.fx_az[az_key] = bp.az1
                n_bursts += 1
                last_land = max(last_land, bp.t1)
                heapq.heappush(heap, (busy_until + 0.3, wid))
                continue
            di = wid
            d = drones[di]
            # choose / keep exclusive cell
            cid = d.get("cell", -1)
            if cid not in cells or rem_d.get(cid, 0.0) <= eps:
                if cid in claims:
                    claims.pop(cid, None)
                open_cells = [c for c in cells if rem_d[c] > eps]
                if not open_cells:
                    d["cell"] = -1
                    continue
                free = [c for c in open_cells if c not in claims]
                pool = free if free else open_cells
                cid = min(pool, key=lambda c: math.hypot(self.plan.cells[c].e - cx, self.plan.cells[c].n - cy))
                claims[cid] = di
                d["cell"] = cid
            # battery swap
            t0 = t_free
            if d["soc"] < self.dp.swap_threshold:
                self.events.append({"t": t0, "kind": "battery_swap", "drone": di, "soc": round(d["soc"], 3)})
                t0 += self.dp.swap_time_s
                d["soc"] = 1.0
                d["swaps"] += 1
            t0 = self._tender_update(t0)
            kg = min(self.spec.unit_mass_kg, rem_d[cid])
            self.hopper -= kg
            rem_d[cid] -= kg
            pats = self._patterns_for(cid)
            dp_idx = drone_pats[cid]
            pi = int(dp_idx[self.cell_sorties[cid] % len(dp_idx)])
            self.cell_sorties[cid] += 1
            dock = dock_xy(cx, cy, yaw, di)
            tm = sortie_timing(dock, pats[pi], self.spec, self.dp)
            t_launch = t0 + self.spec.load_time_s
            t_land = t_launch + tm.flight_time
            e_kwh = sortie_energy_kwh(tm, kg, self.dp)
            soc_before = d["soc"]
            d["soc"] -= e_kwh / self.dp.battery_kwh
            d["flight_s"] += tm.flight_time
            d["energy_kwh"] += e_kwh
            t_apply_mid = t_launch + tm.t_climb + tm.t_out + 0.5 * tm.t_apply
            self.deposited[cid] += kg
            self.dep_rows.append((t_apply_mid, cid, kg))
            self.sortie_rows.append((di, cid, sid, t0, t_launch, t_land, kg, pi, dock[0], dock[1],
                                     tm.flight_time, e_kwh, tm.out_dist + tm.back_dist + tm.apply_dist,
                                     t_launch + tm.t_climb + tm.t_out, t_launch + tm.t_climb + tm.t_out + tm.t_apply,
                                     soc_before, d["soc"]))
            n_sorties += 1
            last_land = max(last_land, t_land)
            d["t_free"] = t_land + self.dp.turnaround_s
            heapq.heappush(heap, (d["t_free"], di))
        self.t = max(last_land + 2.0, t_arr + self.cfg.min_dwell_s)
        visit["t_depart"] = self.t
        visit["sorties"] = n_sorties
        visit["ms_bursts"] = n_bursts
        self.visits.append(visit)

    # ------------------------------------------------------------------ run
    def run(self) -> MissionResult:
        drones = [{"t_free": 0.0, "soc": 1.0, "swaps": 0, "flight_s": 0.0, "energy_kwh": 0.0, "cell": -1}
                  for _ in range(self.cfg.swarm_size)]
        self.events.append({"t": 0.0, "kind": "depart_depot"})
        self._log(0.0, 0.0, 0.0, 0.0, LABEL_CODES["transit"])
        n_st = len(self.plan.stations)
        if self.cfg.max_stations is not None:
            n_st = min(n_st, self.cfg.max_stations)
        for k in range(n_st):
            self._drive_to(lambda k=k: self.station_s[k])
            self.events.append({"t": self.t, "kind": "arrive_station", "sid": k})
            self._sense(self.ctrl.idx)
            i = self.ctrl.idx
            self._log(0.0, 0.0, 0.0, float(self.s[i]), LABEL_CODES["station"])
            self._dwell(k, drones)
            self._log(0.0, 0.0, 0.0, float(self.s[i]), LABEL_CODES["station"])
            self.events.append({"t": self.t, "kind": "depart_station", "sid": k})
        if self.cfg.max_stations is None:
            self._drive_to(lambda: float(self.s[-1]))
            self.events.append({"t": self.t, "kind": "arrive_depot"})
        log = {k: np.asarray(v, dtype=float if k != "mode" else int) for k, v in self.log.items()}
        return self._result(log, drones)

    def _result(self, log: dict, drones: list[dict]) -> MissionResult:
        cols = ("drone", "cell", "station", "t_load", "t_launch", "t_land", "kg", "pattern", "dock_x", "dock_y",
                "flight_s", "energy_kwh", "dist_m", "apply_t0", "apply_t1", "soc0", "soc1")
        rows = np.array(self.sortie_rows, dtype=float) if self.sortie_rows else np.zeros((0, len(cols)))
        sorties = {c: rows[:, i] for i, c in enumerate(cols)}
        for c in ("drone", "cell", "station", "pattern"):
            sorties[c] = sorties[c].astype(int)
        ms_cols = ("station", "cell", "pattern", "kind", "side", "t0", "t_fire0", "t1", "t_free", "kg", "units",
                   "az0", "az1", "aim_x", "aim_y", "range_m", "ms_x", "ms_y", "ms_yaw")
        mrows = np.array(self.ms_rows, dtype=float) if self.ms_rows else np.zeros((0, len(ms_cols)))
        ms_shots = {c: mrows[:, i] for i, c in enumerate(ms_cols)}
        for c in ("station", "cell", "pattern", "kind", "side", "units"):
            ms_shots[c] = ms_shots[c].astype(int)
        dep = np.array(self.dep_rows, dtype=float) if self.dep_rows else np.zeros((0, 3))
        order = np.argsort(dep[:, 0], kind="stable")
        deposits = {"t": dep[order, 0], "cell": dep[order, 1].astype(int), "kg": dep[order, 2]}
        res = MissionResult(
            site=self.site, plan=self.plan, cfg=self.cfg, spec=self.spec, terrain=self.terrain, ms_log=log,
            active_path=self.path, active_labels=self.labels, detours=self.detours, events=self.events,
            visits=self.visits, sorties=sorties, patterns=self.patterns, deposits=deposits,
            cell_target_kg=self.target.copy(), cell_blocked=self.blocked.copy(), kpis={}, ms_shots=ms_shots,
        )
        res.kpis = compute_mission_kpis(res, drones, self.tender)
        return res


def run_mission(
    site: Site,
    twin: FieldTwin,
    cfg: MissionConfig | None = None,
    plan: GlobalPlan | None = None,
) -> MissionResult:
    return _Sim(site, twin, cfg or MissionConfig(), plan).run()


# ---------------------------------------------------------------------------
# KPIs
# ---------------------------------------------------------------------------


def tracking_stats(res: MissionResult, settle_m: float = 4.0) -> dict[str, float]:
    L = res.ms_log
    mode = L["mode"]
    cte = np.abs(L["cte"])
    moving = L["v"] > 0.05
    out: dict[str, float] = {}
    for name in ("lane", "headland", "transit", "weed_detour", "obstacle_detour"):
        m = (mode == LABEL_CODES[name]) & moving
        if name == "lane":
            # skip the settling stretch right after entering a lane
            entered = np.zeros_like(m)
            odo = L["odo"]
            start_odo = None
            for i in range(len(m)):
                if m[i] and (i == 0 or not m[i - 1]):
                    start_odo = odo[i]
                entered[i] = m[i] and start_odo is not None and odo[i] - start_odo > settle_m
            m = entered
        if m.any():
            out[f"cte_rms_{name}_m"] = float(np.sqrt(np.mean(cte[m] ** 2)))
            out[f"cte_max_{name}_m"] = float(cte[m].max())
    lane = (mode == LABEL_CODES["lane"]) & moving
    if lane.any():
        out["lane_within_furrow_tol_pct"] = float(100.0 * np.mean(cte[lane] <= 0.2))
    sat = np.abs(L["steer_cmd"]) >= res.cfg.vehicle.max_steer_rad - 1e-3
    out["steer_saturated_pct"] = float(100.0 * np.mean(sat[moving])) if moving.any() else 0.0
    return out


def compute_mission_kpis(res: MissionResult, drones: list[dict], tender: dict) -> dict[str, Any]:
    L = res.ms_log
    t_end = float(L["t"][-1])
    xy = np.column_stack([L["x"], L["y"]])
    dist = float(np.sum(np.hypot(*np.diff(xy, axis=0).T)))
    st_code = LABEL_CODES["station"]
    at_station = (L["mode"][:-1] == st_code) & (L["mode"][1:] == st_code)
    moving = float(np.diff(L["t"])[~at_station].sum()) if len(L["t"]) > 1 else 0.0
    S = res.sorties
    tgt = res.cell_target_kg
    deposited = np.zeros(len(tgt))
    np.add.at(deposited, res.deposits["cell"], res.deposits["kg"])
    valid = tgt > 0
    comp = np.clip(deposited[valid] / tgt[valid], 0, 1)
    area_m2 = float(sum(c.area_m2 for c, ok in zip(res.plan.cells, valid) if ok))
    dose_actual = deposited[valid] / np.array([c.area_m2 for c, ok in zip(res.plan.cells, valid) if ok])
    cv = float(np.std(dose_actual) / np.mean(dose_actual)) if len(dose_actual) and np.mean(dose_actual) > 0 else 0.0
    area_ha = area_m2 / 1e4
    total_t = float(deposited.sum() / 1000.0)
    dosed_ha = area_ha * min(1.0, float(deposited.sum()) / max(float(tgt[valid].sum()), 1e-9))

    # compaction: wheel ribbons leaving furrows inside bedded areas
    bedded = unary_union(list(res.plan.bedded_areas.values()))
    off_furrow_m = 0.0
    lab = L["mode"]
    for i in range(1, len(L["t"])):
        if lab[i] in (LABEL_CODES["obstacle_detour"], LABEL_CODES["weed_detour"], LABEL_CODES["headland"],
                      LABEL_CODES["transit"]):
            p = Point(L["x"][i], L["y"][i])
            if bedded.contains(p):
                off_furrow_m += math.hypot(L["x"][i] - L["x"][i - 1], L["y"][i] - L["y"][i - 1])
    compaction_m2 = off_furrow_m * 4 * 0.30  # 4 wheels x 0.30 m tyre width

    flight_h = float(S["flight_s"].sum() / 3600.0) if len(S["flight_s"]) else 0.0
    n_sorties = int(len(S["kg"]))
    mission_h = t_end / 3600.0
    mode_cart = res.spec.mode is PayloadMode.CARTRIDGE
    M = res.ms_shots
    ms_kg = float(M["kg"].sum()) if len(M.get("kg", [])) else 0.0
    cost = _cost_per_1000m2(res, mission_h, area_m2, flight_h, dist, tender)

    k: dict[str, Any] = {
        "payload_mode": res.spec.mode.value,
        "dose_kg_per_m2": res.spec.dose_kg_per_m2,
        "swarm_size": res.cfg.swarm_size,
        "mission_time_h": round(mission_h, 2),
        "mission_days_10h_shifts": round(mission_h / 10.0, 1),
        "ms_drive_time_h": round(moving / 3600.0, 2),
        "ms_distance_km": round(dist / 1000.0, 3),
        "planned_distance_km": round(res.plan.length_m / 1000.0, 3),
        "stations": len(res.visits),
        "area_treated_ha": round(area_ha, 3),
        "biomass_applied_t": round(total_t, 2),
        "mean_completion": round(float(comp.mean()) if len(comp) else 0.0, 4),
        "cells_complete_pct": round(100.0 * float(np.mean(comp >= 0.999)) if len(comp) else 0.0, 2),
        "dose_cv": round(cv, 4),
        "co2e_t": round(dosed_ha * CO2E_T_PER_HA * (res.spec.dose_kg_per_m2 / 15.0), 1),
        "n_retained_kg": round(dosed_ha * N_RETAINED_KG_PER_HA * (res.spec.dose_kg_per_m2 / 15.0), 1),
        "sorties": n_sorties,
        "drone_flight_h": round(flight_h, 2),
        "drone_flight_km": round(float(S["dist_m"].sum() / 1000.0) if n_sorties else 0.0, 2),
        "mean_sortie_s": round(float(S["flight_s"].mean()) if n_sorties else 0.0, 1),
        "mean_sortie_m": round(float(S["dist_m"].mean()) if n_sorties else 0.0, 1),
        "battery_swaps": int(sum(d["swaps"] for d in drones)),
        "drone_energy_kwh": round(float(sum(d["energy_kwh"] for d in drones)), 1),
        "drone_utilisation_pct": round(100.0 * flight_h / max(mission_h * res.cfg.swarm_size, 1e-9), 1),
        "swarm_throughput_t_per_h": round(total_t / max(mission_h, 1e-9), 2),
        "ms_effector": "2 pneumatic cartridge air cannons" if mode_cart else "snow-blower chip chute",
        "ms_effector_reach_m": round(res.cfg.effectors.range_max_m, 2),
        "ms_effector_t": round(ms_kg / 1000.0, 2),
        "ms_effector_share_pct": round(100.0 * ms_kg / max(deposited.sum(), 1e-9), 1),
        "ms_effector_bursts": int(len(M.get("kg", []))),
        "ms_effector_units": int(M["units"].sum()) if len(M.get("kg", [])) else 0,
        "ms_effector_busy_h": round(float((M["t_free"] - M["t0"]).sum()) / 3600.0, 2) if len(M.get("kg", [])) else 0.0,
        "tender_trips": int(tender["trips"]),
        "tender_topups": int(tender["deliveries"]),
        "tender_km": round(float(tender["km"]), 2),
        "tender_wait_h": round(float(tender["wait_s"]) / 3600.0, 2),
        "off_furrow_in_beds_m": round(off_furrow_m, 1),
        "bed_compaction_m2": round(compaction_m2, 1),
        "bed_compaction_pct_of_area": round(100.0 * compaction_m2 / max(area_m2, 1), 3),
        "obstacles_detected": [
            {k2: e[k2] for k2 in ("oid", "obstacle_type", "range_m", "action") if k2 in e} | {"t_h": round(e["t"] / 3600, 3)}
            for e in res.events if e["kind"] == "obstacle_detected"
        ],
        "detour_m": round(float(sum(d["length_m"] for d in res.detours)), 1),
        "robot_ops_eur_per_1000m2": round(cost["ops"], 2),
        "cost_eur_per_1000m2": round(cost["total"], 2),
    }
    if not mode_cart:
        k["regensol_baseline_eur_per_1000m2"] = REGENSOL_COST_EUR_PER_1000M2
        k["conventional_eur_per_1000m2"] = CONVENTIONAL_COST_EUR_PER_1000M2
        k["saving_vs_regensol_eur_per_1000m2"] = round(REGENSOL_COST_EUR_PER_1000M2 - cost["total"], 2)
        k["overhead_recovered_pct"] = round(
            100.0 * (REGENSOL_COST_EUR_PER_1000M2 - cost["total"]) / REGENSOL_OVERHEAD_EUR_PER_1000M2, 1)
    else:
        k["cartridge_material_eur_per_1000m2"] = round(cost["material"], 2)
    if mode_cart:
        units = int(round(total_t * 1000.0 / (res.spec.unit_mass_kg / res.spec.units_per_sortie)))
        k["cartridges_placed"] = units
        k["seeds_placed"] = units * res.spec.seeds_per_unit
        k["species"] = list(res.spec.local_tree_species)
    k.update({kk: round(v, 4) for kk, v in tracking_stats(res).items()})
    return k


def _cost_per_1000m2(res: MissionResult, mission_h: float, area_m2: float, flight_h: float, dist_m: float,
                     tender: dict) -> dict[str, float]:
    """Cost model (EUR / 1000 m2) in the scope of the REGENSOL baseline.

    REGENSOL (2,408.77) = conventional (1,895.56) + regenerative overhead (513.21);
    27.7 % of it is biomass transport + spreading, which the robot fleet replaces.
    Robot operations: mothership amortisation/energy, drone flight hours, blower /
    air-cannon busy hours, a remote supervisor shared across 4 fleets, tender
    truck km + depot loading.
    """
    if area_m2 <= 0:
        return {"ops": float("nan"), "total": float("nan")}
    units = area_m2 / 1000.0
    ms_eur_h, drone_eur_flight_h, supervisor_eur_h = 38.0, 22.0, 28.0 / 4.0
    tender_eur_km, tender_eur_trip = 1.9, 45.0
    M = res.ms_shots
    fx_busy_h = float((M["t_free"] - M["t0"]).sum()) / 3600.0 if len(M.get("kg", [])) else 0.0
    fx_eur_h = 9.0  # impeller / compressor energy + wear
    ops = (ms_eur_h * mission_h + drone_eur_flight_h * flight_h + supervisor_eur_h * mission_h
           + fx_eur_h * fx_busy_h + tender_eur_km * tender["km"] + tender_eur_trip * tender["trips"]) / units
    if res.spec.mode is PayloadMode.CARTRIDGE:
        cartridges_per_1000m2 = 1000.0 * res.spec.dose_kg_per_m2 / (res.spec.unit_mass_kg / res.spec.units_per_sortie)
        material = 0.22 * cartridges_per_1000m2  # pressed chips + seed + inoculum, EUR/unit
        return {"ops": ops, "material": material, "total": ops + material}
    replaced = TRANSPORT_SPREAD_SHARE * REGENSOL_COST_EUR_PER_1000M2
    total = REGENSOL_COST_EUR_PER_1000M2 - replaced + ops
    return {"ops": ops, "replaced_transport_spread": replaced, "total": total}
