"""Global route planner + time-based mission (vehicle control, lidar detours, swarm)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import LineString, Point

from prunemaxxing.geo.dem import load_dem
from prunemaxxing.geo.grid import build_field_twin_from_dem, crop_target_field, select_pilot_window
from prunemaxxing.geo.site import load_site
from prunemaxxing.planning import ms_effectors as fx
from prunemaxxing.planning.mission import LABEL_CODES, MissionConfig, run_mission
from prunemaxxing.planning.path_geometry import curvature, fillet_polyline, pieces_to_path
from prunemaxxing.planning.route_planner import plan_global_route

ROOT = Path(__file__).resolve().parents[1]
DEM = ROOT / "viladecans_full_dem_10m.TIF"


@pytest.fixture(scope="module")
def site_twin():
    if not DEM.exists():
        pytest.skip("DEM missing")
    site = load_site()
    twin = build_field_twin_from_dem(select_pilot_window(crop_target_field(load_dem(DEM)), 12.0))
    return site, twin


@pytest.fixture(scope="module")
def plan(site_twin):
    return plan_global_route(*site_twin)


def test_fillet_respects_radius():
    pts = np.array([[0, 0], [0, 30], [40, 30], [40, 0]], dtype=float)
    path, _ = pieces_to_path(fillet_polyline(pts, ["a", "b", "c"], 6.5), 0.25)
    k = np.abs(curvature(path))
    assert np.percentile(k, 99) <= 1 / 6.5 + 0.02


def test_route_has_no_spiral_and_returns_to_depot(plan):
    labels = set(plan.labels.tolist())
    assert labels <= {"transit", "lane", "headland", "weed_detour"}
    assert np.allclose(plan.path[0], plan.path[-1], atol=0.5)
    assert len(plan.lanes) == 12
    assert len(plan.stations) >= 40
    assert not plan.unreachable_cells


def test_route_avoids_mapped_weeds_and_turns_feasibly(site_twin, plan):
    site, _ = site_twin
    line = LineString(plan.path)
    for w in site.weeds:
        assert not line.intersects(w.polygon.buffer(2.0)), w.zid
    k = np.abs(curvature(plan.path))
    assert np.percentile(k, 99.5) <= 1 / 6.0 + 0.02


def test_lanes_are_furrow_centred(site_twin, plan):
    site, _ = site_twin
    for lane in plan.lanes:
        parcel = next(p for p in site.parcels if p.pid == lane.parcel_id)
        k = (lane.x - parcel.polygon.bounds[0]) / parcel.bed_spacing_m - 0.5
        assert abs(k - round(k)) < 1e-6


@pytest.fixture(scope="module")
def short_mission(site_twin, plan):
    site, twin = site_twin
    return run_mission(site, twin, MissionConfig(max_stations=14), plan=plan)


def test_mothership_actually_translates_and_logs_commands(short_mission):
    L = short_mission.ms_log
    xy = np.column_stack([L["x"], L["y"]])
    travelled = float(np.sum(np.hypot(*np.diff(xy, axis=0).T)))
    assert travelled > 300.0
    assert np.ptp(L["x"]) > 50 and np.ptp(L["y"]) > 100
    for key in ("v_cmd", "steer_cmd", "v", "steer", "cte"):
        assert key in L and len(L[key]) == len(L["t"])
    assert np.abs(L["steer_cmd"]).max() > 0.2  # headland turns commanded
    assert np.all(np.diff(L["t"]) >= -1e-9)


def test_lane_tracking_within_furrow_tolerance(short_mission):
    k = short_mission.kpis
    assert k["cte_rms_lane_m"] < 0.05
    assert k["cte_max_lane_m"] < 0.2
    assert k["lane_within_furrow_tol_pct"] >= 99.0


def test_hidden_obstacle_detected_and_bypassed(site_twin, short_mission):
    site, _ = site_twin
    det = [e for e in short_mission.events if e["kind"] == "obstacle_detected"]
    assert any(e["oid"] == "unk_tractor_A" for e in det)
    assert short_mission.detours, "expected an online detour splice"
    L = short_mission.ms_log
    tractor = next(o for o in site.obstacles if o.oid == "unk_tractor_A").polygon
    dmin = min(tractor.distance(Point(x, y)) for x, y in zip(L["x"], L["y"]))
    assert dmin > 1.2  # vehicle half-width + margin
    # actual differs from the global plan around the obstacle
    plan_line = LineString(short_mission.plan.path)
    dev = max(plan_line.distance(Point(x, y)) for x, y in zip(L["x"], L["y"]))
    assert dev > 3.0
    assert (L["mode"] == LABEL_CODES["obstacle_detour"]).any()


def test_swarm_sorties_are_spread_and_timed(short_mission):
    S = short_mission.sorties
    assert len(S["kg"]) > 1000
    assert set(np.unique(S["drone"])) == {0, 1, 2, 3}
    assert S["flight_s"].min() > 8.0  # no instant teleports
    assert np.all(S["t_land"] > S["t_launch"])
    # the swarm worked several stations, not just one corner
    assert len(np.unique(S["station"])) >= 10
    # cells of finished stations are at dose
    done = [c for v in short_mission.visits for c in v["cells"]]
    comp = short_mission.completion_at([short_mission.t_end])[0]
    vals = comp[done]
    assert np.nanmin(vals) > 0.99


def test_drone_state_query_moves(short_mission):
    S = short_mission.sorties
    k = int(len(S["kg"]) // 2)
    t_mid = 0.5 * (S["t_launch"][k] + S["t_land"][k])
    d = int(S["drone"][k])
    a = short_mission.drone_state_at(d, t_mid)
    b = short_mission.drone_state_at(d, t_mid + 2.0)
    assert a["phase"] != 0
    assert np.hypot(a["x"] - b["x"], a["y"] - b["y"]) > 0.5


def test_drone_heading_is_continuous_and_lands_aligned(short_mission):
    S = short_mission.sorties
    k = int(len(S["kg"]) // 3)
    d = int(S["drone"][k])
    ts = np.arange(S["t_load"][k] - 1.0, S["t_land"][k] + 1.0, 0.05)
    yaw = np.unwrap([short_mission.drone_state_at(d, t)["yaw"] for t in ts])
    assert np.abs(np.diff(yaw)).max() < 0.15  # < 3 rad/s, no snaps
    ms_yaw = short_mission.ms_state_at(S["t_land"][k])["yaw"]
    assert abs(np.angle(np.exp(1j * (yaw[-1] - ms_yaw)))) < 0.05


def test_ballistic_arc_hits_target_within_40ft():
    p0 = np.array([0.0, 0.0, 2.6])
    p1 = np.array([fx.REACH_M, 0.0, 0.2])
    pts, T = fx.ballistic(p0, p1, np.radians(30.0))
    assert np.allclose(pts[-1], p1, atol=1e-6)
    assert pts[:, 2].max() > p0[2]  # lofted, not a straight line
    assert 0.5 < T < 3.0
    assert fx.launch_speed(fx.REACH_M, 2.4, np.radians(30.0)) < 15.0  # low-pressure pneumatic range


def test_ms_blower_shares_work_within_reach(short_mission):
    M = short_mission.ms_shots
    k = short_mission.kpis
    assert len(M["kg"]) > 20 and k["ms_effector_share_pct"] > 3.0
    assert set(np.unique(M["kind"])) == {fx.KIND_BLOWER}
    assert M["range_m"].max() <= fx.REACH_M + 1e-6
    assert M["range_m"].min() >= short_mission.cfg.effectors.range_min_m - 1e-6
    # bursts only while parked at a station, never overlapping on the single chute
    order = np.argsort(M["t0"])
    assert np.all(M["t0"][order][1:] >= M["t_free"][order][:-1] - 1e-6)
    st = [s for s in (short_mission.ms_effector_state(t) for t in M["t_fire0"][:40] + 1.0) if s[0]["firing"]]
    assert st and all(s[0]["aim"] is not None for s in st)


@pytest.fixture(scope="module")
def cartridge_mission(site_twin, plan):
    site, twin = site_twin
    return run_mission(site, twin, MissionConfig(payload_mode="cartridge", max_stations=6), plan=plan)


def test_ms_air_cannons_fire_both_sides_and_land_on_target(cartridge_mission):
    res = cartridge_mission
    M = res.ms_shots
    assert set(np.unique(M["kind"])) == {fx.KIND_CANNON}
    assert set(np.unique(M["side"])) == {0, 1}
    assert M["range_m"].max() <= fx.REACH_M + 1e-6
    k = int(np.argmax(M["units"]))
    v = res.ms_volley(k)
    assert len(v["t_fire"]) == int(M["units"][k])
    assert set(np.unique(v["barrel"])) == {0, 1}  # twin barrels alternate
    ms_xy = np.array([M["ms_x"][k], M["ms_y"][k]])
    assert np.hypot(*(v["land"][:, :2] - ms_xy).T).max() <= fx.REACH_M + 0.5
    st = res.ms_effector_state(float(v["t_fire"][1]) + 0.2)
    side = int(M["side"][k])
    assert st[side]["firing"] and len(st[side]["pellets"]) >= 1
    comp = res.completion_at([res.t_end])[0]
    done = [c for vis in res.visits for c in vis["cells"]]
    assert np.nanmin(comp[done]) > 0.99
