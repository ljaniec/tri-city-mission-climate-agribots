"""Fast closed-loop MuJoCo test of the Ackermann mothership rover (flat terrain)."""

from __future__ import annotations

import csv
import math
import time

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from prunemaxxing.sim.rover_physics import PathBuilder, ReferencePath, RoverParams, run_closed_loop  # noqa: E402


def _test_path():
    pb = PathBuilder(0.0, 0.0, 0.0, ds=0.25)
    pb.straight(30.0, "lane").arc(6.5, math.pi / 2, "headland").straight(15.0, "lane")
    return pb.build()


def test_reference_path_sign_and_curvature():
    xy, labels = _test_path()
    ref = ReferencePath(xy, labels)
    assert ref.length == pytest.approx(30 + 6.5 * math.pi / 2 + 15, abs=0.05)
    s, cte = ref.project(np.array([10.0, 0.3]))
    assert s == pytest.approx(10.0, abs=1e-6) and cte == pytest.approx(0.3, abs=1e-6)  # + = left
    mid_arc = ref.kappa[(ref.s > 33) & (ref.s < 38)]
    assert np.allclose(mid_arc, 1 / 6.5, rtol=0.02)


def test_closed_loop_flat_straight_arc_straight(tmp_path):
    xy, labels = _test_path()
    t0 = time.perf_counter()
    res = run_closed_loop(xy, v_max=1.4, terrain="flat", segment_labels=labels)
    wall = time.perf_counter() - t0
    s = res.summary

    assert wall < 20.0
    assert s["completed"], s["end_reason"]
    assert s["model"]["total_mass_kg"] == pytest.approx(RoverParams().total_mass, rel=1e-6)
    assert s["rms_cte"] < 0.15
    assert s["max_cte"] < 0.4

    # really translated along the path (guards against rotate-in-place bugs)
    assert s["distance_travelled_m"] > 40.0
    displacement = math.hypot(res.x[-1] - res.x[0], res.y[-1] - res.y[0])
    assert displacement > 35.0
    assert math.hypot(res.x[-1] - xy[-1, 0], res.y[-1] - xy[-1, 1]) < 0.6
    assert abs(math.atan2(math.sin(res.yaw[-1] - math.pi / 2), math.cos(res.yaw[-1] - math.pi / 2))) < 0.1

    # steering and speed commands are logged and the physics followed them
    n = len(res.t)
    for arr in (res.v_cmd, res.v, res.steer_cmd, res.steer, res.cross_track, res.s_along):
        assert arr.shape == (n,)
    assert res.wheel_omega.shape == (n, 4)
    assert res.v_cmd.max() > 1.0 and res.v.max() > 1.0
    assert np.max(res.steer_cmd) > 0.25  # left turn on the R6.5 arc
    assert np.max(res.steer) > 0.25
    assert np.max(res.wheel_omega) > 1.0 / RoverParams().wheel_radius
    assert s["physics_health"]["pct_ticks_all_4_wheels_in_contact"] > 99.0

    out = res.to_csv(tmp_path / "log.csv")
    with out.open() as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == ["t", "x", "y", "z", "yaw", "v", "v_cmd", "steer", "steer_cmd", "cross_track", "s", "label"]
    assert len(rows) == n + 1
    assert {r[-1] for r in rows[1:]} == {"lane", "headland"}
