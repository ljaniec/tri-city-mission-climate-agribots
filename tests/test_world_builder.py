"""Tests for the high-fidelity MuJoCo world (prunemaxxing.sim.world_builder)."""

from __future__ import annotations

import ctypes.util
import os
import time
from pathlib import Path

import numpy as np
import pytest

_EGL_VENDOR = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
# MUJOCO_GL is read at `import mujoco`; without libEGL the egl backend fails to import,
# so fall back to no GL and let the rendering tests skip.
os.environ.setdefault("MUJOCO_GL", "egl" if ctypes.util.find_library("EGL") else "disable")
if Path(_EGL_VENDOR).exists():
    os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", _EGL_VENDOR)

mujoco = pytest.importorskip("mujoco")

ROOT = Path(__file__).resolve().parents[1]
DEM = ROOT / "viladecans_full_dem_10m.TIF"

pytestmark = pytest.mark.skipif(not DEM.exists(), reason="DEM not present")

SWARM = 4
OBJ = mujoco.mjtObj


def _id(model, obj, name: str) -> int:
    return mujoco.mj_name2id(model, obj, name)


@pytest.fixture(scope="module")
def spec():
    from prunemaxxing.sim.world_builder import build_world

    return build_world(payload_mode="dust", swarm_size=SWARM, verbose=False)


@pytest.fixture(scope="module")
def model(spec):
    from prunemaxxing.sim.world_builder import load_world_model

    return load_world_model(spec)


@pytest.fixture()
def fresh_model(spec):
    from prunemaxxing.sim.world_builder import load_world_model

    return load_world_model(spec)


def _renderer_or_skip(model, height=180, width=320):
    from prunemaxxing.sim.world_builder import make_renderer

    try:
        return make_renderer(model, height=height, width=width)
    except Exception as exc:  # no EGL / GPU context available
        pytest.skip(f"EGL rendering unavailable: {exc}")


def _render(model, data, renderer, camera: str) -> np.ndarray:
    from prunemaxxing.sim.world_builder import focus_shadows

    focus_shadows(renderer, model, data)
    renderer.update_scene(data, camera=camera)
    return renderer.render()


# ---------------------------------------------------------------------------
# Build / cache
# ---------------------------------------------------------------------------


def test_build_is_cached_and_deterministic(spec):
    from prunemaxxing.sim.world_builder import build_world

    xml_before = spec.xml_path.read_text()
    t0 = time.perf_counter()
    again = build_world(payload_mode="dust", swarm_size=SWARM, verbose=False)
    elapsed = time.perf_counter() - t0
    assert again.meta["asset_key"] == spec.meta["asset_key"]
    assert again.xml_path == spec.xml_path
    assert again.xml_path.read_text() == xml_before
    assert elapsed < 10.0, f"cached build took {elapsed:.1f}s"
    for f in spec.meta["asset_files"]:
        assert (spec.out_dir / f).exists(), f
    assert (spec.out_dir / "world_meta.json").exists()


def test_invalid_payload_mode_rejected():
    from prunemaxxing.sim.world_builder import build_world

    with pytest.raises(ValueError):
        build_world(payload_mode="gravel", verbose=False)


def test_spec_frame_and_cells(spec):
    assert spec.half_x == pytest.approx(290.0)
    assert spec.half_y == pytest.approx(285.0)
    assert spec.h_max > spec.h_min
    cg = spec.cells
    assert cg.work.shape == (cg.nrows, cg.ncols)
    assert cg.work.any()
    fx0, fy0, fx1, fy1 = spec.field_box
    centers = cg.centers()[cg.work]
    assert np.all((centers[:, 0] > fx0) & (centers[:, 0] < fx1))
    assert np.all((centers[:, 1] > fy0) & (centers[:, 1] < fy1))


# ---------------------------------------------------------------------------
# Named entities
# ---------------------------------------------------------------------------

REQUIRED_BODIES = (
    ["mothership"]
    + [f"drone{i}" for i in range(SWARM)]
    + [f"drone{i}_cradle" for i in range(SWARM)]
    + [f"fx_dust_{k}" for k in range(200)]
    + [f"fx_cart_{k}" for k in range(40)]
)
REQUIRED_JOINTS = (
    ["ms_free", "ms_wheel_fl", "ms_wheel_fr", "ms_wheel_rl", "ms_wheel_rr", "ms_steer_fl", "ms_steer_fr"]
    + [f"drone{i}_free" for i in range(SWARM)]
    + [f"drone{i}_lift" for i in range(SWARM)]
    + [f"drone{i}_rotor{k}" for i in range(SWARM) for k in range(6)]
)
REQUIRED_CAMERAS = ["cam_overview", "cam_top", "cam_low_field", "ms_front", "ms_mast", "ms_side"] + [
    f"drone{i}_cam" for i in range(SWARM)
]
REQUIRED_SITES = ["ms_lidar"] + [f"ms_dock_{k}" for k in range(4)]


def test_required_names_exist(model):
    missing = []
    for obj, names in (
        (OBJ.mjOBJ_BODY, REQUIRED_BODIES),
        (OBJ.mjOBJ_JOINT, REQUIRED_JOINTS),
        (OBJ.mjOBJ_CAMERA, REQUIRED_CAMERAS),
        (OBJ.mjOBJ_SITE, REQUIRED_SITES),
    ):
        missing += [f"{obj.name}:{n}" for n in names if _id(model, obj, n) < 0]
    assert not missing, missing
    assert _id(model, OBJ.mjOBJ_GEOM, "terrain") >= 0
    assert _id(model, OBJ.mjOBJ_TEXTURE, "tex_field") >= 0


def test_spec_names_resolve(spec, model):
    kinds = {"bodies": OBJ.mjOBJ_BODY, "joints": OBJ.mjOBJ_JOINT, "cameras": OBJ.mjOBJ_CAMERA,
             "sites": OBJ.mjOBJ_SITE, "mocap": OBJ.mjOBJ_BODY, "lights": OBJ.mjOBJ_LIGHT,
             "tendons": OBJ.mjOBJ_TENDON}
    for key, obj in kinds.items():
        bad = [n for n in spec.names[key] if _id(model, obj, n) < 0]
        assert not bad, (key, bad)
    assert spec.names["obstacles"], "site defines obstacles"


def test_obstacle_models_and_marks_exist(spec, model):
    geom_names = [mujoco.mj_id2name(model, OBJ.mjOBJ_GEOM, i) or "" for i in range(model.ngeom)]
    for oid in spec.names["obstacles"]:
        assert any(n.startswith(f"obs_{oid}_") for n in geom_names), f"no model geoms for {oid}"
        marks = [i for i, n in enumerate(geom_names) if n.startswith(f"obs_mark_{oid}")]
        assert marks, f"no highlight marks for {oid}"
        assert np.all(model.geom_rgba[marks, 3] == 0.0), f"marks for {oid} must start hidden"


def test_joint_types_axes_and_ranges(model):
    jt = mujoco.mjtJoint

    def joint(name):
        j = _id(model, OBJ.mjOBJ_JOINT, name)
        return int(model.jnt_type[j]), model.jnt_axis[j], model.jnt_range[j]

    assert joint("ms_free")[0] == jt.mjJNT_FREE
    for k in ("fl", "fr", "rl", "rr"):
        t, axis, _ = joint(f"ms_wheel_{k}")
        assert t == jt.mjJNT_HINGE
        np.testing.assert_allclose(np.abs(axis), [0, 1, 0], atol=1e-9)
    for k in ("fl", "fr"):
        t, axis, _ = joint(f"ms_steer_{k}")
        assert t == jt.mjJNT_HINGE
        np.testing.assert_allclose(np.abs(axis), [0, 0, 1], atol=1e-9)
    for i in range(SWARM):
        assert joint(f"drone{i}_free")[0] == jt.mjJNT_FREE
        t, _, rng = joint(f"drone{i}_lift")
        assert t == jt.mjJNT_SLIDE
        np.testing.assert_allclose(rng, [-1.5, 0.0])
        for k in range(6):
            assert joint(f"drone{i}_rotor{k}")[0] == jt.mjJNT_HINGE


def test_cam_top_is_orthographic(model):
    from prunemaxxing.sim.world_builder import CAM_TOP_EXTENT

    cid = _id(model, OBJ.mjOBJ_CAMERA, "cam_top")
    assert int(model.cam_projection[cid]) == int(mujoco.mjtProjection.mjPROJ_ORTHOGRAPHIC)
    assert model.cam_fovy[cid] == pytest.approx(CAM_TOP_EXTENT)


def test_fx_mocap_parked(model):
    from prunemaxxing.sim.world_builder import fx_mocap_ids

    data = mujoco.MjData(model)
    dust, cart = fx_mocap_ids(model, "dust"), fx_mocap_ids(model, "cartridge")
    assert len(set(dust)) == 200 and len(set(cart)) == 40
    assert np.all(dust >= 0) and np.all(cart >= 0)
    np.testing.assert_allclose(data.mocap_pos[np.r_[dust, cart], 2], -50.0)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def test_mothership_geometry(spec, model):
    from prunemaxxing.sim.world_builder import place_mothership

    data = mujoco.MjData(model)
    x, y = spec.furrow_x(spec.beds[0]["minx"] + 29.3), -121.5
    place_mothership(model, data, spec, x, y, yaw=np.pi / 2, follow_terrain=False)
    mujoco.mj_forward(model, data)
    body = lambda n: data.xpos[_id(model, OBJ.mjOBJ_BODY, n)]  # noqa: E731
    origin = body("mothership")
    assert origin[2] == pytest.approx(spec.terrain.height_at(x, y), abs=1e-6)
    w = {k: body(f"ms_wheel_{k}") - origin for k in ("fl", "fr", "rl", "rr")}
    # heading north: +x body axis -> +y world
    assert w["fl"][1] - w["rl"][1] == pytest.approx(2.4, abs=1e-6)
    assert w["rl"][0] - w["rr"][0] == pytest.approx(-1.7, abs=1e-6)
    for v in w.values():
        assert v[2] == pytest.approx(0.45, abs=1e-6)
    mass = model.body_subtreemass[_id(model, OBJ.mjOBJ_BODY, "mothership")]
    assert 3500.0 < mass < 4500.0
    lidar = data.site_xpos[_id(model, OBJ.mjOBJ_SITE, "ms_lidar")]
    assert lidar[2] - origin[2] > 2.5


def test_mothership_effectors_pose_and_fx_pools(spec, model):
    from prunemaxxing.planning import ms_effectors as fx
    from prunemaxxing.sim import world_effectors as we
    from prunemaxxing.sim.world_builder import place_mothership

    data = mujoco.MjData(model)
    x, y = spec.furrow_x(spec.beds[0]["minx"] + 29.3), -121.5
    place_mothership(model, data, spec, x, y, yaw=0.0, follow_terrain=False)
    adr = lambda n: model.jnt_qposadr[_id(model, OBJ.mjOBJ_JOINT, n)]  # noqa: E731
    site = lambda n: data.site_xpos[_id(model, OBJ.mjOBJ_SITE, n)].copy()  # noqa: E731
    mujoco.mj_forward(model, data)
    origin = data.xpos[_id(model, OBJ.mjOBJ_BODY, "mothership")].copy()
    tip = site("ms_chute_tip") - origin
    assert tip[0] > 0.3 and abs(tip[1]) < 0.05  # chute yaw 0 throws forward
    assert tip[2] == pytest.approx(fx.CHUTE_TIP[2], abs=0.25)
    data.qpos[adr("ms_chute_yaw")] = np.pi / 2
    for s in (0, 1):
        data.qpos[adr(f"ms_cannon{s}_yaw")] = (0.5 - s) * np.pi  # port -> +y, starboard -> -y
        data.qpos[adr(f"ms_cannon{s}_pitch")] = np.radians(30.0)
    mujoco.mj_forward(model, data)
    tip = site("ms_chute_tip") - origin
    assert tip[1] > 0.3 and abs(tip[0]) < 0.05
    for s in (0, 1):
        piv = np.array(fx.CANNON_PIVOTS[s])
        m = site(f"ms_cannon{s}_muzzle0") - origin
        assert np.sign(piv[1]) * m[1] == pytest.approx(abs(piv[1]) + fx.CANNON_BARREL_M * np.cos(np.radians(30)),
                                                       abs=0.03)
        assert m[2] == pytest.approx(piv[2] + fx.CANNON_BARREL_M * np.sin(np.radians(30)), abs=0.05)
    data = mujoco.MjData(model)
    for prefix, n in (("fx_blow_", we.N_FX_BLOW), ("fx_bdust_", we.N_FX_BDUST), ("fx_pellet_", we.N_FX_PELLET),
                      ("fx_trail_", we.N_FX_TRAIL), ("fx_puff_", we.N_FX_PUFF)):
        ids = [model.body_mocapid[_id(model, OBJ.mjOBJ_BODY, f"{prefix}{k}")] for k in range(n)]
        assert min(ids) >= 0
        np.testing.assert_allclose(data.mocap_pos[ids, 2], -50.0)


def test_drone_rotors_and_cradle(model):
    from prunemaxxing.sim.world_builder import place_drone

    data = mujoco.MjData(model)
    place_drone(model, data, 0, 0.0, 0.0, 10.0, lift=-1.2)
    mujoco.mj_forward(model, data)
    d0 = data.xpos[_id(model, OBJ.mjOBJ_BODY, "drone0")]
    cradle = data.xpos[_id(model, OBJ.mjOBJ_BODY, "drone0_cradle")]
    assert d0[2] - cradle[2] > 1.2
    rotors = np.array([data.xpos[_id(model, OBJ.mjOBJ_BODY, f"drone0_rotor{k}_body")] for k in range(6)])
    radii = np.linalg.norm(rotors[:, :2] - d0[:2], axis=1)
    # 2.2 m hexacopter: rotor hubs on a ~0.75 m arm radius, all equal
    assert np.ptp(radii) < 1e-6
    assert 0.6 < radii.mean() < 1.0


def test_hfield_data_matches_terrain_grid(spec, model):
    gid = _id(model, OBJ.mjOBJ_GEOM, "terrain")
    assert int(model.geom_type[gid]) == int(mujoco.mjtGeom.mjGEOM_HFIELD)
    hid = model.geom_dataid[gid]
    nrow, ncol = int(model.hfield_nrow[hid]), int(model.hfield_ncol[hid])
    assert (nrow, ncol) == spec.terrain.heights.shape
    sx, sy, sz, _ = model.hfield_size[hid]
    assert (sx, sy) == pytest.approx((spec.half_x, spec.half_y))
    adr = int(model.hfield_adr[hid])
    h = model.hfield_data[adr : adr + nrow * ncol].reshape(nrow, ncol) * sz + model.geom_pos[gid, 2]
    np.testing.assert_allclose(h, spec.terrain.heights, atol=2e-3)


def test_hfield_ray_heights_match_terrain(spec, model):
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    groups = np.zeros(6, np.uint8)
    groups[3] = 1  # heightfield collision geom only
    tid = _id(model, OBJ.mjOBJ_GEOM, "terrain")
    rng = np.random.default_rng(7)
    pts = np.c_[rng.uniform(-spec.half_x + 5, spec.half_x - 5, 24), rng.uniform(-spec.half_y + 5, spec.half_y - 5, 24)]
    pts = np.r_[pts, [[0.0, 0.0], [-29.05, -121.5], [100.3, 50.7]]]
    top = spec.h_max + 50.0
    geomid = np.zeros(1, np.int32)
    for x, y in pts:
        dist = mujoco.mj_ray(model, data, np.array([x, y, top]), np.array([0.0, 0.0, -1.0]), groups, 1, -1, geomid)
        assert dist > 0 and geomid[0] == tid, (x, y)
        assert top - dist == pytest.approx(spec.terrain.height_at(x, y), abs=0.05), (x, y)


def test_geom_budget_and_decor_collision_free(model):
    assert model.ngeom < 8000
    from prunemaxxing.sim.world_builder import GROUP_DECOR

    decor = np.flatnonzero(model.geom_group == GROUP_DECOR)
    assert decor.size > 0
    assert np.all(model.geom_contype[decor] == 0) and np.all(model.geom_conaffinity[decor] == 0)


# ---------------------------------------------------------------------------
# Runtime toggles
# ---------------------------------------------------------------------------


def _payload_geoms(model):
    from prunemaxxing.sim.world_builder import _is_cart, _is_dust

    names = [mujoco.mj_id2name(model, OBJ.mjOBJ_GEOM, i) or "" for i in range(model.ngeom)]
    dust = [i for i, n in enumerate(names) if _is_dust(n) and not n.startswith("fx_")]
    cart = [i for i, n in enumerate(names) if _is_cart(n) and not n.startswith("fx_")]
    return dust, cart, names


def test_payload_mode_toggle(fresh_model):
    from prunemaxxing.sim.world_builder import set_payload_mode

    m = fresh_model
    dust, cart, names = _payload_geoms(m)
    assert dust and cart
    for i in range(SWARM):
        assert any(names[g].startswith(f"drone{i}_dust") for g in dust)
        assert any(names[g].startswith(f"drone{i}_cart") for g in cart)
    assert np.all(m.geom_rgba[dust, 3] == 1) and np.all(m.geom_rgba[cart, 3] == 0)
    set_payload_mode(m, "cartridge")
    assert np.all(m.geom_rgba[dust, 3] == 0) and np.all(m.geom_rgba[cart, 3] == 1)
    set_payload_mode(m, "DUST")
    assert np.all(m.geom_rgba[dust, 3] == 1) and np.all(m.geom_rgba[cart, 3] == 0)
    with pytest.raises(ValueError):
        set_payload_mode(m, "pellets")


def test_obstacle_highlight_toggle(spec, fresh_model):
    from prunemaxxing.sim.world_builder import set_obstacle_detected

    m = fresh_model
    names = [mujoco.mj_id2name(m, OBJ.mjOBJ_GEOM, i) or "" for i in range(m.ngeom)]
    for oid in spec.names["obstacles"]:
        marks = [i for i, n in enumerate(names) if n.startswith(f"obs_mark_{oid}")]
        set_obstacle_detected(m, oid, True)
        assert np.all(m.geom_rgba[marks, 3] > 0.2)
        set_obstacle_detected(m, oid, False)
        assert np.all(m.geom_rgba[marks, 3] == 0)
    with pytest.raises(KeyError):
        set_obstacle_detected(m, "no_such_obstacle", True)


# ---------------------------------------------------------------------------
# Mulch painter
# ---------------------------------------------------------------------------


def _half_done_grid(spec) -> np.ndarray:
    cg = spec.cells
    grid = np.full((cg.nrows, cg.ncols), np.nan, np.float32)
    grid[cg.work] = 0.0
    work_rows = np.flatnonzero(cg.work.any(axis=1))
    south = work_rows[len(work_rows) // 2 :]
    grid[np.ix_(south, np.arange(cg.ncols))] = np.where(cg.work[south], 1.0, np.nan)
    return grid


def test_mulch_painter_cpu_blend(spec, fresh_model):
    from prunemaxxing.sim.world_builder import MulchPainter

    m = fresh_model
    painter = MulchPainter(spec, m, mode="dust")
    view = painter._view(m)[..., :3]
    np.testing.assert_array_equal(view, painter.base)
    with pytest.raises(ValueError):
        painter.apply(m, None, np.zeros((3, 3)), upload=False)

    painter.apply(m, None, _half_done_grid(spec), upload=False)
    changed = np.any(view != painter.base, axis=-1)
    assert changed.mean() > 0.05
    assert not np.any(changed & (painter.wgt == 0)), "mulch must stay on beds"
    # rows north -> south: the southern half of the field is mulched, the northern stays bare
    h = changed.shape[0]
    assert changed[h // 2 :].mean() > 10 * max(changed[: h // 3].mean(), 1e-6)

    painter.reset(m)
    np.testing.assert_array_equal(view, painter.base)

    cart = MulchPainter(spec, m, mode="cartridge")
    assert not np.array_equal(cart.mulch, painter.mulch)


def test_render_small_frame_not_black(model):
    renderer = _renderer_or_skip(model)
    try:
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        for cam in ("cam_overview", "cam_top", "ms_mast"):
            img = _render(model, data, renderer, cam)
            assert img.shape == (180, 320, 3)
            assert float(img.mean()) > 30.0, cam
            assert float(img.std()) > 8.0, cam
    finally:
        renderer.close()


def test_mulch_painter_upload_and_timing(spec, fresh_model):
    from prunemaxxing.sim.world_builder import MulchPainter

    m = fresh_model
    renderer = _renderer_or_skip(m)
    try:
        data = mujoco.MjData(m)
        mujoco.mj_forward(m, data)
        painter = MulchPainter(spec, m, mode="dust")
        painter.apply(m, renderer, np.zeros((spec.cells.nrows, spec.cells.ncols)), force=True)
        before = _render(m, data, renderer, "cam_top").astype(np.int16)

        ms_full = painter.apply(m, renderer, _half_done_grid(spec), force=True)
        after = _render(m, data, renderer, "cam_top").astype(np.int16)
        assert np.abs(after - before).max() > 20, "uploaded mulch texture must be visible"

        grid = _half_done_grid(spec)
        r, c = np.argwhere(spec.cells.work)[0]
        grid[r, c] = 0.5
        ms_incr = painter.apply(m, renderer, grid)
        assert ms_incr <= ms_full * 1.5
        # target < 150 ms per full update; generous bound for loaded CI machines
        assert ms_full < 450.0, f"full mulch update took {ms_full:.0f} ms"
    finally:
        renderer.close()


def test_render_timing_720p(model):
    renderer = _renderer_or_skip(model, height=720, width=1280)
    try:
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        _render(model, data, renderer, "cam_overview")
        t0 = time.perf_counter()
        n = 5
        for _ in range(n):
            _render(model, data, renderer, "ms_mast")
        ms = (time.perf_counter() - t0) * 1000 / n
        # target < 50 ms; generous bound for shared GPUs
        assert ms < 150.0, f"1280x720 render took {ms:.1f} ms"
    finally:
        renderer.close()
