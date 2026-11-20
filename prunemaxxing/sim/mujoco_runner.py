"""Legacy v1 small-scale MuJoCo replay. Superseded by :mod:`prunemaxxing.sim.replay` (1:1 world)."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Prefer NVIDIA EGL on this host before importing mujoco.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault(
    "__EGL_VENDOR_LIBRARY_FILENAMES",
    "/usr/share/glvnd/egl_vendor.d/10_nvidia.json",
)

import mujoco

from prunemaxxing.planning.payload_modes import PayloadMode
from prunemaxxing.planning.simulator import CoverageSimResult
from prunemaxxing.sim.heightfield import ensure_heightfield

XML_PATH = Path(__file__).with_name("mothership_swarm.xml")

N_DUST_FX = 20
N_CART_FX = 8
N_SPRAY_LOCAL = 4
BED_Z = 0.18  # resting height for dropped cartridges on raised beds


@dataclass
class CameraClip:
    name: str
    frames: list[np.ndarray]


@dataclass
class _DustParticle:
    active: bool = False
    age: float = 0.0
    life: float = 1.0
    origin: np.ndarray = field(default_factory=lambda: np.zeros(3))
    vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rgba: np.ndarray = field(default_factory=lambda: np.array([0.55, 0.40, 0.22, 0.0]))


@dataclass
class _CartDrop:
    active: bool = False
    age: float = 0.0
    life: float = 1.4
    origin: np.ndarray = field(default_factory=lambda: np.zeros(3))
    offset_xy: np.ndarray = field(default_factory=lambda: np.zeros(2))
    fall_duration: float = 0.55


class DepositFX:
    """Kinematic dust spray + cartridge drop visuals (no physics contacts)."""

    def __init__(self, model: mujoco.MjModel, rng: np.random.Generator | None = None):
        self.rng = rng or np.random.default_rng(7)
        self.dust = [_DustParticle() for _ in range(N_DUST_FX)]
        self.carts = [_CartDrop() for _ in range(N_CART_FX)]
        self._dust_body: list[str] = [f"fx_dust_{i}" for i in range(N_DUST_FX)]
        self._dust_geom: list[str] = [f"fx_dust_{i}" for i in range(N_DUST_FX)]
        self._cart_body: list[str] = [f"fx_cart_{i}" for i in range(N_CART_FX)]
        self._cart_geom: list[str] = [f"fx_cart_{i}" for i in range(N_CART_FX)]
        self._spray_geom: list[list[str]] = [
            [f"d{d}_spray{s}" for s in range(N_SPRAY_LOCAL)] for d in range(4)
        ]
        self._next_dust = 0
        self._next_cart = 0
        self._hide_all(model)

    def _hide_all(self, model: mujoco.MjModel) -> None:
        for name in self._dust_geom + self._cart_geom:
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid >= 0:
                rgba = model.geom_rgba[gid].copy()
                rgba[3] = 0.0
                model.geom_rgba[gid] = rgba
        for d in range(4):
            for s in range(N_SPRAY_LOCAL):
                gid = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_GEOM, self._spray_geom[d][s]
                )
                if gid >= 0:
                    rgba = model.geom_rgba[gid].copy()
                    rgba[3] = 0.0
                    model.geom_rgba[gid] = rgba

    def _spawn_dust(self, tip: np.ndarray, intensity: float) -> None:
        if intensity < 0.08:
            return
        n_spawn = 1 + int(intensity > 0.45) + int(intensity > 0.75)
        for _ in range(n_spawn):
            p = self.dust[self._next_dust]
            self._next_dust = (self._next_dust + 1) % N_DUST_FX
            spread = 0.12 + 0.18 * intensity
            lateral = self.rng.normal(0.0, spread, size=2)
            p.active = True
            p.age = 0.0
            p.life = float(self.rng.uniform(0.55, 1.05))
            p.origin = tip + np.array([lateral[0] * 0.25, lateral[1] * 0.25, 0.02])
            p.vel = np.array(
                [
                    lateral[0] * 1.4,
                    lateral[1] * 1.4,
                    float(self.rng.uniform(-1.8, -0.6)),
                ]
            )
            shade = float(self.rng.uniform(-0.05, 0.05))
            p.rgba = np.array(
                [0.55 + shade, 0.40 + shade * 0.7, 0.22 + shade * 0.4, 0.0]
            )

    def _spawn_cart(self, tip: np.ndarray, intensity: float) -> None:
        if intensity < 0.35:
            return
        c = self.carts[self._next_cart]
        self._next_cart = (self._next_cart + 1) % N_CART_FX
        c.active = True
        c.age = 0.0
        c.life = 1.6
        c.fall_duration = float(self.rng.uniform(0.45, 0.70))
        c.origin = tip.copy()
        c.offset_xy = self.rng.normal(0.0, 0.12, size=2)

    def update(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        mode: PayloadMode,
        frame_k: int,
        dt: float = 1.0 / 12.0,
    ) -> None:
        """Advance FX from current effector tips; call after mj_forward of vehicles."""
        for di in range(4):
            tip = _effector_tip(model, data, di)
            intensity = _deposit_intensity(data, di)
            if mode is PayloadMode.DUST:
                if intensity > 0.15 and (frame_k + di) % 2 == 0:
                    self._spawn_dust(tip, intensity)
                self._update_local_spray(model, di, intensity, frame_k, active=True)
            else:
                self._update_local_spray(model, di, 0.0, frame_k, active=False)
                period = 5 + (di % 3)
                if intensity > 0.4 and frame_k % period == di % period:
                    self._spawn_cart(tip, intensity)

        if mode is PayloadMode.DUST:
            self._step_dust(model, data, dt)
            self._hide_carts(model, data)
        else:
            self._step_carts(model, data, dt)
            self._hide_dust(model, data)

    def snapshot_mid_deposit(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        mode: PayloadMode,
    ) -> None:
        """Pose a dramatic mid-deposit burst for still renders."""
        self._hide_all(model)
        for di in range(4):
            tip = _effector_tip(model, data, di)
            if mode is PayloadMode.DUST:
                for _ in range(4):
                    self._spawn_dust(tip, 0.95)
                self._update_local_spray(model, di, 0.95, frame_k=8, active=True)
            else:
                self._update_local_spray(model, di, 0.0, frame_k=0, active=False)
                for _ in range(2):
                    self._spawn_cart(tip, 0.95)

        if mode is PayloadMode.DUST:
            for i, p in enumerate(self.dust):
                if not p.active:
                    continue
                # Stagger ages so the burst fills the column under the effector
                p.age = 0.08 + 0.06 * (i % 6)
                pos = p.origin + p.vel * p.age + np.array([0.0, 0.0, -4.5 * p.age * p.age])
                pos[2] = max(pos[2], BED_Z + 0.02)
                alpha = float(np.clip(1.0 - p.age / p.life, 0.0, 1.0)) * 0.95
                _set_free_body(model, data, self._dust_body[i], pos, _quat_from_yaw(0.0))
                gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, self._dust_geom[i])
                if gid >= 0:
                    rgba = p.rgba.copy()
                    rgba[3] = alpha
                    model.geom_rgba[gid] = rgba
                    # Slightly larger particles for still readability
                    model.geom_size[gid] = model.geom_size[gid] * 1.35
        else:
            for i, c in enumerate(self.carts):
                if not c.active:
                    continue
                # Spread mid-fall / near-bed so d0_cam sees a release trail
                t = 0.12 + 0.18 * (i % 5)
                c.age = t
                frac = min(1.0, t / c.fall_duration)
                ease = frac * frac
                x = c.origin[0] + c.offset_xy[0] * ease
                y = c.origin[1] + c.offset_xy[1] * ease
                z = c.origin[2] + (BED_Z + 0.10 - c.origin[2]) * ease
                _set_free_body(
                    model, data, self._cart_body[i], [x, y, z], _quat_from_yaw(0.15 * i)
                )
                gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, self._cart_geom[i])
                if gid >= 0:
                    rgba = model.geom_rgba[gid].copy()
                    rgba[:3] = (0.28, 0.58, 0.30)
                    rgba[3] = 1.0
                    model.geom_rgba[gid] = rgba

    def _update_local_spray(
        self,
        model: mujoco.MjModel,
        di: int,
        intensity: float,
        frame_k: int,
        active: bool,
    ) -> None:
        """Animate effector-local spray ellipsoids (visible in d0_cam close-ups)."""
        for s in range(N_SPRAY_LOCAL):
            name = self._spray_geom[di][s]
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid < 0:
                continue
            rgba = model.geom_rgba[gid].copy()
            if not active or intensity < 0.12:
                rgba[3] = 0.0
                model.geom_rgba[gid] = rgba
                continue
            phase = 0.9 * frame_k + 1.7 * s + 0.4 * di
            ox = 0.08 * np.sin(phase) * (0.4 + 0.6 * intensity)
            oy = 0.08 * np.cos(phase * 1.1) * (0.4 + 0.6 * intensity)
            oz = -0.18 - 0.12 * s - 0.08 * intensity * (0.5 + 0.5 * np.sin(phase * 1.3))
            model.geom_pos[gid] = np.array([ox, oy, oz], dtype=float)
            grow = 1.0 + 0.35 * s + 0.25 * intensity
            model.geom_size[gid] = np.array([0.04, 0.03, 0.022], dtype=float) * grow
            rgba[3] = float(np.clip(0.55 * intensity * (1.0 - 0.12 * s), 0.0, 0.85))
            model.geom_rgba[gid] = rgba

    def _step_dust(self, model: mujoco.MjModel, data: mujoco.MjData, dt: float) -> None:
        for i, p in enumerate(self.dust):
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, self._dust_geom[i])
            if not p.active:
                if gid >= 0:
                    rgba = model.geom_rgba[gid].copy()
                    rgba[3] = 0.0
                    model.geom_rgba[gid] = rgba
                _set_free_body(
                    model, data, self._dust_body[i], [0, 0, -8], _quat_from_yaw(0.0)
                )
                continue
            if dt > 0:
                p.age += dt
            if p.age >= p.life:
                p.active = False
                if gid >= 0:
                    rgba = model.geom_rgba[gid].copy()
                    rgba[3] = 0.0
                    model.geom_rgba[gid] = rgba
                continue
            pos = p.origin + p.vel * p.age + np.array([0.0, 0.0, -4.5 * p.age * p.age])
            pos[2] = max(float(pos[2]), BED_Z + 0.015)
            alpha = float(np.clip(1.0 - p.age / p.life, 0.0, 1.0))
            if pos[2] <= BED_Z + 0.04:
                alpha *= 0.55
            _set_free_body(model, data, self._dust_body[i], pos, _quat_from_yaw(0.3 * i))
            if gid >= 0:
                rgba = p.rgba.copy()
                rgba[3] = alpha * 0.9
                model.geom_rgba[gid] = rgba

    def _step_carts(self, model: mujoco.MjModel, data: mujoco.MjData, dt: float) -> None:
        for i, c in enumerate(self.carts):
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, self._cart_geom[i])
            if not c.active:
                if gid >= 0:
                    rgba = model.geom_rgba[gid].copy()
                    rgba[3] = 0.0
                    model.geom_rgba[gid] = rgba
                _set_free_body(
                    model, data, self._cart_body[i], [0, 0, -8], _quat_from_yaw(0.0)
                )
                continue
            c.age += dt
            if c.age >= c.life:
                c.active = False
                if gid >= 0:
                    rgba = model.geom_rgba[gid].copy()
                    rgba[3] = 0.0
                    model.geom_rgba[gid] = rgba
                continue
            frac = min(1.0, c.age / c.fall_duration)
            ease = frac * frac
            x = c.origin[0] + c.offset_xy[0] * ease
            y = c.origin[1] + c.offset_xy[1] * ease
            z = c.origin[2] + (BED_Z + 0.10 - c.origin[2]) * ease
            yaw = 0.4 * i + 0.6 * frac
            _set_free_body(model, data, self._cart_body[i], [x, y, z], _quat_from_yaw(yaw))
            if c.age < c.fall_duration:
                alpha = 1.0
            else:
                linger = (c.age - c.fall_duration) / max(1e-6, c.life - c.fall_duration)
                alpha = float(np.clip(1.0 - linger, 0.0, 1.0))
            if gid >= 0:
                rgba = model.geom_rgba[gid].copy()
                rgba[:3] = (0.28, 0.58, 0.30)
                rgba[3] = alpha
                model.geom_rgba[gid] = rgba

    def _hide_dust(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        for i in range(N_DUST_FX):
            self.dust[i].active = False
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, self._dust_geom[i])
            if gid >= 0:
                rgba = model.geom_rgba[gid].copy()
                rgba[3] = 0.0
                model.geom_rgba[gid] = rgba
            _set_free_body(
                model, data, self._dust_body[i], [0, 0, -8], _quat_from_yaw(0.0)
            )

    def _hide_carts(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        for i in range(N_CART_FX):
            self.carts[i].active = False
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, self._cart_geom[i])
            if gid >= 0:
                rgba = model.geom_rgba[gid].copy()
                rgba[3] = 0.0
                model.geom_rgba[gid] = rgba
            _set_free_body(
                model, data, self._cart_body[i], [0, 0, -8], _quat_from_yaw(0.0)
            )


def _quat_from_yaw(yaw: float) -> np.ndarray:
    """MuJoCo quat wxyz from yaw about +Z."""
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)], dtype=float)


def _set_free_body(model: mujoco.MjModel, data: mujoco.MjData, body: str, pos, quat) -> None:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
    jntadr = model.body_jntadr[bid]
    qadr = model.jnt_qposadr[jntadr]
    data.qpos[qadr : qadr + 3] = pos
    data.qpos[qadr + 3 : qadr + 7] = quat


def _effector_tip(model: mujoco.MjModel, data: mujoco.MjData, di: int) -> np.ndarray:
    """World position near the payload scoop under drone di."""
    for name in (f"d{di}_effector", f"drone{di}"):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid >= 0:
            tip = data.xpos[bid].copy()
            if name.startswith("drone"):
                tip[2] -= 0.35
            else:
                tip[2] -= 0.22
            return tip
    return np.array([0.0, 0.0, 1.0])


def _deposit_intensity(data: mujoco.MjData, di: int) -> float:
    """0..1 intensity from lift actuator extension (more negative = depositing)."""
    if di >= data.ctrl.shape[0]:
        return 0.0
    ctrl = float(data.ctrl[di])
    return float(np.clip((-ctrl - 0.05) / 0.35, 0.0, 1.0))


def _patch_xml_heightfield(xml_text: str, size_attr: str, z_offset: float) -> str:
    """Inject generated hfield size and mean-centered Z offset into scene XML."""
    xml_text = re.sub(
        r'(<hfield\s+name="pilot_dem"[^>]*\ssize=")[^"]+(")',
        rf"\g<1>{size_attr}\2",
        xml_text,
        count=1,
    )
    xml_text = re.sub(
        r'(<geom\s+name="terrain"[^>]*\spos=")[^"]+(")',
        rf"\g<1>0 0 {z_offset:.6g}\2",
        xml_text,
        count=1,
    )
    return xml_text


def load_model(
    xml_path: Path | None = None,
    *,
    regenerate_heightfield: bool = False,
) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Load scene with DEM heightfield (auto-built if missing)."""
    path = xml_path or XML_PATH
    spec = ensure_heightfield(force=regenerate_heightfield)
    xml_text = _patch_xml_heightfield(path.read_text(), spec.size_attr, spec.z_offset)
    png = Path(__file__).with_name("assets") / "pilot_dem_hfield.png"
    assets = {"assets/pilot_dem_hfield.png": png.read_bytes()}
    model = mujoco.MjModel.from_xml_string(xml_text, assets=assets)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def set_payload_mode_visuals(model: mujoco.MjModel, mode: PayloadMode) -> None:
    """Toggle dust vs cartridge geom visibility via RGBA alpha."""
    for di in range(4):
        for name, alpha in [
            (f"d{di}_payload_dust", 1.0 if mode is PayloadMode.DUST else 0.0),
            (f"d{di}_payload_cart", 1.0 if mode is PayloadMode.CARTRIDGE else 0.0),
        ]:
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid < 0:
                continue
            rgba = model.geom_rgba[gid].copy()
            rgba[3] = alpha
            if mode is PayloadMode.CARTRIDGE and "cart" in name:
                rgba[:3] = (0.28, 0.58, 0.30)
            model.geom_rgba[gid] = rgba


def world_from_utm(
    e: float,
    n: float,
    origin_e: float,
    origin_n: float,
    scale: float = 0.22,
) -> tuple[float, float]:
    """Map UTM delta into MuJoCo metres (ops bubble ~40 m → ~9 m on scene)."""
    x = (e - origin_e) * scale
    y = (n - origin_n) * scale
    return x, y


def _lerp_frame(a, b, t: float):
    """Interpolate two SimFrame-like pose snapshots for smooth video."""
    me = a.mothership_e * (1 - t) + b.mothership_e * t
    mn = a.mothership_n * (1 - t) + b.mothership_n * t
    # Unwrap yaw briefly
    yaw_a, yaw_b = a.mothership_yaw, b.mothership_yaw
    dyaw = (yaw_b - yaw_a + np.pi) % (2 * np.pi) - np.pi
    myaw = yaw_a + t * dyaw
    n = max(len(a.drones), len(b.drones))
    drones = []
    for i in range(n):
        da = a.drones[i] if i < len(a.drones) else a.drones[-1]
        db = b.drones[i] if i < len(b.drones) else b.drones[-1]
        drones.append(
            (
                da[0] * (1 - t) + db[0] * t,
                da[1] * (1 - t) + db[1] * t,
                da[2] * (1 - t) + db[2] * t,
            )
        )
    return me, mn, myaw, drones


def animate_and_render(
    result: CoverageSimResult,
    cameras: list[str] | None = None,
    width: int = 640,
    height: int = 368,
    max_frames: int = 180,
    payload_mode: PayloadMode | str = PayloadMode.DUST,
    upsample: int = 3,
) -> dict[str, list[np.ndarray]]:
    """
    Drive mothership/drones from twin frames and render named cameras.

    - Origin follows the mothership every frame (follow-cam) so drones stay visible
    - No clamp-to-orbit (that made drones look parked)
    - Upsamples between sim frames for smooth motion
    """
    mode = PayloadMode(payload_mode)
    model, data = load_model()
    set_payload_mode_visuals(model, mode)
    fx = DepositFX(model)

    cameras = cameras or [
        "hero_orbit",
        "top_tactical",
        "ms_front",
        "ms_overview",
        "d0_cam",
        "low_tracking",
    ]
    renderer = mujoco.Renderer(model, height=height, width=width)

    frames_src = result.frames
    if not frames_src:
        raise ValueError("CoverageSimResult has no frames; run with record_frames=True")

    # Keyframes from sim, then upsample for smooth video
    n_keys = min(max(24, max_frames // max(1, upsample)), len(frames_src))
    key_idxs = np.linspace(0, len(frames_src) - 1, num=n_keys, dtype=int)

    out: dict[str, list[np.ndarray]] = {c: [] for c in cameras}
    scale = 0.22
    render_i = 0
    dt = 1.0 / 24.0

    def _emit(me, mn, myaw, drones, k: int) -> None:
        nonlocal render_i
        # Follow-cam origin = current mothership (drones move relative in scene)
        mx, my = 0.0, 0.0
        _set_free_body(
            model, data, "mothership", [mx, my, 0.55], _quat_from_yaw(myaw)
        )
        n_drones = min(4, len(drones))
        for di, (de, dn, alt) in enumerate(drones[:n_drones]):
            rel_e = float(de - me)
            rel_n = float(dn - mn)
            dist = float(np.hypot(rel_e, rel_n))
            # Guaranteed angular separation (sector) blended with real cell bearing
            sector = (2 * np.pi * (di + 0.5) / n_drones) + 0.12 * k
            if dist > 2.0:
                bearing = float(np.arctan2(rel_n, rel_e))
                # Unwrap toward sector so drones don't collapse into one wedge
                dang = (bearing - sector + np.pi) % (2 * np.pi) - np.pi
                ang = sector + 0.55 * dang
                radius_m = min(dist, 40.0)
            else:
                ang = sector
                radius_m = 18.0 + 4.0 * di
            # Outbound pulse — each drone phase-shifted so they don't sync-collapse
            flight = 0.72 + 0.28 * np.sin(0.40 * k + 1.2 * di)
            r = radius_m * scale * flight
            dx = float(r * np.cos(ang))
            dy = float(r * np.sin(ang))
            z = 2.0 + 0.9 * di * 0.15 + 1.2 * flight
            yaw = ang
            _set_free_body(
                model, data, f"drone{di}", [dx, dy, z], _quat_from_yaw(yaw)
            )
            if di < model.nu:
                deposit_wave = 0.5 + 0.5 * np.sin(0.45 * k + 0.5 * di)
                data.ctrl[di] = -0.10 - 0.28 * deposit_wave

        # Keep world cameras framing the mothership cluster
        for cam_name, (az, el, dist) in {
            "hero_orbit": (135.0, -22.0, 14.0),
            "top_tactical": (0.0, -90.0, 22.0),
            "low_tracking": (160.0, -12.0, 11.0),
        }.items():
            cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
            if cid < 0:
                continue
            # Free cameras in XML are fixed; update via MjvCamera in update_scene instead
            _ = (az, el, dist)

        mujoco.mj_forward(model, data)
        fx.update(model, data, mode, frame_k=k, dt=dt)
        mujoco.mj_forward(model, data)

        for cam in cameras:
            if cam in ("hero_orbit", "top_tactical", "low_tracking"):
                # Look-at mothership with cinematic free camera
                cam_id = mujoco.MjvCamera()
                mujoco.mjv_defaultFreeCamera(model, cam_id)
                cam_id.lookat[:] = [0.0, 0.0, 0.8]
                if cam == "hero_orbit":
                    cam_id.distance, cam_id.azimuth, cam_id.elevation = 13.0, 130.0, -24.0
                elif cam == "top_tactical":
                    cam_id.distance, cam_id.azimuth, cam_id.elevation = 20.0, 0.0, -90.0
                else:
                    cam_id.distance, cam_id.azimuth, cam_id.elevation = 10.0, 155.0, -14.0
                renderer.update_scene(data, camera=cam_id)
            else:
                renderer.update_scene(data, camera=cam)
            out[cam].append(renderer.render().copy())
        render_i += 1

    for ki in range(len(key_idxs) - 1):
        ia, ib = int(key_idxs[ki]), int(key_idxs[ki + 1])
        fa, fb = frames_src[ia], frames_src[ib]
        steps = upsample if ki < len(key_idxs) - 2 else upsample
        for s in range(steps):
            t = s / steps
            me, mn, myaw, drones = _lerp_frame(fa, fb, t)
            _emit(me, mn, myaw, drones, k=render_i)
            if render_i >= max_frames:
                break
        if render_i >= max_frames:
            break
    # Final keyframe
    if render_i < max_frames and len(key_idxs):
        fr = frames_src[int(key_idxs[-1])]
        _emit(fr.mothership_e, fr.mothership_n, fr.mothership_yaw, fr.drones, k=render_i)

    renderer.close()
    return out


def render_still_hero(
    payload_mode: PayloadMode | str = PayloadMode.DUST,
    camera: str = "hero_orbit",
    width: int = 1280,
    height: int = 720,
    with_deposit_fx: bool = True,
) -> np.ndarray:
    mode = PayloadMode(payload_mode)
    model, data = load_model()
    set_payload_mode_visuals(model, mode)
    bed_x = [-2.55, -0.85, 0.85, 2.55]
    for i in range(4):
        _set_free_body(
            model,
            data,
            f"drone{i}",
            [bed_x[i], 2.0 + 0.4 * i, 2.4],
            _quat_from_yaw(0.05 * i),
        )
        data.ctrl[i] = -0.35
    mujoco.mj_forward(model, data)
    if with_deposit_fx:
        fx = DepositFX(model)
        fx.snapshot_mid_deposit(model, data, mode)
        mujoco.mj_forward(model, data)
    r = mujoco.Renderer(model, height=height, width=width)
    r.update_scene(data, camera=camera)
    img = r.render().copy()
    r.close()
    return img


def write_effector_showcase_stills(
    out_dir: Path | str | None = None,
) -> dict[str, Path]:
    """Render effector_dust.png and effector_cartridge.png for showcase."""
    out = Path(out_dir) if out_dir else Path("outputs/showcase")
    out.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v2 as imageio
    except ImportError:  # pragma: no cover
        from PIL import Image

        def _write(path: Path, arr: np.ndarray) -> None:
            Image.fromarray(arr).save(path)

    else:

        def _write(path: Path, arr: np.ndarray) -> None:
            imageio.imwrite(path, arr)

    paths: dict[str, Path] = {}
    for mode, name in [
        (PayloadMode.DUST, "effector_dust.png"),
        (PayloadMode.CARTRIDGE, "effector_cartridge.png"),
    ]:
        img = render_still_hero(mode, camera="d0_cam", width=960, height=640)
        path = out / name
        _write(path, img)
        paths[name] = path
    return paths
