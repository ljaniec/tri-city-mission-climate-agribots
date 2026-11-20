"""MJCF generators for the 1:1 mothership UGV, heavy-lift hexacopters and FX pools.

Conventions (all metres, radians):

Mothership body ``mothership`` (freejoint ``ms_free``)
    origin = ground-contact level, centred between the axles; +x forward,
    +y left, +z up.  Wheelbase 2.4, track 1.7 (wheels ride in furrows two beds
    apart), wheel radius 0.45, width 0.30, frame underside 1.0 m above ground.
Drone body ``drone{i}`` (freejoint ``drone{i}_free``)
    origin = centre of mass; skid bottoms at z = -DRONE_SKID_Z below origin.
    6 rotors on arms at 0, 60, ..., 300 deg, motor radius 0.75 m, prop radius
    0.35 m (2.2 m tip-to-tip).  Slide joint ``drone{i}_lift`` (range -1.5..0)
    lowers the winch cradle ``drone{i}_cradle``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

WHEELBASE = 2.4
TRACK = 1.7
WHEEL_R = 0.45
WHEEL_W = 0.30
FRAME_Z0 = 1.0
DECK_Z = 1.40
MODULE_TOP = 2.25
PAD_TOP_Z = 2.36
DOCKS = [(1.45, 1.05), (1.45, -1.05), (-1.45, 1.05), (-1.45, -1.05)]  # dock_0..3: FL, FR, RL, RR

DRONE_ARM_R = 0.75
DRONE_PROP_R = 0.35
DRONE_SKID_Z = 0.44  # origin height above skid contact
CRADLE_Z0 = -0.20  # cradle origin (hook point) below drone origin at lift = 0
LIFT_RANGE = (-1.5, 0.0)

DRONE_COLORS = ["#a78bfa", "#ff9f1a", "#7CFC00", "#ff2d95"]

N_FX_DUST = 200
N_FX_CART = 40
FX_PARK_Z = -50.0


def hex_rgb(h: str) -> tuple[float, float, float]:
    h = h.lstrip("#")
    return tuple(int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


def quat_axis(axis, ang) -> np.ndarray:
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    return np.concatenate([[np.cos(ang / 2)], np.sin(ang / 2) * axis])


def quat_mul(a, b) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def f(v) -> str:
    return " ".join(f"{float(x):.5g}" for x in np.atleast_1d(v))


def cam_xyaxes(forward, up_hint=(0, 0, 1)) -> str:
    fw = np.asarray(forward, float)
    fw /= np.linalg.norm(fw)
    right = np.cross(fw, up_hint)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([0.0, -1.0, 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fw)
    return f(np.concatenate([right, up]))


class G:
    """Tiny MJCF geom string builder."""

    @staticmethod
    def geom(name=None, type="box", size=None, pos=None, quat=None, zaxis=None, fromto=None, material=None,
             rgba=None, cls=None, extra="") -> str:
        a = []
        if name:
            a.append(f'name="{name}"')
        if cls:
            a.append(f'class="{cls}"')
        a.append(f'type="{type}"')
        if size is not None:
            a.append(f'size="{f(size)}"')
        if fromto is not None:
            a.append(f'fromto="{f(fromto)}"')
        if pos is not None:
            a.append(f'pos="{f(pos)}"')
        if quat is not None:
            a.append(f'quat="{f(quat)}"')
        if zaxis is not None:
            a.append(f'zaxis="{f(zaxis)}"')
        if material:
            a.append(f'material="{material}"')
        if rgba is not None:
            a.append(f'rgba="{f(rgba)}"')
        if extra:
            a.append(extra)
        return "<geom " + " ".join(a) + "/>"


g = G.geom


# ---------------------------------------------------------------------------
# Materials
# ---------------------------------------------------------------------------


def robot_assets(swarm_size: int) -> str:
    mats = [
        '<material name="ms_amber" texture="tex_panel" rgba="0.97 0.63 0.07 1" specular="0.55" shininess="0.65"/>',
        '<material name="ms_amber_dark" rgba="0.80 0.48 0.05 1" specular="0.4" shininess="0.5"/>',
        '<material name="ms_white" texture="tex_panel" rgba="0.93 0.93 0.91 1" specular="0.45" shininess="0.6"/>',
        '<material name="ms_dark" rgba="0.17 0.18 0.19 1" specular="0.35" shininess="0.4"/>',
        '<material name="ms_black" rgba="0.055 0.055 0.06 1" specular="0.25" shininess="0.3"/>',
        '<material name="ms_rubber" rgba="0.075 0.075 0.08 1" specular="0.08" shininess="0.1"/>',
        '<material name="ms_rim" rgba="0.74 0.76 0.78 1" specular="0.9" shininess="0.8"/>',
        '<material name="ms_steel" rgba="0.55 0.57 0.6 1" specular="0.7" shininess="0.7"/>',
        '<material name="ms_glass" rgba="0.08 0.12 0.16 1" specular="1" shininess="1"/>',
        '<material name="ms_solar" texture="tex_solar" texuniform="true" texrepeat="1.1 1.1" specular="0.9" shininess="0.9"/>',
        '<material name="ms_beacon" rgba="1.0 0.52 0.05 1" emission="1.0" specular="0.5"/>',
        '<material name="ms_headlight" rgba="1.0 0.98 0.9 1" emission="1.0"/>',
        '<material name="ms_tail" rgba="0.9 0.05 0.04 1" emission="0.8"/>',
        '<material name="ms_coil" rgba="0.2 1.0 0.85 1" emission="0.75"/>',
        '<material name="ms_hazard" rgba="0.08 0.08 0.08 1"/>',
        '<material name="ms_battery" rgba="0.24 0.27 0.30 1" specular="0.5"/>',
        '<material name="ms_battery_ok" rgba="0.25 0.95 0.35 1" emission="0.7"/>',
        '<material name="mulch_dust" texture="tex_dust" texuniform="true" texrepeat="1.5 1.5" specular="0.05"/>',
        '<material name="mulch_chips" texture="tex_chips" texuniform="true" texrepeat="0.8 0.8" specular="0.05"/>',
        '<material name="puck_brown" rgba="0.55 0.38 0.22 1" specular="0.1"/>',
        '<material name="puck_green" rgba="0.36 0.55 0.22 1" specular="0.1"/>',
        '<material name="drone_body" rgba="0.16 0.17 0.19 1" specular="0.6" shininess="0.7"/>',
        '<material name="drone_carbon" rgba="0.09 0.09 0.1 1" specular="0.5" shininess="0.6"/>',
        '<material name="drone_blade" rgba="0.05 0.05 0.06 1" specular="0.6" shininess="0.8"/>',
        '<material name="led_green" rgba="0.2 1 0.3 1" emission="1"/>',
        '<material name="led_red" rgba="1 0.15 0.1 1" emission="1"/>',
        '<material name="led_white" rgba="1 1 1 1" emission="1"/>',
    ]
    for i in range(max(swarm_size, 4)):
        r, gg, b = hex_rgb(DRONE_COLORS[i % len(DRONE_COLORS)])
        mats.append(f'<material name="drone_accent{i}" rgba="{r:.3f} {gg:.3f} {b:.3f} 1" specular="0.6" shininess="0.7"/>')
        mats.append(f'<material name="drone_glow{i}" rgba="{r:.3f} {gg:.3f} {b:.3f} 1" emission="0.9"/>')
    return "\n".join(mats)


# ---------------------------------------------------------------------------
# Mothership
# ---------------------------------------------------------------------------


def _wheel_geoms(prefix: str, side: int) -> list[str]:
    """Geoms of a spinning wheel body (spin axis = local y). side=+1 left, -1 right."""
    out = []
    ya = (0, 1, 0)
    out.append(g(f"{prefix}_tyre", "cylinder", [WHEEL_R - 0.025, WHEEL_W / 2], zaxis=ya, material="ms_rubber",
                 cls="tyre"))
    # sidewall bulge rings
    for s in (-1, 1):
        out.append(g(None, "cylinder", [WHEEL_R - 0.05, 0.012], pos=[0, s * (WHEEL_W / 2 + 0.004), 0], zaxis=ya,
                     material="ms_rubber", cls="robot"))
    # chevron tread lugs (contact radius = WHEEL_R)
    n = 14
    for k in range(n):
        th = 2 * np.pi * k / n
        for s in (-1, 1):
            yoff = s * 0.075
            ang = th + s * 0.06
            q = quat_mul(quat_axis((0, 1, 0), ang), quat_axis((1, 0, 0), s * 0.35))
            p = [(WHEEL_R - 0.018) * np.sin(ang), yoff, (WHEEL_R - 0.018) * np.cos(ang)]
            out.append(g(None, "box", [0.035, 0.07, 0.02], pos=p, quat=q, material="ms_rubber", cls="robot"))
    oy = side * (WHEEL_W / 2 + 0.01)
    out.append(g(f"{prefix}_rim", "cylinder", [0.27, WHEEL_W / 2 + 0.008], zaxis=ya, material="ms_rim", cls="robot"))
    out.append(g(None, "cylinder", [0.2, 0.01], pos=[0, oy + side * 0.005, 0], zaxis=ya, material="ms_dark",
                 cls="robot"))
    out.append(g(f"{prefix}_hub", "cylinder", [0.095, 0.03], pos=[0, oy + side * 0.02, 0], zaxis=ya,
                 material="ms_amber", cls="robot"))
    for k in range(6):
        th = 2 * np.pi * k / 6
        out.append(g(None, "cylinder", [0.022, 0.018], pos=[0.16 * np.sin(th), oy + side * 0.012, 0.16 * np.cos(th)],
                     zaxis=ya, material="ms_rim", cls="robot"))
    # two contrasting sidewall marks so rotation reads at any distance
    for th in (0.0, np.pi):
        out.append(g(None, "box", [0.05, 0.006, 0.03], pos=[0.355 * np.sin(th), side * (WHEEL_W / 2 + 0.02),
                                                             0.355 * np.cos(th)], material="ms_amber", cls="robot"))
    return out


def _fender(prefix, side, pos=(0, 0, 0)) -> list[str]:
    px, py, pz = pos
    out = []
    w = 0.2
    r = WHEEL_R + 0.09
    for k, th in enumerate(np.linspace(-1.0, 1.0, 5)):
        q = quat_axis((0, 1, 0), th)
        p = [px + r * np.sin(th), py, pz + r * np.cos(th)]
        out.append(g(f"{prefix}_fender{k}" if k == 2 else None, "box", [0.13, w, 0.012], pos=p, quat=q,
                     material="ms_amber", cls="robot"))
    return out


def mothership_xml(start_pos=(0.0, 0.0, 0.0)) -> str:
    L = []
    a = L.append
    a(f'<body name="mothership" pos="{f(start_pos)}">')
    a('<freejoint name="ms_free"/>')
    a('<inertial pos="0 0 1.25" mass="3800" diaginertia="1980 6300 6850"/>')
    a('<site name="ms_origin" pos="0 0 0" size="0.05" rgba="1 1 1 0"/>')
    # --- main frame (dark running gear) ---
    for s in (-1, 1):
        a(g(None, "box", [2.1, 0.11, 0.17], pos=[0, s * 0.82, FRAME_Z0 + 0.17], material="ms_dark", cls="robot"))
    for x in (-1.9, -1.2, 0.0, 1.2, 1.9):
        a(g(None, "box", [0.09, 0.82, 0.12], pos=[x, 0, FRAME_Z0 + 0.14], material="ms_dark", cls="robot"))
    a(g("ms_deck", "box", [2.1, 1.0, 0.03], pos=[0, 0, DECK_Z - 0.03], material="ms_dark", cls="robot"))
    # side skirts on frame (amber) with hazard strip
    for s in (-1, 1):
        a(g(None, "box", [2.12, 0.02, 0.2], pos=[0, s * 1.0, FRAME_Z0 + 0.2], material="ms_amber", cls="robot"))
        for k in range(14):
            x = -1.95 + k * 0.3
            a(g(None, "box", [0.075, 0.006, 0.05], pos=[x, s * 1.022, FRAME_Z0 + 0.06],
                quat=quat_axis((0, 1, 0), 0.6), material="ms_hazard", cls="robot"))
    # --- front module (head) ---
    fx0, fx1 = 1.12, 2.1
    a(g("ms_head", "box", [(fx1 - fx0) / 2, 0.98, (MODULE_TOP - DECK_Z) / 2],
        pos=[(fx0 + fx1) / 2, 0, (MODULE_TOP + DECK_Z) / 2], material="ms_amber", cls="robot"))
    a(g(None, "box", [(fx1 - fx0) / 2 + 0.005, 0.985, 0.07], pos=[(fx0 + fx1) / 2, 0, MODULE_TOP - 0.07],
        material="ms_white", cls="robot"))
    a(g(None, "box", [0.02, 0.9, 0.22], pos=[fx1 + 0.012, 0, 1.78], material="ms_black", cls="robot"))  # sensor bar
    for s in (-1, 1):
        a(g(None, "box", [0.015, 0.16, 0.05], pos=[fx1 + 0.025, s * 0.68, 1.78], material="ms_headlight",
            cls="robot"))
        a(g(None, "box", [0.012, 0.05, 0.05], pos=[fx1 + 0.03, s * 0.32, 1.78], material="ms_glass", cls="robot"))
    a(g(None, "box", [0.012, 0.035, 0.035], pos=[fx1 + 0.03, 0, 1.78], material="ms_glass", cls="robot"))
    # bumper with hazard chevrons
    a(g("ms_bumper", "box", [0.08, 1.02, 0.09], pos=[fx1 + 0.08, 0, FRAME_Z0 + 0.12], material="ms_dark",
        cls="robot"))
    for k in range(9):
        y = -0.88 + k * 0.22
        a(g(None, "box", [0.006, 0.05, 0.085], pos=[fx1 + 0.162, y, FRAME_Z0 + 0.12], quat=quat_axis((1, 0, 0), 0.6),
            material="ms_amber", cls="robot"))
    # solar roof on head
    a(g("ms_solar_front", "box", [0.42, 0.46, 0.02], pos=[1.62, 0, MODULE_TOP + 0.03], material="ms_solar",
        cls="robot"))
    # --- rear module (battery bay) ---
    rx0, rx1 = -2.1, -1.12
    a(g("ms_tail", "box", [(rx1 - rx0) / 2, 0.98, (MODULE_TOP - DECK_Z) / 2],
        pos=[(rx0 + rx1) / 2, 0, (MODULE_TOP + DECK_Z) / 2], material="ms_amber", cls="robot"))
    a(g(None, "box", [(rx1 - rx0) / 2 + 0.005, 0.985, 0.07], pos=[(rx0 + rx1) / 2, 0, MODULE_TOP - 0.07],
        material="ms_white", cls="robot"))
    for k in range(6):  # louvres
        z = 1.55 + k * 0.1
        for s in (-1, 1):
            a(g(None, "box", [0.3, 0.006, 0.018], pos=[-1.6, s * 0.988, z], material="ms_black", cls="robot"))
    for s in (-1, 1):
        a(g(None, "box", [0.012, 0.12, 0.05], pos=[rx0 - 0.012, s * 0.75, 1.62], material="ms_tail", cls="robot"))
    a(g(None, "box", [0.08, 1.02, 0.09], pos=[rx0 - 0.08, 0, FRAME_Z0 + 0.12], material="ms_dark", cls="robot"))
    # rear ladder
    for z in (1.15, 1.45, 1.75, 2.05):
        a(g(None, "capsule", [0.018], fromto=[rx0 - 0.08, -0.25, z, rx0 - 0.08, 0.25, z][:6], material="ms_steel",
            cls="robot"))
    for s in (-1, 1):
        a(g(None, "capsule", [0.022], fromto=[rx0 - 0.08, s * 0.25, 1.05, rx0 - 0.08, s * 0.25, 2.2],
            material="ms_steel", cls="robot"))
    # --- hopper (open top, white with amber band) ---
    hx, hy = 1.0, 0.9
    hz0, hz1 = DECK_Z, 2.62
    t = 0.04
    for s in (-1, 1):
        a(g(f"ms_hopper_side{'L' if s > 0 else 'R'}", "box", [hx, t, (hz1 - hz0) / 2], pos=[0, s * (hy - t), (hz0 + hz1) / 2],
            material="ms_white", cls="robot"))
        a(g(None, "box", [t, hy, (hz1 - hz0) / 2], pos=[s * (hx - t), 0, (hz0 + hz1) / 2], material="ms_white",
            cls="robot"))
        a(g(None, "box", [hx + 0.01, 0.006, 0.09], pos=[0, s * (hy + 0.004), hz1 - 0.32], material="ms_amber",
            cls="robot"))
        a(g(None, "box", [hx + 0.01, 0.006, 0.025], pos=[0, s * (hy + 0.004), hz1 - 0.48], material="ms_black",
            cls="robot"))
    # hopper rim + ribs
    for s in (-1, 1):
        a(g(None, "box", [hx + 0.03, 0.045, 0.03], pos=[0, s * hy, hz1 + 0.02], material="ms_dark", cls="robot"))
        a(g(None, "box", [0.045, hy + 0.03, 0.03], pos=[s * hx, 0, hz1 + 0.02], material="ms_dark", cls="robot"))
        for x in (-0.5, 0.5):
            a(g(None, "box", [0.03, 0.012, (hz1 - hz0) / 2 - 0.05], pos=[x, s * (hy + 0.012), (hz0 + hz1) / 2],
                material="ms_dark", cls="robot"))
    a(g(None, "box", [hx - 0.04, hy - 0.04, 0.02], pos=[0, 0, hz0 + 0.05], material="ms_dark", cls="robot"))
    # hopper contents (payload-mode toggled)
    a(g("ms_dust_fill", "box", [hx - 0.05, hy - 0.05, 0.42], pos=[0, 0, 2.43 - 0.42], material="mulch_dust",
        cls="robot", rgba=None))
    for k, (x, y, sx, sy, sz) in enumerate([(0.3, 0.2, 0.55, 0.45, 0.16), (-0.35, -0.15, 0.5, 0.5, 0.13),
                                            (0.1, -0.4, 0.45, 0.35, 0.1), (-0.45, 0.45, 0.4, 0.35, 0.09)]):
        a(g(f"ms_dust_mound{k}", "ellipsoid", [sx, sy, sz], pos=[x, y, 2.43], material="mulch_dust", cls="robot"))
    a(g("ms_cart_base", "box", [hx - 0.05, hy - 0.05, 0.36], pos=[0, 0, 2.30 - 0.36], material="mulch_chips",
        cls="robot"))
    kk = 0
    for i, x in enumerate(np.linspace(-0.78, 0.78, 6)):
        for j, y in enumerate(np.linspace(-0.66, 0.66, 5)):
            hgt = 0.09 + 0.03 * ((i * 3 + j * 5) % 3)
            a(g(f"ms_cart_stack{kk}", "cylinder", [0.075, hgt], pos=[x, y, 2.30 + hgt], material="puck_brown",
                cls="robot"))
            a(g(f"ms_cart_cap{kk}", "cylinder", [0.073, 0.012], pos=[x, y, 2.30 + 2 * hgt + 0.01],
                material="puck_green", cls="robot"))
            kk += 1
    # hopper side logo panel
    for s in (-1, 1):
        a(g(None, "box", [0.55, 0.004, 0.11], pos=[0.1, s * (hy + 0.01), 1.82], material="ms_dark", cls="robot"))
        a(g(None, "box", [0.5, 0.005, 0.03], pos=[0.1, s * (hy + 0.012), 1.86], material="ms_coil", cls="robot"))
    # auger pipes from hopper to front/rear pads
    for sx in (-1, 1):
        for sy in (-1, 1):
            a(g(None, "capsule", [0.06], fromto=[sx * 0.9, sy * 0.55, 1.75, sx * 1.35, sy * 0.95, PAD_TOP_Z - 0.12],
                material="ms_steel", cls="robot"))
    # --- docking pads (cantilevered) ---
    for k, (x, y) in enumerate(DOCKS):
        sy = np.sign(y)
        a(g(f"ms_dock{k}_arm", "box", [0.2, 0.32, 0.05], pos=[x, sy * 0.72, PAD_TOP_Z - 0.12], material="ms_dark",
            cls="robot"))
        a(g(f"ms_dock{k}_strut", "capsule", [0.04], fromto=[x, sy * 0.95, MODULE_TOP - 0.25, x, y, PAD_TOP_Z - 0.06],
            material="ms_steel", cls="robot"))
        a(g(f"ms_dock{k}_pad", "cylinder", [0.58, 0.035], pos=[x, y, PAD_TOP_Z - 0.035], material="ms_dark",
            cls="robot"))
        a(g(f"ms_dock{k}_coil", "cylinder", [0.47, 0.004], pos=[x, y, PAD_TOP_Z + 0.002], material="ms_coil",
            cls="robot"))
        a(g(None, "cylinder", [0.43, 0.005], pos=[x, y, PAD_TOP_Z + 0.004], material="ms_dark", cls="robot"))
        for (dx, dy, hx_, hy_) in ((-0.12, 0, 0.03, 0.17), (0.12, 0, 0.03, 0.17), (0, 0, 0.12, 0.03)):
            a(g(None, "box", [hx_, hy_, 0.003], pos=[x + dx, y + dy, PAD_TOP_Z + 0.009], material="ms_white",
                cls="robot"))
        a(f'<site name="ms_dock_{k}" pos="{f([x, y, PAD_TOP_Z])}" size="0.06" rgba="0.2 1 0.8 0"/>')
    # --- lidar mast ---
    a(g("ms_mast_pole", "cylinder", [0.045, 0.55], pos=[1.98, 0, MODULE_TOP + 0.55], material="ms_dark", cls="robot"))
    a(g(None, "box", [0.1, 0.12, 0.05], pos=[1.98, 0, MODULE_TOP + 0.05], material="ms_dark", cls="robot"))
    a(g("ms_lidar_body", "cylinder", [0.085, 0.06], pos=[1.98, 0, MODULE_TOP + 1.16], material="ms_black",
        cls="robot"))
    a(g(None, "cylinder", [0.087, 0.018], pos=[1.98, 0, MODULE_TOP + 1.16], material="ms_glass", cls="robot"))
    a(g(None, "cylinder", [0.06, 0.01], pos=[1.98, 0, MODULE_TOP + 1.23], material="ms_beacon", cls="robot"))
    a(f'<site name="ms_lidar" pos="{f([1.98, 0, MODULE_TOP + 1.16])}" size="0.05" rgba="1 0 0 0"/>')
    # GNSS mushrooms
    for y in (-0.35, 0.35):
        a(g(None, "cylinder", [0.008, 0.08], pos=[-1.95, y, MODULE_TOP + 0.08], material="ms_dark", cls="robot"))
        a(g(None, "ellipsoid", [0.07, 0.07, 0.03], pos=[-1.95, y, MODULE_TOP + 0.17], material="ms_white",
            cls="robot"))
    # beacons
    for (x, y) in ((2.0, 0.42), (2.0, -0.42), (-2.0, 0.6), (-2.0, -0.6)):
        a(g(None, "cylinder", [0.05, 0.02], pos=[x, y, MODULE_TOP + 0.02], material="ms_dark", cls="robot"))
        a(g(None, "cylinder", [0.04, 0.045], pos=[x, y, MODULE_TOP + 0.085], material="ms_beacon", cls="robot"))
    # --- battery carousel + swap arm on tail ---
    cx = -1.62
    a(g("ms_carousel", "cylinder", [0.36, 0.05], pos=[cx, 0, MODULE_TOP + 0.05], material="ms_dark", cls="robot"))
    for k in range(6):
        th = 2 * np.pi * k / 6
        p = [cx + 0.26 * np.cos(th), 0.26 * np.sin(th), MODULE_TOP + 0.17]
        q = quat_axis((0, 0, 1), th)
        a(g(None, "box", [0.08, 0.055, 0.07], pos=p, quat=q, material="ms_battery", cls="robot"))
        a(g(None, "box", [0.03, 0.03, 0.006], pos=[p[0], p[1], p[2] + 0.072], quat=q, material="ms_battery_ok",
            cls="robot"))
    a(g(None, "cylinder", [0.05, 0.12], pos=[cx, 0, MODULE_TOP + 0.2], material="ms_steel", cls="robot"))
    ab = np.array([-1.22, 0.0, MODULE_TOP + 0.04])
    a(g("ms_arm_base", "cylinder", [0.11, 0.06], pos=ab + [0, 0, 0.02], material="ms_dark", cls="robot"))
    j1 = ab + [0, 0, 0.42]
    j2 = j1 + [-0.05, 0.42, 0.18]
    tip = j2 + [-0.15, 0.22, -0.12]
    a(g("ms_arm_link1", "capsule", [0.055], fromto=np.concatenate([ab + [0, 0, 0.08], j1]), material="ms_white",
        cls="robot"))
    a(g(None, "sphere", [0.075], pos=j1, material="ms_amber", cls="robot"))
    a(g("ms_arm_link2", "capsule", [0.045], fromto=np.concatenate([j1, j2]), material="ms_white", cls="robot"))
    a(g(None, "sphere", [0.06], pos=j2, material="ms_amber", cls="robot"))
    a(g("ms_arm_link3", "capsule", [0.035], fromto=np.concatenate([j2, tip]), material="ms_dark", cls="robot"))
    for s in (-1, 1):
        a(g(None, "box", [0.015, 0.05, 0.04], pos=tip + [s * 0.04, 0.03, -0.03], material="ms_steel", cls="robot"))
    # --- portal legs, knuckles and wheels ---
    corners = {"fl": (WHEELBASE / 2, TRACK / 2), "fr": (WHEELBASE / 2, -TRACK / 2),
               "rl": (-WHEELBASE / 2, TRACK / 2), "rr": (-WHEELBASE / 2, -TRACK / 2)}
    for key, (x, y) in corners.items():
        side = 1 if y > 0 else -1
        inb = -side
        # fixed upper leg (frame to knuckle top)
        a(g(None, "box", [0.15, 0.08, 0.08], pos=[x, y + inb * 0.25, FRAME_Z0 + 0.02], material="ms_dark",
            cls="robot"))
        wheel = [f'<body name="ms_wheel_{key}" pos="0 0 0">',
                 f'<joint name="ms_wheel_{key}" type="hinge" axis="0 1 0" damping="0" armature="0.1"/>',
                 '<inertial pos="0 0 0" mass="80" diaginertia="4.65 8.1 4.65"/>']
        wheel += _wheel_geoms(f"ms_wheel_{key}", side)
        wheel.append("</body>")
        leg = [g(f"ms_leg_{key}", "box", [0.15, 0.065, 0.29], pos=[0, inb * 0.25, 0.27], material="ms_amber_dark",
                 cls="robot"),
               g(None, "cylinder", [0.17, 0.05], pos=[0, inb * 0.21, 0], zaxis=(0, 1, 0), material="ms_dark",
                 cls="robot"),
               g(None, "cylinder", [0.12, 0.03], pos=[0, inb * 0.18, 0], zaxis=(0, 1, 0), material="ms_steel",
                 cls="robot")]
        leg += _fender(f"ms_{key}", side, pos=(0, 0, 0))
        if key in ("fl", "fr"):
            a(f'<body name="ms_knuckle_{key}" pos="{f([x, y, WHEEL_R])}">')
            a(f'<joint name="ms_steer_{key}" type="hinge" axis="0 0 1" range="-0.6 0.6" limited="true" armature="1"/>')
            a('<inertial pos="0 0 0.2" mass="40" diaginertia="1.5 1.5 1"/>')
            a("\n".join(leg))
            a("\n".join(wheel))
            a("</body>")
        else:
            a(f'<body name="ms_axle_{key}" pos="{f([x, y, WHEEL_R])}">')
            a('<inertial pos="0 0 0.2" mass="40" diaginertia="1.5 1.5 1"/>')
            a("\n".join(leg))
            a("\n".join(wheel))
            a("</body>")
    # --- cameras ---
    pitch = np.deg2rad(15)
    a(f'<camera name="ms_front" pos="2.2 0 1.6" xyaxes="{cam_xyaxes([np.cos(pitch), 0, -np.sin(pitch)])}" fovy="70"/>')
    tgt = np.array([0.6, 0, 1.6])
    pm = np.array([-9.0, 0, 6.0])
    a(f'<camera name="ms_mast" pos="{f(pm)}" xyaxes="{cam_xyaxes(tgt - pm)}" fovy="55"/>')
    ps = np.array([2.3, 3.4, 0.75])
    ts = np.array([-0.5, 0.85, 0.42])
    a(f'<camera name="ms_side" pos="{f(ps)}" xyaxes="{cam_xyaxes(ts - ps)}" fovy="62"/>')
    a("</body>")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# Drones
# ---------------------------------------------------------------------------


def drone_xml(i: int, pos) -> tuple[str, str]:
    """Returns (body xml, tendon xml)."""
    acc = f"drone_accent{i}"
    glow = f"drone_glow{i}"
    p = f"drone{i}"
    L = []
    a = L.append
    a(f'<body name="{p}" pos="{f(pos)}">')
    a(f'<freejoint name="{p}_free"/>')
    a('<inertial pos="0 0 0" mass="45" diaginertia="6.3 6.3 10.8"/>')
    # fuselage
    a(g(f"{p}_body", "box", [0.34, 0.24, 0.1], pos=[0, 0, 0.0], material="drone_body", cls="robot"))
    a(g(f"{p}_canopy", "ellipsoid", [0.42, 0.29, 0.13], pos=[0.02, 0, 0.08], material=acc, cls="robot"))
    a(g(None, "ellipsoid", [0.2, 0.15, 0.05], pos=[0.12, 0, 0.19], material="drone_body", cls="robot"))
    a(g(f"{p}_battery", "box", [0.17, 0.13, 0.055], pos=[-0.16, 0, 0.2], material="drone_carbon", cls="robot"))
    a(g(None, "box", [0.172, 0.04, 0.057], pos=[-0.16, 0, 0.2], material=glow, cls="robot"))
    a(g(None, "box", [0.06, 0.1, 0.06], pos=[0.36, 0, -0.04], material="drone_carbon", cls="robot"))  # radar
    a(g(None, "sphere", [0.045], pos=[0.4, 0, -0.1], material="ms_glass", cls="robot"))  # gimbal cam
    # arms, motors
    for k in range(6):
        ang = k * np.pi / 3
        c, s = np.cos(ang), np.sin(ang)
        m = np.array([DRONE_ARM_R * c, DRONE_ARM_R * s, 0.04])
        a(g(f"{p}_arm{k}", "capsule", [0.032], fromto=[0.22 * c, 0.22 * s, 0.02, m[0] * 0.94, m[1] * 0.94, 0.04],
            material="drone_carbon", cls="robot"))
        a(g(None, "cylinder", [0.065, 0.055], pos=m, material="drone_body", cls="robot"))
        a(g(None, "cylinder", [0.067, 0.012], pos=m + [0, 0, 0.03], material=acc, cls="robot"))
        a(g(None, "capsule", [0.02], fromto=np.concatenate([m + [0, 0, -0.05], m + [0, 0, -0.12]]),
            material="drone_carbon", cls="robot"))
        # rotor
        a(f'<body name="{p}_rotor{k}_body" pos="{f(m + [0, 0, 0.09])}">')
        a(f'<joint name="{p}_rotor{k}" type="hinge" axis="0 0 1" armature="0.001"/>')
        a('<inertial pos="0 0 0" mass="0.3" diaginertia="0.008 0.008 0.015"/>')
        a(g(None, "cylinder", [0.035, 0.02], material="drone_body", cls="robot"))
        q0 = quat_axis((0, 0, 1), 0.0)
        for bs in (-1, 1):
            qb = quat_mul(q0, quat_axis((1, 0, 0), bs * 0.12))
            a(g(f"{p}_rotor{k}_blade{'a' if bs > 0 else 'b'}", "box", [0.16, 0.028, 0.004], pos=[bs * 0.18, 0, 0.005],
                quat=qb, material="drone_blade", cls="robot"))
            a(g(None, "box", [0.025, 0.029, 0.0045], pos=[bs * 0.325, 0, 0.005], quat=qb, material=acc, cls="robot"))
        a(g(f"{p}_rotor{k}_disc", "cylinder", [DRONE_PROP_R, 0.0015], pos=[0, 0, 0.004], rgba=[0.6, 0.6, 0.62, 0.10],
            cls="fxvis"))
        a("</body>")
    # landing gear
    for s in (-1, 1):
        a(g(f"{p}_skid{'L' if s > 0 else 'R'}", "capsule", [0.022], fromto=[-0.48, s * 0.34, -0.42, 0.48, s * 0.34, -0.42],
            material="drone_carbon", cls="robot"))
        for sx in (-1, 1):
            a(g(None, "capsule", [0.02], fromto=[sx * 0.22, s * 0.18, -0.08, sx * 0.3, s * 0.34, -0.42],
                material="drone_carbon", cls="robot"))
    a(g(None, "box", [0.012, 0.012, 0.012], pos=[0.45, 0.0, 0.0], material="led_white", cls="robot"))
    a(g(None, "sphere", [0.018], pos=[DRONE_ARM_R, 0, -0.03], material="led_green", cls="robot"))
    a(g(None, "sphere", [0.018], pos=[-DRONE_ARM_R, 0, -0.03], material="led_red", cls="robot"))
    for j, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
        a(f'<site name="{p}_winch{j}" pos="{f([sx * 0.13, sy * 0.1, -0.1])}" size="0.01" rgba="0 0 0 0"/>')
    a(g(None, "box", [0.16, 0.13, 0.02], pos=[0, 0, -0.11], material="drone_carbon", cls="robot"))  # winch housing
    # cradle on lift slide
    a(f'<body name="{p}_cradle" pos="0 0 {CRADLE_Z0}">')
    a(f'<joint name="{p}_lift" type="slide" axis="0 0 1" range="{LIFT_RANGE[0]} {LIFT_RANGE[1]}" limited="true" armature="1"/>')
    a('<inertial pos="0 0 -0.12" mass="8" diaginertia="0.15 0.15 0.15"/>')
    for j, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
        a(f'<site name="{p}_hook{j}" pos="{f([sx * 0.13, sy * 0.1, 0.0])}" size="0.01" rgba="0 0 0 0"/>')
    a(g(f"{p}_cradle_frame", "box", [0.2, 0.16, 0.015], pos=[0, 0, -0.01], material=acc, cls="robot"))
    # DUST: under-slung spreader bin
    a(g(f"{p}_dust_bin", "box", [0.18, 0.14, 0.08], pos=[0, 0, -0.1], material="ms_amber", cls="robot"))
    a(g(f"{p}_dust_hopper", "box", [0.12, 0.09, 0.04], pos=[0, 0, -0.2], material="ms_dark", cls="robot"))
    a(g(f"{p}_dust_fill", "ellipsoid", [0.16, 0.12, 0.04], pos=[0, 0, -0.025], material="mulch_dust", cls="robot"))
    a(g(f"{p}_dust_spinner", "cylinder", [0.14, 0.01], pos=[0, 0, -0.26], material="ms_steel", cls="robot"))
    for k in range(4):
        th = k * np.pi / 4
        a(g(f"{p}_dust_vane{k}", "box", [0.13, 0.006, 0.018], pos=[0, 0, -0.278], quat=quat_axis((0, 0, 1), th),
            material="ms_dark", cls="robot"))
    # CARTRIDGE: magazine of seed/mycelium pucks + release gate
    for k, y in enumerate((-0.09, 0.0, 0.09)):
        a(g(f"{p}_cart_tube{k}", "cylinder", [0.045, 0.12], pos=[0.0, y, -0.14], material="ms_glass", cls="robot"))
        for j in range(4):
            a(g(f"{p}_cart_puck{k}_{j}", "cylinder", [0.04, 0.022], pos=[0.0, y, -0.05 - j * 0.05],
                material="puck_green" if (j + k) % 2 == 0 else "puck_brown", cls="robot"))
    a(g(f"{p}_cart_rack", "box", [0.07, 0.15, 0.015], pos=[0.0, 0, -0.27], material="ms_dark", cls="robot"))
    a(g(f"{p}_cart_gate", "box", [0.05, 0.14, 0.01], pos=[0.05, 0, -0.29], material=acc, cls="robot"))
    a("</body>")
    pitch = np.deg2rad(22)
    a(f'<camera name="{p}_cam" pos="0.42 0 -0.12" xyaxes="{cam_xyaxes([np.sin(pitch), 0, -np.cos(pitch)], up_hint=(1, 0, 0))}" fovy="75"/>')
    a("</body>")
    tend = []
    for j in range(4):
        tend.append(f'<spatial name="{p}_cable{j}" width="0.007" rgba="0.12 0.12 0.13 1">'
                    f'<site site="{p}_winch{j}"/><site site="{p}_hook{j}"/></spatial>')
    return "\n".join(L), "\n".join(tend)


# ---------------------------------------------------------------------------
# FX pools (mocap)
# ---------------------------------------------------------------------------


def fx_xml(seed: int = 21) -> str:
    rng = np.random.default_rng(seed)
    L = []
    for k in range(N_FX_DUST):
        shade = rng.uniform(-0.05, 0.05)
        if k % 4 == 3:  # translucent dust puff
            s = rng.uniform(0.10, 0.20)
            sz = [s, s * rng.uniform(0.8, 1.0), s * rng.uniform(0.6, 0.8)]
            rgba = [0.52 + shade, 0.42 + shade, 0.31 + shade, 0.16]
        else:  # mulch speck
            s = rng.uniform(0.022, 0.045)
            sz = [s, s * rng.uniform(0.6, 0.9), s * rng.uniform(0.4, 0.7)]
            rgba = [0.32 + shade, 0.22 + shade * 0.8, 0.14 + shade * 0.5, 0.9]
        L.append(f'<body name="fx_dust_{k}" mocap="true" pos="{f([0, 0, FX_PARK_Z])}">'
                 f'{g(f"fx_dust_{k}", "ellipsoid", sz, rgba=rgba, cls="fxvis")}</body>')
    for k in range(N_FX_CART):
        L.append(f'<body name="fx_cart_{k}" mocap="true" pos="{f([0, 0, FX_PARK_Z])}">'
                 f'{g(f"fx_cart_{k}", "cylinder", [0.06, 0.035], material="puck_brown", cls="fx")}'
                 f'{g(f"fx_cart_{k}_seed", "cylinder", [0.045, 0.006], pos=[0, 0, 0.036], material="puck_green", cls="fx")}'
                 "</body>")
    return "\n".join(L)


@dataclass
class RobotNames:
    swarm_size: int

    @property
    def drones(self) -> list[str]:
        return [f"drone{i}" for i in range(self.swarm_size)]
