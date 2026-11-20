"""MJCF for the mothership near-field effectors (40 ft / 12.19 m reach).

* Snow-blower chip thrower: impeller housing rising out of the hopper centre,
  slewing discharge chute (hinge ``ms_chute_yaw``, 0 = throwing forward) with a
  hydraulic deflector hood (hinge ``ms_chute_pitch``, + = higher throw) and the
  throw-origin site ``ms_chute_tip``; spinning impeller ``ms_impeller``.
* Twin pneumatic air cannons, port (0) / starboard (1): turret yaw
  ``ms_cannon{s}_yaw`` (0 = forward), elevation ``ms_cannon{s}_pitch`` (+ = up),
  two barrels with recoil slides ``ms_cannon{s}_recoil{b}`` and muzzle sites
  ``ms_cannon{s}_muzzle{b}``; cartridge magazine + air receiver tank + hose.
* Mocap FX pools: chip flakes ``fx_blow_*`` + translucent dust plume
  ``fx_bdust_*``, flying / landed cartridges ``fx_pellet_*`` with tracer specks
  ``fx_trail_*``, muzzle blast puffs ``fx_puff_*``.

Geometry anchors come from :mod:`prunemaxxing.planning.ms_effectors` so the
mission model, the dashboard and the 3D replay agree on where material leaves
the machine.
"""

from __future__ import annotations

import math

import numpy as np

from prunemaxxing.planning.ms_effectors import CANNON_BARREL_M, CANNON_PIVOTS, CHUTE_PIVOT
from prunemaxxing.sim.world_robots import FX_PARK_Z, f, g, quat_axis

N_FX_BLOW = 320
N_FX_BDUST = 64
N_FX_PELLET = 32
N_FX_TRAIL = 160
N_FX_PUFF = 8

HOPPER_TOP = 2.62
HOPPER_HY = 0.9


def _chute(L: list[str]) -> None:
    a = L.append
    px, py, pz = CHUTE_PIVOT
    hz = HOPPER_TOP
    # impeller housing standing in the chip load, with intake slots + spinning vanes
    a(g("ms_blower_housing", "cylinder", [0.30, 0.15], pos=[px, py, hz], material="ms_white", cls="robot"))
    a(g(None, "cylinder", [0.312, 0.022], pos=[px, py, hz + 0.12], material="ms_amber", cls="robot"))
    a(g(None, "cylinder", [0.312, 0.012], pos=[px, py, hz - 0.08], material="ms_dark", cls="robot"))
    for k in range(12):
        th = 2 * math.pi * k / 12
        a(g(None, "box", [0.008, 0.045, 0.07], pos=[px + 0.302 * math.cos(th), py + 0.302 * math.sin(th), hz - 0.005],
            quat=quat_axis((0, 0, 1), th), material="ms_black", cls="robot"))
    a(f'<body name="ms_impeller_body" pos="{f([px, py, hz + 0.155])}">')
    a('<joint name="ms_impeller" type="hinge" axis="0 0 1" limited="false"/>')
    a(g(None, "cylinder", [0.27, 0.006], material="ms_dark", cls="robot"))
    for k in range(6):
        th = 2 * math.pi * k / 6
        a(g(None, "box", [0.06, 0.012, 0.018], pos=[0.21 * math.cos(th), 0.21 * math.sin(th), 0.02],
            quat=quat_axis((0, 0, 1), th + 0.5), material="ms_steel", cls="robot"))
    a("</body>")
    # neck + slewing ring
    a(g("ms_chute_neck", "cylinder", [0.15, (pz - hz - 0.17) / 2], pos=[px, py, (pz + hz + 0.17) / 2],
        material="ms_dark", cls="robot"))
    a(g("ms_chute_ring", "cylinder", [0.215, 0.025], pos=[px, py, pz - 0.025], material="ms_steel", cls="robot"))
    for k in range(16):
        th = 2 * math.pi * k / 16
        a(g(None, "box", [0.014, 0.012, 0.022], pos=[px + 0.222 * math.cos(th), py + 0.222 * math.sin(th), pz - 0.025],
            quat=quat_axis((0, 0, 1), th), material="ms_dark", cls="robot"))

    a(f'<body name="ms_chute" pos="{f([px, py, pz])}">')
    a('<joint name="ms_chute_yaw" type="hinge" axis="0 0 1" limited="false"/>')
    a(g(None, "cylinder", [0.18, 0.02], pos=[0, 0, 0.02], material="ms_dark", cls="robot"))
    a(g("ms_chute_tube", "cylinder", [0.165, 0.13], pos=[0, 0, 0.17], material="ms_amber", cls="robot"))
    # bend from vertical to 25 deg above horizontal (snow-blower spout)
    R, z0 = 0.34, 0.30
    phis = np.radians(np.linspace(0.0, 65.0, 7))
    for p0, p1 in zip(phis[:-1], phis[1:]):
        pm = 0.5 * (p0 + p1)
        pos = [R - R * math.cos(pm), 0.0, z0 + R * math.sin(pm)]
        a(g(None, "cylinder", [0.165, R * (p1 - p0) / 2 + 0.012], pos=pos, zaxis=[math.sin(pm), 0, math.cos(pm)],
            material="ms_amber", cls="robot"))
    pe = phis[-1]
    end = np.array([R - R * math.cos(pe), 0.0, z0 + R * math.sin(pe)])
    d_end = np.array([math.sin(pe), 0.0, math.cos(pe)])
    up_n = np.array([-math.cos(pe), 0.0, math.sin(pe)])  # spout "roof" normal
    a(g(None, "cylinder", [0.172, 0.018], pos=end, zaxis=d_end, material="ms_dark", cls="robot"))
    a(g(None, "cylinder", [0.145, 0.004], pos=end + 0.02 * d_end, zaxis=d_end, material="ms_black", cls="robot"))
    # side "40 FT" reach stripes
    for sgn in (1, -1):
        a(g(None, "box", [0.10, 0.004, 0.025], pos=[0.0, sgn * 0.168, 0.22], material="ms_black", cls="robot"))
    # hydraulic ram for the deflector
    a(g(None, "capsule", [0.022], fromto=np.concatenate([[0.05, 0.0, 0.40] + up_n * 0.12, end + up_n * 0.2]),
        material="ms_steel", cls="robot"))
    hinge = end + up_n * 0.16
    a(f'<body name="ms_chute_deflector" pos="{f(hinge)}">')
    a('<joint name="ms_chute_pitch" type="hinge" axis="0 -1 0" range="-0.4 0.7"/>')
    for j in range(6):
        ang = math.radians(-6.0 * j)
        dd = d_end * math.cos(ang) + up_n * math.sin(ang)
        pos = d_end * (0.04 + 0.058 * j) - up_n * (0.0045 * j * j)
        a(g(None, "box", [0.042, 0.175, 0.007], pos=pos, zaxis=np.cross(dd, [0, 1, 0]), material="ms_amber",
            cls="robot"))
    tip = d_end * 0.34 - up_n * 0.16
    a(f'<site name="ms_chute_tip" pos="{f(tip)}" size="0.03" rgba="1 0.8 0.2 0"/>')
    a("</body>")
    a("</body>")


def _cannon(L: list[str], s: int) -> None:
    a = L.append
    cx, cy, cz = CANNON_PIVOTS[s]
    side = 1.0 if cy > 0 else -1.0
    base_z = cz - 0.12
    # bracket off the hopper wall
    span = abs(cy) - HOPPER_HY
    a(g(None, "box", [0.17, span / 2 + 0.02, 0.025], pos=[cx, side * (HOPPER_HY + span / 2), base_z - 0.06],
        material="ms_dark", cls="robot"))
    for dx in (-0.12, 0.12):
        a(g(None, "capsule", [0.018], fromto=[cx + dx, side * (HOPPER_HY + 0.01), base_z - 0.38,
                                              cx + dx, side * abs(cy), base_z - 0.08], material="ms_dark", cls="robot"))
    # air receiver tank (between the wheels, under the deck edge) + hose
    a(g(f"ms_air_tank{s}", "capsule", [0.12], fromto=[-0.55, side * 1.06, 1.17, 0.55, side * 1.06, 1.17],
        material="ms_white", cls="robot"))
    for x in (-0.35, 0.35):
        a(g(None, "cylinder", [0.124, 0.02], pos=[x, side * 1.06, 1.17], zaxis=[1, 0, 0], material="ms_amber",
            cls="robot"))
    a(g(None, "cylinder", [0.04, 0.012], pos=[0.0, side * 1.185, 1.2], zaxis=[0, side, 0], material="ms_white",
        cls="robot"))
    a(g(None, "cylinder", [0.032, 0.004], pos=[0.0, side * 1.198, 1.2], zaxis=[0, side, 0], material="ms_black",
        cls="robot"))
    a(g(None, "capsule", [0.022], fromto=[0.2, side * 1.1, 1.28, cx + 0.05, side * (abs(cy) - 0.02), base_z - 0.05],
        material="ms_black", cls="robot"))

    a(f'<body name="ms_cannon{s}" pos="{f([cx, cy, base_z])}">')
    a(f'<joint name="ms_cannon{s}_yaw" type="hinge" axis="0 0 1" limited="false"/>')
    a(g(None, "cylinder", [0.15, 0.03], pos=[0, 0, -0.01], material="ms_dark", cls="robot"))
    a(g(None, "cylinder", [0.153, 0.005], pos=[0, 0, 0.022], material="ms_coil", cls="robot"))
    a(g(f"ms_cannon{s}_housing", "box", [0.13, 0.125, 0.055], pos=[-0.01, 0, 0.07], material="ms_amber", cls="robot"))
    for sy in (0.12, -0.12):
        a(g(None, "box", [0.07, 0.012, 0.06], pos=[0.0, sy, 0.12], material="ms_dark", cls="robot"))
    a('<body name="ms_cannon%d_elev" pos="0 0 0.12">' % s)
    a(f'<joint name="ms_cannon{s}_pitch" type="hinge" axis="0 -1 0" range="-0.2 0.9"/>')
    a(g(None, "cylinder", [0.03, 0.125], zaxis=[0, 1, 0], material="ms_steel", cls="robot"))
    a(g(f"ms_cannon{s}_breech", "box", [0.11, 0.11, 0.06], pos=[-0.03, 0, 0.0], material="ms_dark", cls="robot"))
    a(g(None, "capsule", [0.05], fromto=[-0.16, 0, -0.075, 0.30, 0, -0.075], material="ms_white", cls="robot"))
    a(g(None, "cylinder", [0.052, 0.012], pos=[0.12, 0, -0.075], zaxis=[1, 0, 0], material="ms_amber", cls="robot"))
    # cartridge magazine (clear tube, pucks visible)
    a(g(None, "cylinder", [0.05, 0.15], pos=[-0.07, 0, 0.21], rgba=[0.78, 0.88, 0.92, 0.33], cls="robot"))
    a(g(None, "cylinder", [0.053, 0.012], pos=[-0.07, 0, 0.365], material="ms_dark", cls="robot"))
    for k in range(5):
        a(g(None, "cylinder", [0.036, 0.011], pos=[-0.07, 0, 0.09 + 0.05 * k],
            material="puck_brown" if k % 2 else "puck_green", cls="robot"))
    for b, by in enumerate((0.07, -0.07)):
        a(f'<body name="ms_cannon{s}_barrel{b}" pos="0 {by} 0.02">')
        a(f'<joint name="ms_cannon{s}_recoil{b}" type="slide" axis="1 0 0" range="-0.15 0"/>')
        a(g(f"ms_cannon{s}_tube{b}", "cylinder", [0.05, (CANNON_BARREL_M - 0.06) / 2],
            pos=[(CANNON_BARREL_M + 0.06) / 2, 0, 0], zaxis=[1, 0, 0], material="ms_steel", cls="robot"))
        a(g(None, "cylinder", [0.066, 0.04], pos=[CANNON_BARREL_M - 0.04, 0, 0], zaxis=[1, 0, 0],
            material="ms_dark", cls="robot"))
        a(g(None, "cylinder", [0.056, 0.022], pos=[0.36, 0, 0], zaxis=[1, 0, 0], material="ms_amber", cls="robot"))
        a(g(None, "cylinder", [0.042, 0.003], pos=[CANNON_BARREL_M + 0.001, 0, 0], zaxis=[1, 0, 0],
            material="ms_black", cls="robot"))
        a(f'<site name="ms_cannon{s}_muzzle{b}" pos="{f([CANNON_BARREL_M, 0, 0])}" size="0.02" rgba="1 0.5 0 0"/>')
        a("</body>")
    a("</body>")
    a("</body>")


def effector_bodies_xml() -> str:
    """Geoms/bodies to insert inside the ``mothership`` body."""
    L: list[str] = []
    _chute(L)
    for s in range(len(CANNON_PIVOTS)):
        _cannon(L, s)
    return "\n".join(L)


def effector_fx_xml(seed: int = 33) -> str:
    rng = np.random.default_rng(seed)
    L = []
    for k in range(N_FX_BLOW):
        sh = rng.uniform(-0.06, 0.06)
        sz = [rng.uniform(0.022, 0.042), rng.uniform(0.012, 0.024), rng.uniform(0.004, 0.007)]
        q = quat_axis(rng.normal(size=3), rng.uniform(0, math.pi))
        rgba = [0.70 + sh, 0.52 + sh, 0.30 + 0.6 * sh, 1.0] if k % 5 else [0.45 + sh, 0.32 + sh, 0.2, 1.0]
        L.append(f'<body name="fx_blow_{k}" mocap="true" pos="{f([0, 0, FX_PARK_Z])}" quat="{f(q)}">'
                 f'{g(f"fx_blow_{k}", "box", sz, rgba=rgba, cls="fxvis")}</body>')
    for k in range(N_FX_BDUST):
        r = rng.uniform(0.22, 0.42)
        sh = rng.uniform(-0.04, 0.04)
        L.append(f'<body name="fx_bdust_{k}" mocap="true" pos="{f([0, 0, FX_PARK_Z])}">'
                 f'{g(f"fx_bdust_{k}", "sphere", [r], rgba=[0.70 + sh, 0.58 + sh, 0.42 + sh, 0.085], cls="fxvis")}</body>')
    for k in range(N_FX_TRAIL):
        j = k % 8
        L.append(f'<body name="fx_trail_{k}" mocap="true" pos="{f([0, 0, FX_PARK_Z])}">'
                 f'{g(f"fx_trail_{k}", "sphere", [0.06 - 0.005 * j], rgba=[1.0, 0.97, 0.88, 0.85 - 0.08 * j], cls="fxvis")}'
                 "</body>")
    for k in range(N_FX_PELLET):
        L.append(f'<body name="fx_pellet_{k}" mocap="true" pos="{f([0, 0, FX_PARK_Z])}">'
                 f'{g(f"fx_pellet_{k}", "cylinder", [0.045, 0.02], material="puck_brown", cls="fx")}'
                 f'{g(f"fx_pellet_{k}_seed", "cylinder", [0.032, 0.005], pos=[0, 0, 0.021], material="puck_green", cls="fx")}'
                 "</body>")
    for k in range(N_FX_PUFF):
        r = rng.uniform(0.13, 0.2)
        L.append(f'<body name="fx_puff_{k}" mocap="true" pos="{f([0, 0, FX_PARK_Z])}">'
                 f'{g(f"fx_puff_{k}", "sphere", [r], rgba=[0.96, 0.96, 0.93, 0.30], cls="fxvis")}</body>')
    return "\n".join(L)
