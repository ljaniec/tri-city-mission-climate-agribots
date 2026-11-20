"""Primitive-geom props: masia, depot vehicles/props, hidden obstacles, highlight marks.

Every function returns MJCF snippets in *world* coordinates.  Base heights come
from a ``ground(x, y)`` callable (terrain height), so nothing floats.
"""

from __future__ import annotations

import numpy as np

from prunemaxxing.sim.world_robots import f, g, quat_axis, quat_mul


def _R(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]])


class Placer:
    """Places geoms given in a local (x fwd, y left, z up) frame at (cx, cy, yaw)."""

    def __init__(self, cx, cy, yaw, ground, cls, prefix=None, z0=None):
        self.cx, self.cy, self.yaw = cx, cy, yaw
        self.R = _R(yaw)
        self.ground = ground
        self.z0 = float(ground(cx, cy)) if z0 is None else z0
        self.cls = cls
        self.prefix = prefix
        self.out: list[str] = []
        self.k = 0

    def world(self, p):
        xy = self.R @ np.asarray(p[:2], float)
        return np.array([self.cx + xy[0], self.cy + xy[1], self.z0 + p[2]])

    def q(self, quat=None):
        qy = quat_axis((0, 0, 1), self.yaw)
        return qy if quat is None else quat_mul(qy, quat)

    def add(self, type, size, pos=None, quat=None, zaxis=None, material=None, rgba=None, name=None, fromto=None, extra=""):
        nm = None
        if self.prefix is not None:
            nm = f"{self.prefix}_{name or type + str(self.k)}"
            self.k += 1
        if fromto is not None:
            a = self.world(fromto[:3])
            b = self.world(fromto[3:])
            self.out.append(g(nm, type, size, fromto=np.concatenate([a, b]), material=material, rgba=rgba,
                              cls=self.cls, extra=extra))
            return
        if zaxis is not None:
            z = np.asarray(zaxis, float)
            zxy = self.R @ z[:2]
            zaxis = [zxy[0], zxy[1], z[2]]
            self.out.append(g(nm, type, size, pos=self.world(pos), zaxis=zaxis, material=material, rgba=rgba,
                              cls=self.cls, extra=extra))
            return
        self.out.append(g(nm, type, size, pos=self.world(pos), quat=self.q(quat), material=material, rgba=rgba,
                          cls=self.cls, extra=extra))


def prop_assets() -> str:
    return "\n".join([
        '<material name="plaster" texture="tex_plaster" texuniform="true" texrepeat="0.25 0.25" specular="0.1"/>',
        '<material name="plaster_ochre" texture="tex_ochre" texuniform="true" texrepeat="0.25 0.25" specular="0.1"/>',
        '<material name="plaster_flat" rgba="0.90 0.86 0.78 1" specular="0.1"/>',
        '<material name="ochre_flat" rgba="0.86 0.68 0.44 1" specular="0.1"/>',
        '<material name="roof_tile" texture="tex_roof" texuniform="true" texrepeat="0.5 0.5" specular="0.15"/>',
        '<material name="stone" rgba="0.70 0.64 0.55 1" specular="0.1"/>',
        '<material name="stone_dark" rgba="0.52 0.47 0.40 1" specular="0.1"/>',
        '<material name="window_glass" rgba="0.10 0.12 0.14 1" specular="0.9" shininess="0.9"/>',
        '<material name="shutter_green" rgba="0.24 0.38 0.28 1" specular="0.2"/>',
        '<material name="wood_door" rgba="0.36 0.24 0.15 1" specular="0.15"/>',
        '<material name="container" texture="tex_container" texuniform="true" texrepeat="0.4 0.4" specular="0.3"/>',
        '<material name="truck_white" rgba="0.92 0.92 0.9 1" specular="0.6" shininess="0.6"/>',
        '<material name="truck_orange" rgba="0.86 0.36 0.08 1" specular="0.5" shininess="0.5"/>',
        '<material name="truck_grey" rgba="0.35 0.36 0.38 1" specular="0.4"/>',
        '<material name="loader_yellow" rgba="0.95 0.72 0.08 1" specular="0.5"/>',
        '<material name="rubber" rgba="0.07 0.07 0.075 1" specular="0.1"/>',
        '<material name="rim_light" rgba="0.78 0.78 0.74 1" specular="0.7"/>',
        '<material name="pallet_wood" rgba="0.70 0.56 0.36 1" specular="0.05"/>',
        '<material name="bigbag" rgba="0.90 0.89 0.84 1" specular="0.15"/>',
        '<material name="tractor_red" rgba="0.62 0.13 0.09 1" specular="0.35" shininess="0.4"/>',
        '<material name="tractor_rust" rgba="0.42 0.22 0.12 1" specular="0.1"/>',
        '<material name="tractor_grey" rgba="0.30 0.31 0.31 1" specular="0.3"/>',
        '<material name="tractor_cream" rgba="0.86 0.82 0.70 1" specular="0.3"/>',
        '<material name="pump_blue" rgba="0.12 0.30 0.62 1" specular="0.5"/>',
        '<material name="pump_green" rgba="0.18 0.42 0.25 1" specular="0.4"/>',
        '<material name="pvc_blue" rgba="0.18 0.45 0.80 1" specular="0.6" shininess="0.7"/>',
        '<material name="alu" rgba="0.80 0.82 0.84 1" specular="0.9" shininess="0.8"/>',
        '<material name="galv" rgba="0.66 0.68 0.70 1" specular="0.7"/>',
        '<material name="water_puddle" rgba="0.36 0.44 0.48 1" reflectance="0.45" specular="1" shininess="1"/>',
        '<material name="water_sheen" rgba="0.55 0.66 0.72 0.35" specular="1" shininess="1"/>',
        '<material name="mark_fill" rgba="1 0.35 0.05 1" emission="1"/>',
        '<material name="mark_line" rgba="1 0.25 0.02 1" emission="1"/>',
    ])


# ---------------------------------------------------------------------------
# Masia (Catalan farmhouse)
# ---------------------------------------------------------------------------


def _masia_dims(poly, height, main):
    x0, y0, x1, y1 = poly.bounds
    Lx, Ly = x1 - x0, y1 - y0
    He = height * (0.70 if main else 0.72)
    pitch = np.deg2rad(21 if main else 14)
    rise = (Ly / 2) * np.tan(pitch) if main else Ly * np.tan(pitch)
    return Lx, Ly, He, pitch, rise


def masia_assets(poly, height, main: bool, idx: int) -> list[str]:
    """Inline triangular-prism mesh for the gable/attic volume of house `idx`."""
    Lx, Ly, He, pitch, rise = _masia_dims(poly, height, main)
    hx, hy = Lx / 2 - 0.02, Ly / 2 - 0.02
    top_y = 0.0 if main else hy
    v = [(-hx, -hy, 0), (hx, -hy, 0), (-hx, hy, 0), (hx, hy, 0), (-hx, top_y, rise - 0.06), (hx, top_y, rise - 0.06)]
    return [f'<mesh name="masia_gable_{idx}" vertex="{" ".join(f"{c:.4f}" for p in v for c in p)}"/>']


def masia(poly, height, ground, main=True, idx=0) -> list[str]:
    x0, y0, x1, y1 = poly.bounds
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    Lx, Ly, He, pitch, rise = _masia_dims(poly, height, main)
    z0 = float(min(ground(x0, y0), ground(x1, y0), ground(x0, y1), ground(x1, y1))) - 0.15
    P = Placer(cx, cy, 0.0, ground, "decor", z0=z0)
    wall_mat = "plaster" if main else "plaster_ochre"
    flat = "plaster_flat" if main else "ochre_flat"
    P.add("box", [Lx / 2, Ly / 2, He / 2], [0, 0, He / 2], material=wall_mat)
    P.add("box", [Lx / 2 + 0.06, Ly / 2 + 0.06, 0.35], [0, 0, 0.35], material="stone_dark")
    for sx in (-1, 1):
        for sy in (-1, 1):
            P.add("box", [0.3, 0.3, He / 2], [sx * (Lx / 2 - 0.25), sy * (Ly / 2 - 0.25), He / 2 + 0.01],
                  material="stone")
    P.out.append(f'<geom class="decor" type="mesh" mesh="masia_gable_{idx}" material="{flat}" '
                 f'pos="{cx:.4f} {cy:.4f} {z0 + He:.4f}"/>')
    if main:
        slab_w = (Ly / 2) / np.cos(pitch) + 0.55
        for sy in (-1, 1):
            q = quat_axis((1, 0, 0), -sy * pitch)  # each slab falls toward its eave
            P.add("box", [Lx / 2 + 0.45, slab_w / 2, 0.09],
                  [0, sy * (Ly / 4 + 0.1), He + rise / 2 + 0.06], quat=q, material="roof_tile")
        P.add("box", [Lx / 2 + 0.47, 0.18, 0.1], [0, 0, He + rise + 0.03], material="roof_tile")
        # chimney
        P.add("box", [0.45, 0.45, 1.3], [Lx * 0.22, Ly * 0.12, He + rise * 0.7 + 0.9], material=flat)
        P.add("box", [0.55, 0.55, 0.08], [Lx * 0.22, Ly * 0.12, He + rise * 0.7 + 2.2], material="roof_tile")
        # south facade: arched door, 2 floors of shuttered windows, attic 'golfes'
        fy = -Ly / 2 - 0.02
        P.add("box", [0.95, 0.12, 1.25], [0, fy, 1.25], material="wood_door")
        P.add("cylinder", [0.95, 0.12], [0, fy, 2.5], zaxis=(0, 1, 0), material="wood_door")
        P.add("cylinder", [1.15, 0.1], [0, fy + 0.02, 2.5], zaxis=(0, 1, 0), material="stone")
        P.add("box", [1.15, 0.1, 1.25], [0, fy + 0.02, 1.25], material="stone")
        cols = np.linspace(-Lx / 2 + 3.2, Lx / 2 - 3.2, 5)
        for fl, zc in enumerate((1.6, 4.0)):
            for xc in cols:
                if fl == 0 and abs(xc) < 1.5:
                    continue
                P.add("box", [0.62, 0.08, 0.82], [xc, fy, zc], material="stone")
                P.add("box", [0.48, 0.1, 0.68], [xc, fy, zc], material="window_glass")
                for s in (-1, 1):
                    P.add("box", [0.26, 0.05, 0.7], [xc + s * 0.78, fy - 0.03, zc], material="shutter_green")
        for xc in np.linspace(-Lx / 2 + 4, Lx / 2 - 4, 6):
            P.add("box", [0.28, 0.1, 0.32], [xc, fy, He - 0.55], material="window_glass")
            P.add("cylinder", [0.28, 0.1], [xc, fy, He - 0.23], zaxis=(0, 1, 0), material="window_glass")
        # other facades: plain windows
        for (sx, sy, along) in ((0, 1, "x"), (-1, 0, "y"), (1, 0, "y")):
            n = 4 if along == "x" else 3
            span = Lx if along == "x" else Ly
            for zc in (1.6, 4.0):
                for t in np.linspace(-span / 2 + 3, span / 2 - 3, n):
                    if along == "x":
                        P.add("box", [0.45, 0.1, 0.62], [t, sy * (Ly / 2 + 0.02), zc], material="window_glass")
                    else:
                        P.add("box", [0.1, 0.45, 0.62], [sx * (Lx / 2 + 0.02), t, zc], material="window_glass")
        # sundial-ish ochre panel + bench
        P.add("box", [0.6, 0.04, 0.6], [Lx * 0.3, fy - 0.02, He - 1.0], material="ochre_flat")
        P.add("box", [1.2, 0.25, 0.22], [-3.0, fy - 0.6, 0.22], material="stone")
    else:
        q = quat_axis((1, 0, 0), pitch)
        P.add("box", [Lx / 2 + 0.35, Ly / 2 / np.cos(pitch) + 0.35, 0.08], [0, 0, He + rise / 2 + 0.05], quat=q,
              material="roof_tile")
        P.add("box", [1.6, 0.1, 1.6], [-Lx * 0.15, -Ly / 2 - 0.02, 1.6], material="wood_door")
        for xc in (Lx * 0.25, Lx * 0.4):
            P.add("box", [0.4, 0.1, 0.5], [xc, -Ly / 2 - 0.02, 2.4], material="window_glass")
    return P.out


# ---------------------------------------------------------------------------
# Depot props
# ---------------------------------------------------------------------------


def _wheel(P, x, y, z, r, w, mat="rubber", rim="rim_light"):
    P.add("cylinder", [r, w / 2], [x, y, z], zaxis=(0, 1, 0), material=mat)
    P.add("cylinder", [r * 0.55, w / 2 + 0.01], [x, y, z], zaxis=(0, 1, 0), material=rim)


def dump_truck(cx, cy, yaw, ground) -> list[str]:
    P = Placer(cx, cy, yaw, ground, "decor")
    # chassis
    P.add("box", [3.6, 0.45, 0.15], [0, 0, 0.95], material="truck_grey")
    for (x, r) in ((2.6, 0.52), (-1.2, 0.52), (-2.6, 0.52)):
        for s in (-1, 1):
            _wheel(P, x, s * 1.0, r, r, 0.38 if x > 0 else 0.6)
    # cab
    P.add("box", [0.95, 1.22, 1.05], [2.95, 0, 1.95], material="truck_white")
    P.add("box", [0.04, 1.05, 0.45], [3.91, 0, 2.3], material="window_glass")
    for s in (-1, 1):
        P.add("box", [0.5, 0.03, 0.4], [3.1, s * 1.23, 2.35], material="window_glass")
        P.add("box", [0.04, 0.04, 0.25], [3.95, s * 1.3, 2.3], material="truck_grey")
    P.add("box", [0.08, 1.15, 0.3], [3.95, 0, 1.2], material="truck_grey")
    P.add("box", [0.03, 0.18, 0.08], [4.0, 0.85, 1.45], material="led_white")
    P.add("box", [0.03, 0.18, 0.08], [4.0, -0.85, 1.45], material="led_white")
    P.add("box", [0.9, 1.25, 0.08], [2.95, 0, 3.02], material="truck_white")
    P.add("cylinder", [0.08, 0.7], [2.0, 0.95, 2.4], material="galv")
    # tipper body (orange) with chip load
    q = quat_axis((0, 1, 0), -0.05)
    P.add("box", [2.35, 1.22, 0.06], [-0.75, 0, 1.18], quat=q, material="truck_orange")
    for s in (-1, 1):
        P.add("box", [2.35, 0.05, 0.62], [-0.75, s * 1.22, 1.8], quat=q, material="truck_orange")
    P.add("box", [0.05, 1.22, 0.75], [1.62, 0, 1.92], material="truck_orange")
    P.add("box", [0.05, 1.22, 0.62], [-3.12, 0, 1.75], material="truck_orange")
    P.add("box", [2.3, 1.17, 0.05], [-0.75, 0, 2.18], quat=q, material="mulch_chips")
    P.add("ellipsoid", [1.6, 0.95, 0.35], [-0.6, 0, 2.2], material="mulch_chips")
    return P.out


def skid_loader(cx, cy, yaw, ground) -> list[str]:
    P = Placer(cx, cy, yaw, ground, "decor")
    for x in (-0.55, 0.55):
        for s in (-1, 1):
            _wheel(P, x, s * 0.8, 0.38, 0.38, 0.32)
    P.add("box", [1.05, 0.62, 0.35], [0, 0, 0.75], material="loader_yellow")
    P.add("box", [0.55, 0.6, 0.55], [0.1, 0, 1.6], material="window_glass")
    P.add("box", [0.6, 0.65, 0.05], [0.1, 0, 2.18], material="loader_yellow")
    for s in (-1, 1):
        P.add("box", [0.06, 0.03, 0.55], [0.6, s * 0.6, 1.6], material="truck_grey")
        P.add("capsule", [0.07], fromto=[-0.6, s * 0.72, 1.6, 1.15, s * 0.72, 0.7], material="loader_yellow")
    P.add("box", [0.35, 0.85, 0.3], [1.4, 0, 0.42], material="truck_grey")
    P.add("box", [0.2, 0.8, 0.12], [1.3, 0, 0.6], material="mulch_chips")
    return P.out


def container_office(cx, cy, yaw, ground) -> list[str]:
    P = Placer(cx, cy, yaw, ground, "decor")
    P.add("box", [3.0, 1.2, 1.3], [0, 0, 1.38], material="container")
    P.add("box", [3.02, 1.22, 0.06], [0, 0, 2.7], material="truck_grey")
    for x in (-2.0, 0.2):
        P.add("box", [0.55, 0.03, 0.45], [x, -1.215, 1.6], material="window_glass")
    P.add("box", [0.45, 0.03, 0.95], [1.7, -1.215, 1.05], material="wood_door")
    for x in (-2.6, 2.6):
        for s in (-1, 1):
            P.add("box", [0.25, 0.25, 0.06], [x, s * 1.0, 0.04], material="stone_dark")
    P.add("box", [0.35, 0.25, 0.22], [-1.0, -1.35, 2.3], material="truck_white")  # AC unit
    return P.out


def pallets_and_bags(cx, cy, ground, rng) -> list[str]:
    P = Placer(cx, cy, 0.0, ground, "decor")
    for i in range(4):
        x = (i % 2) * 1.4 - 0.7
        y = (i // 2) * 1.4 - 0.7
        P.add("box", [0.6, 0.5, 0.07], [x, y, 0.07], material="pallet_wood")
        P.add("box", [0.5, 0.45, 0.45], [x, y, 0.6], quat=quat_axis((0, 0, 1), rng.uniform(-0.1, 0.1)),
              material="bigbag")
        P.add("ellipsoid", [0.42, 0.38, 0.12], [x, y, 1.05], material="mulch_chips")
        for s in (-1, 1):
            P.add("capsule", [0.02], fromto=[x - 0.35, y + s * 0.35, 1.05, x - 0.35, y + s * 0.35, 1.25],
                  material="bigbag")
    return P.out


# ---------------------------------------------------------------------------
# Hidden obstacles (named obs_{oid}_*)
# ---------------------------------------------------------------------------


def tractor(oid, cx, cy, yaw, ground) -> list[str]:
    P = Placer(cx, cy, yaw, ground, "obs", prefix=f"obs_{oid}", z0=float(ground(cx, cy)) - 0.03)
    # rear big wheels (tread lugs as boxes), front wheels
    for s in (-1, 1):
        P.add("cylinder", [0.78, 0.22], [-1.15, s * 0.98, 0.78], zaxis=(0, 1, 0), material="rubber",
              name=f"rear_tyre_{'l' if s > 0 else 'r'}")
        P.add("cylinder", [0.45, 0.23], [-1.15, s * 0.98, 0.78], zaxis=(0, 1, 0), material="tractor_cream")
        for k in range(16):
            th = 2 * np.pi * k / 16
            q = quat_axis((0, 1, 0), th)
            P.add("box", [0.04, 0.21, 0.035], [-1.15 + 0.78 * np.sin(th), s * 0.98, 0.78 + 0.78 * np.cos(th)],
                  quat=quat_mul(q, quat_axis((0, 0, 1), 0.5 * s)), material="rubber")
        P.add("cylinder", [0.45, 0.13], [1.55, s * 0.82, 0.45 - (0.12 if s < 0 else 0.0)], zaxis=(0, 1, 0),
              material="rubber", name=f"front_tyre_{'l' if s > 0 else 'r'}")
        P.add("cylinder", [0.25, 0.14], [1.55, s * 0.82, 0.45 - (0.12 if s < 0 else 0.0)], zaxis=(0, 1, 0),
              material="tractor_cream")
        # rear fenders
        for k, th in enumerate(np.linspace(-1.1, 1.1, 5)):
            q = quat_axis((0, 1, 0), th)
            P.add("box", [0.26, 0.26, 0.02], [-1.15 + 0.92 * np.sin(th), s * 0.98, 0.78 + 0.92 * np.cos(th)], quat=q,
                  material="tractor_red")
    P.add("box", [1.4, 0.36, 0.28], [0.65, 0, 0.72], material="tractor_grey", name="engine")
    P.add("box", [0.7, 0.55, 0.42], [-0.9, 0, 0.85], material="tractor_grey", name="transmission")
    P.add("box", [1.15, 0.43, 0.27], [0.95, 0, 1.22], material="tractor_red", name="bonnet")
    P.add("box", [0.04, 0.4, 0.3], [2.11, 0, 1.15], material="tractor_grey")
    P.add("box", [0.18, 0.5, 0.18], [2.3, 0, 0.6], material="tractor_grey", name="front_weights")
    P.add("box", [0.25, 0.4, 0.12], [1.0, 0, 1.52], material="tractor_rust")
    P.add("cylinder", [0.055, 0.6], [1.45, 0.22, 1.95], material="tractor_grey", name="exhaust")
    P.add("box", [0.3, 0.3, 0.06], [-1.05, 0, 1.35], material="tractor_grey")  # platform
    P.add("box", [0.22, 0.24, 0.05], [-1.25, 0, 1.55], material="tractor_grey")  # seat
    P.add("box", [0.04, 0.24, 0.22], [-1.45, 0, 1.75], material="tractor_grey")
    P.add("cylinder", [0.2, 0.015], [-0.55, 0, 1.85], quat=quat_axis((0, 1, 0), -0.7), material="tractor_grey")
    P.add("capsule", [0.025], fromto=[-0.45, 0, 1.45, -0.55, 0, 1.83], material="tractor_grey")
    # ROPS frame + canopy roof
    for (x, s) in ((-0.55, 1), (-0.55, -1), (-1.75, 1), (-1.75, -1)):
        P.add("capsule", [0.045], fromto=[x, s * 0.72, 1.2, x, s * 0.72, 2.85], material="tractor_grey")
    P.add("box", [0.75, 0.82, 0.05], [-1.15, 0, 2.9], material="tractor_cream", name="roof")
    P.add("box", [0.12, 0.62, 0.1], [-2.15, 0, 0.75], material="tractor_grey", name="hitch")
    P.add("capsule", [0.04], fromto=[-2.15, 0.35, 0.75, -2.5, 0.35, 0.55], material="tractor_grey")
    P.add("capsule", [0.04], fromto=[-2.15, -0.35, 0.75, -2.5, -0.35, 0.55], material="tractor_grey")
    return P.out


def pump(oid, cx, cy, yaw, ground) -> list[str]:
    P = Placer(cx, cy, yaw, ground, "obs", prefix=f"obs_{oid}", z0=float(ground(cx, cy)))
    # skid frame on the north half
    for s in (-1, 1):
        P.add("box", [1.5, 0.06, 0.08], [0, 0.55 + s * 0.45, 0.08], material="galv", name=f"skid_{s}")
    for x in (-1.3, 0, 1.3):
        P.add("box", [0.05, 0.5, 0.05], [x, 0.55, 0.12], material="galv")
    P.add("box", [0.55, 0.38, 0.4], [-0.55, 0.55, 0.58], material="pump_green", name="engine")
    P.add("box", [0.12, 0.42, 0.45], [-1.15, 0.55, 0.62], material="tractor_grey", name="radiator")
    P.add("box", [0.42, 0.3, 0.2], [-0.55, 0.55, 1.1], material="tractor_grey", name="tank")
    P.add("cylinder", [0.04, 0.35], [-0.35, 0.75, 1.35], material="tractor_grey", name="exhaust")
    P.add("cylinder", [0.3, 0.16], [0.45, 0.55, 0.52], zaxis=(1, 0, 0), material="pump_blue", name="volute")
    P.add("cylinder", [0.12, 0.35], [0.95, 0.55, 0.52], zaxis=(1, 0, 0), material="pump_blue")
    P.add("capsule", [0.1], fromto=[1.3, 0.55, 0.52, 1.55, 0.2, 0.15], material="rubber", name="suction_hose")
    P.add("cylinder", [0.08, 0.35], [0.45, 0.55, 1.05], material="pump_blue", name="outlet")
    P.add("box", [0.3, 0.05, 0.25], [-0.2, 0.92, 0.75], material="truck_white")  # control box
    # pipe stack on the south half (blue PVC + aluminium), pyramid
    k = 0
    for layer, n in enumerate((5, 4, 3, 2)):
        r = 0.085
        for i in range(n):
            y = -0.55 + (i - (n - 1) / 2) * 2 * r * 1.02
            z = r + layer * r * 1.72 + 0.12
            mat = "pvc_blue" if (i + layer) % 3 else "alu"
            P.add("cylinder", [r if mat == "pvc_blue" else r * 0.8, 1.48], [0.05, y, z], zaxis=(1, 0, 0), material=mat,
                  name=f"pipe{k}")
            k += 1
    for x in (-1.1, 1.1):
        P.add("box", [0.06, 0.55, 0.06], [x, -0.55, 0.06], material="pallet_wood")
    return P.out


def obstacle_marker(oid, poly, height, ground, kind) -> list[str]:
    """Initially invisible (alpha 0) glowing outline + translucent fill + map pin."""
    out = []
    c = poly.centroid
    zg = float(ground(c.x, c.y))
    lift = 0.24
    buf = poly.buffer(1.2, join_style=2 if kind in ("tractor", "pump") else 1)
    ring = np.asarray(buf.exterior.simplify(0.15).coords)
    if len(ring) > 26:
        idx = np.linspace(0, len(ring) - 1, 26).astype(int)
        ring = ring[idx]
    fill = poly.buffer(0.6)
    if kind == "waterlogged":
        r = np.sqrt(fill.area / np.pi)
        out.append(g(f"obs_mark_{oid}", "cylinder", [r, 0.01], pos=[c.x, c.y, zg + lift], rgba=[1, 0.35, 0.05, 0],
                     material="mark_fill", cls="mark"))
    else:
        mrr = fill.minimum_rotated_rectangle
        pts = np.asarray(mrr.exterior.coords)[:4]
        e1 = pts[1] - pts[0]
        e2 = pts[2] - pts[1]
        yaw = np.arctan2(e1[1], e1[0])
        out.append(g(f"obs_mark_{oid}", "box", [np.linalg.norm(e1) / 2, np.linalg.norm(e2) / 2, 0.01],
                     pos=[c.x, c.y, zg + lift], quat=quat_axis((0, 0, 1), yaw), rgba=[1, 0.35, 0.05, 0],
                     material="mark_fill", cls="mark"))
    for j in range(len(ring) - 1):
        a, b = ring[j], ring[j + 1]
        if np.linalg.norm(b - a) < 0.05:
            continue
        za = float(ground(*a)) + lift + 0.02
        zb = float(ground(*b)) + lift + 0.02
        out.append(g(f"obs_mark_{oid}_seg{j}", "capsule", [0.08], fromto=[a[0], a[1], za, b[0], b[1], zb],
                     rgba=[1, 0.25, 0.02, 0], material="mark_line", cls="mark"))
    top = zg + max(height, 0.5) + 1.2
    out.append(g(f"obs_mark_{oid}_pin", "cylinder", [0.06, 2.0], pos=[c.x, c.y, top + 2.0], rgba=[1, 0.3, 0.05, 0],
                 material="mark_line", cls="mark"))
    out.append(g(f"obs_mark_{oid}_pinhead", "sphere", [0.55], pos=[c.x, c.y, top + 4.4], rgba=[1, 0.3, 0.05, 0],
                 material="mark_line", cls="mark"))
    return out
