"""1:1-scale MuJoCo world for PruneMaxxing (Parc Agrari del Baix Llobregat).

World frame (see ``world_frame.py``): x = easting - center_e, y = northing -
center_n, z = absolute DEM elevation.  The physics heightfield (geom
``terrain``, group 3, hidden in renders) equals ``Terrain.height_at``; the
visible ground is a set of textured meshes sampled from the same terrain
(the field mesh adds real 3D raised beds on top), which avoids z-fighting and
lets the beds be lit/shaded properly.

Public API
----------
build_world(site=None, payload_mode='dust', swarm_size=4, out_dir=..., texture_px=4096, force=False) -> WorldSpec
load_world_model(spec) -> mujoco.MjModel
set_payload_mode(model, mode)
set_obstacle_detected(model, oid, detected)
MulchPainter(spec, model, mode).apply(model, renderer, completion_grid)
make_renderer(model, height, width) / configure_scene(renderer)
focus_shadows(renderer, model, data, center_xyz | None, half_extent)
place_mothership(...), place_drone(...), park_fx(...)
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", "/usr/share/glvnd/egl_vendor.d/10_nvidia.json")

import numpy as np
from PIL import Image
from shapely.affinity import translate

from prunemaxxing.config import DEFAULT_DEM_PATH
from prunemaxxing.sim import world_effectors as we
from prunemaxxing.sim import world_meshes as wm
from prunemaxxing.sim import world_props as wp
from prunemaxxing.sim import world_robots as wr
from prunemaxxing.sim import world_textures as wt
from prunemaxxing.sim.world_frame import Terrain, WorldFrame, frame_for_site, load_terrain

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "prunemaxxing" / "sim" / "assets" / "world"
BUILDER_VERSION = "1.0.0"

HALF_X = 290.0
HALF_Y = 285.0
TERRAIN_RES = 1.0
GROUND_STEP = 2.0
RIDGE_H = 0.20
BED_PITCH = 0.85
HEADLAND_M = 12.0
SUN_AZ_DEG = 248.0  # compass azimuth the sun shines *from* (WSW, afternoon)
SUN_EL_DEG = 33.0
EXTENT = 300.0
APRON_RINGS = [6, 15, 30, 55, 90, 140, 210, 300, 420, 600, 850]
CAM_TOP_EXTENT = 352.0  # metres visible vertically in the orthographic cam_top
FAR_RGB = (0.74, 0.75, 0.70)  # baked aerial-perspective colour of distant farmland

# geom groups
GROUP_ROBOT = 0  # robots + hidden obstacles (physical things)
GROUP_DECOR = 1  # decorative scenery (trees, reeds, weeds, buildings, props)
GROUP_GROUND = 2  # visible ground meshes, FX pools, obstacle highlight marks
GROUP_PHYSICS = 3  # heightfield collision geom (hidden by default)

MARK_ALPHA = {"fill": 0.32, "seg": 0.95, "pin": 0.85, "pinhead": 0.95}


def sun_direction() -> np.ndarray:
    az = np.deg2rad(SUN_AZ_DEG)
    el = np.deg2rad(SUN_EL_DEG)
    to_sun = np.array([np.sin(az) * np.cos(el), np.cos(az) * np.cos(el), np.sin(el)])
    return -to_sun  # light travel direction


# ---------------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------------


@dataclass
class CellGrid:
    """FieldTwin cell grid in world coords (rows north -> south, like FieldTwin)."""

    nrows: int
    ncols: int
    x0: float  # world x of the west edge of column 0
    y1: float  # world y of the north edge of row 0
    dx: float
    dy: float
    work: np.ndarray  # (nrows, ncols) bool: cell centre inside a parcel bed area

    def center(self, row: float, col: float) -> tuple[float, float]:
        return self.x0 + (col + 0.5) * self.dx, self.y1 - (row + 0.5) * self.dy

    def centers(self) -> np.ndarray:
        cols, rows = np.meshgrid(np.arange(self.ncols), np.arange(self.nrows))
        return np.stack([self.x0 + (cols + 0.5) * self.dx, self.y1 - (rows + 0.5) * self.dy], -1)


@dataclass
class WorldSpec:
    xml_path: Path
    out_dir: Path
    frame: WorldFrame
    terrain: Terrain
    half_x: float
    half_y: float
    h_min: float
    h_max: float
    payload_mode: str
    swarm_size: int
    cells: CellGrid
    field_box: tuple[float, float, float, float]
    field_tex_size: tuple[int, int]
    beds: list[dict]  # per parcel: pid, crop, minx, bx0, by0, bx1, by1, pitch
    names: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def bed_height(self, x, y):
        """Raised-bed profile height above terrain at world (x, y) (0 outside beds)."""
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        out = np.zeros(np.broadcast(x, y).shape)
        for b in self.beds:
            u = np.mod((x - b["minx"]) / b["pitch"], 1.0)
            d = np.where(u < 0.5, u, u - 1.0) * b["pitch"]
            prof = wm.bed_profile(d, RIDGE_H)
            ry = np.clip(np.minimum(y - b["by0"], b["by1"] - y) / 0.6, 0, 1)
            inx = (x >= b["rx0"]) & (x <= b["rx1"])
            out = np.where(inx & (ry > 0), prof * ry, out)
        return float(out) if out.ndim == 0 else out

    def ground_z(self, x, y):
        """Visible ground surface height (terrain + raised beds)."""
        return self.terrain.height_at(x, y) + self.bed_height(x, y)

    def furrow_x(self, x_world: float, parcel_index: int = 0) -> float:
        """Snap a world x to the nearest furrow centre of the given parcel."""
        b = self.beds[parcel_index]
        k = np.round((x_world - b["minx"]) / b["pitch"] - 0.5)
        return float(b["minx"] + (k + 0.5) * b["pitch"])


# ---------------------------------------------------------------------------
# Site -> world-coordinate scene
# ---------------------------------------------------------------------------


def _w(geom, frame: WorldFrame):
    return translate(geom, -frame.center_e, -frame.center_n)


def _site_digest(site) -> str:
    h = hashlib.sha256()
    for p in site.parcels:
        h.update(f"P{p.pid}{p.polygon.wkt}{p.crop}{p.bed_spacing_m}".encode())
    for t in site.tracks + site.ditches:
        h.update(f"T{t.tid}{t.line.wkt}{t.width_m}{t.surface}".encode())
    for z in site.weeds:
        h.update(f"W{z.zid}{z.polygon.wkt}{sorted(z.props.items())}".encode())
    h.update(site.depot.polygon.wkt.encode())
    for s in site.stockpiles:
        h.update(s.wkt.encode())
    for o in site.obstacles:
        h.update(f"O{o.oid}{o.kind}{o.polygon.wkt}{o.height_m}{o.yaw_rad}".encode())
    for s in site.scenery:
        h.update(f"S{s.kind}{s.geometry.wkt}{s.props.get('height_m')}".encode())
    h.update(f"{site.origin_e}{site.origin_n}{site.width_m}{site.height_m}".encode())
    return h.hexdigest()[:16]


def _scene(site, frame) -> tuple[wt.PaintScene, dict]:
    W = lambda g: _w(g, frame)  # noqa: E731
    parcels = []
    beds = []
    for i, p in enumerate(site.parcels):
        pg = W(p.polygon)
        x0, y0, x1, y1 = pg.bounds
        pitch = p.bed_spacing_m or BED_PITCH
        bx0, by0, bx1, by1 = x0 + HEADLAND_M, y0 + HEADLAND_M, x1 - HEADLAND_M, y1 - HEADLAND_M
        k0 = int(np.ceil((bx0 - x0) / pitch))
        k1 = int(np.floor((bx1 - x0) / pitch))
        parcels.append(wt.PaintParcel(p.pid, p.crop, x0, y0, x1, y1, bx0, by0, bx1, by1, pitch, seed=i + 1))
        beds.append(dict(pid=p.pid, crop=p.crop, minx=x0, x0=x0, y0=y0, x1=x1, y1=y1, bx0=bx0, by0=by0, bx1=bx1,
                         by1=by1, pitch=pitch, k0=k0, k1=k1, rx0=x0 + k0 * pitch - pitch / 2,
                         rx1=x0 + k1 * pitch + pitch / 2))
    tracks = [(t.tid, [tuple(c) for c in W(t.line).coords], t.width_m, t.surface) for t in site.tracks]
    ditches = [(d.tid, [tuple(c) for c in W(d.line).coords], d.width_m) for d in site.ditches]
    weeds = [(z.zid, W(z.polygon), z.props.get("species", "Sorghum_halepense")) for z in site.weeds]
    stock = [(W(s).x, W(s).y) for s in site.stockpiles]
    wl = []
    fps = []
    for o in site.obstacles:
        pg = W(o.polygon)
        if o.kind == "waterlogged":
            c = pg.centroid
            wl.append((c.x, c.y, float(np.sqrt(pg.area / np.pi))))
        else:
            fps.append((o.oid, pg))
    crown = {"poplar": 1.9, "olive": 2.6, "carob": 3.0, "citrus": 1.5, "pine": 4.2, "reed": 0.7}
    trees, houses, ghs, poles = [], [], [], []
    for s in site.scenery:
        g = W(s.geometry)
        h = float(s.props.get("height_m", 5.0))
        if s.kind in crown:
            trees.append((s.kind, g.x, g.y, crown[s.kind] * (h / 15 if s.kind == "pine" else 1.0), h))
        elif s.kind == "house":
            houses.append((g, h))
        elif s.kind == "greenhouse":
            ghs.append((g, h))
        elif s.kind == "pole":
            poles.append((g.x, g.y, h))

    def bbox(pts, m):
        if not pts:
            return None
        a = np.asarray(pts)
        return (a[:, 0].min() - m, a[:, 1].min() - m, a[:, 0].max() + m, a[:, 1].max() + m)

    citrus = [(t[1], t[2]) for t in trees if t[0] == "citrus"]
    pines = [(t[1], t[2]) for t in trees if t[0] == "pine"]
    gh_pts = [c for g, _ in ghs for c in g.exterior.coords]
    house_pts = [c for g, _ in houses for c in g.exterior.coords]
    sx0, sy0 = -site.width_m / 2, -site.height_m / 2
    scene = wt.PaintScene(
        site_box=(sx0, sy0, -sx0, -sy0),
        parcels=parcels,
        tracks=tracks,
        ditches=ditches,
        weeds=weeds,
        depot=W(site.depot.polygon),
        stockpiles=stock,
        waterlogged=wl,
        obstacle_footprints=fps,
        trees=trees,
        houses=[g for g, _ in houses],
        greenhouses=[g for g, _ in ghs],
        orchard_box=bbox(citrus, 4.0),
        greenhouse_box=bbox(gh_pts, 3.0),
        yard_box=bbox(house_pts, 9.0),
        pine_box=bbox(pines, 6.0),
        poles=[(p[0], p[1]) for p in poles],
        seed=1234,
    )
    extra = dict(beds=beds, houses=houses, greenhouses=ghs, poles=poles)
    return scene, extra


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def _field_box(beds) -> tuple[float, float, float, float]:
    x0 = min(b["x0"] for b in beds) - 1.0
    y0 = min(b["y0"] for b in beds) - 1.0
    x1 = max(b["x1"] for b in beds) + 1.0
    y1 = max(b["y1"] for b in beds) + 1.0
    # snap outward onto the GROUND_STEP grid anchored at (-HALF_X, -HALF_Y)
    sx = lambda v, fn: -HALF_X + fn((v + HALF_X) / GROUND_STEP) * GROUND_STEP  # noqa: E731
    sy = lambda v, fn: -HALF_Y + fn((v + HALF_Y) / GROUND_STEP) * GROUND_STEP  # noqa: E731
    return (sx(x0, np.floor), sy(y0, np.floor), sx(x1, np.ceil), sy(y1, np.ceil))


def _cell_grid(site, frame, beds) -> CellGrid:
    from prunemaxxing.geo.dem import load_dem
    from prunemaxxing.geo.grid import crop_target_field, select_pilot_window

    dem = select_pilot_window(crop_target_field(load_dem(ROOT / DEFAULT_DEM_PATH)), 12.0)
    x0 = dem.origin_easting - frame.center_e
    y1 = dem.origin_northing - frame.center_n
    cg = CellGrid(dem.height, dem.width, x0, y1, dem.pixel_size_x, dem.pixel_size_y,
                  np.zeros((dem.height, dem.width), bool))
    C = cg.centers()
    for b in beds:
        cg.work |= (C[..., 0] > b["bx0"]) & (C[..., 0] < b["bx1"]) & (C[..., 1] > b["by0"]) & (C[..., 1] < b["by1"])
    return cg


def _code_digest(*objs) -> str:
    import inspect

    h = hashlib.sha256()
    for o in objs:
        h.update(inspect.getsource(o).encode())
    return h.hexdigest()[:12]


def _save_png(path: Path, arr: np.ndarray) -> None:
    Image.fromarray(arr).save(path, compress_level=1)


def build_world(site=None, payload_mode: str = "dust", swarm_size: int = 4, out_dir: Path | str = DEFAULT_OUT,
                texture_px: int = 4096, force: bool = False, verbose: bool = True) -> WorldSpec:
    """Generate (or reuse cached) world assets + MJCF. Deterministic (seeded)."""
    from prunemaxxing.geo.site import load_site

    t_start = time.time()
    payload_mode = str(getattr(payload_mode, "value", payload_mode)).lower()
    if payload_mode not in ("dust", "cartridge"):
        raise ValueError(f"payload_mode must be dust|cartridge, got {payload_mode}")
    swarm_size = max(4, int(swarm_size))
    site = site or load_site()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    frame = frame_for_site(site)
    dem_path = ROOT / DEFAULT_DEM_PATH
    st = dem_path.stat()
    code_assets = _code_digest(wt, wm, _scene, _generate_assets, _ground_axes, _carve, _field_box)
    code_xml = _code_digest(wr, wp, we, _world_xml, _apply_payload_alpha_xml)
    asset_key = hashlib.sha256(
        f"{BUILDER_VERSION}|{_site_digest(site)}|{st.st_size}|{int(st.st_mtime)}|{texture_px}|{code_assets}".encode()
    ).hexdigest()[:16]
    xml_name = f"world_{payload_mode}_n{swarm_size}.xml"
    meta_path = out / "world_meta.json"
    log = (lambda *a: print("[world]", *a, flush=True)) if verbose else (lambda *a: None)

    terrain = load_terrain(frame, HALF_X, HALF_Y, TERRAIN_RES)
    scene, extra = _scene(site, frame)
    beds = extra["beds"]
    cells = _cell_grid(site, frame, beds)
    fbox = _field_box(beds)

    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    assets_ok = (not force and meta.get("asset_key") == asset_key
                 and all((out / f).exists() for f in meta.get("asset_files", [])))
    if not assets_ok:
        log(f"generating assets into {out} (key {asset_key})")
        meta = _generate_assets(site, frame, terrain, scene, extra, fbox, out, texture_px, log)
        meta["asset_key"] = asset_key
    else:
        log(f"assets cached (key {asset_key})")
    xml_path = out / xml_name
    xml_key = hashlib.sha256(f"{asset_key}|{payload_mode}|{swarm_size}|{code_xml}".encode()).hexdigest()[:16]
    if force or not xml_path.exists() or meta.get("xml_keys", {}).get(xml_name) != xml_key:
        log(f"writing {xml_name}")
        xml, names = _world_xml(site, frame, terrain, scene, extra, fbox, meta, payload_mode, swarm_size)
        xml_path.write_text(xml)
        (out / "world.xml").write_text(xml)
        meta.setdefault("xml_keys", {})[xml_name] = xml_key
        meta.setdefault("names", {})[xml_name] = names
    meta["landmarks"] = {
        "houses": [[g.centroid.x, g.centroid.y] for g, _ in extra["houses"]],
        "greenhouse_box": list(scene.greenhouse_box) if scene.greenhouse_box else None,
        "orchard_box": list(scene.orchard_box) if scene.orchard_box else None,
        "depot_box": list(scene.depot.bounds),
        "stockpiles": [list(s) for s in scene.stockpiles],
        "ditches": [[d[0], [list(p) for p in d[1]], d[2]] for d in scene.ditches],
        "obstacles": {oid: [pg.centroid.x, pg.centroid.y] for oid, pg in scene.obstacle_footprints}
        | {f"waterlogged_{i}": [w[0], w[1]] for i, w in enumerate(scene.waterlogged)},
    }
    meta["frame"] = asdict(frame)
    meta["cells"] = dict(nrows=cells.nrows, ncols=cells.ncols, x0=cells.x0, y1=cells.y1, dx=cells.dx, dy=cells.dy)
    meta["field_box"] = list(fbox)
    meta["beds"] = beds
    meta_path.write_text(json.dumps(meta, indent=1, default=float))
    spec = WorldSpec(
        xml_path=xml_path,
        out_dir=out,
        frame=frame,
        terrain=terrain,
        half_x=HALF_X,
        half_y=HALF_Y,
        h_min=float(meta["h_min"]),
        h_max=float(meta["h_max"]),
        payload_mode=payload_mode,
        swarm_size=swarm_size,
        cells=cells,
        field_box=tuple(fbox),
        field_tex_size=tuple(meta["field_tex_size"]),
        beds=beds,
        names=meta["names"][xml_name],
        meta=meta,
    )
    log(f"world ready in {time.time() - t_start:.1f}s -> {xml_path}")
    return spec


# ---------------------------------------------------------------------------
# Asset generation
# ---------------------------------------------------------------------------


def _ditch_dist(scene, X, Y):
    d = np.full(np.broadcast(X, Y).shape, 1e9)
    for _, pts, w in scene.ditches:
        for a, b in zip(pts[:-1], pts[1:]):
            dd, _ = wt.seg_dist(X, Y, a, b)
            d = np.minimum(d, dd)
    return d


def _ground_axes(scene):
    xs = list(np.arange(-HALF_X, HALF_X + 1e-6, GROUND_STEP))
    ys = list(np.arange(-HALF_Y, HALF_Y + 1e-6, GROUND_STEP))
    offs = [-2.6, -1.9, -1.25, -0.6, 0.0, 0.6, 1.25, 1.9, 2.6]
    for _, pts, w in scene.ditches:
        for a, b in zip(pts[:-1], pts[1:]):
            if abs(a[0] - b[0]) < 1e-6:  # vertical ditch -> extra columns
                xs += [a[0] + o for o in offs]
            elif abs(a[1] - b[1]) < 1e-6:
                ys += [a[1] + o for o in offs]

    def clean(v, lim):
        v = np.unique(np.round(np.clip(v, -lim, lim), 3))
        keep = [v[0]]
        for t in v[1:]:
            if t - keep[-1] > 0.25:
                keep.append(t)
            elif abs(t - np.round(t)) < 1e-6:  # keep grid point, drop extra
                keep[-1] = t
        return np.array(keep)

    return clean(xs, HALF_X), clean(ys, HALF_Y)


def _carve(scene, X, Y):
    d = _ditch_dist(scene, X, Y)
    return -0.55 * (1.0 - wt.sstep(0.95, 2.1, d))


def _generate_assets(site, frame, terrain, scene, extra, fbox, out: Path, texture_px: int, log) -> dict:
    t0 = time.time()
    rng = np.random.default_rng(20261120)
    files = []
    hmin = float(terrain.heights.min())
    hmax = float(terrain.heights.max())
    meta: dict = {"h_min": hmin, "h_max": hmax}

    # --- heightfield (binary custom format, row 0 = south, normalised) ---
    nrow, ncol = terrain.heights.shape
    norm = ((terrain.heights - hmin) / max(hmax - hmin, 1e-6)).astype(np.float32)
    with open(out / "terrain_hfield.bin", "wb") as fh:
        fh.write(np.array([nrow, ncol], np.int32).tobytes())
        fh.write(norm.tobytes())
    _save_png(out / "terrain_hfield_preview.png", (np.flipud(norm) * 65535).astype(np.uint16))
    files += ["terrain_hfield.bin"]
    meta["hfield"] = dict(nrow=int(nrow), ncol=int(ncol), size=[HALF_X, HALF_Y, hmax - hmin, 0.5], pos_z=hmin)
    log(f"hfield {ncol}x{nrow}, z {hmin:.2f}..{hmax:.2f} m")

    painter = wt.GroundPainter(scene)
    # --- ground texture (whole hfield extent) ---
    gw = texture_px
    gh = int(round(texture_px * (2 * HALF_Y) / (2 * HALF_X)))
    t1 = time.time()
    Rg = wt.Raster(-HALF_X, -HALF_Y, HALF_X, HALF_Y, gw, gh)
    ground = painter.paint(Rg)["rgb"]
    _save_png(out / "tex_ground.png", wt.to_u8(ground))
    del ground
    files.append("tex_ground.png")
    log(f"ground texture {gw}x{gh} in {time.time() - t1:.1f}s")

    # --- field texture (parcels window, ~8 cm/px) + mulch layers ---
    fx0, fy0, fx1, fy1 = fbox
    fw = texture_px
    fh_ = int(round(texture_px * (fy1 - fy0) / (fx1 - fx0)))
    t1 = time.time()
    Rf = wt.Raster(fx0, fy0, fx1, fy1, fw, fh_)
    lay = painter.paint(Rf, mulch="all")
    base = wt.to_u8(lay["rgb"])
    _save_png(out / "tex_field.png", base)
    np.save(out / "field_base.npy", base)
    np.save(out / "field_mulch_dust.npy", wt.to_u8(lay["dust"]))
    np.save(out / "field_mulch_cartridge.npy", wt.to_u8(lay["cart"]))
    np.save(out / "field_mulch_masks.npy", np.stack([lay["threshold"], lay["weight"]], -1))
    _save_png(out / "preview_field_mulch_dust.png", wt.to_u8(wt.downsample(lay["dust"], 4)))
    _save_png(out / "preview_field_mulch_cartridge.png", wt.to_u8(wt.downsample(lay["cart"], 4)))
    del lay, base
    files += ["tex_field.png", "field_base.npy", "field_mulch_dust.npy", "field_mulch_cartridge.npy",
              "field_mulch_masks.npy"]
    meta["field_tex_size"] = [fw, fh_]
    log(f"field texture {fw}x{fh_} (+mulch layers) in {time.time() - t1:.1f}s")

    # --- apron / far farmland, sky, atlas, small textures ---
    ax = HALF_X + APRON_RINGS[-1]
    ay = HALF_Y + APRON_RINGS[-1]
    meta["apron_box"] = [-ax, -ay, ax, ay]
    _save_png(out / "tex_apron.png", wt.bake_patchwork_texture(-ax, -ay, ax, ay, 2048, painter.seed + 99,
                                                                haze_rgb=FAR_RGB, haze_r=(HALF_X + 40, ax - 20)))
    _save_png(out / "tex_sky.png", wt.bake_skybox(1024, tuple(-sun_direction())))
    _save_png(out / "tex_atlas.png", wt.bake_atlas())
    for name, arr in wt.bake_material_textures().items():
        _save_png(out / f"tex_{name}.png", arr)
        files.append(f"tex_{name}.png")
    files += ["tex_apron.png", "tex_sky.png", "tex_atlas.png"]
    log(f"aux textures done ({time.time() - t0:.1f}s total)")

    # --- terrain meshes ---
    t1 = time.time()
    hat = terrain.height_at
    gxs, gys = _ground_axes(scene)
    GX, GY = np.meshgrid(gxs, gys)
    GZ = hat(GX, GY) + _carve(scene, GX, GY)
    uv = np.stack([(GX + HALF_X) / (2 * HALF_X), (HALF_Y - GY) / (2 * HALF_Y)], -1)
    cx = 0.5 * (gxs[:-1] + gxs[1:])
    cy = 0.5 * (gys[:-1] + gys[1:])
    hole = ((cy[:, None] > fy0) & (cy[:, None] < fy1)) & ((cx[None, :] > fx0) & (cx[None, :] < fx1))
    V, F, T, N = wm.grid_mesh(gxs, gys, GZ, uv, skip=hole)
    b = wm.MeshBuilder()
    b.add(V, F, N, T)
    meta["mesh_ground"] = b.write(out / "mesh_ground.msh")
    files.append("mesh_ground.msh")

    # field mesh with 3D raised beds
    fxs = set(np.round(np.arange(fx0, fx1 + 1e-6, GROUND_STEP), 4))
    for bd in extra["beds"]:
        for k in range(bd["k0"], bd["k1"] + 1):
            rc = bd["minx"] + k * bd["pitch"]
            for o in (-0.425, -0.30, -0.15, 0.0, 0.15, 0.30):
                fxs.add(round(rc + o, 4))
        fxs.add(round(bd["minx"] + bd["k1"] * bd["pitch"] + 0.425, 4))
    fys = set(np.round(np.arange(fy0, fy1 + 1e-6, GROUND_STEP), 4))
    for bd in extra["beds"]:
        for v in np.arange(bd["by0"], bd["by1"] + 1e-6, 2.5):
            fys.add(round(v, 4))
        for v in (bd["by0"], bd["by0"] + 0.3, bd["by0"] + 0.6, bd["by1"] - 0.6, bd["by1"] - 0.3, bd["by1"]):
            fys.add(round(v, 4))
    fxs = np.array(sorted(v for v in fxs if fx0 - 1e-6 <= v <= fx1 + 1e-6))
    fys = np.array(sorted(v for v in fys if fy0 - 1e-6 <= v <= fy1 + 1e-6))
    fxs = fxs[np.concatenate([[True], np.diff(fxs) > 0.05])]
    fys = fys[np.concatenate([[True], np.diff(fys) > 0.05])]
    FX, FY = np.meshgrid(fxs, fys)
    tmp_spec = WorldSpec(Path(), out, frame, terrain, HALF_X, HALF_Y, hmin, hmax, "dust", 4,
                         CellGrid(1, 1, 0, 0, 1, 1, np.zeros((1, 1), bool)), fbox, (1, 1), extra["beds"])
    FZ = hat(FX, FY) + tmp_spec.bed_height(FX, FY)
    fuv = np.stack([(FX - fx0) / (fx1 - fx0), (fy1 - FY) / (fy1 - fy0)], -1)
    V, F, T, N = wm.grid_mesh(fxs, fys, FZ, fuv)
    b = wm.MeshBuilder()
    b.add(V, F, N, T)
    # skirt along the boundary to hide T-junction cracks
    ny, nx = FZ.shape
    loop = ([(0, i) for i in range(nx)] + [(j, nx - 1) for j in range(1, ny)] +
            [(ny - 1, i) for i in range(nx - 2, -1, -1)] + [(j, 0) for j in range(ny - 2, 0, -1)])
    top = np.array([[FX[j, i], FY[j, i], FZ[j, i]] for j, i in loop])
    tuv = np.array([fuv[j, i] for j, i in loop])
    bot = top - [0, 0, 0.35]
    Vs = np.vstack([top, bot])
    n = len(top)
    Fs = []
    for k in range(n):
        a_, b_ = k, (k + 1) % n
        Fs += [(a_, b_, n + b_), (a_, n + b_, n + a_)]
    b.add(Vs, np.array(Fs), np.tile([0, 0, 1.0], (2 * n, 1)), np.vstack([tuv, tuv]))
    meta["mesh_field"] = b.write(out / "mesh_field.msh")
    files.append("mesh_field.msh")

    # apron ring outside the hfield
    rings = np.array(APRON_RINGS, float)
    axs = np.concatenate([-HALF_X - rings[::-1], gxs, HALF_X + rings])
    ays = np.concatenate([-HALF_Y - rings[::-1], gys, HALF_Y + rings])
    AX, AY = np.meshgrid(axs, ays)
    ex = np.clip(AX, -HALF_X, HALF_X)
    ey = np.clip(AY, -HALF_Y, HALF_Y)
    dout = np.hypot(np.abs(AX) - np.minimum(np.abs(AX), HALF_X), np.abs(AY) - np.minimum(np.abs(AY), HALF_Y))
    edge_vals = np.concatenate([terrain.heights[0], terrain.heights[-1], terrain.heights[:, 0], terrain.heights[:, -1]])
    z_far = float(np.percentile(edge_vals, 30)) - 0.15
    z_plane = z_far - 0.6
    s = wt.sstep(0.0, 260.0, dout)
    AZ = hat(ex, ey) + _carve(scene, ex, ey) * (dout < 1e-6)
    AZ = AZ * (1 - s) + z_far * s
    AZ = np.where(dout >= rings[-1] - 1e-6, z_plane - 1.2, AZ)
    auv = np.stack([(AX + ax) / (2 * ax), (ay - AY) / (2 * ay)], -1)
    acx = 0.5 * (axs[:-1] + axs[1:])
    acy = 0.5 * (ays[:-1] + ays[1:])
    inner = (np.abs(acy)[:, None] < HALF_Y) & (np.abs(acx)[None, :] < HALF_X)
    V, F, T, N = wm.grid_mesh(axs, ays, AZ, auv, skip=inner)
    b = wm.MeshBuilder()
    b.add(V, F, N, T)
    meta["mesh_apron"] = b.write(out / "mesh_apron.msh")
    meta["z_plane"] = z_plane
    files.append("mesh_apron.msh")
    log(f"terrain meshes in {time.time() - t1:.1f}s (field verts {meta['mesh_field']['nvert']})")

    # --- vegetation ---
    t1 = time.time()
    builders = {k: wm.MeshBuilder() for k in ("veg_poplar", "veg_olive", "veg_citrus", "veg_pine", "veg_reed",
                                              "veg_plume")}
    ditch_dir = {}
    for s_ in site.scenery:
        g_ = _w(s_.geometry, frame)
        h = float(s_.props.get("height_m", 5.0))
        if s_.kind not in ("poplar", "olive", "carob", "citrus", "pine", "reed"):
            continue
        x, y = g_.x, g_.y
        z = float(hat(x, y))
        r = np.random.default_rng(int(abs(x * 1000) + abs(y * 7919)) % (2**31))
        if s_.kind == "poplar":
            wm.poplar(builders["veg_poplar"], x, y, z, h, r)
        elif s_.kind == "olive":
            wm.olive(builders["veg_olive"], x, y, z, h, r)
        elif s_.kind == "carob":
            wm.carob(builders["veg_olive"], x, y, z, h, r)
        elif s_.kind == "citrus":
            wm.citrus(builders["veg_citrus"], x, y, z, h, r)
        elif s_.kind == "pine":
            wm.pine(builders["veg_pine"], x, y, z, h, r)
        elif s_.kind == "reed":
            # lean slightly toward the nearest ditch (water)
            best = None
            for _, pts, _w_ in scene.ditches:
                for a_, b_ in zip(pts[:-1], pts[1:]):
                    d_, tpar = wt.seg_dist(np.array(x), np.array(y), a_, b_)
                    if best is None or d_ < best[0]:
                        L = np.hypot(b_[0] - a_[0], b_[1] - a_[1])
                        px = a_[0] + (b_[0] - a_[0]) * tpar / L
                        py = a_[1] + (b_[1] - a_[1]) * tpar / L
                        best = (float(d_), px, py)
            lean = None
            if best is not None and best[0] > 1e-3:
                lean = np.array([best[1] - x, best[2] - y, 0.0]) / best[0]
            wm.reed_clump(builders["veg_reed"], x, y, z, h, r, lean, plume_b=builders["veg_plume"])
    for k, b_ in builders.items():
        if b_.nvert:
            meta[f"mesh_{k}"] = b_.write(out / f"{k}.msh")
            files.append(f"{k}.msh")
    bw = wm.MeshBuilder()
    for zid, poly, species in scene.weeds:
        wm.weed_patch(bw, poly, species, hat, np.random.default_rng(wt._skey(zid) + 5),
                      sdf=wt.weed_sdf(zid, poly, scene.seed))
    meta["mesh_veg_weeds"] = bw.write(out / "veg_weeds.msh")
    files.append("veg_weeds.msh")
    log(f"vegetation meshes in {time.time() - t1:.1f}s " +
        str({k: meta[f'mesh_{k}']['nvert'] for k in list(builders) + ['veg_weeds'] if f'mesh_{k}' in meta}))

    # --- structures: greenhouses, poles/wires, stockpiles, ditch water, fallen log ---
    t1 = time.time()
    film, frame_b = wm.MeshBuilder(), wm.MeshBuilder()
    for poly, h in extra["greenhouses"]:
        wm.greenhouse_meshes(film, frame_b, poly, h, hat)
    meta["mesh_gh_film"] = film.write(out / "gh_film.msh")
    meta["mesh_gh_frame"] = frame_b.write(out / "gh_frame.msh")
    bp = wm.MeshBuilder()
    poles = sorted(extra["poles"], key=lambda p: p[0])
    wm.pole_and_wires(bp, [(p[0], p[1]) for p in poles], hat, H=poles[0][2] if poles else 9.0)
    meta["mesh_poles"] = bp.write(out / "poles.msh")
    bs = wm.MeshBuilder()
    for (sx_, sy_) in scene.stockpiles:
        wm.stockpile_mesh(bs, sx_, sy_, 5.2, 3.0, hat, rng)
    meta["mesh_stockpiles"] = bs.write(out / "stockpiles.msh")
    bwat = wm.MeshBuilder()
    for _, pts, w in scene.ditches:
        for a_, b_ in zip(pts[:-1], pts[1:]):
            a_ = np.array(a_)
            b_ = np.array(b_)
            L = np.linalg.norm(b_ - a_)
            d_ = (b_ - a_) / L
            sdir = np.array([-d_[1], d_[0]])
            ts = np.arange(0, L + 1e-6, 2.0)
            P = a_[None] + ts[:, None] * d_[None]
            P = P[(np.abs(P[:, 0]) <= HALF_X + 300) & (np.abs(P[:, 1]) <= HALF_Y + 300)]
            hw = w / 2 + 0.15
            Vw = []
            for p in P:
                for sgn in (-1, 1):
                    q = p + sgn * hw * sdir
                    qx, qy = np.clip(q[0], -HALF_X, HALF_X), np.clip(q[1], -HALF_Y, HALF_Y)
                    Vw.append([q[0], q[1], float(hat(qx, qy)) - 0.30])
            Vw = np.array(Vw)
            Fw = []
            for i in range(len(P) - 1):
                a0, a1, b0, b1 = 2 * i, 2 * i + 1, 2 * i + 2, 2 * i + 3
                Fw += [(a0, b1, a1), (a0, b0, b1)] if np.cross(np.r_[d_, 0], np.r_[sdir, 0])[2] > 0 else [(a0, a1, b1), (a0, b1, b0)]
            Fw = np.array(Fw)
            nrm = wm.smooth_normals(Vw, Fw)
            if nrm[:, 2].mean() < 0:
                Fw = Fw[:, ::-1]
            bwat.add(Vw, Fw, np.tile([0, 0, 1.0], (len(Vw), 1)),
                     np.stack([Vw[:, 0] / 8.0, Vw[:, 1] / 8.0], 1))
    meta["mesh_ditch_water"] = bwat.write(out / "ditch_water.msh")
    files += ["gh_film.msh", "gh_frame.msh", "poles.msh", "stockpiles.msh", "ditch_water.msh"]
    for o in site.obstacles:
        if o.kind == "log":
            pg = _w(o.polygon, frame)
            mrr = np.asarray(pg.minimum_rotated_rectangle.exterior.coords)[:4]
            e = [np.linalg.norm(mrr[(i + 1) % 4] - mrr[i]) for i in range(4)]
            i = int(np.argmax(e))
            ctr = np.asarray(pg.centroid.coords[0])
            ax_ = (mrr[(i + 1) % 4] - mrr[i]) / e[i]
            Lh = e[i] / 2 - 0.45
            a_, c_ = ctr - ax_ * Lh, ctr + ax_ * Lh
            bl = wm.MeshBuilder()
            wm.fallen_log(bl, a_, c_, 0.45, hat, np.random.default_rng(31))
            meta[f"mesh_obs_{o.oid}"] = bl.write(out / f"obs_{o.oid}.msh")
            files.append(f"obs_{o.oid}.msh")
    log(f"structure meshes in {time.time() - t1:.1f}s")
    meta["asset_files"] = files
    meta["build_seconds"] = time.time() - t0
    meta["builder_version"] = BUILDER_VERSION
    return meta


# ---------------------------------------------------------------------------
# MJCF
# ---------------------------------------------------------------------------


def _look(pos, target) -> str:
    return wr.cam_xyaxes(np.asarray(target, float) - np.asarray(pos, float))


def _world_xml(site, frame, terrain, scene, extra, fbox, meta, payload_mode, swarm_size) -> tuple[str, dict]:
    hat = terrain.height_at
    hmin, hmax = meta["h_min"], meta["h_max"]
    fx0, fy0, fx1, fy1 = fbox
    sun = sun_direction()
    sun_pos = -sun * 900.0 + np.array([0, 0, 0])
    A = []  # assets
    a = A.append
    a(f'<hfield name="terrain" file="terrain_hfield.bin" size="{HALF_X} {HALF_Y} {hmax - hmin:.6f} 0.5"/>')
    a('<texture name="sky" type="skybox" file="tex_sky.png" gridsize="3 4" gridlayout=".U..LFRB.D.."/>')
    for t in ("ground", "field", "apron", "atlas"):
        a(f'<texture name="tex_{t}" type="2d" file="tex_{t}.png"/>')
    for t in ("chips", "dust", "plaster", "ochre", "roof", "container", "solar", "tread", "panel", "bark"):
        a(f'<texture name="tex_{t}" type="2d" file="tex_{t}.png"/>')
    a('<material name="mat_ground" texture="tex_ground" specular="0.03" shininess="0.05"/>')
    a('<material name="mat_field" texture="tex_field" specular="0.03" shininess="0.05"/>')
    a('<material name="mat_apron" texture="tex_apron" specular="0.0" shininess="0.0"/>')
    a(f'<material name="mat_far" rgba="{wr.f(FAR_RGB)} 1" specular="0" shininess="0"/>')
    a('<material name="mat_veg" texture="tex_atlas" specular="0.04" shininess="0.08"/>')
    a('<material name="mat_plume" texture="tex_atlas" specular="0.1" shininess="0.2" emission="0.2"/>')
    a('<material name="mat_atlas_metal" texture="tex_atlas" specular="0.5" shininess="0.6"/>')
    a('<material name="mat_film" rgba="0.93 0.95 0.94 0.74" specular="0.6" shininess="0.8"/>')
    a('<material name="mat_pile" texture="tex_chips" specular="0.03" shininess="0.05"/>')
    a('<material name="mat_ditch_water" rgba="0.17 0.23 0.23 1" specular="0.9" shininess="0.95"/>')
    a(wr.robot_assets(swarm_size))
    a(wp.prop_assets())
    mesh_list = ["mesh_ground", "mesh_field", "mesh_apron", "veg_poplar", "veg_olive", "veg_citrus", "veg_pine",
                 "veg_reed", "veg_plume", "veg_weeds", "gh_film", "gh_frame", "poles", "stockpiles", "ditch_water"]
    for o in site.obstacles:
        if o.kind == "log":
            mesh_list.append(f"obs_{o.oid}")
    for m in mesh_list:
        a(f'<mesh name="{m}" file="{m}.msh" inertia="shell"/>')

    B = []  # worldbody
    b = B.append
    b(f'<light name="sun" type="directional" pos="{wr.f(sun_pos)}" dir="{wr.f(sun)}" castshadow="true" '
      f'diffuse="0.95 0.87 0.74" ambient="0.34 0.36 0.40" specular="0.35 0.32 0.28"/>')
    # cool sky fill from the side opposite the sun (lifts shadowed faces)
    fill = -sun * np.array([1, 1, 0]) + np.array([0, 0, -1.6])
    b(f'<light name="sky_fill" type="directional" pos="0 0 500" dir="{wr.f(fill / np.linalg.norm(fill))}" '
      'castshadow="false" diffuse="0.20 0.23 0.28" ambient="0 0 0" specular="0 0 0"/>')
    # terrain: physics hfield (hidden group 3) + visual meshes
    b(f'<geom name="terrain" type="hfield" hfield="terrain" pos="0 0 {hmin:.6f}" group="{GROUP_PHYSICS}" '
      f'contype="1" conaffinity="1" material="mat_ground" castshadow="false"/>')
    b('<geom name="ground_mesh" type="mesh" mesh="mesh_ground" material="mat_ground" class="ground"/>')
    b('<geom name="field_mesh" type="mesh" mesh="mesh_field" material="mat_field" class="ground"/>')
    b('<geom name="apron_mesh" type="mesh" mesh="mesh_apron" material="mat_apron" class="ground"/>')
    b(f'<geom name="far_plane" type="plane" size="0 0 1" pos="0 0 {meta["z_plane"]:.3f}" material="mat_far" class="ground"/>')
    b('<geom name="ditch_water" type="mesh" mesh="ditch_water" material="mat_ditch_water" class="ground"/>')
    # vegetation + structures
    for m in ("veg_poplar", "veg_olive", "veg_citrus", "veg_pine", "veg_reed"):
        b(f'<geom name="decor_{m[4:]}" type="mesh" mesh="{m}" material="mat_veg" class="decor"/>')
    b('<geom name="decor_weeds" type="mesh" mesh="veg_weeds" material="mat_veg" class="decor" castshadow="false"/>')
    b('<geom name="decor_reed_plumes" type="mesh" mesh="veg_plume" material="mat_plume" class="decor"/>')
    b('<geom name="decor_greenhouse_frames" type="mesh" mesh="gh_frame" material="mat_atlas_metal" class="decor"/>')
    b('<geom name="decor_greenhouse_film" type="mesh" mesh="gh_film" material="mat_film" class="decor" castshadow="false"/>')
    b('<geom name="decor_poles" type="mesh" mesh="poles" material="mat_veg" class="decor"/>')
    b('<geom name="decor_stockpiles" type="mesh" mesh="stockpiles" material="mat_pile" class="decor"/>')
    for idx, (poly, h) in enumerate(sorted(extra["houses"], key=lambda t: -t[0].area)):
        A.extend(wp.masia_assets(poly, h, idx == 0, idx))
        B.extend(wp.masia(poly, h, hat, main=idx == 0, idx=idx))
    # depot props
    dx0, dy0, dx1, dy1 = scene.depot.bounds
    dcy = (dy0 + dy1) / 2
    B.extend(wp.dump_truck(dx1 - 4.6, dcy + 1.2, np.pi + 0.06, hat))
    B.extend(wp.container_office(dx0 + 3.3, dcy, np.pi / 2, hat))
    if len(scene.stockpiles) >= 2:
        mx = 0.5 * (scene.stockpiles[0][0] + scene.stockpiles[1][0])
        B.extend(wp.pallets_and_bags(mx - 1.0, dy1 - 2.3, hat, np.random.default_rng(4)))
        B.extend(wp.skid_loader(scene.stockpiles[1][0] + 6.5, dcy - 3.0, 2.6, hat))
    # hidden obstacles + highlight marks
    names_obs = []
    for o in site.obstacles:
        pg = _w(o.polygon, frame)
        c = pg.centroid
        names_obs.append(o.oid)
        if o.kind == "tractor":
            B.extend(wp.tractor(o.oid, c.x, c.y, o.yaw_rad, hat))
        elif o.kind == "pump":
            B.extend(wp.pump(o.oid, c.x, c.y, o.yaw_rad, hat))
        elif o.kind == "log":
            b(f'<geom name="obs_{o.oid}_trunk" type="mesh" mesh="obs_{o.oid}" material="mat_veg" class="obs"/>')
        elif o.kind == "waterlogged":
            r = float(np.sqrt(pg.area / np.pi))
            zc = float(hat(c.x, c.y))
            b(wr.g(f"obs_{o.oid}_water", "box", [r * 0.62, r * 0.42, 0.02], pos=[c.x, c.y, zc + 0.095 - 0.02],
                   quat=wr.quat_axis((0, 0, 1), 0.35), material="water_puddle", cls="obs"))
            b(wr.g(f"obs_{o.oid}_water2", "box", [r * 0.38, r * 0.3, 0.02], pos=[c.x + 1.6, c.y - 1.2, zc + 0.092 - 0.02],
                   quat=wr.quat_axis((0, 0, 1), -0.5), material="water_puddle", cls="obs"))
            b(wr.g(f"obs_{o.oid}_sheen", "cylinder", [r * 0.86, 0.004], pos=[c.x, c.y, zc + 0.11],
                   material="water_sheen", cls="obs", extra='castshadow="false"'))
        B.extend(wp.obstacle_marker(o.oid, pg, o.height_m, hat, o.kind))
    # robots: mothership parked on the depot access, drones docked
    ms_x = float(np.mean([p[0] for p in scene.stockpiles])) if scene.stockpiles else 0.0
    ms_y = dy1 + 6.0
    ms_xml = wr.mothership_xml(start_pos=[ms_x, ms_y, float(hat(ms_x, ms_y))])
    cut = ms_xml.rstrip().rfind("</body>")
    B.append(ms_xml[:cut] + we.effector_bodies_xml() + "\n" + ms_xml[cut:])
    tendons = []
    for i in range(swarm_size):
        dk = wr.DOCKS[i % 4]
        lvl = i // 4
        p = [ms_x + dk[0], ms_y + dk[1], float(hat(ms_x, ms_y)) + wr.PAD_TOP_Z + wr.DRONE_SKID_Z + 2.5 * lvl]
        bx, tx = wr.drone_xml(i, p)
        B.append(bx)
        tendons.append(tx)
    B.append(wr.fx_xml())
    B.append(we.effector_fx_xml())
    # cameras
    tgt = np.array([10.0, 2.0, 2.0])
    pov = tgt + np.array([-235.0, -345.0, 250.0])
    b(f'<camera name="cam_overview" pos="{wr.f(pov)}" xyaxes="{_look(pov, tgt)}" fovy="42"/>')
    # MuJoCo rejects fovy >= 180 at compile time; load_world_model() sets CAM_TOP_EXTENT metres.
    b('<camera name="cam_top" pos="0 0 600" xyaxes="1 0 0 0 1 0" projection="orthographic" fovy="179"/>')
    pl = np.array([-48.0, 92.0, float(hat(-48, 92)) + 1.55])
    tl = np.array([-178.0, 70.0, float(hat(-178, 70)) + 7.0])
    b(f'<camera name="cam_low_field" pos="{wr.f(pl)}" xyaxes="{_look(pl, tl)}" fovy="50"/>')
    pd = np.array([-140.0, -95.0, float(hat(-140, -95)) + 9.0])
    td = np.array([-172.0, -40.0, float(hat(-172, -40)) + 4.0])
    b(f'<camera name="cam_ditch" pos="{wr.f(pd)}" xyaxes="{_look(pd, td)}" fovy="55"/>')
    if extra["houses"]:
        hc = np.mean([[g.centroid.x, g.centroid.y] for g, _ in extra["houses"]], axis=0)
        pf = np.array([150.0, 30.0, float(hat(150, 30)) + 22.0])
        tf = np.array([hc[0] + 5, hc[1] - 25, float(hat(hc[0], hc[1])) + 2.0])
        b(f'<camera name="cam_farmstead" pos="{wr.f(pf)}" xyaxes="{_look(pf, tf)}" fovy="55"/>')

    xml = f"""<mujoco model="prunemaxxing_world">
  <!-- Generated by prunemaxxing.sim.world_builder v{BUILDER_VERSION}. World frame: x=E-{frame.center_e:.3f}, y=N-{frame.center_n:.3f}, z=DEM m. -->
  <compiler angle="radian" meshdir="." texturedir="." autolimits="true"/>
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <size memory="128M"/>
  <statistic extent="{EXTENT}" center="0 0 3"/>
  <visual>
    <global offwidth="1920" offheight="1080" fovy="45" azimuth="35" elevation="-25"/>
    <quality shadowsize="8192" offsamples="8" numslices="32" numstacks="16"/>
    <headlight ambient="0.05 0.05 0.055" diffuse="0.14 0.14 0.14" specular="0.04 0.04 0.04"/>
    <map znear="0.0005" zfar="12" fogstart="0.8" fogend="5.5" haze="0.3" shadowclip="0.85"/>
    <rgba fog="0.79 0.80 0.78 1" haze="0.86 0.85 0.80 1"/>
  </visual>
  <default>
    <geom contype="0" conaffinity="0" group="{GROUP_GROUND}" condim="3"/>
    <default class="robot"><geom group="{GROUP_ROBOT}"/></default>
    <default class="tyre"><geom group="{GROUP_ROBOT}" contype="1" conaffinity="1" friction="1.1 0.02 0.002"/></default>
    <default class="decor"><geom group="{GROUP_DECOR}"/></default>
    <default class="ground"><geom group="{GROUP_GROUND}" castshadow="false"/></default>
    <default class="obs"><geom group="{GROUP_ROBOT}"/></default>
    <default class="mark"><geom group="{GROUP_GROUND}" castshadow="false"/></default>
    <default class="fx"><geom group="{GROUP_GROUND}"/></default>
    <default class="fxvis"><geom group="{GROUP_GROUND}" castshadow="false"/></default>
  </default>
  <asset>
{chr(10).join('    ' + x for x in A)}
  </asset>
  <worldbody>
{chr(10).join('    ' + x for x in B)}
  </worldbody>
  <tendon>
{chr(10).join(tendons)}
  </tendon>
</mujoco>
"""
    xml = _apply_payload_alpha_xml(xml, payload_mode)
    names = {
        "bodies": ["mothership", "ms_knuckle_fl", "ms_knuckle_fr", "ms_wheel_fl", "ms_wheel_fr", "ms_wheel_rl",
                   "ms_wheel_rr", "ms_axle_rl", "ms_axle_rr"]
        + [f"drone{i}" for i in range(swarm_size)]
        + [f"drone{i}_cradle" for i in range(swarm_size)]
        + [f"drone{i}_rotor{k}_body" for i in range(swarm_size) for k in range(6)],
        "joints": ["ms_free", "ms_wheel_fl", "ms_wheel_fr", "ms_wheel_rl", "ms_wheel_rr", "ms_steer_fl", "ms_steer_fr"]
        + [f"drone{i}_free" for i in range(swarm_size)]
        + [f"drone{i}_lift" for i in range(swarm_size)]
        + [f"drone{i}_rotor{k}" for i in range(swarm_size) for k in range(6)],
        "cameras": ["cam_overview", "cam_top", "cam_low_field", "cam_ditch", "cam_farmstead", "ms_front", "ms_mast",
                    "ms_side"] + [f"drone{i}_cam" for i in range(swarm_size)],
        "sites": ["ms_origin", "ms_lidar"] + [f"ms_dock_{k}" for k in range(4)],
        "mocap": [f"fx_dust_{k}" for k in range(wr.N_FX_DUST)] + [f"fx_cart_{k}" for k in range(wr.N_FX_CART)],
        "obstacles": names_obs,
        "obstacle_marks": [f"obs_mark_{o}" for o in names_obs],
        "lights": ["sun", "sky_fill"],
        "tendons": [f"drone{i}_cable{j}" for i in range(swarm_size) for j in range(4)],
    }
    return xml, names


def _is_dust(name: str) -> bool:
    return name.startswith("ms_dust") or ("_dust" in name and name.startswith("drone"))


def _is_cart(name: str) -> bool:
    return name.startswith("ms_cart") or ("_cart" in name and name.startswith("drone"))


def _apply_payload_alpha_xml(xml: str, mode: str) -> str:
    """Give toggled geoms explicit rgba (material colour) with alpha for the default mode."""
    import re

    mats = dict(re.findall(r'<material name="([^"]+)"[^>]*?rgba="([^"]+)"', xml))

    def fix(m):
        tag = m.group(0)
        nm = re.search(r'name="([^"]+)"', tag)
        if not nm:
            return tag
        name = nm.group(1)
        if not (_is_dust(name) or _is_cart(name)) or name.startswith("fx_"):
            return tag
        mat = re.search(r'material="([^"]+)"', tag)
        rgb = "1 1 1"
        if mat and mat.group(1) in mats:
            rgb = " ".join(mats[mat.group(1)].split()[:3])
        vis = (_is_dust(name) and mode == "dust") or (_is_cart(name) and mode == "cartridge")
        tag = re.sub(r'\srgba="[^"]*"', "", tag)
        return tag.replace("<geom ", f'<geom rgba="{rgb} {1 if vis else 0}" ', 1)

    return re.sub(r"<geom [^>]*/>", fix, xml)


# ---------------------------------------------------------------------------
# Runtime helpers
# ---------------------------------------------------------------------------


def load_world_model(spec: WorldSpec | str | Path):
    import mujoco

    path = spec.xml_path if isinstance(spec, WorldSpec) else Path(spec)
    model = mujoco.MjModel.from_xml_path(str(path))
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "cam_top")
    if cid >= 0:
        model.cam_fovy[cid] = CAM_TOP_EXTENT
    return model


def load_world_spec(out_dir: Path | str = DEFAULT_OUT, payload_mode: str = "dust", swarm_size: int = 4) -> WorldSpec:
    """Convenience: build_world() with caching (no regeneration if inputs unchanged)."""
    return build_world(payload_mode=payload_mode, swarm_size=swarm_size, out_dir=out_dir, verbose=False)


_GEOM_CACHE: dict[int, dict] = {}


def _geom_index(model) -> dict:
    key = id(model)
    c = _GEOM_CACHE.get(key)
    if c is not None and c.get("ngeom") == model.ngeom:
        return c
    import mujoco

    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "" for i in range(model.ngeom)]
    c = {"ngeom": model.ngeom, "names": names,
         "dust": [i for i, n in enumerate(names) if _is_dust(n) and not n.startswith("fx_")],
         "cart": [i for i, n in enumerate(names) if _is_cart(n) and not n.startswith("fx_")],
         "marks": {}}
    for i, n in enumerate(names):
        if n.startswith("obs_mark_"):
            rest = n[len("obs_mark_"):]
            for suffix, kind in (("_pinhead", "pinhead"), ("_pin", "pin")):
                if rest.endswith(suffix):
                    c["marks"].setdefault(rest[: -len(suffix)], []).append((i, kind))
                    break
            else:
                if "_seg" in rest:
                    c["marks"].setdefault(rest.rsplit("_seg", 1)[0], []).append((i, "seg"))
                else:
                    c["marks"].setdefault(rest, []).append((i, "fill"))
    _GEOM_CACHE[key] = c
    return c


def set_payload_mode(model, mode) -> None:
    """Show DUST or CARTRIDGE effectors (drone cradles + mothership hopper contents)."""
    mode = str(getattr(mode, "value", mode)).lower()
    if mode not in ("dust", "cartridge"):
        raise ValueError(mode)
    c = _geom_index(model)
    model.geom_rgba[c["dust"], 3] = 1.0 if mode == "dust" else 0.0
    model.geom_rgba[c["cart"], 3] = 1.0 if mode == "cartridge" else 0.0


def set_obstacle_detected(model, oid: str, detected: bool) -> None:
    """Toggle the glowing ground outline / fill / pin around obstacle `oid`."""
    c = _geom_index(model)
    if oid not in c["marks"]:
        raise KeyError(f"unknown obstacle id {oid}; known: {sorted(c['marks'])}")
    for gid, kind in c["marks"][oid]:
        model.geom_rgba[gid, 3] = MARK_ALPHA[kind] if detected else 0.0


def configure_scene(renderer, fog: bool = False, reflection: bool = True) -> None:
    """Enable the visual features the world is tuned for.

    MuJoCo fog also fogs the skybox (turning the sky flat grey), so distance haze
    is baked into the far farmland textures instead and fog stays off by default.
    Reflection (waterlogged puddle) costs one extra scene pass (~10 ms).
    """
    import mujoco

    fl = renderer.scene.flags
    fl[mujoco.mjtRndFlag.mjRND_SHADOW] = True
    fl[mujoco.mjtRndFlag.mjRND_SKYBOX] = True
    fl[mujoco.mjtRndFlag.mjRND_HAZE] = True
    fl[mujoco.mjtRndFlag.mjRND_FOG] = fog
    fl[mujoco.mjtRndFlag.mjRND_REFLECTION] = reflection


def make_renderer(model, height: int = 1080, width: int = 1920, max_geom: int = 12000):
    import mujoco

    r = mujoco.Renderer(model, height=height, width=width, max_geom=max_geom)
    configure_scene(r)
    return r


def focus_shadows(renderer, model, data, center=None, half_extent: float = 45.0) -> None:
    """Re-centre the sun's shadow box on `center` (world xyz) with +/- half_extent metres.

    MuJoCo's directional shadow map covers a square of +/- shadowClip around the
    light position (orthogonal to the light).  The default (center=None) covers the
    whole site (~+/-255 m, ~6 cm/texel); focusing on the robot with ~45 m gives
    ~1 cm/texel crisp shadows.  Call before ``renderer.update_scene``.
    """
    import mujoco

    lid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_LIGHT, "sun")
    sun = sun_direction()
    if center is None:
        pos = -sun * 900.0
        clip = model.stat.extent * model.vis.map.shadowclip
    else:
        pos = np.asarray(center, float) - sun * 400.0
        clip = float(half_extent)
    model.light_pos[lid] = pos
    data.light_xpos[lid] = pos
    renderer._mjr_context.shadowClip = clip


def _quat_from_euler_zyx(yaw, pitch=0.0, roll=0.0) -> np.ndarray:
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    return np.array([cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
                     cr * cp * sy - sr * sp * cy])


def _jnt_qadr(model, name):
    import mujoco

    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
        raise KeyError(name)
    return model.jnt_qposadr[jid]


def place_mothership(model, data, spec: WorldSpec, x: float, y: float, yaw: float, wheel_angle: float = 0.0,
                     steer: float = 0.0, follow_terrain: bool = True) -> None:
    """Kinematic pose: origin on terrain (furrow bottom), pitch/roll from terrain normal."""
    z = float(spec.terrain.height_at(x, y))
    pitch = roll = 0.0
    if follow_terrain:
        n = spec.terrain.normal_at(x, y, h=1.5)
        c, s = np.cos(yaw), np.sin(yaw)
        fwd = np.array([c, s])
        left = np.array([-s, c])
        pitch = -np.arctan2(n[0] * fwd[0] + n[1] * fwd[1], n[2])
        roll = np.arctan2(n[0] * left[0] + n[1] * left[1], n[2])
    q = _quat_from_euler_zyx(yaw, pitch, roll)
    a = _jnt_qadr(model, "ms_free")
    data.qpos[a : a + 3] = [x, y, z]
    data.qpos[a + 3 : a + 7] = q
    for k in ("fl", "fr", "rl", "rr"):
        data.qpos[_jnt_qadr(model, f"ms_wheel_{k}")] = wheel_angle
    for k in ("fl", "fr"):
        data.qpos[_jnt_qadr(model, f"ms_steer_{k}")] = steer


def place_drone(model, data, i: int, x: float, y: float, z: float, yaw: float = 0.0, lift: float = 0.0,
                rotor_phase: float = 0.0, tilt=(0.0, 0.0)) -> None:
    a = _jnt_qadr(model, f"drone{i}_free")
    data.qpos[a : a + 3] = [x, y, z]
    data.qpos[a + 3 : a + 7] = _quat_from_euler_zyx(yaw, tilt[0], tilt[1])
    data.qpos[_jnt_qadr(model, f"drone{i}_lift")] = float(np.clip(lift, wr.LIFT_RANGE[0], wr.LIFT_RANGE[1]))
    for k in range(6):
        data.qpos[_jnt_qadr(model, f"drone{i}_rotor{k}")] = rotor_phase * (1 if k % 2 == 0 else -1) + k * 0.7


def dock_world_pose(model, data, k: int) -> np.ndarray:
    """World xyz of mothership dock site k (call after mj_forward)."""
    import mujoco

    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"ms_dock_{k}")
    return data.site_xpos[sid].copy()


def park_fx(model, data) -> None:
    data.mocap_pos[:, 2] = wr.FX_PARK_Z


def fx_mocap_ids(model, kind: str = "dust") -> np.ndarray:
    """Mocap indices of the FX particles; kind is 'dust' or 'cart' / 'cartridge'."""
    import mujoco

    kind = str(getattr(kind, "value", kind)).lower()
    kind = "cart" if kind.startswith("cart") else "dust"
    n = wr.N_FX_DUST if kind == "dust" else wr.N_FX_CART
    return np.array([model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"fx_{kind}_{k}")]
                     for k in range(n)])


# ---------------------------------------------------------------------------
# Mulch painter
# ---------------------------------------------------------------------------


def _bilinear_matrix(p: np.ndarray, n: int) -> np.ndarray:
    """Dense (len(p) x n) bilinear interpolation weights for fractional indices p (clamped)."""
    p = np.clip(p, 0.0, n - 1.0)
    i0 = np.floor(p).astype(int)
    i0 = np.minimum(i0, n - 2) if n > 1 else np.zeros_like(i0)
    t = (p - i0).astype(np.float32)
    W = np.zeros((len(p), n), np.float32)
    rows = np.arange(len(p))
    if n == 1:
        W[:, 0] = 1
        return W
    W[rows, i0] = 1 - t
    W[rows, i0 + 1] += t
    return W


class MulchPainter:
    """Blends a pre-baked 'mulched' field texture over the base field texture.

    completion grid: (nrows, ncols) like FieldTwin (rows north -> south), values in
    [0, 1]; NaN / negative = not a work cell (no mulch).  The per-cell values are
    bilinearly upsampled, thresholded against a baked organic noise field and
    weighted by a beds-only mask, so coverage grows in natural patches instead of
    blocky squares.  Only cells whose value changed (+1 cell halo) are re-blended;
    then the field texture is re-uploaded to the GPU.
    """

    SOFT = 0.07

    def __init__(self, spec: WorldSpec, model, mode: str = "dust"):
        import mujoco

        mode = str(getattr(mode, "value", mode)).lower()
        if mode not in ("dust", "cartridge"):
            raise ValueError(mode)
        self.mode = mode
        self.tex_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TEXTURE, "tex_field")
        w = int(model.tex_width[self.tex_id])
        h = int(model.tex_height[self.tex_id])
        nch = int(model.tex_nchannel[self.tex_id])
        adr = int(model.tex_adr[self.tex_id])
        self.w, self.h, self.nch = w, h, nch
        self.base = np.load(spec.out_dir / "field_base.npy")
        self.mulch = np.load(spec.out_dir / f"field_mulch_{mode}.npy")
        masks = np.load(spec.out_dir / "field_mulch_masks.npy")
        assert self.base.shape[:2] == (h, w), (self.base.shape, (h, w))
        self.thr = masks[..., 0].astype(np.float32) / 255.0
        self.wgt = masks[..., 1].astype(np.float32) / 255.0
        self.diff = self.mulch.astype(np.int16) - self.base.astype(np.int16)
        cg = spec.cells
        fx0, fy0, fx1, fy1 = spec.field_box
        xs = fx0 + (np.arange(w) + 0.5) * (fx1 - fx0) / w
        ys = fy1 - (np.arange(h) + 0.5) * (fy1 - fy0) / h
        colf = (xs - cg.x0) / cg.dx - 0.5
        rowf = (cg.y1 - ys) / cg.dy - 0.5
        self.Wc = _bilinear_matrix(colf, cg.ncols)
        self.Wr = _bilinear_matrix(rowf, cg.nrows)
        # pixel spans per cell (nearest cell)
        rn = np.clip(np.round(rowf).astype(int), 0, cg.nrows - 1)
        cn = np.clip(np.round(colf).astype(int), 0, cg.ncols - 1)
        self.row_span = [(int(np.searchsorted(rn, r, "left")), int(np.searchsorted(rn, r, "right")))
                         for r in range(cg.nrows)]
        self.col_span = [(int(np.searchsorted(cn, c, "left")), int(np.searchsorted(cn, c, "right")))
                         for c in range(cg.ncols)]
        self.nrows, self.ncols = cg.nrows, cg.ncols
        self.last = np.zeros((cg.nrows, cg.ncols), np.float32)
        self.last_ms = 0.0
        self._view(model)[..., :3] = self.base

    def _view(self, model) -> np.ndarray:
        adr = int(model.tex_adr[self.tex_id])
        return model.tex_data[adr : adr + self.w * self.h * self.nch].reshape(self.h, self.w, self.nch)

    @staticmethod
    def sanitize(grid) -> np.ndarray:
        g = np.asarray(grid, np.float32).copy()
        g[~np.isfinite(g)] = 0.0
        return np.clip(g, 0.0, 1.0)

    def apply(self, model, renderer, completion_grid, force: bool = False, upload: bool = True) -> float:
        """Update the field texture for `completion_grid`; returns elapsed ms."""
        import mujoco
        from scipy.ndimage import binary_dilation

        t0 = time.perf_counter()
        C = self.sanitize(completion_grid)
        if C.shape != (self.nrows, self.ncols):
            raise ValueError(f"completion grid must be {(self.nrows, self.ncols)}, got {C.shape}")
        changed = np.abs(C - self.last) > (0.5 / 255) if not force else np.ones_like(C, bool)
        if not changed.any():
            self.last_ms = (time.perf_counter() - t0) * 1000
            return self.last_ms
        dirty = binary_dilation(changed, structure=np.ones((3, 3), bool))
        view = self._view(model)
        s = self.SOFT
        for r in range(self.nrows):
            cols = np.flatnonzero(dirty[r])
            if cols.size == 0:
                continue
            i0, i1 = self.row_span[r]
            if i1 <= i0:
                continue
            # contiguous runs of dirty columns
            breaks = np.flatnonzero(np.diff(cols) > 1)
            starts = np.r_[cols[0], cols[breaks + 1]]
            ends = np.r_[cols[breaks], cols[-1]]
            WrC = self.Wr[i0:i1] @ C  # (h, ncols)
            for c0, c1 in zip(starts, ends):
                j0 = self.col_span[c0][0]
                j1 = self.col_span[c1][1]
                if j1 <= j0:
                    continue
                cup = WrC @ self.Wc[j0:j1].T
                cup = cup * (1 + 2 * s) - s
                thr = self.thr[i0:i1, j0:j1]
                a = np.clip((cup - thr) / (2 * s) + 0.5, 0.0, 1.0)
                a = a * a * (3 - 2 * a) * self.wgt[i0:i1, j0:j1]
                blend = self.base[i0:i1, j0:j1].astype(np.int16) + (self.diff[i0:i1, j0:j1] * a[..., None]).astype(np.int16)
                view[i0:i1, j0:j1, :3] = blend.astype(np.uint8)
        self.last = C
        if upload and renderer is not None:
            if hasattr(renderer, "_gl_context") and renderer._gl_context is not None:
                renderer._gl_context.make_current()
            ctx = renderer._mjr_context if hasattr(renderer, "_mjr_context") else renderer
            mujoco.mjr_uploadTexture(model, ctx, self.tex_id)
        self.last_ms = (time.perf_counter() - t0) * 1000
        return self.last_ms

    def reset(self, model, renderer=None) -> None:
        import mujoco

        self._view(model)[..., :3] = self.base
        self.last[:] = 0
        if renderer is not None:
            mujoco.mjr_uploadTexture(model, renderer._mjr_context, self.tex_id)


__all__ = [
    "WorldSpec",
    "CellGrid",
    "build_world",
    "load_world_model",
    "load_world_spec",
    "set_payload_mode",
    "set_obstacle_detected",
    "MulchPainter",
    "make_renderer",
    "configure_scene",
    "focus_shadows",
    "place_mothership",
    "place_drone",
    "dock_world_pose",
    "park_fx",
    "fx_mocap_ids",
    "sun_direction",
]
