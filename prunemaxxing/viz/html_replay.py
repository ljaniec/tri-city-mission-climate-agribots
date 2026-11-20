"""Interactive, offline, single-file HTML mission replay.

``export_html_replay(res, out_path)`` writes ONE self-contained ``.html`` file:
the orthophoto-style basemap (WebP data URI), the finished mission (JSON +
base64 typed arrays, coordinates quantised to decimetres relative to the
basemap's SW corner) and an inline vanilla-JS canvas player. No network access
at all: no CDN, fonts, scripts or images are fetched.

The player rebuilds everything from the raw mission arrays at the playhead:
mothership pose (10 Hz log), drone positions (dock -> patch -> sweep -> dock,
timed by t_launch / apply_t0 / apply_t1 / t_land), mulch / cartridge stamps per
application patch, mothership effector bursts, hopper ledger, counters and
per-parcel progress. Charts come from :class:`MissionTimeline` and the climate
box from :func:`impact_summary` / :func:`factor_table`.

URL hash: ``#t=0.42&play=0&follow=1&speed=auto&z=6`` (t = fraction of mission,
z = zoom in px/m).
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from prunemaxxing.kpis.impact import factor_table
from prunemaxxing.planning import ms_effectors as fx
from prunemaxxing.planning.mission import LABEL_CODES, MissionResult
from prunemaxxing.planning.payload_modes import PayloadMode
from prunemaxxing.planning.swarm import DOCK_OFFSETS
from prunemaxxing.viz.basemap import Basemap, build_basemap
from prunemaxxing.viz.mission_stats import DRONE_STATE_COLORS, MissionTimeline
from prunemaxxing.viz.mission_viz import (
    ACTUAL,
    BG,
    BYPASS,
    CHIP,
    FXC,
    INK,
    MS_LEN,
    MS_WID,
    MUTED,
    OBST,
    PANEL,
    PLANNED,
    SEG_COLORS,
    WEED,
)
from prunemaxxing.viz.style import COLORS, DRONE_PATH_COLORS

CROP_LABELS = {
    "sweet_potato_beauregard": "Sweet potato (Beauregard)",
    "carboniato_ohenry": "Carboniato (O'Henry)",
    "cover_then_sweet_potato": "Cover crop, then sweet potato",
    "horticulture_mix": "Mixed horticulture",
}
OBSTACLE_LABELS = {
    "tractor": "Broken-down tractor",
    "log": "Fallen poplar log",
    "pump": "Irrigation pump + pipe stack",
    "waterlogged": "Waterlogged soft ground",
}
OBSTACLE_SHORT = {"tractor": "Tractor", "log": "Log", "pump": "Pump", "waterlogged": "Waterlogged"}
SLOW_MO_SPEED = 20.0
AUTO_PLAY_S = 90.0
CLOCK_START_H = 7.0  # same convention as the dashboard video (mission starts 07:00, day 1)


# ---------------------------------------------------------------------------
# encoding helpers
# ---------------------------------------------------------------------------


def _b64(a: np.ndarray, dtype: str) -> list[str]:
    np_dtype = {"f4": "<f4", "f8": "<f8", "i1": "i1", "u1": "u1", "i2": "<i2", "u2": "<u2", "i4": "<i4",
                "u4": "<u4"}[dtype]
    raw = np.ascontiguousarray(np.asarray(a), dtype=np_dtype).tobytes()
    return [dtype, base64.b64encode(raw).decode("ascii")]


def _dm(v) -> np.ndarray:
    """Metres -> int16 decimetres (local frame)."""
    return np.clip(np.round(np.asarray(v, dtype=float) * 10.0), -32768, 32767).astype("<i2")


def _clean(x: Any, nd: int = 4) -> Any:
    if isinstance(x, dict):
        return {str(k): _clean(v, nd) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v, nd) for v in x]
    if isinstance(x, np.ndarray):
        return _clean(x.tolist(), nd)
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        v = float(x)
        return None if not math.isfinite(v) else round(v, nd)
    if hasattr(x, "value") and isinstance(getattr(x, "value"), str):
        return x.value
    return x


def _basemap_uri(bm: Basemap, max_px: int, quality: int) -> tuple[str, tuple[int, int]]:
    img = Image.fromarray(bm.rgb)
    if max(img.size) > max_px:
        k = max_px / max(img.size)
        img = img.resize((round(img.width * k), round(img.height * k)), Image.LANCZOS)
    buf = io.BytesIO()
    try:
        img.save(buf, "WEBP", quality=quality, method=5)
        mime = "image/webp"
    except (OSError, KeyError, ValueError):
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=quality, optimize=True)
        mime = "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(buf.getvalue()).decode("ascii"), img.size


def _pretty(s: str) -> str:
    return s.replace("_", " ").strip().capitalize()


# ---------------------------------------------------------------------------
# payload
# ---------------------------------------------------------------------------


def build_replay_payload(res: MissionResult, bm: Basemap, title: str | None = None) -> dict[str, Any]:
    """Everything the player needs, as JSON-ready dict (binary arrays under ``bin``)."""
    e0, e1, n0, n1 = bm.extent
    W, H = e1 - e0, n1 - n0

    def lxy(e, n):
        return np.asarray(e, float) - e0, np.asarray(n, float) - n0

    def poly(geom) -> list[float]:
        if geom is None or geom.is_empty:
            return []
        if geom.geom_type == "MultiPolygon":
            geom = max(geom.geoms, key=lambda g: g.area)
        c = np.asarray(geom.exterior.coords)[:-1]
        return np.round(np.column_stack(lxy(c[:, 0], c[:, 1])), 2).ravel().tolist()

    site, plan, cfg, spec = res.site, res.plan, res.cfg, res.spec
    cart = spec.mode is PayloadMode.CARTRIDGE
    mode = spec.mode.value
    tl = MissionTimeline(res)
    k = res.kpis
    bins: dict[str, list[str]] = {}

    # ---- mothership log
    L = res.ms_log
    x, y = lxy(L["x"], L["y"])
    bins["ms_t"] = _b64(L["t"], "f8")
    bins["ms_x"] = _b64(_dm(x), "i2")
    bins["ms_y"] = _b64(_dm(y), "i2")
    bins["ms_yaw"] = _b64(L["yaw"], "f4")
    bins["ms_v"] = _b64(np.clip(np.round(np.abs(L["v"]) * 100), 0, 65535), "u2")
    bins["ms_mode"] = _b64(L["mode"], "u1")
    bins["ms_odo"] = _b64(L["odo"], "f4")
    px, py = lxy(plan.path[:, 0], plan.path[:, 1])
    bins["plan_xy"] = _b64(_dm(np.column_stack([px, py]).ravel()), "i2")

    # ---- application patterns (global id per (cell, pattern))
    gid: dict[tuple[int, int], int] = {}
    off, pts = [0], []
    for cid in sorted(res.patterns):
        for pi, pat in enumerate(res.patterns[cid]):
            gid[(cid, pi)] = len(off) - 1
            a, b = lxy(pat[:, 0], pat[:, 1])
            pts.append(np.column_stack([a, b]).ravel())
            off.append(off[-1] + len(pat))
    bins["pat_off"] = _b64(np.array(off), "u4")
    bins["pat_xy"] = _b64(_dm(np.concatenate(pts) if pts else np.zeros(0)), "i2")

    # ---- drone sorties
    S = res.sorties
    ns = len(S["kg"])
    dx, dy = lxy(S["dock_x"], S["dock_y"])
    bins["s_drone"] = _b64(S["drone"], "u1")
    bins["s_cell"] = _b64(S["cell"], "u2")
    bins["s_pat"] = _b64(np.array([gid[(int(c), int(p))] for c, p in zip(S["cell"], S["pattern"])], dtype=int), "u4")
    for key, col in (("s_tload", "t_load"), ("s_tlaunch", "t_launch"), ("s_a0", "apply_t0"), ("s_a1", "apply_t1"),
                     ("s_tland", "t_land"), ("s_kg", "kg")):
        bins[key] = _b64(S[col], "f4")
    bins["s_dock"] = _b64(_dm(np.column_stack([dx, dy]).ravel()), "i2")

    # ---- mothership effector bursts
    Mh = res.ms_shots
    nm = len(Mh.get("kg", []))
    if nm:
        mx, my = lxy(Mh["ms_x"], Mh["ms_y"])
        bins["m_kind"] = _b64(Mh["kind"], "u1")
        bins["m_side"] = _b64(Mh["side"], "i1")
        bins["m_cell"] = _b64(Mh["cell"], "u2")
        bins["m_pat"] = _b64(np.array([gid[(int(c), int(p))] for c, p in zip(Mh["cell"], Mh["pattern"])]), "u4")
        for key, col in (("m_t0", "t0"), ("m_tf0", "t_fire0"), ("m_t1", "t1"), ("m_tfree", "t_free"), ("m_kg", "kg"),
                         ("m_yaw", "ms_yaw"), ("m_range", "range_m")):
            bins[key] = _b64(Mh[col], "f4")
        bins["m_units"] = _b64(Mh["units"], "u2")
        bins["m_station"] = _b64(Mh["station"], "u1")
        bins["m_ms"] = _b64(_dm(np.column_stack([mx, my]).ravel()), "i2")
    else:
        for key, dt in (("m_kind", "u1"), ("m_side", "i1"), ("m_cell", "u2"), ("m_pat", "u4"), ("m_t0", "f4"),
                        ("m_tf0", "f4"), ("m_t1", "f4"), ("m_tfree", "f4"), ("m_kg", "f4"), ("m_yaw", "f4"),
                        ("m_range", "f4"), ("m_units", "u2"), ("m_station", "u1"), ("m_ms", "i2")):
            bins[key] = _b64(np.zeros(0), dt)

    # ---- deposits (source of truth for applied kg per cell)
    Dp = res.deposits
    bins["d_t"] = _b64(Dp["t"], "f8")
    bins["d_cell"] = _b64(Dp["cell"], "u2")
    bins["d_kg"] = _b64(Dp["kg"], "f4")

    # ---- battery swaps, tender top-ups (hopper ledger: the simulator applies a top-up at the first
    #      load once the truck has arrived and transferred)
    ev = res.events
    sw = sorted((e["t"], e["drone"]) for e in ev if e["kind"] == "battery_swap")
    bins["sw_t"] = _b64(np.array([a for a, _ in sw]), "f4")
    bins["sw_d"] = _b64(np.array([b for _, b in sw], dtype=int), "u1")
    tops = sorted((e["t"], e["kg"]) for e in ev if e["kind"] == "tender_topup")
    loads = np.sort(np.concatenate([S["t_load"], Mh["t0"] if nm else np.zeros(0)]))
    t_arr = np.array([a for a, _ in tops], dtype=float) + cfg.tender_transfer_s
    j = np.searchsorted(loads, t_arr, side="left")
    t_app = np.where(j < len(loads), loads[np.clip(j, 0, max(len(loads) - 1, 0))] if len(loads) else t_arr, t_arr)
    bins["tp_t"] = _b64(np.array([a for a, _ in tops]), "f4")
    bins["tp_tap"] = _b64(t_app, "f8")
    bins["tp_kg"] = _b64(np.array([b for _, b in tops]), "f4")

    # ---- cells
    cells = plan.cells
    parcels = [p.pid for p in site.parcels]
    pidx = {p: i for i, p in enumerate(parcels)}
    done = tl.cell_done_t
    cells_js = {
        "poly": [poly(c.polygon) if c.polygon is not None else
                 np.round([c.e - e0 - 1, c.n - n0 - 1, c.e - e0 + 1, c.n - n0 - 1, c.e - e0 + 1, c.n - n0 + 1,
                           c.e - e0 - 1, c.n - n0 + 1], 2).tolist() for c in cells],
        "parcel": [pidx.get(c.parcel_id, -1) for c in cells],
        "area": [round(float(c.area_m2), 2) for c in cells],
        "target": [round(float(v), 2) for v in res.cell_target_kg],
        "done": [None if not np.isfinite(v) else round(float(v), 1) for v in done],
        "station": [int(c.station) for c in cells],
        "blocked": [bool(b) for b in res.cell_blocked],
    }

    # ---- obstacles, detours, slow-motion windows
    det_ev = {e["oid"]: e for e in ev if e["kind"] == "obstacle_detected"}
    detours = {d["oid"]: d for d in res.detours}
    lt, lmode = L["t"], L["mode"]
    obstacles, slow = [], []
    for ob in site.obstacles:
        e = det_ev.get(ob.oid)
        d = detours.get(ob.oid)
        o = {"oid": ob.oid, "kind": ob.kind, "name": OBSTACLE_LABELS.get(ob.kind, _pretty(ob.kind)),
             "short": OBSTACLE_SHORT.get(ob.kind, _pretty(ob.kind)),
             "note": ob.props.get("note", ""), "height": ob.height_m, "poly": poly(ob.polygon),
             "sensor": "lidar" if ob.height_m > 0.3 else "camera", "t_det": None}
        c = ob.polygon.centroid
        o["c"] = list(np.round(lxy(c.x, c.y), 2))
        if e is not None:
            o.update(t_det=float(e["t"]), range_m=float(e["range_m"]), action=e.get("action", ""),
                     det_xy=list(np.round(lxy(e["xy"][0], e["xy"][1]), 2)))
            i0 = int(np.searchsorted(lt, e["t"]))
            run = np.flatnonzero(lmode[i0:] == LABEL_CODES["obstacle_detour"])
            t_det = float(e["t"])
            if len(run):
                a = i0 + int(run[0])
                b = a
                while b + 1 < len(lmode) and lmode[b + 1] == LABEL_CODES["obstacle_detour"]:
                    b += 1
                t_start, t_end = float(lt[a]), float(lt[b])
            else:
                t_start, t_end = t_det, t_det + 60.0
            # detection while parked: the bypass only starts after the station dwell, so slow down twice
            if t_start - t_det > 90.0:
                slow += [[t_det - 25.0, t_det + 35.0], [t_start - 12.0, t_end + 8.0]]
            else:
                slow.append([t_det - 25.0, t_end + 8.0])
            o["t_bypass"], o["t_clear"] = t_start, t_end
        if d is not None:
            a, b = lxy(d["pts"][:, 0], d["pts"][:, 1])
            ra, rb = lxy(d["replaced"][:, 0], d["replaced"][:, 1])
            o["detour"] = {"t": float(d["t"]), "pts": np.round(np.column_stack([a, b]), 2).ravel().tolist(),
                           "replaced": np.round(np.column_stack([ra, rb]), 2).ravel().tolist(),
                           "length": float(d["length_m"]), "side": d["side"], "offset": float(d["offset_m"])}
        obstacles.append(o)

    # ---- stations (actual parking poses)
    stations = []
    for v in res.visits:
        st = plan.stations[v["sid"]]
        sx, sy = lxy(v["x"], v["y"])
        stations.append({"sid": int(v["sid"]), "ta": float(v["t_arrive"]), "td": float(v["t_depart"]),
                         "x": float(sx), "y": float(sy), "yaw": float(v["yaw"]), "sorties": int(v.get("sorties", 0)),
                         "bursts": int(v.get("ms_bursts", 0)), "ncells": len(v["cells"]),
                         "parcel": pidx.get(st.parcel_id, -1)})

    # ---- event log
    n_st = len(res.visits)
    events = []
    for e in ev:
        kd = e["kind"]
        if kd == "depart_depot":
            events.append({"t": e["t"], "kind": kd, "label": "Mothership leaves the depot, hopper full"})
        elif kd == "arrive_station":
            sid = int(e["sid"])
            vis = next((s for s in stations if s["sid"] == sid), None)
            p = parcels[vis["parcel"]][0] if vis and vis["parcel"] >= 0 else "?"
            extra = f" · {vis['sorties']:,} sorties" if vis else ""
            events.append({"t": e["t"], "kind": kd, "label": f"Station {sid + 1}/{n_st} · parcel {p}{extra}"})
        elif kd == "obstacle_detected":
            name = OBSTACLE_LABELS.get(e.get("obstacle_type", ""), "obstacle").lower()
            sensor = "Camera" if e.get("obstacle_type") == "waterlogged" else "Lidar"
            events.append({"t": e["t"], "kind": kd, "oid": e["oid"],
                           "label": f"{sensor} contact: {name} at {e['range_m']:.1f} m (not in satellite map)"})
        elif kd == "detour_spliced":
            d = detours.get(e["oid"])
            side = f"{d['side']} +{d['offset_m']:.1f} m, " if d else ""
            events.append({"t": e["t"], "kind": kd, "oid": e["oid"],
                           "label": f"Online bypass spliced: {side}{e['length_m']:.0f} m"})
        elif kd == "arrive_depot":
            events.append({"t": e["t"], "kind": kd, "label": "Back at the depot, mission complete"})
    events.sort(key=lambda r: r["t"])

    # ---- geometry
    parcels_js = [{"pid": p.pid, "letter": p.pid.split("_")[0], "crop": p.crop,
                   "crop_label": CROP_LABELS.get(p.crop, _pretty(p.crop)), "poly": poly(p.polygon),
                   "target_t": float(tl.parcel_target_kg[i]) / 1000.0}
                  for i, p in enumerate(site.parcels)]
    weeds_js = [{"id": w.zid, "species": w.props.get("species", "").replace("_", " "),
                 "severity": w.props.get("severity", ""), "poly": poly(w.polygon)} for w in site.weeds]
    inflated = [poly(g) for g in plan.inflated_weeds]
    bx0, by0, bx1, by1 = site.bounds
    sb = [bx0 - e0, by0 - n0, bx1 - e0, by1 - n0]
    depot = list(np.round(lxy(plan.depot_xy[0], plan.depot_xy[1]), 2))
    stock = [list(np.round(lxy(p.x, p.y), 2)) for p in site.stockpiles]

    imp = dict(tl.impact)
    ep = cfg.effectors
    fx_name = "twin air cannons" if cart else "chip blower"
    mode_label = ("Seed + mycorrhiza cartridges" if cart else "Wood-chip mulch") + f" · {spec.dose_kg_per_m2:g} kg/m²"
    meta = {
        "title": title or f"PruneMaxxing · {mode.upper()} mission replay",
        "mode": mode, "mode_label": mode_label, "spec_desc": spec.description,
        "site": "Parc Agrari del Baix Llobregat · Viladecans", "t_end": float(res.t_end),
        "W_m": W, "H_m": H, "origin_utm": [e0, n0], "site_box": sb, "basemap_res": bm.res,
        "dose": spec.dose_kg_per_m2, "unit_kg": spec.unit_mass_kg, "units_per_sortie": spec.units_per_sortie,
        "n_drones": res.n_drones, "lidar_m": cfg.lidar_range_m, "camera_m": cfg.camera_range_m,
        "reach_m": ep.range_max_m, "reach_min_m": ep.range_min_m, "hopper_cap": cfg.hopper_capacity_kg,
        "tender_request_frac": cfg.tender_request_frac, "start_h": CLOCK_START_H, "swap_s": cfg.drone.swap_time_s,
        "dock_offsets": DOCK_OFFSETS.tolist(), "cannon_pivots": [list(p[:2]) for p in fx.CANNON_PIVOTS],
        "cannon_h": fx.CANNON_PIVOTS[0][2], "cannon_elev_deg": ep.cannon_elev_deg, "shot_gap": fx.shot_gap(ep),
        "cannon_sigma": ep.cannon_sigma_m, "cannon_sigma_per_m": ep.cannon_sigma_per_m,
        "cart_unit_kg": ep.unit_kg, "effector": k.get("ms_effector", fx_name), "fx_name": fx_name,
        "n_stations": n_st, "area_ha": k.get("area_treated_ha"), "material_t": k.get("biomass_applied_t"),
        "sorties_total": int(ns), "bursts_total": int(nm), "slow": slow, "slow_speed": SLOW_MO_SPEED,
        "auto_speed": float(res.t_end) / AUTO_PLAY_S, "auto_play_s": AUTO_PLAY_S, "ms_len": MS_LEN, "ms_wid": MS_WID, "depot": depot,
        "stockpiles": stock, "species": [s.replace("_", " ") for s in spec.local_tree_species],
        "parcels": parcels, "label_codes": LABEL_CODES,
    }
    colors = {
        "bg": BG, "panel": PANEL, "ink": INK, "muted": MUTED, "amber": COLORS["amber"], "planned": PLANNED,
        "actual": ACTUAL, "bypass": BYPASS, "weed": WEED, "obst": OBST, "fx": FXC, "chip": CHIP,
        "drones": [a for _, a in DRONE_PATH_COLORS], "drones_planned": [p for p, _ in DRONE_PATH_COLORS],
        "states": DRONE_STATE_COLORS, "seg": SEG_COLORS, "green": SEG_COLORS["lane"],
    }
    kpis = {kk: v for kk, v in k.items() if not isinstance(v, (list, dict))}
    return _clean({
        "meta": meta, "colors": colors, "kpis": kpis, "impact": imp, "factors": factor_table(),
        "timeline": tl.to_json(1200), "parcels": parcels_js, "weeds": weeds_js, "inflated_weeds": inflated,
        "obstacles": obstacles, "stations": stations, "cells": cells_js, "events": events, "bin": bins,
    })


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------


def export_html_replay(res: MissionResult, out_path, basemap: Basemap | None = None, title: str | None = None,
                       basemap_max_px: int = 2400, basemap_quality: int = 86) -> Path:
    """Write a single self-contained HTML replay of ``res`` to ``out_path``.

    ``basemap`` defaults to :func:`build_basemap(res.site) <prunemaxxing.viz.basemap.build_basemap>`;
    pass a prebuilt one to reuse it across missions on the same site.
    """
    out_path = Path(out_path)
    bm = basemap if basemap is not None else build_basemap(res.site)
    payload = build_replay_payload(res, bm, title)
    uri, _ = _basemap_uri(bm, basemap_max_px, basemap_quality)
    data = json.dumps(payload, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    title_txt = html.escape(payload["meta"]["title"])
    doc = "".join([
        "<!DOCTYPE html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">",
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">",
        f"<title>{title_txt}</title><style>{_CSS}</style></head><body>",
        _BODY,
        f"<img id=\"basemap\" alt=\"\" hidden src=\"{uri}\">",
        f"<script type=\"application/json\" id=\"replay-data\">{data}</script>",
        f"<script>{_JS}</script></body></html>\n",
    ])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(doc, encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# page template
# ---------------------------------------------------------------------------

_CSS = r"""
:root{--bg:#0b100c;--panel:#121a13;--panel2:#172118;--card:#151f16;--line:#253226;--line2:#33443a;--ink:#e8efe4;
--muted:#9aab94;--dim:#66765f;--amber:#e0972f;--cyan:#38d6e6;--planned:#f3dc8a;--bypass:#ff8c2a;--weed:#ff4d3d;
--obst:#ff3df2;--fx:#ffe066;--chip:#e0b97a;--green:#7bdc6a;--seed:#9be15d;
--sans:"Inter","Segoe UI",system-ui,-apple-system,"Noto Sans","DejaVu Sans",Roboto,Helvetica,Arial,sans-serif;
--mono:"JetBrains Mono","SFMono-Regular",Menlo,Consolas,"DejaVu Sans Mono","Liberation Mono",monospace}
*{box-sizing:border-box}
html,body{margin:0;height:100%;background:var(--bg);color:var(--ink);font:13px/1.35 var(--sans);overflow:hidden}
button,select{font:inherit;color:inherit}
#app{display:grid;grid-template-rows:58px minmax(0,1fr) 60px;height:100vh;width:100vw}
header{display:flex;align-items:center;gap:20px;padding:0 18px;background:linear-gradient(180deg,#101811,#0b100c);
border-bottom:1px solid var(--line);min-width:0}
.brand{display:flex;flex-direction:column;line-height:1}
.wordmark{color:var(--amber);font-weight:800;font-size:25px;letter-spacing:.6px}
.tag{color:var(--muted);font-size:10px;letter-spacing:2.6px;margin-top:4px}
.hdr-info{flex:1 1 0;min-width:0;display:flex;flex-direction:column;gap:3px}
.hdr-l1{display:flex;align-items:center;gap:10px;white-space:nowrap;overflow:hidden;font-size:15px}
#hdrTitle{overflow:hidden;text-overflow:ellipsis;min-width:0}
.hdr-l2{color:var(--muted);font-size:12px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.pill{padding:2px 9px;border-radius:999px;font-size:11px;font-weight:700;letter-spacing:1px}
.pill.dust{background:rgba(224,185,122,.16);color:var(--chip);border:1px solid rgba(224,185,122,.45)}
.pill.cartridge{background:rgba(155,225,93,.14);color:var(--seed);border:1px solid rgba(155,225,93,.45)}
.phase{display:inline-block;padding:4px 10px;border-radius:6px;background:#0f1a12;border:1px solid var(--line2);
font:600 11.5px var(--mono);letter-spacing:.4px;color:var(--cyan);white-space:nowrap}
.phase.park{color:var(--fx)}.phase.obst{color:var(--bypass);border-color:rgba(255,140,42,.6)}.phase.done{color:var(--green)}
.clock{text-align:right;font-family:var(--mono);line-height:1.1;min-width:178px}
#clk{font-size:22px;font-weight:700;letter-spacing:.5px}
#tplus{color:var(--amber);font-size:12px;margin-top:2px}
main{display:grid;grid-template-columns:minmax(0,1fr) 540px;min-height:0}
#mapwrap{position:relative;min-width:0;min-height:0;overflow:hidden;background:var(--bg)}
#map{position:absolute;left:0;top:0;width:100%;height:100%;cursor:grab;touch-action:none;display:block}
#map.drag{cursor:grabbing}
#chips{position:absolute;left:10px;top:10px;right:60px;display:flex;flex-wrap:wrap;gap:5px;pointer-events:none}
.chip{pointer-events:auto;display:inline-flex;align-items:center;gap:6px;padding:3px 9px 3px 7px;border-radius:999px;
background:rgba(10,15,11,.82);border:1px solid var(--line2);color:var(--muted);font-size:11.5px;cursor:pointer;
user-select:none;backdrop-filter:blur(2px)}
.chip i{width:10px;height:10px;border-radius:3px;display:inline-block;opacity:.35}
.chip.on{color:var(--ink);border-color:#4b6150}.chip.on i{opacity:1}
.chip:hover{border-color:var(--amber)}
#banner{position:absolute;left:50%;top:54px;transform:translateX(-50%);padding:7px 14px;border-radius:7px;
background:rgba(42,20,6,.92);border:1px solid var(--bypass);color:#ffd2ad;font:600 12.5px var(--sans);
box-shadow:0 4px 18px rgba(0,0,0,.45);display:none;white-space:nowrap;pointer-events:none}
#banner b{color:var(--bypass);letter-spacing:.8px;margin-right:6px}
#banner .ob{color:var(--obst)}
#legend{position:absolute;left:10px;bottom:10px;background:rgba(10,15,11,.86);border:1px solid var(--line2);
border-radius:8px;padding:7px 10px 8px;font-size:11px;color:var(--ink);max-width:290px}
.lg-h{font-size:10px;letter-spacing:1.6px;color:var(--muted);cursor:pointer;user-select:none;display:flex;
justify-content:space-between;gap:12px}
#legend.collapsed #lgBody{display:none}
#lgBody{margin-top:5px;display:grid;grid-template-columns:auto 1fr;gap:3px 8px;align-items:center}
#lgBody svg{display:block}
#mapctl{position:absolute;right:10px;top:10px;display:flex;flex-direction:column;gap:5px}
#mapctl button{width:34px;height:32px;border-radius:7px;background:rgba(10,15,11,.86);border:1px solid var(--line2);
cursor:pointer;font-size:16px;display:flex;align-items:center;justify-content:center;padding:0}
#mapctl button:hover{border-color:var(--amber)}
#mapctl button.on{border-color:var(--cyan);color:var(--cyan)}
#tip{position:absolute;pointer-events:none;display:none;max-width:330px;background:rgba(8,12,9,.95);
border:1px solid var(--line2);border-radius:8px;padding:8px 10px;font-size:12px;box-shadow:0 6px 22px rgba(0,0,0,.5);
z-index:5}
#tip h4{margin:0 0 4px;font-size:12.5px;color:var(--amber)}
#tip .r{display:flex;justify-content:space-between;gap:14px;color:var(--muted)}
#tip .r b{color:var(--ink);font-weight:600;font-family:var(--mono);font-size:11.5px}
#tip .note{color:var(--muted);font-size:11px;margin-top:4px}
#tip .tbar{height:5px;border-radius:3px;background:#26332a;margin:5px 0 2px;overflow:hidden}
#tip .tbar i{display:block;height:100%;background:var(--chip)}
#tip table{border-collapse:collapse;font-size:11px}
#tip td{padding:2px 6px 2px 0;vertical-align:top;color:var(--muted)}
#tip td b{color:var(--ink)}
#side{overflow-y:auto;overflow-x:hidden;border-left:1px solid var(--line);background:var(--panel);padding:9px 11px 10px;
display:flex;flex-direction:column;gap:7px;scrollbar-width:thin;scrollbar-color:#33443a transparent}
.card{background:var(--card);border:1px solid var(--line);border-radius:9px;padding:7px 10px}
.lbl{font-size:10px;letter-spacing:1.3px;color:var(--muted);text-transform:uppercase;white-space:nowrap}
.kpis{display:grid;grid-template-columns:1.62fr 1fr;gap:7px}
.hero{display:flex;gap:12px;align-items:center}
#ring{width:86px;height:86px;flex:none}
.hero-txt{min-width:0}
.big{font:700 26px var(--mono);letter-spacing:-.5px;line-height:1.1;margin-top:2px;white-space:nowrap}
.big .u{font-size:14px;color:var(--muted);font-weight:500;margin-left:3px}
.sub{color:var(--muted);font-size:11.5px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.sub b{color:var(--ink);font-family:var(--mono);font-weight:600}
.sw{display:inline-block;width:9px;height:9px;border-radius:2px;margin:0 4px 0 0;vertical-align:-1px}
.hop{min-width:0}
.hop .val{font:700 20px var(--mono);margin:2px 0 4px;white-space:nowrap}
.hop .val .u{font-size:12px;color:var(--muted);font-weight:500;margin-left:3px}
.hop .sub{font-size:11px;margin-top:3px}
.gauge{position:relative;height:12px;border-radius:4px;background:#0d1610;border:1px solid var(--line2);overflow:hidden}
.gauge i{position:absolute;left:0;top:0;bottom:0;background:linear-gradient(90deg,#2a8f9a,var(--cyan));transition:none}
.gauge .thr{position:absolute;top:-1px;bottom:-1px;width:1px;background:var(--amber);opacity:.9}
.grid6{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:7px}
.sm{padding:6px 9px}
.sm .val{font:700 17px var(--mono);margin-top:1px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.sm .val .u{font-size:11px;color:var(--muted);font-weight:500;margin-left:3px}
.sm .s2{color:var(--muted);font-size:10.5px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin-top:1px}
.bar{height:4px;border-radius:2px;background:#233024;margin-top:5px;overflow:hidden}
.bar i{display:block;height:100%;background:var(--amber);width:0}
.dots{display:flex;gap:5px;margin-top:6px}
.dots i{width:12px;height:12px;border-radius:50%;border:1.5px solid;opacity:.28}
.dots i.on{opacity:1}
.panel{background:var(--card);border:1px solid var(--line);border-radius:9px;padding:7px 10px 8px}
.ph{display:flex;align-items:baseline;justify-content:space-between;font-size:10px;letter-spacing:1.4px;color:var(--muted);
text-transform:uppercase;margin-bottom:4px;gap:8px}
.ph .hint{letter-spacing:.2px;text-transform:none;font-size:10.5px;color:var(--dim)}
.chart{margin-top:3px}
.ch-h{display:flex;align-items:center;gap:8px;font-size:11px;color:var(--muted);white-space:nowrap}
.ch-h .nm{color:var(--ink);font-weight:600}
.ch-h .lg{flex:1;overflow:hidden;text-overflow:ellipsis}
.ch-h b{font-family:var(--mono);color:var(--ink);font-weight:600}
.chart canvas{display:block;width:100%;height:46px;cursor:crosshair}
#chB{height:38px}
#chC{height:58px}
#parcels{display:grid;grid-template-columns:1fr 1fr;gap:3px 14px}
.prow{display:grid;grid-template-columns:14px minmax(0,1fr) 46px;gap:6px;align-items:center}
.prow .pl{font:700 12px var(--mono);color:var(--amber)}
.prow .pn{font-size:10.5px;color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.prow .pb{height:6px;border-radius:3px;background:#233024;overflow:hidden;margin-top:2px}
.prow .pb i{display:block;height:100%;width:0;border-radius:3px}
.prow .pv{text-align:right;font:600 11px var(--mono)}
#swim{display:block;width:100%;height:66px}
.swlg{display:flex;flex-wrap:nowrap;gap:7px;font-size:10px;color:var(--muted);letter-spacing:0;text-transform:none;overflow:hidden}
.swlg .sw{width:8px;height:8px;margin-right:3px}
.duo{display:grid;grid-template-columns:minmax(0,1.05fr) minmax(0,1fr);gap:7px;min-height:200px;flex:1 0 auto}
.ev{display:flex;flex-direction:column;min-height:0}
#events{overflow-y:auto;flex:1 1 0;min-height:110px;scrollbar-width:thin;scrollbar-color:#33443a transparent}
.er{display:grid;grid-template-columns:60px 10px minmax(0,1fr);gap:6px;align-items:center;padding:2px 4px;border-radius:4px;
cursor:pointer;font-size:11px;color:var(--ink)}
.er:hover{background:#1d2a1f}
.er .et{font:11px var(--mono);color:var(--muted);white-space:nowrap;text-align:right}
.er .el{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.er i{width:8px;height:8px;border-radius:2px;display:block;background:#dfe7da}
.er i.k-obstacle_detected{background:var(--obst);transform:rotate(45deg)}
.er i.k-detour_spliced{background:var(--bypass);border-radius:50%}
.er i.k-depart_depot,.er i.k-arrive_depot{background:var(--amber);border-radius:50%}
.er.obst .el{color:#ffc3f7}.er.byp .el{color:#ffcf9f}
.er.fut{opacity:.42}
.er.cur{background:rgba(224,151,47,.16);outline:1px solid rgba(224,151,47,.5)}
.imp .bigc{font:700 24px var(--mono);color:var(--green);line-height:1.1;margin:2px 0 1px}
.imp .bigc .u{font-size:12px;color:var(--muted);font-weight:500;margin-left:3px}
.imp .r{display:flex;justify-content:space-between;gap:8px;font-size:11.5px;color:var(--muted);padding:2px 0;
border-top:1px dashed #223024}
.imp .r b{color:var(--ink);font-family:var(--mono);font-weight:600;white-space:nowrap}
.imp .fin{font-size:10.5px;color:var(--dim);margin-top:4px}
.src{cursor:help;color:var(--amber);letter-spacing:.3px;text-transform:none;font-size:11px;border-bottom:1px dotted var(--amber)}
footer{display:flex;align-items:center;gap:10px;padding:0 14px;border-top:1px solid var(--line);
background:linear-gradient(0deg,#101811,#0b100c)}
.tb{width:38px;height:38px;border-radius:9px;background:#142016;border:1px solid var(--line2);cursor:pointer;display:flex;
align-items:center;justify-content:center;padding:0;flex:none}
.tb:hover{border-color:var(--amber)}
.tb.play{width:46px;height:42px;background:var(--amber);border-color:var(--amber);color:#1a1205}
.tb svg{display:block}
#scrubWrap{position:relative;flex:1;min-width:120px;height:44px}
#scrub{position:absolute;left:0;top:0;width:100%;height:100%;cursor:pointer;touch-action:none}
#scrubTip{position:absolute;top:-30px;transform:translateX(-50%);background:rgba(8,12,9,.95);border:1px solid var(--line2);
border-radius:6px;padding:3px 7px;font:11px var(--mono);white-space:nowrap;display:none;pointer-events:none}
.tr-time{font-family:var(--mono);text-align:right;min-width:112px;line-height:1.15}
#trT{font-size:14px;font-weight:700}
.muted{color:var(--muted);font-size:11px}
.spd{display:flex;align-items:center;gap:6px;color:var(--muted);font-size:11px;white-space:nowrap}
.spd select{background:#142016;border:1px solid var(--line2);border-radius:7px;padding:5px 6px;color:var(--ink);
font:12px var(--mono)}
#effSpd{font:11px var(--mono);color:var(--amber);min-width:68px}
.tg{height:32px;padding:0 10px;border-radius:7px;background:#142016;border:1px solid var(--line2);color:var(--muted);
cursor:pointer;font-size:12px;flex:none}
.tg.on{color:var(--ink);border-color:var(--cyan);box-shadow:inset 0 0 0 1px rgba(56,214,230,.25)}
.tg:hover{border-color:var(--amber)}
#help{position:fixed;inset:0;background:rgba(0,0,0,.55);display:flex;align-items:center;justify-content:center;z-index:20}
#help[hidden]{display:none}
#help .box{background:var(--panel);border:1px solid var(--line2);border-radius:12px;padding:18px 22px;min-width:420px}
#help h3{margin:0 0 10px;color:var(--amber)}
#help table{border-collapse:collapse}
#help td{padding:3px 14px 3px 0}
#help kbd{font:12px var(--mono);background:#1d2a1f;border:1px solid var(--line2);border-radius:4px;padding:1px 6px}
#err{position:fixed;left:12px;right:12px;top:12px;z-index:50;background:#3a0b0b;border:2px solid #ff4d3d;color:#ffd0d0;
padding:10px 12px;font:12px var(--mono);white-space:pre-wrap;display:none;border-radius:8px;max-height:40vh;overflow:auto}
@media (max-width:1600px){main{grid-template-columns:minmax(0,1fr) 450px}.big{font-size:23px}#ring{width:80px;height:80px}
.hdr-l2{display:none}#parcels{grid-template-columns:1fr}}
@media (max-width:1420px){main{grid-template-columns:minmax(0,1fr) 410px}.kpis{grid-template-columns:1fr}
.duo{grid-template-columns:1fr}#events{max-height:200px}.clock{min-width:150px}#clk{font-size:19px}
.wordmark{font-size:21px}header{gap:14px}.spd>span.l{display:none}.tr-time{min-width:96px}
.chip{font-size:10.5px;padding:2px 7px 2px 5px;gap:4px}.chip i{width:8px;height:8px}#chips{gap:4px}
.hdr-l1 .xtra{display:none}}
@media (max-height:900px){#app{grid-template-rows:52px minmax(0,1fr) 56px}#swim{height:58px}}
"""

_ICON_PLAY = '<svg width="18" height="18" viewBox="0 0 18 18"><path d="M4 2.5v13l11-6.5z" fill="currentColor"/></svg>'
_ICON_PREV = ('<svg width="16" height="16" viewBox="0 0 16 16"><path d="M3 2.5h2v11H3zM14 2.5v11L6 8z" '
              'fill="currentColor"/></svg>')
_ICON_NEXT = ('<svg width="16" height="16" viewBox="0 0 16 16"><path d="M11 2.5h2v11h-2zM2 2.5v11L10 8z" '
              'fill="currentColor"/></svg>')

_BODY = f"""
<div id="app">
<header>
  <div class="brand"><div class="wordmark">PRUNEMAXXING</div><div class="tag">MISSION REPLAY</div></div>
  <div class="hdr-info">
    <div class="hdr-l1"><span id="modePill" class="pill"></span><span id="hdrTitle"></span></div>
    <div class="hdr-l2" id="hdrSite"></div>
  </div>
  <span id="phase" class="phase"></span>
  <div class="clock"><div id="clk">Day 1 07:00:00</div><div id="tplus">T+ 0.00 h</div></div>
</header>
<main>
  <section id="mapwrap">
    <canvas id="map"></canvas>
    <div id="chips"></div>
    <div id="banner"></div>
    <div id="legend"><div class="lg-h" id="lgHead"><span>LEGEND</span><span id="lgCaret">hide</span></div>
      <div id="lgBody"></div></div>
    <div id="mapctl">
      <button id="zIn" title="Zoom in (wheel / +)">+</button>
      <button id="zOut" title="Zoom out">&minus;</button>
      <button id="zFit" title="Reset view (double-click / 0)">&#x2922;</button>
      <button id="zFollow" title="Follow the mothership (F)">&#x25CE;</button>
    </div>
    <div id="tip"></div>
  </section>
  <aside id="side">
    <section class="kpis">
      <div class="card hero">
        <svg id="ring" viewBox="0 0 100 100">
          <circle cx="50" cy="50" r="41" fill="none" stroke="#223024" stroke-width="10"/>
          <circle id="ringD" cx="50" cy="50" r="41" fill="none" stroke="#7bdc6a" stroke-width="10"
            transform="rotate(-90 50 50)" stroke-dasharray="0 999"/>
          <circle id="ringM" cx="50" cy="50" r="41" fill="none" stroke="#ffe066" stroke-width="10"
            transform="rotate(-90 50 50)" stroke-dasharray="0 999"/>
          <text id="ringT" x="50" y="49" text-anchor="middle" fill="#e8efe4" font-size="19" font-weight="700"
            font-family="JetBrains Mono, Menlo, Consolas, DejaVu Sans Mono, monospace">0%</text>
          <text x="50" y="64" text-anchor="middle" fill="#9aab94" font-size="9.5" letter-spacing="1">OF TARGET</text>
        </svg>
        <div class="hero-txt">
          <div class="lbl" id="heroLbl">Material applied</div>
          <div class="big"><span id="kApplied">0</span><span class="u">t</span></div>
          <div class="sub">of <b id="kTarget">0</b> t target · <span id="kArea"></span></div>
          <div class="sub" style="margin-top:3px"><i class="sw" style="background:var(--green)"></i>drones <b id="kDrone">0</b> t
            &nbsp;<i class="sw" style="background:var(--fx)"></i><span id="kFxName">effector</span> <b id="kMs">0</b> t</div>
        </div>
      </div>
      <div class="card hop">
        <div class="lbl">Mothership hopper</div>
        <div class="val"><span id="kHop">0</span><span class="u">kg</span></div>
        <div class="gauge"><i id="kHopBar"></i><span class="thr" id="kHopThr"></span></div>
        <div class="sub" title="Amber tick: a tender truck is called below this level">capacity <b id="kHopCap"></b> kg</div>
        <div class="sub" title="Tender-truck top-ups delivered so far"><b id="kTop">0</b> top-ups · <b id="kTopT">0</b> t</div>
      </div>
    </section>
    <section class="grid6">
      <div class="card sm"><div class="lbl">Cells at dose</div><div class="val"><span id="kCells">0</span><span class="u" id="kCellsN"></span></div>
        <div class="bar"><i id="kCellsBar"></i></div></div>
      <div class="card sm"><div class="lbl">Drone sorties</div><div class="val"><span id="kSort">0</span></div>
        <div class="s2" id="kSort2"></div></div>
      <div class="card sm"><div class="lbl">Battery swaps</div><div class="val"><span id="kSwap">0</span></div>
        <div class="s2" id="kSwap2">hot-swap 25 s on deck</div></div>
      <div class="card sm"><div class="lbl">Odometer</div><div class="val"><span id="kOdo">0</span><span class="u">km</span></div>
        <div class="s2" id="kOdo2"></div></div>
      <div class="card sm"><div class="lbl">Drones airborne</div><div class="val"><span id="kAir">0</span><span class="u" id="kAirN"></span></div>
        <div class="dots" id="kAirDots"></div></div>
      <div class="card sm"><div class="lbl">Effector share</div><div class="val"><span id="kFx">0</span><span class="u">%</span></div>
        <div class="s2" id="kFx2"></div></div>
    </section>
    <section class="panel">
      <div class="ph"><span>Timeline</span><span class="hint">click or drag a chart to seek</span></div>
      <div class="chart"><div class="ch-h"><span class="nm">Applied</span><span class="lg"><i class="sw" style="background:var(--green)"></i>drones
        <i class="sw" style="background:var(--fx);margin-left:6px"></i><span id="lgFx">effector</span></span><b id="cA"></b></div><canvas id="chA"></canvas></div>
      <div class="chart"><div class="ch-h"><span class="nm">Cells at dose</span><span class="lg"></span><b id="cB"></b></div><canvas id="chB"></canvas></div>
      <div class="chart"><div class="ch-h"><span class="nm">Hopper level</span><span class="lg"><i class="sw" style="background:var(--cyan)"></i>min-max per pixel
        <i class="sw" style="background:var(--amber);margin-left:6px"></i>tender call level</span><b id="cC"></b></div><canvas id="chC"></canvas></div>
    </section>
    <section class="panel"><div class="ph"><span>Parcel progress</span><span class="hint" id="parcHint"></span></div><div id="parcels"></div></section>
    <section class="panel"><div class="ph"><span>Drone activity</span><span class="swlg" id="swLg"></span></div>
      <canvas id="swim"></canvas></section>
    <section class="duo">
      <div class="panel ev"><div class="ph"><span>Events</span><span class="hint">click to jump · &larr; &rarr;</span></div><div id="events"></div></div>
      <div class="panel imp"><div class="ph"><span>Climate impact</span><span class="src" id="srcBtn">sources &#9432;</span></div><div id="impact"></div></div>
    </section>
  </aside>
</main>
<footer>
  <button id="bPrev" class="tb" title="Previous event (&larr;)">{_ICON_PREV}</button>
  <button id="bPlay" class="tb play" title="Play / pause (Space)">{_ICON_PLAY}</button>
  <button id="bNext" class="tb" title="Next event (&rarr;)">{_ICON_NEXT}</button>
  <div id="scrubWrap"><canvas id="scrub"></canvas><div id="scrubTip"></div></div>
  <div class="tr-time"><div id="trT">T+ 0.0 h</div><div id="trEnd" class="muted"></div></div>
  <label class="spd"><span class="l">Speed</span><select id="speed"></select><span id="effSpd"></span></label>
  <button id="bFollow" class="tg" title="Follow the mothership (F)">Follow</button>
  <button id="bSlow" class="tg on" title="Slow motion around obstacle detections (S)">Slow-mo</button>
  <button id="bLoop" class="tg" title="Loop playback (L)">Loop</button>
  <button id="bHelp" class="tg" title="Keyboard shortcuts (?)">?</button>
</footer>
</div>
<div id="help" hidden><div class="box"><h3>Keyboard &amp; mouse</h3><table>
<tr><td><kbd>Space</kbd></td><td>play / pause</td></tr>
<tr><td><kbd>&larr;</kbd> <kbd>&rarr;</kbd></td><td>previous / next event (stations, obstacles, bypasses)</td></tr>
<tr><td><kbd>Shift</kbd>+<kbd>&larr;</kbd> <kbd>&rarr;</kbd></td><td>step 1 % of the mission</td></tr>
<tr><td><kbd>,</kbd> <kbd>.</kbd></td><td>step 1 minute of mission time</td></tr>
<tr><td><kbd>+</kbd> <kbd>&minus;</kbd></td><td>faster / slower playback</td></tr>
<tr><td><kbd>Home</kbd> <kbd>End</kbd></td><td>jump to start / end</td></tr>
<tr><td><kbd>F</kbd></td><td>follow the mothership</td></tr>
<tr><td><kbd>S</kbd></td><td>slow motion around obstacle detections</td></tr>
<tr><td><kbd>L</kbd></td><td>loop</td></tr>
<tr><td><kbd>0</kbd> / double-click</td><td>reset map view</td></tr>
<tr><td>drag / wheel / pinch</td><td>pan / zoom the map</td></tr>
<tr><td><kbd>?</kbd> <kbd>Esc</kbd></td><td>toggle this help</td></tr>
</table><p class="muted">URL hash: <code>#t=0.42&amp;play=0&amp;follow=1&amp;speed=auto&amp;z=6</code></p></div></div>
<div id="err"></div>
"""

_JS = r"""
(function () {
'use strict';
const $ = (id) => document.getElementById(id);
const errBox = $('err');
window.__replayErrors = [];
function showErr(msg) {
  window.__replayErrors.push(String(msg));
  errBox.style.display = 'block';
  errBox.textContent += String(msg) + '\n';
}
window.addEventListener('error', (e) => showErr('JS error: ' + e.message + ' (line ' + e.lineno + ')'));
window.addEventListener('unhandledrejection', (e) => showErr('Unhandled rejection: ' + ((e.reason && e.reason.message) || e.reason)));

// ================================================================== data
const D = JSON.parse($('replay-data').textContent);
const M = D.meta, COL = D.colors, TL = D.timeline;
const TYPED = { f4: Float32Array, f8: Float64Array, i1: Int8Array, u1: Uint8Array, i2: Int16Array, u2: Uint16Array, i4: Int32Array, u4: Uint32Array };
const B = {};
for (const k of Object.keys(D.bin)) {
  const dt = D.bin[k][0], s = D.bin[k][1];
  const bin = atob(s), n = bin.length, u8 = new Uint8Array(n);
  for (let i = 0; i < n; i++) u8[i] = bin.charCodeAt(i);
  B[k] = new TYPED[dt](u8.buffer);
}
function dm(a) { const o = new Float32Array(a.length); for (let i = 0; i < a.length; i++) o[i] = a[i] * 0.1; return o; }
const T_END = M.t_end, Wm = M.W_m, Hm = M.H_m, CART = M.mode === 'cartridge', ND = M.n_drones;
const FONT = '"Inter","Segoe UI",system-ui,-apple-system,"Noto Sans","DejaVu Sans",Roboto,Arial,sans-serif';
const MONO = '"JetBrains Mono",Menlo,Consolas,"DejaVu Sans Mono",monospace';

// ================================================================== helpers
const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);
function ub(a, v, lo, hi) { lo = lo || 0; hi = hi === undefined ? a.length : hi; while (lo < hi) { const m = (lo + hi) >> 1; if (a[m] <= v) lo = m + 1; else hi = m; } return lo; }
function lb(a, v, lo, hi) { lo = lo || 0; hi = hi === undefined ? a.length : hi; while (lo < hi) { const m = (lo + hi) >> 1; if (a[m] < v) lo = m + 1; else hi = m; } return lo; }
const p2 = (n) => (n < 10 ? '0' : '') + n;
function clock(t) {
  const tot = M.start_h * 3600 + t, day = Math.floor(tot / 86400) + 1, r = tot % 86400;
  return 'Day ' + day + ' ' + p2(Math.floor(r / 3600)) + ':' + p2(Math.floor((r % 3600) / 60)) + ':' + p2(Math.floor(r % 60));
}
const tph = (t, d) => 'T+ ' + (t / 3600).toFixed(d === undefined ? 2 : d) + ' h';
function dur(sec) { sec = Math.max(0, sec); const h = Math.floor(sec / 3600), m = Math.round((sec % 3600) / 60); return h ? h + ' h ' + p2(m) + ' min' : m + ' min'; }
const nf = (v, d) => Number(v).toLocaleString('en-US', { minimumFractionDigits: d || 0, maximumFractionDigits: d || 0 });
function mulberry32(a) { return function () { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
function hash01(a, b) { let h = Math.imul(a ^ 0x9e3779b9, 0x85ebca6b) ^ Math.imul(b + 0x632be59b, 0xc2b2ae35); h ^= h >>> 13; h = Math.imul(h, 0x27d4eb2d); h ^= h >>> 15; return (h >>> 0) / 4294967296; }
const ease = (u) => { u = clamp(u, 0, 1); return u * u * (3 - 2 * u); };
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
function pip(px, py, poly) {
  let inside = false; const n = poly.length >> 1;
  for (let i = 0, j = n - 1; i < n; j = i++) {
    const xi = poly[2 * i], yi = poly[2 * i + 1], xj = poly[2 * j], yj = poly[2 * j + 1];
    if ((yi > py) !== (yj > py) && px < ((xj - xi) * (py - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}
function hexA(hex, a) { const v = parseInt(hex.slice(1), 16); return 'rgba(' + (v >> 16) + ',' + ((v >> 8) & 255) + ',' + (v & 255) + ',' + a + ')'; }
function rrect(c, x, y, w, h, r) { c.beginPath(); c.moveTo(x + r, y); c.arcTo(x + w, y, x + w, y + h, r); c.arcTo(x + w, y + h, x, y + h, r); c.arcTo(x, y + h, x, y, r); c.arcTo(x, y, x + w, y, r); c.closePath(); }
function launchSpeed(R, dz, el) { const c = Math.cos(el), tn = Math.tan(el); return Math.sqrt(9.81 * R * R / Math.max(2 * c * c * (dz + R * tn), 1e-6)); }

// ================================================================== mothership log
const msT = B.ms_t, msX = dm(B.ms_x), msY = dm(B.ms_y), msYaw = B.ms_yaw, msMode = B.ms_mode, msOdo = B.ms_odo, msV = B.ms_v;
const NL = msT.length, STATION = M.label_codes.station, ODET = M.label_codes.obstacle_detour;
const MODE_NAME = {}; for (const k in M.label_codes) MODE_NAME[M.label_codes[k]] = k;
const _pose = { x: 0, y: 0, yaw: 0, odo: 0, v: 0, mode: 0, parked: false, i: 0 };
function msPose(t, o) {
  o = o || _pose;
  let i = ub(msT, t) - 1; i = clamp(i, 0, NL - 2);
  const dt = msT[i + 1] - msT[i]; const a = dt > 1e-9 ? clamp((t - msT[i]) / dt, 0, 1) : 0;
  o.x = msX[i] + a * (msX[i + 1] - msX[i]); o.y = msY[i] + a * (msY[i + 1] - msY[i]);
  o.yaw = msYaw[i] + a * (msYaw[i + 1] - msYaw[i]); o.odo = msOdo[i] + a * (msOdo[i + 1] - msOdo[i]);
  o.v = (msV[i] + a * (msV[i + 1] - msV[i])) * 0.01; o.mode = msMode[i];
  o.parked = msMode[i] === STATION && msMode[i + 1] === STATION; o.i = i;
  return o;
}
// decimated track (>= 0.3 m spacing, keep mode changes)
const trkL = [0];
for (let i = 1, lx = msX[0], ly = msY[0]; i < NL; i++) {
  if (Math.hypot(msX[i] - lx, msY[i] - ly) >= 0.3 || msMode[i] !== msMode[trkL[trkL.length - 1]] || i === NL - 1) { trkL.push(i); lx = msX[i]; ly = msY[i]; }
}
const trkI = Int32Array.from(trkL), trkT = Float64Array.from(trkL, (i) => msT[i]);
const detRuns = [];
for (let k = 0; k < trkI.length; k++) {
  if (msMode[trkI[k]] === ODET) { const a = k; while (k + 1 < trkI.length && msMode[trkI[k + 1]] === ODET) k++; detRuns.push([Math.max(0, a - 1), Math.min(trkI.length - 1, k + 1)]); }
}
const planXY = dm(B.plan_xy);

// ================================================================== patterns
const pOff = B.pat_off, pXY = dm(B.pat_xy), NP = pOff.length - 1;
const pLen = new Float32Array(NP), pBox = new Float32Array(NP * 4);
for (let p = 0; p < NP; p++) {
  let L = 0, x0 = 1e9, y0 = 1e9, x1 = -1e9, y1 = -1e9;
  for (let i = pOff[p]; i < pOff[p + 1]; i++) {
    const x = pXY[2 * i], y = pXY[2 * i + 1];
    if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y;
    if (i > pOff[p]) L += Math.hypot(x - pXY[2 * i - 2], y - pXY[2 * i - 1]);
  }
  pLen[p] = L; pBox[4 * p] = x0; pBox[4 * p + 1] = y0; pBox[4 * p + 2] = x1; pBox[4 * p + 3] = y1;
}
function patAt(p, u, out) {
  const a = pOff[p], b = pOff[p + 1];
  if (b - a < 2 || pLen[p] <= 0) { out[0] = pXY[2 * a]; out[1] = pXY[2 * a + 1]; return out; }
  let s = clamp(u, 0, 1) * pLen[p];
  for (let i = a; i < b - 1; i++) {
    const x0 = pXY[2 * i], y0 = pXY[2 * i + 1], x1 = pXY[2 * i + 2], y1 = pXY[2 * i + 3];
    const sl = Math.hypot(x1 - x0, y1 - y0);
    if (s <= sl || i === b - 2) { const f = sl > 0 ? clamp(s / sl, 0, 1) : 0; out[0] = x0 + f * (x1 - x0); out[1] = y0 + f * (y1 - y0); return out; }
    s -= sl;
  }
  return out;
}

// ================================================================== sorties
const NS = B.s_kg.length;
const sDrone = B.s_drone, sCell = B.s_cell, sPat = B.s_pat, sTl = B.s_tload, sTo = B.s_tlaunch, sA0 = B.s_a0, sA1 = B.s_a1, sTd = B.s_tland, sKg = B.s_kg;
const sDock = dm(B.s_dock);
const drones = [];
for (let d = 0; d < ND; d++) {
  const idx = []; for (let k = 0; k < NS; k++) if (sDrone[k] === d) idx.push(k);
  idx.sort((a, b) => sTl[a] - sTl[b]);
  drones.push({ idx: Int32Array.from(idx), tl: Float64Array.from(idx, (k) => sTl[k]), td: Float64Array.from(idx, (k) => sTd[k]) });
}
const landSorted = Float64Array.from(sTd).sort();
const launchSorted = Float64Array.from(sTo).sort();
const swT = B.sw_t, swD = B.sw_d;
const swapSorted = Float64Array.from(swT).sort();
const swByDrone = []; for (let d = 0; d < ND; d++) { const a = []; for (let i = 0; i < swT.length; i++) if (swD[i] === d) a.push(swT[i]); swByDrone.push(Float64Array.from(a).sort()); }

// ================================================================== effector bursts
const NM = B.m_kg.length;
const mKind = B.m_kind, mSide = B.m_side, mCell = B.m_cell, mPat = B.m_pat, mT0 = B.m_t0, mTf0 = B.m_tf0, mT1 = B.m_t1, mTfree = B.m_tfree, mKg = B.m_kg, mUnits = B.m_units, mYaw = B.m_yaw, mStation = B.m_station;
const mMS = dm(B.m_ms);
const shOrd = Int32Array.from({ length: NM }, (_, i) => i).sort((a, b) => mTf0[a] - mTf0[b]);
const shTf0 = Float64Array.from(shOrd, (k) => mTf0[k]);
const msMid = new Float64Array(NM); for (let k = 0; k < NM; k++) msMid[k] = 0.5 * (mTf0[k] + mT1[k]);
const msDepOrd = Int32Array.from({ length: NM }, (_, i) => i).sort((a, b) => msMid[a] - msMid[b]);
const msDepT = Float64Array.from(msDepOrd, (k) => msMid[k]);
const msDepCum = new Float64Array(NM); { let a = 0; for (let i = 0; i < NM; i++) { a += mKg[msDepOrd[i]]; msDepCum[i] = a; } }
const EL = M.cannon_elev_deg * Math.PI / 180;
function pivotXY(k, out) {
  const side = mSide[k] < 0 ? 0 : mSide[k], pv = M.cannon_pivots[side], c = Math.cos(mYaw[k]), s = Math.sin(mYaw[k]);
  out[0] = mMS[2 * k] + c * pv[0] - s * pv[1]; out[1] = mMS[2 * k + 1] + s * pv[0] + c * pv[1]; return out;
}
const CART_PER_KG = 1 / M.cart_unit_kg;

// ================================================================== deposits, cells, parcels
const dT = B.d_t, dCell = B.d_cell, dKg = B.d_kg, NDEP = dT.length;
const dCum = new Float64Array(NDEP); { let a = 0; for (let i = 0; i < NDEP; i++) { a += dKg[i]; dCum[i] = a; } }
const CELLS = D.cells, NC = CELLS.target.length;
const cellBox = new Float32Array(NC * 4);
for (let c = 0; c < NC; c++) {
  const pl = CELLS.poly[c]; let x0 = 1e9, y0 = 1e9, x1 = -1e9, y1 = -1e9;
  for (let i = 0; i < pl.length; i += 2) { x0 = Math.min(x0, pl[i]); x1 = Math.max(x1, pl[i]); y0 = Math.min(y0, pl[i + 1]); y1 = Math.max(y1, pl[i + 1]); }
  cellBox[4 * c] = x0; cellBox[4 * c + 1] = y0; cellBox[4 * c + 2] = x1; cellBox[4 * c + 3] = y1;
}
const cellOff = new Int32Array(NC + 1);
for (let i = 0; i < NDEP; i++) cellOff[dCell[i] + 1]++;
for (let c = 0; c < NC; c++) cellOff[c + 1] += cellOff[c];
const cellDepT = new Float64Array(NDEP), cellDepCum = new Float64Array(NDEP);
{ const fill = Int32Array.from(cellOff.subarray(0, NC)); for (let i = 0; i < NDEP; i++) { const c = dCell[i], j = fill[c]++; cellDepT[j] = dT[i]; cellDepCum[j] = (j > cellOff[c] ? cellDepCum[j - 1] : 0) + dKg[i]; } }
function cellKg(c, t) { const a = cellOff[c], b = cellOff[c + 1]; const j = ub(cellDepT, t, a, b) - 1; return j >= a ? cellDepCum[j] : 0; }
const cellDoneSorted = Float64Array.from(CELLS.done.filter((v) => v !== null)).sort();
const NPARC = D.parcels.length;
const parcDep = [];
for (let p = 0; p < NPARC; p++) {
  const ts = [], cs = []; let a = 0;
  for (let i = 0; i < NDEP; i++) if (CELLS.parcel[dCell[i]] === p) { a += dKg[i]; ts.push(dT[i]); cs.push(a); }
  parcDep.push({ t: Float64Array.from(ts), c: Float64Array.from(cs), target: D.parcels[p].target_t * 1000 });
}
function parcelFrac(p, t) { const P = parcDep[p]; const j = ub(P.t, t) - 1; return j >= 0 ? P.c[j] / Math.max(P.target, 1e-9) : 0; }

// ================================================================== hopper ledger + tender
const NTOP = B.tp_kg.length;
const topT = Float64Array.from(B.tp_tap), topKgCum = new Float64Array(NTOP);
{ let a = 0; for (let i = 0; i < NTOP; i++) { a += B.tp_kg[i]; topKgCum[i] = a; } }
const nH = NS + NM + NTOP, hT = new Float64Array(nH), hD = new Float64Array(nH), hP = new Uint8Array(nH);
{ let q = 0;
  for (let k = 0; k < NS; k++) { hT[q] = sTl[k]; hD[q] = -sKg[k]; hP[q++] = 1; }
  for (let k = 0; k < NM; k++) { hT[q] = mT0[k]; hD[q] = -mKg[k]; hP[q++] = 1; }
  for (let k = 0; k < NTOP; k++) { hT[q] = topT[k]; hD[q] = B.tp_kg[k]; hP[q++] = 0; } }
const hOrd = Int32Array.from({ length: nH }, (_, i) => i).sort((a, b) => (hT[a] - hT[b]) || (hP[a] - hP[b]));
const hopT = new Float64Array(nH), hopV = new Float64Array(nH);
{ let a = M.hopper_cap; for (let i = 0; i < nH; i++) { const o = hOrd[i]; a += hD[o]; hopT[i] = hT[o]; hopV[i] = a; } }
function hopperAt(t) { const i = ub(hopT, t) - 1; return i < 0 ? M.hopper_cap : hopV[i]; }

// ================================================================== coverage jobs (stamp when a patch is finished)
const NJ = NS + NM, jobT = new Float64Array(NJ), jobK = new Int32Array(NJ);
{ const tmp = new Float64Array(NJ);
  for (let k = 0; k < NS; k++) tmp[k] = sA1[k];
  for (let k = 0; k < NM; k++) tmp[NS + k] = mKind[k] === 0 ? mTfree[k] : mT1[k];
  const o = Int32Array.from({ length: NJ }, (_, i) => i).sort((a, b) => tmp[a] - tmp[b]);
  for (let i = 0; i < NJ; i++) { jobT[i] = tmp[o[i]]; jobK[i] = o[i]; } }

// ================================================================== events
const EVENTS = D.events;
const evT = Float64Array.from(EVENTS, (e) => e.t);
const stopT = Float64Array.from(Array.from(new Set(EVENTS.map((e) => Math.round(e.t * 10) / 10))).sort((a, b) => a - b));
const OBST = D.obstacles;
const STN = D.stations;
const stTa = Float64Array.from(STN, (s) => s.ta), stTd = Float64Array.from(STN, (s) => s.td);
function stationAt(t) { const i = ub(stTa, t) - 1; return i >= 0 && t <= STN[i].td + 1e-6 ? i : -1; }
// trails / effector fades only belong to the station being worked (or none while driving)
function lastDepart(t) { const i = ub(stTd, t) - 1; return i >= 0 ? stTd[i] : -Infinity; }

// ================================================================== state
let t = 0, playing = true, baseSpeed = M.auto_speed, effSpeed = M.auto_speed, autoSpeed = true;
let loop = false, follow = false, slowmo = true, dirty = true, panelDirty = true;
let hover = null, hoverT = null, mouse = null;
const L = { coverage: 1, cells: 0, parcels: 1, weeds: 1, planned: 1, actual: 1, bypass: 1, obstacles: 1, stations: 1, mothership: 1, effector: 1, drones: 1, trails: 1, lidar: 1 };

// ================================================================== map canvas + view
const mapC = $('map'), mctx = mapC.getContext('2d');
const bmImg = $('basemap');
let cw = 800, ch = 600, dpr = 1;
const view = { cx: Wm / 2, cy: Hm / 2, s: 2, fit: 2, ts: null };
const SX = (x) => (x - view.cx) * view.s + cw / 2;
const SY = (y) => ch / 2 - (y - view.cy) * view.s;
const WX = (sx) => (sx - cw / 2) / view.s + view.cx;
const WY = (sy) => view.cy - (sy - ch / 2) / view.s;
function fitView() {
  const b = M.site_box, pad = 10;
  const s = Math.min(cw / (b[2] - b[0] + 2 * pad), ch / (b[3] - b[1] + 2 * pad));
  view.s = s; view.fit = s; view.cx = (b[0] + b[2]) / 2; view.cy = (b[1] + b[3]) / 2; view.ts = null; dirty = true;
}
function zoomAt(sx, sy, f) {
  const wx = WX(sx), wy = WY(sy);
  view.s = clamp(view.s * f, view.fit * 0.5, 60);
  view.cx = wx - (sx - cw / 2) / view.s; view.cy = wy + (sy - ch / 2) / view.s; view.ts = null; dirty = true;
}

// textures
function makeChipTexture() {
  const c = document.createElement('canvas'); c.width = c.height = 40; const x = c.getContext('2d'); const r = mulberry32(7);
  x.fillStyle = '#b48a55'; x.fillRect(0, 0, 40, 40);
  const cols = ['#e0b97a', '#c99a5e', '#8f6539', '#f0d6a2', '#7a5432', '#d4a86c'];
  for (let i = 0; i < 360; i++) { x.fillStyle = cols[Math.floor(r() * cols.length)]; const w = 0.8 + r() * 1.8, h = 0.5 + r() * 0.8; x.save(); x.translate(r() * 40, r() * 40); x.rotate(r() * Math.PI); x.fillRect(-w / 2, -h / 2, w, h); x.restore(); }
  return c;
}
function makeHatch(color) {
  const c = document.createElement('canvas'); c.width = c.height = 9; const x = c.getContext('2d');
  x.strokeStyle = color; x.lineWidth = 1.6; x.beginPath(); x.moveTo(-1, 10); x.lineTo(10, -1); x.moveTo(-1, 1); x.lineTo(1, -1); x.moveTo(8, 10); x.lineTo(10, 8); x.stroke();
  return c;
}
const chipTex = makeChipTexture();
const weedPat = mctx.createPattern(makeHatch(COL.weed), 'repeat');

// coverage raster (world space, 4 px/m, north up)
const CR = 4, covC = document.createElement('canvas');
covC.width = Math.ceil(Wm * CR); covC.height = Math.ceil(Hm * CR);
const cctx = covC.getContext('2d');
const chipPat = cctx.createPattern(chipTex, 'repeat');
let covCursor = 0;
function covReset() { cctx.setTransform(1, 0, 0, 1, 0, 0); cctx.globalAlpha = 1; cctx.clearRect(0, 0, covC.width, covC.height); covCursor = 0; }
const _q = [0, 0], _q2 = [0, 0];
function dustAlpha(kg, area) { return 1 - Math.pow(0.12, clamp(kg / Math.max(area * M.dose, 1e-6), 0, 3)); }
function stampDust(g, p, kg, u) {
  const m = 0.55, x0 = pBox[4 * p] - m, y0 = pBox[4 * p + 1] - m, x1 = pBox[4 * p + 2] + m, y1 = pBox[4 * p + 3] + m;
  g.globalAlpha = dustAlpha(kg, (x1 - x0) * (y1 - y0));
  const xe = x0 + (x1 - x0) * (u === undefined ? 1 : u);
  g.fillRect(x0 * CR, (Hm - y1) * CR, (xe - x0) * CR, (y1 - y0) * CR);
}
function cartDots(k, isShot, uMax, cb) {
  if (isShot) {
    const n = mUnits[k], p = mPat[k], R = M.cannon_sigma, Rm = M.cannon_sigma_per_m; pivotXY(k, _q2);
    for (let i = 0; i < n; i++) {
      if ((i + 0.5) / n > uMax) break;
      patAt(p, (i + 0.5) / n, _q); const rr = Math.hypot(_q[0] - _q2[0], _q[1] - _q2[1]), sg = R + Rm * rr;
      const a = hash01(k * 31 + i, 7) * 6.283, r = Math.sqrt(-2 * Math.log(1e-6 + hash01(k, i * 17 + 3))) * sg;
      cb(_q[0] + r * Math.cos(a), _q[1] + r * Math.sin(a), i);
    }
  } else {
    const n = Math.max(1, Math.round(sKg[k] * CART_PER_KG)), p = sPat[k];
    for (let i = 0; i < n; i++) {
      if ((i + 0.5) / n > uMax) break;
      patAt(p, (i + 0.5) / n, _q); cb(_q[0] + (hash01(k, i) - 0.5) * 0.12, _q[1] + (hash01(i, k) - 0.5) * 0.25, i);
    }
  }
}
function stampJob(k) {
  const g = cctx;
  if (!CART) {
    g.fillStyle = chipPat;
    if (k < NS) stampDust(g, sPat[k], sKg[k]); else stampDust(g, mPat[k - NS], mKg[k - NS]);
    return;
  }
  const isShot = k >= NS, kk = isShot ? k - NS : k, p = isShot ? mPat[kk] : sPat[kk];
  // cartridge row: faint chip line along the bed ridge, seed-green pucks on it
  g.globalAlpha = 0.28; g.strokeStyle = COL.chip; g.lineWidth = 0.32 * CR; g.beginPath();
  for (let i = pOff[p]; i < pOff[p + 1]; i++) { const X = pXY[2 * i] * CR, Y = (Hm - pXY[2 * i + 1]) * CR; if (i === pOff[p]) g.moveTo(X, Y); else g.lineTo(X, Y); }
  g.stroke();
  g.globalAlpha = 0.95; g.fillStyle = '#9be15d';
  cartDots(kk, isShot, 1, (x, y) => g.fillRect((x - 0.17) * CR, (Hm - y - 0.17) * CR, 0.34 * CR, 0.34 * CR));
}
function covTo(tt) {
  const k = ub(jobT, tt);
  if (k < covCursor) covReset();
  for (let j = covCursor; j < k; j++) stampJob(jobK[j]);
  if (k > covCursor) covCursor = k;
  cctx.globalAlpha = 1;
}

// ================================================================== drone kinematics
const DRONE_COL = COL.drones;
function sortiePos(k, tt, out) {
  const dx = sDock[2 * k], dy = sDock[2 * k + 1], p = sPat[k];
  if (tt <= sTo[k]) { out[0] = dx; out[1] = dy; return 'loading'; }
  if (tt < sA0[k]) { const u = ease((tt - sTo[k]) / Math.max(sA0[k] - sTo[k], 1e-6)); patAt(p, 0, out); out[0] = dx + u * (out[0] - dx); out[1] = dy + u * (out[1] - dy); return 'outbound'; }
  if (tt < sA1[k]) { patAt(p, (tt - sA0[k]) / Math.max(sA1[k] - sA0[k], 1e-6), out); return 'applying'; }
  if (tt < sTd[k]) { const u = ease((tt - sA1[k]) / Math.max(sTd[k] - sA1[k], 1e-6)); patAt(p, 1, out); out[0] += u * (dx - out[0]); out[1] += u * (dy - out[1]); return 'return'; }
  out[0] = dx; out[1] = dy; return 'idle';
}
function droneAt(d, tt, ms) {
  const dr = drones[d], j = ub(dr.tl, tt) - 1, o = { x: 0, y: 0, st: 'idle', k: -1, air: false, u: 0 };
  if (j >= 0) {
    const k = dr.idx[j];
    if (tt < sTd[k]) {
      const q = [0, 0]; o.st = sortiePos(k, tt, q); o.x = q[0]; o.y = q[1]; o.k = k; o.air = o.st !== 'loading';
      if (o.st === 'applying') o.u = (tt - sA0[k]) / Math.max(sA1[k] - sA0[k], 1e-6);
      return o;
    }
  }
  const off = M.dock_offsets[d], c = Math.cos(ms.yaw), s = Math.sin(ms.yaw);
  o.x = ms.x + c * off[0] - s * off[1]; o.y = ms.y + s * off[0] + c * off[1];
  const sw = swByDrone[d], js = ub(sw, tt) - 1;
  if (js >= 0 && tt < sw[js] + M.swap_s) o.st = 'swap';
  return o;
}

// ================================================================== stats at the playhead
function stats(tt) {
  const kD = ub(dT, tt), applied = kD ? dCum[kD - 1] : 0;
  const kM = ub(msDepT, tt), msKg = kM ? msDepCum[kM - 1] : 0;
  const ms = msPose(tt, {});
  const nTop = ub(topT, tt);
  let air = 0; const airs = [];
  for (let d = 0; d < ND; d++) { const a = droneAt(d, tt, ms); airs.push(a); if (a.air) air++; }
  return {
    applied, msKg, droneKg: Math.max(applied - msKg, 0), cells: ub(cellDoneSorted, tt), sorties: ub(landSorted, tt),
    inflight: ub(launchSorted, tt) - ub(landSorted, tt), swaps: ub(swapSorted, tt), hopper: hopperAt(tt), ms, air, airs,
    tops: nTop, topKg: nTop ? topKgCum[nTop - 1] : 0, bursts: ub(shTf0, tt),
  };
}
const TARGET_KG = TL.target_t * 1000, FINAL_KG = NDEP ? dCum[NDEP - 1] : 1;

// ================================================================== map drawing
function poly(g, pts, close) { g.beginPath(); for (let i = 0; i < pts.length; i += 2) { const X = SX(pts[i]), Y = SY(pts[i + 1]); if (i) g.lineTo(X, Y); else g.moveTo(X, Y); } if (close) g.closePath(); }
function pill(g, text, x, y, o) {
  o = o || {}; g.font = (o.font || '600 11px ') + FONT; const w = g.measureText(text).width + 12, h = o.h || 18;
  let bx = o.align === 'left' ? x : o.align === 'right' ? x - w : x - w / 2; const by = o.valign === 'top' ? y : y - h;
  g.fillStyle = o.bg || 'rgba(8,12,9,0.8)'; rrect(g, bx, by, w, h, 4); g.fill();
  if (o.border) { g.strokeStyle = o.border; g.lineWidth = 1; g.stroke(); }
  g.fillStyle = o.color || COL.ink; g.textBaseline = 'middle'; g.textAlign = 'left'; g.fillText(text, bx + 6, by + h / 2 + 0.5);
  return [bx, by, w, h];
}
function drawParcels(g) {
  g.lineWidth = 1.2; g.strokeStyle = 'rgba(232,239,228,0.42)'; g.setLineDash([]);
  for (const p of D.parcels) { poly(g, p.poly, true); g.stroke(); }
  labelRects.length = 0;
  for (const p of D.parcels) {
    let x0 = 1e9, y1 = -1e9; for (let i = 0; i < p.poly.length; i += 2) { x0 = Math.min(x0, p.poly[i]); y1 = Math.max(y1, p.poly[i + 1]); }
    const txt = view.s > 1.6 ? p.letter + ' · ' + p.crop_label : p.letter;
    labelRects.push(pill(g, txt, SX(x0) + 6, SY(y1) + 6, { align: 'left', valign: 'top', font: '600 11px ', bg: 'rgba(8,12,9,0.72)' }));
  }
}
const labelRects = [];
function hitsLabel(r) { return labelRects.some((q) => r[0] < q[0] + q[2] + 4 && q[0] < r[0] + r[2] + 4 && r[1] < q[1] + q[3] + 4 && q[1] < r[1] + r[3] + 4); }
function drawCells(g, tt) {
  g.lineWidth = 1;
  for (let c = 0; c < NC; c++) {
    if (CELLS.blocked[c]) continue;
    const f = clamp(cellKg(c, tt) / Math.max(CELLS.target[c], 1e-9), 0, 1);
    poly(g, CELLS.poly[c], true);
    if (f >= 0.999) { g.fillStyle = 'rgba(123,220,106,0.30)'; g.fill(); }
    else if (f > 0) { g.fillStyle = hexA('#e0972f', (0.12 + 0.35 * f).toFixed(3)); g.fill(); }
    g.strokeStyle = 'rgba(232,239,228,0.16)'; g.stroke();
  }
}
function drawWeeds(g) {
  g.setLineDash([3, 3]); g.lineWidth = 1; g.strokeStyle = hexA(COL.weed, 0.55);
  for (const w of D.inflated_weeds) { poly(g, w, true); g.stroke(); }
  g.setLineDash([]);
  for (const w of D.weeds) {
    poly(g, w.poly, true); g.fillStyle = 'rgba(255,77,61,0.20)'; g.fill(); g.fillStyle = weedPat; g.fill();
    g.lineWidth = 1.6; g.strokeStyle = COL.weed; g.stroke();
  }
}
function strokeLine(g, arr, i0, i1) { g.beginPath(); for (let i = i0; i < i1; i++) { const X = SX(arr[2 * i]), Y = SY(arr[2 * i + 1]); if (i === i0) g.moveTo(X, Y); else g.lineTo(X, Y); } }
function drawPlanned(g) {
  g.setLineDash([7, 5]); g.lineWidth = 1.5; g.strokeStyle = hexA(COL.planned, 0.9);
  strokeLine(g, planXY, 0, planXY.length >> 1); g.stroke(); g.setLineDash([]);
}
function drawTrack(g, ms) {
  const kEnd = ub(trkT, t);
  if (kEnd < 1) return;
  const path = () => { g.beginPath(); for (let k = 0; k < kEnd; k++) { const i = trkI[k]; const X = SX(msX[i]), Y = SY(msY[i]); if (k) g.lineTo(X, Y); else g.moveTo(X, Y); } g.lineTo(SX(ms.x), SY(ms.y)); };
  path(); g.lineJoin = 'round'; g.lineCap = 'round';
  g.lineWidth = 6; g.strokeStyle = hexA(COL.actual, 0.18); g.stroke();
  g.lineWidth = 2.2; g.strokeStyle = COL.actual; g.stroke();
  for (const r of detRuns) {
    if (trkT[r[0]] > t) continue;
    g.beginPath(); const e = Math.min(r[1], kEnd - 1);
    for (let k = r[0]; k <= e; k++) { const i = trkI[k]; const X = SX(msX[i]), Y = SY(msY[i]); if (k === r[0]) g.moveTo(X, Y); else g.lineTo(X, Y); }
    if (r[1] >= kEnd) g.lineTo(SX(ms.x), SY(ms.y));
    g.lineWidth = 2.6; g.strokeStyle = COL.bypass; g.stroke();
  }
}
function drawBypasses(g) {
  for (const o of OBST) {
    if (!o.detour || o.t_det === null || t < o.detour.t) continue;
    g.setLineDash([5, 4]); g.lineWidth = 2.4; g.strokeStyle = COL.bypass;
    strokeLine(g, o.detour.pts, 0, o.detour.pts.length >> 1); g.stroke(); g.setLineDash([]);
    // the abandoned stretch of the global plan, crossed out
    g.lineWidth = 3; g.strokeStyle = 'rgba(255,77,61,0.55)'; g.setLineDash([2, 5]);
    strokeLine(g, o.detour.replaced, 0, o.detour.replaced.length >> 1); g.stroke(); g.setLineDash([]);
  }
}
function drawStations(g, cur) {
  const r = clamp(view.s * 1.2, 3, 6);
  for (let i = 0; i < STN.length; i++) {
    const s = STN[i], X = SX(s.x), Y = SY(s.y);
    if (t >= s.ta) { g.fillStyle = '#f3f6f0'; g.fillRect(X - r / 2, Y - r / 2, r, r); g.strokeStyle = '#0b100c'; g.lineWidth = 1; g.strokeRect(X - r / 2, Y - r / 2, r, r); }
    else { g.strokeStyle = 'rgba(243,246,240,0.55)'; g.lineWidth = 1; g.strokeRect(X - r / 2, Y - r / 2, r, r); }
    if (view.s > 5) { g.font = '600 10px ' + FONT; g.fillStyle = 'rgba(243,246,240,0.85)'; g.textAlign = 'left'; g.textBaseline = 'middle'; g.fillText('S' + (s.sid + 1), X + r, Y - r); }
  }
  if (cur >= 0) { const s = STN[cur]; g.strokeStyle = COL.amber; g.lineWidth = 2; g.strokeRect(SX(s.x) - r, SY(s.y) - r, 2 * r, 2 * r); }
}
function drawDepot(g) {
  const X = SX(M.depot[0]), Y = SY(M.depot[1]), R = 8;
  g.beginPath();
  for (let i = 0; i < 10; i++) { const a = -Math.PI / 2 + i * Math.PI / 5, r = i % 2 ? R * 0.45 : R; g.lineTo(X + r * Math.cos(a), Y + r * Math.sin(a)); }
  g.closePath(); g.fillStyle = COL.amber; g.fill(); g.strokeStyle = '#000'; g.lineWidth = 1; g.stroke();
  pill(g, 'Depot · chip stockpiles', X + 12, Y + 9, { align: 'left', font: '600 10.5px ', h: 17 });
}
function recentW() { return playing ? Math.max(240, effSpeed * 4) : 600; }
function drawObstacles(g, now) {
  for (const o of OBST) {
    if (o.t_det === null || t < o.t_det) continue;
    const age = t - o.t_det, recent = age < recentW();
    poly(g, o.poly, true); g.fillStyle = hexA(COL.obst, o.kind === 'waterlogged' ? 0.55 : 0.9); g.fill();
    g.lineWidth = 1.4; g.strokeStyle = '#fff'; g.stroke();
    const X = SX(o.c[0]), Y = SY(o.c[1]);
    if (recent) {
      for (let r = 0; r < 2; r++) {
        const ph = ((now / 1000 + r * 0.6) % 1.2) / 1.2;
        g.beginPath(); g.arc(X, Y, 8 + ph * 34, 0, 6.283); g.strokeStyle = hexA(COL.obst, (1 - ph) * 0.85); g.lineWidth = 2; g.stroke();
      }
      if (o.det_xy) {
        g.setLineDash([3, 3]); g.strokeStyle = hexA(COL.bypass, 0.9); g.lineWidth = 1.2; g.beginPath(); g.moveTo(SX(o.det_xy[0]), SY(o.det_xy[1])); g.lineTo(X, Y); g.stroke(); g.setLineDash([]);
        g.beginPath(); g.arc(SX(o.det_xy[0]), SY(o.det_xy[1]), 4, 0, 6.283); g.fillStyle = COL.bypass; g.fill(); g.strokeStyle = '#fff'; g.lineWidth = 1; g.stroke();
      }
    }
    let top = -1e9, bot = 1e9; for (let i = 1; i < o.poly.length; i += 2) { top = Math.max(top, o.poly[i]); bot = Math.min(bot, o.poly[i]); }
    const txt = o.short + ' · not in satellite map', po = { font: '600 10.5px ', h: 17, border: COL.obst, color: '#ffd6fb' };
    g.font = po.font + FONT; const w = g.measureText(txt).width + 12;
    let yb = Math.min(SY(top), Y - 8) - 6;
    if (hitsLabel([X - w / 2, yb - 17, w, 17])) yb = Math.max(SY(bot), Y + 8) + 6 + 17;
    pill(g, txt, X, yb, po);
  }
}
function drawReach(g, ms) {
  if (!ms.parked) return;
  const X = SX(ms.x), Y = SY(ms.y);
  g.beginPath(); g.arc(X, Y, M.reach_m * view.s, 0, 6.283); g.fillStyle = 'rgba(255,224,102,0.07)'; g.fill();
  g.setLineDash([4, 4]); g.lineWidth = 1.3; g.strokeStyle = hexA(COL.fx, 0.85); g.stroke(); g.setLineDash([]);
  g.beginPath(); g.arc(X, Y, M.reach_min_m * view.s, 0, 6.283); g.strokeStyle = hexA(COL.fx, 0.35); g.lineWidth = 1; g.stroke();
  if (view.s > 4.5) { g.font = '600 10px ' + FONT; g.fillStyle = hexA(COL.fx, 0.95); g.textAlign = 'center'; g.textBaseline = 'bottom'; g.fillText('40 ft reach · ' + M.reach_m.toFixed(2) + ' m', X, Y - M.reach_m * view.s - 2); }
}
function drawLidar(g, ms, now) {
  const sx = ms.x + 1.6 * Math.cos(ms.yaw), sy = ms.y + 1.6 * Math.sin(ms.yaw), X = SX(sx), Y = SY(sy), R = M.lidar_m * view.s;
  const a = (now / 1000) * 2.4;
  const grd = g.createRadialGradient(X, Y, 0, X, Y, R); grd.addColorStop(0, 'rgba(56,214,230,0.0)'); grd.addColorStop(1, 'rgba(56,214,230,0.16)');
  g.beginPath(); g.moveTo(X, Y); g.arc(X, Y, R, a, a + 0.55); g.closePath(); g.fillStyle = grd; g.fill();
  g.beginPath(); g.arc(X, Y, R, 0, 6.283); g.setLineDash([2, 5]); g.lineWidth = 1.1; g.strokeStyle = hexA(COL.actual, 0.55); g.stroke(); g.setLineDash([]);
  if (view.s > 4.5) { g.font = '10px ' + FONT; g.fillStyle = hexA(COL.actual, 0.85); g.textAlign = 'left'; g.textBaseline = 'middle'; g.fillText('lidar ' + M.lidar_m + ' m', X + R * 0.72 + 4, Y + R * 0.72); }
}
function drawMothership(g, ms) {
  const X = SX(ms.x), Y = SY(ms.y), k = Math.max(view.s, 18 / M.ms_len);
  const Lh = M.ms_len * k / 2, Wh = M.ms_wid * k / 2, c = Math.cos(ms.yaw), s = Math.sin(ms.yaw);
  const pts = [[Lh, 0], [Lh * 0.62, Wh], [-Lh, Wh], [-Lh, -Wh], [Lh * 0.62, -Wh]];
  g.beginPath(); g.arc(X, Y, Lh + 7, 0, 6.283); g.fillStyle = 'rgba(56,214,230,0.16)'; g.fill();
  g.beginPath();
  pts.forEach((p, i) => { const px = X + c * p[0] - s * p[1], py = Y - (s * p[0] + c * p[1]); if (i) g.lineTo(px, py); else g.moveTo(px, py); });
  g.closePath(); g.fillStyle = COL.actual; g.fill(); g.lineWidth = 1.6; g.strokeStyle = '#06222a'; g.stroke();
  g.beginPath(); g.arc(X - c * Lh * 0.15, Y + s * Lh * 0.15, Wh * 0.55, 0, 6.283); g.fillStyle = '#0b3b44'; g.fill();
}
function drawPartial(g, dst) {
  if (!CART) g.fillStyle = chipPat;
  for (let d = 0; d < ND; d++) {
    const a = dst[d]; if (a.st !== 'applying' || a.k < 0) continue;
    if (!CART) {
      const p = sPat[a.k], m = 0.55, x0 = pBox[4 * p] - m, y0 = pBox[4 * p + 1] - m, x1 = pBox[4 * p + 2] + m, y1 = pBox[4 * p + 3] + m;
      g.globalAlpha = dustAlpha(sKg[a.k], (x1 - x0) * (y1 - y0)) * 0.9;
      g.save(); g.translate(SX(0), SY(Hm)); g.scale(view.s / CR, view.s / CR);
      g.fillRect(x0 * CR, (Hm - y1) * CR, (x1 - x0) * a.u * CR, (y1 - y0) * CR); g.restore();
    } else {
      g.globalAlpha = 0.95; g.fillStyle = '#9be15d'; const r = Math.max(1.2, 0.17 * view.s);
      cartDots(a.k, false, a.u, (x, y) => g.fillRect(SX(x) - r, SY(y) - r, 2 * r, 2 * r));
    }
  }
  g.globalAlpha = 1;
}
function drawEffector(g, now) {
  if (!NM) return;
  const W = Math.max(4, effSpeed * 0.3), rnd = mulberry32(Math.floor(now / 50)), cut = lastDepart(t);
  let j = ub(shTf0, t) - 1;
  for (; j >= 0; j--) {
    const k = shOrd[j];
    if (shTf0[j] < t - W - 120 || shTf0[j] < cut) break;
    const isB = mKind[k] === 0, tEnd = isB ? mTfree[k] : mT1[k];
    if (t > tEnd + W) continue;
    const p = mPat[k];
    if (isB) {
      const ox = mMS[2 * k], oy = mMS[2 * k + 1];
      if (t <= tEnd) {
        const u = clamp((t - mTf0[k]) / Math.max(mTfree[k] - mTf0[k], 1e-6), 0, 1); patAt(p, u, _q);
        const X0 = SX(ox), Y0 = SY(oy), X1 = SX(_q[0]), Y1 = SY(_q[1]);
        const grd = g.createLinearGradient(X0, Y0, X1, Y1); grd.addColorStop(0, hexA(COL.fx, 0.95)); grd.addColorStop(1, hexA(COL.chip, 0.5));
        g.strokeStyle = grd; g.lineWidth = 2.6; g.beginPath(); g.moveTo(X0, Y0); g.lineTo(X1, Y1); g.stroke();
        const nx = -(Y1 - Y0), ny = X1 - X0, nl = Math.hypot(nx, ny) || 1;
        for (let i = 0; i < 26; i++) {
          const f = (i / 26 + now / 900) % 1, sp = (rnd() - 0.5) * f * 0.25 * nl;
          g.fillStyle = i % 3 ? hexA(COL.chip, 0.85) : hexA(COL.fx, 0.9);
          g.fillRect(X0 + f * (X1 - X0) + sp * nx / nl - 1, Y0 + f * (Y1 - Y0) + sp * ny / nl - 1, 2, 2);
        }
        const rr = Math.max(4, 0.9 * view.s);
        for (let i = 0; i < 16; i++) { const a = rnd() * 6.283, r = rnd() * rr; g.fillStyle = hexA(COL.chip, 0.75); g.fillRect(X1 + r * Math.cos(a) - 1, Y1 + r * Math.sin(a) - 1, 2, 2); }
      } else {
        const fa = 1 - (t - tEnd) / W; patAt(p, 0.5, _q);
        g.strokeStyle = hexA(COL.fx, 0.5 * fa); g.lineWidth = 1.4; g.beginPath(); g.moveTo(SX(ox), SY(oy)); g.lineTo(SX(_q[0]), SY(_q[1])); g.stroke();
      }
    } else {
      pivotXY(k, _q2); const PX = SX(_q2[0]), PY = SY(_q2[1]), n = mUnits[k], gap = M.shot_gap;
      if (t <= tEnd) {
        const iNow = clamp(Math.floor((t - mTf0[k]) / gap), 0, n - 1);
        patAt(p, (iNow + 0.5) / n, _q);
        g.strokeStyle = hexA(COL.fx, 0.75); g.lineWidth = 1.3; g.setLineDash([3, 3]); g.beginPath(); g.moveTo(PX, PY); g.lineTo(SX(_q[0]), SY(_q[1])); g.stroke(); g.setLineDash([]);
        cartDots(k, true, 1, (x, y, i) => {
          const tf = mTf0[k] + i * gap; if (t < tf) return;
          const R = Math.hypot(x - _q2[0], y - _q2[1]), v = launchSpeed(Math.max(R, 0.3), M.cannon_h + 0.5, EL), T = Math.max(R, 0.3) / (v * Math.cos(EL));
          const u = (t - tf) / T;
          if (u < 1) {
            const X = PX + u * (SX(x) - PX), Y = PY + u * (SY(y) - PY), hgt = 4 * u * (1 - u);
            g.strokeStyle = hexA(COL.fx, 0.8); g.lineWidth = 1.5; g.beginPath(); g.moveTo(PX + Math.max(0, u - 0.2) * (SX(x) - PX), PY + Math.max(0, u - 0.2) * (SY(y) - PY)); g.lineTo(X, Y); g.stroke();
            g.beginPath(); g.arc(X, Y, 2.2 + hgt * 1.6, 0, 6.283); g.fillStyle = '#fff6c8'; g.fill();
          } else { g.fillStyle = '#9be15d'; g.fillRect(SX(x) - 1.5, SY(y) - 1.5, 3, 3); }
        });
        if (t - (mTf0[k] + iNow * gap) < 0.18) { g.beginPath(); g.arc(PX, PY, 7, 0, 6.283); g.fillStyle = 'rgba(255,246,200,0.85)'; g.fill(); }
      } else {
        const fa = 1 - (t - tEnd) / W; patAt(p, 0.5, _q);
        g.strokeStyle = hexA(COL.fx, 0.45 * fa); g.lineWidth = 1.2; g.beginPath(); g.moveTo(PX, PY); g.lineTo(SX(_q[0]), SY(_q[1])); g.stroke();
      }
    }
  }
}
const _tp = [0, 0];
function drawTrails(g) {
  const W = clamp(effSpeed * 0.5, 14, 2400), cut = Math.max(t - W, lastDepart(t));
  g.lineCap = 'round';
  for (let d = 0; d < ND; d++) {
    const dr = drones[d]; const j1 = ub(dr.tl, t) - 1; if (j1 < 0) continue;
    const j0 = Math.max(lb(dr.td, cut), j1 - 24);
    g.strokeStyle = DRONE_COL[d % DRONE_COL.length]; g.lineWidth = 1.5;
    for (let j = j0; j <= j1; j++) {
      const k = dr.idx[j], ta = Math.max(sTo[k], cut), tb = Math.min(sTd[k], t);
      if (tb <= ta) continue;
      // key times: launch, apply start, pattern vertices, apply end, land
      const p = sPat[k], keys = [sTo[k], sA0[k]];
      let acc = 0; for (let i = pOff[p] + 1; i < pOff[p + 1]; i++) { acc += Math.hypot(pXY[2 * i] - pXY[2 * i - 2], pXY[2 * i + 1] - pXY[2 * i - 1]); keys.push(sA0[k] + (pLen[p] > 0 ? acc / pLen[p] : 1) * (sA1[k] - sA0[k])); }
      keys.push(sTd[k]);
      const ts = [ta]; for (const kt of keys) if (kt > ta && kt < tb) ts.push(kt); ts.push(tb);
      sortiePos(k, ts[0], _tp); let X0 = SX(_tp[0]), Y0 = SY(_tp[1]);
      for (let i = 1; i < ts.length; i++) {
        sortiePos(k, ts[i] - 1e-4, _tp); const X1 = SX(_tp[0]), Y1 = SY(_tp[1]);
        g.globalAlpha = 0.08 + 0.8 * Math.pow(clamp(1 - (t - ts[i]) / W, 0, 1), 1.4);
        g.beginPath(); g.moveTo(X0, Y0); g.lineTo(X1, Y1); g.stroke(); X0 = X1; Y0 = Y1;
      }
    }
  }
  g.globalAlpha = 1;
}
function drawDrones(g, dst) {
  for (let pass = 0; pass < 2; pass++) {
    for (let d = 0; d < ND; d++) {
      const a = dst[d]; if ((pass === 0) === a.air) continue;
      const X = SX(a.x), Y = SY(a.y), col = DRONE_COL[d % DRONE_COL.length];
      if (!a.air) { const r = Math.max(2.4, 0.55 * view.s); g.beginPath(); g.arc(X, Y, r, 0, 6.283); g.fillStyle = col; g.fill(); g.lineWidth = 1; g.strokeStyle = a.st === 'swap' ? COL.bypass : '#0b100c'; g.stroke(); if (a.st === 'swap') { g.beginPath(); g.arc(X, Y, r + 3, 0, 6.283); g.strokeStyle = COL.bypass; g.lineWidth = 1.5; g.stroke(); } continue; }
      const r = Math.max(4, 1.0 * view.s);
      if (a.st === 'applying') { g.beginPath(); g.arc(X, Y, r + 4, 0, 6.283); g.fillStyle = CART ? 'rgba(155,225,93,0.28)' : 'rgba(224,185,122,0.32)'; g.fill(); }
      if (view.s > 7) {
        g.strokeStyle = col; g.lineWidth = 1.6;
        for (let q = 0; q < 4; q++) { const ang = Math.PI / 4 + q * Math.PI / 2; g.beginPath(); g.moveTo(X, Y); g.lineTo(X + r * 1.1 * Math.cos(ang), Y + r * 1.1 * Math.sin(ang)); g.stroke(); g.beginPath(); g.arc(X + r * 1.25 * Math.cos(ang), Y + r * 1.25 * Math.sin(ang), r * 0.42, 0, 6.283); g.stroke(); }
      }
      g.beginPath(); g.arc(X, Y, r * (view.s > 7 ? 0.55 : 1), 0, 6.283); g.fillStyle = col; g.fill(); g.lineWidth = 1.4; g.strokeStyle = '#0b100c'; g.stroke();
    }
  }
}
function drawScale(g) {
  const nice = [5, 10, 20, 25, 50, 100, 200]; let Lm = 50;
  for (const v of nice) { if (v * view.s >= 70) { Lm = v; break; } }
  const w = Lm * view.s, x = cw - w - 22, y = ch - 22;
  g.fillStyle = 'rgba(8,12,9,0.72)'; rrect(g, x - 10, y - 34, w + 20, 46, 6); g.fill();
  g.fillStyle = '#fff'; g.fillRect(x, y, w, 5); g.fillStyle = '#111'; g.fillRect(x + w / 2, y + 1, w / 2 - 1, 3);
  g.font = '600 11px ' + FONT; g.fillStyle = '#fff'; g.textAlign = 'center'; g.textBaseline = 'bottom'; g.fillText(Lm + ' m', x + w / 2, y - 3);
  g.beginPath(); g.moveTo(x + w - 6, y - 26); g.lineTo(x + w - 1, y - 16); g.lineTo(x + w - 11, y - 16); g.closePath(); g.fill();
  g.textBaseline = 'top'; g.font = '700 9px ' + FONT; g.textAlign = 'center'; g.fillText('N', x + w - 6, y - 15);
}
function drawHover(g) {
  if (!hover) return;
  g.lineWidth = 2; g.strokeStyle = COL.amber; g.setLineDash([]);
  if (hover.type === 'cell') { poly(g, CELLS.poly[hover.id], true); g.stroke(); }
  else if (hover.type === 'weed') { poly(g, D.weeds[hover.id].poly, true); g.stroke(); }
  else if (hover.type === 'obst') { poly(g, OBST[hover.id].poly, true); g.stroke(); }
  else if (hover.type === 'station') {
    const s = STN[hover.id]; g.beginPath(); g.arc(SX(s.x), SY(s.y), 9, 0, 6.283); g.stroke();
    for (const c of D.cells.station.map((v, i) => (v === s.sid ? i : -1)).filter((i) => i >= 0)) { poly(g, CELLS.poly[c], true); g.strokeStyle = 'rgba(224,151,47,0.55)'; g.lineWidth = 1; g.stroke(); }
  }
}

function render(now) {
  const g = mctx;
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.fillStyle = COL.bg; g.fillRect(0, 0, cw, ch);
  const ms = msPose(t, {}), x0 = SX(0), y0 = SY(Hm), w = Wm * view.s, h = Hm * view.s;
  g.imageSmoothingEnabled = true; g.imageSmoothingQuality = 'high';
  if (bmImg.complete && bmImg.naturalWidth) g.drawImage(bmImg, x0, y0, w, h);
  g.fillStyle = 'rgba(6,10,7,0.14)'; g.fillRect(x0, y0, w, h);
  if (L.coverage) { covTo(t); g.drawImage(covC, x0, y0, w, h); }
  const dst = []; for (let d = 0; d < ND; d++) dst.push(droneAt(d, t, ms));
  if (L.coverage) drawPartial(g, dst);
  if (L.cells) drawCells(g, t);
  if (L.weeds) drawWeeds(g);
  labelRects.length = 0;
  if (L.parcels) drawParcels(g);
  if (L.planned) drawPlanned(g);
  if (L.bypass) drawBypasses(g);
  if (L.actual) drawTrack(g, ms);
  const cur = stationAt(t);
  if (L.stations) drawStations(g, cur);
  drawDepot(g);
  if (L.mothership) drawReach(g, ms);
  if (L.obstacles) drawObstacles(g, now);
  if (L.lidar) drawLidar(g, ms, now);
  if (L.effector) drawEffector(g, now);
  if (L.trails && L.drones) drawTrails(g);
  if (L.mothership) drawMothership(g, ms);
  if (L.drones) drawDrones(g, dst);
  drawHover(g);
  drawScale(g);
}

// ================================================================== hit testing + tooltips
function hitTest(sx, sy) {
  const wx = WX(sx), wy = WY(sy), ms = msPose(t, {});
  if (L.drones) for (let d = 0; d < ND; d++) { const a = droneAt(d, t, ms); if (a.air && Math.hypot(SX(a.x) - sx, SY(a.y) - sy) < 9) return { type: 'drone', id: d }; }
  if (L.mothership && Math.hypot(SX(ms.x) - sx, SY(ms.y) - sy) < 14) return { type: 'ms', id: 0 };
  if (L.obstacles) for (let i = 0; i < OBST.length; i++) { const o = OBST[i]; if (o.t_det === null || t < o.t_det) continue; if (pip(wx, wy, o.poly) || Math.hypot(SX(o.c[0]) - sx, SY(o.c[1]) - sy) < 10) return { type: 'obst', id: i }; }
  if (L.stations) { let best = -1, bd = 9; for (let i = 0; i < STN.length; i++) { const dd = Math.hypot(SX(STN[i].x) - sx, SY(STN[i].y) - sy); if (dd < bd) { bd = dd; best = i; } } if (best >= 0) return { type: 'station', id: best }; }
  if (L.weeds) for (let i = 0; i < D.weeds.length; i++) if (pip(wx, wy, D.weeds[i].poly)) return { type: 'weed', id: i };
  for (let c = 0; c < NC; c++) { if (wx < cellBox[4 * c] || wx > cellBox[4 * c + 2] || wy < cellBox[4 * c + 1] || wy > cellBox[4 * c + 3]) continue; if (pip(wx, wy, CELLS.poly[c])) return { type: 'cell', id: c }; }
  return null;
}
const row = (k, v) => '<div class="r"><span>' + k + '</span><b>' + v + '</b></div>';
function tipHTML(h) {
  if (h.type === 'cell') {
    const c = h.id, tgt = CELLS.target[c], kg = cellKg(c, t), f = tgt > 0 ? kg / tgt : 0, P = D.parcels[CELLS.parcel[c]], dn = CELLS.done[c];
    let s = '<h4>Work cell ' + c + '</h4>' + row('Parcel', P ? P.letter + ' · ' + esc(P.crop_label) : '-');
    if (CELLS.blocked[c]) return s + '<div class="note">Blocked: under a discovered obstacle, no material applied.</div>';
    s += row('Area', nf(CELLS.area[c], 1) + ' m²') + row('Target', nf(tgt, 0) + ' kg') + row('Applied now', nf(kg, 0) + ' kg · ' + nf(100 * Math.min(f, 9.99), 0) + ' %');
    s += '<div class="tbar"><i style="width:' + clamp(100 * f, 0, 100).toFixed(1) + '%;background:' + (f >= 0.999 ? 'var(--green)' : 'var(--chip)') + '"></i></div>';
    s += row(dn !== null && dn <= t ? 'Completed' : 'Completes', dn === null ? 'never' : tph(dn) + ' · ' + clock(dn).slice(0, 14));
    s += row('Served from', 'station ' + (CELLS.station[c] + 1));
    return s;
  }
  if (h.type === 'station') {
    const s = STN[h.id], P = D.parcels[s.parcel], dwell = s.td - s.ta, st = t < s.ta ? 'upcoming' : t <= s.td ? 'working now' : 'done';
    return '<h4>Station ' + (s.sid + 1) + ' / ' + STN.length + ' · ' + st + '</h4>' + row('Parcel', P ? P.letter + ' · ' + esc(P.crop_label) : '-') +
      row('Arrive', tph(s.ta)) + row('Depart', tph(s.td)) + row('Dwell', dur(dwell)) + row('Drone sorties', nf(s.sorties)) +
      row('Effector bursts', nf(s.bursts) + ' (' + M.fx_name + ')') + row('Cells served', s.ncells);
  }
  if (h.type === 'obst') {
    const o = OBST[h.id];
    let s = '<h4>' + esc(o.name) + '</h4><div class="note" style="color:#ffc3f7;margin:0 0 4px">Not in the satellite / survey map: found online by the ' + o.sensor + '.</div>';
    s += row('Detected', tph(o.t_det) + ' · ' + clock(o.t_det).slice(0, 14)) + row('Detection range', nf(o.range_m, 1) + ' m (' + o.sensor + ' ' + (o.sensor === 'lidar' ? M.lidar_m : M.camera_m) + ' m)');
    s += row('Height', nf(o.height, 2) + ' m');
    if (o.detour) s += row('Bypass', o.detour.side + ' +' + nf(o.detour.offset, 1) + ' m') + row('Bypass length', nf(o.detour.length, 1) + ' m');
    if (o.note) s += '<div class="note">' + esc(o.note) + '</div>';
    return s;
  }
  if (h.type === 'weed') {
    const w = D.weeds[h.id];
    return '<h4>Mapped weed patch</h4>' + row('Species', '<i>' + esc(w.species) + '</i>') + row('Severity', esc(w.severity)) + '<div class="note">Known from the survey: the global route bypasses it (filleted detour) and no cells are planned inside.</div>';
  }
  if (h.type === 'ms') {
    const ms = msPose(t, {}), cur = stationAt(t);
    return '<h4>Mothership UGV</h4>' + row('State', ms.parked ? 'parked at station ' + (STN[cur] ? STN[cur].sid + 1 : '-') : (MODE_NAME[ms.mode] || '').replace('_', ' ')) + row('Speed', nf(ms.v, 2) + ' m/s') +
      row('Hopper', nf(hopperAt(t), 0) + ' kg') + row('Odometer', nf(ms.odo / 1000, 3) + ' km') + row('Effector', esc(M.effector)) + row('Reach', '40 ft = ' + M.reach_m.toFixed(2) + ' m');
  }
  if (h.type === 'drone') {
    const ms = msPose(t, {}), a = droneAt(h.id, t, ms);
    let s = '<h4 style="color:' + DRONE_COL[h.id % DRONE_COL.length] + '">Drone D' + (h.id + 1) + '</h4>' + row('State', a.st);
    if (a.k >= 0) s += row('Cell', sCell[a.k]) + row('Payload', nf(sKg[a.k], 1) + ' kg') + row('Sortie', tph(sTo[a.k]) + ' to ' + tph(sTd[a.k]));
    return s;
  }
  return '';
}
const tipEl = $('tip');
function showTip() {
  if (!hover || !mouse) { tipEl.style.display = 'none'; return; }
  tipEl.innerHTML = tipHTML(hover); tipEl.style.display = 'block';
  const r = tipEl.getBoundingClientRect(); let x = mouse[0] + 16, y = mouse[1] + 14;
  if (x + r.width > cw - 6) x = mouse[0] - r.width - 14; if (y + r.height > ch - 6) y = Math.max(6, ch - r.height - 6);
  tipEl.style.left = x + 'px'; tipEl.style.top = y + 'px';
}

// ================================================================== map interaction
const ptrs = new Map(); let drag = null, pinch = null;
mapC.addEventListener('pointerdown', (e) => {
  mapC.setPointerCapture(e.pointerId); ptrs.set(e.pointerId, [e.offsetX, e.offsetY]);
  if (ptrs.size === 1) { drag = { x: e.offsetX, y: e.offsetY, cx: view.cx, cy: view.cy, moved: false }; mapC.classList.add('drag'); }
  else if (ptrs.size === 2) { const [a, b] = Array.from(ptrs.values()); pinch = { d: Math.hypot(a[0] - b[0], a[1] - b[1]), s: view.s, mx: (a[0] + b[0]) / 2, my: (a[1] + b[1]) / 2 }; drag = null; }
});
mapC.addEventListener('pointermove', (e) => {
  if (ptrs.has(e.pointerId)) ptrs.set(e.pointerId, [e.offsetX, e.offsetY]);
  if (pinch && ptrs.size === 2) {
    const [a, b] = Array.from(ptrs.values()); const d = Math.hypot(a[0] - b[0], a[1] - b[1]);
    zoomAt(pinch.mx, pinch.my, (pinch.s * d / Math.max(pinch.d, 1)) / view.s); setFollow(false); return;
  }
  if (drag) {
    const dx = e.offsetX - drag.x, dy = e.offsetY - drag.y;
    if (Math.abs(dx) + Math.abs(dy) > 3) { drag.moved = true; setFollow(false); }
    view.cx = drag.cx - dx / view.s; view.cy = drag.cy + dy / view.s; view.ts = null; dirty = true; hover = null; showTip(); return;
  }
  mouse = [e.offsetX, e.offsetY]; const h = hitTest(e.offsetX, e.offsetY);
  if ((h && (!hover || h.type !== hover.type || h.id !== hover.id)) || (!h && hover)) { hover = h; dirty = true; }
  mapC.style.cursor = hover ? 'pointer' : '';
  showTip();
});
function endPtr(e) { ptrs.delete(e.pointerId); if (ptrs.size < 2) pinch = null; if (ptrs.size === 0) { drag = null; mapC.classList.remove('drag'); } }
mapC.addEventListener('pointerup', endPtr); mapC.addEventListener('pointercancel', endPtr);
mapC.addEventListener('pointerleave', () => { mouse = null; if (hover) { hover = null; dirty = true; } showTip(); });
mapC.addEventListener('wheel', (e) => { e.preventDefault(); const k = e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? 400 : 1; zoomAt(e.offsetX, e.offsetY, Math.exp(-e.deltaY * k * 0.0016)); }, { passive: false });
mapC.addEventListener('dblclick', (e) => { e.preventDefault(); setFollow(false); fitView(); });
$('zIn').onclick = () => zoomAt(cw / 2, ch / 2, 1.5);
$('zOut').onclick = () => zoomAt(cw / 2, ch / 2, 1 / 1.5);
$('zFit').onclick = () => { setFollow(false); fitView(); };
$('zFollow').onclick = () => setFollow(!follow);

// ================================================================== side panel
const els = {};
function set(id, v) { const e = els[id] || (els[id] = $(id)); if (e._v !== v) { e._v = v; e.textContent = v; } }
function setW(id, frac) { const e = els[id] || (els[id] = $(id)); const v = (100 * clamp(frac, 0, 1)).toFixed(2) + '%'; if (e._w !== v) { e._w = v; e.style.width = v; } }
const RING_C = 2 * Math.PI * 41;
function buildPanel() {
  $('modePill').textContent = CART ? 'CARTRIDGE MODE' : 'DUST MODE';
  $('modePill').className = 'pill ' + M.mode;
  $('hdrTitle').innerHTML = esc(M.mode_label) + ' · ' + nf(M.material_t, CART ? 0 : 0) + ' t over ' + nf(M.area_ha, 2) + ' ha<span class="xtra"> · ' + nf(T_END / 3600, 0) + ' h · ' + nf(M.sorties_total) + ' drone sorties</span>';
  $('hdrSite').textContent = M.site + ' · 4 parcels · 1 mothership UGV + ' + ND + ' heavy-lift drones + ' + M.effector + ' (40 ft reach) · ' + M.n_stations + ' stations';
  set('kTarget', nf(TL.target_t, CART ? 1 : 0)); set('kArea', nf(M.area_ha, 2) + ' ha');
  set('kFxName', M.fx_name); set('lgFx', 'mothership ' + M.fx_name); set('kHopCap', nf(M.hopper_cap));
  $('kHopThr').style.left = (100 * M.tender_request_frac).toFixed(1) + '%';
  set('kCellsN', '/ ' + nf(TL.n_cells)); set('kAirN', '/ ' + ND);
  $('kAirDots').innerHTML = DRONE_COL.slice(0, ND).map((c) => '<i style="border-color:' + c + ';background:' + c + '"></i>').join('');
  set('trEnd', '/ ' + (T_END / 3600).toFixed(1) + ' h · ' + clock(T_END).slice(0, 5));
  set('kOdo2', 'planned route ' + nf(D.kpis.planned_distance_km, 2) + ' km');
  // parcels
  $('parcels').innerHTML = D.parcels.map((p, i) => '<div class="prow"><span class="pl">' + p.letter + '</span><div><div class="pn" title="' + esc(p.crop_label) + '">' + esc(p.crop_label) + ' · ' + nf(p.target_t, CART ? 1 : 0) +
    ' t</div><div class="pb"><i id="pb' + i + '"></i></div></div><span class="pv" id="pv' + i + '">0 %</span></div>').join('');
  set('parcHint', CART ? 'cartridges placed vs target' : 'mulch applied vs target');
  // swimlane legend
  const SL = COL.states; $('swLg').innerHTML = Object.keys(SL).map((k) => '<span><i class="sw" style="background:' + SL[k] + '"></i>' + k + '</span>').join('');
  // events
  $('events').innerHTML = EVENTS.map((e, i) => '<div class="er' + (e.kind === 'obstacle_detected' ? ' obst' : e.kind === 'detour_spliced' ? ' byp' : '') + '" data-i="' + i + '"><span class="et">' + (e.t / 3600).toFixed(2) + ' h</span><i class="k-' + e.kind + '"></i><span class="el" title="' + esc(e.label) + '">' + esc(e.label) + '</span></div>').join('');
  $('events').addEventListener('click', (e) => { const r = e.target.closest('.er'); if (!r) return; const ev = EVENTS[+r.dataset.i]; seek(ev.kind === 'obstacle_detected' || ev.kind === 'detour_spliced' ? ev.t - 15 : ev.t + 1); });
  $('events').addEventListener('wheel', () => { evUserScroll = performance.now(); }, { passive: true });
  // impact
  const I = D.impact;
  let h = '<div class="lbl">CO₂e stored + avoided so far</div><div class="bigc"><span id="iTot">0</span><span class="u">t CO₂e</span></div>';
  h += row('Soil carbon stored', '<span id="iSoil">0</span> t') + row('Burning GHG avoided', '<span id="iBurn">0</span> t') + row('PM2.5 avoided', '<span id="iPm">0</span> kg') + row('Nitrogen retained', '<span id="iN">0</span> kg');
  if (I.saving_eur_total !== null && I.saving_eur_total !== undefined) h += row('Saved vs REGENSOL', '€<span id="iEur">0</span>');
  if (CART) h += row('Cartridges placed', '<span id="iCart">0</span>') + row('Seeds placed', '<span id="iSeed">0</span>');
  h += '<div class="fin">Mission end: ' + nf(I.total_co2e_t, CART ? 1 : 0) + ' t CO₂e · ' + nf(I.co2e_t_per_ha, 1) + ' t/ha' + (CART && M.species.length ? ' · ' + M.species.join(', ') : '') + (!CART ? ' · €' + nf(I.saving_eur_per_1000m2, 0) + ' per 1000 m²' : '') + '</div>';
  $('impact').innerHTML = h;
  const src = $('srcBtn');
  src.addEventListener('mouseenter', () => {
    const rows = D.factors.map((f) => '<tr><td><b>' + esc(f.label) + '</b><br>' + esc(f.source) + (f.assumed ? ' <i>(assumption)</i>' : '') + '</td><td style="white-space:nowrap"><b>' + f.value + '</b> ' + esc(f.unit) + '</td></tr>').join('');
    sideTip.innerHTML = '<h4>Impact factors and sources</h4><table>' + rows + '</table><div class="note">Scaled by the share of material applied at the playhead. Biogenic CO₂ from burning is not counted (no double counting).</div>';
    sideTip.style.display = 'block'; const r = src.getBoundingClientRect(), tr = sideTip.getBoundingClientRect();
    sideTip.style.left = Math.max(8, r.right - tr.width) + 'px'; sideTip.style.top = Math.max(8, r.top - tr.height - 8) + 'px';
  });
  src.addEventListener('mouseleave', () => { sideTip.style.display = 'none'; });
}
const sideTip = document.createElement('div'); sideTip.id = 'sideTip'; sideTip.style.cssText = 'position:fixed;display:none;z-index:30;max-width:470px';
sideTip.className = 'tipbox'; document.body.appendChild(sideTip);
Object.assign(sideTip.style, { background: 'rgba(8,12,9,.97)', border: '1px solid #33443a', borderRadius: '8px', padding: '9px 11px', fontSize: '11.5px', boxShadow: '0 6px 22px rgba(0,0,0,.5)', pointerEvents: 'none' });
let evCur = -2, evUserScroll = -1e9;
function updatePanel() {
  const S = stats(t), ms = S.ms;
  set('clk', clock(t)); set('tplus', tph(t) + ' · ' + nf(100 * t / T_END, 1) + ' % of mission');
  set('trT', tph(t, 1));
  // phase
  const cur = stationAt(t), ph = $('phase'); let txt, cls = '';
  const det = OBST.find((o) => o.t_det !== null && t >= o.t_bypass - 1 && t <= o.t_clear + 5);
  if (t >= T_END - 1) { txt = 'MISSION COMPLETE · BACK AT DEPOT'; cls = 'done'; }
  else if (ms.parked && cur >= 0) { const fxOn = ub(shTf0, t) > 0 && (() => { const k = shOrd[ub(shTf0, t) - 1]; return t <= (mKind[k] === 0 ? mTfree[k] : mT1[k]); })(); txt = 'PARKED · STATION ' + (STN[cur].sid + 1) + '/' + STN.length + ' · ' + S.air + ' DRONES UP' + (fxOn ? ' · ' + M.fx_name.toUpperCase() + ' FIRING' : ''); cls = 'park'; }
  else if (det) { txt = 'BYPASSING · ' + det.name.toUpperCase(); cls = 'obst'; }
  else { const mn = MODE_NAME[ms.mode] || 'transit'; txt = 'DRIVING · ' + ({ lane: 'FURROW LANE', headland: 'HEADLAND U-TURN', transit: 'TRANSFER ON TRACKS', weed_detour: 'MAPPED-WEED BYPASS', obstacle_detour: 'OBSTACLE BYPASS', station: 'STATION' }[mn] || mn.toUpperCase()) + ' · ' + nf(ms.v, 2) + ' m/s'; }
  if (ph._v !== txt) { ph._v = txt; ph.textContent = txt; ph.className = 'phase ' + cls; }
  // hero
  const fD = S.droneKg / TARGET_KG, fM = S.msKg / TARGET_KG;
  const ringD = $('ringD'), ringM = $('ringM');
  ringD.setAttribute('stroke-dasharray', (fD * RING_C).toFixed(2) + ' 999'); ringM.setAttribute('stroke-dasharray', (fM * RING_C).toFixed(2) + ' 999');
  ringM.setAttribute('stroke-dashoffset', (-fD * RING_C).toFixed(2));
  set('ringT', nf(100 * S.applied / TARGET_KG, 0) + '%');
  const dec = CART ? 2 : 1;
  set('kApplied', nf(S.applied / 1000, dec)); set('kDrone', nf(S.droneKg / 1000, dec)); set('kMs', nf(S.msKg / 1000, dec));
  set('kHop', nf(S.hopper, 0)); setW('kHopBar', S.hopper / M.hopper_cap); set('kTop', nf(S.tops)); set('kTopT', nf(S.topKg / 1000, 1));
  set('kCells', nf(S.cells)); setW('kCellsBar', S.cells / TL.n_cells);
  set('kSort', nf(S.sorties)); set('kSort2', S.inflight + ' in flight · ' + nf(100 * S.sorties / Math.max(M.sorties_total, 1), 0) + ' % of ' + nf(M.sorties_total));
  set('kSwap', nf(S.swaps)); set('kSwap2', 'hot-swap ' + M.swap_s.toFixed(0) + ' s on deck');
  set('kOdo', nf(ms.odo / 1000, 2));
  set('kAir', String(S.air));
  const dots = $('kAirDots').children; for (let d = 0; d < dots.length; d++) { const on = S.airs[d].air; if (dots[d]._on !== on) { dots[d]._on = on; dots[d].className = on ? 'on' : ''; } }
  set('kFx', S.applied > 0 ? nf(100 * S.msKg / S.applied, 1) : '0.0'); set('kFx2', nf(S.bursts) + (CART ? ' volleys' : ' bursts') + ' · ' + M.fx_name);
  // charts header values
  set('cA', nf(S.applied / 1000, dec) + ' t'); set('cB', nf(S.cells) + ' / ' + nf(TL.n_cells)); set('cC', nf(S.hopper, 0) + ' kg');
  // parcels
  for (let p = 0; p < NPARC; p++) {
    const f = parcelFrac(p, t), bar = els['pb' + p] || (els['pb' + p] = $('pb' + p)), col = f >= 0.999 ? COL.green : CART ? '#9be15d' : COL.chip;
    setW('pb' + p, f); if (bar._c !== col) { bar._c = col; bar.style.background = col; }
    set('pv' + p, nf(100 * Math.min(f, 1), f > 0 && f < 0.995 ? 1 : 0) + ' %');
  }
  // impact (scaled by share of final material, as MissionTimeline.at)
  const I = D.impact, f = S.applied / FINAL_KG, d1 = CART ? 2 : 1;
  set('iTot', nf(f * I.total_co2e_t, d1)); set('iSoil', nf(f * I.soil_co2e_t, d1)); set('iBurn', nf(f * I.burn_ghg_avoided_co2e_t, d1));
  set('iPm', nf(f * I.pm25_avoided_kg, 0)); set('iN', nf(f * I.n_retained_kg, 0));
  if (els.iEur || $('iEur')) set('iEur', nf(f * I.saving_eur_total, 0));
  if (CART) { set('iCart', nf(f * (I.cartridges_placed || 0), 0)); set('iSeed', nf(f * (I.seeds_placed || 0), 0)); }
  // events
  const ci = ub(evT, t + 1e-6) - 1;
  if (ci !== evCur) {
    evCur = ci; const rows = $('events').children;
    for (let i = 0; i < rows.length; i++) rows[i].className = 'er' + (EVENTS[i].kind === 'obstacle_detected' ? ' obst' : EVENTS[i].kind === 'detour_spliced' ? ' byp' : '') + (i > ci ? ' fut' : '') + (i === ci ? ' cur' : '');
    if (ci >= 0 && performance.now() - evUserScroll > 3000) { const box = $('events'), r = rows[ci]; box.scrollTop = r.offsetTop - box.offsetTop - box.clientHeight / 2 + r.offsetHeight / 2; }
  }
  // banner
  const bn = $('banner');
  const by = OBST.find((o) => o.t_det !== null && o.detour && o.t_bypass - o.t_det > 90 && t >= o.t_bypass - 12 && t <= o.t_clear + 5);
  const ob = by ? null : OBST.find((o) => o.t_det !== null && t >= o.t_det && t - o.t_det < recentW());
  if (ob || by) {
    const html = ob
      ? '<b>' + (ob.sensor === 'lidar' ? 'LIDAR' : 'CAMERA') + ' CONTACT</b><span class="ob">' + esc(ob.name.toLowerCase()) + '</span> at ' + nf(ob.range_m, 1) + ' m · not in satellite map' + (ob.detour ? ' · bypass ' + ob.detour.side + ' +' + nf(ob.detour.offset, 1) + ' m (' + nf(ob.detour.length, 0) + ' m)' : '')
      : '<b>ONLINE BYPASS</b>around the <span class="ob">' + esc(by.name.toLowerCase()) + '</span> · ' + by.detour.side + ' +' + nf(by.detour.offset, 1) + ' m · ' + nf(by.detour.length, 0) + ' m re-planned around it';
    if (bn._v !== html) { bn._v = html; bn.innerHTML = html; } bn.style.display = 'block';
  } else bn.style.display = 'none';
  if (hover) showTip();
}

// ================================================================== charts
const PADL = 40, PADR = 8;
function setupCanvas(c) { const r = c.getBoundingClientRect(); const w = Math.max(20, r.width), h = Math.max(16, r.height); c.width = Math.round(w * dpr); c.height = Math.round(h * dpr); return [w, h]; }
const charts = [];
function chartX(c, tt) { return PADL + (tt / T_END) * (c.w - PADL - PADR); }
function chartY(c, v) { return c.pt + (1 - v / c.ymax) * (c.h - c.pt - c.pb); }
function niceMax(v) { const e = Math.pow(10, Math.floor(Math.log10(Math.max(v, 1e-9)))); for (const m of [1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10]) if (m * e >= v) return m * e; return 10 * e; }
function gridY(g, c, fmt) {
  g.font = '10px ' + MONO; g.textAlign = 'right'; g.textBaseline = 'middle';
  for (let i = 0; i <= 2; i++) { const v = c.ymax * i / 2, y = chartY(c, v); g.strokeStyle = i ? 'rgba(154,171,148,0.13)' : 'rgba(154,171,148,0.35)'; g.lineWidth = 1; g.beginPath(); g.moveTo(PADL, Math.round(y) + 0.5); g.lineTo(c.w - PADR, Math.round(y) + 0.5); g.stroke(); g.fillStyle = COL.muted; g.fillText(fmt(v), PADL - 5, y); }
}
function areaPath(g, c, ys, base) {
  const T = TL.t; g.beginPath(); g.moveTo(chartX(c, T[0]), chartY(c, base ? base[0] : 0));
  for (let i = 0; i < T.length; i++) g.lineTo(chartX(c, T[i]), chartY(c, (base ? base[i] : 0) + ys[i]));
  for (let i = T.length - 1; i >= 0; i--) g.lineTo(chartX(c, T[i]), chartY(c, base ? base[i] : 0));
  g.closePath();
}
function linePath(g, c, ys) { const T = TL.t; g.beginPath(); for (let i = 0; i < T.length; i++) { const X = chartX(c, T[i]), Y = chartY(c, ys[i]); if (i) g.lineTo(X, Y); else g.moveTo(X, Y); } }
function hourTicks(g, c) {
  const H = T_END / 3600, step = [1, 2, 5, 10, 12, 20, 24, 50][[1, 2, 5, 10, 12, 20, 24, 50].findIndex((s) => H / s <= 7)] || 50;
  g.font = '10px ' + MONO; g.fillStyle = COL.muted; g.textAlign = 'center'; g.textBaseline = 'top';
  for (let h = 0; h <= H + 1e-6; h += step) { const X = chartX(c, h * 3600); g.fillRect(X, c.h - c.pb, 1, 3); g.fillText(h + ' h', clamp(X, PADL + 8, c.w - PADR - 12), c.h - c.pb + 3); }
}
function makeChart(id, o) {
  const c = { el: $(id), o, w: 0, h: 0, pt: 7, pb: o.axis ? 15 : 3, ymax: 1, cache: document.createElement('canvas') };
  c.g = c.el.getContext('2d'); charts.push(c);
  const seekAt = (e) => { const r = c.el.getBoundingClientRect(); seek(clamp((e.clientX - r.left - PADL) / (c.w - PADL - PADR), 0, 1) * T_END); };
  c.el.addEventListener('pointerdown', (e) => { c.el.setPointerCapture(e.pointerId); c.dragging = true; seekAt(e); });
  c.el.addEventListener('pointermove', (e) => { const r = c.el.getBoundingClientRect(); hoverT = clamp((e.clientX - r.left - PADL) / (c.w - PADL - PADR), 0, 1) * T_END; if (c.dragging) seekAt(e); });
  c.el.addEventListener('pointerup', () => { c.dragging = false; });
  c.el.addEventListener('pointerleave', () => { hoverT = null; });
  return c;
}
function buildChart(c) {
  const wh = setupCanvas(c.el); c.w = wh[0]; c.h = wh[1]; c.cache.width = c.el.width; c.cache.height = c.el.height;
  const g = c.cache.getContext('2d'); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, c.w, c.h); c.o.draw(g, c); if (c.o.axis) hourTicks(g, c);
}
function drawChartDyn(c) {
  const g = c.g; g.setTransform(1, 0, 0, 1, 0, 0); g.clearRect(0, 0, c.el.width, c.el.height); g.drawImage(c.cache, 0, 0); g.setTransform(dpr, 0, 0, dpr, 0, 0);
  if (hoverT !== null) { const X = chartX(c, hoverT); g.strokeStyle = 'rgba(232,239,228,0.35)'; g.setLineDash([2, 3]); g.lineWidth = 1; g.beginPath(); g.moveTo(X, c.pt); g.lineTo(X, c.h - c.pb); g.stroke(); g.setLineDash([]); }
  const X = chartX(c, t); g.fillStyle = 'rgba(11,16,12,0.45)'; g.fillRect(X, c.pt, c.w - PADR - X, c.h - c.pt - c.pb);
  g.strokeStyle = '#fff'; g.lineWidth = 1.2; g.beginPath(); g.moveTo(X, c.pt - 2); g.lineTo(X, c.h - c.pb); g.stroke();
  const v = c.o.value(t); g.beginPath(); g.arc(X, chartY(c, v), 3.2, 0, 6.283); g.fillStyle = c.o.dot; g.fill(); g.strokeStyle = '#0b100c'; g.lineWidth = 1; g.stroke();
}
function interpTL(arr, tt) { const T = TL.t, i = clamp(ub(T, tt) - 1, 0, T.length - 2), a = clamp((tt - T[i]) / Math.max(T[i + 1] - T[i], 1e-9), 0, 1); return arr[i] + a * (arr[i + 1] - arr[i]); }
const tot = TL.drone_t.map((v, i) => v + TL.ms_t[i]);
makeChart('chA', {
  dot: COL.amber, value: (tt) => interpTL(tot, tt),
  draw(g, c) {
    c.ymax = niceMax(TL.target_t * 1.04); gridY(g, c, (v) => nf(v, 0));
    areaPath(g, c, TL.drone_t); g.fillStyle = 'rgba(123,220,106,0.55)'; g.fill();
    areaPath(g, c, TL.ms_t, TL.drone_t); g.fillStyle = 'rgba(255,224,102,0.8)'; g.fill();
    linePath(g, c, tot); g.strokeStyle = '#e8efe4'; g.lineWidth = 1.2; g.stroke();
    const y = chartY(c, TL.target_t); g.setLineDash([4, 3]); g.strokeStyle = COL.amber; g.beginPath(); g.moveTo(PADL, y); g.lineTo(c.w - PADR, y); g.stroke(); g.setLineDash([]);
    g.font = '10px ' + FONT; g.fillStyle = COL.amber; g.textAlign = 'left'; g.textBaseline = 'top'; g.fillText('target ' + nf(TL.target_t, CART ? 1 : 0) + ' t', PADL + 6, y + 2);
  },
});
makeChart('chB', {
  dot: COL.amber, value: (tt) => ub(cellDoneSorted, tt),
  draw(g, c) {
    c.ymax = TL.n_cells; gridY(g, c, (v) => nf(v));
    areaPath(g, c, TL.cells_done); g.fillStyle = 'rgba(224,151,47,0.28)'; g.fill();
    linePath(g, c, TL.cells_done); g.strokeStyle = COL.amber; g.lineWidth = 1.5; g.stroke();
  },
});
makeChart('chC', {
  dot: COL.actual, value: (tt) => hopperAt(tt), axis: true,
  draw(g, c) {
    c.ymax = M.hopper_cap * 1.05; gridY(g, c, (v) => (v ? nf(v / 1000, 1) + ' t' : '0'));
    const Wp = Math.max(2, Math.ceil(c.w - PADL - PADR)), mn = new Float64Array(Wp), mx = new Float64Array(Wp);
    let cur = M.hopper_cap, j = 0;
    for (let i = 0; i < Wp; i++) { const te = (i + 1) / Wp * T_END; mn[i] = mx[i] = cur; while (j < hopT.length && hopT[j] <= te) { cur = hopV[j++]; if (cur < mn[i]) mn[i] = cur; if (cur > mx[i]) mx[i] = cur; } }
    g.beginPath(); for (let i = 0; i < Wp; i++) g.lineTo(PADL + i + 0.5, chartY(c, mx[i])); for (let i = Wp - 1; i >= 0; i--) g.lineTo(PADL + i + 0.5, chartY(c, mn[i])); g.closePath();
    g.fillStyle = 'rgba(56,214,230,0.42)'; g.fill(); g.strokeStyle = 'rgba(56,214,230,0.9)'; g.lineWidth = 0.8; g.stroke();
    const y = chartY(c, M.hopper_cap * M.tender_request_frac); g.setLineDash([3, 3]); g.strokeStyle = hexA(COL.amber, 0.8); g.beginPath(); g.moveTo(PADL, y); g.lineTo(c.w - PADR, y); g.stroke(); g.setLineDash([]);
  },
});

// swimlanes
const swC = $('swim'), swG = swC.getContext('2d'); let swW = 0, swH = 0;
function drawSwim() {
  const g = swG; g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, swW, swH);
  const Wwin = playing ? clamp(effSpeed * 1.5, 1200, 3 * 3600) : 1200, t0 = t - Wwin / 2, t1 = t + Wwin / 2, PL = 26, ab = 13, laneH = (swH - ab - 2) / ND;
  const X = (tt) => PL + ((tt - t0) / (t1 - t0)) * (swW - PL - 4), SC = COL.states;
  for (let d = 0; d < ND; d++) {
    const y = 1 + d * laneH, hh = laneH - 3;
    g.fillStyle = SC.idle; g.fillRect(PL, y, swW - PL - 4, hh);
    const dr = drones[d]; let j = Math.max(0, lb(dr.td, t0));
    const seg = (a, b, col) => { if (b <= t0 || a >= t1) return; const xa = X(Math.max(a, t0)), xb = X(Math.min(b, t1)); g.fillStyle = col; g.fillRect(xa, y, Math.max(xb - xa, 0.6), hh); };
    for (; j < dr.idx.length && dr.tl[j] < t1; j++) { const k = dr.idx[j]; seg(sTl[k], sTo[k], SC.loading); seg(sTo[k], sA0[k], SC.outbound); seg(sA0[k], sA1[k], SC.applying); seg(sA1[k], sTd[k], SC.return); }
    const sw = swByDrone[d]; for (let i = lb(sw, t0 - M.swap_s); i < sw.length && sw[i] < t1; i++) seg(sw[i], sw[i] + M.swap_s, SC.swap);
    g.fillStyle = DRONE_COL[d % DRONE_COL.length]; g.font = '600 10px ' + MONO; g.textAlign = 'left'; g.textBaseline = 'middle'; g.fillText('D' + (d + 1), 2, y + hh / 2);
  }
  const Xc = X(t); g.strokeStyle = '#fff'; g.lineWidth = 1.2; g.beginPath(); g.moveTo(Xc, 0); g.lineTo(Xc, swH - ab); g.stroke();
  g.font = '10px ' + MONO; g.fillStyle = COL.muted; g.textBaseline = 'top';
  g.textAlign = 'left'; g.fillText('−' + dur(Wwin / 2), PL, swH - ab + 1); g.textAlign = 'center'; g.fillText('now', Xc, swH - ab + 1); g.textAlign = 'right'; g.fillText('+' + dur(Wwin / 2), swW - 4, swH - ab + 1);
}

// scrubber
const scC = $('scrub'), scG = scC.getContext('2d'); let scW = 0, scH = 0; const scCache = document.createElement('canvas');
const SPL = 8, SPR = 8;
const scX = (tt) => SPL + (tt / T_END) * (scW - SPL - SPR);
function buildScrub() {
  const wh = setupCanvas(scC); scW = wh[0]; scH = wh[1]; scCache.width = scC.width; scCache.height = scC.height;
  const g = scCache.getContext('2d'); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, scW, scH);
  const y = scH / 2 - 4; g.fillStyle = '#1b271d'; rrect(g, SPL, y, scW - SPL - SPR, 8, 4); g.fill();
  for (let i = 0; i < STN.length; i++) { const a = scX(STN[i].ta), b = scX(STN[i].td); g.fillStyle = i % 2 ? 'rgba(255,224,102,0.10)' : 'rgba(255,224,102,0.18)'; g.fillRect(a, y + 1, Math.max(1, b - a), 6); }
  g.fillStyle = 'rgba(232,239,228,0.32)'; for (const s of STN) g.fillRect(Math.round(scX(s.ta)), y + 10, 1, 4);
  for (const o of OBST) { if (o.t_det === null) continue; const X = scX(o.t_det); g.fillStyle = COL.obst; g.beginPath(); g.moveTo(X, y - 2); g.lineTo(X - 5, y - 9); g.lineTo(X + 5, y - 9); g.closePath(); g.fill(); g.fillStyle = COL.bypass; g.fillRect(X - 0.5, y - 2, 1.5, 12); }
}
function drawScrub() {
  const g = scG; g.setTransform(1, 0, 0, 1, 0, 0); g.clearRect(0, 0, scC.width, scC.height); g.drawImage(scCache, 0, 0); g.setTransform(dpr, 0, 0, dpr, 0, 0);
  const y = scH / 2 - 4, X = scX(t);
  const grd = g.createLinearGradient(SPL, 0, X, 0); grd.addColorStop(0, 'rgba(224,151,47,0.55)'); grd.addColorStop(1, 'rgba(224,151,47,0.95)');
  g.fillStyle = grd; rrect(g, SPL, y, Math.max(X - SPL, 0.1), 8, 4); g.fill();
  if (hoverT !== null) { const H = scX(hoverT); g.fillStyle = 'rgba(232,239,228,0.6)'; g.fillRect(H - 0.5, y - 6, 1, 20); }
  g.beginPath(); g.arc(X, y + 4, 7, 0, 6.283); g.fillStyle = '#fff'; g.fill(); g.lineWidth = 2; g.strokeStyle = COL.amber; g.stroke();
}
let scDrag = false; const scTip = $('scrubTip');
function scTime(e) { const r = scC.getBoundingClientRect(); return clamp((e.clientX - r.left - SPL) / (scW - SPL - SPR), 0, 1) * T_END; }
scC.addEventListener('pointerdown', (e) => { scC.setPointerCapture(e.pointerId); scDrag = true; seek(scTime(e)); });
scC.addEventListener('pointermove', (e) => {
  const tt = scTime(e); hoverT = tt; if (scDrag) seek(tt);
  let near = null, bd = 7; for (const ev of EVENTS) { if (ev.kind === 'arrive_station') continue; const d = Math.abs(scX(ev.t) - scX(tt)); if (d < bd) { bd = d; near = ev; } }
  scTip.textContent = tph(tt, 2) + ' · ' + clock(tt) + (near ? ' · ' + near.label : ''); scTip.style.display = 'block'; scTip.style.left = clamp(scX(tt), 120, scW - 120) + 'px';
});
scC.addEventListener('pointerup', () => { scDrag = false; });
scC.addEventListener('pointerleave', () => { hoverT = null; scTip.style.display = 'none'; });

// ================================================================== transport
const SPEEDS = [60, 300, 1000, 3000, 6000, 10000, 20000];
const spSel = $('speed');
spSel.innerHTML = '<option value="auto">auto</option>' + SPEEDS.map((s) => '<option value="' + s + '">' + nf(s) + '×</option>').join('');
spSel.addEventListener('change', () => { setSpeed(spSel.value); spSel.blur(); });
function setSpeed(v) {
  if (v === 'auto' || v === undefined) {
    autoSpeed = true; baseSpeed = autoBase(); spSel.value = 'auto';
    spSel.options[0].textContent = 'auto ' + nf(baseSpeed) + '×'; spSel.options[0].title = 'whole mission in about ' + M.auto_play_s + ' s';
  }
  else { autoSpeed = false; baseSpeed = clamp(+v, 1, 100000); const opt = SPEEDS.indexOf(baseSpeed) >= 0 ? String(baseSpeed) : null; if (opt) spSel.value = opt; }
  if (!playing) effSpeed = baseSpeed;
}
function stepSpeed(dir) {
  const all = SPEEDS.slice(); let i = all.findIndex((s) => s >= baseSpeed - 1e-6); if (i < 0) i = all.length - 1;
  if (dir > 0) i = all[i] > baseSpeed + 1e-6 ? i : Math.min(all.length - 1, i + 1); else i = Math.max(0, all[i] < baseSpeed - 1e-6 ? i : i - 1);
  setSpeed(all[i]);
}
const ICON_PLAY = $('bPlay').innerHTML, ICON_PAUSE = '<svg width="18" height="18" viewBox="0 0 18 18"><path d="M4 3h3.6v12H4zM10.4 3H14v12h-3.6z" fill="currentColor"/></svg>';
function setPlaying(p) { playing = p; if (playing && t >= T_END - 1e-6) seek(0); $('bPlay').innerHTML = playing ? ICON_PAUSE : ICON_PLAY; if (!playing) effSpeed = baseSpeed; }
function setFollow(f) { follow = f; $('bFollow').classList.toggle('on', f); $('zFollow').classList.toggle('on', f); if (f && view.s < 5.5) view.ts = Math.min(7, view.fit * 4); dirty = true; }
function seek(tt) { t = clamp(tt, 0, T_END); if (playing) effSpeed = Math.min(effSpeed, targetSpeed(t)); dirty = true; panelDirty = true; }
function jumpEvent(dir) {
  const cur = t; let i;
  if (dir > 0) { i = ub(stopT, cur + 0.5); if (i < stopT.length) seek(stopT[i]); }
  else { i = lb(stopT, cur - 0.5) - 1; if (i >= 0) seek(stopT[i]); else seek(0); }
}
$('bPlay').onclick = () => setPlaying(!playing);
$('bPrev').onclick = () => jumpEvent(-1);
$('bNext').onclick = () => jumpEvent(1);
$('bFollow').onclick = () => setFollow(!follow);
function setSlowmo(on) { slowmo = on; $('bSlow').classList.toggle('on', slowmo); if (autoSpeed) setSpeed('auto'); }
$('bSlow').onclick = () => setSlowmo(!slowmo);
$('bLoop').onclick = () => { loop = !loop; $('bLoop').classList.toggle('on', loop); };
$('bHelp').onclick = () => { $('help').hidden = !$('help').hidden; };
$('help').onclick = () => { $('help').hidden = true; };
for (const b of document.querySelectorAll('button')) b.addEventListener('mouseup', () => b.blur());
document.addEventListener('keydown', (e) => {
  if (e.target && (e.target.tagName === 'SELECT' || e.target.tagName === 'INPUT')) return;
  const k = e.key;
  if (k === ' ') { e.preventDefault(); setPlaying(!playing); }
  else if (k === 'ArrowRight') { e.preventDefault(); if (e.shiftKey) seek(t + 0.01 * T_END); else jumpEvent(1); }
  else if (k === 'ArrowLeft') { e.preventDefault(); if (e.shiftKey) seek(t - 0.01 * T_END); else jumpEvent(-1); }
  else if (k === '.') seek(t + 60); else if (k === ',') seek(t - 60);
  else if (k === '+' || k === '=') stepSpeed(1); else if (k === '-' || k === '_') stepSpeed(-1);
  else if (k === 'f' || k === 'F') setFollow(!follow);
  else if (k === 'l' || k === 'L') $('bLoop').click();
  else if (k === 's' || k === 'S') $('bSlow').click();
  else if (k === '0' || k === 'r' || k === 'R') { setFollow(false); fitView(); }
  else if (k === 'Home') seek(0); else if (k === 'End') seek(T_END);
  else if (k === '?' || k === 'h' || k === 'H') $('help').hidden = !$('help').hidden;
  else if (k === 'Escape') $('help').hidden = true;
});
document.addEventListener('keyup', (e) => { if (e.key === ' ') e.preventDefault(); });

// layer chips + legend
const CHIPS = [['coverage', CART ? 'Cartridges' : 'Mulch', CART ? '#9be15d' : COL.chip], ['cells', 'Cell grid', COL.amber], ['parcels', 'Parcels', '#e8efe4'], ['weeds', 'Weeds', COL.weed],
  ['planned', 'Planned route', COL.planned], ['actual', 'Actual track', COL.actual], ['bypass', 'Bypasses', COL.bypass], ['obstacles', 'Obstacles', COL.obst],
  ['stations', 'Stations', '#f3f6f0'], ['mothership', 'Mothership', COL.actual], ['effector', CART ? 'Air cannons' : 'Chip blower', COL.fx], ['drones', 'Drones', DRONE_COL[0]],
  ['trails', 'Trails', DRONE_COL[2]], ['lidar', 'Lidar', COL.actual]];
$('chips').innerHTML = CHIPS.map((c) => '<span class="chip' + (L[c[0]] ? ' on' : '') + '" data-l="' + c[0] + '"><i style="background:' + c[2] + '"></i>' + c[1] + '</span>').join('');
$('chips').addEventListener('click', (e) => { const el = e.target.closest('.chip'); if (!el) return; const k = el.dataset.l; L[k] = L[k] ? 0 : 1; el.classList.toggle('on', !!L[k]); dirty = true; });
const sv = (inner) => '<svg width="30" height="12" viewBox="0 0 30 12">' + inner + '</svg>';
const LEG = [
  [sv('<line x1="1" y1="6" x2="29" y2="6" stroke="' + COL.planned + '" stroke-width="2" stroke-dasharray="5 3"/>'), 'Planned route (known map)'],
  [sv('<line x1="1" y1="6" x2="29" y2="6" stroke="' + COL.actual + '" stroke-width="2.4"/>'), 'Actual mothership track (10 Hz)'],
  [sv('<line x1="1" y1="6" x2="29" y2="6" stroke="' + COL.bypass + '" stroke-width="2.4" stroke-dasharray="4 3"/>'), 'Online bypass (lidar re-plan)'],
  [sv('<rect x="5" y="1" width="20" height="10" fill="rgba(255,77,61,.25)" stroke="' + COL.weed + '" stroke-width="1.4"/><path d="M7 11l8-10M13 11l8-10M19 11l6-7" stroke="' + COL.weed + '"/>'), 'Mapped weed patch (avoided)'],
  [sv('<rect x="8" y="1.5" width="14" height="9" rx="2" fill="' + COL.obst + '" stroke="#fff"/>'), 'Hidden obstacle, shown once detected'],
  [sv('<rect x="11" y="2" width="8" height="8" fill="#f3f6f0" stroke="#000"/>'), 'Drone service station'],
  [sv('<circle cx="15" cy="6" r="5" fill="rgba(255,224,102,.12)" stroke="' + COL.fx + '" stroke-dasharray="2 2"/>'), M.fx_name.charAt(0).toUpperCase() + M.fx_name.slice(1) + ' reach (40 ft) when parked'],
  [sv('<circle cx="15" cy="6" r="5" fill="none" stroke="' + COL.actual + '" stroke-dasharray="1.5 2"/>'), 'Lidar range ' + M.lidar_m + ' m'],
  [sv(DRONE_COL.slice(0, ND).map((c, i) => '<circle cx="' + (5 + i * 6.7) + '" cy="6" r="2.8" fill="' + c + '"/>').join('')), 'Drones D1-D' + ND + ' with fading trails'],
  [sv(CART ? '<rect x="2" y="1" width="26" height="10" fill="rgba(224,185,122,.25)"/>' + [4, 9, 14, 19, 24].map((x) => '<rect x="' + x + '" y="4" width="2.4" height="2.4" fill="#9be15d"/>').join('') : '<rect x="2" y="1" width="26" height="10" rx="2" fill="#b48a55"/><rect x="2" y="1" width="26" height="10" rx="2" fill="none" stroke="#e0b97a" stroke-width=".8"/>'),
    CART ? 'Seed cartridges placed on bed ridges' : 'Wood-chip mulch applied (density = dose)'],
];
$('lgBody').innerHTML = LEG.map((l) => l[0] + '<span>' + l[1] + '</span>').join('');
$('lgHead').onclick = () => { const lg = $('legend'); lg.classList.toggle('collapsed'); $('lgCaret').textContent = lg.classList.contains('collapsed') ? 'show' : 'hide'; };

// ================================================================== URL hash
function applyHash() {
  const h = new URLSearchParams(location.hash.replace(/^#/, ''));
  if (h.has('speed')) setSpeed(h.get('speed'));
  if (h.has('t')) seek(clamp(parseFloat(h.get('t')) || 0, 0, 1) * T_END);
  if (h.has('play')) setPlaying(h.get('play') !== '0');
  if (h.has('slowmo')) setSlowmo(h.get('slowmo') !== '0');
  if (h.has('loop')) { loop = h.get('loop') === '1'; $('bLoop').classList.toggle('on', loop); }
  if (h.has('follow')) {
    setFollow(h.get('follow') === '1');
    if (follow) { const ms = msPose(t, {}); view.cx = ms.x; view.cy = ms.y; if (view.ts !== null) { view.s = view.ts; view.ts = null; } }
  }
  if (h.has('z')) { view.s = clamp(parseFloat(h.get('z')) || view.fit, view.fit * 0.5, 60); view.ts = null; }
  if (h.has('legend') && h.get('legend') === '0') $('legend').classList.add('collapsed');
  dirty = true; panelDirty = true;
}
let lastHashW = 0, ownHash = location.hash;
window.addEventListener('hashchange', () => { ownHash = location.hash; applyHash(); renderAll(performance.now()); });
function writeHash(now) {
  // a hash edited by the user is applied by the (async) hashchange handler; never overwrite it before that
  if (playing || now - lastHashW < 800 || location.hash !== ownHash) return; lastHashW = now;
  const parts = ['t=' + (t / T_END).toFixed(5), 'play=0', 'follow=' + (follow ? 1 : 0)];
  if (!autoSpeed) parts.push('speed=' + baseSpeed);
  if (follow) parts.push('z=' + view.s.toFixed(1));
  if (!slowmo) parts.push('slowmo=0');
  if (loop) parts.push('loop=1');
  const h = '#' + parts.join('&');
  if (h !== ownHash) { try { history.replaceState(null, '', h); ownHash = location.hash; } catch (e) { /* file:// may refuse; the hash is optional */ } }
}

// ================================================================== main loop
function resize() {
  dpr = window.devicePixelRatio || 1;
  const r = mapC.parentElement.getBoundingClientRect(); const oldFit = view.fit;
  cw = Math.max(50, r.width); ch = Math.max(50, r.height); mapC.width = Math.round(cw * dpr); mapC.height = Math.round(ch * dpr);
  const b = M.site_box, pad = 10; view.fit = Math.min(cw / (b[2] - b[0] + 2 * pad), ch / (b[3] - b[1] + 2 * pad));
  if (!resize.done || Math.abs(view.s - oldFit) < 1e-6) fitView();
  resize.done = true;
  charts.forEach(buildChart); buildScrub(); const wh = setupCanvas(swC); swW = wh[0]; swH = wh[1];
  dirty = true; panelDirty = true;
}
// speed falls in proportion to the distance to the next slow-mo window, so playback never overshoots it
function targetSpeed(tt, base = baseSpeed, sm = slowmo) {
  if (!sm) return base;
  let tg = base;
  for (const w of M.slow) {
    if (tt >= w[0] && tt <= w[1]) return Math.min(M.slow_speed, base);
    if (tt < w[0]) tg = Math.min(tg, M.slow_speed + (w[0] - tt) / 0.35);
  }
  return tg;
}
function stepEff(eff, target, dt) { return target < eff ? target : Math.exp(Math.log(eff) + (Math.log(target) - Math.log(eff)) * (1 - Math.exp(-dt / 0.3))); }
// "auto" speed: the whole mission plays in AUTO_PLAY_S of wall time, slow-motion windows included
function playDuration(base) {
  let tt = 0, eff = base, n = 0; const dt = 1 / 30;
  while (tt < T_END && n < 60000) { eff = stepEff(eff, targetSpeed(tt, base, true), dt); tt += dt * eff; n++; }
  return n * dt;
}
const AUTO_SLOW = (() => {
  if (!M.slow.length) return M.auto_speed;
  let lo = M.auto_speed, hi = M.auto_speed * 8;
  if (playDuration(hi) > M.auto_play_s) return hi;
  for (let i = 0; i < 22; i++) { const mid = Math.sqrt(lo * hi); if (playDuration(mid) > M.auto_play_s) lo = mid; else hi = mid; }
  return hi;
})();
function autoBase() { return slowmo ? AUTO_SLOW : M.auto_speed; }
let lastNow = performance.now(), lastPanel = 0, lastSide = 0;
function renderAll(now) {
  render(now); charts.forEach(drawChartDyn); drawScrub(); drawSwim(); updatePanel(); lastPanel = now; panelDirty = false; dirty = false;
}
function frame(now) {
  try {
    const dt = clamp((now - lastNow) / 1000, 0, 0.1); lastNow = now;
    if (playing) {
      effSpeed = stepEff(effSpeed, targetSpeed(t), dt);
      t += dt * effSpeed;
      if (t >= T_END) { if (loop) t = 0; else { t = T_END; setPlaying(false); } }
      dirty = true;
    }
    const es = $('effSpd'), est = playing ? (effSpeed < baseSpeed * 0.7 ? 'slow-mo ' + nf(effSpeed) + '×' : nf(effSpeed) + '×') : 'paused';
    if (es._v !== est) { es._v = est; es.textContent = est; }
    if (follow) { const ms = msPose(t, {}); const a = 1 - Math.exp(-dt / 0.25); view.cx += (ms.x - view.cx) * a; view.cy += (ms.y - view.cy) * a; dirty = true; }
    if (view.ts !== null) { const a = 1 - Math.exp(-dt / 0.2); view.s += (view.ts - view.s) * a; if (Math.abs(view.s - view.ts) < 0.01) { view.s = view.ts; view.ts = null; } dirty = true; }
    render(now);
    if (playing || dirty || hoverT !== null || now - lastSide > 250) { charts.forEach(drawChartDyn); drawScrub(); drawSwim(); lastSide = now; }
    if (panelDirty || (playing && now - lastPanel > 66) || now - lastPanel > 500) { updatePanel(); lastPanel = now; panelDirty = false; }
    writeHash(now);
    dirty = false;
  } catch (err) { showErr('Frame error: ' + (err && err.stack ? err.stack.split('\n').slice(0, 3).join(' | ') : err)); playing = false; }
  requestAnimationFrame(frame);
}
function init() {
  setSpeed('auto'); effSpeed = baseSpeed;
  buildPanel(); resize();
  if (cw < 1000 || ch < 700) { $('legend').classList.add('collapsed'); $('lgCaret').textContent = 'show'; }
  applyHash(); renderAll(performance.now());
  window.addEventListener('resize', () => { resize(); renderAll(performance.now()); });
  window.__replayState = () => ({ t, tEnd: T_END, playing, follow, slowmo, loop, speed: baseSpeed, effSpeed, zoom: view.s });
  window.__replayReady = true;
  requestAnimationFrame(frame);
}
if (bmImg.complete) init(); else { bmImg.addEventListener('load', init); bmImg.addEventListener('error', init); }
})();
"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    import pickle

    ap = argparse.ArgumentParser(description="Export pickled missions as offline single-file HTML replays.")
    ap.add_argument("pickles", nargs="+", type=Path, help="pickled MissionResult files")
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/dev/html_replay"))
    args = ap.parse_args(argv)
    bm = None
    for pk in args.pickles:
        with open(pk, "rb") as f:
            res = pickle.load(f)
        t0 = time.time()
        if bm is None:
            bm = build_basemap(res.site)
        tb = time.time() - t0
        out = export_html_replay(res, args.out_dir / f"replay_{res.spec.mode.value}.html", basemap=bm)
        print(f"{out}  {out.stat().st_size / 1e6:.2f} MB  basemap {tb:.1f} s  export {time.time() - t0 - tb:.1f} s")


if __name__ == "__main__":
    main()
