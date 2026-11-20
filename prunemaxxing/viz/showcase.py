"""Legacy v1 dashboard for CoverageSimResult. Superseded by :mod:`prunemaxxing.viz.mission_viz`."""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch
from matplotlib.patches import Polygon as MplPolygon

from prunemaxxing.geo.polygons import MapLayers
from prunemaxxing.planning.simulator import CoverageSimResult
from prunemaxxing.viz.style import (
    COLORS,
    FONT_FAMILY,
    LABEL_SIZE,
    TITLE_SIZE,
    drone_path_colors,
)

plt.rcParams.update(
    {
        "font.family": FONT_FAMILY,
        "figure.facecolor": COLORS["bg"],
        "axes.facecolor": COLORS["panel"],
        "axes.edgecolor": COLORS["muted"],
        "axes.labelcolor": COLORS["ink"],
        "text.color": COLORS["ink"],
        "xtick.color": COLORS["muted"],
        "ytick.color": COLORS["muted"],
        "axes.titlecolor": COLORS["ink"],
    }
)


def _utm_extent(twin, margin_frac: float = 0.18) -> tuple[float, float, float, float]:
    """Field bounds with generous margin so the map reads larger / less cramped."""
    dem = twin.dem
    e0, e1 = dem.easting(0), dem.easting(dem.width)
    n0, n1 = dem.northing(dem.height), dem.northing(0)
    de = (e1 - e0) * margin_frac
    dn = (n1 - n0) * margin_frac
    return (e0 - de, e1 + de, n0 - dn, n1 + dn)


def _draw_terrain(ax, twin, completion: np.ndarray | None = None) -> None:
    dem = twin.dem
    extent = _utm_extent(twin)
    elev = dem.elevation
    ax.imshow(
        elev,
        origin="upper",
        extent=extent,
        cmap="gist_earth",
        alpha=0.55,
        aspect="equal",
        zorder=0,
    )
    if completion is not None:
        ax.imshow(
            completion,
            origin="upper",
            extent=extent,
            cmap="YlGn",
            alpha=0.45,
            vmin=0,
            vmax=1,
            aspect="equal",
            zorder=1,
        )
    # Weed overlay
    weed = np.array(
        [
            [1.0 if twin.get(r, c).weed_exclusion_flag else np.nan for c in range(twin.ncols)]
            for r in range(twin.nrows)
        ]
    )
    ax.imshow(
        weed,
        origin="upper",
        extent=extent,
        cmap="Reds",
        alpha=0.55,
        vmin=0,
        vmax=1,
        aspect="equal",
        zorder=2,
    )


def _draw_polygons(ax, layers: MapLayers | None) -> None:
    if layers is None:
        return
    for p in layers.parcels:
        ring = np.array(p.rings_utm[0])
        ax.add_patch(
            MplPolygon(
                ring,
                closed=True,
                fill=False,
                edgecolor=COLORS["canopy"],
                linewidth=1.2,
                linestyle="--",
                zorder=3,
            )
        )
    for w in layers.weeds:
        ring = np.array(w.rings_utm[0])
        ax.add_patch(
            MplPolygon(
                ring,
                closed=True,
                fill=True,
                facecolor=COLORS["weed"],
                edgecolor="white",
                alpha=0.35,
                linewidth=0.8,
                zorder=3,
            )
        )


def _kpi_series(result: CoverageSimResult) -> dict[str, np.ndarray]:
    biomass = np.array([f.biomass_t for f in result.frames], dtype=float)
    mean_c = np.array(
        [float(np.nanmean(f.completion)) for f in result.frames], dtype=float
    )
    steps = np.array([f.step for f in result.frames], dtype=float)
    return {"steps": steps, "biomass": biomass, "completion": mean_c}


def _progressive_len(n_pts: int, frame_i: int, n_frames: int) -> int:
    if n_pts < 2:
        return n_pts
    return max(2, int((frame_i + 1) / max(1, n_frames) * n_pts))


def _draw_drone_paths(
    ax,
    result: CoverageSimResult,
    frame_i: int | None = None,
    *,
    progressive: bool = True,
    lw_planned: float = 0.6,
    lw_actual: float = 1.35,
    alpha_planned: float = 0.30,
    alpha_actual: float = 0.85,
    legend: bool = False,
    zorder: int = 5,
) -> None:
    """Draw all drones' planned (dashed) and actual (solid) flight polylines."""
    n_frames = max(1, len(result.frames))
    planned_map = result.drone_planned or {}
    actual_map = result.drone_actual or {}
    ids = sorted(set(planned_map) | set(actual_map))
    # Draw planned first (underneath), then actuals in reverse so D0 isn't buried
    for di in ids:
        c_plan, _ = drone_path_colors(di)
        planned = planned_map.get(di)
        if planned is not None and len(planned) >= 2:
            ax.plot(
                planned[:, 0],
                planned[:, 1],
                color=c_plan,
                lw=lw_planned,
                ls="--",
                alpha=alpha_planned,
                label="Drone planned" if legend and di == ids[0] else None,
                zorder=zorder,
            )
    for di in reversed(ids):
        _, c_act = drone_path_colors(di)
        actual = actual_map.get(di)
        if actual is not None and len(actual) >= 2:
            if progressive and frame_i is not None:
                n = _progressive_len(len(actual), frame_i, n_frames)
                act = actual[:n]
            else:
                act = actual
            ax.plot(
                act[:, 0],
                act[:, 1],
                color=c_act,
                lw=lw_actual + 0.15 * di,
                ls="-",
                alpha=alpha_actual,
                label=f"D{di} actual" if legend else None,
                zorder=zorder + 1 + di,
            )


def render_dashboard_frame(
    result: CoverageSimResult,
    frame_i: int,
    layers: MapLayers | None,
    camera_imgs: dict[str, np.ndarray] | None,
    kpi: dict[str, np.ndarray],
    fig_size=(16, 9),
) -> np.ndarray:
    fr = result.frames[frame_i]
    twin = result.twin
    extent = _utm_extent(twin)

    # 1920x1080 @ dpi 120 with figsize 16x9 (macro-block friendly)
    fig = plt.figure(figsize=fig_size, dpi=120)
    gs = GridSpec(
        3,
        4,
        figure=fig,
        width_ratios=[1.15, 1.15, 0.95, 0.95],
        height_ratios=[1.0, 1.0, 0.85],
        hspace=0.28,
        wspace=0.28,
        left=0.04,
        right=0.98,
        top=0.90,
        bottom=0.06,
    )

    # Title banner
    fig.suptitle(
        "PRUNEMAXXING  ·  AGR-E2 Digital Twin  ·  Mothership + Wireless Swarm",
        fontsize=TITLE_SIZE + 2,
        fontweight="bold",
        color=COLORS["amber"],
        y=0.97,
    )
    fig.text(
        0.04,
        0.925,
        "Parc Agrari del Baix Llobregat  ·  Viladecans pilot  ·  no-till bed-safe UGV",
        fontsize=LABEL_SIZE,
        color=COLORS["muted"],
    )

    # --- Full map with terrain + trajectories ---
    ax_map = fig.add_subplot(gs[0:2, 0:2])
    _draw_terrain(ax_map, twin, fr.completion)
    _draw_polygons(ax_map, layers)
    if len(result.planned_polyline):
        ax_map.plot(
            result.planned_polyline[:, 0],
            result.planned_polyline[:, 1],
            color=COLORS["planned"],
            lw=1.6,
            ls="--",
            label="Planned CPP",
            zorder=5,
        )
    if len(result.actual_polyline):
        # progressive actual trail
        n_act = _progressive_len(len(result.actual_polyline), frame_i, len(result.frames))
        act = result.actual_polyline[:n_act]
        ax_map.plot(
            act[:, 0],
            act[:, 1],
            color=COLORS["actual"],
            lw=2.2,
            label="MS tracked",
            zorder=6,
        )
    _draw_drone_paths(ax_map, result, frame_i, progressive=True, legend=True)
    # Compact mothership glyph (full field is ~250 m — keep marker proportional)
    ax_map.scatter(
        [fr.mothership_e],
        [fr.mothership_n],
        s=18,
        c=COLORS["amber"],
        marker="s",
        zorder=10,
        edgecolors="white",
        linewidths=0.5,
        label="Mothership",
    )
    yaw = fr.mothership_yaw
    ax_map.plot(
        [fr.mothership_e, fr.mothership_e + 6 * np.cos(yaw)],
        [fr.mothership_n, fr.mothership_n + 6 * np.sin(yaw)],
        color=COLORS["amber"],
        lw=1.1,
        zorder=10,
    )
    if fr.drones:
        for di, d in enumerate(fr.drones):
            _, c_act = drone_path_colors(di)
            ax_map.scatter(
                [d[0]],
                [d[1]],
                s=28,
                c=c_act,
                marker="^",
                zorder=11,
                edgecolors="white",
                linewidths=0.4,
                label=None,
            )
            ax_map.plot(
                [fr.mothership_e, d[0]],
                [fr.mothership_n, d[1]],
                color=c_act,
                alpha=0.35,
                lw=0.9,
                zorder=7,
            )
    ax_map.set_xlim(extent[0], extent[1])
    ax_map.set_ylim(extent[2], extent[3])
    ax_map.set_title("Field map · terrain + biomass overlay", fontsize=TITLE_SIZE)
    ax_map.set_xlabel("Easting (m)")
    ax_map.set_ylabel("Northing (m)")
    ax_map.legend(loc="upper right", fontsize=7, framealpha=0.35, ncol=2)

    # --- Live minimap ---
    ax_mini = fig.add_subplot(gs[0, 2])
    _draw_terrain(ax_mini, twin, fr.completion)
    if len(result.planned_polyline):
        ax_mini.plot(
            result.planned_polyline[:, 0],
            result.planned_polyline[:, 1],
            color=COLORS["planned"],
            lw=1.0,
            ls="--",
        )
    if len(result.actual_polyline):
        n_act = _progressive_len(len(result.actual_polyline), frame_i, len(result.frames))
        ax_mini.plot(
            result.actual_polyline[:n_act, 0],
            result.actual_polyline[:n_act, 1],
            color=COLORS["actual"],
            lw=1.4,
        )
    _draw_drone_paths(
        ax_mini,
        result,
        frame_i,
        progressive=True,
        legend=False,
        lw_planned=0.7,
        lw_actual=1.0,
        alpha_planned=0.5,
        alpha_actual=0.8,
    )
    ax_mini.scatter(
        [fr.mothership_e], [fr.mothership_n], c=COLORS["amber"], s=14, marker="s", zorder=10
    )
    if fr.drones:
        for di, d in enumerate(fr.drones):
            _, c_act = drone_path_colors(di)
            ax_mini.scatter([d[0]], [d[1]], c=c_act, s=22, marker="^", zorder=11)
    # Wide live context (bigger local map)
    pad = 120
    ax_mini.set_xlim(fr.mothership_e - pad, fr.mothership_e + pad)
    ax_mini.set_ylim(fr.mothership_n - pad, fr.mothership_n + pad)
    ax_mini.set_title("Live minimap", fontsize=TITLE_SIZE - 2)
    ax_mini.set_xticks([])
    ax_mini.set_yticks([])
    for spine in ax_mini.spines.values():
        spine.set_color(COLORS["cyan"])
        spine.set_linewidth(2)

    # --- KPI moving charts ---
    ax_bio = fig.add_subplot(gs[0, 3])
    ax_bio.plot(kpi["steps"][: frame_i + 1], kpi["biomass"][: frame_i + 1], color=COLORS["amber"], lw=2)
    ax_bio.fill_between(
        kpi["steps"][: frame_i + 1],
        kpi["biomass"][: frame_i + 1],
        color=COLORS["amber"],
        alpha=0.25,
    )
    ax_bio.set_title("Biomass applied (t)", fontsize=TITLE_SIZE - 2)
    ax_bio.set_xlim(kpi["steps"][0], kpi["steps"][-1])
    ax_bio.set_ylim(0, max(kpi["biomass"][-1] * 1.1, 1))

    ax_cov = fig.add_subplot(gs[1, 3])
    ax_cov.plot(
        kpi["steps"][: frame_i + 1],
        100 * kpi["completion"][: frame_i + 1],
        color=COLORS["canopy"],
        lw=2,
    )
    ax_cov.set_title("Coverage completion (%)", fontsize=TITLE_SIZE - 2)
    ax_cov.set_xlim(kpi["steps"][0], kpi["steps"][-1])
    ax_cov.set_ylim(0, 105)

    # --- Camera panels ---
    cam_order = [
        ("hero_orbit", "Hero / orbit"),
        ("ms_front", "Mothership front cam"),
        ("ms_overview", "Mothership top cam"),
        ("d0_cam", "Drone-0 effector cam"),
    ]
    axes_cam = [
        fig.add_subplot(gs[1, 2]),
        fig.add_subplot(gs[2, 0]),
        fig.add_subplot(gs[2, 1]),
        fig.add_subplot(gs[2, 2]),
    ]
    for ax, (key, title) in zip(axes_cam, cam_order):
        ax.set_title(title, fontsize=TITLE_SIZE - 3)
        ax.set_xticks([])
        ax.set_yticks([])
        if camera_imgs and key in camera_imgs:
            ax.imshow(camera_imgs[key])
        else:
            ax.text(0.5, 0.5, "camera offline", ha="center", va="center", transform=ax.transAxes)
        for spine in ax.spines.values():
            spine.set_color(COLORS["amber"])

    # --- Stats strip ---
    ax_stats = fig.add_subplot(gs[2, 3])
    ax_stats.axis("off")
    k = result.kpis
    mode = result.allocator.payload_mode.value
    lines = [
        f"Step {fr.step}   ·   hopper {fr.hopper_kg:.0f} kg",
        f"Payload mode: {mode.upper()}",
    ]
    if mode == "cartridge":
        lines.append("Units: wood-chip + seed + mycelium")
    lines.extend(
        [
            f"Biomass: {fr.biomass_t:.1f} t",
            f"CO₂e est.: {k.co2e_sequestered_t * (fr.biomass_t / max(k.biomass_applied_t, 1e-6)):.1f} t",
            f"Bed spacing: {result.bed_spacing_m:.2f} m",
            f"Off-road compaction: {k.offroad_compaction_m2:.0f} m²",
            f"Cost vs REGENSOL: {k.vs_regensol_delta_eur_per_1000m2:.0f} €/1000m²",
        ]
    )
    ax_stats.add_patch(
        FancyBboxPatch(
            (0.02, 0.05),
            0.96,
            0.9,
            transform=ax_stats.transAxes,
            boxstyle="round,pad=0.04,rounding_size=0.08",
            facecolor="#243028",
            edgecolor=COLORS["amber"],
            linewidth=1.2,
        )
    )
    ax_stats.text(
        0.06,
        0.88,
        "LIVE TELEMETRY",
        transform=ax_stats.transAxes,
        fontsize=11,
        fontweight="bold",
        color=COLORS["amber"],
        va="top",
    )
    ax_stats.text(
        0.06,
        0.72,
        "\n".join(lines),
        transform=ax_stats.transAxes,
        fontsize=9,
        color=COLORS["ink"],
        va="top",
        family="monospace",
    )

    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    rgb = rgba[:, :, :3].copy()
    plt.close(fig)
    return rgb


def _pad_to_macroblock(frame: np.ndarray, block: int = 16) -> np.ndarray:
    h, w = frame.shape[:2]
    nh = int(np.ceil(h / block) * block)
    nw = int(np.ceil(w / block) * block)
    if nh == h and nw == w:
        return frame
    out = np.zeros((nh, nw, 3), dtype=frame.dtype)
    out[:h, :w] = frame
    return out


def write_video(frames: list[np.ndarray], path: Path, fps: int = 24) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    padded = [_pad_to_macroblock(f) for f in frames]
    imageio.mimsave(path, padded, fps=fps, codec="libx264", quality=8)
    return path


def render_showcase(
    result: CoverageSimResult,
    out_dir: str | Path,
    layers: MapLayers | None = None,
    mujoco_cams: dict[str, list[np.ndarray]] | None = None,
    max_frames: int = 120,
    fps: int = 24,
) -> dict[str, Path]:
    """Build dashboard MP4 + stills + camera clips."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}

    if not result.frames:
        raise ValueError("No simulation frames to visualize")

    # Prefer matching MuJoCo clip length for synced motion; cap matplotlib cost
    cam_len = 0
    if mujoco_cams:
        cam_len = max((len(v) for v in mujoco_cams.values()), default=0)
    n = min(max(max_frames, cam_len), max(len(result.frames), 1), 160)
    idxs = np.linspace(0, len(result.frames) - 1, num=n, dtype=int)
    kpi = _kpi_series(result)

    dash_frames: list[np.ndarray] = []
    for i, src_i in enumerate(idxs):
        cam_now = None
        if mujoco_cams:
            cam_now = {}
            # Map dashboard i → cam frame (clips may be longer via upsample)
            for k, seq in mujoco_cams.items():
                if not seq:
                    continue
                cam_i = int(i / max(1, n - 1) * (len(seq) - 1)) if n > 1 else 0
                cam_now[k] = seq[cam_i]
        dash_frames.append(
            render_dashboard_frame(result, int(src_i), layers, cam_now, kpi)
        )

    dash_path = out_dir / "prunemaxxing_dashboard.mp4"
    write_video(dash_frames, dash_path, fps=fps)
    outputs["dashboard_video"] = dash_path

    # Hero still
    still = dash_frames[len(dash_frames) // 2]
    still_path = out_dir / "dashboard_still.png"
    imageio.imwrite(still_path, still)
    outputs["dashboard_still"] = still_path

    # Camera-only clips
    if mujoco_cams:
        for name, seq in mujoco_cams.items():
            if len(seq) < 2:
                continue
            p = out_dir / f"cam_{name}.mp4"
            write_video(seq, p, fps=fps)
            outputs[f"cam_{name}"] = p
            imageio.imwrite(out_dir / f"cam_{name}_still.png", seq[len(seq) // 2])

    # Top-down trajectory poster
    fig, ax = plt.subplots(figsize=(10, 8), dpi=140)
    _draw_terrain(ax, result.twin, result.frames[-1].completion)
    _draw_polygons(ax, layers)
    if len(result.planned_polyline):
        ax.plot(
            result.planned_polyline[:, 0],
            result.planned_polyline[:, 1],
            color=COLORS["planned"],
            lw=2,
            ls="--",
            label="MS planned",
        )
    if len(result.actual_polyline):
        ax.plot(
            result.actual_polyline[:, 0],
            result.actual_polyline[:, 1],
            color=COLORS["actual"],
            lw=2.4,
            label="MS actual",
        )
    _draw_drone_paths(
        ax,
        result,
        frame_i=None,
        progressive=False,
        legend=True,
        lw_planned=0.85,
        lw_actual=1.2,
        alpha_planned=0.5,
        alpha_actual=0.7,
    )
    ax.set_title("Mothership + swarm planned vs actual paths · biomass completion")
    ax.legend(fontsize=8, ncol=2, framealpha=0.4)
    ax.set_aspect("equal")
    poster = out_dir / "map_poster.png"
    fig.savefig(poster, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    outputs["map_poster"] = poster

    # Cross-track error chart
    if result.track_samples:
        fig, ax = plt.subplots(figsize=(10, 3.5), dpi=140)
        t = [s.t for s in result.track_samples]
        ct = [s.cross_track_m for s in result.track_samples]
        ax.plot(t, ct, color=COLORS["cyan"], lw=1.8)
        ax.fill_between(t, ct, color=COLORS["cyan"], alpha=0.25)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Cross-track (m)")
        ax.set_title("Path-tracking cross-track error (bed-locked pure pursuit)")
        p = out_dir / "cross_track_chart.png"
        fig.savefig(p, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)
        outputs["cross_track"] = p

    return outputs
