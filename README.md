# PruneMaxxing (AGR-E2)

Autonomous mothership UGV + wireless drone swarm that spreads shredded urban/forest prunings
(wood-chip mulch, or seed + mycorrhiza cartridges) to regenerate soil in the Parc Agrari del
Baix Llobregat (Viladecans) — Unite! Tri-City Hackathon 2026.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

python scripts/demo_coverage.py                       # headless mission + KPIs (~3 s)
python scripts/build_world.py                         # MuJoCo world assets (~30 s once, then cached)
python scripts/render_showcase.py --payload dust      # full showcase -> outputs/showcase_dust/
python scripts/render_showcase.py --payload cartridge # -> outputs/showcase_cartridge/
python scripts/validate_rover_mujoco.py               # MuJoCo multibody closed-loop tracking check
pytest
```

Headless MuJoCo needs EGL (`MUJOCO_GL=egl`; the scripts also set the NVIDIA vendor file).
`--skip-mujoco` renders everything except the 3D views; `--preview` makes a short, cheap run.

## What the simulation does

1. **Site model** (`data/`): 12 ha DEM window with 4 parcels (9.1 ha), farm tracks, road,
   irrigation ditches, depot with chip stockpiles and mapped weed patches
   (`site_known.geojson`). Hidden ground truth the planner never sees:
   `unknown_obstacles.geojson` (broken-down tractor, fallen poplar, irrigation pump,
   waterlogged ground). Rendering-only scenery: `scenery.geojson`. Regenerate with
   `python scripts/build_site.py`.
2. **Global route** (`planning/route_planner.py`): lanes on furrow centres (0.85 m beds,
   59.5 m lane spacing so drones cover ±30 m), 6.5 m-radius headland U-turns, filleted
   bypasses around mapped weeds, transfers on the track network (networkx),
   depot → parcels → depot. 56 drone stations; every cell is assigned to its nearest station.
3. **Mission** (`planning/mission.py`):
   - Mothership: pure-pursuit controller drives an actuator-limited bicycle model
     (steer rate/lag, accel limits, slip noise). Commands and response are logged at 10 Hz;
     nothing snaps the state back onto the path.
   - 360° lidar (25 m; camera 20 m for flat hazards) discovers the hidden obstacles. A local
     planner then splices a bypass into the active path, so the actual route differs from
     the planned one.
   - Stop-and-go at stations. Drones (`planning/swarm.py`) fly timed sorties: load, climb,
     transit, apply, return, land. Patterns are a dust zig-zag or cartridge drops along a
     bed ridge. Actual tracks include wind gusts and controller lag. Battery swaps,
     hopper level and a 2-truck tender are tracked.
   - Near-field effectors on the mothership (`planning/ms_effectors.py`), with up to
     40 ft (12.19 m) reach from the parked machine. Dust mode uses a snow-blower style
     chip thrower: an impeller with a slewing chute and a deflector hood (1.2 kg/s,
     35° throw). Cartridge mode uses twin pneumatic air cannons: port and starboard
     turrets, two alternating barrels each, one shot per 0.8 s per turret, with
     ballistic flight and scatter that grows with range. At every station, patches
     that lie entirely inside the 3–12.19 m ring go to the mothership. The swarm works
     the rest in parallel, and both draw from the same hopper and tender logistics.
4. **Visuals** (`viz/`): orthophoto-style basemap, per-sortie mulch painting, map poster,
   tracking report, live dashboard video (time-warped), and MuJoCo 1:1 world replay
   (`sim/world_builder.py`, `sim/replay.py`) with a cinematic video and hero stills.

## Outputs (`outputs/showcase_<payload>/`)

| File | Content |
|------|---------|
| `dashboard.mp4` | live map (planned vs actual route, re-plans, drone sorties, mulch), MuJoCo chase + drone cams, minimap, charts, telemetry |
| `cinematic.mp4` | MuJoCo shots: depot departure, swarm launch, drone follow, drone effector close-up, 40 ft chip blower / air-cannon shot, furrow tracking, obstacle bypass, time-lapse |
| `map_poster.png` | whole-mission map + detour insets + station close-up + KPI card |
| `tracking_report.png` | cross-track error, steer/speed command vs actual, MuJoCo physics check |
| `hero_*.png` | MuJoCo stills (`hero_ms_effector.png`: chip blower or air cannons firing) |
| `control_log.csv` | mothership t, x, y, z, yaw, v, v_cmd, steer, steer_cmd, cte, s, odo, hopper, mode |
| `sorties.csv`, `events.json`, `detours.json`, `kpis.json` | swarm log, events, bypasses, KPIs |

## Headline numbers (4 drones + mothership effectors)

| | Dust (15 kg/m²) | Cartridge (1.2 kg/m²) |
|---|---|---|
| Area / material | 9.08 ha / 1,360 t | 9.08 ha / 109 t (≈218 k cartridges, 653 k seeds) |
| Mission time | 109 h (10.9 × 10 h shifts; 128 h without the blower) | 31 h (33 h without the cannons) |
| Drone sorties | 45,637 | 12,508 |
| Mothership effector (≤ 12.19 m) | chip blower: 229 t = 16.8 % in 4,745 bursts | air cannons: 10.2 t = 9.4 %, 20,492 cartridges in 1,375 volleys |
| Coverage | 100 % of cells at dose, CV 1.85 % | same |
| Lane cross-track RMS / max | 1.1 / 9.2 cm (100 % within ±20 cm) | same controller |
| Hidden obstacles | 4/4 detected, bypassed online (+202 m) | 4/4 |
| Cost (€/1000 m²) | 1,942 vs REGENSOL 2,408.77 (−467 €, 91 % of the overhead recovered) | 570 (528 of it cartridge material) |

MuJoCo multibody check (4 t Ackermann rover on the DEM heightfield, same controller):
0.8 cm RMS on settled lanes and 4.7 cm in headlands (`outputs/validation/`).

## Package layout

| Module | Role |
|--------|------|
| `prunemaxxing.geo` | DEM, CRS, cell twin, site model (`site.py`) |
| `prunemaxxing.planning` | `route_planner`, `path_geometry`, `vehicle`, `swarm`, `ms_effectors`, `mission`, `payload_modes` |
| `prunemaxxing.sim` | `world_frame` (1:1 frame + terrain), `world_builder` (MuJoCo world; `world_robots`, `world_effectors`, `world_props`, `world_textures`, `world_meshes`), `replay`, `rover_physics` |
| `prunemaxxing.viz` | `basemap`, `mission_viz` (poster, tracking report, dashboard) |
| `prunemaxxing.logistics`, `prunemaxxing.kpis` | macro supply/deficit, legacy KPI helpers |

Legacy v1 modules (`mothership_cpp`, `simulator`, `drone_allocator`, `path_tracker`,
`sim/mujoco_runner`, `viz/showcase`) are kept for reference only. The showcase no longer
uses them.

## Target field

WGS84 box `2.0255–2.0355°E`, `41.2825–41.2898°N` cropped from `viladecans_full_dem_10m.TIF`
(EPSG:32631, ~10 m). Pilot block: SW 12 ha window of the target field.
