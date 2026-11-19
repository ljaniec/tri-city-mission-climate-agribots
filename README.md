# PruneMaxxing (AGR-E2)

Autonomous mothership + drone swarm for applying shredded urban/forest prunings to regenerate soil in the Parc Agrari del Baix Llobregat (Unite! Tri-City Hackathon 2026).

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python scripts/demo_coverage.py
pytest
```

## Package layout

| Module | Role |
|--------|------|
| `prunemaxxing.geo` | DEM load (UTM 31N), CRS, cell-state Digital Twin |
| `prunemaxxing.planning` | Mothership CPP, drone task allocation, coverage sim |
| `prunemaxxing.logistics` | Macro supply/deficit routing scaffold |
| `prunemaxxing.kpis` | AGR-E2 agronomic / robotics / economic KPIs |
| `prunemaxxing.sim` | MuJoCo scene scaffold (phase 2) |

## Target field

WGS84 box `2.0255–2.0355°E`, `41.2825–41.2898°N` cropped from `viladecans_full_dem_10m.TIF` (EPSG:32631, ~10 m). Default pilot window: **6 ha** @ **15 kg/m²** (150 t/ha).
