"""MuJoCo scene entrypoints."""

from pathlib import Path

MUJOCO_XML = Path(__file__).with_name("mothership_swarm.xml")

MUJOCO_SCENE_NOTES = """
PruneMaxxing MuJoCo scene (mothership_swarm.xml):
  - Amber UGV mothership with hopper + 4 wireless docking/charge pads (no tethers)
  - 4 heavy-lift drones with sliding lift/grapple effectors
  - Dual payload visuals: wood-dust scoop OR seed+mycelium cartridges
  - Deposit FX: local spray ellipsoids + 20 dust particles / 8 cartridge drops
  - DEM heightfield from viladecans_full_dem_10m.TIF (6 ha pilot), relief exaggerated
  - Raised-bed / furrow overlays (+ subtle bake into heightfield)
  - Cameras: hero_orbit, top_tactical, ms_front, ms_overview, ms_rear_dock, d*_cam
  - Regenerate terrain: python scripts/build_heightfield.py
"""
