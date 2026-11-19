"""Scaffold notes for the MuJoCo mothership + tethered drone scene."""

MUJOCO_SCENE_NOTES = """
Phase 2 MuJoCo scene (not yet compiled):
  - Articulated UGV mothership with bulk hopper and docking pads
  - 4x multi-rotor drones (tethered cable constraint or free-flight + dock recharge)
  - Lift/grapple/spreader end-effectors for shredded woody residue
  - Path-tracking controllers following MothershipPath waypoints
  - Soft-body / particle approximation of chip payload transfer

Wire-in point: prunemaxxing.planning.simulator.run_coverage_simulation
provides the discrete twin loop; MuJoCo will track the same waypoint stream.
"""
