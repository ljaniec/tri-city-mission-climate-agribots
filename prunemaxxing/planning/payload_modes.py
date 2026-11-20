"""Dual payload modes for drone lift effectors."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class PayloadMode(str, Enum):
    DUST = "dust"  # shredded wood shavings / fine residue
    CARTRIDGE = "cartridge"  # formed wood-chip cartridges + seed + mycelium


@dataclass(frozen=True)
class PayloadSpec:
    mode: PayloadMode
    unit_mass_kg: float  # payload per sortie
    description: str
    rgba: tuple[float, float, float, float]
    contains_seed: bool = False
    contains_mycelium: bool = False
    local_tree_species: tuple[str, ...] = ()
    dose_kg_per_m2: float = 15.0
    load_time_s: float = 5.0
    apply_speed_mps: float = 2.0
    apply_alt_agl_m: float = 3.0
    units_per_sortie: int = 1  # cartridges per magazine (1 = bulk spreader)
    seeds_per_unit: int = 0


DUST_SPEC = PayloadSpec(
    mode=PayloadMode.DUST,
    unit_mass_kg=25.0,
    description="Shredded woody prunings / wood dust (<5 cm, C/N~40), spinner-disc spreader",
    rgba=(0.55, 0.40, 0.22, 1.0),
    dose_kg_per_m2=15.0,
    load_time_s=5.0,
    apply_speed_mps=2.0,
    apply_alt_agl_m=3.0,
)

CARTRIDGE_SPEC = PayloadSpec(
    mode=PayloadMode.CARTRIDGE,
    unit_mass_kg=8.0,
    description="Pressed wood-chip cartridges (0.5 kg) with local tree seed + mycorrhizal mycelium",
    rgba=(0.35, 0.55, 0.28, 1.0),
    contains_seed=True,
    contains_mycelium=True,
    local_tree_species=("Quercus_ilex", "Pinus_halepensis", "Ceratonia_siliqua"),
    dose_kg_per_m2=1.2,
    load_time_s=6.0,
    apply_speed_mps=1.5,
    apply_alt_agl_m=2.2,
    units_per_sortie=16,
    seeds_per_unit=3,
)


def get_payload_spec(mode: PayloadMode | str) -> PayloadSpec:
    m = PayloadMode(mode)
    return CARTRIDGE_SPEC if m is PayloadMode.CARTRIDGE else DUST_SPEC
