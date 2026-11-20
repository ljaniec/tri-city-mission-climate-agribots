"""Climate / soil / air-quality impact of a mission, with every factor traceable.

The counterfactual for Viladecans prunings is pile burning (or landfill). Spreading
them as mulch / cartridges instead:

* stores soil carbon (REGENSOL trial figure, ``CO2E_T_PER_HA``),
* retains nitrogen (``N_RETAINED_KG_PER_HA``),
* avoids the non-CO2 greenhouse gases and smoke of open burning (IPCC Tier 1
  agricultural-residue emission factors; Andreae & Merlet PM2.5),
* adds a little plant-available water storage via soil organic carbon
  (Minasny & McBratney 2018 meta-analysis — honest, small),
* replaces tractor transport + spreading in the REGENSOL cost structure.

Biogenic CO2 released by burning is shown for context only and never added to the
soil-carbon figure (no double counting).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from prunemaxxing.config import (
    CO2E_T_PER_HA,
    CONVENTIONAL_COST_EUR_PER_1000M2,
    INCORPORATION_DEPTH_CM,
    N_RETAINED_KG_PER_HA,
    REGENSOL_COST_EUR_PER_1000M2,
    SOC_INCREASE_PP,
)


@dataclass(frozen=True)
class Factor:
    value: float
    unit: str
    label: str
    source: str
    assumed: bool = False


FACTORS: dict[str, Factor] = {
    "dry_matter_frac": Factor(0.60, "kg DM / kg fresh", "dry matter of fresh chipped prunings", "assumption (40 % moisture)", True),
    "combustion_frac": Factor(0.85, "-", "fraction of a pruning pile actually combusted",
                              "IPCC 2006 V4 Table 2.6 range 0.8-0.9 for crop residues", True),
    "ef_ch4": Factor(2.7, "g / kg DM", "CH4 from open burning", "IPCC 2006 V4 Table 2.5, agricultural residues"),
    "ef_n2o": Factor(0.07, "g / kg DM", "N2O from open burning", "IPCC 2006 V4 Table 2.5, agricultural residues"),
    "ef_co": Factor(92.0, "g / kg DM", "CO from open burning", "IPCC 2006 V4 Table 2.5, agricultural residues"),
    "ef_nox": Factor(2.5, "g / kg DM", "NOx from open burning", "IPCC 2006 V4 Table 2.5, agricultural residues"),
    "ef_co2": Factor(1515.0, "g / kg DM", "biogenic CO2 from open burning (context only)",
                     "IPCC 2006 V4 Table 2.5, agricultural residues"),
    "ef_pm25": Factor(3.9, "g / kg DM", "PM2.5 from open burning", "Andreae & Merlet 2001, agricultural residues"),
    "gwp_ch4": Factor(27.0, "kg CO2e / kg", "GWP100 of non-fossil CH4", "IPCC AR6 WG1 Table 7.15"),
    "gwp_n2o": Factor(273.0, "kg CO2e / kg", "GWP100 of N2O", "IPCC AR6 WG1 Table 7.15"),
    "soil_co2e": Factor(CO2E_T_PER_HA, "t CO2e / ha", "soil carbon stored at 15 kg/m2 mulch",
                        "REGENSOL trials (AGR-E2 brief), scaled by dose"),
    "n_retained": Factor(N_RETAINED_KG_PER_HA, "kg N / ha", "nitrogen retained at 15 kg/m2",
                         "REGENSOL trials (AGR-E2 brief), scaled by dose"),
    "soc_increase": Factor(SOC_INCREASE_PP, "pp SOC", "soil organic carbon increase at 15 kg/m2",
                           "REGENSOL trials (AGR-E2 brief), scaled by dose"),
    "awc_per_soc": Factor(1.16, "mm / 100 mm soil per pp SOC", "plant-available water per 1 pp SOC",
                          "Minasny & McBratney 2018, Eur. J. Soil Sci. 69:39"),
    "depth_cm": Factor(INCORPORATION_DEPTH_CM, "cm", "incorporation depth", "AGR-E2 brief"),
}


def factor_table() -> list[dict[str, Any]]:
    return [{"key": k, "value": f.value, "unit": f.unit, "label": f.label, "source": f.source, "assumed": f.assumed}
            for k, f in FACTORS.items()]


def impact_from_material(material_t: float, area_ha: float, dose_kg_per_m2: float) -> dict[str, float]:
    """Impact of spreading ``material_t`` fresh tonnes over ``area_ha`` at the given dose."""
    F = {k: f.value for k, f in FACTORS.items()}
    dose_scale = dose_kg_per_m2 / 15.0
    burned_dm_t = material_t * F["dry_matter_frac"] * F["combustion_frac"]
    ch4_t = burned_dm_t * F["ef_ch4"] / 1000.0
    n2o_t = burned_dm_t * F["ef_n2o"] / 1000.0
    soc_pp = F["soc_increase"] * dose_scale
    awc_mm = soc_pp * F["awc_per_soc"] * F["depth_cm"] / 10.0
    return {
        "material_t": material_t,
        "area_ha": area_ha,
        "soil_co2e_t": area_ha * F["soil_co2e"] * dose_scale,
        "n_retained_kg": area_ha * F["n_retained"] * dose_scale,
        "burn_avoided_dm_t": burned_dm_t,
        "burn_ghg_avoided_co2e_t": ch4_t * F["gwp_ch4"] + n2o_t * F["gwp_n2o"],
        "burn_ch4_avoided_kg": ch4_t * 1000.0,
        "burn_n2o_avoided_kg": n2o_t * 1000.0,
        "pm25_avoided_kg": burned_dm_t * F["ef_pm25"],
        "co_avoided_kg": burned_dm_t * F["ef_co"],
        "nox_avoided_kg": burned_dm_t * F["ef_nox"],
        "biogenic_co2_not_released_t": burned_dm_t * F["ef_co2"] / 1000.0,
        "soc_increase_pp": soc_pp,
        "awc_gain_mm": awc_mm,
        "awc_gain_m3": awc_mm * 10.0 * area_ha,
    }


def impact_summary(res) -> dict[str, Any]:
    """Whole-mission impact + economics for a :class:`MissionResult`.

    ``area_treated_ha`` is the planned field area, so it is scaled by the share of
    the target dose actually applied (1.0 for a finished mission).
    """
    k = res.kpis
    done = min(1.0, float(res.deposits["kg"].sum()) / max(float(res.cell_target_kg.sum()), 1e-9))
    area_ha = float(k["area_treated_ha"]) * done
    out: dict[str, Any] = impact_from_material(float(k["biomass_applied_t"]), area_ha, float(k["dose_kg_per_m2"]))
    out["total_co2e_t"] = out["soil_co2e_t"] + out["burn_ghg_avoided_co2e_t"]
    out["co2e_t_per_ha"] = out["total_co2e_t"] / max(area_ha, 1e-9)
    out["payload_mode"] = k["payload_mode"]
    out["mission_time_h"] = float(k["mission_time_h"])
    out["cost_eur_per_1000m2"] = float(k["cost_eur_per_1000m2"])
    out["regensol_eur_per_1000m2"] = REGENSOL_COST_EUR_PER_1000M2
    out["conventional_eur_per_1000m2"] = CONVENTIONAL_COST_EUR_PER_1000M2
    out["saving_eur_per_1000m2"] = k.get("saving_vs_regensol_eur_per_1000m2")
    out["saving_eur_total"] = (None if out["saving_eur_per_1000m2"] is None
                               else out["saving_eur_per_1000m2"] * area_ha * 10.0)
    out["drone_energy_kwh"] = float(k.get("drone_energy_kwh", 0.0))
    out["drone_energy_kwh_per_t"] = out["drone_energy_kwh"] / max(out["material_t"], 1e-9)
    out["bed_compaction_pct"] = float(k.get("bed_compaction_pct_of_area", 0.0))
    out["cartridges_placed"] = k.get("cartridges_placed")
    out["seeds_placed"] = k.get("seeds_placed")
    out["species"] = k.get("species", [])
    return out
