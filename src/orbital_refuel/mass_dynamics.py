"""
mass_dynamics.py
-----------------

Implements the mass dynamics of a spacecraft during a mission
with multiple rendezvous and refueling operations.

It is modeled with two physically distinct tanks:
    - MANEUVER propellant (the one the service itself burns to change orbit —
        it is depleted via Tsiolkovsky's equation at each impulse)
    - CARGO propellant (the one that is transferred to each target satellite
        upon arrival — it is not burned, it is delivered)

"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class MassBudget:
    """
    Represents the mass budget of a spacecraft for a mission,
    including dry mass, maneuver propellant, cargo propellant,
    and specific impulse.

    Arguments
    ---------
    dry_mass_kg : float
        The dry mass of the spacecraft in kilograms (structural mass).
    maneuver_prop_kg : float
        The amount of propellant available for maneuvers in kilograms.
    cargo_prop_kg : float
        The total amount of propellant to be delivered as cargo in kilograms.
    isp_s : float
        The specific impulse of the spacecraft's engine in seconds.
    g0 : float, optional
        The standard gravity in m/s^2 (default is 9.80665 m/s^2).
    """
    dry_mass_kg: float
    maneuver_prop_kg: float
    cargo_prop_kg: float
    isp_s: float
    g0: float = 9.80665  # m/s^2

    @property
    def c0_km_s(self) -> float:
        """
        Effective exhaust velocity in km/s (Isp·g0 is in m/s).
        """
        return self.isp_s * self.g0 / 1000.0


@dataclass
class LegResult:
    """
    Represents the result of a single leg of a mission.

    Arguments
    ---------
    leg_index : int
        The index of the leg in the mission sequence.
    dv_km_s : float
        The delta-v required for this leg in km/s.
    mass_total_before_kg : float
        The total mass of the spacecraft before the maneuver (dry + maneuver + cargo) in kilograms      
    propellant_required_kg : float
        The amount of propellant required for this leg according to Tsiolkovsky's equation
    maneuver_remaining_before_kg : float
        The amount of maneuver propellant remaining before this leg in kilograms
    maneuver_remaining_after_kg : float
        The amount of maneuver propellant remaining after this leg in kilograms
    feasible : bool
        Whether the leg is feasible (i.e., enough maneuver propellant is available)
    cargo_delivered_kg : float
        The amount of cargo propellant delivered at the end of this leg in kilograms
    cargo_remaining_after_kg : float
        The amount of cargo propellant remaining after this leg in kilograms

    """
    leg_index: int
    dv_km_s: float
    mass_total_before_kg: float 
    propellant_required_kg: float
    maneuver_remaining_before_kg: float
    maneuver_remaining_after_kg: float
    feasible: bool
    cargo_delivered_kg: float
    cargo_remaining_after_kg: float


def simulate_mission_mass(dv_per_leg: list[float], cargo_delivered_per_leg: list[float],
                           budget: MassBudget) -> dict:
    """
    Simulates the mass of the spacecraft leg by leg.

    Parameters
    ----------
    dv_per_leg : list of float
        List of delta-v values (in km/s) for each leg of the mission.
    cargo_delivered_per_leg : list of float
        List of cargo propellant amounts (in kg) to be delivered at the end of each
        leg. Must have the same length as dv_per_leg.
    budget : MassBudget
        The mass budget of the spacecraft, including dry mass, maneuver propellant,
        cargo propellant, and specific impulse.

    Returns
    -------
    dict
        A dictionary containing:
        - 'feasible': bool, whether the entire mission is feasible
        - 'fail_at_leg': int or None, the index of the leg where feasibility failed
            (None if the mission is feasible)
        - 'legs': list of LegResult, the results for each leg
        - 'final_maneuver_kg': float, remaining maneuver propellant after the mission
        - 'final_cargo_kg': float, remaining cargo propellant after the mission
        - 'final_total_mass_kg': float, total mass of the spacecraft after the mission
        - 'total_propellant_used_kg': float, total maneuver propellant used during the
            mission
    """
    assert len(dv_per_leg) == len(cargo_delivered_per_leg), \
        "cargo_delivered_per_leg must have an entry for each leg in dv_per_leg"

    maneuver = budget.maneuver_prop_kg
    cargo = budget.cargo_prop_kg
    c0 = budget.c0_km_s

    legs: list[LegResult] = []
    feasible_overall = True
    fail_at_leg = None
    total_propellant_used = 0.0

    for i, (dv, cargo_out) in enumerate(zip(dv_per_leg, cargo_delivered_per_leg)):
        m_total_before = budget.dry_mass_kg + maneuver + cargo
        propellant_needed = m_total_before * (1.0 - np.exp(-dv / c0))

        leg_feasible = propellant_needed <= maneuver + 1e-9

        if not leg_feasible:
            legs.append(LegResult(
                leg_index=i, dv_km_s=dv,
                mass_total_before_kg=m_total_before,
                propellant_required_kg=propellant_needed,
                maneuver_remaining_before_kg=maneuver,
                maneuver_remaining_after_kg=maneuver,
                feasible=False,
                cargo_delivered_kg=0.0,
                cargo_remaining_after_kg=cargo,
            ))
            feasible_overall = False
            fail_at_leg = i
            break

        maneuver_before = maneuver
        maneuver -= propellant_needed
        total_propellant_used += propellant_needed

        cargo_delivered = min(cargo_out, cargo)
        cargo -= cargo_delivered

        legs.append(LegResult(
            leg_index=i, dv_km_s=dv,
            mass_total_before_kg=m_total_before,
            propellant_required_kg=propellant_needed,
            maneuver_remaining_before_kg=maneuver_before,
            maneuver_remaining_after_kg=maneuver,
            feasible=True,
            cargo_delivered_kg=cargo_delivered,
            cargo_remaining_after_kg=cargo,
        ))

    final_total_mass = budget.dry_mass_kg + maneuver + cargo

    return {
        "feasible": feasible_overall,
        "fail_at_leg": fail_at_leg,
        "legs": legs,
        "final_maneuver_kg": maneuver,
        "final_cargo_kg": cargo,
        "final_total_mass_kg": final_total_mass,
        "total_propellant_used_kg": total_propellant_used,
    }
