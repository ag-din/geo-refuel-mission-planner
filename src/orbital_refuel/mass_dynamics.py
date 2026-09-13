"""
mass_dynamics.py
-----------------
Integración RIGUROSA de masa y la ecuación del cohete de Tsiolkovsky en
la misión 

  1. Se modelan DOS tanques físicamente distintos:
       - propelente de MANIOBRA (el que el propio servicio quema para
         cambiar de órbita — se agota vía Tsiolkovsky en cada impulso)
       - carga de COMBUSTIBLE A ENTREGAR (lo que se transfiere a cada
         satélite objetivo al llegar — no se quema, se entrega)

  2. La masa que entra en la exponencial de Tsiolkovsky en cada impulso
     es la masa INSTANTÁNEA TOTAL de la nave (seca + maniobra restante +
     CARGA QUE TODAVÍA LLEVA A BORDO) — la carga aún no entregada pesa y
     hay que acelerarla también, aunque no se "gaste" hasta el rendezvous.
     Ignorar esto (tratar solo la masa seca+maniobra) sería la
     simplificación que el usuario pidió evitar.

  3. FACTIBILIDAD DURA: si en algún tramo el propelente de maniobra
     requerido excede lo que queda en el tanque de maniobra, la misión
     se declara INFACTIBLE en ese punto — no se seguye computando como
     si nada (a diferencia de la v. anterior, que no tenía ningún tipo
     de chequeo de suficiencia de combustible).

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
    g0: float = 9.80665         # m/s^2

    @property
    def c0_km_s(self) -> float:
        """
        Effective exhaust velocity in km/s (Isp·g0 is in m/s).
        """
        return self.isp_s * self.g0 / 1000.0


@dataclass
class LegResult:
    leg_index: int
    dv_km_s: float
    mass_total_before_kg: float     # seca + maniobra + carga, ANTES del impulso
    propellant_required_kg: float   # lo que pide Tsiolkovsky para este Δv
    maneuver_remaining_before_kg: float
    maneuver_remaining_after_kg: float
    feasible: bool                  # ¿alcanzaba el propelente de maniobra?
    cargo_delivered_kg: float       # entregada DESPUÉS de este impulso (al llegar)
    cargo_remaining_after_kg: float


def simulate_mission_mass(dv_per_leg: list[float], cargo_delivered_per_leg: list[float],
                           budget: MassBudget) -> dict:
    """
    Simula la masa de la nave tramo a tramo, SIN simplificaciones:

      - Antes de cada impulso, la masa que "ve" Tsiolkovsky es la masa
        TOTAL instantánea (seca + maniobra restante + carga restante
        A BORDO, incluso la que todavía no se entregó).
      - El propelente que ese impulso exige se descuenta del tanque de
        MANIOBRA. Si no alcanza, la misión se declara infactible EN ESE
        TRAMO (no se sigue simulando más allá, tal como pasaría en la
        realidad: la nave se queda sin combustible ahí mismo).
      - Justo después del impulso (al llegar al objetivo del tramo), se
        entrega la carga correspondiente de ese tramo (si la hay),
        descontándola del tanque de carga — esto aliviana la nave para
        los tramos siguientes.

    len(cargo_delivered_per_leg) debe ser igual a len(dv_per_leg): la
    entrega de carga del tramo i ocurre al llegar al destino del tramo i.

    Devuelve un dict con: feasible (bool, misión completa sin quedarse
    sin combustible), fail_at_leg (índice del tramo donde falló, o None),
    legs (lista de LegResult), final_maneuver_kg, final_cargo_kg,
    final_total_mass_kg, total_propellant_used_kg.
    """
    assert len(dv_per_leg) == len(cargo_delivered_per_leg), \
        "cargo_delivered_per_leg debe tener una entrada por cada tramo de dv_per_leg"

    maneuver = budget.maneuver_prop_kg
    cargo = budget.cargo_prop_kg
    c0 = budget.c0_km_s

    legs: list[LegResult] = []
    feasible_overall = True
    fail_at_leg = None
    total_propellant_used = 0.0

    for i, (dv, cargo_out) in enumerate(zip(dv_per_leg, cargo_delivered_per_leg)):
        m_total_before = budget.dry_mass_kg + maneuver + cargo  # <- masa TOTAL, incluye carga aún a bordo
        propellant_needed = m_total_before * (1.0 - np.exp(-dv / c0))

        leg_feasible = propellant_needed <= maneuver + 1e-9

        if not leg_feasible:
            legs.append(LegResult(
                leg_index=i, dv_km_s=dv,
                mass_total_before_kg=m_total_before,
                propellant_required_kg=propellant_needed,
                maneuver_remaining_before_kg=maneuver,
                maneuver_remaining_after_kg=maneuver,  # no se ejecuta el impulso
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

        cargo_delivered = min(cargo_out, cargo)  # no se puede entregar más de lo que queda
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
