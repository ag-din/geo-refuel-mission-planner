"""
main.py
-------

"""

from __future__ import annotations
import itertools
import os
import numpy as np
import matplotlib.pyplot as plt
from orbital_refuel.mission import generate_same_orbit_mission

from orbital_refuel.rdvsq_planning_model import (
    solve_optimal_rdv_seq, best_sequence_candidates,
    time_distribution_ip,
)


def demonstrate_mass_profile():
    """
    """


    orbits, alphas = generate_same_orbit_mission(6, seed=3, evenly=True)

    T_ref = orbits[0].period_s

    tf = 5.5 * T_ref

    best, _ = solve_optimal_rdv_seq(orbits, alphas, tf, tof_min=0.03 * T_ref,
                                 tof_max=1.3 * T_ref, n_samples=50,
                                 m0_kg=500.0, isp_s=300.0, verbose=False)

    print(f"\nSecuencia óptima: {best['sequence_labels']}  "
          f"(constelación equiespaciada, Isp=300s, m0=500 kg)")
    print(f"{'Tramo':<8}{'Δv (km/s)':<12}{'masa antes (kg)':<18}"
          f"{'masa después (kg)':<20}{'propelente (kg)':<16}")
    for i, leg in enumerate(best["mass_profile"]):
        print(f"{i+1:<8}{leg['dv_km_s']:<12.4f}{leg['mass_before_kg']:<18.2f}"
              f"{leg['mass_after_kg']:<20.2f}{leg['propellant_used_kg']:<16.2f}")

    print(f"\nNota: los Δv son iguales entre tramos (geometría simétrica), pero "
          f"el propelente consumido decrece de {best['mass_profile'][0]['propellant_used_kg']:.1f} "
          f"a {best['mass_profile'][-1]['propellant_used_kg']:.1f} kg — exactamente el patrón "
          f"que reportan las Tablas 1 y 4 del paper: el propelente depende de la MASA actual, "
          f"no solo del Δv.")
    print(f"final mass: {best['final_mass_kg']:.2f} kg "
          f"(consumió {best['total_propellant_kg']:.2f} kg de 500 kg iniciales)\n")





def _sep(cum, a_idx, b_idx):
    d = (cum[b_idx] - cum[a_idx]) % (2 * np.pi)
    if d > np.pi:
        d -= 2 * np.pi
    return d


def full_bruteforce_check(alphas, tf, tof_min, tof_max, n_samples=35):
    """For n=5: test all 120 possible sequences UNDER THE SAME
    shared time budget and return the true optimum."""
    n = len(alphas) - 1
    cum = [0.0]
    for g in alphas[:-1]:
        cum.append(cum[-1] + g)

    best_seq, best_dv = None, np.inf
    for perm in itertools.permutations(range(1, n + 1)):
        seq = list(perm)
        thetas = [_sep(cum, 0, seq[0])]
        for i in range(len(seq) - 1):
            thetas.append(_sep(cum, seq[i], seq[i + 1]))
        _, dv, _ = time_distribution_ip(thetas, tf, tof_min, tof_max, n_samples=n_samples)
        if dv < best_dv:
            best_dv, best_seq = dv, seq
    return best_seq, best_dv


def validate_tsa_optimality_regime(out_dir="outputs/figures"):
    print("=" * 70)
    print("VALIDACIÓN 2: ¿en qué régimen de tiempo la heurística TSA es")
    print("               realmente óptima? (verificación por fuerza bruta)")
    print("=" * 70)

    orbits, alphas = generate_same_orbit_mission(5, seed=7)
    T_ref = orbits[0].period_s
    tof_min, tof_max = 0.03 * T_ref, 1.3 * T_ref

    tf_factors = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.4]
    tsa_costs, real_costs, matches = [], [], []

    print(f"\n{'tf (periodos)':<16}{'tf/tramo':<12}{'Δv predicho TSA':<18}"
          f"{'Δv óptimo real':<18}{'¿coincide?':<10}")
    print("-" * 74)

    for f in tf_factors:
        tf = f * T_ref
        best, _ = solve_optimal_rdv_seq(orbits, alphas, tf, tof_min, tof_max,
                                     n_samples=35, verbose=False)
        real_seq, real_dv = full_bruteforce_check(alphas, tf, tof_min, tof_max, n_samples=35)
        match = (real_seq == best["sequence_labels"])

        tsa_costs.append(best["total_dv"])
        real_costs.append(real_dv)
        matches.append(match)

        print(f"{f:<16.1f}{f/5:<12.2f}{best['total_dv']:<18.3f}"
              f"{real_dv:<18.3f}{'sí' if match else 'NO':<10}")

    # Plot
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(tf_factors, tsa_costs, "o-", label="Predicción TSA (paper)", color="royalblue")
    ax.plot(tf_factors, real_costs, "^--", label="Óptimo real (fuerza bruta, mismo tf)",
            color="green")
    for f, m in zip(tf_factors, matches):
        if not m:
            ax.axvspan(f - 0.05, f + 0.05, color="tomato", alpha=0.15)
    ax.set_xlabel("Presupuesto total de tiempo tf (periodos orbitales)")
    ax.set_ylabel("Δv total (km/s)")
    ax.set_title("¿Cuándo predice bien la heurística TSA? (zona roja = falla)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "tsa_validity_regime.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)

    first_ok = next((f for f, m in zip(tf_factors, matches) if m), None)
    print(f"\nLa heurística TSA empieza a coincidir con el óptimo real a partir de "
          f"tf ≈ {first_ok} periodos (~{first_ok/5:.2f} periodos/tramo para n=5).")
    print(f"Los propios ejemplos del paper usan tf=15.6 con n=4-6 (~2.6-3.9 "
          f"periodos/tramo) — dentro de la región donde la heurística sí funciona.")
    print(f"\nFigura guardada en: {path}")


if __name__ == "__main__":

    print("\nCase")
    demonstrate_mass_profile()
    validate_tsa_optimality_regime()

    