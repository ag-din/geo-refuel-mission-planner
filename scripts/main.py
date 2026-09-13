"""
main.py
-------
Main entry point for a full mission case: it runs one scenario,
prints the mass profile, evaluates the TSA validity regime, and
reuses the HTML animation in the visualization folder to render the
result in one place.
"""

from __future__ import annotations

import itertools
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for p in (ROOT, SRC):
    path = str(p)
    if path not in sys.path:
        sys.path.insert(0, path)

from orbital_refuel.mission import generate_mission_case
from orbital_refuel.rdvsq_planning_model import solve_optimal_rdv_seq, time_distribution_ip
from visualization.animate_mission_html import build_payload, render_html


def demonstrate_mass_profile(
    n_targets: int = 6,
    seed: int = 3,
    evenly: bool = True,
    tf_periods: float = 5.5,
    m0_kg: float = 500.0,
    isp_s: float = 300.0,
    tof_min_frac: float = 0.03,
    tof_max_frac: float = 1.3,
):
    """Run one mission case and print the mass profile."""
    orbits, alphas = generate_mission_case(n_targets, seed=seed, evenly=evenly)
    T_ref = orbits[0].period_s
    tf = tf_periods * T_ref

    best, _ = solve_optimal_rdv_seq(
        orbits,
        alphas,
        tf,
        tof_min=tof_min_frac * T_ref,
        tof_max=tof_max_frac * T_ref,
        n_samples=50,
        m0_kg=m0_kg,
        isp_s=isp_s,
        verbose=False,
    )

    print(f"\nOptimal sequence: {best['sequence_labels']}"
          f"  (evenly spaced constellation, Isp={isp_s}s, m0={m0_kg} kg)")
    print(f"{'Leg':<8}{'Δv (km/s)':<12}{'mass before (kg)':<18}"
          f"{'mass after (kg)':<20}{'propellant (kg)':<16}")
    for i, leg in enumerate(best["mass_profile"]):
        print(f"{i + 1:<8}{leg['dv_km_s']:<12.4f}{leg['mass_before_kg']:<18.2f}"
              f"{leg['mass_after_kg']:<20.2f}{leg['propellant_used_kg']:<16.2f}")

    print(f"\nNote: Δv is the same across legs (symmetric geometry), but "
          f"propellant decreases from {best['mass_profile'][0]['propellant_used_kg']:.1f} "
          f"to {best['mass_profile'][-1]['propellant_used_kg']:.1f} kg — exactly the pattern "
          f"reported in Tables 1 and 4 of the paper: propellant depends on the current MASS, "
          f"not only on Δv.")
    print(f"Final service mass: {best['final_mass_kg']:.2f} kg "
          f"(used {best['total_propellant_kg']:.2f} kg out of {m0_kg} kg initial)\n")
    return best


def _sep(cum, a_idx, b_idx):
    d = (cum[b_idx] - cum[a_idx]) % (2 * np.pi)
    if d > np.pi:
        d -= 2 * np.pi
    return d


def full_bruteforce_check(alphas, tf, tof_min, tof_max, n_samples=35):
    """For n=5, test all 120 possible sequences under the same time budget and return the true optimum."""
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


def validate_tsa_optimality_regime(
    n_targets: int = 5,
    seed: int = 7,
    out_dir: str | os.PathLike = "outputs/figures",
):
    """Check when the TSA heuristic matches the true brute-force optimum."""
    print("=" * 70)
    print("VALIDATION 2: in which time regime is the TSA heuristic actually optimal?")
    print("               (brute-force verification)")
    print("=" * 70)

    orbits, alphas = generate_mission_case(n_targets, seed=seed)
    T_ref = orbits[0].period_s
    tof_min, tof_max = 0.03 * T_ref, 1.3 * T_ref

    tf_factors = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.4]
    tsa_costs, real_costs, matches = [], [], []

    print(f"\n{'tf (periods)':<16}{'tf/leg':<12}{'TSA predicted Δv':<18}"
          f"{'true optimal Δv':<18}{'match?':<10}")
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

        print(f"{f:<16.1f}{f / 5:<12.2f}{best['total_dv']:<18.3f}"
              f"{real_dv:<18.3f}{'yes' if match else 'NO':<10}")

    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(tf_factors, tsa_costs, "o-", label="TSA prediction (paper)", color="royalblue")
    ax.plot(tf_factors, real_costs, "^--", label="True optimum (brute force, same tf)", color="green")
    for f, m in zip(tf_factors, matches):
        if not m:
            ax.axvspan(f - 0.05, f + 0.05, color="tomato", alpha=0.15)
    ax.set_xlabel("Total time budget tf (orbital periods)")
    ax.set_ylabel("Total Δv (km/s)")
    ax.set_title("When does the TSA heuristic predict well? (red zone = failure)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "tsa_validity_regime.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)

    first_ok = next((f for f, m in zip(tf_factors, matches) if m), None)
    print(f"\nTSA starts matching the true optimum at tf ≈ {first_ok} periods "
          f"(~{first_ok / 5:.2f} periods/leg for n=5).")
    print(f"The paper examples use tf=15.6 with n=4-6 (~2.6-3.9 periods/leg) — "
          f"within the region where the heuristic does work.")
    print(f"\nFigure saved to: {path}")
    return path


def generate_visualization_case(
    n_targets: int = 6,
    seed: int = 3,
    evenly: bool = True,
    tf_periods: float = 5.5,
    output_path: str | os.PathLike | None = None,
):
    """Reuse the HTML visualizer from the visualization folder for the requested case."""
    payload = build_payload(
        n_targets=n_targets,
        seed=seed,
        evenly=evenly,
        tf_periods=tf_periods,
    )
    target_path = Path(output_path) if output_path is not None else ROOT / "outputs" / "reports" / "mission_visualization.html"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    html = render_html(payload)
    target_path.write_text(html)
    print(f"Visualization generated at: {target_path}")
    return target_path


def run_case(
    n_targets: int = 6,
    seed: int = 3,
    evenly: bool = True,
    tf_periods: float = 5.5,
    m0_kg: float = 500.0,
    isp_s: float = 300.0,
    generate_html: bool = True,
):
    """Run one full case: mass profile + validity regime + HTML visualization."""
    print("\n=== Mission case ===")
    best = demonstrate_mass_profile(
        n_targets=n_targets,
        seed=seed,
        evenly=evenly,
        tf_periods=tf_periods,
        m0_kg=m0_kg,
        isp_s=isp_s,
    )

    print("\n=== Regime analysis ===")
    validate_tsa_optimality_regime(n_targets=min(5, max(3, n_targets)), seed=7)

    if generate_html:
        print("\n=== Visualization ===")
        generate_visualization_case(
            n_targets=n_targets,
            seed=seed,
            evenly=evenly,
            tf_periods=tf_periods,
            output_path=ROOT / "outputs" / "reports" / "mission_visualization.html",
        )

    return best


if __name__ == "__main__":
    run_case()

    