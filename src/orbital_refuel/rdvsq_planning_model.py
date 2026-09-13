"""
rdvsq_planning_model.py
----------------------

Direct implementation of the model of:

  Shen, H., Tsiotras, P. (2002). "Optimal Scheduling for Servicing
  Multiple Satellites in a Circular Constellation." AIAA/AAS
  Astrodynamics Specialist Conference.

    The problem is solved in two
    steps:

  1. OPTIMAL SEQUENCE PROBLEM

  2. OPTIMAL TIME DISTRIBUTION PROBLEM

"""

from __future__ import annotations
import warnings
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds, minimize_scalar

from .keplerian import CircularOrbit, MU_EARTH, R_EARTH
from .transfer import transfer_delta_v
from .lambert import find_all_lambert_solutions, solve_lambert

GEO_RADIUS_KM = R_EARTH + 35786.0


def best_sequence_candidates(alphas: list[float]) -> list[list[int]]:
    """
    Identifies the candidate sequences with minimum total sweep angle (TSA).

    Parameters
    ----------
    alphas : list of float
        List of angular gaps (in radians) between consecutive satellites,
        including the gap between the last target and the depot.
    
    Returns
    -------
    list of list of int
        List of candidate sequences (as lists of indices) that achieve the
        minimum TSA. Each sequence is a permutation of the indices 1..n
        (the depot is always index 0 and is not included in the sequence).

    """
    a = np.array(alphas, dtype=float)
    n = len(a) - 1

    term1 = np.array([a[i] - a[:i].sum() for i in range(n + 1)])
    term2 = np.array([a[i] - a[i + 1:].sum() for i in range(n + 1)])

    A1, k1 = term1.max(), int(term1.argmax())
    A2, k2 = term2.max(), int(term2.argmax())
    A = max(A1, A2)

    candidates = []
    if abs(A1 - A) < 1e-9 and 0 <= k1 < n:
        candidates.append(list(range(1, k1 + 1)) + list(range(n, k1, -1)))
    if abs(A2 - A) < 1e-9 and 0 <= k2 <= n:
        candidates.append(list(range(n, k2, -1)) + list(range(1, k2 + 1)))

    unique = []
    for s in candidates:
        if s not in unique:
            unique.append(s)
    return unique


def total_sweep_angle(alphas: list[float], sequence: list[int]) -> float:
    """
    Computes the total sweep angle (TSA) for a given sequence of satellites.

    Parameters
    ----------
    alphas : list of float
        List of angular gaps (in radians) between consecutive satellites,
        including the gap between the last target and the depot.
    sequence : list of int
        List of satellite indices in the order they are visited.

    Returns
    -------
    float
        The total sweep angle (TSA) for the given sequence.
    """
    n = len(alphas) - 1
    
    cum = [0.0]
    for g in alphas[:-1]:
        cum.append(cum[-1] + g)
    

    def sep(u_a, u_b):
        """
        Computes the angular separation between two angles u_a and u_b,
        taking into account the circular nature of the orbit.

        Parameters
        ----------
        u_a : float
            Angle of the first satellite (in radians).
        u_b : float
            Angle of the second satellite (in radians).
        
        Returns
        -------
        float
            The angular separation between u_a and u_b (in radians).
        """
        d = (u_b - u_a) % (2 * np.pi)
        if d > np.pi:
            d -= 2 * np.pi
        return abs(d)

    tsa = sep(cum[0], cum[sequence[0]])
    for i in range(len(sequence) - 1):
        tsa += sep(cum[sequence[i]], cum[sequence[i + 1]])
    return tsa


# ---------------------------------------------------------------------
# Mass and propellant (eqs. 1-2 of Section III of Tsiotras & de Nailly,
# 2005 — "Optimal Single Service Vehicle Refueling")
# ---------------------------------------------------------------------
def mass_consumption_profile(dv_per_leg: list[float], m0: float,
                              isp_s: float, g0: float = 9.80665):
    """
    Given the list of leg Δv (km/s) and the initial service mass m0 (kg),
    apply the rocket equation leg by leg:

        m(t_k^+) = m(t_k^-) * exp(-ΔV_k / (Isp * g0))     (eq. 1 of the paper)

    Parameters
    ----------
    dv_per_leg : list of float
        List of Δv (km/s) for each leg of the mission.
    m0 : float
        Initial mass of the service satellite (kg).
    isp_s : float
        Specific impulse of the propulsion system (seconds).
    g0 : float
        Standard gravity (m/s^2). Default is 9.80665 m/s^2.
    
    Returns
    -------
    list of dict
        List of dictionaries containing the mass profile for each leg:
        - 'dv_km_s': Δv for the leg (km/s)
        - 'mass_before_kg': mass before the leg (kg)
        - 'mass_after_kg': mass after the leg (kg)
        - 'propellant_used_kg': propellant used for the leg (kg)
    """
    c0 = isp_s * g0 / 1000.0  # km/s
    m = m0
    profile = []
    for dv in dv_per_leg:
        m_before = m
        m_after = m_before * np.exp(-dv / c0)
        propellant_used = m_before - m_after
        profile.append({
            "dv_km_s": dv,
            "mass_before_kg": m_before,
            "mass_after_kg": m_after,
            "propellant_used_kg": propellant_used,
        })
        m = m_after
    return profile


# ---------------------------------------------------------------------
# Step 2: optimal time distribution problem
# ---------------------------------------------------------------------
def cost_curve(theta: float, tof_min: float, tof_max: float, n_samples: int,
               r_km: float = GEO_RADIUS_KM, mu: float = MU_EARTH, max_M: int = 3):
    """Samples Cθ(t): Δv (km/s) as a function of flight time t for
    two bodies in the SAME circular orbit separated by angle theta (rad) —
    but now using the REAL cost curve with "coasting", not only Lambert
    zero-revolution transfer:

      1. For each tof on the grid, all Lambert solutions from 0 to max_M
         revolutions (lambert.find_all_lambert_solutions) are tested and the
         cheapest one is selected — this reproduces the "without coasting"
         curve (dotted line in Fig. 1 of the Tsiotras & de Nailly paper),
         which is no longer a single branch but several intertwined families.
      2. The running MINIMUM is applied to that curve: the cost at time t is
         the minimum achievable using any tof <= t — because one can always
         execute the cheapest transfer found so far and then wait (coasting,
         at no additional cost) for the remaining time. This reproduces the
         solid line of the paper exactly: decreasing segments connected by
         plateaus where adding more time does not reduce the cost further.

    Parameters
    ----------
    theta : float
        Angular separation between the two satellites (radians).
    tof_min : float
        Minimum time of flight (seconds).
    tof_max : float
        Maximum time of flight (seconds).
    n_samples : int
        Number of time samples between tof_min and tof_max.
    r_km : float
        Radius of the circular orbit (km). Default is GEO radius.
    mu : float
        Gravitational parameter (km^3/s^2). Default is Earth's mu.
    max_M : int
        Maximum number of revolutions to consider in Lambert solutions.

    Returns
    -------
    tofs : np.ndarray
        Time of flight values (seconds).
    costs : np.ndarray
        Minimum achievable costs (km/s) for each time of flight.
    """
    a = CircularOrbit("a", r_km, 0.0, 0.0, 0.0)
    b = CircularOrbit("b", r_km, 0.0, 0.0, theta)
    tofs = np.linspace(tof_min, tof_max, n_samples)

    raw_costs = np.full(n_samples, np.inf)
    _, v1_local = a.state_at(0.0)
    for i, t in enumerate(tofs):
        r1, _ = a.state_at(0.0)
        r2, v2_local = b.state_at(float(t))
        try:
            sols = find_all_lambert_solutions(r1, r2, float(t), mu, max_M=max_M)
        except ValueError:
            continue
        for s in sols:
            dv = np.linalg.norm(s["v1"] - v1_local) + np.linalg.norm(v2_local - s["v2"])
            if dv < raw_costs[i]:
                raw_costs[i] = dv

    costs = np.minimum.accumulate(raw_costs)  # <- el "coasting": mínimo acumulado
    return tofs, costs


def find_breakpoints(theta: float, tof_min: float, tof_max: float,
                      r_km: float = GEO_RADIUS_KM, mu: float = MU_EARTH,
                      max_M: int = 3, n_scan: int = 250):
    """
    Exact β/A breakpoints (not sampled on a grid): for each Lambert branch
    (M revolutions, low/high path), find its true local minimum via bounded
    1D optimization (scipy.optimize.minimize_scalar), not via a grid. The β
    values are exactly the minima of each branch that achieve a new cost record
    when traversing time in increasing order — this reproduces the Section
    "Optimal Time Distribution" of the paper (Fig. 2/3): each βij is where
    "one curve is followed by a step".

    It follows the simplification used by the paper itself in its IP2/"Case 3"
    (it only uses constant segments; see the text after eq. 16): each A_j is
    taken equal to the next β (instead of computing the exact crossing of the
    connecting ramps, which the paper describes as "small compared with the
    plateau width").

    Returns a list of (beta_j, A_j, cost_j), ordered by time.
    """
    a = CircularOrbit("a", r_km, 0.0, 0.0, 0.0)
    b = CircularOrbit("b", r_km, 0.0, 0.0, theta)
    _, v1_local = a.state_at(0.0)

    def branch_cost(tof, M, low_path):
        r1, _ = a.state_at(0.0)
        r2, v2_local = b.state_at(tof)
        try:
            v1t, v2t = solve_lambert(r1, r2, tof, mu, M=M, low_path=low_path)
        except ValueError:
            return np.inf
        dv = np.linalg.norm(v1t - v1_local) + np.linalg.norm(v2_local - v2t)
        return float(dv) if np.isfinite(dv) else np.inf  # blindaje anti-NaN

    candidates = []
    for M in range(0, max_M + 1):
        for low_path in ([True] if M == 0 else [True, False]):
            scan = np.linspace(tof_min, tof_max, n_scan)
            vals = np.array([branch_cost(t, M, low_path) for t in scan])
            finite = np.isfinite(vals)
            if not finite.any():
                continue
            idx = np.where(finite)[0]

            lo = scan[idx[0]] + 1e-3 * (tof_max - tof_min)
            hi = scan[idx[-1]] - 1e-3 * (tof_max - tof_min)
            if hi <= lo:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                res = minimize_scalar(lambda t: branch_cost(t, M, low_path),
                                       bounds=(lo, hi), method="bounded",
                                       options={"xatol": 1e-8})
            if np.isfinite(res.fun):
                candidates.append((float(res.x), float(res.fun)))

    candidates.sort()
    breakpoints = []
    best = np.inf
    for tau, cost in candidates:
        if cost < best - 1e-9:
            breakpoints.append([tau, None, cost])
            best = cost
    for j in range(len(breakpoints) - 1):
        breakpoints[j][1] = breakpoints[j + 1][0]
    if breakpoints:
        breakpoints[-1][1] = tof_max
    return [tuple(bp) for bp in breakpoints]


def time_distribution_ip_exact(thetas: list[float], tf: float,
                                tof_min: float, tof_max: float,
                                r_km: float = GEO_RADIUS_KM, mu: float = MU_EARTH,
                                max_M: int = 3):
    """
    Literal reimplementation of IP2 (Case 3, eqs. 14-16 in the paper): uses the
    exact β/A breakpoints from find_breakpoints() as bins (instead of a grid),
    and the paper's TWO time constraints — (16b) Σ A_ij·x_ij ≥ tf and (16c)
    Σ β_ij·x_ij ≤ tf — instead of a single inequality. Since the cost is
    constant inside each [β_j, A_j], the exact choice of t_i within the bin does
    not change the Δv, so the remaining time (tf − Σβ_i) is distributed at no
    cost across the legs with slack so that the sum equals exactly tf whenever
    possible.

    Returns (tofs, dv_total, dv_per_leg) or (None, inf, None).
    """
    n = len(thetas)
    bins_per_leg = [find_breakpoints(th, tof_min, tof_max, r_km, mu, max_M) for th in thetas]

    if any(len(bins) == 0 for bins in bins_per_leg):
        return None, np.inf, None

    n_vars = sum(len(bins) for bins in bins_per_leg)
    offsets = np.cumsum([0] + [len(bins) for bins in bins_per_leg])

    c = np.zeros(n_vars)
    A_beta = np.zeros(n_vars)  # coeficientes para (16c): suma de beta_ij*x_ij <= tf
    A_A = np.zeros(n_vars)     # coeficientes para (16b): suma de A_ij*x_ij >= tf
    A_eq = np.zeros((n, n_vars))
    for i, bins in enumerate(bins_per_leg):
        for j, (beta, A_end, cost) in enumerate(bins):
            k = offsets[i] + j
            c[k] = cost
            A_beta[k] = beta
            A_A[k] = A_end
            A_eq[i, k] = 1.0

    constraints = [
        LinearConstraint(A_eq, np.ones(n), np.ones(n)),   # (16a): un bin por tramo
        LinearConstraint(A_A.reshape(1, -1), tf, np.inf),   # (16b)
        LinearConstraint(A_beta.reshape(1, -1), -np.inf, tf),  # (16c)
    ]
    res = milp(c, constraints=constraints, integrality=np.ones(n_vars), bounds=Bounds(0, 1))

    if not res.success:
        return None, np.inf, None

    x = res.x
    chosen = []
    for i, bins in enumerate(bins_per_leg):
        k_local = int(np.argmax(x[offsets[i]:offsets[i + 1]]))
        beta, A_end, cost = bins[k_local]
        chosen.append([beta, A_end, cost])

    # Repartir el sobrante de tiempo (no cambia el costo, cada bin es plano)
    total_beta = sum(ch[0] for ch in chosen)
    slack = max(0.0, tf - total_beta)
    tofs = []
    for beta, A_end, cost in chosen:
        room = A_end - beta
        take = min(slack, room)
        tofs.append(beta + take)
        slack -= take

    dv_per_leg = [ch[2] for ch in chosen]
    return tofs, float(sum(dv_per_leg)), dv_per_leg


def time_distribution_ip(thetas: list[float], tf: float,
                          tof_min: float, tof_max: float, n_samples: int = 60,
                          r_km: float = GEO_RADIUS_KM, mu: float = MU_EARTH):
    """
    Distributes tf across len(thetas) legs while minimizing total Δv,
    reproducing IP1 from the paper via explicit time discretization rather
    than analytically identifying the flat segments (which would require
    multi-revolution Lambert — see module note).

    Formulation (binary integer program, same structure as IP1):
        variables  x_ik ∈ {0,1}   (leg i uses time bin k)
        min        Σ_i Σ_k c_ik x_ik
        s.a.       Σ_k x_ik = 1                    ∀i   (exactly one bin per leg)
                   Σ_i Σ_k t_ik x_ik ≤ tf                (do not exceed total time)

    Returns (optimal_tofs, total_delta_v) or (None, inf) if infeasible.
    """
    n = len(thetas)
    tof_grid, cost_grid = [], []
    for theta in thetas:
        tofs, costs = cost_curve(theta, tof_min, tof_max, n_samples, r_km, mu)
        tof_grid.append(tofs)
        cost_grid.append(costs)

    # Variables: x[i][k], aplanadas en un solo vector para scipy.optimize.milp
    n_bins = n_samples
    n_vars = n * n_bins

    def idx(i, k):
        return i * n_bins + k

    c = np.zeros(n_vars)
    for i in range(n):
        for k in range(n_bins):
            cost = cost_grid[i][k]
            # Bins infactibles (Lambert no converge) reciben costo muy alto
            # en vez de excluir la variable, para mantener el programa bien
            # definido (milp no admite "huecos" fáciles en las variables).
            c[idx(i, k)] = cost if np.isfinite(cost) else 1e6

    # Restricción de igualdad: exactamente un bin por tramo
    A_eq = np.zeros((n, n_vars))
    for i in range(n):
        for k in range(n_bins):
            A_eq[i, idx(i, k)] = 1.0
    b_eq = np.ones(n)

    # Restricción de desigualdad: tiempo total <= tf
    A_time = np.zeros((1, n_vars))
    for i in range(n):
        for k in range(n_bins):
            A_time[0, idx(i, k)] = tof_grid[i][k]

    constraints = [
        LinearConstraint(A_eq, b_eq, b_eq),
        LinearConstraint(A_time, -np.inf, tf),
    ]
    integrality = np.ones(n_vars)  # todas binarias
    bounds = Bounds(0, 1)

    res = milp(c, constraints=constraints, integrality=integrality, bounds=bounds)

    if not res.success:
        return None, np.inf, None

    x = res.x.reshape(n, n_bins)
    chosen_k = np.argmax(x, axis=1)  # cada fila debería tener un solo "1"
    tofs_opt = [float(tof_grid[i][chosen_k[i]]) for i in range(n)]
    dv_per_leg = [float(cost_grid[i][chosen_k[i]]) for i in range(n)]
    dv_total = float(sum(dv_per_leg))

    return tofs_opt, dv_total, dv_per_leg


# ---------------------------------------------------------------------
# Orchestrator: solves the full model (sequence + distribution)
# ---------------------------------------------------------------------
def solve_optimal_rdv_seq(orbits, alphas, tf: float, tof_min: float, tof_max: float,
                       n_samples: int = 60, r_km: float = GEO_RADIUS_KM,
                       mu: float = MU_EARTH, verbose: bool = True,
                       m0_kg: float = None, isp_s: float = 300.0,
                       time_method: str = "grid", max_M: int = 3,
                       dock_time_s: float = 0.0):
    """
    Solves the complete paper problem (Problem 2: without returning to the
    depot): identifies the candidate sequence(s) with minimum TSA, solves the
    optimal time distribution for each, and returns the best one.

    time_method="grid" (default, fast): time_distribution_ip — real cost curve
    (multi-rev + coasting) but discretized on a grid.
    time_method="exact" (slower, more faithful): time_distribution_ip_exact
    — exact β/A breakpoints via 1D optimization per branch, with the paper's
    TWO time constraints (eqs. 16b/16c). This gives equal or better results
    than "grid" (never worse) because it has no discretization error, at the
    cost of more computation time.

    dock_time_s: fixed docking/handling time at EACH visited satellite (same for
    all — not in the paper, it is a custom extension). It is subtracted from the
    total `tf` BEFORE distributing time across transfer legs
    (tf_transferencias = tf - n_legs * dock_time_s), so the sequence logic and
    integer program do not change at all — only the time budget available for
    travel. Dock time does not consume propellant by itself (it is operational
    time, not a maneuver), so it does not affect the mass profile via
    Tsiolkovsky. It is reported separately in the result (`dock_time_s`,
    `total_duration_with_docking`) so that anyone building a schedule or
    animation can insert the real pause between arrival at a target and the
    departure toward the next one.

    If m0_kg is given (initial service mass, kg), it also computes the leg-by-leg
    propellant consumption profile via Tsiolkovsky (eqs. 1-2 of Section III of
    Tsiotras & de Nailly, 2005), reproducing the pattern in Tables 1/4: equal
    Δv across legs does NOT imply equal propellant, because the vehicle becomes
    lighter after each burn.

    Returns a dict with: sequence (indices in `orbits`), thetas, tofs,
    total_dv, tsa, dv_per_leg, mass_profile (if m0_kg was given), and the
    details of all evaluated candidates (to show, as the paper does, that there
    may be a TSA tie that is only resolved by computing the real cost).
    """
    candidates = best_sequence_candidates(alphas)
    n_legs = len(alphas) - 1
    tf_transfers = tf - n_legs * dock_time_s

    if tf_transfers <= 0:
        raise ValueError(
            f"tf={tf:.1f}s no alcanza ni para el acoplamiento: "
            f"{n_legs} paradas x {dock_time_s:.1f}s = {n_legs*dock_time_s:.1f}s "
            f"ya superan el tf total. No queda tiempo para transferir."
        )

    results = []
    for seq_labels in candidates:
        # seq_labels son etiquetas 1..n en el orden físico de `orbits`
        seq_idx = seq_labels  # orbits[label] ya es el índice correcto

        # ángulos de separación reales entre pasos consecutivos de la secuencia
        cum = [0.0]
        for g in alphas[:-1]:
            cum.append(cum[-1] + g)

        def sep(u_a, u_b):
            d = (u_b - u_a) % (2 * np.pi)
            if d > np.pi:
                d -= 2 * np.pi
            return d  # con signo: determina sentido de la transferencia

        thetas = [sep(cum[0], cum[seq_idx[0]])]
        for i in range(len(seq_idx) - 1):
            thetas.append(sep(cum[seq_idx[i]], cum[seq_idx[i + 1]]))

        tsa = total_sweep_angle(alphas, seq_idx)

        if time_method == "exact":
            tofs, dv_total, dv_per_leg = time_distribution_ip_exact(
                thetas, tf_transfers, tof_min, tof_max, r_km, mu, max_M)
        else:
            tofs, dv_total, dv_per_leg = time_distribution_ip(
                thetas, tf_transfers, tof_min, tof_max, n_samples, r_km, mu)

        entry = {
            "sequence_labels": seq_labels,
            "sequence_idx": [0] + seq_idx,
            "thetas": thetas,
            "tofs": tofs,
            "dv_per_leg": dv_per_leg,
            "total_dv": dv_total,
            "tsa_deg": np.degrees(tsa),
            "dock_time_s": dock_time_s,
            "total_duration_with_docking": (
                (sum(tofs) + n_legs * dock_time_s) if tofs is not None else None
            ),
        }

        if m0_kg is not None and dv_per_leg is not None:
            entry["mass_profile"] = mass_consumption_profile(dv_per_leg, m0_kg, isp_s)
            entry["final_mass_kg"] = entry["mass_profile"][-1]["mass_after_kg"]
            entry["total_propellant_kg"] = m0_kg - entry["final_mass_kg"]

        results.append(entry)
        if verbose:
            names = " -> ".join(["DEPOT"] + [orbits[i].name for i in seq_idx])
            extra = ""
            if m0_kg is not None and dv_per_leg is not None:
                extra = f"   propelente={entry['total_propellant_kg']:.2f} kg  m_final={entry['final_mass_kg']:.2f} kg"
            dock_note = f"   (+{dock_time_s*n_legs/3600:.1f}h de acople)" if dock_time_s > 0 else ""
            print(f"  Candidata TSA={np.degrees(tsa):.1f}°:  {names}   "
                  f"Δv={dv_total:.4f} km/s{extra}{dock_note}")

    best = min(results, key=lambda r: r["total_dv"])
    return best, results
