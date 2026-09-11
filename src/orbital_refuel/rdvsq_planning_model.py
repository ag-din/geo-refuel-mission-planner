"""
rdvsq_planning_model.py
----------------------

Implementación directa del modelo de:

  Shen, H., Tsiotras, P. (2002). "Optimal Scheduling for Servicing
  Multiple Satellites in a Circular Constellation." AIAA/AAS
  Astrodynamics Specialist Conference.

    El problema se resuelve en dos
    pasos:

  1. PROBLEMA DE SECUENCIA ÓPTIMA

  2. PROBLEMA DE DISTRIBUCIÓN ÓPTIMA DE TIEMPO

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

    n = len(alphas) - 1
    
    cum = [0.0]
    for g in alphas[:-1]:
        cum.append(cum[-1] + g)
    

    def sep(u_a, u_b):
        d = (u_b - u_a) % (2 * np.pi)
        if d > np.pi:
            d -= 2 * np.pi
        return abs(d)

    tsa = sep(cum[0], cum[sequence[0]])
    for i in range(len(sequence) - 1):
        tsa += sep(cum[sequence[i]], cum[sequence[i + 1]])
    return tsa


# ---------------------------------------------------------------------
# Masa y propelente (ecs. 1-2 de la Sección III de Tsiotras & de Nailly,
# 2005 — "Optimal Single Service Vehicle Refueling")
# ---------------------------------------------------------------------
def mass_consumption_profile(dv_per_leg: list[float], m0: float,
                              isp_s: float, g0: float = 9.80665):
    """
    Dada la lista de Δv (km/s) de cada tramo y la masa inicial del
    servicio m0 (kg), aplica la ecuación del cohete tramo a tramo:

        m(t_k^+) = m(t_k^-) * exp(-ΔV_k / (Isp * g0))     (ec. 1 del paper)

    Devuelve una lista de dicts, uno por tramo, con la masa ANTES y
    DESPUÉS del impulso y el propelente consumido — reproduciendo el
    patrón que muestra el paper en sus Tablas 1 y 4: aunque el Δv de
    varios tramos sea idéntico, el propelente consumido en cada uno
    DISMINUYE a medida que la nave se aliviana (el propelente gastado
    es proporcional a la masa que aún carga, no solo al Δv).
    """
    c0 = isp_s * g0 / 1000.0  # km/s (Isp*g0 está en m/s)
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
# Paso 2: problema de distribución óptima de tiempo 
# ---------------------------------------------------------------------
def cost_curve(theta: float, tof_min: float, tof_max: float, n_samples: int,
               r_km: float = GEO_RADIUS_KM, mu: float = MU_EARTH, max_M: int = 3):
    """Muestrea Cθ(t): Δv (km/s) en función del tiempo de vuelo t, para
    dos cuerpos en la MISMA órbita circular separados por ángulo theta
    (rad) — pero ahora usando la curva de costo REAL con "coasting",
    no solo Lambert de cero-revolución:

      1. Para cada tof de la grilla, se prueban TODAS las soluciones de
         Lambert de 0 a max_M revoluciones (lambert.find_all_lambert_
         solutions) y se toma la más barata — esto reproduce la curva
         "sin coasting" (línea punt eada de la Fig. 1 del paper de
         Tsiotras & de Nailly), que ya NO es una sola rama sino varias
         familias entrelazadas.
      2. Se aplica el MÍNIMO ACUMULADO sobre esa curva: el costo en el
         tiempo t es el mínimo alcanzable usando CUALQUIER tof <= t —
         porque siempre se puede ejecutar la transferencia más barata
         encontrada hasta ahí y luego esperar (coasting, sin costo
         adicional) el tiempo restante. Esto reproduce exactamente la
         línea sólida del paper: tramos que decrecen, conectados por
         "mesetas" donde agregar más tiempo no baja más el costo.

    Devuelve (tofs, costs) con costs=inf donde ninguna solución converge.
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
    Breakpoints β/A EXACTOS (no muestreados en grilla): para cada rama
    de Lambert (M revoluciones, camino low/high), encuentra su mínimo
    local real vía optimización 1D acotada (scipy.optimize.minimize_scalar),
    no vía grilla. Los β son exactamente los mínimos de cada rama que
    logran un nuevo récord de costo al recorrer el tiempo creciente —
    esto reproduce la Sección "Optimal Time Distribution" del paper
    (Fig. 2/3): cada βij es donde "una curva es seguida por un escalón".

    Sigue la simplificación que el propio paper adopta en su IP2/"Case 3"
    (solo usa los tramos constantes, ver el texto tras la ec. 16): cada
    A_j se toma igual al siguiente β (en vez de calcular el cruce exacto
    de las rampas de conexión, que el propio paper describe como
    "pequeño comparado con el ancho de la meseta").

    Devuelve una lista de (beta_j, A_j, costo_j), ordenada por tiempo.
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
            # margen pequeño hacia adentro del rango valido, para que el
            # optimizador acotado no tantee justo en el borde singular
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
    Reimplementación LITERAL de IP2 (Case 3, ecs. 14-16 del paper): usa
    los breakpoints β/A exactos de find_breakpoints() como bins (en vez
    de una grilla), y las DOS restricciones de tiempo del paper — (16b)
    Σ A_ij·x_ij ≥ tf y (16c) Σ β_ij·x_ij ≤ tf — en vez de una sola
    desigualdad. Como el costo es constante dentro de cada [β_j, A_j],
    la elección exacta de t_i dentro del bin no cambia el Δv, así que el
    sobrante de tiempo (tf − Σβ_i) se reparte sin costo entre los tramos
    con margen, para que la suma dé exactamente tf cuando es posible.

    Devuelve (tofs, dv_total, dv_per_leg) o (None, inf, None).
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
    Distribuye tf entre len(thetas) tramos minimizando el Δv total,
    replicando IP1 del paper mediante una discretización explícita del
    tiempo en vez de identificar analíticamente los tramos planos
    (que requeriría Lambert multi-revolución — ver nota del módulo).

    Formulación (programa entero binario, igual estructura que IP1):
        variables  x_ik ∈ {0,1}   (tramo i usa el bin k de tiempo)
        min        Σ_i Σ_k c_ik x_ik
        s.a.       Σ_k x_ik = 1                    ∀i   (exactamente un bin por tramo)
                   Σ_i Σ_k t_ik x_ik ≤ tf                (no exceder el tiempo total)

    Devuelve (tofs_óptimos, delta_v_total) o (None, inf) si es infactible.
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
# Orquestador: resuelve el modelo completo (secuencia + distribución)
# ---------------------------------------------------------------------
def solve_optimal_rdv_seq(orbits, alphas, tf: float, tof_min: float, tof_max: float,
                       n_samples: int = 60, r_km: float = GEO_RADIUS_KM,
                       mu: float = MU_EARTH, verbose: bool = True,
                       m0_kg: float = None, isp_s: float = 300.0,
                       time_method: str = "grid", max_M: int = 3,
                       dock_time_s: float = 0.0):
    """
    Resuelve el problema completo del paper (Problem 2: sin retorno al
    depot): identifica la(s) secuencia(s) candidata(s) de mínimo TSA,
    resuelve la distribución óptima de tiempo para CADA una, y devuelve
    la mejor.

    time_method="grid" (default, rápido): time_distribution_ip — curva
    de costo real (multi-rev + coasting) pero discretizada en grilla.
    time_method="exact" (más lento, más fiel): time_distribution_ip_exact
    — breakpoints β/A exactos vía optimización 1D por rama, con las DOS
    restricciones de tiempo del paper (ecs. 16b/16c). Da resultados
    iguales o mejores que "grid" (nunca peores) porque no tiene error de
    discretización, a costa de más tiempo de cómputo.

    dock_time_s: tiempo fijo de acoplamiento/entrega en CADA satélite
    visitado (igual para todos — no está en el paper, es una extensión
    propia). Se descuenta del `tf` total ANTES de repartir tiempo entre
    los tramos de transferencia (tf_transferencias = tf - n_legs *
    dock_time_s), así que la lógica de secuencia y de programación
    entera no cambia en absoluto — solo el presupuesto disponible para
    viajar. El tiempo de acople no consume combustible por sí mismo (es
    tiempo de operación, no una maniobra), así que no afecta el perfil
    de masa vía Tsiolkovsky. Se reporta por separado en el resultado
    (`dock_time_s`, `total_duration_with_docking`) para que quien arme
    un cronograma o animación pueda insertar la pausa real entre la
    llegada a un objetivo y la salida hacia el siguiente.

    Si se da m0_kg (masa inicial del servicio, kg), además calcula el
    perfil de consumo de propelente tramo a tramo vía Tsiolkovsky (ecs.
    1-2 de la Sección III de Tsiotras & de Nailly, 2005), replicando el
    patrón de sus Tablas 1/4: Δv iguales entre tramos NO implican
    propelente igual, porque la nave se aliviana en cada quemado.

    Devuelve un dict con: sequence (índices en `orbits`), thetas, tofs,
    total_dv, tsa, dv_per_leg, mass_profile (si m0_kg se dio), y el
    detalle de todas las candidatas evaluadas (para mostrar, como hace
    el paper, que puede haber empate en TSA que solo se resuelve
    calculando el costo real).
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
