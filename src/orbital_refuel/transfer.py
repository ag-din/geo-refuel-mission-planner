"""
transfer.py
-----------
Costo Δv de transferir el satélite de servicio de la órbita `orbit_a`
a la órbita `orbit_b`, saliendo en el tiempo `t_depart` y empleando un
tiempo de vuelo `tof`. 

  1. El satélite objetivo SE MUEVE durante el tránsito: su posición de
     encuentro es su posición en t_depart + tof, no su posición actual.
  2. El costo depende explícitamente de `tof` (como en la Fig. 1 del
     paper de Tsiotras & de Nailly) — hay que optimizar sobre tof.
  3. El cambio de plano orbital (inclinación + RAAN) se captura solo
     por trabajar en 3D con vectores reales vía Lambert, sin fórmula
     ad-hoc.
"""

from __future__ import annotations
import numpy as np
from scipy.optimize import minimize_scalar

from .keplerian import CircularOrbit, MU_EARTH
from .lambert import solve_lambert


def transfer_delta_v(orbit_a: CircularOrbit, orbit_b: CircularOrbit,
                      t_depart: float, tof: float, mu: float = MU_EARTH) -> float:
    """Δv total (km/s) para ir de orbit_a a orbit_b, saliendo en t_depart
    con tiempo de vuelo tof. Devuelve inf si Lambert no converge (geometría
    degenerada) en vez de lanzar excepción, para que la optimización externa
    pueda simplemente descartar ese punto."""
    r1, v1_local = orbit_a.state_at(t_depart)
    r2, v2_local = orbit_b.state_at(t_depart + tof)

    try:
        v1_transfer, v2_transfer = solve_lambert(r1, r2, tof, mu, prograde=True)
    except (ValueError, ZeroDivisionError, FloatingPointError):
        return np.inf

    dv1 = np.linalg.norm(v1_transfer - v1_local)
    dv2 = np.linalg.norm(v2_local - v2_transfer)
    return float(dv1 + dv2)


def optimal_transfer(orbit_a: CircularOrbit, orbit_b: CircularOrbit,
                      t_depart: float, tof_min: float, tof_max: float,
                      mu: float = MU_EARTH, n_scan: int = 25):
    """
    Encuentra el tiempo de vuelo tof* en [tof_min, tof_max] que minimiza
    el Δv total, mediante un escaneo grueso (para localizar la región del
    mínimo, ya que la curva Δv-vs-tof puede tener más de un mínimo local,
    ) seguido de un refinamiento local.

    Devuelve (tof_óptimo, delta_v_óptimo).
    """
    tof_grid = np.linspace(tof_min, tof_max, n_scan)
    dv_grid = np.array([transfer_delta_v(orbit_a, orbit_b, t_depart, t, mu)
                         for t in tof_grid])

    best_idx = int(np.argmin(dv_grid))
    if not np.isfinite(dv_grid[best_idx]):
        return None, np.inf

    lo = tof_grid[max(0, best_idx - 1)]
    hi = tof_grid[min(len(tof_grid) - 1, best_idx + 1)]

    def objective(t):
        return transfer_delta_v(orbit_a, orbit_b, t_depart, t, mu)

    res = minimize_scalar(objective, bounds=(lo, hi), method="bounded",
                           options={"xatol": 1.0}) 

    if res.fun < dv_grid[best_idx]:
        return float(res.x), float(res.fun)
    return float(tof_grid[best_idx]), float(dv_grid[best_idx])
