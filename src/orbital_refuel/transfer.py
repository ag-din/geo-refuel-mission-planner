"""
transfer.py
-----------
Cost of Δv to transfer the service satellite from orbit `orbit_a`
to orbit `orbit_b`, departing at time `t_depart` and using flight time `tof`.

  1. The target satellite moves during the transfer: its rendezvous
     position is its position at t_depart + tof, not its current position.
  2. The cost depends explicitly on `tof` and it must be optimized over tof.
  3. The orbital plane change (inclination + RAAN) is captured only by
     working in 3D with real vectors via Lambert, without an ad hoc formula.
"""

from __future__ import annotations
import numpy as np
from scipy.optimize import minimize_scalar

from .keplerian import CircularOrbit, MU_EARTH
from .lambert import solve_lambert


def transfer_delta_v(orbit_a: CircularOrbit, orbit_b: CircularOrbit,
                      t_depart: float, tof: float, mu: float = MU_EARTH) -> float:
    """
    Total Δv (km/s) to go from orbit_a to orbit_b, leaving at t_depart
    with flight time tof. Returns inf if Lambert does not converge
    (degenerate geometry) instead of raising an exception, so the outer
    optimizer can simply discard that point.

    Parameters
    ----------
    orbit_a : CircularOrbit
        The departure orbit (service satellite).
    orbit_b : CircularOrbit
        The target orbit (target satellite).
    t_depart : float
        Time of departure (seconds).
    tof : float
        Time of flight (seconds).
    mu : float
        Gravitational parameter (km^3/s^2). Default is Earth's mu.

    Returns
    -------
    float
        Total Δv (km/s) for the transfer, or np.inf if Lambert fails.
    """
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
    Finds the flight time tof* in [tof_min, tof_max] that minimizes total
    Δv using a coarse scan (to locate the minimum region, since the
    Δv-vs-tof curve can have multiple local minima) followed by local
    refinement.

    Parameters
    ----------
    orbit_a : CircularOrbit
        The departure orbit (service satellite).
    orbit_b : CircularOrbit
        The target orbit (target satellite).
    t_depart : float
        Time of departure (seconds).
    tof_min : float
        Minimum time of flight (seconds).
    tof_max : float
        Maximum time of flight (seconds).
    mu : float
        Gravitational parameter (km^3/s^2). Default is Earth's mu.
    n_scan : int
        Number of points in the coarse scan of tof. Default is 25.

    Returns
    -------
    tuple
        (tof*, Δv*) where tof* is the optimal time of flight (seconds)
        and Δv* is the corresponding total Δv (km/s).
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
