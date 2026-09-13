
"""
mission.py
----------

Generates a mission scenario with a servicier and multiple target satellites
in the same circular GEO orbit. All the bodies share the same radius, 
inclination, and RAAN, but differ in phase.
"""

import numpy as np
from .keplerian import CircularOrbit, R_EARTH

# Constant
GEO_RADIUS_KM = R_EARTH + 35786.0

def generate_mission_case(n_targets: int, seed: int = 42,
                                 r_km: float = GEO_RADIUS_KM,
                                 evenly: bool = False):
    """
    Generates a mission scenario with a servicier and multiple target satellites
    in the same circular GEO orbit. All the bodies share the same radius,
    inclination, and RAAN, but differ in phase.

    Parameters
    ----------
    n_targets : int
        Number of target satellites (not counting the servicier).
    seed : int
        Random seed for reproducibility.
    r_km : float
        Radius of the circular orbit in kilometers. Default is GEO radius.
    evenly : bool
        If True, the targets are evenly spaced around the orbit. If False,
        the targets are randomly spaced.
    
    Returns
    -------
    orbits : list of CircularOrbit
        List of CircularOrbit objects, index 0 is the servicier/depot, 1
        to n are the target satellites, in physical order around the circle
        (orbit-wise from the depot).
    gaps : list of float
        List of angular gaps (in radians) between consecutive satellites,
        including the gap between the last target and the depot.

    """
    rng = np.random.default_rng(seed)

    if evenly:
        gaps = np.full(n_targets + 1, 2 * np.pi / (n_targets + 1))
    else:
        cuts = np.sort(rng.uniform(0, 2 * np.pi, n_targets))
        edges = np.concatenate([[0.0], cuts, [2 * np.pi]])
        gaps = np.diff(edges)

    depot = CircularOrbit("DEPOT", r_km, 0.0, 0.0, 0.0)
    orbits = [depot]
    u = 0.0
    for k in range(n_targets):
        u += gaps[k]
        orbits.append(CircularOrbit(f"TARGET-{k+1}", r_km, 0.0, 0.0, u))

    return orbits, list(gaps)