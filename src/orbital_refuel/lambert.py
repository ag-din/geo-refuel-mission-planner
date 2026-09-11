"""
lambert.py
----------

Contains a function to solve Lambert's problem, including MULTI-REVOLUTION solutions.
Given two position vectors r1, r2 (3D, km) and a time of flight tof (s),
find the elliptical transfer orbit(s) that connect both points in that exact time,
and return the required velocities v1, v2 (km/s) at the start and end of the transfer.

"""

from __future__ import annotations
import numpy as np
from scipy.optimize import brentq


def _stumpff_C(z):
    """
    Stumpff function C, defined as:
    
    C(z) = (1 - cos(sqrt(z))) / z for z > 0,
    C(z) = (cosh(sqrt(-z)) - 1) / (-z) for z < 0, 
    and C(0) = 1/2.

    Parameters
    ----------
    z : float or np.ndarray
        Input value(s) for which to compute the Stumpff function C(z).
    
    Returns
    -------
    float or np.ndarray
        The computed value(s) of the Stumpff function C(z).
    """
    z = np.asarray(z, dtype=float)
    out = np.full_like(z, 0.5)
    pos = z > 1e-8
    neg = z < -1e-8
    sz_pos = np.sqrt(z[pos])
    out[pos] = (1 - np.cos(sz_pos)) / z[pos]
    sz_neg = np.sqrt(-z[neg])
    out[neg] = (np.cosh(sz_neg) - 1) / (-z[neg])
    return out


def _stumpff_S(z):
    """
    Stumpff function S, defined as:
    
    S(z) = (sqrt(z) - sin(sqrt(z))) / (sqrt(z))^3 for z > 0,
    S(z) = (sinh(sqrt(-z)) - sqrt(-z)) / (sqrt(-z))^3 for z < 0,
    and S(0) = 1/6.

    Parameters
    ----------
    z : float or np.ndarray
        Input value(s) for which to compute the Stumpff function S(z).

    Returns
    -------
    float or np.ndarray
        The computed value(s) of the Stumpff function S(z).
    """
    z = np.asarray(z, dtype=float)
    out = np.full_like(z, 1.0 / 6.0)
    pos = z > 1e-8
    neg = z < -1e-8
    sz_pos = np.sqrt(z[pos])
    out[pos] = (sz_pos - np.sin(sz_pos)) / sz_pos**3
    sz_neg = np.sqrt(-z[neg])
    out[neg] = (np.sinh(sz_neg) - sz_neg) / sz_neg**3
    return out


def _geometry(r1_vec, r2_vec, prograde):
    """
    Calculates the geometry of the transfer orbit, 
    including the magnitudes of the position vectors and
    the parameter A (semi-major axis).
    """
    r1 = np.linalg.norm(r1_vec)
    r2 = np.linalg.norm(r2_vec)
    cos_dnu = np.clip(np.dot(r1_vec, r2_vec) / (r1 * r2), -1.0, 1.0)
    cross = np.cross(r1_vec, r2_vec)
    if prograde:
        dnu = np.arccos(cos_dnu) if cross[2] >= 0 else 2 * np.pi - np.arccos(cos_dnu)
    else:
        dnu = 2 * np.pi - np.arccos(cos_dnu) if cross[2] >= 0 else np.arccos(cos_dnu)
    A = np.sin(dnu) * np.sqrt(r1 * r2 / (1 - np.cos(dnu)))
    return r1, r2, A


def _velocities_at(z, r1_vec, r2_vec, r1, r2, A, mu):
    """
    Calculates the velocities at the start and end of the transfer orbit
    for a given Stumpff parameter z, using the geometry of the transfer.

    Parameters
    ----------
    z : float
        Stumpff parameter.
    r1_vec : np.ndarray
        Initial position vector.
    r2_vec : np.ndarray
        Final position vector.
    r1 : float
        Magnitude of the initial position vector.
    r2 : float
        Magnitude of the final position vector.
    A : float
        Semi-major axis parameter of the transfer orbit.
    mu : float
        Standard gravitational parameter of the central body.

    Returns
    -------
    v1 : np.ndarray
        Velocity vector at the initial position.
    v2 : np.ndarray
        Velocity vector at the final position.
    y : float
        Auxiliary variable used in the calculation of velocities.
    """
    C, S = _stumpff_C(z), _stumpff_S(z)
    y = r1 + r2 + A * (z * S - 1) / np.sqrt(C)
    f = 1 - y / r1
    g = A * np.sqrt(y / mu)
    gdot = 1 - y / r2
    v1 = (r2_vec - f * r1_vec) / g
    v2 = (gdot * r2_vec - r1_vec) / g
    return v1, v2, float(y)


def find_all_lambert_solutions(r1_vec: np.ndarray, r2_vec: np.ndarray, tof: float,
                                mu: float, max_M: int = 0, prograde: bool = True,
                                n_scan: int = 400):
    """
    Returns ALL Lambert solutions (0 to max_M revolutions)
    for the given tof, as a list of dicts {M, low_path, v1, v2, z}.

    Parameters
    ----------
    r1_vec : np.ndarray
        Initial position vector (3D, km).
    r2_vec : np.ndarray
        Final position vector (3D, km).
    tof : float
        Time of flight (s).
    mu : float
        Standard gravitational parameter of the central body (km^3/s^2).
    max_M : int, optional
        Maximum number of revolutions to consider (default is 0).
    prograde : bool, optional
        If True, compute the prograde solution; if False, compute the retrograde solution.
    n_scan : int, optional
        Number of points to scan in the z range for root finding (default is 400).

    Returns
    -------
    list of dict
        A list of dictionaries, each containing the solution parameters:
        - 'M': number of revolutions
        - 'low_path': True for low-path solution, False for high-path solution
        - 'v1': velocity vector at the initial position (km/s)
        - 'v2': velocity vector at the final position (km/s)
        - 'z': the value of the auxiliary variable z
    """
    if tof <= 0:
        raise ValueError("El tiempo de vuelo (tof) debe ser positivo.")

    r1, r2, A = _geometry(r1_vec, r2_vec, prograde)
    if not np.isfinite(A) or abs(A) < 1e-9:
        raise ValueError("Geometría degenerada (Δν ≈ 0 o 180°).")

    def F(z):
        C, S = _stumpff_C(z), _stumpff_S(z)
        y = r1 + r2 + A * (z * S - 1) / np.sqrt(C)
        y = np.where(y < 0, 1e-6, y)
        return (y / C) ** 1.5 * S + A * np.sqrt(y) - np.sqrt(mu) * tof

    z_max = (2 * np.pi * (max_M + 1)) ** 2 - 1e-6
    z_grid = np.linspace(-20.0, z_max, n_scan * (max_M + 1))
    f_grid = F(z_grid)
    signs = np.sign(f_grid)
    sign_changes = np.where(np.diff(signs) != 0)[0]

    roots = []
    for k in sign_changes:
        try:
            z = brentq(lambda zz: float(F(zz)), z_grid[k], z_grid[k + 1],
                       xtol=1e-10, maxiter=200)
        except ValueError:
            continue  # cambio de signo espurio (artefacto de la grilla), se descarta
        v1, v2, y = _velocities_at(z, r1_vec, r2_vec, r1, r2, A, mu)
        if y > 0 and np.all(np.isfinite(v1)):
            roots.append((z, v1, v2))

    # M=0 -> 1st root; for  M>=1, (low, high)
    solutions = []
    if roots:
        solutions.append({"M": 0, "low_path": True, "v1": roots[0][1], "v2": roots[0][2], "z": roots[0][0]})
    idx = 1
    M = 1
    while idx + 1 < len(roots) + 1 and M <= max_M:
        if idx >= len(roots):
            break
        solutions.append({"M": M, "low_path": True, "v1": roots[idx][1], "v2": roots[idx][2], "z": roots[idx][0]})
        if idx + 1 < len(roots):
            solutions.append({"M": M, "low_path": False, "v1": roots[idx + 1][1], "v2": roots[idx + 1][2], "z": roots[idx + 1][0]})
        idx += 2
        M += 1

    return solutions


def solve_lambert(r1_vec: np.ndarray, r2_vec: np.ndarray, tof: float, mu: float,
                   prograde: bool = True, M: int = 0, low_path: bool = True):
    """
    Solve Lambert's problem for a given number of revolutions M (default M=0: zero-rev behavior).   
    
    Parameters
    ----------
    r1_vec : np.ndarray
        Initial position vector (3D, km).
    r2_vec : np.ndarray
        Final position vector (3D, km).
    tof : float
        Time of flight (s).
    mu : float
        Standard gravitational parameter of the central body (km^3/s^2).
    prograde : bool, optional
        If True, compute the prograde solution; if False, compute the retrograde solution.
    M : int, optional
        Number of revolutions (default is 0 for zero-rev solution).
    low_path : bool, optional
        If True, return the low-path solution; if False, return the high-path solution (only relevant for M >= 1).
    
    Returns
    -------
    v1 : np.ndarray
        Velocity vector at the initial position (km/s).
    v2 : np.ndarray
        Velocity vector at the final position (km/s).
    
    Raises
    ------
    ValueError
        If no valid Lambert solution is found for the specified parameters.
    """
    solutions = find_all_lambert_solutions(r1_vec, r2_vec, tof, mu, max_M=M, prograde=prograde)
    for s in solutions:
        if s["M"] == M and (M == 0 or s["low_path"] == low_path):
            return s["v1"], s["v2"]

    raise ValueError(f"No solution found for M={M} "
                      f"(low_path={low_path}) with tof={tof:.1f}s; probably "
                      f"tof too short to fit {M} complete revolutions.")
