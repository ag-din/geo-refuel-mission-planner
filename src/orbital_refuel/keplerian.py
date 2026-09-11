"""
keplerian.py
------------

Definition of a class for circular orbits and functions to 
calculate the position and velocity of a satellite in ECI frame
at a given time.

"""

import numpy as np
from dataclasses import dataclass

# Constants
MU_EARTH = 398600.4418  # km^3/s^2
R_EARTH = 6378.137      # km


@dataclass(frozen=True)
class CircularOrbit:
    """
    Circular orbit defined by its radius, inclination,
    RAAN and argument of latitude at t0.

    Arguments:
    ----------
    name : str
        Name of the orbit that a satellite is in.
    radius_km : float
        Orbital radius in kilometers.
    inclination_rad : float
        Orbital inclination in radians.
    raan_rad : float
        Right ascension of ascending node in radians.
    u0_rad : float
        Argument of latitude at t0 in radians.
    """
    name: str
    radius_km: float
    inclination_rad: float
    raan_rad: float
    u0_rad: float

    @property
    def mean_motion(self) -> float:
        """
        Mean motion defined as n = sqrt(mu / r^3), rad/s,
        where mu is the standard gravitational parameter of Earth (km^3/s^2
        and r is the orbital radius (km).

        Parameters
        ----------
        None

        Returns
        -------
        float
            Mean motion in rad/s.
        """
        return np.sqrt(MU_EARTH / self.radius_km**3)

    @property
    def period_s(self) -> float:
        """
        Orbital period definied as T = 2π / n, s,
        where n is the mean motion (rad/s).

        Parameters
        ----------
        None

        Returns
        -------
        float
            Orbital period in seconds.
        """
        return 2 * np.pi / self.mean_motion

    def state_at(self, t: float):
        """
        Calculates the position and velocity vectors of a satellite in 
        ECI frame at time t (seconds since t0) for an circular orbit, 
        the position in ECI frame is given by:

            X = r (cosΩ cos u - sinΩ sin u cos i)
            Y = r (sinΩ cos u + cosΩ sin u cos i)
            Z = r (sin u sin i)

        where r is the orbital radius, Ω is the RAAN, i is the inclination, 
        u is the argument of latitude (u = u0 + n*t).

        Parameters
        ----------
        t : float
            Time in seconds since t0.
        
        Returns
        -------
        pos : np.ndarray
            Satellite position vector in ECI frame (km).
        vel : np.ndarray
            Satellite velocity vector in ECI frame (km/s).
        """
        n = self.mean_motion
        u = self.u0_rad + n * t
        r, i, Om = self.radius_km, self.inclination_rad, self.raan_rad

        cu, su = np.cos(u), np.sin(u)
        cO, sO = np.cos(Om), np.sin(Om)
        ci, si = np.cos(i), np.sin(i)

        pos = np.array([
            r * (cO * cu - sO * su * ci),
            r * (sO * cu + cO * su * ci),
            r * (su * si),
        ])

        vel = r * n * np.array([
            -(cO * su + sO * cu * ci),
            -(sO * su - cO * cu * ci),
            (cu * si),
        ])

        return pos, vel
