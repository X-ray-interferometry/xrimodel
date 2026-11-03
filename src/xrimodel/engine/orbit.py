"""Two-body Keplerian orbits, to place the parts of a binary in a :class:`System3D`.

The orbit lies in the frame's x-y plane with the centre of mass at the origin and the
angular momentum along +z (so for i < 90 deg the orbit runs counterclockwise on the sky).
``phase`` is the mean anomaly / 2 pi, counted from periastron; ``omega`` (deg) rotates the
periastron from the +x axis within the plane. A System3D renders one epoch: render several
phases for an orbit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..units import G_CGS, MSUN_G
from .primitives import unit_cm


def _eccentric_anomaly(mean_anomaly, e):
    E = np.where(e < 0.8, mean_anomaly, np.pi)
    for _ in range(50):
        E = E - (E - e * np.sin(E) - mean_anomaly) / (1 - e * np.cos(E))
    return E


@dataclass
class KeplerOrbit:
    """Masses in Msun; give ``period_days`` or the semi-major axis ``a`` (of the relative
    orbit, in ``length_unit``)."""

    m1: float
    m2: float
    period_days: float | None = None
    a: float | None = None
    e: float = 0.0
    omega: float = 0.0  # deg
    length_unit: str | float = "cm"

    def __post_init__(self):
        gm = G_CGS * (self.m1 + self.m2) * MSUN_G
        L = unit_cm(self.length_unit, self.m1 + self.m2)
        if (self.period_days is None) == (self.a is None):
            raise ValueError("give exactly one of period_days and a")
        if self.a is None:
            p = self.period_days * 86400.0
            self.a = (gm * p**2 / (4 * np.pi**2)) ** (1 / 3) / L
        else:
            self.period_days = 2 * np.pi * np.sqrt((self.a * L) ** 3 / gm) / 86400.0
        self._gm, self._L = gm, L

    def state(self, phase):
        """Positions (length units) and velocities (km/s) of both bodies at ``phase``:
        dict with r1, r2, v1, v2 (each (3,) or (n, 3) for an array of phases) and the
        separation r (length units)."""
        ph = np.asarray(phase, float)
        e, a_cm = self.e, self.a * self._L
        E = _eccentric_anomaly(2 * np.pi * np.mod(ph, 1.0), e)
        nu = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2))
        r = a_cm * (1 - e * np.cos(E))
        th = nu + np.deg2rad(self.omega)
        rel = np.stack([r * np.cos(th), r * np.sin(th), np.zeros_like(r)], -1)
        h = np.sqrt(self._gm * a_cm * (1 - e**2))  # specific angular momentum
        vr, vt = self._gm / h * e * np.sin(nu), h / r
        vrel = np.stack([vr * np.cos(th) - vt * np.sin(th), vr * np.sin(th) + vt * np.cos(th),
                         np.zeros_like(r)], -1)
        f1, f2 = self.m2 / (self.m1 + self.m2), self.m1 / (self.m1 + self.m2)
        return dict(r1=-f1 * rel / self._L, r2=f2 * rel / self._L, v1=-f1 * vrel / 1e5,
                    v2=f2 * vrel / 1e5, r=r / self._L)
