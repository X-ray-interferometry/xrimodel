"""Unit conventions used throughout xrimodel.

Internal units (all plain floats, no astropy Quantities in hot paths):

* sky offsets: micro-arcseconds (uas), measured on the tangent plane around a
  reference position. ``x`` points East (RA increasing), ``y`` points North.
* energies: keV
* times: seconds since the file's ``MJDREF`` + ``TIMEZERO``
* photon flux: photons s^-1 cm^-2; flux density: photons s^-1 cm^-2 keV^-1
* physical lengths inside the engine: cm; distances: pc
"""

import numpy as np

UAS_PER_DEG = 3600.0e6
UAS_PER_RAD = np.degrees(1.0) * UAS_PER_DEG
KEV_TO_ERG = 1.602176634e-9
C_CM_S = 2.99792458e10
PC_CM = 3.0856775814913673e18
G_CGS = 6.67430e-8
MSUN_G = 1.98840987e33


def uas_to_deg(x):
    return np.asarray(x) / UAS_PER_DEG


def deg_to_uas(x):
    return np.asarray(x) * UAS_PER_DEG


def uas_to_rad(x):
    return np.asarray(x) / UAS_PER_RAD


def length_to_uas(length_cm, distance_pc):
    """Small-angle conversion of a projected length (cm) at a distance (pc) to uas."""
    return np.asarray(length_cm) / (distance_pc * PC_CM) * UAS_PER_RAD
