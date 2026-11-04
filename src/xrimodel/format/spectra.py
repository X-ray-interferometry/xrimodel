"""Binned photon spectra.

A :class:`Spectrum` is a photon flux density on energy bins: ``values[i]`` is the mean
flux density (photons s^-1 cm^-2 keV^-1) in ``[edges[i], edges[i+1])``. Within a bin the
spectrum is taken as constant when sampling. A *template* is a spectrum whose integral
is 1, i.e. a probability density in energy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..units import KEV_TO_ERG


@dataclass
class Spectrum:
    edges: np.ndarray  # keV, length n+1, ascending
    values: np.ndarray  # photons/s/cm^2/keV, length n
    name: str = ""

    def __post_init__(self):
        self.edges = np.asarray(self.edges, dtype=float)
        self.values = np.asarray(self.values, dtype=float)
        if self.edges.ndim != 1 or self.values.shape != (self.edges.size - 1,):
            raise ValueError("Spectrum needs n+1 edges and n values")
        if np.any(np.diff(self.edges) <= 0):
            raise ValueError("Spectrum edges must be strictly ascending")
        if np.any(self.values < 0):
            raise ValueError("Spectrum values must be non-negative")

    @property
    def centres(self) -> np.ndarray:
        return 0.5 * (self.edges[1:] + self.edges[:-1])

    @property
    def widths(self) -> np.ndarray:
        return np.diff(self.edges)

    def photon_flux(self, emin=None, emax=None) -> float:
        """Photons s^-1 cm^-2 between emin and emax (default: whole spectrum)."""
        return float(np.sum(self._bin_overlap(emin, emax) * self.values))

    def energy_flux(self, emin=None, emax=None) -> float:
        """erg s^-1 cm^-2 between emin and emax (bin-centre approximation)."""
        return float(np.sum(self._bin_overlap(emin, emax) * self.values * self.centres) * KEV_TO_ERG)

    def _bin_overlap(self, emin, emax):
        lo = self.edges[:-1] if emin is None else np.clip(self.edges[:-1], emin, None)
        hi = self.edges[1:] if emax is None else np.clip(self.edges[1:], None, emax)
        return np.clip(hi - lo, 0.0, None)

    def cdf_at(self, e) -> np.ndarray:
        """Cumulative photon flux from the first edge up to energy ``e`` (piecewise linear)."""
        cum = np.concatenate([[0.0], np.cumsum(self.values * self.widths)])
        return np.interp(e, self.edges, cum, left=0.0, right=cum[-1])

    def normalised(self) -> Spectrum:
        total = self.photon_flux()
        if total <= 0:
            raise ValueError("cannot normalise a spectrum with zero flux")
        return Spectrum(self.edges, self.values / total, self.name)

    def scaled(self, factor: float) -> Spectrum:
        return Spectrum(self.edges, self.values * factor, self.name)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw ``n`` photon energies (keV), uniform within each bin."""
        p = self.values * self.widths
        p = p / p.sum()
        b = rng.choice(p.size, size=n, p=p)
        return self.edges[b] + rng.random(n) * self.widths[b]


# --------------------------------------------------------------------------- constructors


def energy_grid(emin=0.3, emax=12.0, de=0.05) -> np.ndarray:
    """Linear bin edges; the default (~50 eV) oversamples a CCD-like response by ~2."""
    n = round((emax - emin) / de)
    return np.linspace(emin, emax, n + 1)


def _mean_over_bins(edges, f, sub=8):
    """Mean of f over each bin by Gauss-Legendre quadrature (``sub`` points)."""
    xg, wg = np.polynomial.legendre.leggauss(sub)
    lo, hi = edges[:-1, None], edges[1:, None]
    e = 0.5 * (hi - lo) * xg[None, :] + 0.5 * (hi + lo)
    return 0.5 * np.sum(f(e) * wg[None, :], axis=1)


def powerlaw(edges, gamma, norm=1.0, name="powerlaw") -> Spectrum:
    """N(E) = norm * E^-gamma photons/s/cm^2/keV (norm at 1 keV, as XSPEC powerlaw)."""
    return Spectrum(edges, _mean_over_bins(np.asarray(edges, float), lambda e: norm * e**-gamma), name)


def cutoffpl(edges, gamma, e_cut, norm=1.0, name="cutoffpl") -> Spectrum:
    """N(E) = norm * E^-gamma * exp(-E / e_cut) photons/s/cm^2/keV (as XSPEC cutoffpl)."""
    return Spectrum(edges, _mean_over_bins(np.asarray(edges, float),
                                           lambda e: norm * e**-gamma * np.exp(-e / e_cut)), name)


# 2 / (h^3 c^2) in photons / (s cm^2 sr keV^3), times (1 km / 10 kpc)^2, times 8 pi / 3
_H_KEV_S, _C_CM_S, _KPC_CM = 4.135667696e-18, 2.99792458e10, 3.0856775814913673e21
_DISKBB_K = 8 * np.pi / 3 * 2 / (_H_KEV_S**3 * _C_CM_S**2) * (1e5 / (10 * _KPC_CM)) ** 2


def diskbb(edges, t_in, norm=1.0, name="diskbb") -> Spectrum:
    """Multicolour disc blackbody (Mitsuda et al. 1984; XSPEC diskbb): T ~ r^-3/4 from the
    inner temperature ``t_in`` (keV) outwards, ``norm`` = (R_in / km / (D / 10 kpc))^2 cos i
    as in XSPEC. Photons/s/cm^2/keV."""
    u = np.linspace(-14.0, 0.0, 401)  # ln(T / t_in) along the disc
    x = np.exp(u)

    def f(e):
        # integral over T of (T/t_in)^-11/3 B_E(T) dT/t_in, written in u = ln(T/t_in)
        y = e[..., None] / (t_in * x)
        integrand = np.where(y < 700, x ** (-8.0 / 3.0) / np.expm1(np.minimum(y, 700)), 0.0)
        return norm * _DISKBB_K * e**2 * np.trapezoid(integrand, u, axis=-1)

    return Spectrum(edges, _mean_over_bins(np.asarray(edges, float), f), name)


def gaussian_line(edges, e0, sigma, flux=1.0, name="line") -> Spectrum:
    """Gaussian line with total photon flux ``flux`` (photons/s/cm^2), sigma in keV."""
    from math import erf, sqrt

    edges = np.asarray(edges, float)
    cdf = np.array([0.5 * (1 + erf((e - e0) / (sqrt(2) * sigma))) for e in edges])
    return Spectrum(edges, flux * np.diff(cdf) / np.diff(edges), name)


def blackbody(edges, kt, norm=1.0, name="bbody") -> Spectrum:
    """Photon spectrum shape E^2 / (exp(E/kT) - 1), scaled so the peak bin equals ``norm``."""
    def f(e):
        return e**2 / np.expm1(e / kt)

    v = _mean_over_bins(np.asarray(edges, float), f)
    return Spectrum(edges, norm * v / v.max(), name)


def add(*spectra: Spectrum, name="sum") -> Spectrum:
    edges = spectra[0].edges
    for s in spectra[1:]:
        if not np.array_equal(s.edges, edges):
            raise ValueError("can only add spectra on the same energy grid")
    return Spectrum(edges, np.sum([s.values for s in spectra], axis=0), name)


# --------------------------------------------------------------------------- absorption


def sigma_photoabs(e_kev) -> np.ndarray:
    """Approximate photoelectric cross-section per H atom (cm^2), solar abundances.

    A single power law, sigma = 2.4e-22 (E/keV)^(-8/3), roughly following Morrison &
    McCammon (1983) between 0.5 and 10 keV but without edges. This is a placeholder:
    swap in a tabulated tbabs cross-section for science work.
    """
    return 2.4e-22 * np.asarray(e_kev, float) ** (-8.0 / 3.0)


SIGMA_T = 6.6524587e-25  # cm^2
ME_KEV = 510.99895


def sigma_kn(e_kev) -> np.ndarray:
    """Klein-Nishina total cross-section per electron (cm^2): electron scattering out of
    the line of sight, applied with a SpecMap's electron column ``NE``."""
    x = np.asarray(e_kev, float) / ME_KEV
    x = np.maximum(x, 1e-6)
    small = x < 1e-3
    a = (1 + x) / x**3 * (2 * x * (1 + x) / (1 + 2 * x) - np.log1p(2 * x))
    b = np.log1p(2 * x) / (2 * x) - (1 + 3 * x) / (1 + 2 * x) ** 2
    full = 0.75 * (a + b)
    series = 1 - 2 * x + 5.2 * x**2
    return SIGMA_T * np.where(small, series, full)


ABS_MODELS = {"powerlaw-approx": sigma_photoabs}


def cross_section(name: str):
    """Look up an absorption cross-section function by name (registers 'tbabs' on demand)."""
    if name not in ABS_MODELS and name == "tbabs":
        from ..adapters.xspec import register_tbabs

        register_tbabs()
    try:
        return ABS_MODELS[name]
    except KeyError:
        raise KeyError(f"unknown absorption model {name!r}; known: {sorted(ABS_MODELS)}") from None
