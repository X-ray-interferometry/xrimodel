"""XSPEC models as spectra/templates via ``xspectrampoline`` (no HEASoft install needed).

``xspectrampoline`` ships the XSPEC model library as wheels (Linux x86-64, macOS).
It is GPL-3.0-or-later, so it is an optional extra here (``pip install xrimodel[xspec]``)
rather than a core dependency of the MIT-licensed package.

Example::

    from xrimodel.adapters.xspec import xspec_spectrum
    edges = energy_grid(0.3, 12.0, 0.05)
    s = xspec_spectrum("C_powerlaw", [1.7], edges)          # norm 1 at 1 keV
    t = xspec_spectrum("C_powerlaw", [1.7], edges).normalised()  # as a template
"""

from __future__ import annotations

import numpy as np

from ..format.spectra import Spectrum

_LIB = None


def _library():
    global _LIB
    if _LIB is None:
        try:
            import xspectrampoline
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError("install the optional dependency: pip install xspectrampoline") from exc
        _LIB = xspectrampoline.get_libraries()
    return _LIB


def xspec_spectrum(model: str, params, edges, norm: float = 1.0, name: str | None = None) -> Spectrum:
    """Evaluate an XSPEC library model on bin ``edges`` (keV).

    ``model`` is the library symbol (e.g. ``"C_powerlaw"``); ``params`` are the model
    parameters without the XSPEC ``norm``, which multiplies the result. XSPEC models
    return photons/cm^2/s per bin; this divides by the bin width to give a flux density.
    """
    edges = np.ascontiguousarray(edges, dtype=np.float64)
    fn = _library().get_model(model)
    out = np.zeros(edges.size - 1, dtype=np.float64)
    err = np.zeros(edges.size - 1, dtype=np.float64)
    fn(edges, np.ascontiguousarray(params, dtype=np.float64), out, err)
    return Spectrum(edges, norm * np.clip(out, 0.0, None) / np.diff(edges), name or model)


def register_tbabs(emin=0.05, emax=100.0, nbins=4000):
    """Register ``abs_model='tbabs'`` (Wilms et al. 2000) for SpecMap NH absorption.

    The cross-section per H atom is tabulated as sigma(E) = -ln T(E; nH=1e22) / 1e22 on a
    fine log grid from XSPEC's ``tbabs`` (abundances as currently set in the library,
    ``angr`` by default; XSPEC recommends ``wilm``), then interpolated in log-log.
    """
    from ..format.spectra import ABS_MODELS

    edges = np.geomspace(emin, emax, nbins + 1)
    out = np.zeros(nbins)
    err = np.zeros(nbins)
    _library().get_model("C_tbabs")(edges, np.array([1.0]), out, err)
    ec = np.sqrt(edges[1:] * edges[:-1])
    sig = -np.log(np.clip(out, 1e-300, 1.0)) / 1e22
    ok = sig > 0
    le, ls = np.log(ec[ok]), np.log(sig[ok])

    def sigma_tbabs(e):
        return np.exp(np.interp(np.log(np.asarray(e, float)), le, ls))

    ABS_MODELS["tbabs"] = sigma_tbabs
    return sigma_tbabs


def pexmon_templates(edges, gamma=1.7, fold_e=150.0, mu_edges=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
                     abund=1.0, fe_abund=1.0):
    """Reflection-only pexmon spectra (Nandra et al. 2007), one per emission-angle bin.

    Returns (templates, mu_edges): template k is pexmon with rel_refl = -1 evaluated at the
    inclination arccos(mu_mid_k), for use as SpecMap templates indexed by TIDX. Only the
    shapes matter (templates are normalised when used).
    """
    mu_edges = np.asarray(mu_edges, dtype=float)
    mids = 0.5 * (mu_edges[1:] + mu_edges[:-1])
    out = []
    for k, mu in enumerate(mids):
        inc = float(np.degrees(np.arccos(np.clip(mu, 0.02, 0.98))))
        out.append(xspec_spectrum("C_pexmon", [gamma, fold_e, -1.0, 0.0, abund, fe_abund, inc],
                                  edges, name=f"pexmon_mu{k}"))
    return out, mu_edges
