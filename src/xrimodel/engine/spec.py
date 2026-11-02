"""Plain-dict descriptions of spectra and drivers, so a whole scene is JSON (the ModelSpec).

Spectrum specs::

    {"kind": "powerlaw", "gamma": 1.7, "norm": 1e-3}      # photons/s/cm^2/keV at 1 keV
    {"kind": "gaussian_line", "e0": 6.4, "sigma": 0.01, "flux": 1e-4}
    {"kind": "blackbody", "kt": 0.3, "norm": 1e-2}        # peak-bin flux density
    {"kind": "xspec", "model": "C_powerlaw", "params": [1.7], "norm": 1e-3}
    {"kind": "sum", "parts": [spec, spec, ...]}

Driver specs mirror the classes in :mod:`xrimodel.format.drivers` with a ``kind`` key.
"""

from __future__ import annotations

import numpy as np

from ..format import drivers as drv
from ..format import spectra as sp


def build_spectrum(spec: dict, edges) -> sp.Spectrum:
    kind = spec["kind"]
    args = {k: v for k, v in spec.items() if k != "kind"}
    if kind == "powerlaw":
        return sp.powerlaw(edges, **args)
    if kind == "gaussian_line":
        return sp.gaussian_line(edges, **args)
    if kind == "blackbody":
        return sp.blackbody(edges, **args)
    if kind == "xspec":
        from ..adapters.xspec import xspec_spectrum

        return xspec_spectrum(args["model"], args.get("params", []), edges,
                              norm=args.get("norm", 1.0))
    if kind == "sum":
        return sp.add(*[build_spectrum(p, edges) for p in args["parts"]])
    raise ValueError(f"unknown spectrum kind {kind!r}")


def driver_to_spec(d: drv.Driver) -> dict:
    if isinstance(d, drv.Constant):
        return {"kind": "constant", "name": d.name}
    if isinstance(d, drv.LightCurve):
        return {"kind": "lightcurve", "name": d.name, "x": d.x.tolist(), "flux": d.flux.tolist(),
                "periodic": d.periodic, "period": d.period, "phase0": d.phase0,
                "dperiod": list(d.dperiod)}
    if isinstance(d, drv.PSDDriver):
        return {"kind": "psd", "name": d.name, "frequency": d.frequency.tolist(),
                "power": d.power.tolist(), "seed": d.seed, "transform": d.transform, "dt": d.dt,
                "span": list(d.span) if d.span else None}
    raise TypeError(type(d))


def driver_from_spec(s: dict) -> drv.Driver:
    s = dict(s)
    kind = s.pop("kind")
    if kind == "constant":
        return drv.Constant(**s)
    if kind == "lightcurve":
        s["dperiod"] = tuple(s.get("dperiod", ()))
        return drv.LightCurve(**s)
    if kind == "psd":
        return drv.PSDDriver(**s)
    raise ValueError(f"unknown driver kind {kind!r}")


def jsonable(x):
    if isinstance(x, dict):
        return {k: jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    return x
