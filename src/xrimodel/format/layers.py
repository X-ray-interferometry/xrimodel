"""Layer types and the :class:`Model` container.

Every layer is one row of the SIMPUT source catalogue. Its normalisation is its
time-averaged photon flux; a named driver (see :mod:`.drivers`) modulates it in time.

=================  =============================================================
SeparableLayer     image x spectrum x driver (plain SIMPUT 1.1). No image = point.
SpecMapLayer       rest-frame templates + per-pixel maps NORM, G, DELAY, SIGMA, NH,
                   TIDX. Observed photon: E = G * E_rest (E_rest from template TIDX),
                   Gaussian broadening with fractional width SIGMA, survival
                   exp(-NH sigma(E)), arrival t = t_emit + DELAY.
SpecCubeLayer      dense flux density per pixel and energy bin.
PhotonLayer        pre-sampled photons (time, offset, energy) collected over a given
                   area and exposure.
=================  =============================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .drivers import Driver
from .grid import Grid
from .spectra import Spectrum

SPECMAP_PLANES = ("NORM", "G", "DELAY", "SIGMA", "NH", "TIDX")
SPECMAP_DEFAULTS = {"G": 1.0, "DELAY": 0.0, "SIGMA": 0.0, "NH": 0.0, "TIDX": 0}


@dataclass
class SeparableLayer:
    name: str
    spectrum: Spectrum  # absolute: photons/s/cm^2/keV, time-averaged
    grid: Grid | None = None  # None -> point source at (x, y)
    image: np.ndarray | None = None  # relative weights per grid leaf
    x: float = 0.0  # uas, used when grid is None
    y: float = 0.0
    driver: str | None = None

    kind = "separable"

    def __post_init__(self):
        if (self.grid is None) != (self.image is None):
            raise ValueError("give both grid and image, or neither (point source)")
        if self.image is not None:
            self.image = np.asarray(self.image, float).ravel()
            if self.image.size != self.grid.n:
                raise ValueError("image size does not match grid")
            if np.any(self.image < 0) or self.image.sum() <= 0:
                raise ValueError("image weights must be non-negative with a positive sum")

    def mean_photon_flux(self) -> float:
        return self.spectrum.photon_flux()


@dataclass
class SpecMapLayer:
    name: str
    grid: Grid
    templates: list[Spectrum]  # rest-frame, each normalised to unit integral
    maps: dict[str, np.ndarray]  # plane name -> flat array over grid leaves
    driver: str | None = None
    abs_model: str = "powerlaw-approx"

    kind = "specmap"

    def __post_init__(self):
        if "NORM" not in self.maps:
            raise ValueError("SpecMapLayer needs a NORM map (photons/s/cm^2 per pixel)")
        self.templates = [t.normalised() for t in self.templates]
        full = {}
        for p in SPECMAP_PLANES:
            v = self.maps.get(p, SPECMAP_DEFAULTS.get(p))
            v = np.broadcast_to(np.asarray(v, float), (self.grid.n,)).copy() if np.ndim(v) == 0 \
                else np.asarray(v, float).ravel()
            if v.size != self.grid.n:
                raise ValueError(f"map {p} has {v.size} values, grid has {self.grid.n}")
            full[p] = v
        if np.any(full["NORM"] < 0):
            raise ValueError("NORM must be non-negative")
        if np.any(full["G"] <= 0):
            raise ValueError("G must be positive")
        tidx = full["TIDX"]
        if np.any((tidx < 0) | (tidx >= len(self.templates)) | (tidx != np.round(tidx))):
            raise ValueError("TIDX must index the template list")
        self.maps = full

    def mean_photon_flux(self) -> float:
        """Unabsorbed photon flux (absorption is applied when sampling)."""
        return float(self.maps["NORM"].sum())


@dataclass
class SpecCubeLayer:
    name: str
    grid: Grid
    energy_edges: np.ndarray  # keV, n+1
    cube: np.ndarray  # (nE, npix): photons/s/cm^2/keV per pixel
    driver: str | None = None

    kind = "speccube"

    def __post_init__(self):
        self.energy_edges = np.asarray(self.energy_edges, float)
        self.cube = np.asarray(self.cube, float).reshape(self.energy_edges.size - 1, self.grid.n)
        if np.any(self.cube < 0):
            raise ValueError("cube must be non-negative")

    def bin_flux(self) -> np.ndarray:
        """Photons/s/cm^2 per (energy bin, pixel)."""
        return self.cube * np.diff(self.energy_edges)[:, None]

    def mean_photon_flux(self) -> float:
        return float(self.bin_flux().sum())


@dataclass
class PhotonLayer:
    name: str
    time: np.ndarray  # s
    x: np.ndarray  # uas offsets
    y: np.ndarray
    energy: np.ndarray  # keV
    area: float  # cm^2 the photons were collected over
    tstart: float = 0.0
    tstop: float = 0.0

    kind = "photons"

    def __post_init__(self):
        for k in ("time", "x", "y", "energy"):
            setattr(self, k, np.asarray(getattr(self, k), float))
        if not self.tstop:
            self.tstop = float(self.time.max()) if self.time.size else 0.0

    def mean_photon_flux(self) -> float:
        dur = self.tstop - self.tstart
        return float(self.time.size / (self.area * dur)) if dur > 0 else 0.0


Layer = SeparableLayer | SpecMapLayer | SpecCubeLayer | PhotonLayer


@dataclass
class Model:
    """A complete source model: reference position, layers, and the drivers they use."""

    ra: float  # deg, tangent point (REFRA)
    dec: float  # deg (REFDEC)
    layers: list[Layer] = field(default_factory=list)
    drivers: dict[str, Driver] = field(default_factory=dict)
    mjdref: float = 0.0
    spec: dict | None = None  # the ModelSpec (JSON-able) that generated this file

    def add(self, layer: Layer) -> Layer:
        if len(layer.name) > 40 or not layer.name or "'" in layer.name or "," in layer.name:
            raise ValueError("layer names must be 1-40 characters without quotes or commas "
                             f"(they become FITS extension names): {layer.name!r}")
        if any(lay.name == layer.name for lay in self.layers):
            raise ValueError(f"duplicate layer name {layer.name!r}")
        self.layers.append(layer)
        return layer

    def add_driver(self, driver: Driver) -> Driver:
        self.drivers[driver.name] = driver
        return driver

    def layer(self, name: str) -> Layer:
        for lay in self.layers:
            if lay.name == name:
                return lay
        raise KeyError(name)

    def validate(self):
        for lay in self.layers:
            d = getattr(lay, "driver", None)
            if d is not None and d not in self.drivers:
                raise ValueError(f"layer {lay.name!r} refers to unknown driver {d!r}")
