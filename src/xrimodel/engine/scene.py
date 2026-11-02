"""The :class:`Scene`: where components live and how they become a :class:`Model`."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from ..format.drivers import Driver
from ..format.grid import QuadtreeGrid, RegularGrid
from ..format.layers import Model
from ..format.spectra import energy_grid
from .components import COMPONENTS
from .spec import driver_from_spec, driver_to_spec, jsonable

SPEC_VERSION = 1


@dataclass
class Scene:
    """Sky frame, distance, energy grid, spatial grid, drivers and components.

    ``grid`` is shared by all extended components. ``energy_edges`` (keV) is the grid
    on which spectra and templates are evaluated (default 0.3-12 keV in 50 eV bins).
    """

    ra: float
    dec: float
    distance_pc: float
    grid: RegularGrid | QuadtreeGrid
    energy_edges: np.ndarray = field(default_factory=energy_grid)
    components: list = field(default_factory=list)
    drivers: dict[str, Driver] = field(default_factory=dict)
    mjdref: float = 0.0

    def add(self, component):
        if any(c.name == component.name for c in self.components):
            raise ValueError(f"duplicate component name {component.name!r}")
        self.components.append(component)
        return component

    def add_driver(self, driver: Driver):
        self.drivers[driver.name] = driver
        return driver

    def render(self) -> Model:
        model = Model(ra=self.ra, dec=self.dec, mjdref=self.mjdref, spec=self.to_spec())
        for d in self.drivers.values():
            model.add_driver(d)
        for c in self.components:
            out = c.render(self)
            for layer in out if isinstance(out, list) else [out]:
                model.add(layer)
        model.validate()
        return model

    # ------------------------------------------------------------------ ModelSpec

    def to_spec(self) -> dict:
        if isinstance(self.grid, RegularGrid):
            g = {"kind": "regular", **asdict(self.grid)}
        else:
            g = {"kind": "quadtree", "uniq": self.grid.uniq.tolist(),
                 "root_size": self.grid.root_size, "x0": self.grid.x0, "y0": self.grid.y0}
        return jsonable({
            "xrimodel_spec": SPEC_VERSION,
            "ra": self.ra, "dec": self.dec, "distance_pc": self.distance_pc, "mjdref": self.mjdref,
            "grid": g,
            "energy_edges": np.asarray(self.energy_edges),
            "drivers": [driver_to_spec(d) for d in self.drivers.values()],
            "components": [{"kind": c.kind, **asdict(c)} for c in self.components],
        })

    @classmethod
    def from_spec(cls, spec: dict) -> Scene:
        g = dict(spec["grid"])
        kind = g.pop("kind")
        grid = RegularGrid(**g) if kind == "regular" else QuadtreeGrid(
            uniq=np.asarray(g["uniq"], np.int64), root_size=g["root_size"], x0=g["x0"], y0=g["y0"])
        scene = cls(ra=spec["ra"], dec=spec["dec"], distance_pc=spec["distance_pc"], grid=grid,
                    energy_edges=np.asarray(spec["energy_edges"]), mjdref=spec.get("mjdref", 0.0))
        for d in spec.get("drivers", []):
            scene.add_driver(driver_from_spec(d))
        for c in spec.get("components", []):
            c = dict(c)
            scene.add(COMPONENTS[c.pop("kind")](**c))
        return scene
