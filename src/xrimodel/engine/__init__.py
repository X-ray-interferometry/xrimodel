"""Model engine (needs JAX): build a :class:`Scene` from components and render it."""

from .binary import XRBinary
from .components import COMPONENTS, GaussianBlob, PointSource, ThinDisc
from .orbit import KeplerOrbit
from .primitives import (
                         PRIMITIVES,
                         BetaWind,
                         BiCone,
                         CustomBody,
                         CustomVolume,
                         FlaredDisc,
                         GasTorus,
                         Jet,
                         Lamp,
                         Loop,
                         RocheLobe,
                         Shell,
                         Sphere,
                         Torus,
)
from .scene import Scene
from .spec import build_spectrum
from .system import System3D

COMPONENTS[XRBinary.kind] = XRBinary
COMPONENTS[System3D.kind] = System3D

__all__ = ["COMPONENTS", "PRIMITIVES", "BetaWind", "BiCone", "CustomBody", "CustomVolume",
           "FlaredDisc", "GasTorus", "GaussianBlob", "Jet", "KeplerOrbit", "Lamp", "Loop", "PointSource", "RocheLobe",
           "Scene", "Shell", "Sphere", "System3D", "ThinDisc", "Torus", "XRBinary",
           "build_spectrum"]
