"""Model engine (needs JAX): build a :class:`Scene` from components and render it."""

from .binary import XRBinary
from .components import COMPONENTS, GaussianBlob, PointSource, ThinDisc
from .scene import Scene
from .spec import build_spectrum

COMPONENTS[XRBinary.kind] = XRBinary

__all__ = ["COMPONENTS", "GaussianBlob", "PointSource", "Scene", "ThinDisc", "XRBinary",
           "build_spectrum"]
