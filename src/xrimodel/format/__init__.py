"""Extended-SIMPUT model files: layers, drivers, grids, I/O and photon sampling.

This subpackage depends only on numpy and astropy, so simulators (XRISALIS) can use
it without the model engine.
"""

from .drivers import (
    Constant,
    LightCurve,
    PSDDriver,
    broken_powerlaw_psd,
    lorentzian_psd,
    powerlaw_psd,
)
from .grid import QuadtreeGrid, RegularGrid, offsets_to_radec, radec_to_offsets
from .io import read_model, specmap_projection, write_model
from .layers import Model, PhotonLayer, SeparableLayer, SpecCubeLayer, SpecMapLayer
from .sampler import Photons, sample_photons, transfer_function
from .spectra import (
    Spectrum,
    blackbody,
    cutoffpl,
    diskbb,
    energy_grid,
    gaussian_line,
    powerlaw,
)

__all__ = [
    "Constant",
    "LightCurve",
    "Model",
    "PSDDriver",
    "PhotonLayer",
    "Photons",
    "QuadtreeGrid",
    "RegularGrid",
    "SeparableLayer",
    "SpecCubeLayer",
    "SpecMapLayer",
    "Spectrum",
    "blackbody",
    "broken_powerlaw_psd",
    "cutoffpl",
    "diskbb",
    "energy_grid",
    "gaussian_line",
    "lorentzian_psd",
    "offsets_to_radec",
    "powerlaw",
    "powerlaw_psd",
    "radec_to_offsets",
    "read_model",
    "sample_photons",
    "specmap_projection",
    "transfer_function",
    "write_model",
]
