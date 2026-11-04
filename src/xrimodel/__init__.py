"""xrimodel: spectral-timing source models for X-ray interferometry.

``xrimodel.format`` (numpy + astropy only) reads, writes and samples model files.
``xrimodel.engine`` (needs JAX) builds models from components.
"""

from ._version_info import version as __version__
from .format import (
    Constant,
    LightCurve,
    Model,
    PhotonLayer,
    Photons,
    PSDDriver,
    QuadtreeGrid,
    RegularGrid,
    SeparableLayer,
    SpecCubeLayer,
    SpecMapLayer,
    Spectrum,
    blackbody,
    broken_powerlaw_psd,
    cutoffpl,
    diskbb,
    energy_grid,
    gaussian_line,
    lorentzian_psd,
    offsets_to_radec,
    powerlaw,
    powerlaw_psd,
    radec_to_offsets,
    read_model,
    sample_photons,
    specmap_projection,
    transfer_function,
    write_model,
)

__all__ = [
    "Constant", "LightCurve", "Model", "PSDDriver", "PhotonLayer", "Photons", "QuadtreeGrid",
    "RegularGrid", "SeparableLayer", "SpecCubeLayer", "SpecMapLayer", "Spectrum", "__version__",
    "blackbody", "broken_powerlaw_psd", "cutoffpl", "diskbb", "energy_grid", "gaussian_line", "lorentzian_psd",
    "offsets_to_radec", "powerlaw", "powerlaw_psd", "radec_to_offsets", "read_model", "sample_photons", "specmap_projection",
    "transfer_function", "write_model",
]
