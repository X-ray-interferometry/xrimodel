"""The notebooks' XRISALIS helpers now live in XRISALIS as :mod:`xrisalis.recipes`
(``observe``, ``subset``, ``scaled``, ``dirty_beam``, ``model_image``, ``predicted``,
``show``, ...), with the instrument as an argument (``setup=``). This module re-exports
them so the notebooks keep their ``import xri_helpers as xri``; without XRISALIS (or with
a version before ``recipes``) the import fails and the notebooks skip their XRISALIS cells.
"""

from xrisalis.recipes import *
from xrisalis.recipes import bundled as _bundled

_setup = _bundled()
ARF_10BL, RMF_, INSTRUMENT = _setup.arf, _setup.rmf, _setup.instrument
