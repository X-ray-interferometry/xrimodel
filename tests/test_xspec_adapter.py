import numpy as np
import pytest

pytest.importorskip("xspectrampoline")

import xrimodel as xm
from xrimodel.adapters.xspec import register_tbabs, xspec_spectrum


def test_xspec_powerlaw_matches_builtin():
    e = xm.energy_grid(0.5, 10.0, 0.05)
    s = xspec_spectrum("C_powerlaw", [1.8], e, norm=2.0)
    ref = xm.powerlaw(e, 1.8, norm=2.0)
    assert np.allclose(s.values, ref.values, rtol=1e-6)


def test_tbabs_cross_section_is_sensible():
    sigma = register_tbabs()
    # about 2e-22 cm^2 at 1 keV and falling steeply with energy
    assert 1e-22 < sigma(1.0) < 4e-22
    assert sigma(6.0) < sigma(1.0) / 50
