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


def test_cutoffpl_and_diskbb_match_xspec():
    e = xm.energy_grid(0.3, 12.0, 0.05)
    cut = xspec_spectrum("C_cutoffPowerLaw", [1.8, 6.0], e, norm=2.0)
    assert np.allclose(xm.cutoffpl(e, 1.8, 6.0, norm=2.0).values, cut.values, rtol=1e-6)
    for t_in in (0.6, 1.6):
        ref = xspec_spectrum("C_diskbb", [t_in], e, norm=100.0)
        bright = ref.values > 1e-4 * ref.values.max()  # XSPEC's own quadrature drifts in the tail
        got = xm.diskbb(e, t_in, norm=100.0).values
        assert np.allclose(got[bright], ref.values[bright], rtol=0.01)
