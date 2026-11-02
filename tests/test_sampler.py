import numpy as np
import pytest

import xrimodel as xm
from xrimodel.format.spectra import sigma_photoabs


def _one_pixel_specmap(**maps):
    grid = xm.RegularGrid(nx=1, ny=1, scale=1.0)
    e = xm.energy_grid(1.0, 12.0, 0.01)
    m = xm.Model(ra=0.0, dec=0.0)
    base = {"NORM": np.array([1.0])}
    base.update({k: np.array([v]) for k, v in maps.items()})
    m.add(xm.SpecMapLayer("pix", grid, [xm.gaussian_line(e, 4.0, 0.01)], base))
    return m


def test_point_source_counts_and_energies():
    e = xm.energy_grid(1.0, 10.0, 0.01)
    m = xm.Model(ra=0.0, dec=0.0)
    m.add(xm.SeparableLayer("pt", xm.powerlaw(e, 2.0, norm=1.0), x=5.0, y=-2.0))
    ph = xm.sample_photons(m, 0.0, 100.0, area=10.0, rng=0)
    expected = 0.9 * 10 * 100  # photons/s/cm^2 * cm^2 * s
    assert abs(len(ph) - expected) < 5 * np.sqrt(expected)
    assert np.all(ph.x == 5.0) and np.all(ph.y == -2.0)
    # mean energy of E^-2 on [1, 10]: ln(10) / 0.9
    assert ph.energy.mean() == pytest.approx(np.log(10) / 0.9, rel=0.03)
    assert np.all(np.diff(ph.time) >= 0)


def test_specmap_energy_shift_and_broadening():
    m = _one_pixel_specmap(G=1.5, SIGMA=0.02)
    ph = xm.sample_photons(m, 0.0, 10000.0, area=1.0, rng=1)
    assert ph.energy.mean() == pytest.approx(6.0, abs=0.01)
    assert ph.energy.std() == pytest.approx(np.hypot(6.0 * 0.02, 1.5 * 0.01), rel=0.05)


def test_specmap_absorption_survival():
    nh = 3e22
    m = _one_pixel_specmap(NH=nh)
    ph = xm.sample_photons(m, 0.0, 20000.0, area=1.0, rng=2)
    expected = 20000 * np.exp(-nh * sigma_photoabs(4.0))
    assert abs(len(ph) - expected) < 5 * np.sqrt(expected)


def test_delay_shifts_driver():
    """A step driver switching on at t=100 reaches a pixel delayed by 50 s at t=150."""
    m = _one_pixel_specmap(DELAY=50.0)
    m.add_driver(xm.LightCurve([0.0, 99.999, 100.0, 1000.0], [0.0, 0.0, 2.0, 2.0], name="step"))
    m.layers[0].driver = "step"
    ph = xm.sample_photons(m, 0.0, 400.0, area=10.0, rng=3)
    assert ph.time.min() >= 150.0
    assert len(ph) > 0


def test_extended_layer_positions_follow_image():
    grid = xm.RegularGrid(nx=8, ny=8, scale=2.0)
    img = np.zeros(grid.n)
    img[5] = 1.0
    m = xm.Model(ra=0.0, dec=0.0)
    m.add(xm.SeparableLayer("one", xm.powerlaw(xm.energy_grid(), 2.0, 1.0), grid, img))
    ph = xm.sample_photons(m, 0.0, 10.0, area=1.0, rng=4)
    xc, yc = grid.centres()
    assert np.all(np.abs(ph.x - xc[5]) <= 1.0) and np.all(np.abs(ph.y - yc[5]) <= 1.0)


def test_psd_realisation_rms():
    f = np.linspace(1e-3, 0.5, 500)
    target_rms = 0.1
    psd = np.full(f.size, target_rms**2 / (f[-1] - f[0]))
    d = xm.PSDDriver(f, psd, seed=11, transform="linear").realise(0.0, 2e5)
    assert d.flux.mean() == pytest.approx(1.0, abs=0.01)
    assert d.flux.std() == pytest.approx(target_rms, rel=0.05)
    d2 = xm.PSDDriver(f, psd, seed=11, transform="linear").realise(0.0, 2e5)
    assert np.array_equal(d.flux, d2.flux)  # same seed, same light curve


def test_photon_layer_thinning_and_area_guard():
    m = xm.Model(ra=0.0, dec=0.0)
    n = 10000
    rng = np.random.default_rng(5)
    m.add(xm.PhotonLayer("p", np.sort(rng.random(n) * 100), np.zeros(n), np.zeros(n),
                         np.full(n, 3.0), area=100.0, tstart=0.0, tstop=100.0))
    ph = xm.sample_photons(m, 0.0, 50.0, area=25.0, rng=6)
    assert abs(len(ph) - n * 0.5 * 0.25) < 5 * np.sqrt(n * 0.125)
    with pytest.raises(ValueError):
        xm.sample_photons(m, 0.0, 50.0, area=200.0)


def test_transfer_function_matches_projection():
    grid = xm.RegularGrid(nx=6, ny=5, scale=1.0)
    rng = np.random.default_rng(7)
    e = xm.energy_grid(1.0, 12.0, 0.05)
    lay = xm.SpecMapLayer("s", grid, [xm.gaussian_line(e, 6.4, 0.05)],
                          {"NORM": rng.random(grid.n), "G": 0.7 + 0.6 * rng.random(grid.n),
                           "DELAY": 100 * rng.random(grid.n)})
    psi = xm.transfer_function(lay, e, np.linspace(0, 100, 11))
    spec, _ = xm.specmap_projection(lay, edges=e)
    assert np.allclose(psi.sum(axis=1), spec.values * np.diff(e), rtol=1e-10, atol=1e-14)
    assert psi.sum() == pytest.approx(lay.mean_photon_flux(), rel=1e-6)


def test_shared_psd_driver_is_coherent_across_layers():
    """Regression: a delayed layer must follow the SAME realisation as its driver."""
    f = np.linspace(1e-3, 0.2, 200)
    psd = np.full(f.size, 0.3**2 / (f[-1] - f[0]))
    e = xm.energy_grid(1.0, 10.0, 0.1)
    m = xm.Model(ra=0.0, dec=0.0)
    m.add_driver(xm.PSDDriver(f, psd, name="drv", seed=3))
    m.add(xm.SeparableLayer("direct", xm.powerlaw(e, 2.0, 50.0), driver="drv"))
    grid = xm.RegularGrid(nx=1, ny=1, scale=1.0)
    m.add(xm.SpecMapLayer("echo", grid, [xm.gaussian_line(e, 6.4, 0.05)],
                          {"NORM": np.array([50.0]), "DELAY": np.array([20.0])}, driver="drv"))
    ph = xm.sample_photons(m, 0.0, 2000.0, area=10.0, rng=4)
    tb = np.arange(0.0, 2001.0, 2.0)
    a, _ = np.histogram(ph.time[ph.layer == 0], tb)
    b, _ = np.histogram(ph.time[ph.layer == 1], tb)
    a, b = a - a.mean(), b - b.mean()
    lags = np.arange(-20, 21)
    cc = [np.corrcoef(a[max(0, -k):a.size - max(0, k)], b[max(0, k):b.size - max(0, -k)])[0, 1]
          for k in lags]
    assert lags[int(np.argmax(cc))] * 2.0 == pytest.approx(20.0, abs=2.0)
    assert max(cc) > 0.8


def test_psd_span_gives_one_light_curve_for_any_window():
    f = np.linspace(1e-3, 0.1, 100)
    d = xm.PSDDriver(f, np.full(f.size, 0.01 / 0.099), seed=5, span=(0.0, 1000.0))
    t = np.linspace(300.0, 400.0, 50)
    assert np.allclose(d.prepare(0.0, 500.0)(t), d.prepare(250.0, 1000.0)(t))
    with pytest.raises(ValueError):
        d.prepare(900.0, 1100.0)
