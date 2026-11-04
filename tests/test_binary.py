"""The 3-D X-ray binary component against Cyg X-1 numbers and basic geometry."""

import numpy as np
import pytest

pytest.importorskip("jax")

import xrimodel as xm
from xrimodel.engine import Scene, XRBinary
from xrimodel.engine.binary import RSUN_CM
from xrimodel.units import C_CM_S

AU = 1.495978707e13
CORONA = {"kind": "powerlaw", "gamma": 1.7, "norm": 2.0}


def _render(phase=0.25, n=40, scale=8.0, x0=-40.0, y0=20.0, **kw):
    grid = xm.RegularGrid(nx=n, ny=n, scale=scale, x0=x0, y0=y0)
    sc = Scene(ra=299.59, dec=35.20, distance_pc=2220.0, grid=grid)
    b = XRBinary("cyg", CORONA, phase=phase, supersample=1, n_steps=96, n_fine=64, n_body=64, **kw)
    sc.add(b)
    return sc.render(), b


def test_separation_matches_miller_jones_2021():
    b = XRBinary("cyg", CORONA)
    assert b.separation_cm() / AU == pytest.approx(0.244, abs=0.002)


def test_corona_doppler_follows_the_orbit():
    """The compact object's line-of-sight speed is 2 pi a_bh sin(i) / P at quadrature and
    zero at conjunction (phase 0)."""
    b = XRBinary("cyg", CORONA)
    a_bh = b.separation_cm() * b.m_donor / (b.m_donor + b.m_compact)
    k_bh = 2 * np.pi * a_bh * np.sin(np.deg2rad(b.inclination)) / (b.period_days * 86400)
    m, _ = _render(phase=0.25, n=8)
    g = m.layer("cyg_corona").maps["G"][0]
    assert abs(g - 1) * C_CM_S == pytest.approx(k_bh, rel=0.01)
    m0, _ = _render(phase=0.0, n=8)
    assert abs(m0.layer("cyg_corona").maps["G"][0] - 1) * C_CM_S < 0.01 * k_bh


def test_wind_column_to_corona_peaks_at_superior_conjunction():
    _, b0 = _render(phase=0.0, n=8)
    _, b5 = _render(phase=0.5, n=8)
    assert b0._last_frame["col_bh"] > 3 * b5._last_frame["col_bh"]


def test_reflection_flux_scale_and_phase_dependence():
    m5, b = _render(phase=0.5)
    m0, _ = _render(phase=0.0)
    corona = m5.layer("cyg_corona").mean_photon_flux()
    star5, star0 = m5.layer("cyg_star").mean_photon_flux(), m0.layer("cyg_star").mean_photon_flux()
    f = b._last_frame
    # fraction of corona photons intercepted by the donor (sphere of radius R at distance a)
    hit = 0.5 * (1 - np.sqrt(1 - (22.3 * RSUN_CM / f["a"]) ** 2))
    assert 0.1 * b.albedo * hit < star5 / corona < 2 * b.albedo * hit
    assert star5 > 2 * star0  # the lit face points at us at phase 0.5


def test_disc_shadows_the_donor():
    thin, _ = _render(phase=0.5, disc_h_over_r=0.01)
    thick, _ = _render(phase=0.5, disc_h_over_r=0.3)
    assert thick.layer("cyg_star").mean_photon_flux() < 0.8 * thin.layer("cyg_star").mean_photon_flux()


def test_delays_are_bounded_by_light_travel():
    m, b = _render(phase=0.25)
    f = b._last_frame
    tau_max = 2 * (f["a"] + 22.3 * RSUN_CM) / C_CM_S
    for name in ("cyg_star", "cyg_disc"):
        mp = m.layer(name).maps
        on = mp["NORM"] > 0
        assert np.all((mp["DELAY"][on] >= 0) & (mp["DELAY"][on] <= tau_max))


def test_donor_follows_the_orbit_counterclockwise_on_the_sky():
    """The donor's light centre sits where the orbit puts it (within a stellar radius:
    the lit cap leans towards the compact object), and with the angular momentum towards
    the observer (i < 90 deg) its position angle increases with phase."""
    pas = []
    for phase in (0.35, 0.45, 0.55):
        m, b = _render(phase=phase, n=48, scale=6.0)
        f = b._last_frame
        star = m.layer("cyg_star")
        x, y = star.grid.centres()
        w = star.maps["NORM"] / star.maps["NORM"].sum()
        cx, cy = (w * x).sum(), (w * y).sum()
        u_a, u_b = b.sky_axes()
        d = f["donor"] - f["bh"]
        expect = (d @ f["e_a"] * u_a + d @ f["e_b"] * u_b) / f["cm_per_uas"]
        assert np.hypot(cx - expect[0], cy - expect[1]) < 22.3 * RSUN_CM / f["cm_per_uas"]
        pas.append(np.arctan2(cx, cy))
    assert np.all(np.diff(np.unwrap(pas)) > 0)


def test_reflection_falls_back_without_xspec(monkeypatch):
    import sys

    from xrimodel.engine.binary import reflection_templates

    monkeypatch.setitem(sys.modules, "xrimodel.adapters.xspec", None)
    tmpls, mu_edges = reflection_templates(xm.energy_grid(), 1.7, (0.0, 0.5, 1.0))
    assert len(tmpls) == 1 and list(mu_edges) == [0.0, 1.0]
    t = tmpls[0]
    i64, i55 = np.argmin(abs(t.centres - 6.4)), np.argmin(abs(t.centres - 5.5))
    assert t.values[i64] > 5 * t.values[i55]  # Fe K-alpha line on the reflected continuum


def test_disc_reflection_converges_on_an_adaptive_grid():
    """The disc's reflection is concentrated within a few r_in (~1 uas): refine the grid
    towards the compact object and the flux converges."""
    out = []
    for level in (8, 10):
        qt = xm.QuadtreeGrid.refine(64.0, lambda x, y, s: np.hypot(x, y) < 4 * s + 2.0,
                                    max_level=level, min_level=4)
        sc = Scene(ra=299.59, dec=35.20, distance_pc=2220.0, grid=qt)
        sc.add(XRBinary("cyg", CORONA, phase=0.5, n_steps=64, n_fine=32))
        m = sc.render()
        out.append(m.layer("cyg_disc").mean_photon_flux() / m.layer("cyg_corona").mean_photon_flux())
    assert out[0] == pytest.approx(out[1], rel=0.03)


def test_low_mass_binary_without_wind_and_with_an_adc():
    """Without the donor's wind nothing inside the system absorbs the corona, and a
    thin accretion-disc corona of radial Thomson depth tau scatters ~tau of the corona's
    photons (less what the disc hides of its far side)."""
    tau = 0.05
    kw = dict(m_donor=0.6, m_compact=1.7, period_days=9.8445, inclination=62.5, r_donor=7.0,
              wind=False, r_disc_out_rg=2e6, disc_h_over_r=0.05, adc={"r_out": 0.05, "tau": tau},
              nh_ism=0.0, supersample=1, n_steps=96, n_fine=64, n_body=64)
    grid = xm.RegularGrid(nx=48, ny=48, scale=0.1)
    sc = Scene(ra=326.17, dec=38.32, distance_pc=7200.0, grid=grid)
    b = sc.add(XRBinary("cx2", CORONA, phase=0.5, **kw))
    m = sc.render()
    names = [L.name for L in m.layers]
    assert not any(n.startswith("cx2_wind") for n in names)
    assert b._last_frame["col_bh"] < 1e18
    adc = sum(L.mean_photon_flux() for L in m.layers if L.name.startswith("cx2_adc"))
    assert 0.4 * tau < adc / m.layer("cx2_corona").mean_photon_flux() < tau
