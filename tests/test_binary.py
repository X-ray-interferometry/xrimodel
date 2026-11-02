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


def test_donor_volume_equivalent_radius():
    b = XRBinary("cyg", CORONA)
    f = b.frame()
    phi, phi_s, t_l1, axis = b._roche(f)
    g = np.linspace(-t_l1, t_l1, 90)
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    d = f["donor"] / f["a"]
    pts = d + np.stack([X, Y, Z], -1).reshape(-1, 3)
    inside = ((np.asarray(phi(pts)) < phi_s) & ((pts - d) @ axis < t_l1)
              & (np.linalg.norm(pts - d, axis=-1) < t_l1))
    r_eq = (3 * inside.sum() * (g[1] - g[0]) ** 3 / (4 * np.pi)) ** (1 / 3) * f["a"] / RSUN_CM
    assert r_eq == pytest.approx(22.3, rel=0.02)


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


def test_donor_appears_where_the_orbit_puts_it():
    m, b = _render(phase=0.5, n=48, scale=6.0)
    f = b._last_frame
    star = m.layer("cyg_star")
    x, y = star.grid.centres()
    w = star.maps["NORM"] / star.maps["NORM"].sum()
    u_a, u_b = b.sky_axes()
    d = f["donor"] - f["bh"]
    expect = (d @ f["e_a"] * u_a + d @ f["e_b"] * u_b) / f["cm_per_uas"]
    r_star_uas = 22.3 * RSUN_CM / f["cm_per_uas"]
    # the lit cap is biased towards the compact object, but within a stellar radius
    assert np.hypot((w * x).sum() - expect[0], (w * y).sum() - expect[1]) < r_star_uas


def test_binary_round_trips_through_modelspec(tmp_path):
    m, _ = _render(phase=0.3, n=16)
    xm.write_model(m, tmp_path / "b.fits")
    m2 = Scene.from_spec(xm.read_model(tmp_path / "b.fits").spec).render()
    for a_, b_ in zip(m.layers, m2.layers):
        assert b_.mean_photon_flux() == pytest.approx(a_.mean_photon_flux(), rel=1e-9)


def test_reflection_falls_back_without_xspec(monkeypatch):
    import sys

    from xrimodel.engine.binary import reflection_templates

    monkeypatch.setitem(sys.modules, "xrimodel.adapters.xspec", None)
    tmpls, mu_edges = reflection_templates(xm.energy_grid(), 1.7, (0.0, 0.5, 1.0))
    assert len(tmpls) == 1 and list(mu_edges) == [0.0, 1.0]
    t = tmpls[0]
    i64, i55 = np.argmin(abs(t.centres - 6.4)), np.argmin(abs(t.centres - 5.5))
    assert t.values[i64] > 5 * t.values[i55]  # Fe K-alpha line on the reflected continuum


def test_orbit_runs_counterclockwise_on_the_sky():
    """Angular momentum towards the observer (i < 90 deg): the donor's position angle,
    seen from the compact object, increases with phase."""
    pas = []
    for phase in (0.35, 0.45, 0.55):
        m, _ = _render(phase=phase, n=48, scale=6.0)
        star = m.layer("cyg_star")
        x, y = star.grid.centres()
        w = star.maps["NORM"] / star.maps["NORM"].sum()
        pas.append(np.degrees(np.arctan2((w * x).sum(), (w * y).sum())))
    d = np.diff(np.unwrap(np.radians(pas)))
    assert np.all(d > 0)


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


def test_wind_layers_split_by_column():
    m, b = _render(phase=0.0, n=24)
    winds = [L for L in m.layers if L.name.startswith("cyg_wind")]
    assert len(winds) >= 2
    edges = list(b.wind_nh_bins) + [np.inf]
    for L in winds:
        k = int(L.name[-1])
        nh = L.maps["NH"][L.maps["NORM"] > 0] - b.nh_ism
        assert np.all((nh >= edges[k] * 0.999) & (nh <= edges[k + 1] * 1.001))
