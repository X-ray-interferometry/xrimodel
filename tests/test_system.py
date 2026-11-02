"""System3D against closed-form results."""

import numpy as np
import pytest

pytest.importorskip("jax")

import xrimodel as xm
from xrimodel.engine import (
    BetaWind,
    GasTorus,
    Jet,
    Lamp,
    Loop,
    Scene,
    Shell,
    Sphere,
    System3D,
    build_spectrum,
)
from xrimodel.engine.primitives import RSUN_CM, clump_factor
from xrimodel.engine.system import NE_PER_H, SIGMA_T
from xrimodel.units import PC_CM, UAS_PER_RAD

D_PC = 100.0
UAS_PER_RSUN = RSUN_CM / (D_PC * PC_CM) * UAS_PER_RAD  # ~46.5
LAMP = {"kind": "powerlaw", "gamma": 2.0, "norm": 1.0}


def _render(system, grid, distance_pc=D_PC):
    sc = Scene(ra=0.0, dec=0.0, distance_pc=distance_pc, grid=grid)
    sc.add(system)
    return sc.render()


def _flux(model, prefix):
    return sum(L.mean_photon_flux() for L in model.layers if L.name.startswith(prefix))


def _centre_grid(half_uas, level=9):
    return xm.QuadtreeGrid.refine(2 * half_uas, lambda x, y, s: np.hypot(x, y) < 4 * s + 0.5,
                                  max_level=level, min_level=5)


@pytest.mark.parametrize("extended", [False, True])
def test_thin_sphere_scatters_tau_of_the_light(extended):
    """A lamp (or a small emitting ball) at the centre of a uniform, optically thin
    sphere: the scattered fraction is tau = n_e sigma_T (R - r_in)."""
    n0, r_in = 1.5e11, 0.01
    tau = NE_PER_H * n0 * SIGMA_T * (1.0 - r_in) * RSUN_CM
    gas = Shell(name="gas", r_in=r_in, r_out=1.0, n0=n0, absorption="none")
    if extended:
        a, j0 = 1e-3, 1e3
        src = Shell(name="src", r_out=a, n0=1.0, scatter=0.0, absorption="none",
                    emission={"spectrum": LAMP, "j0": j0, "illuminates": True, "n_points": 8})
        system = System3D(name="s", length_unit="Rsun", volumes=[gas, src], extent=1.05)
        n_emit = 4 * np.pi * j0 * (4 / 3) * np.pi * (a * RSUN_CM) ** 3
        ref = n_emit / (4 * np.pi * (D_PC * PC_CM) ** 2)
    else:
        system = System3D(name="s", length_unit="Rsun", lamps=[Lamp(name="lamp", spectrum=LAMP)],
                          volumes=[gas], extent=1.05)
        ref = build_spectrum(LAMP, xm.energy_grid()).photon_flux()
    m = _render(system, _centre_grid(1.1 * UAS_PER_RSUN))
    assert _flux(m, "s_gas") / ref == pytest.approx(tau, rel=0.03)


def test_incident_column_is_added_to_the_scattered_light():
    """Lamp inside a neutral absorbing shell (0.2-0.3 R), scattering shell outside
    (0.5-1 R): rays passing outside the absorber (impact parameter 0.32-0.48 R) see only
    the column between lamp and scatterer, n1 (0.3 - 0.2) R."""
    n1 = 3e22 / (0.1 * RSUN_CM)
    system = System3D(
        name="s", length_unit="Rsun", lamps=[Lamp(name="lamp", spectrum=LAMP)],
        volumes=[Shell(name="abs", r_in=0.2, r_out=0.3, n0=n1, scatter=0.0),
                 Shell(name="sc", r_in=0.5, r_out=1.0, n0=1e9, absorption="none")],
        extent=1.05, column_grid=(48, 96, 256))
    m = _render(system, xm.RegularGrid(nx=48, ny=48, scale=2.2 * UAS_PER_RSUN / 48))
    layers = [L for L in m.layers if L.name.startswith("s_sc")]
    x, y = layers[0].grid.centres()
    b = np.hypot(x, y) / UAS_PER_RSUN
    w = sum(L.maps["NORM"] for L in layers)
    wnh = sum(L.maps["NORM"] * L.maps["NH"] for L in layers)
    sel = (b > 0.32) & (b < 0.48)
    assert wnh[sel].sum() / w[sel].sum() == pytest.approx(3e22, rel=0.03)


@pytest.mark.parametrize("profile", ["uniform", "gaussian"])
def test_gas_torus_column_to_the_nucleus(profile):
    """Edge-on, the line of sight from the nucleus crosses the tube once along its
    diameter: NH = n0 2a (uniform) or n0 a sqrt(pi/2) erf(sqrt 2) (gaussian, sigma = a/2,
    cut at a), and NE = 1.2 NH; face-on there is no column."""
    from scipy.special import erf

    n0, a = 1e24 / (0.6 * RSUN_CM), 0.3
    nh = n0 * a * RSUN_CM * (2.0 if profile == "uniform" else np.sqrt(np.pi / 2) * erf(np.sqrt(2)))
    for inc, expect in ((90.0, nh), (0.0, 0.0)):
        system = System3D(name="s", length_unit="Rsun", inclination=inc,
                          lamps=[Lamp(name="lamp", spectrum=LAMP)],
                          volumes=[GasTorus(name="torus", r_major=1.0, r_minor=a, n0=n0,
                                            profile=profile)])
        m = _render(system, xm.RegularGrid(nx=8, ny=8, scale=40.0))
        lamp = m.layer("s_lamp")
        assert lamp.maps["NH"][0] == pytest.approx(expect, rel=1e-3, abs=1e18)
        assert lamp.maps["NE"][0] == pytest.approx(NE_PER_H * expect, rel=1e-3, abs=1e18)


def test_jet_to_counterjet_ratio():
    """Steady two-sided jet at 60 deg to the line of sight: the approaching side is
    brighter by ((1 + beta cos i) / (1 - beta cos i))^2."""
    gam, inc = 2.0, 60.0
    beta = np.sqrt(1 - 1 / gam**2)
    jet = Jet(name="jet", r0=0.05, r1=1.0, half_angle=10.0, index=0.0, lorentz=gam,
              emission={"spectrum": LAMP, "j0": 1.0}, n_fine=4096)
    system = System3D(name="s", length_unit="AU", inclination=inc, volumes=[jet], chunk=1024)
    m = _render(system, xm.RegularGrid(nx=128, ny=128, scale=16.0), distance_pc=1000.0)
    lay = [L for L in m.layers if L.name.startswith("s_jet_emit")]
    x, _ = lay[0].grid.centres()
    w = sum(L.maps["NORM"] for L in lay)
    mu = beta * np.cos(np.deg2rad(inc))
    assert w[x > 0].sum() / w[x < 0].sum() == pytest.approx(((1 + mu) / (1 - mu)) ** 2, rel=0.03)
    g = sum(L.maps["NORM"] * L.maps["G"] for L in lay)
    assert g[x > 0].sum() / w[x > 0].sum() == pytest.approx(1 / (gam * (1 - mu)), rel=1e-3)


@pytest.mark.parametrize("beta", [0.0, 0.1])
def test_sphere_reflection(beta):
    """A sphere lit by a distant lamp between it and the observer: at full phase
    mu_i = mu_e, so I = albedo C F / 2 and the flux is I pi R^2 / D^2; a mirror moving
    towards the observer shifts the light by (1 + beta) / (1 - beta)."""
    from xrimodel.engine.raytrace import REFLECTION_C

    d = 200.0
    star = Sphere(name="star", radius=1.0, albedo=0.5, reflection="grey",
                  velocity=(0.0, 0.0, beta * 2.99792458e5))
    system = System3D(name="s", length_unit="Rsun", bodies=[star], extent=1.05,
                      lamps=[Lamp(name="lamp", spectrum=LAMP, position=(0.0, 0.0, d),
                                  visible=False)], supersample=4)
    m = _render(system, xm.RegularGrid(nx=64, ny=64, scale=2.2 * UAS_PER_RSUN / 64))
    lay = m.layer("s_star")
    g = (lay.maps["NORM"] * lay.maps["G"]).sum() / lay.maps["NORM"].sum()
    assert g == pytest.approx((1 + beta) / (1 - beta), rel=1e-3)
    if beta == 0.0:
        nph = 4 * np.pi * (D_PC * PC_CM) ** 2 * build_spectrum(LAMP, xm.energy_grid()).photon_flux()
        i_r = 0.5 * REFLECTION_C * nph / (4 * np.pi * ((d - 1) * RSUN_CM) ** 2) / 2
        assert lay.mean_photon_flux() == pytest.approx(
            i_r * np.pi * RSUN_CM**2 / (D_PC * PC_CM) ** 2, rel=0.02)


def test_spot_flux_is_its_projected_area():
    """A polar spot of angular radius 30 deg facing the observer: flux = I pi (R sin 30)^2 / D^2."""
    i0 = 1e10
    star = Sphere(name="star", radius=1.0, reflection=None,
                  emission={"spectrum": LAMP, "spots": [{"lat": 90, "lon": 0, "radius": 30,
                                                         "intensity": i0}]})
    m = _render(System3D(name="s", length_unit="Rsun", bodies=[star], supersample=4),
                xm.RegularGrid(nx=64, ny=64, scale=2.2 * UAS_PER_RSUN / 64))
    expect = i0 * np.pi * (0.5 * RSUN_CM) ** 2 / (D_PC * PC_CM) ** 2
    assert _flux(m, "s_star_emit") == pytest.approx(expect, rel=0.02)


def test_loop_flux_is_its_emission_measure():
    """A flat loop (height 0) along 60 deg of a great circle: a torus segment of volume
    pi t^2 R omega (Pappus), flux j0 V / D^2."""
    j0, t = 1e5, 0.02
    loop = Loop(name="loop", r_star=1.0, foot1=(0, 0), foot2=(0, 60), height=0.0, thickness=t,
                emission={"spectrum": LAMP, "j0": j0}, n_fine=2048)
    m = _render(System3D(name="s", length_unit="Rsun", inclination=30.0, volumes=[loop],
                         chunk=1024), xm.RegularGrid(nx=96, ny=96, scale=2.4 * UAS_PER_RSUN / 96))
    vol = np.pi * (t * RSUN_CM) ** 2 * RSUN_CM * np.pi / 3
    assert _flux(m, "s_loop_emit") == pytest.approx(j0 * vol / (D_PC * PC_CM) ** 2, rel=0.03)


def test_clumping_conserves_mass():
    import jax.numpy as jnp

    rng = np.random.default_rng(1)
    u = rng.normal(size=(400000, 3))
    p = u / np.linalg.norm(u, axis=1, keepdims=True) * rng.uniform(1, 10, 400000)[:, None]
    fac = np.asarray(clump_factor({"f_vol": 0.1, "size": 0.05, "interclump": 0.05, "seed": 3},
                                  jnp.zeros(3), 1.0)(jnp.asarray(p)))
    assert fac.mean() == pytest.approx(1.0, rel=0.02)
    assert np.mean(fac > 1) == pytest.approx(0.1, rel=0.05)


def test_spec_round_trip(tmp_path):
    system = System3D(
        name="s", length_unit="Rsun", inclination=40.0, lamps=[Lamp(name="lamp", spectrum=LAMP)],
        bodies=[Sphere(name="star", centre=(2.0, 0.0, 0.0), radius=0.5)],
        volumes=[BetaWind(name="wind", star=(2.0, 0.0, 0.0), r_star=0.5, extent=3.0,
                          refine_body="star", clumping={"f_vol": 0.2, "size": 0.1},
                          focus={"towards": (0.0, 0.0, 0.0), "amp": 2.0, "width": 20.0})])
    grid = xm.RegularGrid(nx=12, ny=12, scale=12.0)
    m = _render(system, grid)
    xm.write_model(m, tmp_path / "s.fits")
    m2 = Scene.from_spec(xm.read_model(tmp_path / "s.fits").spec).render()
    assert [L.name for L in m2.layers] == [L.name for L in m.layers]
    for a_, b_ in zip(m.layers, m2.layers):
        assert b_.mean_photon_flux() == pytest.approx(a_.mean_photon_flux(), rel=1e-9)
