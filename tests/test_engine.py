import numpy as np
import pytest

pytest.importorskip("jax")

import xrimodel as xm
from xrimodel.engine import GaussianBlob, PointSource, Scene, ThinDisc


def _scene(pa=30.0, inc=60.0):
    sc = Scene(ra=266.4, dec=-29.0, distance_pc=8178.0, grid=xm.RegularGrid(64, 64, 4.0))
    sc.add(PointSource("pt", {"kind": "powerlaw", "gamma": 2.0, "norm": 1e-3}, x=3.0))
    sc.add(GaussianBlob("blob", {"kind": "blackbody", "kt": 0.3, "norm": 1e-3}, sigma=10.0, x=20.0))
    sc.add(ThinDisc("disc", {"kind": "gaussian_line", "e0": 6.4, "sigma": 0.005}, flux=1e-4,
                    mass_msun=4.3e6, r_out=40.0, inclination=inc, position_angle=pa))
    return sc


def test_disc_flux_and_kinematics():
    pa = 30.0
    m = _scene(pa=pa).render()
    d = m.layer("disc").maps
    assert d["NORM"].sum() == pytest.approx(1e-4)
    on = d["NORM"] > 0
    assert d["G"][on].min() < 0.9 and d["G"][on].max() > 1.1
    # red- and blueshifted sides lie on opposite ends of the projected major axis
    x, y = m.layer("disc").grid.centres()
    blue, red = on & (d["G"] > 1.05), on & (d["G"] < 0.95)
    sep = np.array([x[blue].mean() - x[red].mean(), y[blue].mean() - y[red].mean()])
    major = np.array([np.sin(np.deg2rad(pa)), np.cos(np.deg2rad(pa))])
    assert abs(sep @ major) / np.linalg.norm(sep) > 0.95


def test_disc_delays_increase_towards_far_side():
    pa = 0.0  # major axis North-South, so the minor axis is East-West
    m = _scene(pa=pa).render()
    d = m.layer("disc").maps
    x, _ = m.layer("disc").grid.centres()
    on = d["NORM"] > 0
    corr = np.corrcoef(x[on], d["DELAY"][on])[0, 1]
    assert abs(corr) > 0.8  # delay tracks the minor-axis coordinate
    rg_c = 6.674e-8 * 4.3e6 * 1.988e33 / 2.998e10**3
    assert d["DELAY"].max() < (np.hypot(40, 10) + 40 + 10) * rg_c


def test_gaussian_blob_normalised_image():
    m = _scene().render()
    blob = m.layer("blob")
    x, y = blob.grid.centres()
    w = blob.image / blob.image.sum()
    assert (w * x).sum() == pytest.approx(20.0, abs=0.05)
    assert np.sqrt((w * (y - (w * y).sum()) ** 2).sum()) == pytest.approx(10.0, rel=0.03)


def test_spec_roundtrip_through_file(tmp_path):
    sc = _scene()
    m = sc.render()
    xm.write_model(m, tmp_path / "s.fits")
    m2 = Scene.from_spec(xm.read_model(tmp_path / "s.fits").spec).render()
    for a, b in zip(m.layers, m2.layers):
        assert b.mean_photon_flux() == pytest.approx(a.mean_photon_flux(), rel=1e-12)
    assert np.allclose(m.layer("disc").maps["G"], m2.layer("disc").maps["G"])


def test_quadtree_scene_renders():
    qt = xm.QuadtreeGrid.refine(256.0, lambda x, y, s: np.hypot(x, y) < 40 + s, max_level=7)
    sc = Scene(ra=266.4, dec=-29.0, distance_pc=8178.0, grid=qt)
    sc.add(ThinDisc("disc", {"kind": "gaussian_line", "e0": 6.4, "sigma": 0.005}, flux=1e-4,
                    mass_msun=4.3e6, r_out=40.0))
    m = sc.render()
    assert m.layer("disc").maps["NORM"].sum() == pytest.approx(1e-4)
    ph = xm.sample_photons(m, 0, 1e6, area=10.0, rng=0)
    assert abs(len(ph) - 1000) < 5 * np.sqrt(1000)


def test_disc_photon_weight_scales_as_g_cubed():
    from xrimodel.engine.components import _disc_kernel

    px = np.array([[0.0, 0.0]])  # minor axis is y' after the PA rotation below
    w, g, _ = _disc_kernel(np.array([[10.0, -10.0]]), px, 6.0, 40.0, 0.5, np.sqrt(0.75),
                           0.0, 3.0, 10.0, False, 1.0)
    w, g = np.asarray(w)[0], np.asarray(g)[0]
    assert w[0] / w[1] == pytest.approx((g[0] / g[1]) ** 3, rel=1e-10)
    assert g[0] != pytest.approx(g[1])
