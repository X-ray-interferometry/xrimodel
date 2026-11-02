import numpy as np
import pytest
from astropy.io import fits

import xrimodel as xm

# ------------------------------------------------------------------ grids


def test_quadtree_tiles_root_exactly():
    qt = xm.QuadtreeGrid.refine(100.0, lambda x, y, s: np.hypot(x, y) < 3 * s, max_level=6)
    assert np.isclose(np.sum(qt.sizes() ** 2), 100.0**2)
    assert qt.sizes().min() == pytest.approx(100.0 / 64)
    assert qt.n < 64**2  # adaptive: far fewer leaves than the uniform finest grid


def test_offsets_are_true_angles():
    """Tangent-plane offsets (x East, y North) map to sky positions at the right
    separation and position angle (astropy), at Cyg X-1's declination."""
    from astropy.coordinates import SkyCoord

    x, y = np.array([5000.0, 0.0, -3000.0]), np.array([0.0, 5000.0, 4000.0])
    ra, dec = xm.offsets_to_radec(x, y, 299.59, 35.20)
    c0, c = SkyCoord(299.59, 35.20, unit="deg"), SkyCoord(ra, dec, unit="deg")
    assert np.allclose(c0.separation(c).to_value("uas"), np.hypot(x, y), rtol=1e-6)
    assert np.allclose(c0.position_angle(c).to_value("rad"), np.mod(np.arctan2(x, y), 2 * np.pi),
                       atol=1e-6)


def test_regular_grid_orientation():
    g = xm.RegularGrid(nx=4, ny=2, scale=1.0)
    x, y = g.centres()
    img = g.to_image(x)
    assert img[0, 0] > img[0, -1]  # columns run East -> West, as in sky images
    assert g.to_image(y)[0, 0] < g.to_image(y)[-1, 0]


# ------------------------------------------------------------------ spectra


def test_powerlaw_integral():
    e = xm.energy_grid(1.0, 10.0, 0.01)
    s = xm.powerlaw(e, 2.0, norm=3.0)
    assert s.photon_flux() == pytest.approx(3.0 * (1 - 1 / 10), rel=1e-6)


def test_gaussian_line_flux_and_sampling():
    e = xm.energy_grid(5.0, 8.0, 0.01)
    s = xm.gaussian_line(e, 6.4, 0.05, flux=2.0)
    assert s.photon_flux() == pytest.approx(2.0, rel=1e-6)
    draw = s.sample(20000, np.random.default_rng(0))
    assert draw.mean() == pytest.approx(6.4, abs=2e-3)


# ------------------------------------------------------------------ I/O


def _model_all_layers(grid):
    e = xm.energy_grid(0.5, 10.0, 0.05)
    m = xm.Model(ra=299.59, dec=35.20, mjdref=60000.0)
    m.add_driver(xm.LightCurve([0, 0.5], [0.5, 1.5], name="orbit", periodic=True, period=1000.0))
    f = np.geomspace(1e-3, 1.0, 50)
    m.add_driver(xm.PSDDriver(f, xm.broken_powerlaw_psd(f, 0.2, 0.05), name="flicker", seed=7))
    m.add(xm.SeparableLayer("pt", xm.powerlaw(e, 1.7, 1e-2), x=12.5, y=-3.25, driver="flicker"))
    rng = np.random.default_rng(1)
    img = rng.random(grid.n)
    m.add(xm.SeparableLayer("ext", xm.blackbody(e, 0.4, 1e-2), grid, img, driver="orbit"))
    maps = {"NORM": img * 1e-4, "G": 0.8 + 0.4 * rng.random(grid.n), "DELAY": 10 * rng.random(grid.n),
            "NH": 1e22 * rng.random(grid.n), "SIGMA": 0.01 * np.ones(grid.n)}
    m.add(xm.SpecMapLayer("smap", grid, [xm.gaussian_line(e, 6.4, 0.02)], maps, driver="flicker"))
    cube = rng.random((e.size - 1, grid.n)) * 1e-5
    cube[:, :3] = 0.0
    m.add(xm.SpecCubeLayer("cube", grid, e, cube))
    m.add(xm.PhotonLayer("phl", time=np.sort(rng.random(100) * 50), x=rng.normal(0, 5, 100),
                         y=rng.normal(0, 5, 100), energy=rng.uniform(1, 9, 100), area=50.0,
                         tstart=0.0, tstop=50.0))
    return m


@pytest.mark.parametrize("grid", [
    xm.RegularGrid(nx=24, ny=16, scale=1.5, x0=10.0, y0=-5.0),
    xm.QuadtreeGrid.refine(40.0, lambda x, y, s: np.hypot(x, y) < 2 * s, max_level=5, x0=3.0),
])
def test_roundtrip_all_layer_types(tmp_path, grid):
    m = _model_all_layers(grid)
    m.spec = {"hello": [1, 2, 3]}
    path = tmp_path / "m.fits"
    xm.write_model(m, path)
    r = xm.read_model(path)
    assert [la.name for la in r.layers] == [la.name for la in m.layers]
    assert set(r.drivers) == {"orbit", "flicker"}
    assert r.spec == {"hello": [1, 2, 3]}
    for a, b in zip(m.layers, r.layers):
        assert b.kind == a.kind
        assert b.mean_photon_flux() == pytest.approx(a.mean_photon_flux(), rel=1e-5)
    pt = r.layer("pt")
    assert (pt.x, pt.y) == (12.5, -3.25)
    sm, sm0 = r.layer("smap"), m.layer("smap")
    for p in ("G", "DELAY", "NH"):
        assert np.allclose(sm.maps[p], sm0.maps[p], rtol=1e-6)
    cb, cb0 = r.layer("cube"), m.layer("cube")
    nz = cb0.cube > 0
    assert np.all(cb.cube[~nz] == 0)
    assert np.max(np.abs(cb.cube[nz] / cb0.cube[nz] - 1)) < 4e-4
    ph = r.layer("phl")
    assert np.allclose(ph.x, m.layer("phl").x) and ph.area == 50.0


def test_simput_row_flux_matches_layer(tmp_path):
    """SIMPUT-only readers normalise by FLUX: it must equal the layer's energy flux."""
    grid = xm.RegularGrid(nx=24, ny=16, scale=1.5)
    m = _model_all_layers(grid)
    path = tmp_path / "m.fits"
    xm.write_model(m, path)
    cat = fits.getdata(path, "SRC_CAT")
    ext = m.layer("ext")
    i = list(cat["SRC_NAME"]).index("ext")
    assert cat["FLUX"][i] == pytest.approx(ext.spectrum.energy_flux(cat["E_MIN"][i], cat["E_MAX"][i]),
                                           rel=1e-5)
    assert cat["IMAGE"][i].startswith("[IMG_ext")


def test_read_plain_simput(tmp_path):
    """A SIMPUT 1.1 file from another tool (point source + extended source with image)."""
    energy = np.linspace(0.5, 10, 200)
    flux_density = energy**-2.0
    spec = fits.BinTableHDU.from_columns([
        fits.Column("NAME", "8A", array=["s1"]),
        fits.Column("ENERGY", "PE()", array=np.array([energy], dtype=object)),
        fits.Column("FLUXDENSITY", "PE()", array=np.array([flux_density], dtype=object))],
        name="SPECTRUM")
    img = fits.ImageHDU(np.ones((10, 20)), name="IMAGE")
    for k, v in dict(CTYPE1="RA---TAN", CTYPE2="DEC--TAN", CRVAL1=10.0, CRVAL2=20.0,
                     CRPIX1=10.5, CRPIX2=5.5, CDELT1=-1e-9, CDELT2=1e-9).items():
        img.header[k] = v
    cat = fits.BinTableHDU.from_columns([
        fits.Column("SRC_ID", "J", array=[1, 2]),
        fits.Column("RA", "D", array=[10.0, 10.0]), fits.Column("DEC", "D", array=[20.0, 20.0]),
        fits.Column("E_MIN", "E", array=[2.0, 2.0]), fits.Column("E_MAX", "E", array=[10.0, 10.0]),
        fits.Column("FLUX", "E", array=[1e-11, 2e-11]),
        fits.Column("SPECTRUM", "32A", array=["[SPECTRUM,1][NAME=='s1']"] * 2),
        fits.Column("IMAGE", "32A", array=["NULL", "[IMAGE,1]"]),
        fits.Column("TIMING", "32A", array=["NULL", "NULL"])], name="SRC_CAT")
    path = tmp_path / "plain.simput"
    fits.HDUList([fits.PrimaryHDU(), cat, spec, img]).writeto(path)
    m = xm.read_model(path)
    assert m.layers[0].grid is None
    assert m.layers[0].spectrum.energy_flux(2.0, 10.0) == pytest.approx(1e-11, rel=1e-6)
    g = m.layers[1].grid
    assert g.scale == pytest.approx(3.6, rel=1e-6)  # 1e-9 deg = 3.6 uas
    assert abs(g.x0) < 1e-3 and abs(g.y0) < 1e-3


def test_simput_row_sits_on_image_reference(tmp_path):
    """SIMPUT puts the image reference pixel at the row's RA/DEC: they must agree."""
    grid = xm.RegularGrid(nx=10, ny=6, scale=2.0, x0=500.0, y0=-300.0)
    m = xm.Model(ra=83.6, dec=22.0)
    m.add(xm.SeparableLayer("off", xm.powerlaw(xm.energy_grid(), 2.0), grid, np.ones(grid.n)))
    xm.write_model(m, tmp_path / "o.fits")
    cat = fits.getdata(tmp_path / "o.fits", "SRC_CAT")
    hdr = fits.getheader(tmp_path / "o.fits", "IMG_off")
    assert cat["RA"][0] == pytest.approx(hdr["CRVAL1"], abs=1e-12)
    assert cat["DEC"][0] == pytest.approx(hdr["CRVAL2"], abs=1e-12)
    assert hdr["CRPIX1"] == (10 + 1) / 2 and hdr["CRPIX2"] == (6 + 1) / 2
