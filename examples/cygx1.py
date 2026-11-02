"""Cyg X-1 rebuilt from 3-D parts, at three orbital phases.

System parameters from Miller-Jones et al. (2021, Science 371, 1046): D = 2.22 kpc,
M_BH = 21.2 Msun, M_donor = 40.6 Msun, R_donor = 22.3 Rsun, i = 27.5 deg, P = 5.6 d,
Omega = 64 deg. Wind (Mdot = 3e-6 Msun/yr, v_inf = 2100 km/s, beta = 1.05), ISM column
(5.4e21 cm^-2) and corona flux (F(0.5-10 keV) ~ 7.4e-9 erg/s/cm^2, Gamma = 1.7) from
Hanke et al. (2009, ApJ 690, 330). The thermal disc (diskbb, kT = 0.3 keV) normalisation
is illustrative. For accurate disc-reflection fluxes, which come from within ~1 uas of the
compact object, render on a QuadtreeGrid refined towards it (see tests/test_binary.py).

Run:  python examples/cygx1.py    (a few minutes; uses XSPEC models if xspectrampoline is
installed: pexmon for reflection, diskbb for the disc)
Writes cygx1_phase*.fits and cygx1_phase0.50.png next to this script.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

import xrimodel as xm
from xrimodel.engine import Scene, XRBinary
from xrimodel.units import UAS_PER_RAD

here = Path(__file__).parent
gamma = 1.7
norm = 7.4e-9 / xm.powerlaw(xm.energy_grid(), gamma, 1.0).energy_flux(0.5, 10.0)
try:
    import xspectrampoline  # noqa: F401

    disc = {"kind": "xspec", "model": "C_diskbb", "params": [0.3], "norm": 2e4}
except ImportError:
    disc = {"kind": "blackbody", "kt": 0.3, "norm": 1.0}

grid = xm.RegularGrid(nx=320, ny=320, scale=1.0)  # 1 uas pixels, compact object at 0
scene = Scene(ra=299.590315, dec=35.201604, distance_pc=2220.0, grid=grid)
f = np.geomspace(1e-3, 1.0, 300)
scene.add_driver(xm.PSDDriver(f, xm.broken_powerlaw_psd(f, rms=0.3, f_break=0.05),
                              name="corona", seed=7))

for phase in (0.0, 0.25, 0.5):
    scene.components = []
    scene.add(XRBinary("cygx1", {"kind": "powerlaw", "gamma": gamma, "norm": norm},
                       phase=phase, driver="corona", disc_spectrum=disc))
    model = scene.render()
    xm.write_model(model, here / f"cygx1_phase{phase:.2f}.fits", overwrite=True)
    print(f"phase {phase:.2f}:", ", ".join(f"{L.name} {L.mean_photon_flux():.2e}"
                                         for L in model.layers), "photons/s/cm^2")

# quick look at phase 0.5: band images of the extended layers, delay map, spectra
m = xm.read_model(here / "cygx1_phase0.50.fits")
ext_layers = [L for L in m.layers if L.grid.n > 1]
g = ext_layers[0].grid
ext = [g.x0 + g.nx/2*g.scale, g.x0 - g.nx/2*g.scale, g.y0 - g.ny/2*g.scale, g.y0 + g.ny/2*g.scale]
pix_sr = (g.scale / UAS_PER_RAD) ** 2
def band_image(lo, hi, layers):
    tot = np.zeros(g.n)
    for L in layers:
        _, pix = xm.specmap_projection(L, edges=np.linspace(lo, hi, 41))
        tot += pix
    return g.to_image(tot) / pix_sr  # photons/s/cm2/sr
broad = band_image(0.5, 10.0, ext_layers)
fe = band_image(6.3, 6.5, ext_layers)
w = sum(L.maps["NORM"] for L in ext_layers); wd = sum(L.maps["NORM"] * L.maps["DELAY"] for L in ext_layers)
delay = g.to_image(np.where(w > 0, wd / np.where(w > 0, w, 1), np.nan))
fig, ax = plt.subplots(1, 4, figsize=(19, 4.2), gridspec_kw={"width_ratios": [1, 1, 1, 1.1]})
for a_, img, lab, kw in [(ax[0], broad, "0.5-10 keV (ph s$^{-1}$ cm$^{-2}$ sr$^{-1}$)", {}),
                         (ax[1], fe, "6.3-6.5 keV (ph s$^{-1}$ cm$^{-2}$ sr$^{-1}$)", {}),
                         (ax[2], delay, "delay after corona (s)", {"cmap": "viridis"})]:
    pos = img[np.isfinite(img) & (img > 0)]
    norm = LogNorm(vmin=np.percentile(pos, 2), vmax=pos.max()) if "delay" not in lab else None
    im = a_.imshow(img, origin="lower", extent=ext, norm=norm, cmap=kw.get("cmap", "magma"))
    a_.plot(0, 0, "+", color="c", ms=10, mew=1.5)
    fig.colorbar(im, ax=a_, label=lab, fraction=0.046)
    a_.set_xlabel("East offset (µas)"); a_.set_ylabel("North offset (µas)")
e = xm.energy_grid(0.5, 10.0, 0.02)
for L in m.layers:
    s, _ = xm.specmap_projection(L, edges=e)
    a3 = ax[3]; a3.step(s.centres, s.values * s.centres**2, where="mid", label=L.name.split("_", 1)[1])
ax[3].set_xscale("log"); ax[3].set_yscale("log"); ax[3].set_ylim(1e-5, 30)
ax[3].set_xlabel("Energy (keV)"); ax[3].set_ylabel("E$^2$ N(E) (keV ph s$^{-1}$ cm$^{-2}$ keV$^{-1}$)")
ax[3].legend(frameon=False, fontsize=9)
fig.tight_layout(); fig.savefig(here / "cygx1_phase0.50.png", dpi=110)
