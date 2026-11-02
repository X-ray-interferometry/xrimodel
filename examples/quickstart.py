"""Quickstart: a variable corona illuminating a reverberating disc, plus a soft blob.

Run:  python examples/quickstart.py
Writes quickstart.fits (extended SIMPUT) and quickstart.png next to this script.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import xrimodel as xm
from xrimodel.engine import GaussianBlob, PointSource, Scene, ThinDisc

here = Path(__file__).parent

# Sgr A*-like geometry: 1 r_g = GM/c^2 is ~5 uas at 8.2 kpc for 4.3e6 Msun
scene = Scene(ra=266.41683, dec=-29.00781, distance_pc=8178.0,
              grid=xm.RegularGrid(nx=160, ny=160, scale=1.5))

# one stochastic driver, shared by the corona and (delayed) by the disc
f = np.geomspace(1e-5, 2e-2, 400)
scene.add_driver(xm.PSDDriver(f, xm.broken_powerlaw_psd(f, rms=0.4, f_break=2e-4),
                              name="corona", seed=42))

scene.add(PointSource("corona", {"kind": "powerlaw", "gamma": 2.0, "norm": 2e-3}, driver="corona"))
scene.add(ThinDisc("fe_line", {"kind": "gaussian_line", "e0": 6.4, "sigma": 0.003}, flux=3e-4,
                   mass_msun=4.3e6, r_in=6, r_out=30, inclination=60, position_angle=30,
                   h_lamp=10, driver="corona"))
scene.add(GaussianBlob("hot_spot", {"kind": "blackbody", "kt": 0.25, "norm": 5e-4},
                       sigma=15.0, x=80.0, y=-60.0))

model = scene.render()
xm.write_model(model, here / "quickstart.fits", overwrite=True)

# sample 1 day on 100 cm^2 from the file, as a simulator would
model = xm.read_model(here / "quickstart.fits")
ph = xm.sample_photons(model, 0.0, 86400.0, area=100.0, rng=1)
print(f"{len(ph)} photons:", {n: int((ph.layer == i).sum()) for i, n in enumerate(ph.layer_names)})

# quick look -----------------------------------------------------------------
disc = model.layer("fe_line")
g = disc.grid
fig, ax = plt.subplots(1, 3, figsize=(13, 4))
ext = [g.x0 + g.nx / 2 * g.scale, g.x0 - g.nx / 2 * g.scale,
       g.y0 - g.ny / 2 * g.scale, g.y0 + g.ny / 2 * g.scale]
gmap = np.where(disc.maps["NORM"] > 0, disc.maps["G"], np.nan)
im = ax[0].imshow(g.to_image(gmap), origin="lower", extent=ext, cmap="coolwarm")
fig.colorbar(im, ax=ax[0], label="G = E_obs / E_rest")
ax[0].set_xlabel("East offset (uas)")
ax[0].set_ylabel("North offset (uas)")

ax[1].hist(ph.energy, bins=np.linspace(0.3, 12, 235), histtype="step", color="k")
ax[1].set_yscale("log")
ax[1].set_xlabel("Energy (keV)")
ax[1].set_ylabel("Photons per bin")

tb = np.arange(0, 86400 + 1, 600.0)
for i, (name, c) in enumerate([("corona", "k"), ("fe_line", "C3")]):
    sel = ph.layer == ph.layer_names.index(name)
    rate, _ = np.histogram(ph.time[sel], tb)
    ax[2].plot(0.5 * (tb[1:] + tb[:-1]) / 3600, rate / rate.mean(), color=c, label=name)
ax[2].set_xlabel("Time (h)")
ax[2].set_ylabel("Rate / mean")
ax[2].legend(frameon=False)
fig.tight_layout()
fig.savefig(here / "quickstart.png", dpi=120)
