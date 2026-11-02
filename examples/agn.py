"""An illustrative Seyfert nucleus built from System3D primitives (parameters are
round numbers for demonstration, not a fit to any source).

A corona at the centre lights an obscuring torus (Compton reflection with Fe K-alpha from
its inner face), a clumpy broad-line region (scattering plus a 6.4 keV line), and
ionisation cones (electron scattering of the nucleus). At 15 Mpc, 1 pc is 13.8 mas.

In the plot, black is no emission: outside the cones there is no gas, and the torus's
outer faces (dashed outline: its whole silhouette) do not see the corona, so only its
inner face lights up. Light scattered once by the cones onto the torus is not modelled
(single scattering).

Run:  python examples/agn.py     (about a minute)
Writes agn.fits and agn.png next to this script.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

import xrimodel as xm
from xrimodel.engine import BiCone, Lamp, Scene, Shell, System3D, Torus
from xrimodel.units import UAS_PER_RAD

here = Path(__file__).parent
corona = {"kind": "powerlaw", "gamma": 1.8, "norm": 0.02}
agn = System3D(
    name="agn", length_unit="pc", inclination=50.0, node_pa=30.0, nh_ism=2e20,
    lamps=[Lamp(name="corona", spectrum=corona)],
    bodies=[Torus(name="torus", r_major=0.6, r_minor=0.3, albedo=0.1, mass_msun=4e7)],
    volumes=[
        Shell(name="blr", r_in=0.005, r_out=0.03, n0=2e6, index=1.0, absorption="xi",
              clumping={"f_vol": 0.05, "size": 0.15, "seed": 1},
              lines=[{"e0": 6.4, "sigma": 0.03, "yield": 0.5}]),
        BiCone(name="cones", r_in=0.05, r_out=3.0, half_angle=35.0, n0=3e4, index=2.0,
               v_r=500.0, absorption="none"),
    ],
    extent=3.2)

grid = xm.RegularGrid(nx=200, ny=200, scale=220.0)  # 0.22 mas pixels
scene = Scene(ra=182.6357, dec=39.4057, distance_pc=15.0e6, grid=grid)
scene.add(agn)
model = scene.render()
xm.write_model(model, here / "agn.fits", overwrite=True)
for L in model.layers:
    print(f"{L.name:14s} {L.mean_photon_flux():.3e} photons/s/cm^2")

# band images of the extended light
ext_layers = [L for L in model.layers if L.grid.n > 1]
pix_sr = (grid.scale / UAS_PER_RAD) ** 2


def band_image(lo, hi):
    tot = np.zeros(grid.n)
    for L in ext_layers:
        _, pix = xm.specmap_projection(L, edges=np.linspace(lo, hi, 41))
        tot += pix
    return grid.to_image(tot) / pix_sr


# outline of the torus (its whole silhouette, lit or not): render it once more with a
# uniformly emitting surface and no reflection
outline = System3D(name="o", length_unit="pc", inclination=agn.inclination,
                   node_pa=agn.node_pa, supersample=1, extent=agn.extent,
                   bodies=[Torus(name="torus", r_major=0.6, r_minor=0.3, reflection=None,
                                 emission={"spectrum": corona, "intensity": 1.0})])
sc_o = Scene(ra=scene.ra, dec=scene.dec, distance_pc=scene.distance_pc, grid=grid)
sc_o.add(outline)
silhouette = grid.to_image(sc_o.render().layer("o_torus_emit").maps["NORM"] > 0)

half = grid.nx / 2 * grid.scale / 1e3
extent = [half, -half, -half, half]
cmap = plt.get_cmap("magma").copy()
cmap.set_bad(cmap(0.0))  # no emission at all: no gas there, or an unlit surface
fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
for a_, (lo, hi), lab in [(ax[0], (2.0, 10.0), "2-10 keV"), (ax[1], (6.3, 6.5), "6.3-6.5 keV")]:
    img = band_image(lo, hi)
    pos = img[img > 0]
    im = a_.imshow(np.where(img > 0, img, np.nan), origin="lower", extent=extent, cmap=cmap,
                   norm=LogNorm(vmin=np.percentile(pos, 1), vmax=pos.max()))
    a_.contour(silhouette.astype(float), levels=[0.5], extent=extent, origin="lower",
               colors="c", linewidths=0.8, linestyles="--")
    a_.set_xlabel("East offset (mas)")
    a_.set_ylabel("North offset (mas)")
    fig.colorbar(im, ax=a_, label=f"{lab} (ph s$^{{-1}}$ cm$^{{-2}}$ sr$^{{-1}}$)", fraction=0.046)
fig.tight_layout()
fig.savefig(here / "agn.png", dpi=130)
