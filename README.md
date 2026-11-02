# xrimodel

Spectral-timing source models for X-ray interferometry. Models are built from components, stored as **extended SIMPUT** FITS files, and sampled into photon lists (time, position, energy) for simulators such as [XRISALIS](https://github.com/X-ray-interferometry/XRISALIS).

* `xrimodel.format` (numpy + astropy only): layers, drivers, grids, file I/O, photon sampler.
  This is generally is all a simulator needs as a baseline.
* `xrimodel.engine` (needs JAX): a `Scene` of components that renders to a model.
* `xrimodel.adapters`: XSPEC models via [xspectrampoline](https://pypi.org/project/xspectrampoline/)
  (no HEASoft needed); energy-delay transfer functions for [nDspec](https://ndspec.readthedocs.io).

## Install

```bash
pip install -e ".[dev]"          # format + engine + tests
pip install -e ".[engine]"       # without test tools
pip install xspectrampoline      # optional: XSPEC models and tbabs absorption (GPL-3)
```

## Layer types

| Layer | Holds | Use for |
| --- | --- | --- |
| `SeparableLayer` | image x spectrum (point if no image) | point sources, simple extended sources; plain SIMPUT |
| `SpecMapLayer` | rest-frame templates + per-pixel `NORM`, `G`, `DELAY`, `SIGMA`, `NH`, `TIDX` | discs, winds, binaries; kerrz/Gradus maps |
| `SpecCubeLayer` | dense flux per pixel and energy bin | external simulations |
| `PhotonLayer` | pre-sampled photons | Monte Carlo codes, real data |

Any layer can follow a named **driver**: a `LightCurve` (time or periodic phase) or a `PSDDriver` (power spectrum + seed). SpecMap layers add a per-pixel `DELAY`, so reprocessed emission follows its driver with light-travel delays. Objects and surfaces render with grids. These are currently split into `RegularGrid` or the adaptive `QuadtreeGrid` (for when you need higher resolution in specific areas).

## Example

```python
import numpy as np
import xrimodel as xm
from xrimodel.engine import Scene, PointSource, ThinDisc

scene = Scene(ra=266.41683, dec=-29.00781, distance_pc=8178.0,
              grid=xm.RegularGrid(nx=160, ny=160, scale=1.5))          # 1.5 uas pixels
f = np.geomspace(1e-5, 2e-2, 400)
scene.add_driver(xm.PSDDriver(f, xm.broken_powerlaw_psd(f, rms=0.4, f_break=2e-4),
                              name="corona", seed=42))
scene.add(PointSource("corona", {"kind": "powerlaw", "gamma": 2.0, "norm": 2e-3}, driver="corona"))
scene.add(ThinDisc("fe_line", {"kind": "gaussian_line", "e0": 6.4, "sigma": 0.003}, flux=3e-4,
                   mass_msun=4.3e6, r_out=30, inclination=60, position_angle=30, driver="corona"))

xm.write_model(scene.render(), "model.fits")
photons = xm.sample_photons(xm.read_model("model.fits"), 0.0, 86400.0, area=100.0, rng=1)
image = photons.to_xrisalis_image()   # for the current XRISALIS API
```

`examples/quickstart.py` runs this and plots the disc's G map, the photon spectrum and the
light curves (the line lags the corona).

## 3-D scenes

`System3D` composes a scene from named parts in a 3-D frame (a length unit, inclination,
sky position angle and offset) and ray-marches it along the line of sight:

* **lamps**: point X-ray sources, which also illuminate everything else;
* **bodies** (opaque): `Sphere` (rotating, with emitting spots), `RocheLobe`, `FlaredDisc`
  (any axis), `Torus`. The front-most body hides what is behind it; visible surfaces
  reflect the lamps (pexmon or approximate templates by emission angle, or grey) and may
  emit their own light;
* **volumes** (optically thin): `BetaWind` (focused and/or clumped), `Shell`, `GasTorus`
  (a torus of absorbing gas: transmission through a Compton-thick or clumpy torus), `BiCone`,
  `Jet` (Lorentz factor, counter-jet, knots), `Loop` (coronal loops). They scatter and
  reprocess the lamps' light (with emission lines), absorb (neutral, ionisation threshold
  or none), and may emit. An emitting volume can itself illuminate the scene, as seeded
  point illuminators drawn from its emissivity;
* `CustomBody` / `CustomVolume` take your own JAX functions (rendered, but not written to
  the ModelSpec).

Shadows and the absorbing column between each illuminator and every emitter come from a
precomputed grid around the illuminator; those columns are added to the emitter's NH
(absorbing gas) and NE (all electrons: Klein-Nishina scattering out of the line of sight).
Every emitter carries Doppler factors (including the lamps' own motion), light-travel
delays and columns, and reprocessed layers follow their illuminator's driver.

```python
from xrimodel.engine import BiCone, Lamp, Scene, Shell, System3D, Torus

agn = System3D(name="agn", length_unit="pc", inclination=50.0, node_pa=30.0,
               lamps=[Lamp(name="corona", spectrum={"kind": "powerlaw", "gamma": 1.8,
                                                    "norm": 0.02})],
               bodies=[Torus(name="torus", r_major=0.6, r_minor=0.3)],
               volumes=[BiCone(name="cones", r_in=0.05, r_out=3.0, half_angle=35.0,
                               n0=3e4, index=2.0, absorption="none")])
scene = Scene(ra=182.64, dec=39.41, distance_pc=15e6,
              grid=xm.RegularGrid(nx=200, ny=200, scale=220.0))
scene.add(agn)
model = scene.render()   # layers agn_corona, agn_torus, agn_cones0..3
```

`examples/agn.py` adds a clumpy broad-line region and plots the images. The renderer
warns when the electron-scattering optical depth along a ray exceeds ~0.3, where the
single-scattering treatment stops being accurate.

### An X-ray binary

One of our `System3D` presets is an `XRBinary`: a Roche-lobe donor, a flared disc and a point-like corona, plus the donor's beta-law wind (optionally focused towards the compact object, `focus_amp`, and clumped, `clump_f_vol`). This shows off a lot of the functionality already: the front-most body hides what is behind it, every emitter is absorbed by the wind column in front of it, the disc shadows the donor from the corona, and reflected and scattered light is normalised from the corona's photon luminosity. It renders into SpecMap layers with Doppler, delay, column and emission-angle (pexmon template) maps:

```python
from xrimodel.engine import Scene, XRBinary

scene = Scene(ra=299.590315, dec=35.201604, distance_pc=2220.0,
              grid=xm.RegularGrid(nx=320, ny=320, scale=1.0))
scene.add(XRBinary("cygx1", {"kind": "powerlaw", "gamma": 1.7, "norm": 2.0}, phase=0.5))
model = scene.render()   # layers: cygx1_corona, _disc_bb, _star, _disc, _wind0.._wind3
```

The wind is split into layers by the absorbing column in front of the emission, so each layer's per-pixel NH is stacked and accurate even when winds overlap. The disc's reflection comes from within ~1 uas of the compact object so is only an approximation in this case. `examples/cygx1.py` specifically renders what we expect Cyg X-1 (Miller-Jones et al. 2021 system parameters) to look like at three orbital phases in a few minutes.

## Notebooks

`notebooks/` holds executed examples: the format and sampler, Cyg X-1 around its orbit, AGN tori, cones and jets, and an active star simulated end to end through XRISALIS.