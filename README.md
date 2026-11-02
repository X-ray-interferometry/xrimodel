# xrimodel

Spectral-timing source models for X-ray interferometry. Models are built from components,
stored as **extended SIMPUT** FITS files, and sampled into photon lists (time, position,
energy) for simulators such as [XRISALIS](https://github.com/X-ray-interferometry/XRISALIS).

* `xrimodel.format` (numpy + astropy only): layers, drivers, grids, file I/O, photon sampler.
  This is all a simulator needs.
* `xrimodel.engine` (needs JAX): a `Scene` of components that renders to a model.
* `xrimodel.adapters`: XSPEC models via [xspectrampoline](https://pypi.org/project/xspectrampoline/)
  (no HEASoft needed); energy-delay transfer functions for [nDspec](https://ndspec.readthedocs.io).

Status: 0.1, first cut. See `docs/format.md` for the file format and `docs/roadmap.md` for
what comes next.

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

Any layer can follow a named **driver**: a `LightCurve` (time or periodic phase) or a
`PSDDriver` (power spectrum + seed). SpecMap layers add a per-pixel `DELAY`, so reprocessed
emission follows its driver with light-travel delays. Grids are `RegularGrid` or adaptive
`QuadtreeGrid`.

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

## 3-D scenes: an X-ray binary

`XRBinary` builds a binary from 3-D parts and ray-marches it along the line of sight: a
Roche-lobe donor, a flared disc and a point-like corona (opaque bodies or point
sources), plus the donor's beta-law wind (an absorbing, scattering volume). The front-most
body hides what is behind it, every emitter is absorbed by the wind column in front of it,
the disc shadows the donor from the corona, and reflected and scattered light is
normalised from the corona's photon luminosity. It renders into SpecMap layers with
Doppler, delay, column and emission-angle (pexmon template) maps:

```python
from xrimodel.engine import Scene, XRBinary

scene = Scene(ra=299.590315, dec=35.201604, distance_pc=2220.0,
              grid=xm.RegularGrid(nx=320, ny=320, scale=1.0))
scene.add(XRBinary("cygx1", {"kind": "powerlaw", "gamma": 1.7, "norm": 2.0}, phase=0.5))
model = scene.render()   # layers: cygx1_corona, _disc_bb, _star, _disc, _wind0.._wind3
```

The wind is split into layers by the absorbing column in front of the emission, so each
layer's per-pixel NH stays accurate. The disc's reflection comes from within ~1 uas of the
compact object: use a `QuadtreeGrid` refined towards it for accurate disc fluxes.
`examples/cygx1.py` rebuilds Cyg X-1 (Miller-Jones et al. 2021 system parameters) at
three orbital phases in a few minutes.

## Tests

```bash
pytest
```

The statistical tests check counts, energy shifts, broadening, absorption, delays, PSD rms,
transfer functions, and file round trips for every layer type on both grid types.
