# xrimodel file format 0.1 (extended SIMPUT)

An xrimodel file is a valid [SIMPUT 1.1](https://hea-www.harvard.edu/heasarc/formats/simput-1.1.0.pdf)
FITS file. Every layer of a model is one `SRC_CAT` row. Layers SIMPUT cannot express carry a
separable approximation in the standard columns, so SIMPUT-only tools (e.g. SIXTE) still get
the right total flux, spectrum and band image. The exact layer is stored in an extension named
in the extra `LAYER` column, which those tools ignore.

## Conventions

| Quantity | Unit / convention |
| --- | --- |
| Sky offsets | uas on the tangent plane at `REFRA`/`REFDEC` (primary header); x East, y North |
| Energies | keV |
| Times | s since `MJDREF` (TT); `TIMEZERO = 0` |
| Flux density | photon s^-1 cm^-2 keV^-1 |
| Per-pixel flux (`NORM`) | photon s^-1 cm^-2 integrated over the pixel, time-averaged |

A float64 RA in degrees only resolves ~0.2 uas, so photon lists and point sources also store
their offsets (`X_OFF`, `Y_OFF`, uas) and readers should prefer them.

## Primary header

`XRIFMT` (format version), `XRIMODEL` (writer version), `REFRA`, `REFDEC` (deg), `MJDREF`.

## SRC_CAT

The standard SIMPUT columns, plus:

| Column | Meaning |
| --- | --- |
| `LAYTYPE` | `separable`, `specmap`, `speccube` or `photons` |
| `LAYER` | EXTNAME of the exact layer extension (empty for plain separable layers) |
| `X_OFF`, `Y_OFF` | uas offsets of point sources / centroids |

`FLUX` always equals the energy flux of the row's `SPECTRUM` between `E_MIN` and `E_MAX`, so the
SIMPUT renormalisation is a no-op. For `specmap` and `speccube` rows, `SPECTRUM` is the
pixel-summed observed spectrum and `IMAGE` is the band image (regular grids only; for adaptive
grids `IMAGE` is `NULL` and RA/DEC is the flux centroid). `TIMING` points at the layer's driver.

## SPECTRUM

One row per spectrum: `NAME`, `ENERGY` (bin centres), `FLUXDENSITY`, and `E_EDGES` (n+1 bin
edges, keV). With `E_EDGES` present the spectrum is piecewise constant in each bin. Templates
of a SpecMap layer are rows named `<layer>:T<k>`, normalised to unit photon flux.

## Drivers (TIMING)

* `LC_<name>`: SIMPUT `LIGHTCURVE` (`TIME` or `PHASE` + `FLUX`, `PERIODIC`, `PERIOD`, `PHASE0`,
  `DPERIOD`), linearly interpolated, mean ~1.
* `PSD_<name>`: SIMPUT `POWSPEC` (`FREQUENCY` Hz, `POWER` fractional rms^2/Hz) plus `SEED`,
  `XFORM` (`exponential` = log-normal, default, or `linear`), optional `DT` (default
  1/(8 f_max)) and optional `SPANSTRT`/`SPANSTOP`. Realised with Timmer & Koenig at sampling
  time. With a span, the realisation covers exactly that interval, so every sampling window
  inside it sees the same light curve; without one, each sampling call realises the driver
  once over the union of what its layers need. All layers sharing a driver always see one
  realisation, so delayed layers stay coherent with it.

Each driver extension carries `DRVNAME`. Several layers can share one driver.

## SPECMAP (`SMAP_<layer>`)

Rest-frame templates plus per-pixel planes:

| Plane | Meaning | Default |
| --- | --- | --- |
| `NORM` | photon flux per pixel before absorption | required |
| `G` | energy shift, E_obs = G E_rest (Doppler x gravitational) | 1 |
| `DELAY` | light-travel delay relative to the driver (s) | 0 |
| `SIGMA` | fractional Gaussian broadening (unresolved velocity spread in the pixel) | 0 |
| `NH` | absorbing column along the line of sight (cm^-2) | 0 |
| `TIDX` | which template (0-based) | 0 |
| `NE` | electron column along the line of sight (cm^-2): Klein-Nishina scattering out of the beam | 0 |

Regular grid: a 3-D image (plane, y, x) with a TAN WCS on axes 1-2 and `PLANEn` keywords.
Adaptive grid: a BINTABLE with `NUNIQ` and one column per plane. Header: `NTEMPL`, `TEMPLk`,
`ABSMODEL` (`powerlaw-approx` or `tbabs`).

A photon from pixel p: arrival time t with rate NORM_p x area x D(t - DELAY_p); E_rest drawn
from template TIDX_p; E = G_p E_rest (1 + SIGMA_p N(0,1)); kept with probability
exp(-NH_p sigma(E) - NE_p sigma_KN(E)). `NH` counts only gas that absorbs (neutral or
weakly ionised), `NE` all electrons in front, so Compton-thick and ionised columns lose
photons to scattering too. Files without an `NE` plane read with NE = 0.

## SPECCUBE (`SCUBE_<layer>` + `SCEB_<layer>`)

Flux density per pixel and energy bin. Regular grid: tile-compressed (GZIP_2) int16 image
(E, y, x), log-quantised: value = 10^(`LOGZERO` + (q + 32767) `LOGSTEP`), q = `QZERO` (-32768)
meaning 0; ten decades below the maximum, relative error < 2e-4. Adaptive grid: BINTABLE
`NUNIQ` + vector `FLUX`. Energy bins in `SCEB_<layer>` (`E_MIN`, `E_MAX`).

## PHOTONS (`PHL_<layer>`)

SIMPUT `PHOTONS` (`TIME`, `RA`, `DEC`, `ENERGY`) plus `X_OFF`, `Y_OFF`, and header `AREA` (cm^2),
`TSTART`, `TSTOP`. Sampling a smaller area thins the list; a larger one is an error.

## Adaptive grids (QTGRID)

Tables on an adaptive grid carry `GRIDTYPE = 'QUADTREE'`, `HDUCLAS2 = 'QTGRID'`, `QTROOT` (root
side, uas), `GRIDX0`, `GRIDY0` (root centre). Leaf ids: `NUNIQ = 4 * 4^level + morton(ix, iy)`
with ix East, iy North, cell side `QTROOT / 2^level`. Separable layers on adaptive grids use a
`QIMG_<layer>` table (`NUNIQ`, `WEIGHT`).

## MODELSPEC

A uint8 image holding the UTF-8 JSON scene description that produced the file;
`xrimodel.engine.Scene.from_spec` rebuilds the scene from it.

## Not yet in 0.1

Per-epoch layers for geometric time evolution (orbits, eclipses): planned as a `LIGHTCURVE`
whose rows reference one layer set per epoch/phase, interpolated by the sampler.
