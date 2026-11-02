# Roadmap

Design plan: https://claude.ai/artifact/A8xqM1DpaJBzz7ETLUKgz9

## Done in 0.1

* Format: separable, SpecMap, SpecCube and photon layers; regular and quadtree grids;
  light-curve and PSD drivers; MODELSPEC; reads plain SIMPUT 1.1.
* Sampler: exact Poisson thinning with per-pixel delays, G shifts, broadening, absorption.
* Engine: `Scene`, `PointSource`, `GaussianBlob`, `ThinDisc` (Newtonian Keplerian, special-
  relativistic Doppler, optional sqrt(1 - 2/r) redshift, lamp-post delays).
* Adapters: XSPEC spectra and tbabs cross-sections via xspectrampoline; `transfer_function`
  for nDspec.

## Phase 2 (started 2 Nov 2026)

* `engine.raytrace`: line-of-sight ray marching helpers (implicit-solid hits refined by
  bisection inside bounding spheres, shadow tests, normals from field gradients, Doppler).
* `engine.XRBinary`: Roche donor, flared disc (H/R ~ r^(1/8), tapered to zero at r_in),
  corona and disc-blackbody point sources, beta-law wind with an ionisation threshold and
  the donor's shadow; depth compositing and wind absorption (wind split into column bins);
  reflection normalised from the corona (pexmon templates by emission angle, approximate
  fallback); electron-scattering Doppler shifts; light-travel delays; Cyg X-1 example and
  tests (Kepler separation, Roche radius, orbital Doppler, dips at superior conjunction,
  disc shadow, sky rotation sense, convergence on adaptive grids).
* Still to do: a general scene API for user-composed bodies and volumes (XRBinary composes
  them internally), image/cube import as engine components, focused and clumpy winds, wind
  absorption of the illuminating light, per-epoch layers (phase 3).

## Next

1. **XRISALIS integration** (branch `xrimodel-integration` on `package-restructure`):
   `XrimodelSource` samples photons with xrimodel and thins them onto baselines by the ARF;
   attitude interpolated per photon. Still open there: absolute RA/Dec positions and the
   missing cos(dec) in RA offsets.
2. **Per-epoch layers** for geometric time evolution (orbits, eclipses, precession).
3. **Roche binary** (3a): port the colleague's `roche.py` as an implicit-surface component with
   orbital ephemeris, Doppler maps, wind absorption and irradiation delays.
4. **Black hole disc** (3b): kerrz and Gradus.jl producers of `G`/`DELAY` maps, angle-dependent
   reflection templates (`TIDX`), checked against each other and against relline.
5. **GUI** (Panel) on `Scene.to_spec` / `from_spec`.
6. Fitting: JAX gradients through the engine.

## Decisions (2 Nov 2026)

* The `SPECMAP`/`SPECCUBE`/`QTGRID` extensions will eventually be proposed to the SIMPUT/SIXTE
  maintainers. Checking that SIXTE loads xrimodel files is low priority for now.
* The model energy grid is set per file (already the case: each spectrum and template stores
  its own bin edges; `Scene.energy_edges` sets it per scene).
* Polarisation is out of scope.
* XRISALIS's existing source generators (`images.py`) become thin wrappers over xrimodel.
* ModelSim (`model_gen` branch) and XRI-image-gen will be archived.
