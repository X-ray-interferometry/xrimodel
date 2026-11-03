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
* `engine.System3D` (2 Nov 2026): general scenes of lamps, bodies (Sphere, RocheLobe,
  FlaredDisc, Torus, custom) and volumes (BetaWind, Shell, BiCone, Jet, Loop, custom);
  several illuminators, including extended ones (emitting volumes discretised into point
  illuminators); per-illuminator column/shadow grids, so the illuminating light is
  absorbed and shadowed; focused and clumped winds; lamp motion in all Doppler factors.
  XRBinary is now a preset. Tests against closed forms: thin-sphere scattering fraction
  tau, incident column on limb rays, jet/counter-jet ratio, spot and loop fluxes, sphere
  reflection and the moving-mirror shift, clump mass conservation, ModelSpec round trip.
  `examples/agn.py`. `GasTorus` (absorbing, scattering, optionally clumpy torus) and a SpecMap
  electron-column plane `NE` (Klein-Nishina losses, for Compton-thick and ionised columns).
* `engine.KeplerOrbit` (3 Nov 2026): two-body positions and velocities (any eccentricity) to place
  binaries in a System3D; notebooks 04 (Algol-like eclipsing binary) and 05 (binary SMBH).
* Still to do: image/cube import as engine components, time-dependent clumps, extended
  illuminators that are bodies (surface emission), beamed (non-isotropic) illuminators,
  per-epoch layers (phase 3).

## Next

1. **XRISALIS integration** (branch `xrimodel-integration` on `package-restructure`):
   `XrimodelSource` samples photons with xrimodel and thins them onto baselines by the ARF;
   attitude interpolated per photon; sky positions projected onto the tangent plane at the
   nominal pointing (the cos(dec) fix).
2. **Per-epoch layers** for geometric time evolution (orbits, eclipses, precession).
3. **Roche binary** (3a): XRBinary over the orbit through per-epoch layers (ephemeris,
   eclipses, phase-dependent Doppler, absorption and irradiation delays).
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
