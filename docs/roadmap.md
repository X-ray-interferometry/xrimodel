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

## Next

1. **XRISALIS phase 0** (in the XRISALIS repo): real photon times with interpolated pointing,
   RA/Dec and unit fixes, fringes from TRUE energy, SIMPUT IMAGE/LIGHTCURVE/POWSPEC reader
   (or simply `xrimodel.read_model` + `sample_photons`).
2. **Per-epoch layers** for geometric time evolution (orbits, eclipses, precession).
3. **Roche binary** (3a): port the colleague's `roche.py` as an implicit-surface component with
   orbital ephemeris, Doppler maps, wind absorption and irradiation delays.
4. **Black hole disc** (3b): kerrz and Gradus.jl producers of `G`/`DELAY` maps, angle-dependent
   reflection templates (`TIDX`), checked against each other and against relline.
5. **GUI** (Panel) on `Scene.to_spec` / `from_spec`.
6. Fitting: JAX gradients through the engine.

## Open questions

* Propose `SPECMAP`/`SPECCUBE`/`QTGRID` to the SIMPUT/SIXTE maintainers?
* One fixed model energy grid or per file (currently per scene, default 0.3-12 keV, 50 eV)?
* Polarisation (SIMPUT `STOKES` keywords) in scope?
