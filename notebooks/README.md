# Example notebooks

Executed notebooks (outputs included, so they read on GitHub without running). Each one ends its
examples with simulated XRISALIS observations; the quickstart spells the pipeline out and the others
call `xri_helpers.py` (in this folder), which wraps it. The simulated spacecraft rolls through half a turn over each exposure, and images are reconstructed from the detected energies with `reconstruct`'s `deconvolve` (chromatic decoherence). Without XRISALIS installed those cells skip.

| Notebook | Shows | Runtime |
| --- | --- | --- |
| `01_quickstart.ipynb` | A scene with the engine (corona, reverberating disc, blob); SpecMap planes; the extended SIMPUT file; sampling photons; light curves, the line's lag and the transfer function ψ(E, τ); a model built by hand; then an active star (spots, coronal loops) through XRISALIS step by step (`XrimodelSource` → `generate_photons` → `redistribute` → `reconstruct`), compared with the model convolved with the dirty beam; how the roll samples the uv plane (a line's dirty beam against a continuum's); the effect of the deconvolution, with `xri_helpers` | ~4 min |
| `02_cygx1.ipynb` | The `XRBinary` preset with Cyg X-1's parameters: images at four phases, reprocessed fluxes and the column to the corona around the orbit, spectra, velocity and delay maps, reverberation responses, focused and clumpy winds; XRISALIS images at two phases of all the light, the reprocessed light and the donor alone | ~5 min |
| `03_agn_and_jets.ipynb` | `System3D`: a Seyfert nucleus (opaque torus, clumpy BLR, ionisation cones), a Compton-thick gas torus (smooth and clumpy) and the nucleus's transmitted spectrum against inclination, a Seyfert 2 with a compact gas torus as a scene and through XRISALIS (at its real brightness, and brightened to show what the array resolves), a relativistic jet with knots and its Doppler asymmetry; XRISALIS images of the nucleus and of a compact jet, each against the model convolved with the dirty beam | ~25 min |
| `04_binary_star.ipynb` | An Algol-like eclipsing binary: a Roche-lobe-filling K subgiant with a corona and a loop, an X-ray-dark B dwarf that eclipses it (`KeplerOrbit` for the orbit); X-ray light curve with eclipse and ellipsoidal dips, velocity curve, images through the eclipse, the corona as an extended illuminator, and XRISALIS images at quadrature and ingress | ~9 min |
| `05_binary_smbh.ipynb` | A binary supermassive black hole: two moving coronae with their own drivers, mini-discs and a circumbinary disc lit by both; velocity curves, the two echoes of the circumbinary disc and their delays around the orbit, and XRISALIS images of the pair moving on the sky | ~3 min |

Run them with `jupyter lab` from this folder; they write their FITS files next to themselves.
Parameters are illustrative round numbers unless a notebook cites a source.
