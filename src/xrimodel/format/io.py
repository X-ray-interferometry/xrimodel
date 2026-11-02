"""Read and write models as extended SIMPUT (FITS).

The file is a valid SIMPUT 1.1 file: every layer is a ``SRC_CAT`` row with a
``SPECTRUM`` (and, where possible, ``IMAGE`` and ``TIMING``) reference. For layers
SIMPUT cannot express, the row carries a separable approximation (pixel-summed
spectrum x band image), so SIMPUT-only readers such as SIXTE still get the right
total flux and morphology. The exact layer lives in an extra extension named in
the ``LAYER`` column, which SIMPUT-only readers ignore. See docs/format.md.
"""

from __future__ import annotations

import json
import re

import numpy as np
from astropy.io import fits

from .._version_info import version as _pkg_version
from ..units import UAS_PER_DEG
from .drivers import Constant, LightCurve, PSDDriver
from .grid import QuadtreeGrid, RegularGrid, offsets_to_radec, radec_to_offsets
from .layers import (
    SPECMAP_PLANES,
    Model,
    PhotonLayer,
    SeparableLayer,
    SpecCubeLayer,
    SpecMapLayer,
)
from .spectra import Spectrum, cross_section

FORMAT_VERSION = "0.1"
# [EXT,1], [EXT,1][NAME=='x'] or [EXT,1][#row==n], optionally prefixed by a file name
_REF = re.compile(r"[^\[]*\[([^,\]]+)(?:,\d+)?\](?:\[(?:NAME=='([^']*)'|#row==(\d+))\])?")
_NULL = "NULL"


# =========================================================================== helpers


def _ref(extname, name=None):
    return f"[{extname},1]" + (f"[NAME=='{name}']" if name is not None else "")


def _parse_ref(s):
    """Return (extname, selector) where selector is a row name (str), a 1-based row (int)
    or None."""
    s = (s or "").strip()
    if not s or s.upper() == _NULL:
        return None, None
    if not s.startswith("[") and "[" not in s:
        raise ValueError(f"cannot parse SIMPUT reference {s!r}")
    if not s.startswith("["):
        raise NotImplementedError(f"references to other files are not supported: {s!r}")
    m = _REF.match(s)
    if not m:
        raise ValueError(f"cannot parse SIMPUT reference {s!r}")
    sel = m.group(2) if m.group(2) is not None else (int(m.group(3)) if m.group(3) else None)
    return m.group(1).strip(), sel


def _vla(arrays, fmt="PD()"):
    return np.array([np.asarray(a, float) for a in arrays] + [np.zeros(0)], dtype=object)[:-1], fmt


def specmap_projection(layer: SpecMapLayer, edges=None, chunk=4096):
    """Separable approximation of a SpecMapLayer: (pixel-summed spectrum, per-pixel flux).

    Uses each template's CDF with the per-pixel energy shift G and absorption NH at the
    observed bin centre. Broadening (SIGMA) is ignored here.
    """
    m = layer.maps
    g = m["G"]
    if edges is None:
        lo = min(t.edges[0] for t in layer.templates) * g.min()
        hi = max(t.edges[-1] for t in layer.templates) * g.max()
        edges = np.linspace(lo, hi, 513)
    ec = 0.5 * (edges[1:] + edges[:-1])
    sigma = cross_section(layer.abs_model)(ec)
    spec = np.zeros(ec.size)
    pix = np.zeros(layer.grid.n)
    tidx = m["TIDX"].astype(int)
    for k, tmpl in enumerate(layer.templates):
        sel = np.flatnonzero(tidx == k)
        for s in range(0, sel.size, chunk):
            idx = sel[s:s + chunk]
            cdf = tmpl.cdf_at(edges[None, :] / g[idx, None])  # (npix, nedges)
            binf = np.diff(cdf, axis=1) * m["NORM"][idx, None]
            binf *= np.exp(-m["NH"][idx, None] * sigma[None, :])
            spec += binf.sum(axis=0)
            pix[idx] = binf.sum(axis=1)
    return Spectrum(edges, spec / np.diff(edges), layer.name), pix


def _projection(layer):
    """(absolute spectrum, grid or None, image weights or None) for the SRC_CAT row."""
    if isinstance(layer, SeparableLayer):
        return layer.spectrum, layer.grid, layer.image
    if isinstance(layer, SpecMapLayer):
        spec, pix = specmap_projection(layer)
        return spec, layer.grid, pix
    if isinstance(layer, SpecCubeLayer):
        bf = layer.bin_flux()
        spec = Spectrum(layer.energy_edges, bf.sum(axis=1) / np.diff(layer.energy_edges), layer.name)
        return spec, layer.grid, bf.sum(axis=0)
    raise TypeError(type(layer))


def _image_hdu(extname, grid: RegularGrid, weights, model: Model):
    hdu = fits.ImageHDU(grid.to_image(weights).astype(np.float32), name=extname)
    hdu.header["HDUCLASS"] = "HEASARC/SIMPUT"
    hdu.header["HDUCLAS1"] = "IMAGE"
    hdu.header["HDUVERS"] = "1.1.0"
    for k, v in grid.wcs_header(model.ra, model.dec).items():
        hdu.header[k] = v
    hdu.header["RADESYS"] = "FK5"
    hdu.header["EQUINOX"] = 2000.0
    return hdu


def _simput_header(hdr, clas1):
    hdr["HDUCLASS"] = "HEASARC/SIMPUT"
    hdr["HDUCLAS1"] = clas1
    hdr["HDUVERS"] = "1.1.0"


# =========================================================================== writing


def write_model(model: Model, path, overwrite=False):
    """Write ``model`` to ``path`` as extended SIMPUT."""
    model.validate()
    prim = fits.PrimaryHDU()
    prim.header["XRIMODEL"] = (_pkg_version, "xrimodel version that wrote this file")
    prim.header["XRIFMT"] = (FORMAT_VERSION, "xrimodel extended-SIMPUT format version")
    prim.header["REFRA"] = (model.ra, "[deg] tangent point of all uas offsets")
    prim.header["REFDEC"] = (model.dec, "[deg]")
    prim.header["MJDREF"] = model.mjdref

    hdus, spectra, rows = [], [], []

    # drivers ---------------------------------------------------------------
    timing_ref = {}
    for name, drv in model.drivers.items():
        if isinstance(drv, Constant):
            timing_ref[name] = _NULL
            continue
        if isinstance(drv, LightCurve):
            ext = f"LC_{name}"
            xcol = "PHASE" if drv.periodic else "TIME"
            h = fits.BinTableHDU.from_columns(
                [fits.Column(xcol, "D", array=drv.x), fits.Column("FLUX", "E", array=drv.flux)],
                name=ext)
            _simput_header(h.header, "LIGHTCURVE")
            h.header["MJDREF"] = model.mjdref
            h.header["TIMEZERO"] = 0.0
            h.header["TIMESYS"] = "TT"
            h.header["TIMEUNIT"] = "s"
            h.header["CLOCKCOR"] = "NO"
            h.header["PERIODIC"] = int(drv.periodic)
            if drv.periodic:
                h.header["PERIOD"] = drv.period
                h.header["PHASE0"] = drv.phase0
                for i, d in enumerate(drv.dperiod, start=1):
                    h.header["DPERIOD" if i == 1 else f"D{i}PERIOD"] = d
            else:
                h.header["TSTART"] = float(drv.x.min())
                h.header["TSTOP"] = float(drv.x.max())
        elif isinstance(drv, PSDDriver):
            ext = f"PSD_{name}"
            h = fits.BinTableHDU.from_columns(
                [fits.Column("FREQUENCY", "D", unit="Hz", array=drv.frequency),
                 fits.Column("POWER", "D", unit="Hz**(-1)", array=drv.power)], name=ext)
            _simput_header(h.header, "POWSPEC")
            h.header["SEED"] = (int(drv.seed), "xrimodel: RNG seed of the realisation")
            h.header["XFORM"] = (drv.transform, "xrimodel: exponential or linear transform")
            if drv.dt:
                h.header["DT"] = (drv.dt, "[s] xrimodel: realisation step")
            if drv.span is not None:
                h.header["SPANSTRT"] = (drv.span[0], "[s] xrimodel: realisation start")
                h.header["SPANSTOP"] = (drv.span[1], "[s] xrimodel: realisation stop")
        else:
            raise TypeError(f"cannot write driver of type {type(drv).__name__}")
        h.header["DRVNAME"] = name
        hdus.append(h)
        timing_ref[name] = _ref(ext)

    # layers ----------------------------------------------------------------
    for i, lay in enumerate(model.layers, start=1):
        row = dict(SRC_ID=i, SRC_NAME=lay.name, LAYTYPE=lay.kind, LAYER="",
                   TIMING=timing_ref.get(getattr(lay, "driver", None), _NULL), X_OFF=0.0, Y_OFF=0.0)

        if isinstance(lay, PhotonLayer):
            ext = f"PHL_{lay.name}"
            ra, dec = offsets_to_radec(lay.x, lay.y, model.ra, model.dec)
            h = fits.BinTableHDU.from_columns([
                fits.Column("TIME", "D", unit="s", array=lay.time),
                fits.Column("RA", "D", unit="deg", array=ra),
                fits.Column("DEC", "D", unit="deg", array=dec),
                fits.Column("ENERGY", "E", unit="keV", array=lay.energy),
                fits.Column("X_OFF", "D", unit="uas", array=lay.x),
                fits.Column("Y_OFF", "D", unit="uas", array=lay.y)], name=ext)
            _simput_header(h.header, "PHOTONS")
            for k, v in dict(REFRA=model.ra, REFDEC=model.dec, MJDREF=model.mjdref,
                             TIMEZERO=0.0, TIMESYS="TT", TIMEUNIT="s", CLOCKCOR="NO",
                             TSTART=lay.tstart, TSTOP=lay.tstop, AREA=lay.area,
                             RADESYS="FK5", EQUINOX=2000.0).items():
                h.header[k] = v
            hdus.append(h)
            e = lay.energy
            dur = lay.tstop - lay.tstart
            emin, emax = (float(e.min()), float(e.max())) if e.size else (0.1, 10.0)
            row.update(RA=model.ra, DEC=model.dec, E_MIN=emin, E_MAX=emax,
                       FLUX=float(e.sum() * 1.602176634e-9 / (lay.area * dur)) if dur > 0 else 0.0,
                       SPECTRUM=_ref(ext), IMAGE=_NULL, LAYER=ext)
            rows.append(row)
            continue

        spec, grid, weights = _projection(lay)
        spectra.append(Spectrum(spec.edges, spec.values, lay.name))
        nz = spec.values > 0
        emin = float(spec.edges[:-1][nz].min()) if nz.any() else float(spec.edges[0])
        emax = float(spec.edges[1:][nz].max()) if nz.any() else float(spec.edges[-1])
        row.update(E_MIN=emin, E_MAX=emax, FLUX=spec.energy_flux(emin, emax),
                   SPECTRUM=_ref("SPECTRUM", lay.name))

        if grid is None:  # point source
            ra, dec = offsets_to_radec(lay.x, lay.y, model.ra, model.dec)
            row.update(RA=float(ra), DEC=float(dec), IMAGE=_NULL, X_OFF=lay.x, Y_OFF=lay.y)
        else:
            if isinstance(grid, RegularGrid):
                # SIMPUT places the image's reference pixel (the grid centre) at RA/DEC
                ra_c, dec_c = offsets_to_radec(grid.x0, grid.y0, model.ra, model.dec)
                row.update(RA=float(ra_c), DEC=float(dec_c), X_OFF=grid.x0, Y_OFF=grid.y0)
                ext = f"IMG_{lay.name}"
                hdus.append(_image_hdu(ext, grid, weights, model))
                row["IMAGE"] = _ref(ext)
            else:  # SIMPUT 1.1 has no adaptive images: place the flux at its centroid
                xc, yc = grid.centres()
                w = np.asarray(weights) / np.sum(weights)
                cx, cy = float(np.sum(w * xc)), float(np.sum(w * yc))
                ra, dec = offsets_to_radec(cx, cy, model.ra, model.dec)
                row.update(RA=float(ra), DEC=float(dec), IMAGE=_NULL, X_OFF=cx, Y_OFF=cy)

        if isinstance(lay, SeparableLayer) and isinstance(grid, QuadtreeGrid):
            ext = f"QIMG_{lay.name}"
            h = fits.BinTableHDU.from_columns(
                [fits.Column("NUNIQ", "K", array=grid.uniq),
                 fits.Column("WEIGHT", "D", array=lay.image)], name=ext)
            h.header["HDUCLAS1"] = "QTIMAGE"
            for k, v in grid.header().items():
                h.header[k] = v
            hdus.append(h)
            row["LAYER"] = ext

        if isinstance(lay, SpecMapLayer):
            hdus.append(_specmap_hdu(lay, model))
            row["LAYER"] = f"SMAP_{lay.name}"
            for k, t in enumerate(lay.templates):
                spectra.append(Spectrum(t.edges, t.values, f"{lay.name}:T{k}"))

        if isinstance(lay, SpecCubeLayer):
            hdus.extend(_speccube_hdus(lay, model))
            row["LAYER"] = f"SCUBE_{lay.name}"

        rows.append(row)

    # SRC_CAT ---------------------------------------------------------------
    def col(name, fmt, unit=None):
        return fits.Column(name, fmt, unit=unit, array=[r.get(name, 0) for r in rows])

    cat = fits.BinTableHDU.from_columns([
        col("SRC_ID", "J"), col("SRC_NAME", "48A"),
        col("RA", "D", "deg"), col("DEC", "D", "deg"),
        fits.Column("IMGROTA", "E", unit="deg", array=np.zeros(len(rows))),
        fits.Column("IMGSCAL", "E", array=np.ones(len(rows))),
        col("E_MIN", "E", "keV"), col("E_MAX", "E", "keV"), col("FLUX", "E", "erg/s/cm**2"),
        col("SPECTRUM", "80A"), col("IMAGE", "80A"), col("TIMING", "80A"),
        col("LAYTYPE", "16A"), col("LAYER", "64A"),
        col("X_OFF", "D", "uas"), col("Y_OFF", "D", "uas"),
    ], name="SRC_CAT")
    _simput_header(cat.header, "SRC_CAT")
    cat.header["RADESYS"] = "FK5"
    cat.header["EQUINOX"] = 2000.0

    out = [prim, cat]
    if spectra:
        e_arr, _ = _vla([0.5 * (s.edges[1:] + s.edges[:-1]) for s in spectra])
        f_arr, _ = _vla([s.values for s in spectra])
        ed_arr, _ = _vla([s.edges for s in spectra])
        sh = fits.BinTableHDU.from_columns([
            fits.Column("NAME", "48A", array=[s.name for s in spectra]),
            fits.Column("ENERGY", "PD()", unit="keV", array=e_arr),
            fits.Column("FLUXDENSITY", "PD()", unit="photon/s/cm**2/keV", array=f_arr),
            fits.Column("E_EDGES", "PD()", unit="keV", array=ed_arr),
        ], name="SPECTRUM")
        _simput_header(sh.header, "SPECTRUM")
        out.append(sh)
    out.extend(hdus)
    if model.spec is not None:
        blob = np.frombuffer(json.dumps(model.spec).encode(), dtype=np.uint8)
        sp = fits.ImageHDU(blob, name="MODELSPEC")
        sp.header["CONTENT"] = "ModelSpec JSON, UTF-8 bytes"
        out.append(sp)
    fits.HDUList(out).writeto(path, overwrite=overwrite)


def _specmap_hdu(lay: SpecMapLayer, model: Model):
    ext = f"SMAP_{lay.name}"
    if isinstance(lay.grid, RegularGrid):
        data = np.stack([lay.grid.to_image(lay.maps[p]) for p in SPECMAP_PLANES]).astype(np.float32)
        h = fits.ImageHDU(data, name=ext)
        for k, v in lay.grid.wcs_header(model.ra, model.dec).items():
            h.header[k] = v
        h.header["WCSAXES"] = 3
        h.header["CTYPE3"] = "PLANE"
        h.header["CRPIX3"], h.header["CRVAL3"], h.header["CDELT3"] = 1.0, 1.0, 1.0
    else:
        cols = [fits.Column("NUNIQ", "K", array=lay.grid.uniq)]
        cols += [fits.Column(p, "D" if p == "DELAY" else "E", array=lay.maps[p]) for p in SPECMAP_PLANES]
        h = fits.BinTableHDU.from_columns(cols, name=ext)
        for k, v in lay.grid.header().items():
            h.header[k] = v
    h.header["HDUCLAS1"] = "SPECMAP"
    for i, p in enumerate(SPECMAP_PLANES, start=1):
        h.header[f"PLANE{i}"] = p
    h.header["NORMUNIT"] = "photon/s/cm**2 per pixel"
    h.header["DELAYUNI"] = "s"
    h.header["NHUNIT"] = "cm**-2"
    h.header["ABSMODEL"] = lay.abs_model
    h.header["NTEMPL"] = len(lay.templates)
    for k in range(len(lay.templates)):
        h.header[f"TEMPL{k}"] = f"{lay.name}:T{k}"
    return h


def _speccube_hdus(lay: SpecCubeLayer, model: Model):
    ext = f"SCUBE_{lay.name}"
    eb = fits.BinTableHDU.from_columns([
        fits.Column("E_MIN", "D", unit="keV", array=lay.energy_edges[:-1]),
        fits.Column("E_MAX", "D", unit="keV", array=lay.energy_edges[1:])], name=f"SCEB_{lay.name}")
    if isinstance(lay.grid, RegularGrid):
        cube = lay.cube
        vmax = cube.max() if cube.size and cube.max() > 0 else 1.0
        logzero = np.log10(vmax) - 10.0  # 10 decades of dynamic range
        step = 10.0 / 65534
        with np.errstate(divide="ignore"):
            q = np.round((np.log10(cube) - logzero) / step) - 32767
        q = np.where(cube <= 10**logzero, -32768, np.clip(q, -32767, 32767)).astype(np.int16)
        data = q.reshape(cube.shape[0], *lay.grid.shape)
        h = fits.CompImageHDU(data, name=ext, compression_type="GZIP_2",
                              tile_shape=(cube.shape[0], 32, 32))
        for k, v in lay.grid.wcs_header(model.ra, model.dec).items():
            h.header[k] = v
        h.header["QUANT"] = ("LOG10", "value = 10**(LOGZERO + (q + 32767) * LOGSTEP)")
        h.header["LOGZERO"] = logzero
        h.header["LOGSTEP"] = step
        h.header["QZERO"] = (-32768, "stored value meaning exactly zero")
    else:
        h = fits.BinTableHDU.from_columns([
            fits.Column("NUNIQ", "K", array=lay.grid.uniq),
            fits.Column("FLUX", f"{lay.cube.shape[0]}E", array=lay.cube.T)], name=ext)
        for k, v in lay.grid.header().items():
            h.header[k] = v
        h.header["QUANT"] = "NONE"
    h.header["HDUCLAS1"] = "SPECCUBE"
    h.header["BUNIT"] = "photon/s/cm**2/keV per pixel"
    h.header["EBOUNDS"] = f"SCEB_{lay.name}"
    return [h, eb]


# =========================================================================== reading


def _read_spectra(hdul, ext="SPECTRUM"):
    """All spectra in one extension: (dict by NAME, list in row order)."""
    if ext not in hdul:
        return {}, []
    by_name, rows = {}, []
    d = hdul[ext].data
    has_names = "NAME" in d.columns.names
    for i in range(len(d)):
        nm = str(d["NAME"][i]).strip() if has_names else ""
        e = np.atleast_1d(np.asarray(d["ENERGY"][i], float))
        f = np.atleast_1d(np.asarray(d["FLUXDENSITY"][i], float))
        if "E_EDGES" in d.columns.names and len(d["E_EDGES"][i]) == e.size + 1:
            edges = np.asarray(d["E_EDGES"][i], float)
        elif e.size == 1:  # a single energy: a narrow bin around it
            edges = np.array([e[0] * 0.999, e[0] * 1.001])
            f = f / (edges[1] - edges[0])
        else:  # plain SIMPUT: energies are sample points; build bins around them
            mid = 0.5 * (e[1:] + e[:-1])
            edges = np.concatenate([[e[0] - (mid[0] - e[0])], mid, [e[-1] + (e[-1] - mid[-1])]])
            edges[0] = max(edges[0], 1e-6)
        spec = Spectrum(edges, f, nm)
        rows.append(spec)
        if nm:
            by_name[nm] = spec
    return by_name, rows


def _read_driver(h):
    hdr, d = h.header, h.data
    name = hdr.get("DRVNAME", h.name)
    if hdr.get("HDUCLAS1") == "POWSPEC":
        return PSDDriver(d["FREQUENCY"], d["POWER"], name=name, seed=hdr.get("SEED", 0),
                         transform=hdr.get("XFORM", "exponential"), dt=hdr.get("DT"),
                         span=(hdr["SPANSTRT"], hdr["SPANSTOP"]) if "SPANSTRT" in hdr else None)
    if hdr.get("PERIODIC", 0):
        dp = tuple(hdr[k] for k in ("DPERIOD", "D2PERIOD", "D3PERIOD") if k in hdr)
        return LightCurve(d["PHASE"], d["FLUX"], name=name, periodic=True, period=hdr["PERIOD"],
                          phase0=hdr.get("PHASE0", 0.0), dperiod=dp)
    return LightCurve(d["TIME"], d["FLUX"], name=name)


def _grid_from_image(h, x_row, y_row, imgscal=1.0, imgrota=0.0):
    """Grid of an IMAGE extension; returns (grid, flip_columns).

    xrimodel images carry their grid in GRIDSCAL/GRIDX0/GRIDY0. For plain SIMPUT images,
    the reference pixel (CRPIX) sits at the catalogue row's position (x_row, y_row) and
    the pixel scale is CDELT / IMGSCAL, as SIMPUT prescribes; CRVAL is not used.
    """
    hdr = h.header
    shape = h.data.shape[-2:]
    if "GRIDSCAL" in hdr:
        return RegularGrid.from_header(hdr, shape), False
    if imgrota or hdr.get("CROTA2", 0.0) or any(k in hdr for k in ("PC1_2", "PC2_1", "CD1_2")):
        raise NotImplementedError("rotated SIMPUT images are not supported yet")
    ny, nx = shape
    cd1 = hdr["CD1_1"] if "CD1_1" in hdr else hdr["CDELT1"]
    cd2 = hdr["CD2_2"] if "CD2_2" in hdr else hdr["CDELT2"]
    scale = abs(cd2) * UAS_PER_DEG / (imgscal or 1.0)
    crpix1, crpix2 = hdr.get("CRPIX1", (nx + 1) / 2), hdr.get("CRPIX2", (ny + 1) / 2)
    flip = cd1 > 0
    if flip:  # columns will be reversed so that they run East -> West
        crpix1 = nx + 1 - crpix1
    x0 = x_row - ((nx + 1) / 2 - crpix1) * scale
    y0 = y_row + ((ny + 1) / 2 - crpix2) * scale
    return RegularGrid(nx=nx, ny=ny, scale=float(scale), x0=float(x0), y0=float(y0)), flip


def read_model(path) -> Model:
    """Read an xrimodel file, or a plain SIMPUT 1.1 file (as separable/photon layers)."""
    with fits.open(path) as hdul:
        ph = hdul[0].header
        cat = hdul["SRC_CAT"]
        ra0 = ph.get("REFRA", float(cat.data["RA"][0]))
        dec0 = ph.get("REFDEC", float(cat.data["DEC"][0]))
        model = Model(ra=ra0, dec=dec0, mjdref=ph.get("MJDREF", 0.0))
        spec_cache = {}

        def spectra_in(ext="SPECTRUM"):
            if ext not in spec_cache:
                spec_cache[ext] = _read_spectra(hdul, ext)
            return spec_cache[ext]

        spectra = spectra_in()[0]

        ext_drivers = {}
        for h in hdul[1:]:
            if h.header.get("HDUCLAS1") in ("LIGHTCURVE", "POWSPEC"):
                drv = _read_driver(h)
                model.add_driver(drv)
                ext_drivers[h.name] = drv.name

        cols = cat.data.columns.names
        for r in cat.data:
            name = str(r["SRC_NAME"]).strip() if "SRC_NAME" in cols else f"src{r['SRC_ID']}"
            name = name or f"src{r['SRC_ID']}"
            kind = str(r["LAYTYPE"]).strip() if "LAYTYPE" in cols else ""
            layext = str(r["LAYER"]).strip() if "LAYER" in cols else ""
            t_ext, _ = _parse_ref(r["TIMING"] if "TIMING" in cols else "")
            driver = ext_drivers.get(t_ext.upper()) if t_ext else None
            s_ext, s_name = _parse_ref(r["SPECTRUM"])

            if kind == "photons" or (s_ext and s_ext.upper() != "SPECTRUM"
                                     and hdul[s_ext].header.get("HDUCLAS1") == "PHOTONS"):
                h = hdul[layext or s_ext]
                d = h.data
                if "X_OFF" in d.columns.names:
                    x, y = d["X_OFF"], d["Y_OFF"]
                else:
                    x, y = radec_to_offsets(d["RA"], d["DEC"], ra0, dec0)
                t = d["TIME"] if "TIME" in d.columns.names else np.zeros(len(d))
                model.add(PhotonLayer(name, t, x, y, d["ENERGY"], area=h.header.get("AREA", 1.0),
                                      tstart=h.header.get("TSTART", 0.0),
                                      tstop=h.header.get("TSTOP", 0.0)))
                continue

            if kind == "specmap":
                model.add(_read_specmap(hdul[layext], name, spectra, driver))
                continue
            if kind == "speccube":
                model.add(_read_speccube(hdul, layext, name, driver))
                continue

            # separable (also plain SIMPUT rows)
            by_name, by_row = spectra_in(s_ext or "SPECTRUM")
            if isinstance(s_name, str):
                spec = by_name[s_name]
            else:
                spec = by_row[(s_name or 1) - 1]
            if kind != "separable":  # plain SIMPUT: shape only, normalised by FLUX
                ef = spec.energy_flux(r["E_MIN"], r["E_MAX"])
                spec = spec.scaled(r["FLUX"] / ef) if ef > 0 else spec
            i_ext, _ = _parse_ref(r["IMAGE"] if "IMAGE" in cols else "")
            if layext.startswith("QIMG_"):
                h = hdul[layext]
                grid = QuadtreeGrid.from_header(h.header, h.data["NUNIQ"])
                model.add(SeparableLayer(name, spec, grid, h.data["WEIGHT"], driver=driver))
            elif i_ext:
                h = hdul[i_ext]
                if "X_OFF" in cols:
                    xr, yr = float(r["X_OFF"]), float(r["Y_OFF"])
                else:
                    xr, yr = radec_to_offsets(r["RA"], r["DEC"], ra0, dec0)
                grid, flip = _grid_from_image(
                    h, float(xr), float(yr),
                    imgscal=float(r["IMGSCAL"]) if "IMGSCAL" in cols else 1.0,
                    imgrota=float(r["IMGROTA"]) if "IMGROTA" in cols else 0.0)
                img = np.asarray(h.data, float)
                if flip:
                    img = img[:, ::-1]
                model.add(SeparableLayer(name, spec, grid, img, driver=driver))
            else:
                if "X_OFF" in cols:
                    x, y = float(r["X_OFF"]), float(r["Y_OFF"])
                else:
                    x, y = radec_to_offsets(r["RA"], r["DEC"], ra0, dec0)
                model.add(SeparableLayer(name, spec, x=float(x), y=float(y), driver=driver))

        if "MODELSPEC" in hdul:
            model.spec = json.loads(bytes(hdul["MODELSPEC"].data).decode())
    return model


def _read_specmap(h, name, spectra, driver):
    hdr = h.header
    templates = [spectra[hdr[f"TEMPL{k}"]] for k in range(hdr["NTEMPL"])]
    planes = [hdr[f"PLANE{i}"] for i in range(1, len(SPECMAP_PLANES) + 1)]
    if hdr.get("GRIDTYPE") == "QUADTREE":
        grid = QuadtreeGrid.from_header(hdr, h.data["NUNIQ"])
        maps = {p: np.asarray(h.data[p], float) for p in planes}
    else:
        data = np.asarray(h.data, float)
        grid = RegularGrid.from_header(hdr, data.shape[1:])
        maps = {p: data[i].ravel() for i, p in enumerate(planes)}
    return SpecMapLayer(name, grid, templates, maps, driver=driver,
                        abs_model=hdr.get("ABSMODEL", "powerlaw-approx"))


def _read_speccube(hdul, ext, name, driver):
    h = hdul[ext]
    hdr = h.header
    eb = hdul[hdr["EBOUNDS"]].data
    edges = np.concatenate([eb["E_MIN"], eb["E_MAX"][-1:]])
    if hdr.get("GRIDTYPE") == "QUADTREE":
        grid = QuadtreeGrid.from_header(hdr, h.data["NUNIQ"])
        cube = np.asarray(h.data["FLUX"], float).T
    else:
        q = np.asarray(h.data, np.int32)
        grid = RegularGrid.from_header(hdr, q.shape[1:])
        cube = 10.0 ** (hdr["LOGZERO"] + (q + 32767) * hdr["LOGSTEP"])
        cube[q == hdr.get("QZERO", -32768)] = 0.0
        cube = cube.reshape(q.shape[0], -1)
    return SpecCubeLayer(name, grid, edges, cube, driver=driver)
