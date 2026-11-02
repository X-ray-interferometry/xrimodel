"""Draw photons (time, position, energy) from a :class:`~.layers.Model`.

Each layer is an inhomogeneous Poisson process. Pixel p of a layer has rate
``R_p * D(t - tau_p)``: its mean photon flux R_p times the collecting area, modulated
by the layer's driver D, delayed by the pixel's light-travel delay tau_p (0 unless a
SpecMap layer has a DELAY map). Arrival times come from Lewis-Shedler thinning:
candidates at the constant rate R_p * max(D) are kept with probability
D(t - tau_p) / max(D). This is exact and needs no time x space cube.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..units import UAS_PER_RAD
from .drivers import Constant
from .grid import offsets_to_radec
from .layers import Model, PhotonLayer, SeparableLayer, SpecCubeLayer, SpecMapLayer
from .spectra import cross_section


@dataclass
class Photons:
    """Sampled photons, sorted by arrival time. Offsets in uas, energies in keV."""

    time: np.ndarray
    x: np.ndarray
    y: np.ndarray
    energy: np.ndarray
    layer: np.ndarray  # index into ``layer_names``
    layer_names: list[str]
    ra0: float
    dec0: float

    def __len__(self):
        return self.time.size

    def radec(self):
        """RA, Dec in degrees (see grid.offsets_to_radec for the precision caveat)."""
        return offsets_to_radec(self.x, self.y, self.ra0, self.dec0)

    def select(self, mask) -> Photons:
        return Photons(self.time[mask], self.x[mask], self.y[mask], self.energy[mask],
                       self.layer[mask], self.layer_names, self.ra0, self.dec0)

    def to_xrisalis_image(self, t0: float = 0.0):
        """Build an ``xrisalis.images.image`` (energies in J, loc in rad, integer toa).

        ``toa`` is ``floor(time - t0)``, i.e. an index into XRISALIS's 1-s pointing array,
        until XRISALIS accepts real times. ``loc`` follows the generator convention
        ``[x_east, y_north]`` in radians.
        """
        from xrisalis import images  # optional dependency

        im = images.image(len(self))
        im.energies = self.energy * 1.602176634e-16  # keV -> J
        im.loc = np.column_stack([self.x, self.y]) / UAS_PER_RAD
        im.toa = np.floor(self.time - t0).astype(int)
        return im


def _uniform_in_pixel(xc, yc, size, rng):
    return xc + (rng.random(xc.size) - 0.5) * size, yc + (rng.random(xc.size) - 0.5) * size


def _thin_times(n, t0, t1, driver, delay, dmax, rng):
    """Uniform candidate times on [t0, t1], kept with probability D(t - delay)/dmax."""
    t = t0 + (t1 - t0) * rng.random(n)
    if isinstance(driver, Constant):
        return t, np.ones(n, bool)
    keep = rng.random(n) * dmax < driver(t - delay)
    return t, keep


def sample_photons(model: Model, t0: float, t1: float, area: float, rng=None,
                   layers=None) -> Photons:
    """Sample photons arriving in [t0, t1] (s) on a collecting area (cm^2).

    ``rng`` is a numpy Generator or a seed. ``layers`` optionally restricts sampling to
    the named layers. Detector and mirror responses are applied by the simulator.
    """
    rng = np.random.default_rng(rng)
    model.validate()
    out = {k: [] for k in ("time", "x", "y", "energy", "layer")}
    names = [lay.name for lay in model.layers]
    active = [lay for lay in model.layers if layers is None or lay.name in layers]
    prepared = _prepare_drivers(model, active, t0, t1)

    for li, lay in enumerate(model.layers):
        if lay not in active:
            continue
        if isinstance(lay, PhotonLayer):
            res = _sample_photon_layer(lay, t0, t1, area, rng)
        else:
            res = _sample_rate_layer(lay, prepared, t0, t1, area, rng)
        if res is None:
            continue
        for k, v in zip(("time", "x", "y", "energy"), res):
            out[k].append(v)
        out["layer"].append(np.full(res[0].size, li, dtype=np.int32))

    cat = {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in out.items()}
    order = np.argsort(cat["time"], kind="stable")
    return Photons(cat["time"][order], cat["x"][order], cat["y"][order], cat["energy"][order],
                   cat["layer"][order].astype(np.int32), names, model.ra, model.dec)


def _sample_photon_layer(lay: PhotonLayer, t0, t1, area, rng):
    if area > lay.area * (1 + 1e-12):
        raise ValueError(f"layer {lay.name!r} holds photons for {lay.area} cm^2; "
                         f"cannot sample a larger area ({area} cm^2)")
    sel = (lay.time >= t0) & (lay.time < t1) & (rng.random(lay.time.size) < area / lay.area)
    return lay.time[sel], lay.x[sel], lay.y[sel], lay.energy[sel]


def _emission_window(lay, t0, t1):
    """Interval of emission times a layer needs for arrivals in [t0, t1]."""
    delays = lay.maps["DELAY"] if isinstance(lay, SpecMapLayer) else np.zeros(1)
    return t0 - max(float(delays.max()), 0.0), t1 - min(float(delays.min()), 0.0)


def _prepare_drivers(model, active, t0, t1):
    """Realise each driver ONCE over the union of its layers' windows.

    Layers sharing a driver must see the same realisation, or delayed (reprocessed)
    emission would not be coherent with its driver.
    """
    windows = {}
    for lay in active:
        name = getattr(lay, "driver", None)
        if name is None or isinstance(lay, PhotonLayer):
            continue
        a, b = _emission_window(lay, t0, t1)
        lo, hi = windows.get(name, (a, b))
        windows[name] = (min(lo, a), max(hi, b))
    return {name: model.drivers[name].prepare(lo - 1e-9, hi + 1e-9)
            for name, (lo, hi) in windows.items()}


def _sample_rate_layer(lay, prepared, t0, t1, area, rng):
    drv_name = getattr(lay, "driver", None)
    driver = prepared[drv_name] if drv_name else Constant()
    dmax = driver.max_on(*_emission_window(lay, t0, t1))
    if dmax <= 0:
        return None

    rate = lay.mean_photon_flux() * area
    n = rng.poisson(rate * (t1 - t0) * dmax)
    if n == 0:
        return None

    if isinstance(lay, SeparableLayer):
        t, keep = _thin_times(n, t0, t1, driver, 0.0, dmax, rng)
        t = t[keep]
        m = t.size
        if lay.grid is None:
            x, y = np.full(m, lay.x), np.full(m, lay.y)
        else:
            p = rng.choice(lay.grid.n, size=m, p=lay.image / lay.image.sum())
            xc, yc = lay.grid.centres()
            x, y = _uniform_in_pixel(xc[p], yc[p], lay.grid.sizes()[p], rng)
        return t, x, y, lay.spectrum.sample(m, rng)

    if isinstance(lay, SpecCubeLayer):
        bf = lay.bin_flux().ravel()
        flat = rng.choice(bf.size, size=n, p=bf / bf.sum())
        ebin, p = np.divmod(flat, lay.grid.n)
        t, keep = _thin_times(n, t0, t1, driver, 0.0, dmax, rng)
        ebin, p, t = ebin[keep], p[keep], t[keep]
        xc, yc = lay.grid.centres()
        x, y = _uniform_in_pixel(xc[p], yc[p], lay.grid.sizes()[p], rng)
        de = np.diff(lay.energy_edges)
        e = lay.energy_edges[ebin] + rng.random(ebin.size) * de[ebin]
        return t, x, y, e

    if isinstance(lay, SpecMapLayer):
        m_ = lay.maps
        norm = m_["NORM"]
        p = rng.choice(lay.grid.n, size=n, p=norm / norm.sum())
        t, keep = _thin_times(n, t0, t1, driver, m_["DELAY"][p], dmax, rng)
        p, t = p[keep], t[keep]
        # rest-frame energies per template
        e_rest = np.empty(p.size)
        tidx = m_["TIDX"][p].astype(int)
        for k, tmpl in enumerate(lay.templates):
            sel = tidx == k
            if sel.any():
                e_rest[sel] = tmpl.sample(int(sel.sum()), rng)
        e = m_["G"][p] * e_rest
        sig = m_["SIGMA"][p]
        if np.any(sig > 0):
            e = e * (1.0 + sig * rng.standard_normal(e.size))
            e = np.abs(e)
        nh = m_["NH"][p]
        survive = rng.random(e.size) < np.exp(-nh * cross_section(lay.abs_model)(e))
        p, t, e = p[survive], t[survive], e[survive]
        xc, yc = lay.grid.centres()
        x, y = _uniform_in_pixel(xc[p], yc[p], lay.grid.sizes()[p], rng)
        return t, x, y, e

    raise TypeError(f"cannot sample layer of type {type(lay).__name__}")


def transfer_function(layer: SpecMapLayer, energy_edges, delay_edges, chunk=4096):
    """Energy-delay transfer function psi(E, tau) of a SpecMap layer.

    Returns an array (n_energy_bins, n_delay_bins) of photons/s/cm^2: the mean photon
    flux arriving in each energy bin with each delay. Convolving the driver light curve
    with psi along tau gives the layer's light curve in each energy band; this is the
    impulse response that spectral-timing codes such as nDspec turn into lags and cross
    spectra. Broadening (SIGMA) is ignored, as in ``specmap_projection``.
    """
    energy_edges = np.asarray(energy_edges, float)
    delay_edges = np.asarray(delay_edges, float)
    m = layer.maps
    sigma = cross_section(layer.abs_model)(0.5 * (energy_edges[1:] + energy_edges[:-1]))
    j = np.digitize(m["DELAY"], delay_edges) - 1
    inside = (j >= 0) & (j < delay_edges.size - 1)
    psi = np.zeros((energy_edges.size - 1, delay_edges.size - 1))
    tidx = m["TIDX"].astype(int)
    for k, tmpl in enumerate(layer.templates):
        sel = np.flatnonzero((tidx == k) & inside)
        for s in range(0, sel.size, chunk):
            idx = sel[s:s + chunk]
            cdf = tmpl.cdf_at(energy_edges[None, :] / m["G"][idx, None])
            binf = np.diff(cdf, axis=1) * m["NORM"][idx, None]
            binf *= np.exp(-m["NH"][idx, None] * sigma[None, :])
            np.add.at(psi.T, j[idx], binf)
    return psi
