"""A general 3-D scene: lamps, opaque bodies and optically thin volumes, ray-marched along
the line of sight into SpecMap layers.

The system frame is Cartesian, in the system's ``length_unit``. The observer looks from
n = (0, -sin i, cos i) (``inclination`` i from the frame's z axis); the frame's x axis
lies in the sky plane at position angle ``node_pa`` (East of North) and the frame
origin is at sky offset (``x``, ``y``) uas. (e_x, e_b, n) is right-handed with
e_b = (0, cos i, sin i), so with the z axis towards the observer (i < 90 deg) a
rotation about +z runs counterclockwise on the sky.

What the renderer does, per sky ray (sub-pixel samples of the scene's grid):

* finds the front-most body surface; everything behind it is hidden;
* samples the volumes in front of it (coarse samples along the whole ray plus fine
  samples per volume, clustered towards its dense part), giving the absorbing column in
  front of every emitter;
* electron-scatters (and line-reprocesses) the light of every illuminator off every
  volume, with the illuminator's photon flux, light-travel delay, Doppler shift and the
  column and shadows between illuminator and gas;
* reflects every illuminator off the visible surface (angular law mu_i / (mu_i + mu_e),
  pexmon or approximate templates by emission angle), shadowed by any body on the
  segment to the illuminator (sampled at 48 points);
* adds the bodies' and volumes' own emission.

Columns and shadows *from* each illuminator come from a precomputed spherical grid around
it (log-spaced in radius): the cumulative absorbing column and whether a body has been
crossed. With ``absorb_illumination`` the columns between illuminator and emitter are
added to the emitter's NH (absorbing gas) and NE (all electrons, Klein-Nishina scattering
out of the beam): exact for grey electron scattering, an approximation for reflection,
whose spectrum then ignores the change of the incident shape. Every emitter's NE also
counts the electrons in front of it along the line of sight.

Delays are relative to light emitted at the frame origin and travelling straight to the
observer. Spectra are rest-frame shapes; with D(b, k) = 1 / (gamma (1 - b.k)) the
Doppler factor of a source moving at b seen from direction k, a lamp moving at b_l
appears with G = D(b_l, n) and weight G^3 (band photon flux), and light reprocessed by
gas or a surface moving at b has G = D(b_l, k) (1 - b.k) / (1 - b.n) (k from lamp to
emitter). Weights: steady volume emission D(b, n)^2; scattering D(b_l, k)^3 (1 - b.k)
D(b, n)^2 (the lamp's beaming, the gas's collision rate, aberration); surfaces
D(b_l, k)^3 gamma (1 - b.k) D(b, n)^3; own surface emission D(b, n)^3; moving knots
D(b, n)^3. Ionisation parameters use the illuminators' unabsorbed luminosities.

Layers: ``<system>_<lamp>`` (point sources), ``<system>_<body>`` (reflection),
``<system>_<volume><k>`` (scattering, split by the column in front of the emission: bin
k of ``nh_bins``), ``<system>_<body>_emit`` and ``<system>_<volume>_emit<k>`` (own
emission). With several illuminator groups (lamps, or extended illuminators), reflection
and scattering layers get the group's name: ``<system>_<body>_<group>``.
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass, field
from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.ndimage import map_coordinates

from ..format.grid import RegularGrid
from ..format.layers import SpecMapLayer
from ..format.spectra import Spectrum, gaussian_line, sigma_photoabs
from ..units import C_CM_S, PC_CM, UAS_PER_RAD
from .primitives import PRIMITIVES, unit_cm
from .raytrace import (
    REFLECTION_C,
    doppler,
    first_hit,
    segment_blocked,
    slab_samples,
    sphere_samples,
    surface_normal,
)
from .spec import build_spectrum

SIGMA_T = 6.6524587e-25
NE_PER_H = 1.2


# --------------------------------------------------------------------------- templates


def reflection_templates(edges, spec: dict, incident: Spectrum, mu_edges):
    """pexmon by emission angle for a power-law illuminator (if XSPEC models are
    installed); otherwise one approximate template: the incident spectrum times a
    photoabsorption-limited albedo, plus Fe K-alpha."""
    if spec.get("kind") == "powerlaw":
        try:
            from ..adapters.xspec import pexmon_templates

            return pexmon_templates(edges, gamma=float(spec.get("gamma", 2.0)),
                                    mu_edges=mu_edges)
        except ImportError:
            pass
    e = incident.centres
    sig_t = NE_PER_H * SIGMA_T
    cont = incident.values * sig_t / (sig_t + sigma_photoabs(e))
    line = gaussian_line(edges, 6.4, 0.02, flux=0.9 * np.interp(6.4, e, cont))
    return [Spectrum(edges, cont + line.values, "refl_approx")], np.array([0.0, 1.0])


def scatter_template(incident: Spectrum, scatter: float, lines):
    """Scattered continuum plus lines; returns (template, photons per Thomson-scattered
    incident photon)."""
    v = scatter * incident.values
    nflux = incident.photon_flux()
    for ln in lines:
        v = v + gaussian_line(incident.edges, float(ln["e0"]), float(ln.get("sigma", 0.01)),
                              flux=float(ln["yield"]) * nflux).values
    t = Spectrum(incident.edges, v, "scattered")
    return t, t.photon_flux() / nflux


# --------------------------------------------------------------------------- component


@dataclass(kw_only=True)
class System3D:
    name: str
    length_unit: str | float = "cm"
    mass_msun: float = 1.0  # for length_unit "rg"
    inclination: float = 0.0  # deg
    node_pa: float = 0.0  # deg East of North, of the frame's x axis
    x: float = 0.0  # uas, sky offset of the frame origin
    y: float = 0.0
    nh_ism: float = 0.0
    lamps: list = field(default_factory=list)
    bodies: list = field(default_factory=list)
    volumes: list = field(default_factory=list)
    extent: float | None = None  # half-length of the rays (length units); default: bounds
    n_coarse: int = 128
    nh_bins: tuple = (0.0, 1e21, 1e22, 1e23)  # cm^-2, lower edges of the column bins
    mu_edges: tuple = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    absorb_illumination: bool = True
    column_grid: tuple = (48, 96, 128)  # (theta, phi, log r) cells around each illuminator
    supersample: int = 2
    chunk: int = 4096

    kind: ClassVar[str] = "system3d"

    # ------------------------------------------------------------------ spec

    def to_dict(self):
        def prim(p):
            if not getattr(p, "serialisable", True):
                warnings.warn(f"{p.name}: {type(p).__name__} holds Python functions; the "
                              "ModelSpec records only its name, so this scene cannot be "
                              "rebuilt from the file", stacklevel=3)
                return {"kind": p.kind, "name": p.name, "serialisable": False}
            return {"kind": p.kind, **asdict(p)}

        d = {k: v for k, v in asdict(self).items() if k not in ("lamps", "bodies", "volumes")}
        return {"kind": self.kind, **d, "lamps": [prim(p) for p in self.lamps],
                "bodies": [prim(p) for p in self.bodies],
                "volumes": [prim(p) for p in self.volumes]}

    @classmethod
    def from_dict(cls, d):
        d = {k: v for k, v in d.items() if k != "kind"}
        for key in ("lamps", "bodies", "volumes"):
            for p in d.get(key, []):
                if p.get("serialisable") is False:
                    raise ValueError(f"{p['name']}: a {p['kind']} (custom functions) cannot be "
                                     "rebuilt from the ModelSpec")
            d[key] = [PRIMITIVES[p["kind"]](**{k: v for k, v in p.items() if k != "kind"})
                      for p in d.get(key, [])]
        for key in ("nh_bins", "mu_edges", "column_grid"):
            if key in d:
                d[key] = tuple(d[key])
        return cls(**d)

    # ------------------------------------------------------------------ geometry

    def frame(self):
        inc = np.deg2rad(self.inclination)
        return dict(n=np.array([0.0, -np.sin(inc), np.cos(inc)]),
                    e_a=np.array([1.0, 0.0, 0.0]),
                    e_b=np.array([0.0, np.cos(inc), np.sin(inc)]))

    def sky_axes(self):
        """Unit sky vectors (East, North) of the frame's e_a (x axis) and e_b."""
        pa = np.deg2rad(self.node_pa)
        return np.array([np.sin(pa), np.cos(pa)]), np.array([np.cos(pa), -np.sin(pa)])

    def _extent_cm(self, L, lamps, bodies, volumes):
        if self.extent is not None:
            return float(self.extent) * L
        r = [np.linalg.norm(lp.pos) for lp in lamps]
        for o in list(bodies) + list(volumes):
            r.append(float(np.linalg.norm(np.asarray(o.bound["centre"])) + o.bound["radius"]))
        return 1.05 * max(r) if r else L

    # ------------------------------------------------------------------ rendering

    def render(self, scene):
        L = unit_cm(self.length_unit, self.mass_msun)
        names = [p.name for p in list(self.lamps) + list(self.bodies) + list(self.volumes)]
        if len(set(names)) != len(names):
            raise ValueError(f"{self.name}: lamp, body and volume names must be unique: {names}")
        lamps = [p.build(L, self.mass_msun) for p in self.lamps]
        bodies = [p.build(L, self.mass_msun) for p in self.bodies]
        vols = [p.build(L, self.mass_msun) for p in self.volumes]
        f = self.frame()
        n, e_a, e_b = (jnp.asarray(f[k]) for k in ("n", "e_a", "e_b"))
        u_a, u_b = self.sky_axes()
        d_cm = scene.distance_pc * PC_CM
        cm_per_uas = d_cm / UAS_PER_RAD
        edges = scene.energy_edges
        S = self._extent_cm(L, lamps, bodies, vols)

        inside_fns = tuple(b.inside for b in bodies)

        def any_inside(p):
            out = jnp.zeros(p.shape[:-1], dtype=bool)
            for fn in inside_fns:
                out = out | fn(p)
            return out

        # ---- illuminator groups: lamps, and emitting volumes flagged to illuminate
        rng = np.random.default_rng(0)
        groups = []
        for lp in lamps:
            if not lp.illuminates:
                continue
            sp = build_spectrum(lp.spectrum, edges)
            nph = 4 * np.pi * d_cm**2 * sp.photon_flux()
            groups.append(dict(name=lp.name, spec=lp.spectrum, spectrum=sp, driver=lp.driver,
                               pos=lp.pos[None, :], vel=lp.vel[None, :], nph=np.array([nph]),
                               lx=np.array([4 * np.pi * d_cm**2 * sp.energy_flux()])))
        for v in vols:
            em = v.emission
            if not (em and em["illuminates"]):
                continue
            pts, nph = _discretise(v, em, any_inside, rng=np.random.default_rng(em["seed"]))
            sp = build_spectrum(em["spectrum"], edges)
            e_mean = sp.energy_flux() / sp.photon_flux()
            groups.append(dict(name=v.name, spec=em["spectrum"], spectrum=sp,
                               driver=em["driver"], pos=pts,
                               vel=np.asarray(v.velocity(jnp.asarray(pts))), nph=nph,
                               lx=nph * e_mean))
        del rng
        all_pos = jnp.asarray(np.concatenate([g["pos"] for g in groups])) if groups \
            else jnp.zeros((0, 3))
        all_lx = jnp.asarray(np.concatenate([g["lx"] for g in groups])) if groups \
            else jnp.zeros(0)

        # ---- gas: per-volume densities (none inside bodies), absorbing density
        xi_crit = [10.0**v.log_xi_crit for v in vols]

        def gas(p):
            excl = any_inside(p)
            nhs = [jnp.where(excl, 0.0, v.density(p)) for v in vols]
            tot = sum(nhs) if nhs else jnp.zeros(p.shape[:-1])
            if any(v.absorption == "xi" for v in vols) and all_pos.shape[0]:
                flux = jnp.zeros_like(tot)
                for k in range(all_pos.shape[0]):
                    flux = flux + all_lx[k] / jnp.maximum(jnp.sum((p - all_pos[k]) ** 2, -1), 1.0)
                xi = flux / (NE_PER_H * jnp.maximum(tot, 1e-30))
            else:
                xi = jnp.zeros_like(tot)
            nabs = jnp.zeros_like(tot)
            for v, nh, xc in zip(vols, nhs, xi_crit):
                if v.absorption == "neutral":
                    nabs = nabs + nh
                elif v.absorption == "xi":
                    nabs = nabs + nh * xc / (xi + xc)
            return nhs, nabs, NE_PER_H * tot, excl

        # ---- column / shadow grids around every illuminator point
        n_th, n_ph, n_r = self.column_grid
        r_min, r_max = 1e-4 * S, 4.0 * S
        th_c = (np.arange(n_th) + 0.5) * np.pi / n_th
        ph_c = (np.arange(n_ph) + 0.5) * 2 * np.pi / n_ph
        r_g = np.geomspace(r_min, r_max, n_r)
        T, P_ = np.meshgrid(th_c, ph_c, indexing="ij")
        dirs = jnp.asarray(np.stack([np.sin(T) * np.cos(P_), np.sin(T) * np.sin(P_), np.cos(T)],
                                    -1).reshape(-1, 3))
        r_j = jnp.asarray(r_g)

        @jax.jit
        def column_grid(pos):
            pts = pos + dirs[:, None, :] * r_j[None, :, None]
            _, nabs, ne, excl = gas(pts)
            dr = jnp.diff(r_j)

            def wrap(a):
                return jnp.concatenate([a[:, -1:], a, a[:, :1]], axis=1)

            def cum(q):
                seg = 0.5 * (q[:, 1:] + q[:, :-1]) * dr
                c = jnp.concatenate([q[:, :1] * r_j[0], q[:, :1] * r_j[0]
                                     + jnp.cumsum(seg, axis=1)], axis=1)
                return wrap(c.reshape(n_th, n_ph, n_r))

            blk = jax.lax.cummax(excl.astype(float), axis=1).reshape(n_th, n_ph, n_r)
            return jnp.stack([cum(nabs), cum(ne)]), wrap(blk)

        grids = [column_grid(jnp.asarray(pp)) for g in groups for pp in g["pos"]]
        col_grids = jnp.stack([c for c, _ in grids]) if grids else None
        blk_grids = jnp.stack([b for _, b in grids]) if grids else None
        g_slices, k0 = [], 0
        for g in groups:
            g_slices.append(slice(k0, k0 + len(g["nph"])))
            k0 += len(g["nph"])
        dlog = np.log(r_g[1] / r_g[0])

        def from_illum(p, pos, colg, blkg):
            """(absorbing column, electron column, lit fraction) between the illuminator
            at pos and points p."""
            w = p - pos
            r = jnp.linalg.norm(w, axis=-1)
            th = jnp.arccos(jnp.clip(w[..., 2] / jnp.maximum(r, 1e-30), -1, 1))
            ph = jnp.mod(jnp.arctan2(w[..., 1], w[..., 0]), 2 * np.pi)
            c = [th / np.pi * n_th - 0.5, ph / (2 * np.pi) * n_ph + 0.5,
                 jnp.log(jnp.maximum(r, r_min) / r_min) / dlog]
            col = map_coordinates(colg[0], c, order=1, mode="nearest")
            cole = map_coordinates(colg[1], c, order=1, mode="nearest")
            # shadow from the node at or inside the point's radius (towards the
            # illuminator), so gas just in front of a lit surface is not darkened
            c_in = [c[0], c[1], jnp.floor(c[2])]
            lit = jnp.clip(1.0 - map_coordinates(blkg, c_in, order=1, mode="nearest"), 0.0, 1.0)
            col, cole = jnp.maximum(col, 0.0), jnp.maximum(cole, 0.0)
            if not self.absorb_illumination:
                col, cole = jnp.zeros_like(col), jnp.zeros_like(cole)
            return col, cole, lit

        # ---- templates and per-(object, group) constants
        mu_edges = np.asarray(self.mu_edges, float)
        refl = [[reflection_templates(edges, g["spec"], g["spectrum"], mu_edges)
                 if b.reflection == "neutral" else ([g["spectrum"]], np.array([0.0, 1.0]))
                 for g in groups] for b in bodies]
        scat = [[scatter_template(g["spectrum"], v.scatter, v.lines) for g in groups]
                for v in vols]
        albedo = jnp.asarray([b.albedo if b.reflection else 0.0 for b in bodies] or [0.0])
        nh_bins = jnp.asarray(self.nh_bins, dtype=float)
        nb = nh_bins.size

        def binned(w, col, qs):
            """Sum over samples of w * q for each q, per column bin: (R, nb, len(qs))."""
            kb = jnp.clip(jnp.searchsorted(nh_bins, col, side="right") - 1, 0, nb - 1)
            oh = jax.nn.one_hot(kb, nb) * w[..., None]  # (R, N, nb)
            return jnp.stack([jnp.sum(oh * q[..., None], axis=1) for q in qs], -1)

        n_c = self.n_coarse

        def dl(bl, k):
            """Doppler factor of an illuminator moving at bl (units of c) towards k."""
            return 1.0 / (jnp.sqrt(1.0 - jnp.sum(bl * bl)) * (1.0 - jnp.sum(bl * k, -1)))

        @jax.jit
        def trace(xs, ys, col_grids, blk_grids):
            sx, sy = (xs - self.x) * cm_per_uas, (ys - self.y) * cm_per_uas
            p0 = ((sx * u_a[0] + sy * u_a[1])[:, None] * e_a
                  + (sx * u_b[0] + sy * u_b[1])[:, None] * e_b)
            R = xs.size
            out = {}

            # -- bodies: front-most surface
            if bodies:
                hits = []
                for b in bodies:
                    bd = b.bound
                    if bd["kind"] == "slab":
                        sb = slab_samples(p0, n, bd["centre"], bd["radius"], bd["half"], bd["n"],
                                          axis=bd["axis"])
                    else:
                        sb, _, _ = sphere_samples(p0, n, bd["centre"], bd["radius"], bd["n"])
                    hits.append(first_hit(b.inside, p0, sb, n))
                s_hit = jnp.stack([h[1] for h in hits], -1)
                which = jnp.argmax(s_hit, axis=-1)
                s_front = jnp.max(s_hit, axis=-1)
                any_hit = jnp.any(jnp.stack([h[0] for h in hits], -1), axis=-1)
            else:
                which = jnp.zeros(R, dtype=int)
                s_front = jnp.full(R, -jnp.inf)
                any_hit = jnp.zeros(R, dtype=bool)

            # -- ray samples: coarse over the whole ray, fine per volume
            parts = [jnp.broadcast_to(jnp.linspace(S, -S, n_c), (R, n_c))]
            names = [b.name for b in bodies]
            for v in vols:
                nf = v.n_fine
                if v.sampling == "uniform" or v.refine is None:
                    sv, _, _ = sphere_samples(p0, n, v.bound["centre"], v.bound["radius"], nf)
                    parts.append(jnp.where(sv < -1e299, S, sv))
                    continue
                rf = v.refine
                _, s_c, b_imp = sphere_samples(p0, n, rf["centre"], rf["radius"], 2)
                half = jnp.sqrt(jnp.clip(rf["radius"] ** 2 - b_imp**2, 0.0, None))
                if rf["body"] in names:
                    on_body = any_hit & (which == names.index(rf["body"]))
                else:
                    on_body = jnp.zeros(R, dtype=bool)
                anchor = jnp.where(on_body, s_front, s_c)
                t2 = (jnp.arange(1, nf + 1) / nf) ** 2
                s_fwd = anchor[:, None] + (s_c + half - anchor)[:, None] * t2[None, :]
                s_bwd = anchor[:, None] - jnp.where(on_body, 0.0, anchor - (s_c - half))[:, None] \
                    * t2[None, :]
                parts += [s_fwd, s_bwd, anchor[:, None]]
            s_all = -jnp.sort(-jnp.clip(jnp.concatenate(parts, 1), -S, S), 1)
            ds = 0.5 * (jnp.concatenate([s_all[:, :1], s_all[:, :-1]], 1)
                        - jnp.concatenate([s_all[:, 1:], s_all[:, -1:]], 1))
            p = p0[:, None, :] + s_all[..., None] * n
            nhs, nabs, ne, excl = gas(p)
            front = s_all > s_front[:, None]
            col_f = jnp.cumsum(nabs * ds, axis=1) - nabs * ds
            ne_f = jnp.cumsum(ne * ds, axis=1) - ne * ds
            scat_nh = [nh for v, nh in zip(vols, nhs) if v.scatter > 0 or v.lines]
            out["tau_es"] = jnp.sum(jnp.where(front, NE_PER_H * SIGMA_T * sum(scat_nh) * ds, 0.0),
                                    axis=1) if scat_nh else jnp.zeros(R)
            tau_o = -jnp.sum(p * n, -1) / C_CM_S

            # -- volumes: scattering of each illuminator group
            for iv, v in enumerate(vols):
                nh = nhs[iv]
                if v.scatter > 0 or v.lines:
                    vel = v.velocity(p)
                    beta = vel / C_CM_S
                    bn = 1 - jnp.sum(beta * n, -1)
                    d_out = doppler(vel, n)
                    for ig, g in enumerate(groups):
                        sl = g_slices[ig]
                        boost = scat[iv][ig][1]

                        def step(acc, xk, nh=nh, beta=beta, bn=bn, boost=boost, d_out=d_out):
                            pos, bl, nphk, colg, blkg = xk
                            w = p - pos
                            dk = jnp.linalg.norm(w, axis=-1)
                            col_in, ne_in, lit = from_illum(p, pos, colg, blkg)
                            fl = nphk / (4 * np.pi * jnp.maximum(dk, 1.0) ** 2) * lit
                            k_in = w / jnp.maximum(dk, 1.0)[..., None]
                            d_in = dl(bl, k_in)
                            bk = 1 - jnp.sum(beta * k_in, -1)
                            gw = d_in * bk / bn
                            eta = NE_PER_H * nh * SIGMA_T * fl / (4 * np.pi) * boost
                            ww = jnp.where(front, eta * d_in**3 * bk * d_out**2 * ds, 0.0)
                            tau = (dk - jnp.sum(w * n, -1)) / C_CM_S - jnp.dot(pos, n) / C_CM_S
                            col = col_f + col_in
                            return acc + binned(ww, col, (jnp.ones_like(gw), gw, gw**2, tau,
                                                          col, ne_f + ne_in)), None

                        xk = (jnp.asarray(g["pos"]), jnp.asarray(g["vel"]) / C_CM_S,
                              jnp.asarray(g["nph"]), col_grids[sl], blk_grids[sl])
                        acc, _ = jax.lax.scan(step, jnp.zeros((R, nb, 6)), xk)
                        out[f"scat:{iv}:{ig}"] = acc
                em = v.emission
                if em:
                    jc, jk = em["j"](p)
                    jc, jk = jnp.where(excl, 0.0, jc), jnp.where(excl, 0.0, jk)
                    gv = doppler(v.velocity(p), n)
                    ww = jnp.where(front, (jc * gv**2 + jk * gv**3) * ds, 0.0)
                    out[f"vemit:{iv}"] = binned(ww, col_f, (jnp.ones_like(gv), gv, gv**2, tau_o,
                                                            col_f, ne_f))

            # -- surfaces: reflection of each illuminator group, own emission
            if bodies:
                ph = p0 + jnp.where(any_hit, s_front, 0.0)[:, None] * n
                nrm = jnp.zeros_like(ph)
                vel_s = jnp.zeros_like(ph)
                for ib, b in enumerate(bodies):
                    sel = (any_hit & (which == ib))[:, None]
                    nrm = jnp.where(sel, surface_normal(b.field, ph), nrm)
                    vel_s = jnp.where(sel, b.velocity(ph), vel_s)
                nrm = jnp.nan_to_num(nrm)
                mu_e = jnp.sum(nrm * n, -1)
                g_s = doppler(vel_s, n)
                b_s = vel_s / C_CM_S
                gam_s = 1.0 / jnp.sqrt(1.0 - jnp.sum(b_s * b_s, -1))
                bn_s = 1 - jnp.sum(b_s * n, -1)
                col_s = jnp.sum(jnp.where(front, nabs * ds, 0.0), axis=1)
                ne_s = jnp.sum(jnp.where(front, ne * ds, 0.0), axis=1)
                vis = any_hit & (mu_e > 0)
                mu_c = jnp.clip(mu_e, 1e-6, 1.0)
                tau_s0 = -jnp.sum(ph * n, -1) / C_CM_S
                p_off = ph + nrm * (1e-6 * S)
                alb = albedo[which]
                for ig, g in enumerate(groups):
                    sl = g_slices[ig]

                    def sstep(acc, xk):
                        pos, bl, nphk, colg, blkg = xk
                        to_c = pos - ph
                        dist = jnp.linalg.norm(to_c, axis=-1)
                        mu_i = jnp.sum(nrm * to_c, -1) / dist
                        k_in = -to_c / jnp.maximum(dist, 1.0)[:, None]
                        d_in = dl(bl, k_in)
                        bk = 1 - jnp.sum(b_s * k_in, -1)
                        gr = d_in * bk / bn_s
                        lit = ~segment_blocked(inside_fns, p_off, jnp.broadcast_to(pos, ph.shape))
                        col_in, ne_in, _ = from_illum(ph, pos, colg, blkg)
                        ok = vis & lit & (mu_i > 0)
                        i_r = jnp.where(ok, alb * REFLECTION_C * nphk / (4 * np.pi * dist**2)
                                        * mu_i / (mu_i + mu_c), 0.0)
                        w = jnp.nan_to_num(i_r) * d_in**3 * gam_s * bk * g_s**3
                        tau = (dist + jnp.sum(to_c * n, -1)) / C_CM_S - jnp.dot(pos, n) / C_CM_S
                        col = col_s + col_in
                        return acc + jnp.stack([w, w * gr, w * gr**2, w * tau, w * col,
                                                w * mu_c, w * (ne_s + ne_in)], -1), None

                    xk = (jnp.asarray(g["pos"]), jnp.asarray(g["vel"]) / C_CM_S,
                          jnp.asarray(g["nph"]), col_grids[sl], blk_grids[sl])
                    acc, _ = jax.lax.scan(sstep, jnp.zeros((R, 7)), xk)
                    out[f"refl:{ig}"] = acc
                for ib, b in enumerate(bodies):
                    if b.emission:
                        i_e = jnp.where(vis & (which == ib), b.emission["intensity"](ph), 0.0)
                        w = i_e * g_s**3
                        out[f"bemit:{ib}"] = jnp.stack([w, w * g_s, w * g_s**2, w * tau_s0,
                                                        w * col_s, w * mu_c, w * ne_s], -1)
                out["which"] = jnp.where(any_hit, which, -1)
            return out

        # ---- trace sub-pixel rays over the grid, in chunks
        xc, yc = scene.grid.centres()
        size = scene.grid.sizes()
        ss = self.supersample
        uu = (np.arange(ss) + 0.5) / ss - 0.5
        du, dv = (m.ravel() for m in np.meshgrid(uu, uu))
        rx = (xc[:, None] + du[None, :] * size[:, None]).ravel()
        ry = (yc[:, None] + dv[None, :] * size[:, None]).ravel()
        area = np.repeat((size / ss / UAS_PER_RAD) ** 2, ss * ss)  # sr per sub-ray
        nray = rx.size
        chunks = []
        for i0 in range(0, nray, self.chunk):
            m = min(self.chunk, nray - i0)
            pad = self.chunk - m
            xs = np.pad(rx[i0:i0 + m], (0, pad), mode="edge")
            ys = np.pad(ry[i0:i0 + m], (0, pad), mode="edge")
            o = trace(jnp.asarray(xs), jnp.asarray(ys), col_grids, blk_grids)
            chunks.append({k: np.asarray(v)[:m] for k, v in o.items()})
        res = {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]} if chunks else {}

        tau_max = float(res["tau_es"].max()) if res else 0.0
        if tau_max > 0.3:
            warnings.warn(f"{self.name}: electron-scattering optical depth along some rays is "
                          f"{tau_max:.2g}; the single-scattering, optically thin treatment "
                          "is inaccurate above ~0.3", stacklevel=2)
        npix = scene.grid.n
        ssq = ss * ss

        def pix(v):
            return v.reshape(npix, ssq, *v.shape[1:]).sum(axis=1)

        layers = []
        multi = len(groups) > 1

        def add_volume_layers(acc, base, tmpl, driver):
            acc = pix(acc * area[:, None, None])  # (npix, nb, 5)
            for kb in range(nb):
                w = acc[:, kb, 0]
                if w.sum() <= 0:
                    continue
                sf = np.where(w > 0, w, 1.0)
                gm = np.where(w > 0, acc[:, kb, 1] / sf, 1.0)
                g2 = acc[:, kb, 2] / sf
                maps = {"NORM": w, "G": gm,
                        "SIGMA": np.where(w > 0, np.sqrt(np.clip(g2 - gm**2, 0, None)) / gm, 0.0),
                        "DELAY": acc[:, kb, 3] / sf, "NH": self.nh_ism + acc[:, kb, 4] / sf,
                        "NE": acc[:, kb, 5] / sf}
                layers.append(SpecMapLayer(f"{base}{kb}", scene.grid, [tmpl], maps,
                                           driver=driver))

        def add_surface_layer(acc, mask, name, tmpls, tedges, driver):
            w = pix(np.where(mask, acc[:, 0], 0.0) * area)
            if w.sum() <= 0:
                return
            q = [pix(np.where(mask, acc[:, k], 0.0) * area) for k in range(1, 7)]
            sf = np.where(w > 0, w, 1.0)
            gm = np.where(w > 0, q[0] / sf, 1.0)
            g2 = q[1] / sf
            mu = q[4] / sf
            tidx = np.clip(np.digitize(mu, tedges) - 1, 0, len(tmpls) - 1)
            maps = {"NORM": w, "G": gm,
                    "SIGMA": np.where(w > 0, np.sqrt(np.clip(g2 - gm**2, 0, None)) / gm, 0.0),
                    "DELAY": q[2] / sf, "NH": self.nh_ism + q[3] / sf, "TIDX": tidx,
                    "NE": q[5] / sf}
            layers.append(SpecMapLayer(name, scene.grid, tmpls, maps, driver=driver))

        for iv, v in enumerate(vols):
            for ig, g in enumerate(groups):
                if f"scat:{iv}:{ig}" in res:
                    base = f"{self.name}_{v.name}" + (f"_{g['name']}" if multi else "")
                    add_volume_layers(res[f"scat:{iv}:{ig}"], base, scat[iv][ig][0], g["driver"])
            if f"vemit:{iv}" in res:
                em = v.emission
                add_volume_layers(res[f"vemit:{iv}"], f"{self.name}_{v.name}_emit",
                                  build_spectrum(em["spectrum"], edges), em["driver"])
        if bodies:
            which = res["which"]
            for ib, b in enumerate(bodies):
                for ig, g in enumerate(groups):
                    if b.reflection and f"refl:{ig}" in res:
                        tm, te = refl[ib][ig]
                        add_surface_layer(res[f"refl:{ig}"], which == ib,
                                          f"{self.name}_{b.name}"
                                          + (f"_{g['name']}" if multi else ""), tm, te,
                                          g["driver"])
                if f"bemit:{ib}" in res:
                    em = b.emission
                    add_surface_layer(res[f"bemit:{ib}"], which == ib, f"{self.name}_{b.name}_emit",
                                      [build_spectrum(em["spectrum"], edges)], np.array([0.0, 1.0]),
                                      em["driver"])

        # ---- lamps: visibility and column along the line of sight
        info = {}
        n_np = np.asarray(f["n"])
        for lp in lamps:
            s_los = np.linspace(0.0, 2 * S, 20001)
            los = jnp.asarray(lp.pos[None, :] + s_los[:, None] * n_np)
            _, nabs_los, ne_los, excl = gas(los)
            blocked = bool(np.asarray(excl[1:]).any())
            col = float(np.trapezoid(np.asarray(nabs_los), s_los))
            col_e = float(np.trapezoid(np.asarray(ne_los), s_los))
            g_l = float(doppler(jnp.asarray(lp.vel), n))
            info[lp.name] = dict(column=col, electron_column=col_e, blocked=blocked, g=g_l)
            if not lp.visible or blocked:
                continue
            sky = (lp.pos @ f["e_a"]) * u_a + (lp.pos @ f["e_b"]) * u_b
            grid = RegularGrid(nx=1, ny=1, scale=1e-3, x0=self.x + sky[0] / cm_per_uas,
                               y0=self.y + sky[1] / cm_per_uas)
            sp = build_spectrum(lp.spectrum, edges)
            layers.append(SpecMapLayer(
                f"{self.name}_{lp.name}", grid, [sp],
                {"NORM": np.array([sp.photon_flux() * g_l**3]), "G": np.array([g_l]),
                 "NH": np.array([self.nh_ism + col]), "NE": np.array([col_e]),
                 "DELAY": np.array([-float(lp.pos @ n_np) / C_CM_S])}, driver=lp.driver))
        self._last = dict(lamps=info, cm_per_uas=cm_per_uas, extent_cm=S, tau_es_max=tau_max,
                          groups=[dict(name=g["name"], nph=g["nph"], pos=g["pos"]) for g in groups])
        return layers


def _discretise(v, em, any_inside, rng, n_cand=20000, max_cand=2_560_000, min_hits=4000):
    """Point illuminators from an emitting volume: positions drawn ~ emissivity, each
    carrying an equal share of the volume's photon rate 4 pi integral(j dV), estimated by
    Monte Carlo over the bounding sphere (more candidates until ``min_hits`` land in the
    emitting region)."""
    c, R = np.asarray(v.bound["centre"]), float(v.bound["radius"])
    while True:
        u = rng.normal(size=(n_cand, 3))
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        pts = c + u * R * rng.random(n_cand)[:, None] ** (1 / 3)
        pj = jnp.asarray(pts)
        jc, jk = em["j"](pj)
        j = np.asarray(jnp.where(any_inside(pj), 0.0, jc + jk))
        hits = int(np.count_nonzero(j))
        if hits >= min_hits or n_cand >= max_cand:
            break
        n_cand *= 4
    if j.sum() <= 0:
        raise ValueError(f"{v.name}: no emission inside its bound to illuminate with")
    rel_err = j.std() / j.mean() / np.sqrt(n_cand)
    if rel_err > 0.02:
        warnings.warn(f"{v.name}: its photon rate as an illuminator is uncertain by "
                      f"{100 * rel_err:.0f}% (emission fills little of its bound)", stacklevel=3)
    total = 4 * np.pi * j.mean() * (4 / 3) * np.pi * R**3  # photons / s
    k = em["n_points"]
    cdf = np.cumsum(j) / j.sum()
    idx = np.searchsorted(cdf, (np.arange(k) + rng.random()) / k)
    return pts[np.minimum(idx, n_cand - 1)], np.full(k, total / k)
