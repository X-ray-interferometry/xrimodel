"""An X-ray binary built from 3-D parts: Roche-lobe donor, flared disc, corona, wind.

The component renders, at one orbital phase, into these layers:

==================  ==============================================================
``<name>_corona``   point source at the compact object (the illuminator)
``<name>_disc_bb``  thermal disc emission (unresolved: a point source)
``<name>_star``     X-rays reflected off the donor's irradiated face, shadowed by the disc
``<name>_disc``     X-rays reflected off the flared disc's face
``<name>_wind<k>``  electron-scattered corona light (plus Fe XXV/XXVI lines) from the
                    donor's beta-law wind, split by the absorbing column in front of the
                    emission (bin k of ``wind_nh_bins``) so each layer's NH is accurate
==================  ==============================================================

Every layer carries Doppler factors (orbital motion, corotation, Keplerian disc, wind
outflow), light-travel delays relative to the corona, and absorbing columns (ISM plus
the not-ionised part of the wind in front of each emitter). The reprocessed layers follow
the corona's driver, delayed. Reflected and scattered intensities are normalised from the
corona's photon luminosity, so their fluxes relative to the corona are predictions.
``corona_spectrum`` is the corona's emitted spectrum (as a stationary observer at the
system's distance would see it); the orbital Doppler factor is applied on top.

Frame: orbital plane (X, Y), Z along the orbital angular momentum, centre of mass at the
origin. The observer direction is n = (0, -sin i, cos i). Phase 0 is superior conjunction
of the compact object (it is behind the donor). On the sky, the +X node (where the compact
object recedes fastest, phase 0.75) lies at position angle ``node_pa`` East of North and
the orbit runs counterclockwise for i < 90 deg. The compact object is at sky offset (x, y).

Simplifications (v0): circular orbit; the donor is the Roche equipotential with the given
volume-equivalent radius (the wind and its shadow use that radius as a sphere); the wind
is spherical (not focused), its ionisation is a smooth threshold in xi = L / (n r^2) with
L the corona luminosity in the scene's energy band, and it does not absorb the light that
illuminates the reflectors; the disc does not shadow the wind; the disc's height tapers to
zero at ``r_disc_in``; reflection uses one angular law, mu_i / (mu_i + mu_e), with pexmon
(or an approximate) spectral shape per emission-angle bin.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from ..format.grid import RegularGrid
from ..format.layers import SpecMapLayer
from ..format.spectra import Spectrum, gaussian_line, sigma_photoabs
from ..units import C_CM_S, G_CGS, MSUN_G, PC_CM, UAS_PER_RAD
from .raytrace import (
    REFLECTION_C,
    doppler,
    first_hit,
    segment_blocked,
    slab_samples,
    sphere_samples,
    sphere_shadowed,
    surface_normal,
)
from .spec import build_spectrum

RSUN_CM = 6.957e10
AU_CM = 1.495978707e13
YEAR_S = 3.15576e7
M_H = 1.6735575e-24
SIGMA_T = 6.6524587e-25
MU_H = 1.4  # mass per hydrogen atom in units of m_H (solar composition)
NE_PER_H = 1.2


# --------------------------------------------------------------------------- spectra


def reflection_templates(edges, gamma, mu_edges):
    """pexmon templates per emission-angle bin if XSPEC models are installed, else one
    approximate template: E^-gamma times a photoabsorption-limited albedo, plus Fe K-alpha."""
    try:
        from ..adapters.xspec import pexmon_templates

        return pexmon_templates(edges, gamma=gamma, mu_edges=mu_edges)
    except ImportError:
        e = 0.5 * (edges[1:] + edges[:-1])
        sig_t = NE_PER_H * SIGMA_T
        albedo = sig_t / (sig_t + sigma_photoabs(e))
        cont = Spectrum(edges, e**-gamma * albedo, "refl_cont")
        line = gaussian_line(edges, 6.4, 0.02,
                             flux=0.9 * np.interp(6.4, e, cont.values), name="fe_ka")
        t = Spectrum(edges, cont.values + line.values, "refl_approx")
        return [t], np.array([0.0, 1.0])


def wind_template(corona: Spectrum, line_ew_kev: float):
    """Electron-scattered corona continuum plus Fe XXV (6.70 keV) and Fe XXVI (6.97 keV)."""
    v = corona.values.copy()
    for e0 in (6.70, 6.97):
        cont = np.interp(e0, corona.centres, corona.values)
        v = v + gaussian_line(corona.edges, e0, 0.01, flux=0.5 * line_ew_kev * cont).values
    return Spectrum(corona.edges, v, "wind")


# --------------------------------------------------------------------------- component


@dataclass
class XRBinary:
    name: str
    corona_spectrum: dict  # observed (unabsorbed) spectrum of the corona, a spectrum spec
    m_donor: float = 40.6  # Msun
    m_compact: float = 21.2  # Msun
    period_days: float = 5.599829
    inclination: float = 27.51  # deg
    node_pa: float = 64.0  # deg East of North
    phase: float = 0.0
    r_donor: float = 22.3  # Rsun, volume-equivalent
    disc_spectrum: dict | None = None  # thermal disc (unresolved), a spectrum spec
    r_disc_in: float = 0.01  # units of a (the reflecting part of the disc)
    r_disc_out: float = 0.15  # units of a
    disc_h_over_r: float = 0.1
    albedo: float = 0.1  # band photon albedo at normal incidence
    mdot: float = 3e-6  # Msun / yr
    v_inf: float = 2100.0  # km/s
    beta: float = 1.05
    log_xi_crit: float = 2.0  # wind gas with log xi above this no longer absorbs
    wind_line_ew: float = 0.05  # keV, Fe XXV + XXVI against the scattered continuum
    wind_extent: float = 2.5  # wind traced out to this many a along each ray
    v_wind_base: float = 20.0  # km/s, beta-law speed at the donor's surface
    nh_ism: float = 5.4e21
    x: float = 0.0  # uas, sky offset of the compact object
    y: float = 0.0
    driver: str | None = None
    supersample: int = 2
    n_steps: int = 128  # coarse wind samples along each ray
    n_fine: int = 96  # wind samples on each side of the donor, clustered at its surface
    n_body: int = 96  # samples across the donor's bounding sphere when finding its surface
    n_disc: int = 1024  # samples across the disc's slab when finding its surface
    wind_nh_bins: tuple = (0.0, 1e21, 1e22, 1e23)  # cm^-2, lower edges of the wind layers
    chunk: int = 8192

    kind = "xrbinary"

    # ------------------------------------------------------------------ geometry

    def separation_cm(self):
        m = (self.m_donor + self.m_compact) * MSUN_G
        p = self.period_days * 86400.0
        return (G_CGS * m * p**2 / (4 * np.pi**2)) ** (1.0 / 3.0)

    def frame(self):
        a = self.separation_cm()
        q = self.m_compact / self.m_donor
        th = np.pi / 2 + 2 * np.pi * self.phase
        u = np.array([np.cos(th), np.sin(th), 0.0])
        bh = a * self.m_donor / (self.m_donor + self.m_compact) * u
        donor = -a * self.m_compact / (self.m_donor + self.m_compact) * u
        inc = np.deg2rad(self.inclination)
        n = np.array([0.0, -np.sin(inc), np.cos(inc)])
        e_a = np.array([1.0, 0.0, 0.0])
        e_b = np.array([0.0, np.cos(inc), np.sin(inc)])
        omega = 2 * np.pi / (self.period_days * 86400.0)
        return dict(a=a, q=q, bh=bh, donor=donor, n=n, e_a=e_a, e_b=e_b, omega=omega)

    def sky_axes(self):
        """Unit sky vectors (East, North components) of the frame axes e_a and e_b.

        (e_a, e_b, n) is right-handed but (East, North, towards the observer) is
        left-handed, so the map is a reflection: e_a (the +X node, where the compact
        object recedes fastest, at phase 0.75) points to position angle node_pa, and e_b
        to node_pa + 90 deg. With the orbital angular momentum towards the observer
        (i < 90 deg) the orbit then runs counterclockwise on the sky (position angle
        increasing with phase).
        """
        pa = np.deg2rad(self.node_pa)
        return np.array([np.sin(pa), np.cos(pa)]), np.array([np.cos(pa), -np.sin(pa)])

    def _roche(self, f):
        """Roche potential (units G M_donor / a) and the surface level for r_donor."""
        a, q = f["a"], f["q"]
        d, b = f["donor"] / a, f["bh"] / a

        def phi(u):
            r1 = jnp.linalg.norm(u - d, axis=-1)
            r2 = jnp.linalg.norm(u - b, axis=-1)
            return -1.0 / r1 - q / r2 - 0.5 * (1 + q) * (u[..., 0] ** 2 + u[..., 1] ** 2)

        # L1 between the stars, by bisection on d(phi)/dt along the axis
        axis = (b - d) / np.linalg.norm(b - d)
        dphi = jax.grad(lambda t: phi(d + t * axis))
        lo, hi = 1e-3, 1.0 - 1e-3
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if float(dphi(mid)) > 0 else (lo, mid)
        t_l1 = 0.5 * (lo + hi)
        phi_l1 = float(phi(d + t_l1 * axis))

        # surface level whose volume-equivalent radius is r_donor (grid volume, bisection)
        r_target = self.r_donor * RSUN_CM / a
        g = np.linspace(-t_l1, t_l1, 72)
        X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
        pts = d + np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
        ph = np.asarray(phi(jnp.asarray(pts)))
        # the donor's lobe: towards the companion up to L1, and within the L1 distance of the
        # donor's centre (far from the system the centrifugal term makes phi small again)
        lobe = ((pts - d) @ axis < t_l1) & (np.linalg.norm(pts - d, axis=-1) < t_l1)
        cell = (g[1] - g[0]) ** 3
        if (4 / 3) * np.pi * r_target**3 >= np.sum((ph < phi_l1) & lobe) * cell:
            raise ValueError("r_donor overfills the Roche lobe")
        lo, hi = 1.01 * float(np.min(ph)), phi_l1  # phi is negative; deeper = smaller star
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            r_eq = (3 * np.sum((ph < mid) & lobe) * cell / (4 * np.pi)) ** (1 / 3)
            lo, hi = (mid, hi) if r_eq < r_target else (lo, mid)
        return phi, 0.5 * (lo + hi), t_l1, axis

    # ------------------------------------------------------------------ rendering

    def render(self, scene):
        f = self.frame()
        a, n = f["a"], jnp.asarray(f["n"])
        bh, donor = jnp.asarray(f["bh"]), jnp.asarray(f["donor"])
        omega = f["omega"]
        d_cm = scene.distance_pc * PC_CM
        cm_per_uas = d_cm / UAS_PER_RAD
        phi, phi_s, t_l1, axis = self._roche(f)
        axis = jnp.asarray(axis)
        r_star = self.r_donor * RSUN_CM

        # spectra and the corona's photon luminosity
        edges = scene.energy_edges
        corona = build_spectrum(self.corona_spectrum, edges)
        n_ph = 4 * np.pi * d_cm**2 * corona.photon_flux()  # photons / s
        l_x = 4 * np.pi * d_cm**2 * corona.energy_flux()  # erg / s (in the band)
        gamma_c = float(self.corona_spectrum.get("gamma", 1.7))
        refl_tmpls, mu_edges = reflection_templates(edges, gamma_c, (0.0, 0.2, 0.4, 0.6, 0.8, 1.0))
        wind_t = wind_template(corona, self.wind_line_ew)
        wind_boost = wind_t.photon_flux() / corona.photon_flux()

        # ---- implicit solids (positions in cm)
        def inside_donor(p):
            u = p / a
            du = u - donor / a
            return ((phi(u) < phi_s) & (jnp.sum(du * axis, axis=-1) < t_l1)
                    & (jnp.sum(du * du, axis=-1) < t_l1**2))

        def field_donor(p):
            return phi(p / a)

        r_in, r_out, hr = self.r_disc_in * a, self.r_disc_out * a, self.disc_h_over_r

        def disc_height(r):
            # flared, H/R ~ r^(1/8), so the face sees the centre; tapered to zero at r_in so
            # the truncation leaves no wall facing the corona
            rr = jnp.maximum(r, 1.0)
            return hr * r_out * (rr / r_out) ** 1.125 * jnp.clip(1.0 - r_in / rr, 0.0, 1.0)

        def inside_disc(p):
            w = p - bh
            r = jnp.hypot(w[..., 0], w[..., 1])
            return (r >= r_in) & (r <= r_out) & (jnp.abs(w[..., 2]) <= disc_height(r))

        def field_disc(p):
            w = p - bh
            r = jnp.sqrt(w[..., 0] ** 2 + w[..., 1] ** 2 + 1.0)
            return jnp.maximum(jnp.maximum(jnp.abs(w[..., 2]) - disc_height(r), r - r_out),
                               r_in - r)

        def v_corot(p):  # rigid rotation of the binary frame about Z
            return omega * jnp.stack([-p[..., 1], p[..., 0], jnp.zeros_like(p[..., 0])], -1)

        v_bh = v_corot(bh)
        gm_bh = G_CGS * self.m_compact * MSUN_G

        def v_disc(p):
            w = p - bh
            r = jnp.hypot(w[..., 0], w[..., 1])
            vk = jnp.sqrt(gm_bh / r)
            return v_bh + jnp.stack([-w[..., 1] / r * vk, w[..., 0] / r * vk,
                                     jnp.zeros_like(r)], -1)

        mdot = self.mdot * MSUN_G / YEAR_S
        v_inf = self.v_inf * 1e5
        xi_crit = 10.0**self.log_xi_crit

        # beta law v = v_inf (1 - b R/r)^beta with b set by the base speed v(R) = v_base
        b_beta = 1.0 - (self.v_wind_base / self.v_inf) ** (1.0 / self.beta)

        def wind(p):
            """(n_H, absorbing n_H, velocity) of the wind at p; zero inside the donor."""
            w = p - donor
            r = jnp.linalg.norm(w, axis=-1)
            rs = jnp.maximum(r, r_star)
            v = v_inf * (1 - b_beta * r_star / rs) ** self.beta
            nh = mdot / (4 * np.pi * rs**2 * v * MU_H * M_H)
            nh = jnp.where(inside_donor(p) | (r < r_star), 0.0, nh)
            d2 = jnp.sum((p - bh) ** 2, axis=-1)
            xi = l_x / (NE_PER_H * jnp.maximum(nh, 1e-30) * d2)
            vel = v_corot(donor) + (w / jnp.maximum(r, 1.0)[..., None]) * v[..., None]
            return nh, nh * xi_crit / (xi + xi_crit), vel

        solids = (inside_donor, inside_disc)

        # sky offset 0 is the compact object; s = 0 is the sky plane through the CM
        origin = jnp.asarray(np.asarray(f["bh"]) - np.dot(f["bh"], f["n"]) * f["n"])
        e_a, e_b = jnp.asarray(f["e_a"]), jnp.asarray(f["e_b"])
        u_a, u_b = self.sky_axes()
        s_max = self.wind_extent * a
        r_donor_bound = t_l1 * a
        r_disc_bound = r_out * np.hypot(1.0, hr) * 1.01
        r_shell = min(2.5 * r_star, s_max)
        h_max = hr * r_out
        nc, nf = self.n_steps, self.n_fine
        nh_bins = jnp.asarray(self.wind_nh_bins, dtype=float)

        @jax.jit
        def trace(xs, ys):
            # sky offsets (uas, relative to the compact object) -> ray origins (cm)
            sx, sy = (xs - self.x) * cm_per_uas, (ys - self.y) * cm_per_uas
            p0 = (origin + (sx * u_a[0] + sy * u_a[1])[:, None] * e_a
                  + (sx * u_b[0] + sy * u_b[1])[:, None] * e_b)

            # surfaces: the donor across its bounding sphere, the thin disc across its slab
            sb, _, _ = sphere_samples(p0, n, donor, r_donor_bound, self.n_body)
            sd = slab_samples(p0, n, bh, r_disc_bound, h_max, self.n_disc)
            hits = [first_hit(inside_donor, p0, sb, n), first_hit(inside_disc, p0, sd, n)]
            s_front = jnp.maximum(hits[0][1], hits[1][1])
            which = jnp.where(hits[0][1] >= hits[1][1], 0, 1)
            any_hit = hits[0][0] | hits[1][0]
            hit_donor = hits[0][0] & (which == 0)

            # wind samples: coarse over the whole ray outside a shell round the donor, and
            # fine inside it, clustered quadratically at the donor's surface (where the ray
            # hits it) or at the closest approach to the donor (where it passes by)
            _, s_c, b_imp = sphere_samples(p0, n, donor, r_shell, 2)
            half = jnp.sqrt(jnp.clip(r_shell**2 - b_imp**2, 0.0, None))
            anchor = jnp.where(hit_donor, s_front, s_c)
            t2 = (jnp.arange(1, nf + 1) / nf) ** 2
            s_fwd = anchor[:, None] + (s_c + half - anchor)[:, None] * t2[None, :]
            s_bwd = anchor[:, None] - jnp.where(hit_donor, 0.0, anchor - (s_c - half))[:, None] \
                * t2[None, :]
            s_crs = jnp.broadcast_to(jnp.linspace(s_max, -s_max, nc), (xs.size, nc))
            in_shell = jnp.abs(s_crs - s_c[:, None]) < half[:, None]
            s_crs = jnp.where(in_shell, anchor[:, None], s_crs)
            s_all = -jnp.sort(-jnp.concatenate([s_fwd, s_bwd, anchor[:, None], s_crs], 1), 1)
            ds = 0.5 * (jnp.concatenate([s_all[:, :1], s_all[:, :-1]], 1)
                        - jnp.concatenate([s_all[:, 1:], s_all[:, -1:]], 1))
            p = p0[:, None, :] + s_all[..., None] * n

            nh, nabs, vel = wind(p)
            front = s_all > s_front[:, None]
            col = jnp.cumsum(nabs * ds, axis=1) - nabs * ds  # column in front of each sample
            d_bh = jnp.linalg.norm(p - bh, axis=-1)
            # gas behind the donor (as seen from the corona) is not illuminated
            lit_w = ~sphere_shadowed(p, bh, donor, r_star)
            f_irr = jnp.where(lit_w, n_ph / (4 * np.pi * d_bh**2), 0.0)
            # electron scattering off moving gas: E_out / E_in = (1 - b.k_in) / (1 - b.n)
            k_in = (p - bh) / jnp.maximum(d_bh, 1.0)[..., None]
            beta = vel / C_CM_S
            g_w = (1 - jnp.sum(beta * k_in, -1)) / (1 - jnp.sum(beta * n, -1))
            eta = NE_PER_H * nh * SIGMA_T * f_irr / (4 * np.pi) * wind_boost  # ph/s/cm3/sr
            # band photon emissivity of a steady flow transforms as g^2
            w_w = jnp.where(front, eta * g_w**2 * ds, 0.0)
            tau_w = (d_bh - jnp.sum((p - bh) * n, axis=-1)) / C_CM_S
            # split by the absorbing column in front, so each layer's mean NH is accurate
            kbin = jnp.clip(jnp.searchsorted(nh_bins, col, side="right") - 1, 0,
                            nh_bins.size - 1)
            onehot = jax.nn.one_hot(kbin, nh_bins.size)  # (R, N, nbin)
            wb = w_w[..., None] * onehot
            wind_out = tuple(jnp.sum(wb * q[..., None], axis=1) for q in
                             (jnp.ones_like(g_w), g_w, g_w**2, tau_w, col))

            # visible surface: reflection of the corona
            ph = p0 + jnp.where(any_hit, s_front, 0.0)[:, None] * n
            nrm = jnp.where((which == 0)[:, None], surface_normal(field_donor, ph),
                            surface_normal(field_disc, ph))
            to_c = bh - ph
            dist = jnp.linalg.norm(to_c, axis=-1)
            mu_i = jnp.sum(nrm * to_c, axis=-1) / dist
            mu_e = jnp.sum(nrm * n, axis=-1)
            lit = ~segment_blocked(solids, ph + nrm * (1e-4 * a), bh)
            f_s = n_ph / (4 * np.pi * dist**2)
            ok = any_hit & lit & (mu_i > 0) & (mu_e > 0)
            i_rest = jnp.where(ok, self.albedo * REFLECTION_C * f_s * mu_i
                               / (mu_i + jnp.maximum(mu_e, 1e-6)), 0.0)
            i_rest = jnp.nan_to_num(i_rest)
            vel_s = jnp.where((which == 0)[:, None], v_corot(ph), v_disc(ph))
            g_s = jnp.where(ok, doppler(vel_s, n), 1.0)
            tau_s = jnp.where(ok, (dist - jnp.sum((ph - bh) * n, axis=-1)) / C_CM_S, 0.0)
            col_s = jnp.sum(jnp.where(front, nabs * ds, 0.0), axis=1)  # column in front
            mu_e = jnp.where(ok, jnp.clip(mu_e, 0.0, 1.0), 0.0)
            surf = (i_rest * g_s**3, g_s, tau_s, col_s, mu_e, which, ok)
            return wind_out, surf

        # ---- trace sub-pixel rays over the grid, in chunks
        xc, yc = scene.grid.centres()
        size = scene.grid.sizes()
        ss = self.supersample
        u = (np.arange(ss) + 0.5) / ss - 0.5
        du, dv = (m.ravel() for m in np.meshgrid(u, u))
        rx = (xc[:, None] + du[None, :] * size[:, None]).ravel()
        ry = (yc[:, None] + dv[None, :] * size[:, None]).ravel()
        area = np.repeat((size / ss / UAS_PER_RAD) ** 2, ss * ss)  # sr per sub-ray
        nray = rx.size
        outs = []
        for i0 in range(0, nray, self.chunk):
            sl = slice(i0, min(i0 + self.chunk, nray))
            m = sl.stop - sl.start
            pad = self.chunk - m
            xs = np.pad(rx[sl], (0, pad), mode="edge")
            ys = np.pad(ry[sl], (0, pad), mode="edge")
            wo, so = trace(jnp.asarray(xs), jnp.asarray(ys))
            outs.append(([np.asarray(v)[:m] for v in wo], [np.asarray(v)[:m] for v in so]))
        wind_r = [np.concatenate([o[0][k] for o in outs]) for k in range(5)]
        surf_r = [np.concatenate([o[1][k] for o in outs]) for k in range(7)]

        npix = scene.grid.n
        layers = []

        def pix_sum(v):
            return v.reshape(npix, ss * ss).sum(axis=1)

        # ---- wind layers, one per bin of absorbing column in front of the emission
        for kb in range(nh_bins.size):
            wsum = pix_sum(wind_r[0][:, kb] * area)
            if wsum.sum() <= 0:
                continue
            safe = np.where(wsum > 0, wsum, 1.0)
            gm = np.where(wsum > 0, pix_sum(wind_r[1][:, kb] * area) / safe, 1.0)
            g2 = pix_sum(wind_r[2][:, kb] * area) / safe
            maps = {"NORM": wsum,
                    "G": gm,
                    "SIGMA": np.where(wsum > 0, np.sqrt(np.clip(g2 - gm**2, 0, None)) / gm, 0.0),
                    "DELAY": pix_sum(wind_r[3][:, kb] * area) / safe,
                    "NH": self.nh_ism + pix_sum(wind_r[4][:, kb] * area) / safe}
            layers.append(SpecMapLayer(f"{self.name}_wind{kb}", scene.grid, [wind_t], maps,
                                       driver=self.driver))

        # ---- reflection layers: donor (which == 0) and disc (which == 1)
        i_obs, g_s, tau_s, col_s, mu_e, which, ok = surf_r
        for k, label in ((0, "star"), (1, "disc")):
            w = np.where(ok & (which == k), i_obs, 0.0) * area
            ws = pix_sum(w)
            if ws.sum() <= 0:
                continue
            sf = np.where(ws > 0, ws, 1.0)
            gm = np.where(ws > 0, pix_sum(w * g_s) / sf, 1.0)
            g2 = pix_sum(w * g_s**2) / sf
            mu = pix_sum(w * mu_e) / sf
            tidx = np.clip(np.digitize(mu, mu_edges) - 1, 0, len(refl_tmpls) - 1)
            maps = {"NORM": ws,
                    "G": np.where(ws > 0, gm, 1.0),
                    "SIGMA": np.where(ws > 0, np.sqrt(np.clip(g2 - gm**2, 0, None)) / gm, 0.0),
                    "DELAY": pix_sum(w * tau_s) / sf,
                    "NH": self.nh_ism + pix_sum(w * col_s) / sf,
                    "TIDX": tidx}
            layers.append(SpecMapLayer(f"{self.name}_{label}", scene.grid, refl_tmpls, maps,
                                       driver=self.driver))

        # ---- point sources at the compact object: visibility and column along the LOS
        p_bh = np.asarray(f["bh"])
        s_los = np.linspace(0.0, s_max, 20001)  # fine enough for the dense wind base
        los = p_bh[None, :] + s_los[:, None] * f["n"]
        blocked = bool(np.asarray(inside_donor(jnp.asarray(los))).any()
                       or np.asarray(inside_disc(jnp.asarray(los[1:]))).any())
        _, nabs_los, _ = wind(jnp.asarray(los))
        col_bh = float(np.trapezoid(np.asarray(nabs_los), s_los))
        g_bh = float(doppler(jnp.asarray(v_bh), n))
        point_grid = RegularGrid(nx=1, ny=1, scale=1e-3, x0=self.x, y0=self.y)
        vis = 0.0 if blocked else 1.0
        for label, spec, drv in (("corona", corona, self.driver),
                                 ("disc_bb", None if self.disc_spectrum is None
                                  else build_spectrum(self.disc_spectrum, edges), None)):
            if spec is None or vis == 0.0:
                continue
            layers.append(SpecMapLayer(
                f"{self.name}_{label}", point_grid, [spec],
                {"NORM": np.array([spec.photon_flux() * vis]), "G": np.array([g_bh]),
                 "NH": np.array([self.nh_ism + col_bh])}, driver=drv))
        self._last_frame = dict(f, cm_per_uas=cm_per_uas, col_bh=col_bh, blocked=blocked,
                                n_ph=n_ph, l_x=l_x)
        return layers
