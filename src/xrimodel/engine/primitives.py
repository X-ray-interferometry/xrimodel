"""Building blocks of a :class:`~xrimodel.engine.system.System3D`.

Three kinds, all positioned in the system frame (lengths in the system's
``length_unit``, velocities in km/s, angular velocities in rad/s, densities in H atoms
per cm^3, emissivities in photons/s/cm^3/sr):

* **Lamps**: point sources of X-rays. They appear in the image (absorbed by the
  column in front of them) and illuminate bodies and volumes.
* **Bodies**: opaque solids. The front-most body hides what is behind it. A body's
  visible surface reflects the illuminators' light (``reflection`` "neutral": pexmon or
  an approximation by emission angle; "grey": the incident spectrum, for ionised
  surfaces; None: no reflection) and may emit its own light (``emission``).
* **Volumes**: optically thin gas. A volume electron-scatters the illuminators' light
  (scaled by ``scatter``, plus emission ``lines`` with yields relative to the scattered
  photons), absorbs (``absorption``: "neutral", "xi" with a smooth ionisation threshold
  ``log_xi_crit``, or "none") and may emit (``emission``). Any volume can be clumped
  (``clumping``). An emitting volume with ``emission["illuminates"]`` lights the scene
  as ``n_points`` seeded point illuminators drawn from its emissivity.

Emission dicts::

    {"spectrum": spec, "intensity": I0, "spots": [...], "driver": name}   # bodies
    {"spectrum": spec, "j0": j0, "power": 1, "driver": name,
     "illuminates": False, "n_points": 8, "seed": 0}                       # volumes

A body emits ``I0`` photons/s/cm^2/sr (band-integrated) from its whole surface plus
``spots`` ({"lat", "lon", "radius" in deg, "intensity"}; Sphere only, in the frame of
its rotation axis). A volume emits ``j = j0 (n_H / n_ref)^power``, with ``n_ref`` the
primitive's reference density (``n0``; for a wind, its unfocused density at twice the
stellar radius) and ``power`` 1 by default (2 for a Loop).

Clumping dicts: ``{"f_vol": 0.1, "size": 0.05, "interclump": 0.0, "seed": 0}``: the
gas sits in cells of size ``size * r`` (r from the volume's centre) on a log-polar
lattice; a fraction ``f_vol`` of cells (chosen by a seeded hash) are clumps holding
the mass of the rest; ``interclump`` is the density left between clumps (relative to
the smooth density). The mean density is conserved.

``CustomBody`` and ``CustomVolume`` take your own JAX functions of position (cm, system
frame); they render but cannot be written to the ModelSpec.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np

from ..units import C_CM_S, G_CGS, MSUN_G

jax.config.update("jax_enable_x64", True)

RSUN_CM = 6.957e10
AU_CM = 1.495978707e13
PC_CM = 3.0856775814913673e18
YEAR_S = 3.15576e7
M_H = 1.6735575e-24
MU_H = 1.4  # mass per hydrogen atom in units of m_H (solar composition)
KM = 1e5

LENGTH_UNITS = {"cm": 1.0, "km": 1e5, "Rsun": RSUN_CM, "AU": AU_CM, "pc": PC_CM}


def unit_cm(unit, mass_msun=1.0):
    """cm per length unit: a number, a name in LENGTH_UNITS, or "rg" (G M / c^2)."""
    if isinstance(unit, (int, float)):
        return float(unit)
    if unit == "rg":
        return G_CGS * mass_msun * MSUN_G / C_CM_S**2
    return LENGTH_UNITS[unit]


# --------------------------------------------------------------------------- helpers


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def _basis(axis):
    """Rows (e1, e2, axis): an orthonormal frame with the given third axis (e1 along the
    frame's x axis, or y if the axis is close to x)."""
    z = _unit(axis)
    t = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = _unit(t - (t @ z) * z)
    return np.stack([e1, np.cross(z, e1), z])


def _rotation_velocity(rotation, L):
    """v(p) [cm/s] of a rigid rotation {"omega", "axis", "centre"} (or None)."""
    if not rotation:
        return lambda p: jnp.zeros_like(p)
    om = float(rotation["omega"]) * _unit(rotation.get("axis", (0.0, 0.0, 1.0)))
    c = np.asarray(rotation.get("centre", (0.0, 0.0, 0.0)), float) * L
    om, c = jnp.asarray(om), jnp.asarray(c)
    return lambda p: jnp.cross(jnp.broadcast_to(om, p.shape), p - c)


def _hash01(i, j, k, seed):
    """Deterministic uniform [0, 1) from integer lattice indices."""
    x = (i.astype(jnp.uint32) * jnp.uint32(73856093)) ^ (j.astype(jnp.uint32)
                                                         * jnp.uint32(19349663))
    x = x ^ (k.astype(jnp.uint32) * jnp.uint32(83492791)) ^ jnp.uint32(
        (seed * 2654435761) & 0xFFFFFFFF)
    x = x ^ (x >> 16)
    x = x * jnp.uint32(0x7FEB352D)
    x = x ^ (x >> 15)
    x = x * jnp.uint32(0x846CA68B)
    x = x ^ (x >> 16)
    return x.astype(jnp.float64) / 4294967296.0


def clump_factor(clumping, centre, r_min):
    """Density multiplier of a clumped volume (see the module docstring), or None."""
    if not clumping:
        return None
    f_v = float(clumping.get("f_vol", 0.1))
    s = float(clumping.get("size", 0.05))
    ic = float(clumping.get("interclump", 0.0))
    seed = int(clumping.get("seed", 0))
    c = jnp.asarray(centre)
    hi = (1.0 - ic * (1.0 - f_v)) / f_v

    def fac(p):
        w = p - c
        r = jnp.maximum(jnp.linalg.norm(w, axis=-1), r_min)
        th = jnp.arccos(jnp.clip(w[..., 2] / r, -1.0, 1.0))
        ph = jnp.arctan2(w[..., 1], w[..., 0]) + np.pi
        i = jnp.floor(jnp.log(r / r_min) / s).astype(jnp.int32)
        j = jnp.floor(th / s).astype(jnp.int32)
        dphi = s / jnp.maximum(jnp.sin((j + 0.5) * s), s)
        k = jnp.floor(ph / dphi).astype(jnp.int32)
        return jnp.where(_hash01(i, j, k, seed) < f_v, hi, ic)

    return fac


def _with_clumps(density, fac):
    return density if fac is None else (lambda p: density(p) * fac(p))


def _volume_emission(em, n_ref, density, power=1.0):
    """{"j": j(p) -> (continuous, moving-blob) emissivities, ...} from an emission dict."""
    if not em:
        return None
    j0, pw = float(em["j0"]), float(em.get("power", power))

    def j(p):
        nh = density(p)
        return j0 * (nh / n_ref) ** pw, jnp.zeros_like(nh)

    return dict(j=j, spectrum=em["spectrum"], driver=em.get("driver"),
                illuminates=bool(em.get("illuminates", False)),
                n_points=int(em.get("n_points", 8)), seed=int(em.get("seed", 0)))


# --------------------------------------------------------------------------- base classes


@dataclass(kw_only=True)
class Lamp:
    """A point X-ray source. ``spectrum`` is its unabsorbed spectrum as seen from the
    system's distance (it sets the photon luminosity that illuminates the scene)."""

    name: str
    spectrum: dict
    position: tuple = (0.0, 0.0, 0.0)
    velocity: tuple = (0.0, 0.0, 0.0)  # km/s
    driver: str | None = None
    illuminates: bool = True
    visible: bool = True

    kind: ClassVar[str] = "lamp"
    role: ClassVar[str] = "lamp"

    def build(self, L, mass_msun):
        return SimpleNamespace(name=self.name, spectrum=self.spectrum, driver=self.driver,
                               pos=np.asarray(self.position, float) * L,
                               vel=np.asarray(self.velocity, float) * KM,
                               illuminates=self.illuminates, visible=self.visible)


@dataclass(kw_only=True)
class _Body:
    name: str
    albedo: float = 0.1
    reflection: str | None = "neutral"  # "neutral", "grey" or None
    emission: dict | None = None
    velocity: tuple = (0.0, 0.0, 0.0)  # km/s, bulk
    rotation: dict | None = None  # {"omega": rad/s, "axis": (..), "centre": (..)}; centre
    # defaults to the body's centre
    n_samples: int = 96

    role: ClassVar[str] = "body"

    def _velocity(self, L):
        """Bulk velocity plus the rigid ``rotation`` (about the body's own centre unless
        the rotation dict gives one)."""
        v0 = jnp.asarray(np.asarray(self.velocity, float) * KM)
        rotation = self.rotation
        if rotation and "centre" not in rotation:
            rotation = dict(rotation, centre=getattr(self, "centre", (0.0, 0.0, 0.0)))
        rot = _rotation_velocity(rotation, L)
        return lambda p: v0 + rot(p)

    def _surface_emission(self, intensity_fn=None):
        em = self.emission
        if not em:
            return None
        i0 = float(em.get("intensity", 0.0))
        fn = intensity_fn or (lambda p: jnp.full(p.shape[:-1], i0))
        return dict(intensity=fn, spectrum=em["spectrum"], driver=em.get("driver"))

    def _built(self, **kw):
        return SimpleNamespace(name=self.name, albedo=self.albedo, reflection=self.reflection,
                               **kw)


@dataclass(kw_only=True)
class _Volume:
    name: str
    absorption: str = "neutral"  # "neutral", "xi" or "none"
    log_xi_crit: float = 2.0
    scatter: float = 1.0  # electron-scattering strength (1 = Thomson)
    lines: list = field(default_factory=list)  # [{"e0", "sigma", "yield"}]
    clumping: dict | None = None
    emission: dict | None = None
    velocity: tuple = (0.0, 0.0, 0.0)  # km/s, bulk
    n_fine: int = 96

    role: ClassVar[str] = "volume"
    sampling: ClassVar[str] = "cluster"
    emission_power: ClassVar[float] = 1.0

    def _built(self, density, velocity, n_ref, bound, refine, clump_centre, r_clump):
        dens = _with_clumps(density, clump_factor(self.clumping, clump_centre, r_clump))
        return SimpleNamespace(
            name=self.name, density=dens, velocity=velocity, absorption=self.absorption,
            log_xi_crit=self.log_xi_crit, scatter=self.scatter, lines=list(self.lines),
            emission=_volume_emission(self.emission, n_ref, dens, self.emission_power),
            bound=bound,
            refine=refine, n_fine=self.n_fine, sampling=self.sampling)


# --------------------------------------------------------------------------- bodies


@dataclass(kw_only=True)
class Sphere(_Body):
    """A sphere, optionally rotating, with emitting spots (stellar active regions)."""

    centre: tuple = (0.0, 0.0, 0.0)
    radius: float = 1.0

    kind: ClassVar[str] = "sphere"

    def build(self, L, mass_msun):
        c, R = jnp.asarray(np.asarray(self.centre, float) * L), self.radius * L

        def field(p):
            return jnp.linalg.norm(p - c, axis=-1) - R

        def spot_intensity(p):
            em = self.emission
            out = jnp.full(p.shape[:-1], float(em.get("intensity", 0.0)))
            ax = (self.rotation or {}).get("axis", (0.0, 0.0, 1.0))
            B = jnp.asarray(_basis(ax))
            u = (p - c) @ B.T
            u = u / jnp.linalg.norm(u, axis=-1, keepdims=True)
            for s in em.get("spots", []):
                la, lo = np.deg2rad(s["lat"]), np.deg2rad(s["lon"])
                d = jnp.asarray([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])
                on = (u @ d) > np.cos(np.deg2rad(s["radius"]))
                out = out + jnp.where(on, float(s["intensity"]), 0.0)
            return out

        return self._built(
            inside=lambda p: field(p) < 0, field=field, velocity=self._velocity(L),
            bound=dict(kind="sphere", centre=c, radius=R * 1.001, n=self.n_samples),
            emission=self._surface_emission(spot_intensity if self.emission else None))


@dataclass(kw_only=True)
class RocheLobe(_Body):
    """A star filling the Roche equipotential with volume-equivalent radius ``r_eq``.

    The orbit lies in the frame's x-y plane; ``q`` = companion mass / this star's mass;
    ``cm`` defaults to the centre of mass. Give ``rotation`` (omega about z through the
    centre of mass) for corotation.
    """

    centre: tuple = (0.0, 0.0, 0.0)
    companion: tuple = (1.0, 0.0, 0.0)
    q: float = 1.0
    r_eq: float = 0.3
    cm: tuple | None = None

    kind: ClassVar[str] = "roche"

    def potential(self, L):
        """(phi(u), level, t_l1, axis, d, a): phi of positions in units of a about the CM."""
        c, comp = np.asarray(self.centre, float) * L, np.asarray(self.companion, float) * L
        a = float(np.linalg.norm(comp - c))
        q = self.q
        cm = c + (comp - c) * q / (1 + q) if self.cm is None else np.asarray(self.cm, float) * L
        d, b = (c - cm) / a, (comp - cm) / a

        def phi(u):
            r1 = jnp.linalg.norm(u - d, axis=-1)
            r2 = jnp.linalg.norm(u - b, axis=-1)
            return -1.0 / r1 - q / r2 - 0.5 * (1 + q) * (u[..., 0] ** 2 + u[..., 1] ** 2)

        axis = (b - d) / np.linalg.norm(b - d)
        dphi = jax.grad(lambda t: phi(d + t * axis))
        lo, hi = 1e-3, 1.0 - 1e-3
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if float(dphi(mid)) > 0 else (lo, mid)
        t_l1 = 0.5 * (lo + hi)
        phi_l1 = float(phi(d + t_l1 * axis))

        r_target = self.r_eq * L / a
        g = np.linspace(-t_l1, t_l1, 72)
        X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
        pts = d + np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
        ph = np.asarray(phi(jnp.asarray(pts)))
        # the lobe: towards the companion up to L1, and within the L1 distance of the
        # star's centre (far from the system the centrifugal term makes phi small again)
        lobe = ((pts - d) @ axis < t_l1) & (np.linalg.norm(pts - d, axis=-1) < t_l1)
        cell = (g[1] - g[0]) ** 3
        if (4 / 3) * np.pi * r_target**3 >= np.sum((ph < phi_l1) & lobe) * cell:
            raise ValueError(f"{self.name}: r_eq overfills the Roche lobe")
        lo, hi = 1.01 * float(np.min(ph)), phi_l1
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            r_e = (3 * np.sum((ph < mid) & lobe) * cell / (4 * np.pi)) ** (1 / 3)
            lo, hi = (mid, hi) if r_e < r_target else (lo, mid)
        return phi, 0.5 * (lo + hi), t_l1, axis, d, a, cm

    def build(self, L, mass_msun):
        phi, level, t_l1, axis, d, a, cm = self.potential(L)
        axis_j, cm_j = jnp.asarray(axis), jnp.asarray(cm)

        def inside(p):
            u = (p - cm_j) / a
            du = u - d
            return ((phi(u) < level) & (jnp.sum(du * axis_j, axis=-1) < t_l1)
                    & (jnp.sum(du * du, axis=-1) < t_l1**2))

        return self._built(
            inside=inside, field=lambda p: phi((p - cm_j) / a), velocity=self._velocity(L),
            bound=dict(kind="sphere", centre=jnp.asarray(cm + d * a), radius=t_l1 * a,
                       n=self.n_samples),
            emission=self._surface_emission(), t_l1=t_l1, a=a)


@dataclass(kw_only=True)
class FlaredDisc(_Body):
    """A flared disc, H/R ~ r^(flare - 1) (H/R = ``h_over_r`` at ``r_out``), tapered to
    zero height at ``r_in`` so the truncation leaves no wall facing the centre.
    Keplerian about ``mass_msun`` (0: no rotation), sense +1 = counterclockwise about
    ``axis``; plus the bulk ``velocity``."""

    centre: tuple = (0.0, 0.0, 0.0)
    axis: tuple = (0.0, 0.0, 1.0)
    r_in: float = 0.1
    r_out: float = 1.0
    h_over_r: float = 0.1
    flare: float = 1.125
    mass_msun: float = 0.0
    sense: int = 1
    n_samples: int = 1024

    kind: ClassVar[str] = "flared_disc"

    def build(self, L, mass_msun):
        c = jnp.asarray(np.asarray(self.centre, float) * L)
        B = jnp.asarray(_basis(self.axis))
        r_in, r_out, hr, fl = self.r_in * L, self.r_out * L, self.h_over_r, self.flare

        def height(r):
            rr = jnp.maximum(r, 1.0)
            return hr * r_out * (rr / r_out) ** fl * jnp.clip(1.0 - r_in / rr, 0.0, 1.0)

        def local(p):
            return (p - c) @ B.T

        def inside(p):
            w = local(p)
            r = jnp.hypot(w[..., 0], w[..., 1])
            return (r >= r_in) & (r <= r_out) & (jnp.abs(w[..., 2]) <= height(r))

        def field(p):
            w = local(p)
            r = jnp.sqrt(w[..., 0] ** 2 + w[..., 1] ** 2 + 1.0)
            return jnp.maximum(jnp.maximum(jnp.abs(w[..., 2]) - height(r), r - r_out), r_in - r)

        v0 = self._velocity(L)
        gm = G_CGS * self.mass_msun * MSUN_G

        def velocity(p):
            w = local(p)
            r = jnp.maximum(jnp.hypot(w[..., 0], w[..., 1]), 1.0)
            vk = self.sense * jnp.sqrt(gm / r)
            vl = jnp.stack([-w[..., 1] / r * vk, w[..., 0] / r * vk, jnp.zeros_like(r)], -1)
            return v0(p) + vl @ B

        return self._built(
            inside=inside, field=field, velocity=velocity,
            bound=dict(kind="slab", centre=c, radius=r_out * np.hypot(1.0, hr) * 1.01,
                       half=hr * r_out, axis=B[2], n=self.n_samples),
            emission=self._surface_emission())


@dataclass(kw_only=True)
class Torus(_Body):
    """A torus of major radius ``r_major`` and tube radius ``r_minor`` about ``axis``
    (an AGN's obscuring torus), Keplerian about ``mass_msun`` if given."""

    centre: tuple = (0.0, 0.0, 0.0)
    axis: tuple = (0.0, 0.0, 1.0)
    r_major: float = 1.0
    r_minor: float = 0.3
    mass_msun: float = 0.0
    sense: int = 1
    n_samples: int = 256

    kind: ClassVar[str] = "torus"

    def build(self, L, mass_msun):
        c = jnp.asarray(np.asarray(self.centre, float) * L)
        B = jnp.asarray(_basis(self.axis))
        R, a = self.r_major * L, self.r_minor * L

        def field(p):
            w = (p - c) @ B.T
            return jnp.sqrt((jnp.hypot(w[..., 0], w[..., 1]) - R) ** 2 + w[..., 2] ** 2) - a

        v0 = self._velocity(L)
        gm = G_CGS * self.mass_msun * MSUN_G

        def velocity(p):
            w = (p - c) @ B.T
            r = jnp.maximum(jnp.hypot(w[..., 0], w[..., 1]), 1.0)
            vk = self.sense * jnp.sqrt(gm / r)
            vl = jnp.stack([-w[..., 1] / r * vk, w[..., 0] / r * vk, jnp.zeros_like(r)], -1)
            return v0(p) + vl @ B

        return self._built(
            inside=lambda p: field(p) < 0, field=field, velocity=velocity,
            bound=dict(kind="sphere", centre=c, radius=(R + a) * 1.001, n=self.n_samples),
            emission=self._surface_emission())


# --------------------------------------------------------------------------- volumes


@dataclass(kw_only=True)
class BetaWind(_Volume):
    """A star's beta-law wind, v = v_inf (1 - b R/r)^beta with v(R) = ``v_base``.

    ``focus`` = {"towards": point, "amp": A, "width": deg, "v_factor": f}: with
    F = exp(-theta^2 / 2 width^2) (theta from the direction towards the point), the mass
    flux is raised by 1 + A F and v_inf is lowered to v_inf (1 - (1 - f) F), as in the
    focused winds of high-mass X-ray binaries; the density, mass flux / v, rises by both.
    ``mdot`` is the isotropic mass-loss rate without focusing. ``refine_body`` names the star's body so ray samples cluster at its surface.
    """

    star: tuple = (0.0, 0.0, 0.0)
    r_star: float = 1.0
    mdot: float = 1e-6  # Msun / yr
    v_inf: float = 2000.0  # km/s
    beta: float = 1.0
    v_base: float = 20.0  # km/s
    extent: float = 10.0  # radius of the wind's bounding sphere
    focus: dict | None = None
    refine_body: str | None = None
    absorption: str = "xi"

    kind: ClassVar[str] = "beta_wind"

    def build(self, L, mass_msun):
        c = jnp.asarray(np.asarray(self.star, float) * L)
        R = self.r_star * L
        mdot = self.mdot * MSUN_G / YEAR_S
        v_inf = self.v_inf * KM
        b = 1.0 - (self.v_base / self.v_inf) ** (1.0 / self.beta)
        fo = self.focus
        if fo:
            to = _unit(np.asarray(fo["towards"], float) * L - np.asarray(self.star, float) * L)
            to = jnp.asarray(to)
            amp, wid, vf = float(fo.get("amp", 0.0)), np.deg2rad(fo.get("width", 30.0)), \
                float(fo.get("v_factor", 1.0))

        def profile(p, focused=True):
            w = p - c
            r = jnp.linalg.norm(w, axis=-1)
            rs = jnp.maximum(r, R)
            if fo and focused:
                th = jnp.arccos(jnp.clip(jnp.sum(w * to, -1) / jnp.maximum(r, 1.0), -1, 1))
                f = jnp.exp(-0.5 * (th / wid) ** 2)
                vinf, boost = v_inf * (1 - (1 - vf) * f), 1 + amp * f
            else:
                vinf, boost = v_inf, 1.0
            v = vinf * (1 - b * R / rs) ** self.beta
            nh = boost * mdot / (4 * np.pi * rs**2 * v * MU_H * M_H)
            return w, r, jnp.where(r < R, 0.0, nh), v

        v0 = jnp.asarray(np.asarray(self.velocity, float) * KM)

        def velocity(p):
            w, r, _, v = profile(p)
            return v0 + (w / jnp.maximum(r, 1.0)[..., None]) * v[..., None]

        density = lambda p: profile(p)[2]
        n_ref = float(profile(c[None] + jnp.asarray([[2 * R, 0.0, 0.0]]), focused=False)[2][0])
        ext = self.extent * L
        return self._built(
            density, velocity, n_ref, bound=dict(centre=c, radius=ext),
            refine=dict(centre=c, radius=min(2.5 * R, ext), body=self.refine_body),
            clump_centre=c, r_clump=R)


@dataclass(kw_only=True)
class Shell(_Volume):
    """Gas between ``r_in`` and ``r_out`` with n_H = n0 (r / r_in)^-index, moving
    radially at ``v_r`` (km/s) plus the bulk ``velocity`` (BLR/NLR clouds when clumped)."""

    centre: tuple = (0.0, 0.0, 0.0)
    r_in: float = 0.0
    r_out: float = 1.0
    n0: float = 1e10
    index: float = 0.0
    v_r: float = 0.0

    kind: ClassVar[str] = "shell"

    def build(self, L, mass_msun):
        c = jnp.asarray(np.asarray(self.centre, float) * L)
        r_in, r_out = self.r_in * L, self.r_out * L
        r_ref = max(r_in, 1e-3 * r_out)
        v0 = jnp.asarray(np.asarray(self.velocity, float) * KM)

        def density(p):
            r = jnp.linalg.norm(p - c, axis=-1)
            n = self.n0 * (jnp.maximum(r, r_ref) / r_ref) ** (-self.index)
            return jnp.where((r >= r_in) & (r <= r_out), n, 0.0)

        def velocity(p):
            w = p - c
            r = jnp.maximum(jnp.linalg.norm(w, axis=-1), 1.0)
            return v0 + w / r[..., None] * self.v_r * KM

        return self._built(density, velocity, self.n0, bound=dict(centre=c, radius=r_out),
                           refine=dict(centre=c, radius=r_out, body=None),
                           clump_centre=c, r_clump=r_ref)


@dataclass(kw_only=True)
class GasTorus(_Volume):
    """A torus of gas (an AGN's obscuring torus as an absorber rather than an opaque
    body): major radius ``r_major``, tube radius ``r_minor`` about ``axis``, n_H = n0
    uniform in the tube (``profile`` "uniform") or n0 exp(-d^2 / 2 (r_minor/2)^2) for
    distance d from the tube's centre line, cut at r_minor ("gaussian"). Keplerian about
    ``mass_msun`` if given. It absorbs and scatters the light that crosses it
    (transmission through a Compton-thick torus, with ``NH`` and electron scattering
    ``NE``) and scatters the illuminators' light once; add Fe K-alpha as a line, e.g.
    ``lines=[{"e0": 6.4, "sigma": 0.01, "yield": y}]``. Its Compton *reflection* (many
    scatterings in thick gas) is not modelled: use the opaque ``Torus`` body for that.
    Clumping makes it a clumpy torus."""

    centre: tuple = (0.0, 0.0, 0.0)
    axis: tuple = (0.0, 0.0, 1.0)
    r_major: float = 1.0
    r_minor: float = 0.3
    n0: float = 1e5
    profile: str = "uniform"
    mass_msun: float = 0.0
    sense: int = 1
    n_fine: int = 384

    kind: ClassVar[str] = "gas_torus"
    sampling: ClassVar[str] = "uniform"

    def build(self, L, mass_msun):
        c = jnp.asarray(np.asarray(self.centre, float) * L)
        B = jnp.asarray(_basis(self.axis))
        R, a = self.r_major * L, self.r_minor * L
        if self.profile not in ("uniform", "gaussian"):
            raise ValueError(f"{self.name}: profile must be 'uniform' or 'gaussian'")
        gauss = self.profile == "gaussian"
        v0 = jnp.asarray(np.asarray(self.velocity, float) * KM)
        gm = G_CGS * self.mass_msun * MSUN_G

        def density(p):
            w = (p - c) @ B.T
            d2 = (jnp.hypot(w[..., 0], w[..., 1]) - R) ** 2 + w[..., 2] ** 2
            n = self.n0 * jnp.exp(-2.0 * d2 / a**2) if gauss else jnp.full(d2.shape, self.n0)
            return jnp.where(d2 < a**2, n, 0.0)

        def velocity(p):
            w = (p - c) @ B.T
            r = jnp.maximum(jnp.hypot(w[..., 0], w[..., 1]), 1.0)
            vk = self.sense * jnp.sqrt(gm / r)
            vl = jnp.stack([-w[..., 1] / r * vk, w[..., 0] / r * vk, jnp.zeros_like(r)], -1)
            return v0 + vl @ B

        return self._built(density, velocity, self.n0, bound=dict(centre=c, radius=R + a),
                           refine=None, clump_centre=c, r_clump=max(R - a, 1e-3 * R))


@dataclass(kw_only=True)
class BiCone(_Volume):
    """A (bi)cone of half-opening angle ``half_angle`` about ``axis`` from ``apex``
    (ionisation cones, polar scattering), n_H = n0 (r / r_in)^-index, radial ``v_r``."""

    apex: tuple = (0.0, 0.0, 0.0)
    axis: tuple = (0.0, 0.0, 1.0)
    half_angle: float = 30.0  # deg
    r_in: float = 0.0
    r_out: float = 1.0
    n0: float = 1e4
    index: float = 0.0
    v_r: float = 0.0
    two_sided: bool = True
    absorption: str = "none"

    kind: ClassVar[str] = "bicone"

    def build(self, L, mass_msun):
        c = jnp.asarray(np.asarray(self.apex, float) * L)
        ax = jnp.asarray(_unit(self.axis))
        r_in, r_out = self.r_in * L, self.r_out * L
        r_ref = max(r_in, 1e-3 * r_out)
        cos_h = np.cos(np.deg2rad(self.half_angle))
        v0 = jnp.asarray(np.asarray(self.velocity, float) * KM)

        def density(p):
            w = p - c
            r = jnp.linalg.norm(w, axis=-1)
            mu = jnp.sum(w * ax, -1) / jnp.maximum(r, 1.0)
            inside = (jnp.abs(mu) if self.two_sided else mu) >= cos_h
            n = self.n0 * (jnp.maximum(r, r_ref) / r_ref) ** (-self.index)
            return jnp.where(inside & (r >= r_in) & (r <= r_out), n, 0.0)

        def velocity(p):
            w = p - c
            r = jnp.maximum(jnp.linalg.norm(w, axis=-1), 1.0)
            return v0 + w / r[..., None] * self.v_r * KM

        return self._built(density, velocity, self.n0, bound=dict(centre=c, radius=r_out),
                           refine=dict(centre=c, radius=r_out, body=None),
                           clump_centre=c, r_clump=r_ref)


@dataclass(kw_only=True)
class Jet(_Volume):
    """A conical jet (and counter-jet if ``two_sided``) from ``base`` along ``axis``,
    from ``r0`` to ``r1``, density and emissivity profile ~ (r / r0)^-index, bulk
    Lorentz factor ``lorentz`` along the axis. ``knots`` = [{"r", "size", "j"}] adds
    Gaussian blobs on the axis moving with the flow (Doppler weight g^3 rather than the
    steady jet's g^2). Needs an ``emission`` dict to shine."""

    base: tuple = (0.0, 0.0, 0.0)
    axis: tuple = (0.0, 0.0, 1.0)
    half_angle: float = 5.0  # deg
    r0: float = 1.0
    r1: float = 100.0
    n0: float = 1.0
    index: float = 2.0
    lorentz: float = 2.0
    two_sided: bool = True
    knots: list = field(default_factory=list)
    absorption: str = "none"
    scatter: float = 0.0

    kind: ClassVar[str] = "jet"
    sampling: ClassVar[str] = "uniform"

    def build(self, L, mass_msun):
        c = jnp.asarray(np.asarray(self.base, float) * L)
        ax = jnp.asarray(_unit(self.axis))
        r0, r1 = self.r0 * L, self.r1 * L
        cos_h = np.cos(np.deg2rad(self.half_angle))
        beta = np.sqrt(1 - 1 / self.lorentz**2)

        def side(p):
            w = p - c
            r = jnp.linalg.norm(w, axis=-1)
            mu = jnp.sum(w * ax, -1) / jnp.maximum(r, 1.0)
            return w, r, mu

        def density(p):
            _, r, mu = side(p)
            inside = (jnp.abs(mu) if self.two_sided else mu) >= cos_h
            n = self.n0 * (jnp.maximum(r, r0) / r0) ** (-self.index)
            return jnp.where(inside & (r >= r0) & (r <= r1), n, 0.0)

        v0 = jnp.asarray(np.asarray(self.velocity, float) * KM)

        def velocity(p):
            _, _, mu = side(p)
            return v0 + jnp.sign(mu)[..., None] * beta * C_CM_S * ax

        out = self._built(density, velocity, self.n0, bound=dict(centre=c, radius=r1),
                          refine=None, clump_centre=c, r_clump=r0)
        if out.emission and self.knots:
            j_cont = out.emission["j"]
            kn = [(float(k["r"]) * L, float(k["size"]) * L, float(k["j"])) for k in self.knots]
            sides = (1.0, -1.0) if self.two_sided else (1.0,)

            def j(p):
                jc, _ = j_cont(p)
                jk = jnp.zeros_like(jc)
                for rk, sk, amp in kn:
                    for sg in sides:
                        d2 = jnp.sum((p - (c + sg * rk * ax)) ** 2, -1)
                        jk = jk + amp * jnp.exp(-0.5 * d2 / sk**2)
                return jc, jk

            out.emission["j"] = j
        return out


@dataclass(kw_only=True)
class Loop(_Volume):
    """A coronal loop: a tube of radius ``thickness`` along the arc between two
    footpoints (lat, lon in deg, in the frame of the star's rotation axis) in the plane of
    their great circle, rising to ``height`` above the surface at the middle
    (r = r_star + height sin(pi t), t = 0..1 along the arc; the tube's radius is measured
    in that plane's polar coordinates, so it is exact for a flat arc), with flat ends,
    uniform density ``n0``, corotating. X-ray emissivity goes as n^2 by default
    (``emission["power"]``)."""

    star: tuple = (0.0, 0.0, 0.0)
    r_star: float = 1.0
    foot1: tuple = (0.0, 0.0)
    foot2: tuple = (0.0, 20.0)
    height: float = 0.1
    thickness: float = 0.01
    n0: float = 1e10
    rotation: dict | None = None
    absorption: str = "none"
    scatter: float = 0.0
    n_fine: int = 512

    kind: ClassVar[str] = "loop"
    sampling: ClassVar[str] = "uniform"
    emission_power: ClassVar[float] = 2.0

    def _plane(self):
        B = _basis((self.rotation or {}).get("axis", (0.0, 0.0, 1.0)))

        def u(ll):
            la, lo = np.deg2rad(ll)
            return np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]) @ B

        u1, u2 = u(self.foot1), u(self.foot2)
        om = float(np.arccos(np.clip(u1 @ u2, -1, 1)))
        if not 1e-6 < om < np.pi - 1e-6:
            raise ValueError(f"{self.name}: footpoints must be distinct and not antipodal")
        nrm = _unit(np.cross(u1, u2))
        return u1, np.cross(nrm, u1), nrm, om

    def build(self, L, mass_msun):
        e1, e2, nrm, om = self._plane()
        c = np.asarray(self.star, float) * L
        R, H, th = self.r_star * L, self.height * L, self.thickness * L
        cj, e1j, e2j, nj = (jnp.asarray(v) for v in (c, e1, e2, nrm))
        t = np.linspace(0, 1, 65)
        arc = c + (np.cos(t * om)[:, None] * e1 + np.sin(t * om)[:, None] * e2) \
            * (R + H * np.sin(np.pi * t))[:, None]
        mid = arc.mean(axis=0)
        rad = float(np.max(np.linalg.norm(arc - mid, axis=1))) + 2 * th
        rot = _rotation_velocity(
            None if not self.rotation else dict(self.rotation, centre=self.star), L)
        v0 = jnp.asarray(np.asarray(self.velocity, float) * KM)

        def density(p):
            w = p - cj
            z = jnp.sum(w * nj, -1)
            a1, a2 = jnp.sum(w * e1j, -1), jnp.sum(w * e2j, -1)
            tt = jnp.arctan2(a2, a1) / om
            rho = jnp.hypot(a1, a2)
            d2 = (rho - (R + H * jnp.sin(np.pi * tt))) ** 2 + z**2
            return jnp.where((tt >= 0) & (tt <= 1) & (d2 < th**2), self.n0, 0.0)

        return self._built(density, lambda p: v0 + rot(p), self.n0,
                           bound=dict(centre=jnp.asarray(mid), radius=rad), refine=None,
                           clump_centre=cj, r_clump=R)


# --------------------------------------------------------------------------- escape hatch


@dataclass(kw_only=True)
class CustomBody(_Body):
    """A body from your own JAX functions of position p (cm, system frame): ``inside``
    (bool), ``field`` (positive outside; its gradient is the normal) and optionally
    ``velocity_fn`` (cm/s). Not serialisable."""

    inside: Callable
    field: Callable
    bound_centre: tuple = (0.0, 0.0, 0.0)  # cm
    bound_radius: float = 1.0  # cm
    velocity_fn: Callable | None = None

    kind: ClassVar[str] = "custom_body"
    serialisable: ClassVar[bool] = False

    def build(self, L, mass_msun):
        vel = self.velocity_fn or self._velocity(L)
        return self._built(
            inside=self.inside, field=self.field, velocity=vel,
            bound=dict(kind="sphere", centre=jnp.asarray(self.bound_centre, dtype=float),
                       radius=float(self.bound_radius), n=self.n_samples),
            emission=self._surface_emission())


@dataclass(kw_only=True)
class CustomVolume(_Volume):
    """A volume from your own JAX functions of position p (cm, system frame):
    ``density`` (n_H, cm^-3) and optionally ``velocity_fn`` (cm/s). Not serialisable."""

    density: Callable
    bound_centre: tuple = (0.0, 0.0, 0.0)  # cm
    bound_radius: float = 1.0  # cm
    n_ref: float = 1.0
    velocity_fn: Callable | None = None

    kind: ClassVar[str] = "custom_volume"
    serialisable: ClassVar[bool] = False

    def build(self, L, mass_msun):
        v0 = jnp.asarray(np.asarray(self.velocity, float) * KM)
        vel = self.velocity_fn or (lambda p: jnp.broadcast_to(v0, p.shape))
        c = jnp.asarray(self.bound_centre, dtype=float)
        return self._built(self.density, vel, self.n_ref,
                           bound=dict(centre=c, radius=float(self.bound_radius)),
                           refine=dict(centre=c, radius=float(self.bound_radius), body=None),
                           clump_centre=c, r_clump=1e-3 * float(self.bound_radius))


PRIMITIVES = {cls.kind: cls for cls in (Lamp, Sphere, RocheLobe, FlaredDisc, Torus, BetaWind,
                                        Shell, GasTorus, BiCone, Jet, Loop, CustomBody, CustomVolume)}
