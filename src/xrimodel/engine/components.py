"""Model components. Each one renders itself into a format layer on the scene's grid.

A component's fields are plain numbers, strings and spec dicts, so ``asdict`` gives
its part of the ModelSpec. Positions are tangent-plane offsets in uas (x East,
y North); physical sizes use gravitational radii r_g = GM/c^2 of ``mass_msun``.

Pixel quantities are integrated by supersampling each pixel (``supersample`` points per
side), evaluated with JAX so the same kernels run on CPU or GPU and can later be
differentiated.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from ..format.layers import SeparableLayer, SpecMapLayer
from ..units import C_CM_S, G_CGS, MSUN_G, length_to_uas
from .spec import build_spectrum

jax.config.update("jax_enable_x64", True)


def _subpixel_points(xc, yc, size, ss):
    """(npix, ss*ss) sample positions spread evenly inside each square pixel."""
    u = (jnp.arange(ss) + 0.5) / ss - 0.5
    du, dv = jnp.meshgrid(u, u)
    du, dv = du.ravel(), dv.ravel()
    return (xc[:, None] + du[None, :] * size[:, None],
            yc[:, None] + dv[None, :] * size[:, None])


@dataclass
class PointSource:
    name: str
    spectrum: dict
    x: float = 0.0
    y: float = 0.0
    driver: str | None = None

    kind = "point"

    def render(self, scene):
        spec = build_spectrum(self.spectrum, scene.energy_edges)
        return SeparableLayer(self.name, spec, x=self.x, y=self.y, driver=self.driver)


@dataclass
class GaussianBlob:
    """Circular Gaussian surface brightness (sigma in uas) with one spectrum."""

    name: str
    spectrum: dict
    sigma: float
    x: float = 0.0
    y: float = 0.0
    driver: str | None = None

    kind = "gaussian"

    def render(self, scene):
        g = scene.grid
        xc, yc = (jnp.asarray(a) for a in g.centres())
        h = jnp.asarray(g.sizes()) / 2
        erf = jax.scipy.special.erf
        s2 = jnp.sqrt(2.0) * self.sigma
        wx = erf((xc + h - self.x) / s2) - erf((xc - h - self.x) / s2)
        wy = erf((yc + h - self.y) / s2) - erf((yc - h - self.y) / s2)
        img = np.asarray(0.25 * wx * wy)
        spec = build_spectrum(self.spectrum, scene.energy_edges)
        return SeparableLayer(self.name, spec, g, img, driver=self.driver)


@jax.jit
def _disc_kernel(px, py, r_in, r_out, cos_i, sin_i, pa, q, h_lamp, grav, sense):
    """Per-sample disc quantities: weight, g, delay (in r_g/c). Positions in r_g."""
    # rotate the sky offsets so the disc's projected major axis lies along x'
    c, s = jnp.cos(pa), jnp.sin(pa)
    xp = c * px + s * py
    yp = -s * px + c * py
    # de-project onto the disc plane: (r cos phi, r sin phi) = (x', y' / cos i)
    dx, dy = xp, yp / cos_i
    r = jnp.hypot(dx, dy)
    inside = (r >= r_in) & (r <= r_out)
    rs = jnp.where(inside, r, r_in)  # keep the maths finite outside the disc
    phi = jnp.arctan2(dy, dx)
    # Keplerian (Newtonian) speed; line-of-sight unit vector n = (0, -sin i, cos i)
    beta = jnp.sqrt(1.0 / rs)
    beta_n = -sense * beta * jnp.cos(phi) * sin_i
    gamma = 1.0 / jnp.sqrt(1.0 - beta**2)
    g = 1.0 / (gamma * (1.0 - beta_n))
    g = jnp.where(grav, g * jnp.sqrt(jnp.clip(1.0 - 2.0 / rs, 1e-6, None)), g)
    # lamp-post illuminator L = (0, 0, h). Delay relative to the direct ray from L:
    # |r - L| - (r - L).n = sqrt(r^2 + h^2) + r sin(phi) sin(i) + h cos(i)
    delay = jnp.sqrt(rs**2 + h_lamp**2) + rs * jnp.sin(phi) * sin_i + h_lamp * cos_i
    # photon-number weight per unit sky area: emissivity r^-q times g^3
    # (I_nu / nu^3 is invariant, so photon-number intensity scales as g^3)
    w = jnp.where(inside, rs**-q * g**3, 0.0)
    return w, g, delay


@dataclass
class ThinDisc:
    """Geometrically thin Keplerian disc emitting a (rest-frame) line or continuum template.

    Flat space with special-relativistic Doppler shifts; ``gravitational_redshift`` adds
    sqrt(1 - 2 r_g / r) but no light bending (face-on at r = 6 this gives g = 0.745, not
    the Schwarzschild 0.707), so use kerrz/Gradus maps for GR work. With ``sense = 1`` the
    disc rotates clockwise on the sky (angular momentum towards the observer); ``-1``
    reverses it.
    ``flux`` is the observed time-averaged photon flux (photons/s/cm^2). ``h_lamp`` is the
    height (r_g) of an on-axis illuminator that sets the DELAY map; give the disc the
    illuminator's driver to make it reverberate.
    """

    name: str
    template: dict  # rest-frame spectrum spec; only its shape is used
    flux: float
    mass_msun: float
    r_in: float = 6.0  # r_g
    r_out: float = 400.0  # r_g
    inclination: float = 30.0  # deg
    position_angle: float = 0.0  # deg, of the projected major axis, East of North
    emissivity_index: float = 3.0
    h_lamp: float = 10.0  # r_g
    gravitational_redshift: bool = False
    x: float = 0.0
    y: float = 0.0
    driver: str | None = None
    supersample: int = 6
    sense: int = 1

    kind = "thin_disc"

    def rg_cm(self):
        return G_CGS * self.mass_msun * MSUN_G / C_CM_S**2

    def render(self, scene):
        grid = scene.grid
        rg_uas = float(length_to_uas(self.rg_cm(), scene.distance_pc))
        xc, yc = (jnp.asarray(a) for a in grid.centres())
        size = jnp.asarray(grid.sizes())
        px, py = _subpixel_points(xc - self.x, yc - self.y, size, self.supersample)
        inc = np.deg2rad(self.inclination)
        # PA measured East of North: rotate so the major axis maps onto x'
        pa = np.deg2rad(90.0 - self.position_angle)
        w, g, delay = _disc_kernel(px / rg_uas, py / rg_uas, self.r_in, self.r_out,
                                   np.cos(inc), np.sin(inc), pa, self.emissivity_index,
                                   self.h_lamp, self.gravitational_redshift, float(self.sense))
        area = (size / self.supersample) ** 2  # sky area of each sample, uas^2
        wa = w * area[:, None]
        tot = wa.sum(axis=1)
        safe = jnp.where(tot > 0, tot, 1.0)
        g_mean = jnp.where(tot > 0, (wa * g).sum(axis=1) / safe, 1.0)
        g_var = (wa * (g - g_mean[:, None]) ** 2).sum(axis=1) / safe
        d_mean = (wa * delay).sum(axis=1) / safe
        norm = np.asarray(tot)
        if norm.sum() <= 0:
            raise ValueError(f"disc {self.name!r} covers no pixels of the grid")
        norm = norm / norm.sum() * self.flux
        tg = self.rg_cm() / C_CM_S
        maps = {
            "NORM": norm,
            "G": np.where(norm > 0, np.asarray(g_mean), 1.0),
            "SIGMA": np.where(norm > 0, np.sqrt(np.asarray(g_var)) / np.asarray(g_mean), 0.0),
            "DELAY": np.where(norm > 0, np.asarray(d_mean) * tg, 0.0),
        }
        tmpl = build_spectrum(self.template, scene.energy_edges)
        return SpecMapLayer(self.name, grid, [tmpl], maps, driver=self.driver)


COMPONENTS = {c.kind: c for c in (PointSource, GaussianBlob, ThinDisc)}
