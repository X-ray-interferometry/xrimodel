"""3-D scenes rendered by ray marching along the line of sight.

Geometry lives in a Cartesian system frame (cm). The observer looks from direction ``n``
(unit vector towards the observer); each sky pixel is a ray ``p(s) = p0 + s n``. Three
kinds of object take part:

* **Opaque bodies** (implicit solids: ``inside(p)``, an outward ``field(p)`` whose
  gradient is the surface normal, and a velocity field). The front-most body hit by a
  ray hides everything behind it. Bodies may emit from their visible surface, here by
  reflecting the light of an illuminating point source, with shadowing by any opaque
  body between the surface and the illuminator.
* **Volumes** (optically thin media with a hydrogen density, an absorbing fraction and
  an emissivity). They emit along the ray in front of the visible surface and absorb
  everything behind them: each emitter gets the absorbing column between itself and the
  observer, so there are no pairwise occlusion rules.
* **Point sources**, visible unless a body lies in front, absorbed by the column in front.

All emitters are reported per ray as observed photon intensities with Doppler factors,
light-travel delays relative to the illuminator, and absorbing columns; pixel maps are
flux-weighted moments of these over sub-pixel rays.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from ..units import C_CM_S

jax.config.update("jax_enable_x64", True)


def doppler(v, n):
    """Special-relativistic Doppler factor g = E_obs / E_emit for velocity v (cm/s)."""
    beta = v / C_CM_S
    b2 = jnp.sum(beta * beta, axis=-1)
    gamma = 1.0 / jnp.sqrt(1.0 - b2)
    return 1.0 / (gamma * (1.0 - jnp.sum(beta * n, axis=-1)))


@dataclass
class RayGeometry:
    """Map sky offsets (cm on the sky plane) to rays in the system frame.

    ``e_a`` and ``e_b`` span the sky plane, ``n`` points to the observer, and
    (e_a, e_b, n) is right-handed. ``origin`` is the system-frame point at sky offset 0.
    Rays run from s = +s_max (nearest the observer) to s = -s_max.
    """

    e_a: np.ndarray
    e_b: np.ndarray
    n: np.ndarray
    origin: np.ndarray
    s_max: float
    n_steps: int

    def points(self, a, b):
        s = jnp.linspace(self.s_max, -self.s_max, self.n_steps)
        p0 = self.origin + a[:, None] * self.e_a + b[:, None] * self.e_b
        return p0[:, None, :] + s[None, :, None] * self.n, s


def first_hit(inside_fn, p0, s, n, n_bisect=24):
    """Front-most crossing of an implicit solid along each ray.

    p0 (R, 3) are the ray origins (s = 0) and s (R, N) the sample positions along each ray,
    descending (front first). Returns (hit, s_hit): whether the ray enters the solid, and
    the crossing refined by bisection.
    """
    ins = inside_fn(p0[:, None, :] + s[..., None] * n)  # (R, N)
    hit = ins.any(axis=1)
    k = jnp.argmax(ins, axis=1)  # first True from the front
    rows = jnp.arange(s.shape[0])
    s_in = s[rows, k]
    s_out = jnp.where(k > 0, s[rows, jnp.maximum(k - 1, 0)], s_in)

    def body(_, lohi):
        lo, hi = lohi  # lo: inside, hi: outside (larger s)
        mid = 0.5 * (lo + hi)
        inside_mid = inside_fn((p0 + mid[:, None] * n)[:, None, :])[:, 0]
        return jnp.where(inside_mid, mid, lo), jnp.where(inside_mid, hi, mid)

    lo, hi = jax.lax.fori_loop(0, n_bisect, body, (s_in, s_out))
    return hit, jnp.where(hit, 0.5 * (lo + hi), -jnp.inf)


def sphere_samples(p0, n, centre, radius, n_samples):
    """Samples (R, n_samples) along each ray across a bounding sphere, front first.

    Rays that miss the sphere get samples at -inf (never inside anything).
    """
    d = centre - p0
    s_c = d @ n
    b2 = jnp.sum(d * d, axis=-1) - s_c**2
    h = jnp.sqrt(jnp.clip(radius**2 - b2, 0.0, None))
    t = jnp.linspace(1.0, -1.0, n_samples)
    s = s_c[:, None] + h[:, None] * t[None, :]
    return jnp.where((b2 < radius**2)[:, None], s, -1e300), s_c, jnp.sqrt(jnp.clip(b2, 0, None))


def slab_samples(p0, n, centre, radius, half_thickness, n_samples, axis=(0.0, 0.0, 1.0)):
    """Samples (R, n_samples) along each ray through the part of a bounding sphere that lies
    within a slab |(p - centre) . axis| <= half_thickness (for thin, disc-like bodies),
    front first."""
    axis = jnp.asarray(axis, dtype=float)
    d = centre - p0
    s_c = d @ n
    b2 = jnp.sum(d * d, axis=-1) - s_c**2
    h = jnp.sqrt(jnp.clip(radius**2 - b2, 0.0, None))
    na = jnp.dot(n, axis)
    na = jnp.where(jnp.abs(na) > 1e-12, na, 1e-12)
    z0 = (p0 - centre) @ axis
    s1, s2 = (half_thickness - z0) / na, (-half_thickness - z0) / na
    lo = jnp.maximum(jnp.minimum(s1, s2), s_c - h)
    hi = jnp.minimum(jnp.maximum(s1, s2), s_c + h)
    t = jnp.linspace(1.0, 0.0, n_samples)
    s = lo[:, None] + (hi - lo)[:, None] * t[None, :]
    return jnp.where(((b2 < radius**2) & (hi > lo))[:, None], s, -1e300)


def sphere_shadowed(p, light, centre, radius):
    """True where the segment from p to a point light passes through a sphere."""
    seg = light - p
    t = jnp.clip(jnp.sum((centre - p) * seg, axis=-1) / jnp.sum(seg * seg, axis=-1), 0.0, 1.0)
    closest = p + t[..., None] * seg
    return jnp.sum((closest - centre) ** 2, axis=-1) < radius**2


def segment_blocked(inside_fns, p_from, p_to, n_samples=48):
    """True where any solid lies on the open segment between p_from and p_to."""
    t = (jnp.arange(n_samples) + 0.5) / n_samples
    q = p_from[:, None, :] + t[None, :, None] * (p_to - p_from)[:, None, :]
    blocked = jnp.zeros(p_from.shape[0], dtype=bool)
    for fn in inside_fns:
        blocked = blocked | fn(q).any(axis=1)
    return blocked


def surface_normal(field_fn, p):
    """Outward unit normal from the gradient of a scalar field (positive outside)."""
    g = jax.vmap(jax.grad(field_fn))(p)
    return g / jnp.linalg.norm(g, axis=-1, keepdims=True)


# reflection law: I = albedo * C * F_irr * mu_i / (mu_i + mu_e), with C chosen so that the
# reflected photon flux at normal incidence equals albedo * F_irr
REFLECTION_C = 1.0 / (2.0 * np.pi * (1.0 - np.log(2.0)))
