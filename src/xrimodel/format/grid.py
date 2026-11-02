"""Spatial grids on the tangent plane.

A grid is a set of pixels ("leaves"), each with a centre (x, y) and a side length,
all in micro-arcseconds relative to the model's reference position. Every map
stored on a grid is a flat array with one value per leaf, so code that works on
pixel centres and sizes works for both grid types.

* :class:`RegularGrid`: an nx by ny block of square pixels, written to FITS as an
  image with a TAN WCS.
* :class:`QuadtreeGrid`: an adaptive grid of square leaves at different refinement
  levels inside a square root cell. Each leaf is addressed by a NUNIQ-style integer,
  ``uniq = 4 * 4**level + morton(ix, iy)`` (the IVOA MOC convention, applied to a flat
  square instead of HEALPix).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..units import UAS_PER_DEG

# --------------------------------------------------------------------------- tangent plane


def offsets_to_radec(x_uas, y_uas, ra0_deg, dec0_deg):
    """Gnomonic (TAN) de-projection of tangent-plane offsets to RA/Dec in degrees.

    ``x`` points East, ``y`` North. Note: a float64 RA in degrees resolves only about
    0.2 uas near RA = 300 deg, so photon lists also carry the offsets themselves.
    """
    xi = np.deg2rad(np.asarray(x_uas, dtype=float) / UAS_PER_DEG)
    eta = np.deg2rad(np.asarray(y_uas, dtype=float) / UAS_PER_DEG)
    a0, d0 = np.deg2rad(ra0_deg), np.deg2rad(dec0_deg)
    denom = np.cos(d0) - eta * np.sin(d0)
    ra = a0 + np.arctan2(xi, denom)
    dec = np.arctan2(np.sin(d0) + eta * np.cos(d0), np.hypot(xi, denom))
    return np.rad2deg(ra) % 360.0, np.rad2deg(dec)


def radec_to_offsets(ra_deg, dec_deg, ra0_deg, dec0_deg):
    """Inverse of :func:`offsets_to_radec`; returns (x, y) in uas."""
    a, d = np.deg2rad(ra_deg), np.deg2rad(dec_deg)
    a0, d0 = np.deg2rad(ra0_deg), np.deg2rad(dec0_deg)
    cosc = np.sin(d0) * np.sin(d) + np.cos(d0) * np.cos(d) * np.cos(a - a0)
    xi = np.cos(d) * np.sin(a - a0) / cosc
    eta = (np.cos(d0) * np.sin(d) - np.sin(d0) * np.cos(d) * np.cos(a - a0)) / cosc
    return np.rad2deg(xi) * UAS_PER_DEG, np.rad2deg(eta) * UAS_PER_DEG


# --------------------------------------------------------------------------- regular grid


@dataclass(frozen=True)
class RegularGrid:
    """nx by ny square pixels of side ``scale`` uas, centred on offset (x0, y0).

    Arrays on this grid are stored in FITS order, ``image[iy, ix]``, flattened row-major.
    Following sky convention, column index ``ix`` increases towards the West
    (x decreases), so the FITS image displays East-left like any sky image.
    """

    nx: int
    ny: int
    scale: float  # uas per pixel
    x0: float = 0.0
    y0: float = 0.0

    kind = "regular"

    @property
    def n(self) -> int:
        return self.nx * self.ny

    @property
    def shape(self) -> tuple[int, int]:
        return (self.ny, self.nx)

    def centres(self) -> tuple[np.ndarray, np.ndarray]:
        ix = np.arange(self.nx)
        iy = np.arange(self.ny)
        x = self.x0 - (ix + 0.5 - self.nx / 2) * self.scale
        y = self.y0 + (iy + 0.5 - self.ny / 2) * self.scale
        xx, yy = np.meshgrid(x, y)
        return xx.ravel(), yy.ravel()

    def sizes(self) -> np.ndarray:
        return np.full(self.n, float(self.scale))

    def to_image(self, values) -> np.ndarray:
        return np.asarray(values).reshape(self.shape)

    def wcs_header(self, ra0: float, dec0: float) -> dict:
        """TAN WCS keywords for an image on this grid (axes 1, 2)."""
        ra_c, dec_c = offsets_to_radec(self.x0, self.y0, ra0, dec0)
        return {
            "CTYPE1": "RA---TAN",
            "CTYPE2": "DEC--TAN",
            "CUNIT1": "deg",
            "CUNIT2": "deg",
            "CRVAL1": float(ra_c),
            "CRVAL2": float(dec_c),
            "CRPIX1": self.nx / 2 + 0.5,
            "CRPIX2": self.ny / 2 + 0.5,
            "CDELT1": -self.scale / UAS_PER_DEG,
            "CDELT2": self.scale / UAS_PER_DEG,
            "GRIDTYPE": "REGULAR",
            "GRIDSCAL": (float(self.scale), "[uas] pixel side"),
            "GRIDX0": (float(self.x0), "[uas] grid centre, East offset from REFRA"),
            "GRIDY0": (float(self.y0), "[uas] grid centre, North offset from REFDEC"),
        }

    @classmethod
    def from_header(cls, hdr, shape) -> RegularGrid:
        ny, nx = shape
        return cls(nx=nx, ny=ny, scale=hdr["GRIDSCAL"], x0=hdr["GRIDX0"], y0=hdr["GRIDY0"])


# --------------------------------------------------------------------------- quadtree grid


def _part1by1(n):
    n = np.asarray(n, dtype=np.int64) & 0x00000000FFFFFFFF
    n = (n | (n << 16)) & 0x0000FFFF0000FFFF
    n = (n | (n << 8)) & 0x00FF00FF00FF00FF
    n = (n | (n << 4)) & 0x0F0F0F0F0F0F0F0F
    n = (n | (n << 2)) & 0x3333333333333333
    n = (n | (n << 1)) & 0x5555555555555555
    return n


def _compact1by1(n):
    n = np.asarray(n, dtype=np.int64) & 0x5555555555555555
    n = (n | (n >> 1)) & 0x3333333333333333
    n = (n | (n >> 2)) & 0x0F0F0F0F0F0F0F0F
    n = (n | (n >> 4)) & 0x00FF00FF00FF00FF
    n = (n | (n >> 8)) & 0x0000FFFF0000FFFF
    n = (n | (n >> 16)) & 0x00000000FFFFFFFF
    return n


def encode_uniq(level, ix, iy):
    level = np.asarray(level, dtype=np.int64)
    return 4 * (np.int64(4) ** level) + (_part1by1(ix) | (_part1by1(iy) << 1))


def decode_uniq(uniq):
    uniq = np.asarray(uniq, dtype=np.int64)
    # level = floor(log4(uniq / 4)); use bit length to stay exact
    bits = np.floor(np.log2(uniq)).astype(np.int64)
    level = (bits - 2) // 2
    morton = uniq - 4 * (np.int64(4) ** level)
    return level, _compact1by1(morton), _compact1by1(morton >> 1)


@dataclass(frozen=True)
class QuadtreeGrid:
    """Adaptive grid: leaves of a quadtree over a square root cell.

    ``root_size`` is the root cell side in uas, centred on offset (x0, y0). At level L
    the cell side is ``root_size / 2**L``; leaf (ix, iy) at that level has ix increasing
    to the East and iy to the North.
    """

    uniq: np.ndarray
    root_size: float
    x0: float = 0.0
    y0: float = 0.0

    kind = "quadtree"

    @property
    def n(self) -> int:
        return len(self.uniq)

    @property
    def shape(self) -> tuple[int]:
        return (self.n,)

    def _decoded(self):
        return decode_uniq(self.uniq)

    def sizes(self) -> np.ndarray:
        level, _, _ = self._decoded()
        return self.root_size / (2.0**level)

    def centres(self):
        level, ix, iy = self._decoded()
        s = self.root_size / (2.0**level)
        x = self.x0 - self.root_size / 2 + (ix + 0.5) * s
        y = self.y0 - self.root_size / 2 + (iy + 0.5) * s
        return x, y

    @classmethod
    def refine(cls, root_size, refine_fn, max_level, x0=0.0, y0=0.0, min_level=0):
        """Build a quadtree by recursive subdivision.

        ``refine_fn(x, y, size) -> bool array`` decides, for cell centres and sides in
        uas, which cells are split further. Cells are split unconditionally up to
        ``min_level`` and never beyond ``max_level``.
        """
        leaves = []
        level = 0
        ix = np.array([0])
        iy = np.array([0])
        while True:
            s = root_size / 2.0**level
            x = x0 - root_size / 2 + (ix + 0.5) * s
            y = y0 - root_size / 2 + (iy + 0.5) * s
            if level >= max_level:
                split = np.zeros(ix.size, bool)
            elif level < min_level:
                split = np.ones(ix.size, bool)
            else:
                split = np.asarray(refine_fn(x, y, np.full(ix.size, s)), bool)
            leaves.append(encode_uniq(np.full((~split).sum(), level), ix[~split], iy[~split]))
            if not split.any():
                break
            px, py = ix[split], iy[split]
            ix = np.concatenate([2 * px, 2 * px + 1, 2 * px, 2 * px + 1])
            iy = np.concatenate([2 * py, 2 * py, 2 * py + 1, 2 * py + 1])
            level += 1
        return cls(uniq=np.sort(np.concatenate(leaves)), root_size=float(root_size), x0=x0, y0=y0)

    def header(self) -> dict:
        return {
            "GRIDTYPE": "QUADTREE",
            "HDUCLAS2": "QTGRID",
            "QTROOT": (float(self.root_size), "[uas] side of the root cell"),
            "GRIDX0": (float(self.x0), "[uas] root centre, East offset from REFRA"),
            "GRIDY0": (float(self.y0), "[uas] root centre, North offset from REFDEC"),
        }

    @classmethod
    def from_header(cls, hdr, uniq) -> QuadtreeGrid:
        return cls(uniq=np.asarray(uniq, np.int64), root_size=hdr["QTROOT"],
                   x0=hdr["GRIDX0"], y0=hdr["GRIDY0"])


Grid = RegularGrid | QuadtreeGrid
