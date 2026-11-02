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

The binary is a preset of :class:`~xrimodel.engine.system.System3D`: a ``RocheLobe``
donor ("star"), a ``FlaredDisc`` ("disc"), the donor's ``BetaWind`` ("wind") and lamps
"corona" and "disc_bb", in a frame centred on the compact object. Options: the wind can
be focused towards the compact object (``focus_amp``, ``focus_width``, ``focus_v_factor``)
and clumped (``clump_f_vol``, ``clump_size``, ``clump_interclump``, ``clump_seed``); with
``absorb_illumination`` the wind's column between the corona and each reflecting or
scattering point is added to that emission's NH.

Simplifications: circular orbit; the wind's ionisation is a smooth threshold in
xi = L / (n r^2) with L the corona luminosity in the scene's energy band; the disc's
height tapers to zero at ``r_disc_in``; reflection uses one angular law,
mu_i / (mu_i + mu_e), with pexmon (or an approximate) spectral shape per emission-angle
bin, and ignores the change of the incident shape by absorption.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..format.spectra import Spectrum
from ..units import G_CGS, MSUN_G, PC_CM
from .primitives import BetaWind, FlaredDisc, Lamp, RocheLobe
from .spec import build_spectrum
from .system import System3D
from .system import reflection_templates as _refl

RSUN_CM = 6.957e10
AU_CM = 1.495978707e13


def reflection_templates(edges, gamma, mu_edges):
    """pexmon templates per emission-angle bin for a power law of index ``gamma`` if XSPEC
    models are installed, else one approximate template."""
    spec = {"kind": "powerlaw", "gamma": gamma, "norm": 1.0}
    return _refl(edges, spec, build_spectrum(spec, edges), np.asarray(mu_edges, float))


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
    focus_amp: float = 0.0  # focused wind: density enhancement towards the compact object
    focus_width: float = 30.0  # deg
    focus_v_factor: float = 1.0  # v_inf multiplier along the focus
    clump_f_vol: float | None = None  # clumped wind: volume filling factor (None: smooth)
    clump_size: float = 0.05  # clump size / distance from the donor
    clump_interclump: float = 0.0
    clump_seed: int = 0
    absorb_illumination: bool = True
    nh_ism: float = 5.4e21
    x: float = 0.0  # uas, sky offset of the compact object
    y: float = 0.0
    driver: str | None = None
    supersample: int = 2
    n_steps: int = 128  # coarse samples along each ray
    n_fine: int = 96  # wind samples on each side of the donor, clustered at its surface
    n_body: int = 96  # samples across the donor's bounding sphere when finding its surface
    n_disc: int = 1024  # samples across the disc's slab when finding its surface
    wind_nh_bins: tuple = (0.0, 1e21, 1e22, 1e23)  # cm^-2, lower edges of the wind layers
    column_grid: tuple = (48, 96, 128)
    chunk: int = 4096

    kind = "xrbinary"

    # ------------------------------------------------------------------ geometry

    def separation_cm(self):
        m = (self.m_donor + self.m_compact) * MSUN_G
        p = self.period_days * 86400.0
        return (G_CGS * m * p**2 / (4 * np.pi**2)) ** (1.0 / 3.0)

    def frame(self):
        """Positions (cm) in the centre-of-mass frame, observer direction and sky-plane axes."""
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
        return System3D(name="", node_pa=self.node_pa).sky_axes()

    def _donor(self, f, origin):
        return RocheLobe(name="star", centre=tuple(f["donor"] - origin),
                         companion=tuple(f["bh"] - origin), q=f["q"],
                         r_eq=self.r_donor * RSUN_CM, cm=tuple(-origin),
                         rotation={"omega": f["omega"], "axis": (0.0, 0.0, 1.0),
                                   "centre": tuple(-origin)},
                         albedo=self.albedo, n_samples=self.n_body)

    def _roche(self, f):
        """Roche potential (units G M_donor / a, positions in units of a about the centre
        of mass), the surface level for r_donor, the L1 distance and the donor-L1 axis."""
        phi, level, t_l1, axis, _, _, _ = self._donor(f, np.zeros(3)).potential(1.0)
        return phi, level, t_l1, axis

    def system(self, edges=None):
        """The binary as a :class:`System3D` centred on the compact object."""
        f = self.frame()
        a, om = f["a"], f["omega"]
        bh, donor = f["bh"], f["donor"]

        def corot(p):  # km/s
            return tuple(np.cross([0.0, 0.0, om], p) / 1e5)

        lines = []
        if self.wind_line_ew and edges is not None:
            corona = build_spectrum(self.corona_spectrum, edges)
            for e0 in (6.70, 6.97):
                cont = np.interp(e0, corona.centres, corona.values)
                lines.append({"e0": e0, "sigma": 0.01,
                              "yield": 0.5 * self.wind_line_ew * cont / corona.photon_flux()})
        clump = None if self.clump_f_vol is None else {
            "f_vol": self.clump_f_vol, "size": self.clump_size,
            "interclump": self.clump_interclump, "seed": self.clump_seed}
        focus = None if not (self.focus_amp or self.focus_v_factor != 1.0) else {
            "towards": tuple(bh - bh), "amp": self.focus_amp, "width": self.focus_width,
            "v_factor": self.focus_v_factor}
        lamps = [Lamp(name="corona", spectrum=self.corona_spectrum, velocity=corot(bh),
                      driver=self.driver)]
        if self.disc_spectrum is not None:
            lamps.append(Lamp(name="disc_bb", spectrum=self.disc_spectrum, velocity=corot(bh),
                              illuminates=False))
        return System3D(
            name=self.name, inclination=self.inclination, node_pa=self.node_pa, x=self.x,
            y=self.y, nh_ism=self.nh_ism, lamps=lamps,
            bodies=[self._donor(f, bh),
                    FlaredDisc(name="disc", r_in=self.r_disc_in * a, r_out=self.r_disc_out * a,
                               h_over_r=self.disc_h_over_r, mass_msun=self.m_compact,
                               velocity=corot(bh), albedo=self.albedo, n_samples=self.n_disc)],
            volumes=[BetaWind(name="wind", star=tuple(donor - bh), r_star=self.r_donor * RSUN_CM,
                              mdot=self.mdot, v_inf=self.v_inf, beta=self.beta,
                              v_base=self.v_wind_base, extent=self.wind_extent * a,
                              velocity=corot(donor), log_xi_crit=self.log_xi_crit, lines=lines,
                              focus=focus, clumping=clump, refine_body="star",
                              n_fine=self.n_fine)],
            extent=self.wind_extent * a, n_coarse=self.n_steps, nh_bins=self.wind_nh_bins,
            absorb_illumination=self.absorb_illumination, column_grid=self.column_grid,
            supersample=self.supersample, chunk=self.chunk)

    # ------------------------------------------------------------------ rendering

    def render(self, scene):
        f = self.frame()
        sysm = self.system(scene.energy_edges)
        layers = sysm.render(scene)
        d_cm = scene.distance_pc * PC_CM
        corona = build_spectrum(self.corona_spectrum, scene.energy_edges)
        lamp = sysm._last["lamps"]["corona"]
        self._last_frame = dict(f, cm_per_uas=sysm._last["cm_per_uas"], col_bh=lamp["column"],
                                blocked=lamp["blocked"],
                                n_ph=4 * np.pi * d_cm**2 * corona.photon_flux(),
                                l_x=4 * np.pi * d_cm**2 * corona.energy_flux())
        return layers


__all__ = ["AU_CM", "RSUN_CM", "Spectrum", "XRBinary", "reflection_templates"]
