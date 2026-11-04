"""Time drivers: dimensionless multipliers on a layer's mean photon rate.

A driver maps time t (s) to a non-negative factor with mean ~1, so a layer's
normalisation stays its time-averaged flux. A layer can carry a per-pixel delay
map; its rate at pixel p is then ``mean_rate_p * driver(t - delay_p)``.

* :class:`Constant`: always 1.
* :class:`LightCurve`: tabulated, in TIME or PHASE (periodic, with period derivatives
  as SIMPUT's Taylor series). Written as a SIMPUT ``LIGHTCURVE`` extension.
* :class:`PSDDriver`: a power spectrum in fractional rms^2/Hz plus a seed. Written as a
  SIMPUT ``POWSPEC`` extension and realised with Timmer & Koenig (1995) when photons
  are sampled, so the same file and seed give the same light curve.

PSD shapes for it: :func:`lorentzian_psd` (zero-centred or peaked; sum several for
broadband noise and QPOs), :func:`powerlaw_psd` (rms set over a band) and
:func:`broken_powerlaw_psd`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


class Driver:
    name: str

    def __call__(self, t) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def prepare(self, t0: float, t1: float) -> Driver:
        """Return a driver that can be evaluated on [t0, t1] (realises stochastic ones)."""
        return self

    def max_on(self, t0: float, t1: float) -> float:
        """An upper bound of the driver on [t0, t1], for Poisson thinning."""
        raise NotImplementedError  # pragma: no cover


@dataclass
class Constant(Driver):
    name: str = "constant"

    def __call__(self, t):
        return np.ones_like(np.asarray(t, dtype=float))

    def max_on(self, t0, t1):
        return 1.0


@dataclass
class LightCurve(Driver):
    """Tabulated relative flux, linearly interpolated.

    If ``periodic`` is True, ``x`` is phase in [0, 1) and phase(t) = phase0 + t/P
    - (dP/dt) t^2 / (2 P^2), the first terms of SIMPUT 1.1's Taylor series (higher
    derivatives are stored but not yet used). Otherwise ``x`` is time (s) and the curve
    is held at its end values outside the table.
    """

    x: np.ndarray
    flux: np.ndarray
    name: str = "lightcurve"
    periodic: bool = False
    period: float | None = None
    phase0: float = 0.0
    dperiod: tuple = ()  # (dP/dt, d2P/dt2, ...)

    def __post_init__(self):
        self.x = np.asarray(self.x, float)
        self.flux = np.asarray(self.flux, float)
        if np.any(self.flux < 0):
            raise ValueError("light curve flux must be non-negative")
        if self.periodic and not self.period:
            raise ValueError("a periodic light curve needs a period")

    def phase(self, t):
        t = np.asarray(t, float)
        ph =self.phase0 + t / self.period
        if len(self.dperiod) > 0 and self.dperiod[0]:
            ph = ph - self.dperiod[0] * t**2 / (2 * self.period**2)
        return np.mod(ph, 1.0)

    def __call__(self, t):
        if self.periodic:
            xp = np.concatenate([self.x, [self.x[0] + 1.0]])
            fp = np.concatenate([self.flux, [self.flux[0]]])
            return np.interp(self.phase(t), xp, fp)
        return np.interp(np.asarray(t, float), self.x, self.flux)

    def max_on(self, t0, t1):
        if self.periodic:
            return float(self.flux.max())
        inside = (self.x >= t0) & (self.x <= t1)
        cands = [self(np.array([t0]))[0], self(np.array([t1]))[0]]
        if inside.any():
            cands.append(self.flux[inside].max())
        return float(max(cands))


@dataclass
class PSDDriver(Driver):
    """Stochastic driver defined by a PSD (fractional rms^2 / Hz, one-sided).

    ``transform='exponential'`` gives a log-normal light curve (Uttley, McHardy & Vaughan
    2005): the Gaussian realisation y is mapped to exp(y - var/2), which stays positive
    and has mean 1; its fractional rms is then sqrt(exp(var) - 1), close to the PSD's
    rms for small variance. ``'linear'`` gives 1 + y, clipped at zero.

    One realisation must serve every layer that shares the driver, or delayed copies
    stop being coherent. Give ``span=(tstart, tstop)`` to fix the realised interval
    (then any sampling window inside it sees the same light curve); otherwise
    :func:`~.sampler.sample_photons` realises each driver once per call, over the union
    of the intervals all its layers need.
    """

    frequency: np.ndarray
    power: np.ndarray
    name: str = "psd"
    seed: int = 0
    transform: str = "exponential"
    dt: float | None = None  # realisation step (s); default 1 / (8 f_max)
    span: tuple[float, float] | None = None  # fixed realisation interval (s)
    _realised: LightCurve | None = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        self.frequency = np.asarray(self.frequency, float)
        self.power = np.asarray(self.power, float)
        if self.transform not in ("exponential", "linear"):
            raise ValueError("transform must be 'exponential' or 'linear'")
        if self.span is not None:
            self.span = (float(self.span[0]), float(self.span[1]))

    def realise(self, t0: float, t1: float) -> LightCurve:
        """Realise on [t0, t1], or on ``span`` if set (which must contain [t0, t1])."""
        if self.span is not None:
            if t0 < self.span[0] - 1e-6 or t1 > self.span[1] + 1e-6:
                raise ValueError(f"driver {self.name!r}: [{t0}, {t1}] is outside its span "
                                 f"{self.span}")
            t0, t1 = self.span
        # oversample the highest PSD frequency so interpolation keeps its power
        dt = self.dt or 1.0 / (8.0 * self.frequency.max())
        n = int(np.ceil((t1 - t0) / dt)) + 1
        n = max(n + n % 2, 8)  # even length for the real FFT
        rng = np.random.default_rng(self.seed)
        f = np.fft.rfftfreq(n, dt)[1:]
        df = 1.0 / (n * dt)
        p = np.interp(f, self.frequency, self.power, left=0.0, right=0.0)
        amp = 0.5 * n * np.sqrt(p * df)
        coeff = np.concatenate([[0.0], amp * (rng.standard_normal(f.size)
                                              + 1j * rng.standard_normal(f.size))])
        if n % 2 == 0:  # the Nyquist term must be real
            coeff[-1] = amp[-1] * rng.standard_normal() * np.sqrt(2)
        y = np.fft.irfft(coeff, n)
        if self.transform == "exponential":
            var = float(np.sum(p) * df)  # expected variance of y, not this realisation's
            flux = np.exp(y - var / 2.0)
        else:
            flux = np.clip(1.0 + y, 0.0, None)
        t = t0 + dt * np.arange(n)
        return LightCurve(t, flux, name=self.name)

    def prepare(self, t0, t1):
        lc = self.realise(t0, t1)
        out = PSDDriver(self.frequency, self.power, self.name, self.seed, self.transform,
                        self.dt, self.span)
        out._realised = lc
        return out

    def __call__(self, t):
        if self._realised is None:
            raise RuntimeError("call prepare(t0, t1) before evaluating a PSD driver")
        return self._realised(t)

    def max_on(self, t0, t1):
        if self._realised is None:
            raise RuntimeError("call prepare(t0, t1) first")
        return self._realised.max_on(t0, t1)


def broken_powerlaw_psd(f, rms, f_break, slope_low=0.0, slope_high=2.0):
    """Bending power law in rms^2/Hz, normalised to the given fractional rms over f."""
    f = np.asarray(f, float)
    shape = (f / f_break) ** -slope_low / (1 + (f / f_break) ** (slope_high - slope_low))
    shape = shape / np.trapezoid(shape, f)
    return rms**2 * shape


def lorentzian_psd(f, rms, width, centre=0.0):
    """Lorentzian in rms^2/Hz with half width at half maximum ``width`` (Hz) about
    ``centre`` (Hz; 0 for zero-centred broadband noise), normalised analytically so that
    its integral over 0 < f < infinity is ``rms``^2 (Belloni, Psaltis & van der Klis 2002
    shapes). Sum several for a multi-Lorentzian power spectrum."""
    f = np.asarray(f, float)
    norm = np.pi / 2 + np.arctan(centre / width)
    return rms**2 / norm * width / (width**2 + (f - centre) ** 2)


def powerlaw_psd(f, rms, index, f_lo, f_hi):
    """Power law P ~ f^-index in rms^2/Hz, normalised analytically so that its integral
    over [``f_lo``, ``f_hi``] is ``rms``^2 (the convention for the very-low-frequency noise
    of Z sources, e.g. 0.001-1 Hz; Hasinger & van der Klis 1989). It is not truncated
    outside that band: the frequency grid given to :class:`PSDDriver` sets that."""
    f = np.asarray(f, float)
    if np.isclose(index, 1.0):
        band = np.log(f_hi / f_lo)
    else:
        band = (f_hi ** (1 - index) - f_lo ** (1 - index)) / (1 - index)
    return rms**2 / band * f ** (-index)
