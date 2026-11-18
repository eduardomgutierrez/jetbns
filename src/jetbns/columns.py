"""Exterior p-n column along the light cone of a particle leaving the jet head.

``lightcone_column`` is the primary conversion depth of the NPC export (np-converter decision 2026-11-18; ported from
np-converter ``analysis/npc/diagnostics/columns.py``, where it was the primary exterior column of the WP4 audit): the
all-baryon p-n column met by a particle that leaves the head at (r_h, t) and moves radially outward at ``speed``
(default c),

    col_b_lightcone = int_{r_h}^{r_exit} (rho(r, t_r)/m_p) sigma_pn Gamma_a(r, t_r) (1 - beta_a(r, t_r)) dr,
    t_r = t + (r - r_h)/speed,

with rho the proper rest-mass density (``ejecta.density``, g cm^-3) and beta_a = ejecta.velocity/c the radial speed of
the ejecta in the lab frame at (r, t_r). Gamma_a (1 - beta_a) = sqrt((1 - beta_a)/(1 + beta_a)) is the number of
targets met per unit lab path by an ultrarelativistic particle moving radially outward through the outflowing medium.
r_exit is the root of r = R(t_r) (R = ejecta.optical_depth_outer_radius), found by bracketing and bisection. If the
light cone is still inside R at max_radius_factor r_h (100 by default; e.g. the smooth-tail limit R = 3 R_nominal of
NumericalEjecta grows faster than light when the fastest recorded shell exceeds c/3), the integral stops there and the
result says exited=False.

The frozen lab-time slice (the profile at t for every r, as in the exporter's ``snapshot_columns``) is not the column an
outgoing particle actually crosses. For a ballistic homologous flow and speed = c the light-cone column equals, by
particle conservation, the lab-frame column int Gamma (rho/m_p) sigma_pn dr of the slice at t weighted per shell by the
dilution (c t (1 - beta)/(c t - r_h))^2 (tested); at the last samples of the v3 suite it is 1.8-5.3 times the frozen
column with the Gamma_a (1 - beta_a) factor. For a steady wind it equals n_u sigma_pn r/(2 Gamma_a).

The grid is geometric in x = r - r_h from 1e-6 of the path to the exit, plus r_h itself; [x_1, exit] is integrated by
the trapezoid in ln x and [0, x_1] by the trapezoid in x. At the 99 % samples of the v3 suite the relative errors
against 8192 points are at most 7.6e-7, 1.9e-7 and 4.6e-8 for 256, 512 and 1024 points (the plain trapezoid in x on
the same grid: 1.2e-4 at 512, independent of the model). Every grid point needs its own profile evaluation at its own
time, which makes the function slow for NumericalEjecta (``density_and_velocity`` is used when available, one
quadrature for both). For speed < c the integrand keeps the factor Gamma_a (1 - beta_a), as specified; the exact rate
per unit path of a projectile of speed beta_p c would be Gamma_a (1 - beta_a/beta_p) (equal for speed = c).

Units are cgs; times are lab-frame seconds after the merger.
"""

from __future__ import annotations

import math
import operator
from dataclasses import asdict, dataclass

import numpy as np

from .constants import PROTON_MASS, SPEED_OF_LIGHT
from .npc import PN_CROSS_SECTION  # sigma_pn [cm^2]; npc imports this module lazily

LIGHTCONE_SAMPLES = 512             # points of the light-cone grid
LIGHTCONE_MAX_RADIUS_FACTOR = 100.0  # stop the light cone at this multiple of the head radius if it has not exited
LIGHTCONE_GRID_START = 1e-6         # first nonzero offset r - r_h of the light-cone grid, as a fraction of the path

__all__ = ["LightconeColumn", "lightcone_column", "LIGHTCONE_SAMPLES", "LIGHTCONE_MAX_RADIUS_FACTOR"]


@dataclass(frozen=True)
class LightconeColumn:
    """Result of :func:`lightcone_column`.

    col_b_lightcone  all-baryon p-n column met along the light cone (dimensionless)
    exit_radius_cm   radius where the light cone leaves the ejecta (root of r = optical_depth_outer_radius(t_r), from the
                     inside to 1e-13); when exited is False, the stop radius max_radius_factor * r_h [cm]
    exit_time_s      lab time t + (exit_radius_cm - r_h)/speed [s]
    exited           False when the light cone is still inside optical_depth_outer_radius at the stop radius
    """

    col_b_lightcone: float
    exit_radius_cm: float
    exit_time_s: float
    exited: bool

    def as_dict(self) -> dict:
        return asdict(self)


def _scalar(value, name, *, positive=False, nonnegative=False) -> float:
    a = np.asarray(value, dtype=float)
    if a.ndim != 0:
        raise ValueError(f"{name} must be a scalar")
    x = float(a)
    if not np.isfinite(x) or (positive and x <= 0) or (nonnegative and x < 0):
        raise ValueError(f"{name} must be finite{' and positive' if positive else ''}"
                         f"{' and nonnegative' if nonnegative else ''}, got {value!r}")
    return x


def _outer(ejecta, time_s) -> float:
    return _scalar(ejecta.optical_depth_outer_radius(time_s), "optical_depth_outer_radius", positive=True)


def _lightcone_exit(ejecta, r_h, t, speed, stop):
    """(radius, exited): the first root of r = R(t + (r - r_h)/speed) beyond r_h, approached from the inside (the
    returned radius is inside to 1e-13), bracketed by doubling r and capped at ``stop``; (stop, False) if no root."""
    def inside(r):
        return r < _outer(ejecta, t + (r - r_h) / speed)

    lo = hi = r_h
    while True:
        hi = min(2.0 * hi, stop)
        if not inside(hi):
            break
        if hi >= stop:
            return stop, False
        lo = hi
    for _ in range(200):
        if hi - lo <= 1e-13 * hi:
            break
        mid = 0.5 * (lo + hi)
        if inside(mid):
            lo = mid
        else:
            hi = mid
    return lo, True


def _density_velocity(ejecta, r, t):
    both = getattr(ejecta, "density_and_velocity", None)     # one quadrature for both (NumericalEjecta)
    rho, v = both(r, t) if callable(both) else (ejecta.density(r, t), ejecta.velocity(r, t))
    return _scalar(rho, "density"), _scalar(v, "velocity")


def lightcone_column(ejecta, radius_cm, time_s, *, pn_cross_section_cm2=PN_CROSS_SECTION,
                     samples=LIGHTCONE_SAMPLES, speed=SPEED_OF_LIGHT,
                     max_radius_factor=LIGHTCONE_MAX_RADIUS_FACTOR) -> LightconeColumn:
    """All-baryon p-n column along the light cone of a particle leaving the head (a :class:`LightconeColumn`).

    ejecta             object with density(r, t) [g cm^-3, proper], velocity(r, t) [cm s^-1, lab, radial] (or
                       density_and_velocity(r, t)) and optical_depth_outer_radius(t) [cm], called with scalars
    radius_cm, time_s  head radius [cm] and lab-frame time after the merger [s] where the particle starts
    samples            grid points (>= 3): r_h plus samples - 1 points geometric in r - r_h
    speed              particle speed [cm s^-1], 0 < speed <= c; it sets the retarded times t_r = t + (r - r_h)/speed
    max_radius_factor  stop radius in units of r_h if the light cone has not left the ejecta (> 1)

    A head at or beyond optical_depth_outer_radius(t) gives a zero column with exit_radius_cm = r_h. ValueError for
    invalid arguments, non-finite or negative densities, |beta| >= 1. See the module docstring for the definition.
    """
    r_h = _scalar(radius_cm, "radius_cm", positive=True)
    t = _scalar(time_s, "time_s", nonnegative=True)
    sigma = _scalar(pn_cross_section_cm2, "pn_cross_section_cm2", positive=True)
    v_p = _scalar(speed, "speed", positive=True)
    if v_p > SPEED_OF_LIGHT:
        raise ValueError(f"speed must not exceed c, got {speed!r}")
    factor = _scalar(max_radius_factor, "max_radius_factor", positive=True)
    if factor <= 1.0:
        raise ValueError("max_radius_factor must exceed 1")
    try:
        n = operator.index(samples)
    except TypeError:
        raise ValueError(f"samples must be an integer, got {samples!r}") from None
    if n < 3:
        raise ValueError("samples must be at least 3")
    if r_h >= _outer(ejecta, t):
        return LightconeColumn(0.0, r_h, t, True)

    r_exit, exited = _lightcone_exit(ejecta, r_h, t, v_p, factor * r_h)
    path = r_exit - r_h
    x = np.concatenate([[0.0], np.geomspace(LIGHTCONE_GRID_START * path, path, n - 1)])
    rho, beta = np.empty(n), np.empty(n)
    for i, xi in enumerate(x):
        rho[i], v = _density_velocity(ejecta, r_h + xi, t + xi / v_p)
        beta[i] = v / SPEED_OF_LIGHT
    if np.any(rho < 0):
        raise ValueError("ejecta densities must be nonnegative")
    if np.any(np.abs(beta) >= 1):
        raise ValueError("ejecta speeds must satisfy |beta| < 1")
    integrand = rho / PROTON_MASS * sigma * np.sqrt((1.0 - beta) / (1.0 + beta))   # Gamma_a (1 - beta_a)
    # [0, x_1]: trapezoid in x; [x_1, path]: trapezoid in u = ln x of x f(x) (uniform in u), which converges much faster
    # than the trapezoid in x when the integrand falls off toward the exit and stays second order otherwise
    du = math.log(1.0 / LIGHTCONE_GRID_START) / (n - 2)
    g = integrand[1:] * x[1:]
    column = 0.5 * (integrand[0] + integrand[1]) * x[1] + du * (g.sum() - 0.5 * (g[0] + g[-1]))
    return LightconeColumn(float(column), r_exit, t + path / v_p, exited)
