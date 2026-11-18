"""Tests for jetbns.columns.lightcone_column on synthetic ejecta with closed-form answers.

Ported from np-converter analysis/tests/test_columns.py (light-cone part); the frozen-slice comparisons that used
np-converter's radial_columns are computed here with the same 1536-point geometric trapezoid.
"""
import math

import numpy as np
import pytest

import jetbns
from jetbns import LightconeColumn, lightcone_column
from jetbns.constants import PROTON_MASS as M_P_CGS
from jetbns.constants import SOLAR_MASS as MSUN_CGS
from jetbns.constants import SPEED_OF_LIGHT as C_CGS
from jetbns.npc import PN_CROSS_SECTION

SIGMA = 3e-26
T = 2.0                                  # lab time [s]
R0, OUTER, RH = 1e9, 1e11, 1e10          # inner edge, outer limit R, head radius [cm]


class PowerLawShell:
    """Static-profile stand-in: rho = rho0 (r/r0)^-k for r0 <= r <= R (zero outside), uniform radial speed beta c."""

    def __init__(self, rho0, k, beta=0.0, r0=R0, outer=OUTER):
        self.rho0, self.k, self.beta, self.r0, self.outer = rho0, k, beta, r0, outer

    def density(self, r, t):
        r = np.asarray(r, dtype=float)
        return np.where((r >= self.r0) & (r <= self.outer), self.rho0 * (r / self.r0) ** -self.k, 0.0)

    def velocity(self, r, t):
        return np.full_like(np.asarray(r, dtype=float), self.beta * C_CGS)

    def lab_mass_density(self, r, t):
        return self.density(r, t) / math.sqrt(1.0 - self.beta**2)

    def optical_depth_outer_radius(self, t):
        return self.outer


def shell_volume_integral(k):
    """int_RH^R r^(2-k) dr."""
    return math.log(OUTER / RH) if k == 3 else (OUTER**(3 - k) - RH**(3 - k)) / (3 - k)


def rho0_for(mass_msun, k, beta=0.0):
    """rho0 giving an exterior lab-frame mass `mass_msun` at RH."""
    gamma = 1.0 / math.sqrt(1.0 - beta**2)
    return mass_msun * MSUN_CGS / (4 * math.pi * gamma * R0**k * shell_volume_integral(k))


def power_integral(k, a=RH, b=OUTER):
    """int_a^b (r/r0)^-k dr."""
    return R0**k * (math.log(b / a) if k == 1 else (a**(1 - k) - b**(1 - k)) / (k - 1))


def rel(a, b):
    return abs(a - b) / abs(b)


def frozen_col_b(ejecta, r_h, t, samples=1536):
    """Frozen-slice all-baryon column int (rho/m_p) sigma Gamma_a (1 - beta_a) dr (the snapshot_columns convention)."""
    r = np.geomspace(r_h, ejecta.optical_depth_outer_radius(t), samples)
    beta = np.asarray(ejecta.velocity(r, t)) / C_CGS
    return float(np.trapezoid(np.asarray(ejecta.density(r, t)) / M_P_CGS * SIGMA
                              * np.sqrt((1 - beta) / (1 + beta)), r))


class HomologousShell:
    """Ballistic homologous stand-in: proper density rho = t^-3 f(r/t) with f = a (v/v_min)^-k for v_min <= v <= v_max
    (zero outside), radial velocity r/t, outer limit v_max t."""

    def __init__(self, a, k, vmin, vmax):
        self.a, self.k, self.vmin, self.vmax = a, k, vmin, vmax

    def _inside(self, r, t):
        v = np.asarray(r, dtype=float) / t
        return v, (v >= self.vmin) & (v <= self.vmax)

    def density(self, r, t):
        v, inside = self._inside(r, t)
        return np.where(inside, t**-3 * self.a * (np.clip(v, self.vmin, None) / self.vmin) ** -self.k, 0.0)

    def velocity(self, r, t):
        v, inside = self._inside(r, t)
        return np.where(inside, v, 0.0)

    def lab_mass_density(self, r, t):
        beta = self.velocity(r, t) / C_CGS
        return self.density(r, t) / np.sqrt(1.0 - beta**2)

    def optical_depth_outer_radius(self, t):
        return self.vmax * t


def conservation_column(lab_density, r_h, t, outer, points=None):
    """Particle-conservation closed form for a homologous flow from the origin: sigma/m_p int_{r_h}^{outer} n_lab(r, t)
    (c t (1 - beta)/(c t - r_h))^2 dr, beta = r/(c t), by adaptive quadrature on the slice at t."""
    from scipy.integrate import quad
    ct = C_CGS * t

    def f(r):
        return float(lab_density(r, t)) / M_P_CGS * (ct * (1.0 - r / ct) / (ct - r_h)) ** 2

    value, _ = quad(f, r_h, outer, points=points, epsrel=1e-12, epsabs=0, limit=1000)
    return SIGMA * value


def test_cross_section_is_the_npc_default():
    assert SIGMA == PN_CROSS_SECTION
    assert jetbns.columns.PN_CROSS_SECTION == PN_CROSS_SECTION


def test_lightcone_static_medium_equals_col_b_static():
    rho0 = rho0_for(3e-4, 2.0)
    shell = PowerLawShell(rho0, 2.0)
    c = lightcone_column(shell, RH, T)
    assert isinstance(c, LightconeColumn) and c.exited
    assert rel(c.col_b_lightcone, SIGMA / M_P_CGS * rho0 * power_integral(2.0)) < 1e-4
    assert rel(c.col_b_lightcone, frozen_col_b(shell, RH, T)) < 1e-4
    assert rel(c.exit_radius_cm, OUTER) < 1e-12 and c.exit_radius_cm <= OUTER       # approached from the inside
    assert math.isclose(c.exit_time_s, T + (c.exit_radius_cm - RH) / C_CGS, rel_tol=1e-14)
    slow = lightcone_column(shell, RH, T, speed=0.5 * C_CGS)                         # only the timing changes
    assert slow.col_b_lightcone == c.col_b_lightcone
    assert math.isclose(slow.exit_time_s, T + (slow.exit_radius_cm - RH) / (0.5 * C_CGS), rel_tol=1e-14)


def test_lightcone_steady_wind_is_n_sigma_r_over_two_gamma():
    """Stationary relativistic wind rho ~ r^-2 to large R: column = n_u sigma r_h Gamma_a (1 - beta_a) (1 - r_h/R),
    which tends to n_u sigma r_h/(2 Gamma_a) for beta_a -> 1 (the role attribute of the export)."""
    beta = 0.999
    shell = PowerLawShell(1e-10, 2.0, beta=beta, outer=50 * RH)
    c = lightcone_column(shell, RH, T)
    assert c.exited
    n_u = float(shell.density(RH, T)) / M_P_CGS
    gamma = 1.0 / math.sqrt(1.0 - beta**2)
    exact = n_u * SIGMA * RH * math.sqrt((1 - beta) / (1 + beta)) * (1.0 - 1.0 / 50)
    assert rel(c.col_b_lightcone, exact) < 1e-4
    assert rel(c.col_b_lightcone, n_u * SIGMA * RH / (2.0 * gamma) * (1.0 - 1.0 / 50)) < 1e-3


@pytest.mark.parametrize("k", [3.0, 6.0])
def test_lightcone_homologous_flow_equals_particle_conservation(k):
    t, vmin, vmax = 2.0, 0.05 * C_CGS, 0.8 * C_CGS
    shell = HomologousShell(3e-5, k, vmin, vmax)
    r_h = 0.3 * C_CGS * t
    c = lightcone_column(shell, r_h, t)
    expected = conservation_column(shell.lab_mass_density, r_h, t, vmax * t)
    assert rel(c.col_b_lightcone, expected) < 1e-4
    r_exit = vmax * (C_CGS * t - r_h) / (C_CGS - vmax)                               # where r = v_max t_r
    assert c.exited and rel(c.exit_radius_cm, r_exit) < 1e-10
    assert rel(c.exit_time_s, (C_CGS * t - r_h) / (C_CGS - vmax)) < 1e-10
    assert c.col_b_lightcone > frozen_col_b(shell, r_h, t)                           # the frozen slice undercounts
    grid = np.geomspace(r_h, vmax * t, 20001)
    lab_slice = SIGMA / M_P_CGS * np.trapezoid(shell.lab_mass_density(grid, t), grid)
    assert c.col_b_lightcone < lab_slice                                              # spherical dilution


def test_lightcone_converges_at_second_order():
    t, shell = 2.0, HomologousShell(3e-5, 6.0, 0.05 * C_CGS, 0.8 * C_CGS)
    r_h = 0.3 * C_CGS * t
    exact = conservation_column(shell.lab_mass_density, r_h, t, shell.vmax * t)
    e = [rel(lightcone_column(shell, r_h, t, samples=n).col_b_lightcone, exact) for n in (129, 257)]
    assert 3.0 < e[0] / e[1] < 5.0


def test_lightcone_homologous_tail_against_particle_conservation():
    ejecta = jetbns.HomologousTail()
    t = 2.0
    ct = C_CGS * t
    r_h = 0.3 * ct
    c = lightcone_column(ejecta, r_h, t)
    expected = conservation_column(ejecta.lab_mass_density, r_h, t, ejecta.limit_beta * ct,
                                   points=[ejecta.nominal_beta * ct])
    assert rel(c.col_b_lightcone, expected) < 1e-3
    assert c.exited and rel(c.exit_radius_cm, ejecta.limit_beta * (ct - r_h) / (1 - ejecta.limit_beta)) < 1e-10
    assert c.col_b_lightcone > frozen_col_b(ejecta, r_h, t)


def test_lightcone_that_never_exits_stops_and_is_flagged():
    rho0 = rho0_for(3e-4, 2.0)
    shell = PowerLawShell(rho0, 2.0, outer=3 * RH)                                   # matter ends at 3 r_h ...
    shell.optical_depth_outer_radius = lambda t: 2.0 * C_CGS * t                     # ... but the limit outruns light
    c = lightcone_column(shell, RH, T, max_radius_factor=10.0, samples=4096)
    assert not c.exited and c.exit_radius_cm == 10.0 * RH
    assert math.isclose(c.exit_time_s, T + 9.0 * RH / C_CGS, rel_tol=1e-14)
    assert rel(c.col_b_lightcone, SIGMA / M_P_CGS * rho0 * power_integral(2.0, RH, 3 * RH)) < 2e-3


def test_lightcone_prefers_density_and_velocity():
    shell = PowerLawShell(rho0_for(3e-4, 2.0), 2.0)
    plain = lightcone_column(shell, RH, T).col_b_lightcone
    shell.density_and_velocity = lambda r, t: (2.0 * float(shell.density(r, t)), float(shell.velocity(r, t)))
    assert rel(lightcone_column(shell, RH, T).col_b_lightcone, 2.0 * plain) < 1e-14


def test_lightcone_head_outside_gives_zero():
    c = lightcone_column(PowerLawShell(1.0, 2.0), OUTER, T)
    assert c.as_dict() == {"col_b_lightcone": 0.0, "exit_radius_cm": OUTER, "exit_time_s": T, "exited": True}


@pytest.mark.parametrize("kwargs, match", [
    ({"speed": 0.0}, "speed"), ({"speed": 1.01 * C_CGS}, "speed"), ({"speed": float("nan")}, "speed"),
    ({"samples": 2}, "samples"), ({"samples": 3.5}, "samples"), ({"max_radius_factor": 1.0}, "max_radius_factor"),
    ({"pn_cross_section_cm2": -1.0}, "pn_cross_section_cm2"), ({"radius_cm": float("nan")}, "radius_cm"),
    ({"time_s": -1.0}, "time_s"),
])
def test_lightcone_invalid_arguments_raise(kwargs, match):
    args = {"radius_cm": RH, "time_s": T, **kwargs}
    with pytest.raises(ValueError, match=match):
        lightcone_column(PowerLawShell(1.0, 2.0), args.pop("radius_cm"), args.pop("time_s"), **args)


def test_lightcone_bad_profiles_raise():
    nan = PowerLawShell(1.0, 2.0)
    nan.density = lambda r, t: np.where(np.asarray(r) > 2 * RH, np.nan, 1.0)
    with pytest.raises(ValueError, match="density"):
        lightcone_column(nan, RH, T)
    with pytest.raises(ValueError, match="beta"):
        lightcone_column(PowerLawShell(1.0, 2.0, beta=1.0), RH, T)
