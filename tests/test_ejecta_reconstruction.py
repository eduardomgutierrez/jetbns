"""Regression for the short recorded pulse missed by the old uniform grid."""
import numpy as np
import pytest

from jetbns import HomologousTail, NumericalEjecta, OutflowHistory
from jetbns.constants import SOLAR_MASS, SPEED_OF_LIGHT


def test_short_pulse_at_late_time_against_independent_quadrature():
    times = np.linspace(.005, .025, 1001)
    rates = 1e30*np.exp(-((times-.012)/.003)**2)
    ejecta = NumericalEjecta(OutflowHistory(times, np.full_like(times, .4), rates),
                             cutoff_mode="smooth", integration_samples=128)
    radius = ejecta.extraction_radius_cm+.4*SPEED_OF_LIGHT*(10.-.012)
    grid = np.linspace(times[0], times[-1], 100001)
    beta = (radius-ejecta.extraction_radius_cm)/(SPEED_OF_LIGHT*(10.-grid))
    integrand = np.interp(grid, times, rates)*np.exp(-((beta-.4)/.035)**2)
    integrand *= np.sqrt(1-beta**2)/(np.sqrt(np.pi)*.035*(10.-grid))
    expected = np.trapezoid(integrand, grid)/(4*np.pi*radius**2*SPEED_OF_LIGHT)
    assert ejecta.density(radius, 10.) == pytest.approx(expected, rel=.002)


def test_homologous_tail_conserves_baryon_mass():
    ejecta = HomologousTail()
    for time in (.2, 1., 10.):
        assert ejecta.mass(time) == pytest.approx(ejecta.mass_msun*SOLAR_MASS, rel=1e-4)
