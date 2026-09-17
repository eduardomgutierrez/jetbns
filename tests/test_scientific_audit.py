"""Regressions for failures that shape-only and self-equation tests missed."""
from dataclasses import replace

import numpy as np
import pytest

from jetbns import (
    HomologousTail,
    NpcConfig,
    NumericalEjecta,
    OutflowHistory,
    PropagationResult,
    evaluate_npc_inputs,
)
from jetbns.constants import SOLAR_MASS, SPEED_OF_LIGHT
from jetbns.npc import metzger_free_neutron_fraction


def test_late_time_reconstruction_resolves_short_recorded_pulse():
    # A 3-ms burst must not disappear when queried ten seconds later.
    launch = np.linspace(.005, .025, 1001)
    rate = 1e30 * np.exp(-((launch - .012) / .003)**2)
    history = OutflowHistory(launch, np.full_like(launch, .4), rate,
                             np.full_like(launch, .2))
    model = NumericalEjecta(history, cutoff_mode="smooth", integration_samples=128)
    time = 10.
    radius = model.extraction_radius_cm + .4 * SPEED_OF_LIGHT * (time - .012)
    density = model.density(radius, time)
    # Independent dense quadrature over the measured burst (extrapolated rate ~0).
    grid = np.linspace(launch[0], launch[-1], 100001)
    flight = time - grid
    beta = (radius - model.extraction_radius_cm) / (SPEED_OF_LIGHT * flight)
    integrand = (np.interp(grid, launch, rate) * np.exp(-((beta-.4)/.035)**2)
                 * np.sqrt(1-beta**2) / (np.sqrt(np.pi)*.035*flight))
    expected = np.trapezoid(integrand, grid) / (4*np.pi*radius**2*SPEED_OF_LIGHT)
    assert density == pytest.approx(expected, rel=2e-3)
    assert model.electron_fraction(radius, time) == pytest.approx(.2)
    assert replace(model, integration_samples=512).density(radius, time) == pytest.approx(
        density, rel=1e-3
    )


def test_homologous_tail_conserves_baryon_mass_and_advects_density():
    model = HomologousTail()
    for t in (.2, 1., 10.):
        assert model.mass(t) == pytest.approx(model.mass_msun*SOLAR_MASS, rel=1e-4)
    radius = np.geomspace(model.inner_radius(1.)*1.01, model.outer_radius(1.)*.99, 128)
    assert model.density(3*radius, 3.) == pytest.approx(model.density(radius, 1.)/27)
    assert model.velocity(radius, 1.) == pytest.approx(radius)
    assert model.velocity(model.outer_radius(1.), 1.) < SPEED_OF_LIGHT
    assert model.mass_above(model.nominal_outer_radius(1.), 1.) > 0


def test_mass_coordinate_uses_lab_frame_baryon_density():
    model = HomologousTail()
    r = np.geomspace(model.inner_radius(1.), model.outer_radius(1.), 8192)
    proper_integral = np.trapezoid(4*np.pi*r**2*model.density(r, 1.), r)
    assert model.mass(1.) > proper_integral


@pytest.mark.parametrize("kwargs", [
    {"electron_fraction": np.nan}, {"magnetic_field_at_reference_g": np.nan},
    {"free_neutron_decay_time_s": np.inf}, {"proton_free_fraction": -1},
])
def test_config_rejects_nonfinite_or_invalid_physics(kwargs):
    with pytest.raises(ValueError):
        NpcConfig(**kwargs)


def test_neutron_decay_limits_and_invalid_mass():
    assert metzger_free_neutron_fraction(.1, 0, 900) == pytest.approx(.8/np.e)
    assert metzger_free_neutron_fraction(.6, 0, 0) == 0
    with pytest.raises(ValueError):
        metzger_free_neutron_fraction(.1, -1, 0)


def test_outflow_rejects_nan_composition():
    with pytest.raises(ValueError):
        OutflowHistory(np.array([0., 1.]), np.array([.2, .3]), np.ones(2),
                       np.array([.1, np.nan]))


def test_cold_shock_closure_relative_speed_and_target_rates():
    model = HomologousTail()
    times = np.array([1., 2.])
    radii = .3*SPEED_OF_LIGHT*times
    beta_a = np.full(2, .3)
    beta_rel = np.full(2, np.sqrt(1-1/9))
    beta_h = (beta_rel+beta_a)/(1+beta_rel*beta_a)
    trajectory = PropagationResult(times, radii, beta_h, beta_a, np.ones(2), False)
    inputs = evaluate_npc_inputs(trajectory, model)
    # Transform the upstream-frame shock speed to the downstream frame.
    expected = (inputs.shock_beta_upstream_frame-beta_rel)/(1-
                inputs.shock_beta_upstream_frame*beta_rel)
    assert inputs.shock_beta_downstream_frame == pytest.approx(expected)
    assert inputs.hydrodynamic_compression_ratio == pytest.approx([15., 15.])
    assert inputs.neutron_to_proton_optical_depth == pytest.approx(
        inputs.neutron_on_proton_collision_rate_s1
        * inputs.effective_path_length_upstream_cm/SPEED_OF_LIGHT
    )
    no_free_p = evaluate_npc_inputs(trajectory, model,
                                   config=NpcConfig(proton_free_fraction=0))
    assert np.all(no_free_p.proton_number_density_cm3 == 0)
    assert np.all(no_free_p.neutron_on_proton_collision_rate_s1 == 0)


def test_invalid_profile_composition_is_not_silently_replaced():
    class BadComposition(HomologousTail):
        def electron_fraction(self, radius, time):
            raise ValueError("invalid measured composition")
    tr = PropagationResult(np.array([1., 2.]), np.array([1e9, 2e9]),
                           np.array([.8, .8]), np.array([.2, .2]), np.ones(2), False)
    with pytest.raises(ValueError, match="invalid measured"):
        evaluate_npc_inputs(tr, BadComposition())
