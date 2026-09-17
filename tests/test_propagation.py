import numpy as np
import pytest

from jetbns import ConstantEngine, HomologousPowerLaw, JetHead, NumericalEjecta, OutflowHistory


def make_jet(luminosity: float = 3.0e49) -> JetHead:
    ejecta = HomologousPowerLaw(
        mass_msun=2.0e-3,
        max_beta=0.35,
        inner_radius_cm=1.0e6,
        initial_outer_radius_cm=1.0e9,
    )
    engine = ConstantEngine(
        launch_time_s=0.1,
        launch_radius_cm=8.45e7,
        opening_angle_rad=np.deg2rad(6.8),
        luminosity_erg_s=luminosity,
    )
    return JetHead(engine, ejecta)


def test_head_speed_lies_between_ambient_and_jet() -> None:
    jet = make_jet()
    # Allow the engine signal time to travel from the launch surface.
    head, ambient, tilde_l = jet.state(1.0e8, 0.11)
    assert 0 <= ambient < head < jet.engine.beta < 1
    assert tilde_l > 0


def test_propagation_is_monotonic_and_breaks_out() -> None:
    jet = make_jet()
    result = jet.propagate(max_time_s=2.0, time_step_s=2.0e-4)
    assert result.broke_out
    assert np.all(np.diff(result.time_s) > 0)
    assert np.all(np.diff(result.radius_cm) > 0)
    assert result.breakout_radius_cm == pytest.approx(
        jet.ejecta.outer_radius(result.breakout_time_s), rel=1e-8
    )


def test_more_luminous_jet_breaks_out_earlier() -> None:
    weak = make_jet(1.0e49).propagate(max_time_s=3.0, time_step_s=5.0e-4)
    strong = make_jet(1.0e50).propagate(max_time_s=3.0, time_step_s=5.0e-4)
    assert weak.broke_out and strong.broke_out
    assert strong.breakout_time_s < weak.breakout_time_s


def test_time_step_convergence() -> None:
    jet = make_jet()
    coarse = jet.propagate(max_time_s=2.0, time_step_s=5.0e-4)
    fine = jet.propagate(max_time_s=2.0, time_step_s=2.5e-4)
    assert coarse.breakout_time_s == pytest.approx(fine.breakout_time_s, rel=2e-3)


def test_smooth_numerical_ejecta_propagates_beyond_nominal_edge() -> None:
    history = OutflowHistory(
        time_s=np.array([0.005, 0.01, 0.02]),
        velocity_c=np.array([0.4, 0.35, 0.3]),
        mass_loss_rate_g_s=np.full(3, 1.0e32),
    )
    ejecta = NumericalEjecta(
        history,
        cutoff_mode="smooth",
        beta_width=0.05,
        integration_samples=32,
    )
    engine = ConstantEngine(
        launch_time_s=0.1,
        launch_radius_cm=8.45e7,
        opening_angle_rad=np.deg2rad(6.8),
        luminosity_erg_s=1.0e50,
    )
    jet = JetHead(engine, ejecta, breakout_optical_depth_samples=32)
    result = jet.propagate(max_time_s=3.0, time_step_s=2.0e-3)

    assert result.broke_out
    assert result.breakout_radius_cm > ejecta.outer_radius(result.breakout_time_s)
    head_beta, ambient_beta, _ = jet.state(result.breakout_radius_cm, result.breakout_time_s)
    shock_beta = jet.shock_beta_in_ambient_frame(head_beta, ambient_beta)
    optical_depth = ejecta.optical_depth(
        result.breakout_radius_cm,
        result.breakout_time_s,
        samples=jet.breakout_optical_depth_samples,
    )
    assert optical_depth == pytest.approx(1.0 / shock_beta, rel=2e-2)
