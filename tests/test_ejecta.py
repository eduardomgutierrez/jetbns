import numpy as np
import pytest

from jetbns import BrokenPowerLaw, HomologousPowerLaw, NumericalEjecta, OutflowHistory
from jetbns.constants import SOLAR_MASS, SPEED_OF_LIGHT


def test_homologous_profile_conserves_mass() -> None:
    model = HomologousPowerLaw()
    assert model.mass(0.2) == pytest.approx(model.mass_msun * SOLAR_MASS, rel=2e-4)


def test_homologous_velocity_reaches_configured_outer_speed() -> None:
    model = HomologousPowerLaw(max_beta=0.4)
    assert model.velocity(model.outer_radius(0.2), 0.2) == pytest.approx(0.4 * SPEED_OF_LIGHT)


def test_sharp_broken_profile_conserves_mass() -> None:
    model = BrokenPowerLaw(tail=False)
    assert model.mass(0.2) == pytest.approx(model.mass_msun * SOLAR_MASS, rel=3e-4)


def test_smooth_tail_is_continuous_at_nominal_outer_radius() -> None:
    model = BrokenPowerLaw(tail=True)
    time = 0.2
    outer = model.nominal_outer_radius(time)
    below, above = model.density(np.array([outer * (1 - 1e-8), outer * (1 + 1e-8)]), time)
    assert above == pytest.approx(below, rel=2e-7)
    assert model.density(model.outer_radius(time) * 1.01, time) == 0.0


def test_numerical_profile_is_physical_and_vectorized() -> None:
    history = OutflowHistory(
        time_s=np.array([0.005, 0.01, 0.02, 0.04, 0.08]),
        velocity_c=np.array([0.42, 0.38, 0.34, 0.29, 0.24]),
        mass_loss_rate_g_s=np.array([3e31, 2.5e31, 1.8e31, 1.1e31, 5e30]),
        electron_fraction=np.array([0.08, 0.09, 0.10, 0.12, 0.15]),
    )
    model = NumericalEjecta(history, integration_samples=128)
    time = 0.2
    radius = np.geomspace(model.inner_radius(time) * 1.01, model.outer_radius(time) * 0.99, 16)
    density = model.density(radius, time)
    velocity = model.velocity(radius, time)
    electron_fraction = model.electron_fraction(radius, time)
    assert density.shape == radius.shape
    assert np.all(density >= 0)
    assert np.any(density > 0)
    assert np.all((velocity >= 0) & (velocity < SPEED_OF_LIGHT))
    assert np.all((electron_fraction >= 0) & (electron_fraction <= 1))


def test_numerical_cutoff_modes_above_fastest_recorded_shell() -> None:
    history = OutflowHistory(
        time_s=np.array([0.005, 0.01, 0.02]),
        velocity_c=np.array([0.40, 0.35, 0.30]),
        mass_loss_rate_g_s=np.full(3, 1.0e30),
        electron_fraction=np.full(3, 0.1),
    )
    sharp = NumericalEjecta(history, cutoff_mode="sharp", integration_samples=64)
    smooth = NumericalEjecta(history, cutoff_mode="smooth", integration_samples=64)
    time = 0.2
    radius = 1.05 * sharp.outer_radius(time)

    assert sharp.density(radius, time) == 0.0
    assert sharp.velocity(radius, time) == 0.0
    assert smooth.density(radius, time) > 0.0
    assert smooth.velocity(radius, time) > 0.4 * SPEED_OF_LIGHT
    assert sharp.optical_depth_outer_radius(time) == sharp.outer_radius(time)
    assert smooth.optical_depth_outer_radius(time) == 3.0 * smooth.outer_radius(time)


def test_numerical_kernel_shape_controls_tail_falloff() -> None:
    history = OutflowHistory(
        time_s=np.array([0.005, 0.01, 0.02]),
        velocity_c=np.array([0.40, 0.35, 0.30]),
        mass_loss_rate_g_s=np.full(3, 1.0e30),
    )
    exponential = NumericalEjecta(
        history, cutoff_mode="smooth", kernel_shape=1.0, integration_samples=64
    )
    gaussian = NumericalEjecta(
        history, cutoff_mode="smooth", kernel_shape=2.0, integration_samples=64
    )
    radius = 1.2 * exponential.outer_radius(0.2)

    assert exponential.density(radius, 0.2) > gaussian.density(radius, 0.2)


def test_hdf5_loader_is_the_persisted_input_path(tmp_path) -> None:
    import h5py

    path = tmp_path / "outflow.h5"
    with h5py.File(path, "w") as handle:
        group = handle.create_group("itheta=00000")
        group["time"] = [0.01, 0.02, 0.03]
        group["vel"] = np.array([0.3, 0.25, 0.2]) * SPEED_OF_LIGHT
        group["rho"] = [1.0e5, 8.0e4, 6.0e4]
        group["ye"] = [0.1, 0.12, 0.15]
    history = OutflowHistory.from_hdf5(
        path,
        bin_name="itheta=00000",
        extraction_radius_cm=4.42e7,
        solid_angle_sr=0.2,
    )
    assert history.solid_angle_sr == 0.2
    assert history.velocity_c == pytest.approx([0.3, 0.25, 0.2])
    expected_mass_rate = (
        np.array([1.0e5, 8.0e4, 6.0e4])
        * (1 - np.array([0.3, 0.25, 0.2]) ** 2) ** -0.5
        * np.array([0.3, 0.25, 0.2])
        * SPEED_OF_LIGHT
        * (4.42e7) ** 2
        * 0.2
    )
    assert history.mass_loss_rate_g_s == pytest.approx(expected_mass_rate)

    model = NumericalEjecta.from_hdf5(
        path,
        bin_name="itheta=00000",
        extraction_radius_cm=4.42e7,
        solid_angle_sr=0.2,
        integration_samples=32,
    )
    assert model.history.solid_angle_sr == 0.2


@pytest.mark.parametrize(
    "constructor",
    [
        lambda: HomologousPowerLaw(max_beta=1.0),
        lambda: BrokenPowerLaw(inner_index=3.0),
        lambda: OutflowHistory(np.array([1.0, 0.5]), np.array([0.2, 0.3]), np.ones(2)),
    ],
)
def test_invalid_parameters_are_rejected(constructor) -> None:
    with pytest.raises(ValueError):
        constructor()


# Mass launched after the end of the record (WP4 audit, 2026-11-18).
def short_record_history() -> OutflowHistory:
    time = np.linspace(0.01, 0.03, 40)
    return OutflowHistory(
        time_s=time,
        velocity_c=np.linspace(0.5, 0.2, 40),
        mass_loss_rate_g_s=np.geomspace(1e31, 3e30, 40),
        electron_fraction=np.full(40, 0.3),
        solid_angle_sr=0.0119,
    )


def launched_mass(model: NumericalEjecta, time: float) -> float:
    """Analytic mass launched before ``time``: recorded trapezoid plus the power-law continuation."""
    history = model.history
    scale = 4.0 * np.pi / history.solid_angle_sr
    recorded = np.trapezoid(history.mass_loss_rate_g_s, history.time_s) * scale
    last, a = history.time_s[-1], model.post_simulation_mass_index
    continued = history.mass_loss_rate_g_s[-1] * scale * last / (a - 1) * (1 - (last / time) ** (a - 1))
    return recorded + continued


def enclosed_mass(model: NumericalEjecta, time: float) -> float:
    return model.mass_above(model.inner_radius(time) * 1.0001, time, samples=4096)


@pytest.mark.parametrize("samples", [512, 256])
@pytest.mark.parametrize("time", [1.0, 2.0, 4.0, 8.0])
def test_numerical_mass_after_record_is_conserved(samples: int, time: float) -> None:
    model = NumericalEjecta(short_record_history(), cutoff_mode="smooth", integration_samples=samples)
    assert enclosed_mass(model, time) == pytest.approx(launched_mass(model, time), rel=0.01)


@pytest.mark.parametrize("time", [1.0, 8.0])
def test_numerical_mass_after_record_converges_in_launch_samples(time: float) -> None:
    coarse = NumericalEjecta(short_record_history(), cutoff_mode="smooth", integration_samples=256)
    fine = NumericalEjecta(short_record_history(), cutoff_mode="smooth", integration_samples=4096)
    assert enclosed_mass(coarse, time) == pytest.approx(enclosed_mass(fine, time), rel=0.01)
def _write_bin(path, *, time, vel_c, rho, bernoulli=None, theta=False):
    import h5py

    with h5py.File(path, "w") as handle:
        if theta:
            # 51 bin centres over [0, pi], as in the THC_M1 outflow files
            handle["theta"] = (np.arange(51) + 0.5) * np.pi / 51
        group = handle.create_group("itheta=00000")
        group["time"] = np.asarray(time, dtype=float)
        group["vel"] = np.asarray(vel_c, dtype=float) * SPEED_OF_LIGHT
        group["rho"] = np.asarray(rho, dtype=float)
        group["ye"] = np.full(len(time), 0.2)
        if bernoulli is not None:
            group["bernoulli"] = np.asarray(bernoulli, dtype=float)


def _bernoulli_file(path, theta=False):
    # x = bernoulli/c^2; one negative (floored), one exactly 0.5, one zero (floored), two ordinary.
    x = np.array([-0.01, 0.5, 0.0, 0.1, 0.02])
    _write_bin(
        path,
        time=[0.01, 0.02, 0.03, 0.04, 0.05],
        vel_c=[0.30, 0.40, 0.25, 0.20, 0.10],
        rho=[1.0e5, 8.0e4, 6.0e4, 4.0e4, 2.0e4],
        bernoulli=x * SPEED_OF_LIGHT**2,
        theta=theta,
    )
    return x


def test_asymptotic_velocity_is_the_bernoulli_closed_form(tmp_path) -> None:
    path = tmp_path / "outflow.h5"
    x = _bernoulli_file(path)
    history = OutflowHistory.from_hdf5(
        path, bin_name="itheta=00000", extraction_radius_cm=4.42e7, velocity="asymptotic"
    )
    expected = np.full_like(x, 1.0e-12)
    positive = x > 0
    expected[positive] = np.sqrt(x[positive] * (2 + x[positive])) / (1 + x[positive])
    assert history.velocity_c[1] == pytest.approx(np.sqrt(0.5 * 2.5) / 1.5, rel=1e-14)
    assert history.velocity_c[0] == 1.0e-12
    assert history.velocity_c[2] == 1.0e-12
    np.testing.assert_allclose(history.velocity_c, expected, rtol=1e-14, atol=0)
    assert history.velocity_source == "asymptotic"
    assert history.dropped_samples == 0
    floored = OutflowHistory.from_hdf5(
        path,
        bin_name="itheta=00000",
        extraction_radius_cm=4.42e7,
        velocity="asymptotic",
        velocity_floor=1.0e-6,
    )
    assert floored.velocity_c[0] == 1.0e-6


def test_mass_loss_rate_always_comes_from_recorded_velocity(tmp_path) -> None:
    path = tmp_path / "outflow.h5"
    _bernoulli_file(path)
    kwargs = dict(bin_name="itheta=00000", extraction_radius_cm=4.42e7, solid_angle_sr=0.0119)
    recorded = OutflowHistory.from_hdf5(path, velocity="recorded", **kwargs)
    asymptotic = OutflowHistory.from_hdf5(path, velocity="asymptotic", **kwargs)
    np.testing.assert_array_equal(recorded.mass_loss_rate_g_s, asymptotic.mass_loss_rate_g_s)
    beta = np.array([0.30, 0.40, 0.25, 0.20, 0.10])
    expected = (
        np.array([1.0e5, 8.0e4, 6.0e4, 4.0e4, 2.0e4])
        * (1 - beta**2) ** -0.5 * beta * SPEED_OF_LIGHT * 4.42e7**2 * 0.0119
    )
    np.testing.assert_allclose(recorded.mass_loss_rate_g_s, expected, rtol=1e-13)
    assert recorded.velocity_c == pytest.approx(beta)
    assert recorded.velocity_source == "recorded"
    assert not np.array_equal(recorded.velocity_c, asymptotic.velocity_c)


def test_asymptotic_velocity_requires_bernoulli_dataset(tmp_path) -> None:
    path = tmp_path / "outflow.h5"
    _write_bin(path, time=[0.01, 0.02, 0.03], vel_c=[0.3, 0.2, 0.1], rho=[1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="bernoulli"):
        OutflowHistory.from_hdf5(
            path, bin_name="itheta=00000", extraction_radius_cm=4.42e7, velocity="asymptotic"
        )
    history = OutflowHistory.from_hdf5(path, bin_name="itheta=00000", extraction_radius_cm=4.42e7)
    assert history.velocity_source == "recorded"
    with pytest.raises(ValueError, match="velocity"):
        OutflowHistory.from_hdf5(
            path, bin_name="itheta=00000", extraction_radius_cm=4.42e7, velocity="vel"
        )


def test_non_increasing_times_are_dropped_and_counted(tmp_path) -> None:
    path = tmp_path / "outflow.h5"
    # an exact duplicate at 0.02 and a step back to 0.025 (< 0.03) are both dropped
    _write_bin(
        path,
        time=[0.01, 0.02, 0.02, 0.03, 0.025, 0.04],
        vel_c=[0.3, 0.25, 0.25, 0.2, 0.9, 0.1],
        rho=[1.0, 2.0, 2.0, 3.0, 99.0, 4.0],
    )
    kwargs = dict(bin_name="itheta=00000", extraction_radius_cm=4.42e7)
    history = OutflowHistory.from_hdf5(path, **kwargs)
    assert history.dropped_samples == 2
    np.testing.assert_array_equal(history.time_s, [0.01, 0.02, 0.03, 0.04])
    assert history.velocity_c == pytest.approx([0.3, 0.25, 0.2, 0.1])
    assert len(history.mass_loss_rate_g_s) == 4
    with pytest.raises(ValueError, match="strictly increasing"):
        OutflowHistory.from_hdf5(path, drop_non_increasing=False, **kwargs)
    model = NumericalEjecta.from_hdf5(path, integration_samples=32)
    assert model.history.dropped_samples == 2


def test_dataset_shape_mismatch_is_rejected(tmp_path) -> None:
    import h5py

    path = tmp_path / "outflow.h5"
    _write_bin(path, time=[0.01, 0.02, 0.03], vel_c=[0.3, 0.2, 0.1], rho=[1.0, 1.0, 1.0])
    with h5py.File(path, "a") as handle:
        del handle["itheta=00000/rho"]
        handle["itheta=00000/rho"] = [1.0, 1.0]
    with pytest.raises(ValueError, match="shape"):
        OutflowHistory.from_hdf5(path, bin_name="itheta=00000", extraction_radius_cm=4.42e7)


def test_exporter_numerical_settings(tmp_path) -> None:
    import importlib.util
    import sys
    from pathlib import Path

    source = Path(__file__).parents[1] / "examples" / "export_npc_models.py"
    spec = importlib.util.spec_from_file_location("export_npc_models", source)
    exporter = importlib.util.module_from_spec(spec)
    sys.modules["export_npc_models"] = exporter
    spec.loader.exec_module(exporter)

    for model in exporter.NUMERICAL + exporter.NUMERICAL_2025:
        path = tmp_path / model.profile
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            _bernoulli_file(path, theta=True)

    # The production models are unchanged.
    assert [m.name for m in exporter.NUMERICAL] == ["num_dd2_equal", "num_sfho", "num_sly"]
    for model in exporter.NUMERICAL:
        ejecta = exporter.build(model, tmp_path)
        assert ejecta.beta_width == 0.05
        assert (ejecta.post_simulation_mass_index, ejecta.post_simulation_velocity_index) == (
            5 / 3,
            1 / 4,
        )
        assert ejecta.history.velocity_source == "recorded"
        assert ejecta.cutoff_mode == "smooth" and ejecta.kernel_shape == 2.0
        assert ejecta.integration_samples == 256 and ejecta.history_subsamples == 1

    expected = {
        "num_dd2_equal_2025": (1.50, 0.13),
        "num_sfho_2025": (7.16, 5.23),
        "num_sly_2025": (9.52, 6.91),
    }
    assert [m.name for m in exporter.NUMERICAL_2025] == list(expected)
    for base, model in zip(exporter.NUMERICAL, exporter.NUMERICAL_2025):
        assert (model.profile, model.luminosity_iso_erg_s, model.launch_s) == (
            base.profile,
            base.luminosity_iso_erg_s,
            base.launch_s,
        )
        ejecta = exporter.build(model, tmp_path)
        assert ejecta.beta_width == 0.035
        assert (
            ejecta.post_simulation_mass_index,
            ejecta.post_simulation_velocity_index,
        ) == expected[model.name]
        assert ejecta.history.velocity_source == "asymptotic"
