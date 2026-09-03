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
