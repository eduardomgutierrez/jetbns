import h5py
import numpy as np
import pytest

from jetbns import (
    ConstantEngine,
    HomologousPowerLaw,
    JetHead,
    NpcConfig,
    evaluate_npc_inputs,
    relative_lorentz_factor,
)
from jetbns.constants import (
    BOLTZMANN_CONSTANT,
    ELECTRON_MASS,
    ELEMENTARY_CHARGE,
    PROTON_MASS,
    RADIATION_CONSTANT,
    SPEED_OF_LIGHT,
)
from jetbns.npc import ERG_PER_KEV, PN_CROSS_SECTION


def make_solution():
    ejecta = HomologousPowerLaw(
        mass_msun=2.0e-3,
        max_beta=0.35,
        initial_outer_radius_cm=1.0e9,
    )
    engine = ConstantEngine(
        launch_time_s=0.1,
        launch_radius_cm=8.45e7,
        opening_angle_rad=np.deg2rad(6.8),
        luminosity_erg_s=3.0e49,
    )
    result = JetHead(engine, ejecta).propagate(max_time_s=2.0, time_step_s=2.0e-3)
    return ejecta, result


def test_relative_lorentz_factor_matches_collinear_velocity_difference() -> None:
    head_beta = np.array([0.8, 0.9])
    ambient_beta = np.array([0.2, 0.3])
    relative_beta = (head_beta - ambient_beta) / (1 - head_beta * ambient_beta)
    expected = 1 / np.sqrt(1 - relative_beta**2)
    assert relative_lorentz_factor(head_beta, ambient_beta) == pytest.approx(expected)


def test_equations_match_hand_calculation() -> None:
    ejecta, result = make_solution()
    config = NpcConfig(
        magnetic_field_at_reference_g=1.0e13,
        magnetic_reference_radius_cm=1.0e6,
        path_length="radius",
    )
    data = evaluate_npc_inputs(result, ejecta, config=config)

    index = len(data.time_s) // 2
    radius = data.radius_cm[index]
    density = ejecta.density(radius, data.time_s[index])
    number_density = density / PROTON_MASS
    ambient_gamma = 1 / np.sqrt(1 - data.ambient_beta[index] ** 2)
    magnetic_field = 1.0e13 * (1.0e6 / radius) ** 2
    temperature = (
        density
        * SPEED_OF_LIGHT**2
        * data.relative_lorentz_factor[index] ** 2
        / RADIATION_CONSTANT
    ) ** 0.25
    gyration = ELEMENTARY_CHARGE * magnetic_field / (
        PROTON_MASS * SPEED_OF_LIGHT**3 * number_density * PN_CROSS_SECTION
    )
    max_bh = 2 * ELECTRON_MASS * SPEED_OF_LIGHT**2 / (
        6 * BOLTZMANN_CONSTANT * temperature
    )

    assert data.upstream_number_density_cm3[index] == pytest.approx(number_density)
    assert data.upstream_magnetic_field_g[index] == pytest.approx(magnetic_field)
    assert data.pn_optical_depth[index] == pytest.approx(
        number_density * PN_CROSS_SECTION * radius / ambient_gamma
    )
    assert data.gyration_parameter[index] == pytest.approx(gyration)
    assert data.downstream_temperature_k[index] == pytest.approx(temperature)
    assert data.downstream_temperature_kev[index] == pytest.approx(
        BOLTZMANN_CONSTANT * temperature / ERG_PER_KEV
    )
    assert data.max_lorentz_factor_bethe_heitler[index] == pytest.approx(max_bh)
    assert data.max_lorentz_factor[index] == pytest.approx(min(gyration, max_bh))
    assert data.max_observer_energy_erg[index] == pytest.approx(
        data.head_lorentz_factor[index]
        * min(gyration, max_bh)
        * PROTON_MASS
        * SPEED_OF_LIGHT**2
    )


def test_breakout_is_excluded_and_remaining_path_is_supported() -> None:
    ejecta, result = make_solution()
    assert result.broke_out
    data = evaluate_npc_inputs(
        result, ejecta, config=NpcConfig(path_length="remaining_ejecta")
    )
    assert data.time_s.size == result.time_s.size - 1
    expected = np.array(
        [ejecta.outer_radius(t) - r for t, r in zip(data.time_s, data.radius_cm, strict=True)]
    )
    assert data.path_length_cm == pytest.approx(expected)
    assert np.all(data.path_length_cm > 0)


def test_composition_and_observer_boost_are_explicit() -> None:
    ejecta, result = make_solution()
    full = evaluate_npc_inputs(result, ejecta)
    half = evaluate_npc_inputs(
        result,
        ejecta,
        config=NpcConfig(target_nucleon_fraction=0.5),
        observer_lorentz_factor=2.0,
    )
    assert half.upstream_number_density_cm3 == pytest.approx(
        0.5 * full.upstream_number_density_cm3
    )
    assert half.pn_optical_depth == pytest.approx(0.5 * full.pn_optical_depth)
    assert half.gyration_parameter == pytest.approx(2.0 * full.gyration_parameter)
    assert np.all(half.observer_lorentz_factor == 2.0)


def test_hdf5_export_records_units_configuration_and_metadata(tmp_path) -> None:
    ejecta, result = make_solution()
    config = NpcConfig(path_length="remaining_ejecta")
    data = evaluate_npc_inputs(result, ejecta, config=config)
    output = tmp_path / "npc_inputs.h5"
    data.to_hdf5(output, config=config, metadata={"model": "test trajectory"})

    with h5py.File(output) as handle:
        assert handle.attrs["schema"] == "jetbns.npc-inputs.v1"
        assert handle["pn_optical_depth"].attrs["unit"] == "1"
        assert handle["upstream_density_g_cm3"].attrs["unit"] == "g cm^-3"
        assert handle["configuration"].attrs["path_length"] == "remaining_ejecta"
        assert handle["metadata"].attrs["model"] == "test trajectory"
        assert handle["time_s"][:] == pytest.approx(data.time_s)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"target_nucleon_fraction": 0},
        {"target_nucleon_fraction": 1.1},
        {"pn_inelasticity": 0},
        {"magnetic_field_at_reference_g": 0},
        {"path_length": "invalid"},
    ],
)
def test_configuration_rejects_invalid_physics(kwargs) -> None:
    with pytest.raises(ValueError):
        NpcConfig(**kwargs)
