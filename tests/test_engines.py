import numpy as np
import pytest

from jetbns import ConstantEngine, PowerLawEngine


def test_constant_engine_obeys_activity_window_and_retarded_time() -> None:
    engine = ConstantEngine(
        launch_time_s=0.1,
        launch_radius_cm=1.0e7,
        luminosity_erg_s=1.0e49,
        duration_s=0.5,
    )
    assert engine.source_luminosity(0.09) == 0
    assert engine.source_luminosity(0.2) == 1.0e49
    assert engine.source_luminosity(0.61) == 0
    assert engine.luminosity(1.0e10, 0.2) == 0


def test_isotropic_equivalent_conversion_uses_one_jet_solid_angle() -> None:
    angle = 0.1
    engine = ConstantEngine.from_isotropic_equivalent(
        1.0e52, opening_angle_rad=angle, launch_time_s=0
    )
    assert engine.luminosity_erg_s == pytest.approx(1.0e52 * (1 - np.cos(angle)) / 2)


def test_power_law_engine_has_continuous_plateau_and_decline() -> None:
    engine = PowerLawEngine(
        launch_time_s=0.1,
        luminosity_erg_s=2.0e49,
        plateau_duration_s=0.2,
        decay_index=2,
    )
    assert engine.source_luminosity(0.1) == 2.0e49
    assert engine.source_luminosity(0.3) == pytest.approx(2.0e49)
    assert engine.source_luminosity(0.5) == pytest.approx(0.5e49)

