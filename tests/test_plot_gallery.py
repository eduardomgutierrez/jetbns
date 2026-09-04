import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


def load_gallery_module():
    pytest.importorskip("matplotlib")
    path = Path(__file__).parents[1] / "examples" / "plot_numerical_parameter_gallery.py"
    specification = importlib.util.spec_from_file_location("plot_numerical_gallery", path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def test_gallery_has_ten_distinct_representative_scenarios() -> None:
    gallery = load_gallery_module()
    scenarios = gallery.REPRESENTATIVE_SCENARIOS

    assert len(scenarios) == 10
    assert len({scenario.name for scenario in scenarios}) == 10
    assert {scenario.cutoff_mode for scenario in scenarios} == {"sharp", "smooth"}
    assert {scenario.launch_time_s for scenario in scenarios} == {0.1, 0.3, 1.0, 3.0}
    assert {scenario.luminosity_iso_erg_s for scenario in scenarios} == {
        1e49,
        1e50,
        1e51,
        1e52,
    }


def test_cumulative_optical_depth_is_nonnegative_and_decreases_outward() -> None:
    gallery = load_gallery_module()
    radius = np.geomspace(1e8, 1e10, 100)
    density = 1e4 * (radius / radius[0]) ** -2
    optical_depth = gallery.cumulative_optical_depth(radius, density)

    assert optical_depth[-1] == 0
    assert np.all(optical_depth >= 0)
    assert np.all(np.diff(optical_depth) <= 0)
