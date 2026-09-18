"""Regression checks for the reproducible A&A 2020 comparison script."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "examples/compare_aanda2020_models.py"
    spec = importlib.util.spec_from_file_location("aanda2020_comparison", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_appendix_suite_contains_published_configurations():
    module = _module()
    names = tuple(case.name for case in module.CASES)
    assert names == (
        "A", "B", "C", "M-ref", "M-L4", "M-th30", "M-th45",
        "M-ti500", "M-M3", "M-M2-2", "M-M1",
    )
    assert module.CASES[0].luminosity_erg_s == 2e50
    assert module.CASES[3].opening_angle_deg == 15
    assert module.CASES[-1].mass_msun == .1


def test_appendix_case_runs_to_breakout():
    module = _module()
    trajectory = module.run_case(module.CASES[0], time_step_s=.01, max_time_s=1.)
    assert trajectory.broke_out
    assert trajectory.breakout_time_s > module.CASES[0].launch_time_s
    assert trajectory.breakout_radius_cm > 0
