"""Regression checks for the reproducible A&A 2020 comparison script."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


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


# Reference numbers: variant "b" (exact homologous r^-3.5 shell, jetbns defaults, time step 1 ms) of
# reduced/paper/audit/salafia.json in the np-converter repository, 2026-11-05; M-th45 see its comment.
# name -> (t_bo - t_inj [s], breakout radius [cm])
AUDIT_VARIANT_B = {
    "A": (0.08558350648070577, 1122813728.21049),
    "B": (0.15385654902855672, 2491504458.520177),
    "C": (0.13314091968161273, 2206253675.516346),
    "M-ref": (0.3038437584729114, 4253270619.344437),
    "M-L4": (0.2207532677478417, 3256874522.203062),
    "M-th30": (0.9525315349501664, 12032138697.106567),
    "M-th45": (4.760401154707643, 57694952671.81884),   # recomputed 2026-11-18 after the reconfinement-root fix (was 3.321 s, 4.043e10 cm with the shipped search, which switched this case to a conical jet)
    "M-ti500": (0.9808702663445852, 17768232370.47165),
    "M-M3": (0.11557689739624123, 1995631217.9503586),
    "M-M2-2": (0.4297681534917177, 5763317975.0837965),
    "M-M1": (1.0794444845902385, 13554040539.11653),
}


@pytest.fixture(scope="module")
def trajectories():
    module = _module()
    return module, {case.name: module.run_case(case) for case in module.CASES}


def test_reference_columns():
    module = _module()
    th45 = {case.name: case for case in module.CASES}["M-th45"]
    assert th45.sim_time_s is None and th45.model_time_s == 1.01


def test_full_table_matches_audit(trajectories):
    module, runs = trajectories
    for case in module.CASES:
        t_ref, r_ref = AUDIT_VARIANT_B[case.name]
        traj = runs[case.name]
        assert traj.broke_out
        assert traj.breakout_time_s - case.launch_time_s == pytest.approx(t_ref, rel=1e-2), case.name
        assert traj.breakout_radius_cm == pytest.approx(r_ref, rel=1e-2), case.name


def test_median_time_ratio_to_simulation(trajectories):
    module, runs = trajectories
    ratios = [(runs[c.name].breakout_time_s - c.launch_time_s) / c.sim_time_s
              for c in module.CASES if c.sim_time_s is not None]
    assert 1.0 <= np.median(ratios) <= 1.25   # the audit found 1.11
