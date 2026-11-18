"""Scientific interface checks for the optional notebook helpers."""

from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

pytest.importorskip("pandas", reason="requires the notebooks extra")


@pytest.fixture
def playground(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "examples"))
    import npc_playground

    return npc_playground


def test_playground_uses_requested_engine_and_keeps_nonbreakout(playground, tmp_path):
    run = playground.run_model(
        playground.ANALYTICAL[0],
        tmp_path,
        lorentz_factor=20,
        opening_angle_deg=6,
        max_duration_s=0.1,
        sample_count=12,
    )
    assert run.jet.engine.lorentz_factor == 20
    expected = run.model.luminosity_iso_erg_s * (1 - np.cos(np.deg2rad(6))) / 2
    assert run.jet.engine.luminosity_erg_s == pytest.approx(expected)
    assert not run.trajectory.broke_out
    assert run.trajectory.breakout_time_s is None
    assert len(playground.table(run)) == 12
    figures = playground.plot_comparisons([run], normalized=True)
    assert figures[0].axes[0].get_xlabel() == "lab time [s]"
    for figure in figures:
        playground.plt.close(figure)


def test_playground_save_preserves_config_and_does_not_overwrite(playground, tmp_path):
    run = playground.run_model(playground.ANALYTICAL[0], tmp_path, sample_count=12)
    assert run.trajectory.broke_out
    paths = playground.save_run(run, tmp_path / "trial")
    assert all(path.is_file() for path in paths)
    with h5py.File(paths[0]) as handle:
        np.testing.assert_array_equal(
            handle["relative_lorentz_factor"], run.npc.relative_lorentz_factor
        )
        assert handle.attrs["schema"] == "jetbns.npc-inputs.v5"
        assert "exploratory" in handle["metadata"].attrs["status"]
    with h5py.File(paths[1]) as handle:
        assert handle.attrs["time_step_s"] == run.time_step_s
        assert handle["engine"].attrs["lorentz_factor"] == run.jet.engine.lorentz_factor
        assert "endpoint_radial_columns" in handle
    with pytest.raises(FileExistsError):
        playground.save_run(run, tmp_path / "trial")


def test_playground_missing_profile_is_actionable(playground, tmp_path):
    with pytest.raises(FileNotFoundError, match="Missing local profile"):
        playground.run_model(playground.NUMERICAL[0], tmp_path)


def test_playground_numerical_kernel_and_composition_from_hdf5(playground, tmp_path):
    from dataclasses import replace

    with h5py.File(tmp_path / "synthetic.h5", "w") as handle:
        handle["theta"] = [0.05, 0.15, 0.25]
        group = handle.create_group("itheta=00000")
        group["time"] = [0.01, 0.02, 0.03]
        group["vel"] = np.array([0.3, 0.25, 0.2]) * 2.99792458e10
        group["rho"] = [1e5, 8e4, 6e4]
        group["ye"] = [0.12, 0.12, 0.12]
    model = replace(playground.NUMERICAL[0], profile="synthetic.h5", launch_s=0.1)
    run = playground.run_model(
        model, tmp_path, kernel_shape=1.0, beta_width=0.04, max_duration_s=0.1, sample_count=12
    )
    assert run.ejecta.kernel_shape == 1
    assert run.ejecta.beta_width == 0.04
    np.testing.assert_allclose(run.npc.electron_fraction, 0.12)


def test_dashboard_slider_release_replaces_latest_run(playground, monkeypatch, tmp_path):
    pytest.importorskip("ipywidgets", reason="requires the notebooks extra")
    solves = []

    def solve(model, *_args, **_kwargs):
        solves.append((model, _kwargs))
        return SimpleNamespace(model=model)

    monkeypatch.setattr(playground, "run_model", solve)
    monkeypatch.setattr(playground, "summary", lambda _runs: "summary")
    monkeypatch.setattr(playground, "table", lambda _run: "table")
    monkeypatch.setattr(playground, "plot_run", lambda _run: playground.plt.figure())
    monkeypatch.setattr(
        playground,
        "plot_comparisons",
        lambda *_args, **_kwargs: [playground.plt.figure() for _ in range(4)],
    )

    dashboard = playground.launch_dashboard(tmp_path)
    controls = dashboard.children[1].children
    run_button = dashboard.children[3].children[0]
    run_button.click()
    assert len(solves) == 1

    mass_slider = controls[1]
    assert not mass_slider.continuous_update
    mass_slider.value = 2e-4
    assert len(solves) == 2
    assert solves[-1][0].mass_msun == pytest.approx(2e-4)

    preset = dashboard.children[0]
    preset.value = playground.NUMERICAL[0].name
    assert len(solves) == 3
    assert solves[-1][0].kind == "numerical"

    width_slider = controls[9]
    assert not width_slider.continuous_update
    width_slider.value = 0.08
    assert len(solves) == 4
    assert solves[-1][1]["beta_width"] == pytest.approx(0.08)
