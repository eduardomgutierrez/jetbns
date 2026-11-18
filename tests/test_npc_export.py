"""The portable reader validates physics relations without importing jetbns."""
import importlib.util
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest


def load_example(name):
    path = Path(__file__).parents[1]/"examples"/f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_export_reader_and_summary_agree_and_detect_corruption(tmp_path):
    exporter = load_example("export_npc_models")
    reader = load_example("read_npc_inputs")
    model = exporter.ANALYTICAL[0]
    coarse = exporter.solve(model, tmp_path)
    fine = exporter.solve(model, tmp_path, fine=1)
    checks = exporter.audit_model(coarse, fine)
    path = tmp_path/"inputs.h5"
    with h5py.File(path, "w") as handle:
        handle.attrs["schema"] = exporter.SCHEMA
        row = exporter.write_model(handle, model, fine, checks, tmp_path)
        summary = handle.create_group("prebreakout_summary")
        summary.create_dataset("name", data=[model.name], dtype=h5py.string_dtype())
        for name, value in row.items():
            summary[name] = [value]
    assert reader.validate(path) == [model.name]
    with h5py.File(path) as handle:
        group = handle[f"models/{model.name}"]
        assert group.attrs["propagation_model"] == "JetCocoon"
        assert set(group["jet_cocoon"]) == set(exporter.COCOON_FIELDS)
        assert handle.attrs["schema"] == "jetbns.npc-model-export.v4"
        assert group.attrs["npc_schema"] == "jetbns.npc-inputs.v5"
        assert group.attrs["sampling"] == exporter.SAMPLING
        assert group.attrs["sample_fraction_of_propagation"] == 1-1e-4
        t = group["time_s"][:]
        t0, t_bo = fine[2].time_s[0], group.attrs["breakout_time_s"]
        assert t.size == 261
        np.testing.assert_allclose((t_bo-t[-1])/(t_bo-t0), 1e-4, rtol=1e-9)
        assert np.isclose(group["snapshot_columns"].attrs["time_s"], t0+.99*(t_bo-t0),
                          rtol=1e-12)
        assert "superseded" in group["snapshot_columns"].attrs["role"]
        for key in ("exterior_pn_column_lightcone", "exterior_pn_column_static",
                    "exterior_grey_optical_depth", "downstream_temperature_thermalised_kev",
                    "photon_rich_threshold_photons_per_baryon", "pair_loading_estimate"):
            assert "role" in group[key].attrs and "frame" in group[key].attrs
        assert group["lightcone_exited"].dtype == np.int8
        # early samples (head near the launch radius) stop at 100 r_h; near breakout all exit
        assert np.all(group["lightcone_exited"][-int(.3*t.size):] == 1)
        lightcone = group["exterior_pn_column_lightcone"][:]
        legacy = group["pn_optical_depth"][:]
        # the legacy local depth exceeds the light-cone column by a large factor near breakout
        assert 10 < legacy[-1]/lightcone[-1] < 1000
        # breakout criterion: tau_grey = 1/beta'_s at breakout, so tau_grey ~ 1 at the last sample
        assert .9 < group["exterior_grey_optical_depth"][-1] < 1.2
        gamma = group["relative_lorentz_factor"][:]
        legacy_kev = group["downstream_temperature_kev"][:]
        np.testing.assert_allclose(
            group["downstream_temperature_thermalised_kev"][:],
            legacy_kev*((gamma-1)*(4*gamma+3)/gamma**2)**.25, rtol=1e-12)
        gamma_u = 1/np.sqrt(1-group["shock_beta_upstream_frame"][:]**2)
        np.testing.assert_allclose(group["pair_loading_estimate"][:], 100*gamma_u, rtol=1e-12)
        np.testing.assert_allclose(group["photon_rich_threshold_photons_per_baryon"][:],
                                   1836.15267343*gamma_u, rtol=1e-9)
    snap = reader.snapshot(path, model.name)
    assert snap["proton_number_density_cm3"] == fine[-1].proton_number_density_cm3[-1]
    assert snap["neutron_on_proton_radial_column_depth"] < snap["neutron_to_proton_optical_depth"]
    for key, match in (("neutron_to_proton_optical_depth", "target optical depth"),
                       ("exterior_pn_column_static", "static column"),
                       ("exterior_grey_optical_depth", "static column")):
        broken = tmp_path/f"broken_{key}.h5"
        shutil.copy(path, broken)
        with h5py.File(broken, "r+") as handle:
            handle[f"models/{model.name}/{key}"][-1] *= 2
        with pytest.raises(ValueError, match=match):
            reader.validate(broken)
    broken = tmp_path/"rising.h5"
    shutil.copy(path, broken)
    with h5py.File(broken, "r+") as handle:
        handle[f"models/{model.name}/exterior_pn_column_lightcone"][-1] *= 2
    with pytest.raises(ValueError, match="increases"):
        reader.validate(broken)
    broken = tmp_path/"no_sampling.h5"
    shutil.copy(path, broken)
    with h5py.File(broken, "r+") as handle:
        del handle[f"models/{model.name}"].attrs["sampling"]
    with pytest.raises(ValueError, match="sampling"):
        reader.validate(broken)


def test_sample_times_union_and_deduplication():
    exporter = load_example("export_npc_models")
    t = exporter.sample_times(1.0, 3.0)
    assert t.size == 261 and np.all(np.diff(t) > 0)
    assert t[0] == 1.0 and np.any(np.isclose(t, 1.0+.99*2.0, rtol=1e-14, atol=0))
    np.testing.assert_allclose((3.0-t[-1])/2.0, 1e-4, rtol=1e-12)
    remaining = np.sort((3.0-t)/2.0)
    assert np.isclose(remaining[remaining < .01].min(), 1e-4)
    # a geometric time equal to a uniform one within 1e-12 relative is kept once
    original = exporter.GEOMETRIC_REMAINING
    try:
        exporter.GEOMETRIC_REMAINING = (.5, .01)  # .01 remaining = the 0.99 uniform sample
        t = exporter.sample_times(1.0, 3.0)
        assert t.size == 260 and np.all(np.diff(t) > 1e-12*t[1:])
    finally:
        exporter.GEOMETRIC_REMAINING = original


def test_reader_rejects_v3_schema(tmp_path):
    reader = load_example("read_npc_inputs")
    path = tmp_path/"v3.h5"
    with h5py.File(path, "w") as f:
        f.attrs["schema"] = "jetbns.npc-model-export.v3"
    with pytest.raises(ValueError, match="stale"):
        reader.validate(path)


def test_reader_rejects_pre_audit_schema(tmp_path):
    reader = load_example("read_npc_inputs")
    path = tmp_path/"old.h5"
    with h5py.File(path, "w") as f:
        f.attrs["schema"] = "jetbns.npc-model-export.v1"
    with pytest.raises(ValueError, match="stale"):
        reader.validate(path)


def test_resolution_failure_is_not_marked_as_passed(tmp_path):
    exporter = load_example("export_npc_models")
    model = exporter.ANALYTICAL[0]
    result = exporter.solve(model, tmp_path)
    # Independent perturbation must trip the acceptance gate.
    from dataclasses import replace
    bad = list(result)
    bad[-1] = replace(result[-1], upstream_density_g_cm3=np.ones_like(result[-1].time_s))
    with pytest.raises(RuntimeError, match="resolution check failed"):
        exporter.audit_model(result, tuple(bad))
