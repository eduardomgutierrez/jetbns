"""The portable reader validates physics relations without importing jetbns."""
import importlib.util
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
    snap = reader.snapshot(path, model.name)
    assert snap["proton_number_density_cm3"] == fine[-1].proton_number_density_cm3[-1]
    assert snap["neutron_on_proton_radial_column_depth"] < snap["neutron_to_proton_optical_depth"]
    with h5py.File(path, "r+") as handle:
        handle[f"models/{model.name}/neutron_to_proton_optical_depth"][-1] *= 2
    with pytest.raises(ValueError, match="target optical depth"):
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
