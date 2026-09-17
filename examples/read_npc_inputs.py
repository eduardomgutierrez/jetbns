"""Validate/read the transfer file. Requires only numpy and h5py, not jetbns."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

NPC_CONVERGENCE_FIELDS = (
    "breakout_time_s", "upstream_density_g_cm3", "electron_fraction",
    "relative_lorentz_factor", "neutron_to_proton_optical_depth",
    "proton_to_neutron_optical_depth", "breakout_condition",
)
COCOON_CONVERGENCE_FIELDS = (
    "cocoon_energy_erg", "cocoon_radius_cm",
    "cocoon_pressure_erg_cm3", "jet_cross_section_cm2",
)


def validate(path: str | Path) -> list[str]:
    """Reject stale schemas, malformed arrays, and inconsistent target depths."""
    with h5py.File(path) as handle:
        if handle.attrs.get("schema") != "jetbns.npc-model-export.v3":
            raise ValueError("Unsupported/stale export: regenerate with the audited pipeline")
        names = list(handle["models"])
        summary = handle["prebreakout_summary"]
        ordered = list(summary["name"].asstr()[:])
        if set(names) != set(ordered) or len(names) != len(ordered):
            raise ValueError("Model list and summary disagree")
        for name in names:
            g = handle["models"][name]
            t = g["time_s"][:]
            if t.ndim != 1 or t.size < 2 or np.any(np.diff(t) <= 0):
                raise ValueError(f"{name}: invalid time grid")
            if t[-1] >= g.attrs["breakout_time_s"]:
                raise ValueError(f"{name}: sample reaches or exceeds breakout")
            if not g["convergence"].attrs["passed"]:
                raise ValueError(f"{name}: unresolved numerical solution")
            convergence = g["convergence"].attrs
            npc_tolerance = g.attrs["npc_convergence_tolerance"]
            cocoon_tolerance = g.attrs["cocoon_convergence_tolerance"]
            if max(convergence[key] for key in NPC_CONVERGENCE_FIELDS) >= npc_tolerance:
                raise ValueError(f"{name}: NPC convergence tolerance exceeded")
            if max(convergence[key] for key in COCOON_CONVERGENCE_FIELDS) >= cocoon_tolerance:
                raise ValueError(f"{name}: cocoon convergence tolerance exceeded")
            if g.attrs.get("propagation_model") != "JetCocoon":
                raise ValueError(f"{name}: missing cocoon coupling")
            for key, ds in g["jet_cocoon"].items():
                if ds.shape != t.shape or np.any(~np.isfinite(ds[:])) or np.any(ds[:] < 0):
                    raise ValueError(f"{name}/{key}: invalid cocoon series")
                if "unit" not in ds.attrs or "frame" not in ds.attrs:
                    raise ValueError(f"{name}/{key}: missing cocoon unit/frame")
            for key, ds in g["snapshot_columns"].items():
                if not np.isfinite(ds[()]) or ds[()] < 0:
                    raise ValueError(f"{name}/{key}: invalid radial column")
            for key, ds in g.items():
                if not isinstance(ds, h5py.Dataset):
                    continue
                values = ds[:]
                if values.shape != t.shape or np.any(~np.isfinite(values)):
                    raise ValueError(f"{name}/{key}: nonfinite or misaligned data")
                if "unit" not in ds.attrs or "frame" not in ds.attrs:
                    raise ValueError(f"{name}/{key}: missing unit/frame")
                if np.any(values < 0):
                    raise ValueError(f"{name}/{key}: negative value")
            cfg = g["configuration"].attrs
            length = g["effective_path_length_upstream_cm"][:]
            if not np.allclose(length, g["path_length_cm"][:]/g["ambient_lorentz_factor"][:]):
                raise ValueError(f"{name}: effective length mismatch")
            for density, tau, rate in (
                ("proton_number_density_cm3", "neutron_to_proton_optical_depth",
                 "neutron_on_proton_collision_rate_s1"),
                ("free_neutron_number_density_cm3", "proton_to_neutron_optical_depth",
                 "proton_on_neutron_collision_rate_s1"),
            ):
                expected = g[density][:]*cfg["pn_cross_section_cm2"]
                if not np.allclose(g[tau][:], expected*length, rtol=1e-10, atol=0):
                    raise ValueError(f"{name}: target optical depth mismatch")
                if not np.allclose(g[rate][:], expected*2.99792458e10, rtol=1e-10, atol=0):
                    raise ValueError(f"{name}: collision rate mismatch")
            baryons = g["upstream_baryon_number_density_cm3"][:]
            if np.any(g["proton_number_density_cm3"][:]
                      + g["free_neutron_number_density_cm3"][:] > baryons*(1+1e-12)):
                raise ValueError(f"{name}: species exceed baryon budget")
            for key, ds in summary.items():
                if key != "name" and not np.isclose(ds[ordered.index(name)], g[key][-1]):
                    raise ValueError(f"{name}: summary differs from trajectory endpoint")
        return names


def snapshot(path: str | Path, model: str) -> dict[str, float]:
    """Return all endpoint scalars; call validate once before reading a new file."""
    with h5py.File(path) as handle:
        group = handle["models"][model]
        result = {key: float(ds[-1]) for key, ds in group.items()
                  if isinstance(ds, h5py.Dataset)}
        result.update({key: float(ds[()]) for key, ds in group["snapshot_columns"].items()})
        result.update({key: float(ds[-1]) for key, ds in group["jet_cocoon"].items()})
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", nargs="?", default="npc_prebreakout_models.h5")
    parser.add_argument("--model", help="print this model's snapshot as JSON")
    args = parser.parse_args()
    names = validate(args.file)
    if args.model:
        print(json.dumps(snapshot(args.file, args.model), indent=2, allow_nan=False))
    else:
        print(f"Validated {len(names)} models: {', '.join(names)}")


if __name__ == "__main__":
    main()
