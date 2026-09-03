"""Screen NPC conditions using one or more numerical ejecta HDF5 profiles."""

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from explore_npc_parameter_space import npc_distance

from jetbns import ConstantEngine, JetHead, NumericalEjecta, evaluate_npc_inputs

LAUNCH_TIMES_S = (0.1, 0.3, 1.0, 3.0)
LUMINOSITIES_ISO = (1e49, 1e50, 1e51, 1e52)
BETA_WIDTHS = (0.02, 0.035, 0.05)


def solid_angle(path: Path, bin_name: str) -> float:
    """Infer a cell-centered polar bin's solid angle from the HDF5 theta grid."""
    with h5py.File(path) as handle:
        theta = np.asarray(handle["theta"], dtype=float)
        names = sorted(name for name in handle if name.startswith("itheta="))
    index = names.index(bin_name)
    if theta.ndim != 1 or theta.size != len(names) or theta.size < 2:
        raise ValueError(f"{path}: cannot infer solid angle from theta grid")
    spacing = theta[1] - theta[0]
    return float(
        2 * np.pi * (np.cos(theta[index] - spacing / 2) - np.cos(theta[index] + spacing / 2))
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profiles", nargs="+", type=Path, help="numerical outflow HDF5 files")
    parser.add_argument("--bin-name", default="itheta=00000", help="polar angular-bin group")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[tuple[object, ...]] = []
    failures = 0
    for profile in args.profiles:
        omega = solid_angle(profile, args.bin_name)
        for width in BETA_WIDTHS:
            ejecta = NumericalEjecta.from_hdf5(
                profile,
                bin_name=args.bin_name,
                extraction_radius_cm=4.42e7,
                solid_angle_sr=omega,
                beta_width=width,
                integration_samples=96,
            )
            for launch in LAUNCH_TIMES_S:
                for luminosity in LUMINOSITIES_ISO:
                    engine = ConstantEngine.from_isotropic_equivalent(
                        luminosity,
                        launch_time_s=launch,
                        launch_radius_cm=8.45e7,
                        opening_angle_rad=np.deg2rad(10),
                        lorentz_factor=100,
                    )
                    try:
                        trajectory = JetHead(engine, ejecta).propagate(
                            max_time_s=launch + 12, time_step_s=1e-2
                        )
                        data = evaluate_npc_inputs(trajectory, ejecta)
                    except ValueError as error:
                        failures += 1
                        print(f"skipped {profile.name}, t={launch}, L={luminosity:g}: {error}")
                        continue
                    distance = npc_distance(
                        data.relative_lorentz_factor,
                        data.pn_optical_depth,
                        data.gyration_parameter,
                    )
                    index = int(np.argmin(distance))
                    rows.append(
                        (
                            str(profile), launch, luminosity, width, trajectory.broke_out,
                            data.time_s[index], data.radius_cm[index],
                            data.relative_lorentz_factor[index], data.pn_optical_depth[index],
                            data.gyration_parameter[index],
                            data.downstream_temperature_kev[index], distance[index],
                        )
                    )
    if not rows:
        raise RuntimeError("no numerical trajectories could be evaluated")

    names = (
        "profile", "launch_time_s", "luminosity_iso_erg_s", "beta_width", "broke_out",
        "sample_time_s", "sample_radius_cm", "relative_lorentz_factor",
        "pn_optical_depth", "gyration_parameter", "downstream_temperature_kev",
        "npc_distance",
    )
    numeric = np.asarray([row[1:] for row in rows], dtype=float)
    output = Path(__file__).resolve().parent / "output"
    output.mkdir(exist_ok=True)
    with h5py.File(output / "npc_numerical_sweep.h5", "w") as handle:
        handle.attrs["description"] = "Closest-to-NPC sample for numerical ejecta runs"
        handle.attrs["bin_name"] = args.bin_name
        handle.attrs["failed_runs"] = failures
        text_type = h5py.string_dtype("utf-8")
        handle.create_dataset("profile", data=[row[0] for row in rows], dtype=text_type)
        for index, name in enumerate(names[1:]):
            handle.create_dataset(name, data=numeric[:, index])

    figure, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
    points = axis.scatter(numeric[:, 7], numeric[:, 8], c=numeric[:, 6], cmap="viridis")
    axis.axvspan(0.1, 2, color="tab:green", alpha=0.12, label=r"target $\tau_{pn}$")
    axis.axhspan(1, numeric[:, 8].max() * 3, color="tab:orange", alpha=0.08,
                 label=r"target $\xi(1)$")
    axis.set(xscale="log", yscale="log", xlabel=r"$\tau_{pn}$", ylabel=r"$\xi(1)$",
             title=f"Numerical ejecta NPC screening: {len(rows)} trajectories")
    axis.set_ylim(0.3, numeric[:, 8].max() * 3)
    figure.colorbar(points, ax=axis, label=r"$\Gamma_{\rm rel}$")
    axis.legend()
    figure.savefig(output / "npc_numerical_sweep.png", dpi=180)

    compatible = (numeric[:, 6] > 2) & (numeric[:, 7] >= 0.1) & (
        numeric[:, 7] <= 2
    ) & (numeric[:, 8] > 1)
    best = int(np.argmin(numeric[:, 10]))
    print(f"evaluated {len(rows)} trajectories; failed: {failures}; compatible: {compatible.sum()}")
    print(dict(zip(names, rows[best], strict=True)))
    print(f"wrote {output / 'npc_numerical_sweep.h5'}")
    print(f"wrote {output / 'npc_numerical_sweep.png'}")


if __name__ == "__main__":
    main()
