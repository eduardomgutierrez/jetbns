"""Explore which jet/ejecta parameters approach the NPC-compatible regime.

This is a deliberately broad screening calculation, not an inference grid.  It
uses a coarse integration step and writes one representative (closest-to-NPC)
sample per trajectory to HDF5, plus a tau--xi diagnostic plot.
"""

from itertools import product
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from jetbns import BrokenPowerLaw, ConstantEngine, JetHead, evaluate_npc_inputs

MASSES_MSUN = np.array([1e-5, 1e-4, 1e-3, 1e-2])
LAUNCH_TIMES_S = np.array([0.1, 0.3, 1.0, 3.0])
LUMINOSITIES_ISO = np.array([1e49, 1e50, 1e51, 1e52, 1e53])
TAIL_EXPONENTS = np.array([2.0, 4.0, 8.0, 16.0])


def npc_distance(gamma: np.ndarray, tau: np.ndarray, xi: np.ndarray) -> np.ndarray:
    """Logarithmic distance from Gamma_rel>=2, 0.1<=tau<=2, and xi>=1."""
    gamma_penalty = np.maximum(0, np.log10(2 / gamma))
    tau_penalty = np.maximum(0, np.log10(0.1 / tau)) + np.maximum(0, np.log10(tau / 2))
    xi_penalty = np.maximum(0, -np.log10(xi))
    return gamma_penalty + tau_penalty + xi_penalty


def main() -> None:
    rows = []
    for mass, launch, luminosity, exponent in product(
        MASSES_MSUN, LAUNCH_TIMES_S, LUMINOSITIES_ISO, TAIL_EXPONENTS
    ):
        ejecta = BrokenPowerLaw(
            mass_msun=float(mass),
            break_beta=0.3,
            max_beta=0.6,
            tail=True,
            tail_exponent=float(exponent),
            # Keeps this finite numerical boundary subluminal. See the README.
            tail_extent=1.1,
        )
        engine = ConstantEngine.from_isotropic_equivalent(
            float(luminosity),
            launch_time_s=float(launch),
            launch_radius_cm=8.45e7,
            opening_angle_rad=np.deg2rad(10),
            lorentz_factor=100,
        )
        trajectory = JetHead(engine, ejecta).propagate(
            max_time_s=float(launch + 12), time_step_s=5e-3
        )
        data = evaluate_npc_inputs(trajectory, ejecta)
        distance = npc_distance(
            data.relative_lorentz_factor, data.pn_optical_depth, data.gyration_parameter
        )
        index = int(np.argmin(distance))
        rows.append(
            (
                mass,
                launch,
                luminosity,
                exponent,
                trajectory.broke_out,
                data.time_s[index],
                data.radius_cm[index],
                data.relative_lorentz_factor[index],
                data.pn_optical_depth[index],
                data.gyration_parameter[index],
                distance[index],
            )
        )

    names = (
        "mass_msun",
        "launch_time_s",
        "luminosity_iso_erg_s",
        "tail_exponent",
        "broke_out",
        "sample_time_s",
        "sample_radius_cm",
        "relative_lorentz_factor",
        "pn_optical_depth",
        "gyration_parameter",
        "npc_distance",
    )
    table = np.asarray(rows)
    output = Path(__file__).resolve().parent / "output"
    output.mkdir(exist_ok=True)
    with h5py.File(output / "npc_parameter_sweep.h5", "w") as handle:
        handle.attrs["description"] = "Closest-to-NPC sample from each screening trajectory"
        handle.attrs["tail_extent"] = 1.1
        handle.attrs["time_step_s"] = 5e-3
        for index, name in enumerate(names):
            handle.create_dataset(name, data=table[:, index])

    figure, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
    xi_upper = table[:, 9].max() * 3
    points = axis.scatter(
        table[:, 8], table[:, 9], c=table[:, 7], s=18 + 2 * table[:, 3], cmap="viridis"
    )
    axis.axvspan(0.1, 2, color="tab:green", alpha=0.12, label=r"target $\tau_{pn}$")
    axis.axhspan(1, xi_upper, color="tab:orange", alpha=0.08, label=r"target $\xi(1)$")
    axis.set(
        xscale="log",
        yscale="log",
        xlabel=r"$\tau_{pn}$",
        ylabel=r"$\xi(1)$",
        title="Closest NPC conditions in 320 jet/ejecta scenarios",
    )
    axis.set_ylim(0.3, xi_upper)
    figure.colorbar(points, ax=axis, label=r"$\Gamma_{\rm rel}$")
    axis.legend()
    figure.savefig(output / "npc_parameter_sweep.png", dpi=180)

    compatible = (table[:, 7] > 2) & (table[:, 8] >= 0.1) & (table[:, 8] <= 2) & (table[:, 9] > 1)
    best = table[np.argmin(table[:, 10])]
    print(f"evaluated {len(table)} scenarios; fully compatible: {compatible.sum()}")
    print(dict(zip(names, best, strict=True)))
    print(f"wrote {output / 'npc_parameter_sweep.h5'}")
    print(f"wrote {output / 'npc_parameter_sweep.png'}")


if __name__ == "__main__":
    main()
