"""Generate ten representative multipanel numerical-ejecta diagnostics."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from jetbns import ConstantEngine, JetHead, NumericalEjecta, evaluate_npc_inputs
from jetbns.constants import SPEED_OF_LIGHT


@dataclass(frozen=True)
class Scenario:
    """One representative point in the numerical parameter grid."""

    name: str
    cutoff_mode: str
    launch_time_s: float
    luminosity_iso_erg_s: float
    beta_width: float


REPRESENTATIVE_SCENARIOS = (
    Scenario("early_weak_narrow_sharp", "sharp", 0.1, 1e49, 0.02),
    Scenario("early_weak_narrow_smooth", "smooth", 0.1, 1e49, 0.02),
    Scenario("fiducial_sharp", "sharp", 0.3, 1e50, 0.035),
    Scenario("fiducial_smooth", "smooth", 0.3, 1e50, 0.035),
    Scenario("powerful_sharp", "sharp", 1.0, 1e51, 0.035),
    Scenario("powerful_smooth", "smooth", 1.0, 1e51, 0.035),
    Scenario("late_powerful_wide_sharp", "sharp", 3.0, 1e52, 0.05),
    Scenario("late_powerful_wide_smooth", "smooth", 3.0, 1e52, 0.05),
    Scenario("early_powerful_wide_smooth", "smooth", 0.1, 1e52, 0.05),
    Scenario("late_weak_narrow_smooth", "smooth", 3.0, 1e49, 0.02),
)


def solid_angle(path: Path, bin_name: str) -> float:
    """Infer the solid angle of a cell-centered polar HDF5 bin."""
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


def cumulative_optical_depth(radius: np.ndarray, density: np.ndarray) -> np.ndarray:
    """Integrate the default 0.16 cm^2/g opacity inward on a sampled grid."""
    increments = 0.16 * 0.5 * (density[:-1] + density[1:]) * np.diff(radius)
    return np.concatenate((np.cumsum(increments[::-1])[::-1], [0.0]))


def make_figure(
    profile: Path,
    bin_name: str,
    scenario: Scenario,
    *,
    integration_samples: int = 96,
    radial_samples: int = 180,
    time_step_s: float = 1e-2,
    max_duration_s: float = 12.0,
) -> tuple[plt.Figure, bool]:
    """Solve and plot one scenario; return its figure and breakout flag."""
    ejecta = NumericalEjecta.from_hdf5(
        profile,
        bin_name=bin_name,
        solid_angle_sr=solid_angle(profile, bin_name),
        beta_width=scenario.beta_width,
        kernel_shape=2.0,
        cutoff_mode=scenario.cutoff_mode,
        integration_samples=integration_samples,
    )
    engine = ConstantEngine.from_isotropic_equivalent(
        scenario.luminosity_iso_erg_s,
        launch_time_s=scenario.launch_time_s,
        launch_radius_cm=8.45e7,
        opening_angle_rad=np.deg2rad(10),
        lorentz_factor=100,
    )
    jet = JetHead(engine, ejecta, breakout_optical_depth_samples=max(32, radial_samples))
    trajectory = jet.propagate(
        max_time_s=scenario.launch_time_s + max_duration_s,
        time_step_s=time_step_s,
    )
    npc = evaluate_npc_inputs(trajectory, ejecta)

    figure, axes = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)
    profile_times = np.unique(
        [
            trajectory.time_s[0],
            trajectory.time_s[len(trajectory.time_s) // 2],
            trajectory.time_s[-1],
        ]
    )
    maximum_density = 0.0
    maximum_optical_depth = 0.0
    for time in profile_times:
        inner = ejecta.inner_radius(float(time)) * 1.001
        outer = ejecta.optical_depth_outer_radius(float(time)) * 0.999
        radius = np.geomspace(inner, outer, radial_samples)
        density = np.asarray(ejecta.density(radius, float(time)))
        velocity = np.asarray(ejecta.velocity(radius, float(time))) / SPEED_OF_LIGHT
        optical_depth = cumulative_optical_depth(radius, density)
        maximum_density = max(maximum_density, float(np.max(density)))
        maximum_optical_depth = max(maximum_optical_depth, float(np.max(optical_depth)))
        scaled_radius = radius / ejecta.outer_radius(float(time))
        label = f"t={time:.2g} s"
        axes[0, 0].plot(scaled_radius, density, label=label)
        axes[0, 1].plot(scaled_radius, velocity, label=label)
        axes[0, 2].plot(scaled_radius, optical_depth, label=label)

    for axis in axes[0]:
        axis.axvline(1, color="0.3", linestyle="--", linewidth=1, label=r"nominal $r_{max}$")
        axis.set_xscale("log")
        axis.set_xlabel(r"$r/r_{max}$")
    axes[0, 0].set(yscale="log", ylabel=r"density [g cm$^{-3}$]")
    axes[0, 1].set(ylabel=r"ejecta velocity $\beta$")
    axes[0, 2].set(yscale="log", ylabel=r"electron-scattering $\tau(r)$")
    axes[0, 0].set_ylim(maximum_density * 1e-8, maximum_density * 3)
    axes[0, 2].set_ylim(0.3, maximum_optical_depth * 3)
    axes[0, 2].axhline(1, color="tab:red", linestyle=":", linewidth=1)
    axes[0, 0].legend(fontsize=8)

    time = trajectory.time_s
    nominal_edge = np.asarray([ejecta.outer_radius(float(value)) for value in time])
    opacity_edge = np.asarray([ejecta.optical_depth_outer_radius(float(value)) for value in time])
    axes[1, 0].plot(time, trajectory.radius_cm, label="jet head")
    axes[1, 0].plot(time, nominal_edge, "--", label=r"nominal $r_{max}$")
    if scenario.cutoff_mode == "smooth":
        axes[1, 0].plot(time, opacity_edge, ":", label="tail integration edge")
    axes[1, 0].set(xlabel="time [s]", ylabel="radius [cm]", yscale="log")
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].plot(time, trajectory.head_beta, label=r"$\beta_h$")
    axes[1, 1].plot(time, trajectory.ambient_beta, label=r"$\beta_a$")
    axes[1, 1].set(xlabel="time [s]", ylabel="velocity / c")
    axes[1, 1].legend(fontsize=8)

    axes[1, 2].plot(npc.time_s, npc.pn_optical_depth, label=r"$\tau_{pn}$")
    axes[1, 2].plot(npc.time_s, npc.relative_lorentz_factor, label=r"$\Gamma_{rel}$")
    axes[1, 2].plot(npc.time_s, npc.gyration_parameter, label=r"$\xi(1)$")
    axes[1, 2].axhspan(0.1, 2, color="tab:green", alpha=0.12, label=r"target $\tau_{pn}$")
    axes[1, 2].set(xlabel="time [s]", ylabel="NPC diagnostics", yscale="log")
    axes[1, 2].legend(fontsize=8, ncol=2)

    status = "breakout" if trajectory.broke_out else "embedded at end"
    figure.suptitle(
        f"{scenario.name}: {scenario.cutoff_mode}, "
        rf"$t_{{launch}}={scenario.launch_time_s:g}$ s, "
        rf"$L_{{iso}}={scenario.luminosity_iso_erg_s:.0e}$ erg/s, "
        rf"$\Delta\beta={scenario.beta_width:g}$ ({status})"
    )
    return figure, trajectory.broke_out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path, help="numerical outflow HDF5 file")
    parser.add_argument("--bin-name", default="itheta=00000", help="polar angular-bin group")
    parser.add_argument("--output", type=Path, default=Path("examples/output/numerical_gallery"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for index, scenario in enumerate(REPRESENTATIVE_SCENARIOS, start=1):
        figure, broke_out = make_figure(args.profile, args.bin_name, scenario)
        destination = args.output / f"{index:02d}_{scenario.name}.png"
        figure.savefig(destination, dpi=160)
        plt.close(figure)
        print(f"wrote {destination} (breakout={broke_out})")


if __name__ == "__main__":
    main()
