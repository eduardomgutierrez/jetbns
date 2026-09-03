"""Solve a jet trajectory and plot deterministic NPC Monte Carlo inputs."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from jetbns import (
    ConstantEngine,
    HomologousPowerLaw,
    JetHead,
    NpcConfig,
    evaluate_npc_inputs,
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    ejecta = HomologousPowerLaw(mass_msun=2.0e-3, max_beta=0.35)
    engine = ConstantEngine.from_isotropic_equivalent(
        5.0e51,
        launch_time_s=0.1,
        launch_radius_cm=8.45e7,
        opening_angle_rad=np.deg2rad(6.8),
        lorentz_factor=10.0,
    )
    trajectory = JetHead(engine, ejecta).propagate(max_time_s=3, time_step_s=2e-4)
    config = NpcConfig(path_length="radius")
    inputs = evaluate_npc_inputs(trajectory, ejecta, config=config)

    figure, axes = plt.subplots(3, 3, figsize=(13, 10), constrained_layout=True)
    axes[0, 0].plot(inputs.time_s, inputs.relative_lorentz_factor)
    axes[0, 0].set(ylabel=r"relative Lorentz factor $\Gamma_{\rm rel}$")
    axes[0, 1].plot(inputs.time_s, inputs.pn_optical_depth)
    axes[0, 1].set_yscale("log")
    axes[0, 1].set(ylabel=r"optical depth $\tau_{pn}$")
    axes[0, 2].plot(inputs.time_s, inputs.gyration_parameter)
    axes[0, 2].set_yscale("log")
    axes[0, 2].set(ylabel=r"gyration parameter $\xi(1)$")
    axes[1, 0].plot(inputs.time_s, inputs.upstream_number_density_cm3)
    axes[1, 0].set_yscale("log")
    axes[1, 0].set(ylabel=r"number density [cm$^{-3}$]")
    axes[1, 1].plot(inputs.time_s, inputs.upstream_magnetic_field_g)
    axes[1, 1].set_yscale("log")
    axes[1, 1].set(ylabel="magnetic field [G]")
    axes[1, 2].plot(inputs.time_s, inputs.downstream_temperature_kev)
    axes[1, 2].set(ylabel=r"$k_B T_d$ [keV]")
    axes[2, 0].plot(inputs.time_s, inputs.max_lorentz_factor_bethe_heitler)
    axes[2, 0].set_yscale("log")
    axes[2, 0].set(ylabel="Bethe--Heitler limit")
    axes[2, 1].plot(inputs.time_s, inputs.max_lorentz_factor_gyration)
    axes[2, 1].set_yscale("log")
    axes[2, 1].set(ylabel="gyration limit")
    axes[2, 2].plot(inputs.time_s, inputs.max_observer_energy_erg)
    axes[2, 2].set_yscale("log")
    axes[2, 2].set(ylabel="maximum observer energy [erg]")
    for axis in axes.flat:
        axis.set_xlabel("time after merger [s]")

    output = root / "examples" / "output"
    output.mkdir(exist_ok=True)
    figure.savefig(output / "npc_inputs.png", dpi=180)
    inputs.to_hdf5(
        output / "npc_inputs.h5",
        config=config,
        metadata={"ejecta_model": type(ejecta).__name__},
    )
    print(f"wrote {output / 'npc_inputs.png'}")
    print(f"wrote {output / 'npc_inputs.h5'}")


if __name__ == "__main__":
    main()
