"""Run and plot a minimal engine--ejecta--jet-head calculation."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from jetbns import ConstantEngine, HomologousPowerLaw, JetHead
from jetbns.constants import SPEED_OF_LIGHT


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    ejecta = HomologousPowerLaw(
        mass_msun=2.0e-3,
        density_index=2.0,
        max_beta=0.35,
        initial_outer_radius_cm=1.0e9,
    )
    engine = ConstantEngine.from_isotropic_equivalent(
        5.0e51,
        launch_time_s=0.1,
        launch_radius_cm=8.45e7,
        opening_angle_rad=np.deg2rad(6.8),
        lorentz_factor=10.0,
    )
    result = JetHead(engine, ejecta).propagate(max_time_s=3.0, time_step_s=2.0e-4)

    outer_radius = np.array([ejecta.outer_radius(time) for time in result.time_s])
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].plot(result.time_s, result.radius_cm / 1e9, label="jet head")
    axes[0].plot(result.time_s, outer_radius / 1e9, "--", label="ejecta edge")
    axes[0].set(xlabel="time after merger [s]", ylabel=r"radius [$10^9$ cm]")
    axes[0].legend()
    axes[1].plot(result.time_s, result.head_beta, label="jet head")
    axes[1].plot(result.time_s, result.ambient_beta, label="local ejecta")
    axes[1].axhline(engine.beta, color="black", linestyle=":", label="unshocked jet")
    axes[1].set(xlabel="time after merger [s]", ylabel=r"velocity $v/c$", ylim=(0, 1.05))
    axes[1].legend()
    output = root / "examples" / "output"
    output.mkdir(exist_ok=True)
    figure.savefig(output / "jet_head_propagation.png", dpi=180)
    status = "breakout" if result.broke_out else "no breakout"
    print(
        f"{status} at t={result.time_s[-1]:.4f} s, "
        f"r={result.radius_cm[-1]:.4e} cm, "
        f"r/c={result.radius_cm[-1] / SPEED_OF_LIGHT:.4f} s"
    )
    print(f"wrote {output / 'jet_head_propagation.png'}")


if __name__ == "__main__":
    main()
