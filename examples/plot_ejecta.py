"""Plot density and velocity profiles for the three initial ejecta models."""

import matplotlib.pyplot as plt
import numpy as np

from jetbns import BrokenPowerLaw, HomologousPowerLaw, NumericalEjecta, OutflowHistory
from jetbns.constants import SPEED_OF_LIGHT


def main() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    time = 0.2
    models = {
        "single power law": HomologousPowerLaw(),
        "broken + tail": BrokenPowerLaw(tail=True),
        "numerical example": NumericalEjecta(
            OutflowHistory(
                time_s=np.array([0.005, 0.01, 0.02, 0.04, 0.08]),
                velocity_c=np.array([0.42, 0.38, 0.34, 0.29, 0.24]),
                mass_loss_rate_g_s=np.array([3e31, 2.5e31, 1.8e31, 1.1e31, 5e30]),
                electron_fraction=np.array([0.08, 0.09, 0.10, 0.12, 0.15]),
            )
        ),
    }
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for label, model in models.items():
        radius = np.geomspace(model.inner_radius(time) * 1.001, model.outer_radius(time), 300)
        axes[0].loglog(radius, model.density(radius, time), label=label)
        axes[1].semilogx(radius, model.velocity(radius, time) / SPEED_OF_LIGHT, label=label)
    axes[0].set(xlabel="radius [cm]", ylabel=r"density [g cm$^{-3}$]")
    # The exponential tail is evaluated down to machine-small densities. Those
    # values are numerically valid but visually compress the useful profiles.
    axes[0].set_ylim(1.0, 3.0e8)
    axes[1].set(xlabel="radius [cm]", ylabel=r"velocity $v/c$")
    axes[1].set_ylim(0, 0.7)
    axes[0].legend()
    output = root / "examples" / "output"
    output.mkdir(exist_ok=True)
    figure.savefig(output / "ejecta_profiles.png", dpi=180)
    print(f"wrote {output / 'ejecta_profiles.png'}")


if __name__ == "__main__":
    main()
