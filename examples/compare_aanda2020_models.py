"""Reproduce the Appendix A.1 analytical jet-propagation test suite.

The configurations and reference values are transcribed from Table A.1 of
Salafia et al., *A&A* 636, A105 (2020),
https://www.aanda.org/articles/aa/pdf/2020/04/aa36335-19.pdf.

The paper uses a homologously expanding ambient density proportional to
``r**-3.5`` and quotes ``L_j`` as the luminosity of one jet.  The cleaned
library has no single-power-law profile with an inner divergent normalization,
so this script represents the profile with a normalized broken law (inner
index 2, outer index 3.5, break at half the maximum velocity).  This is a
documented, finite-inner-radius approximation to the paper setup.  Results
should therefore be used as a regression/physics sanity check, not as a claim
to reproduce the published simulations exactly.

Run from the repository root::

    python examples/compare_aanda2020_models.py

It prints a comparison table and writes a TSV report and a two-panel PNG in
``examples/output/aanda2020``.  The output files are generated artifacts and
are intentionally ignored by Git.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from jetbns import BrokenPowerLaw, Cocoon, ConstantEngine, JetCocoon


@dataclass(frozen=True)
class AppendixCase:
    name: str
    group: str
    mass_msun: float
    luminosity_erg_s: float
    opening_angle_deg: float
    launch_time_s: float
    max_beta: float
    reference_time_s: float | None
    reference_radius_1e9_cm: float | None
    reference_energy_1e49_erg: float | None = None


# Published values: time and radius for N14; time and cocoon energy for G18.
CASES = (
    AppendixCase("A", "G18", .01, 2e50, 10, .080, .20, .10, None, 1.5),
    AppendixCase("B", "G18", .01, 2e50, 10, .240, .20, .20, None, 2.0),
    AppendixCase("C", "G18", .01, 2e50, 10, .040, .40, .12, None, 1.5),
    AppendixCase("M-ref", "N14", .01, 2e50, 15, .050, .40, .23, 3.7),
    AppendixCase("M-L4", "N14", .01, 4e50, 15, .050, .40, .20, 3.2),
    AppendixCase("M-th30", "N14", .01, 2e50, 30, .050, .40, .63, 8.9),
    AppendixCase("M-th45", "N14", .01, 2e50, 45, .050, .40, 1.01, 13.0),
    AppendixCase("M-ti500", "N14", .01, 2e50, 15, .500, .40, .90, 17.5),
    AppendixCase("M-M3", "N14", .001, 2e50, 15, .050, .40, .11, 2.0),
    AppendixCase("M-M2-2", "N14", .02, 2e50, 15, .050, .40, .32, 5.0),
    AppendixCase("M-M1", "N14", .10, 2e50, 15, .050, .40, .75, 11.0),
)


def run_case(case: AppendixCase, *, time_step_s: float, max_time_s: float | None):
    """Run one case with the coupled jet/cocoon solver."""
    ejecta = BrokenPowerLaw(
        mass_msun=case.mass_msun,
        inner_index=2.0,
        outer_index=3.5,
        break_beta=.5 * case.max_beta,
        max_beta=case.max_beta,
        inner_radius_cm=1.0e7,
    )
    # Table A.1 calls this L_j: it is already one-sided, so do not use the
    # isotropic-equivalent constructor here.
    engine = ConstantEngine(
        luminosity_erg_s=case.luminosity_erg_s,
        launch_time_s=case.launch_time_s,
        launch_radius_cm=1.0e7,
        opening_angle_rad=np.deg2rad(case.opening_angle_deg),
        lorentz_factor=10.0,
        duration_s=max(2.0, (max_time_s or 2.0) + 1.0),
    )
    solver = JetCocoon(
        engine,
        ejecta,
        cocoon=Cocoon(density_samples=32),
        breakout_optical_depth_samples=96,
    )
    stop = max_time_s or max(2.0, case.launch_time_s + 1.2)
    trajectory = solver.propagate(max_time_s=stop, time_step_s=time_step_s)
    if not trajectory.broke_out:
        raise RuntimeError(f"{case.name} did not break out by {stop:g} s")
    return trajectory


def _fmt(value: float | None, precision: int = 3) -> str:
    return "—" if value is None else f"{value:.{precision}g}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("examples/output/aanda2020"))
    parser.add_argument(
        "--time-step", type=float, default=.002, help="Maximum integration step [s]"
    )
    parser.add_argument("--max-time", type=float, default=None, help="Common stopping time [s]")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | str | None]] = []
    for case in CASES:
        trajectory = run_case(case, time_step_s=args.time_step, max_time_s=args.max_time)
        rows.append({
            "name": case.name, "group": case.group,
            "model_time_s": trajectory.breakout_time_s,
            "published_time_s": case.reference_time_s,
            "model_radius_1e9_cm": trajectory.breakout_radius_cm / 1e9,
            "published_radius_1e9_cm": case.reference_radius_1e9_cm,
            "model_energy_1e49_erg": trajectory.cocoon_energy_erg[-1] / 1e49,
            "published_energy_1e49_erg": case.reference_energy_1e49_erg,
        })

    columns = tuple(rows[0])
    report = args.output_dir / "table_a1_comparison.tsv"
    with report.open("w", encoding="utf-8") as handle:
        handle.write("\t".join(columns) + "\n")
        for row in rows:
            handle.write("\t".join("" if row[c] is None else str(row[c]) for c in columns) + "\n")

    print("case      group  t_model  t_paper  r_model  r_paper  E_model  E_paper")
    for row in rows:
        print(f"{row['name']:8s} {row['group']:>4s}  {_fmt(row['model_time_s']):>7s}"
              f"  {_fmt(row['published_time_s']):>7s}  {_fmt(row['model_radius_1e9_cm']):>7s}"
              f"  {_fmt(row['published_radius_1e9_cm']):>7s}"
              f"  {_fmt(row['model_energy_1e49_erg']):>7s}"
              f"  {_fmt(row['published_energy_1e49_erg']):>7s}")

    names = [str(row["name"]) for row in rows]
    x = np.arange(len(rows))
    figure, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, constrained_layout=True)
    axes[0].plot(x, [row["model_time_s"] for row in rows], "o-", label="jetbns")
    axes[0].plot(
        x, [row["published_time_s"] for row in rows], "s--", label="Salafia et al. Table A.1"
    )
    axes[0].set_ylabel("breakout time [s]")
    axes[0].legend()
    axes[1].plot(x, [row["model_radius_1e9_cm"] for row in rows], "o-", label="jetbns")
    paper_radius = [
        np.nan if row["published_radius_1e9_cm"] is None else row["published_radius_1e9_cm"]
        for row in rows
    ]
    axes[1].plot(x, paper_radius, "s--", label="Salafia et al. Table A.1")
    axes[1].set_ylabel(r"breakout radius [$10^9$ cm]")
    axes[1].set_xticks(x, names, rotation=45, ha="right")
    axes[1].legend()
    figure.savefig(args.output_dir / "table_a1_comparison.png", dpi=180)
    plt.close(figure)
    print(f"\nWrote {report}")


if __name__ == "__main__":
    main()
