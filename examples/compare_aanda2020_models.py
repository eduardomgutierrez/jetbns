"""Reproduce the Appendix A.1 analytical jet-propagation test suite.

The configurations and reference values are transcribed from Table A.1 of
Salafia et al., *A&A* 636, A105 (2020), arXiv:1907.07599,
https://www.aanda.org/articles/aa/pdf/2020/04/aa36335-19.pdf.
The "sim" column holds the simulation values (Gottlieb, Nakar & Piran 2018,
G18, MNRAS 473, 576; Nagakura et al. 2014, N14, ApJL 784, L28) and the
"model" column the semi-analytic model of Salafia et al.  M-th45 has no
simulation value: that run did not break out within the simulated window, so
only the published model value is listed.

Set-up (audit of 2026-11-05, variant "b"):

* Ambient medium: ``HomologousShell``, rho proportional to r**-3.5 between an
  inner edge and r_max(t) = r_max,ref + beta_max c (t - t_ref), normalised to
  the ejecta mass over 4 pi.  N14 Eqs. 1-5: r_max,ref = 1.3e8 cm at t_ref =
  10 ms, shell inner edge and jet base at the escape radius
  r_esc = (2 G M_c r_max^2 / v_max^2)**(1/3) with M_c = 2.7 Msun, both
  evaluated at the injection time.  G18 Sect. 2: the same law at t = 0 between
  4.5e7 cm and 1.3e8 cm, jet base at 4.5e7 cm; the wind injected after t = 0 is
  neglected (the region between the jet base and the shell is empty).
* Jet: one-sided L_j (Table A.1), the papers' terminal Lorentz factors (G18 80,
  N14 100), constant engine launched at the injection time.
* Solver: JetCocoon defaults (calibration 0.65, breakout where
  tau = 1/beta'_s with 0.16 cm**2/g), time step 1 ms.
* Times: Table A.1 lists breakout times since the jet injection, so the model
  time reported here is t_bo - t_inj (the trajectory itself is in lab seconds
  after the merger).

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

from jetbns import ConstantEngine, HomologousShell, JetCocoon
from jetbns.constants import SOLAR_MASS, SPEED_OF_LIGHT


G_NEWTON = 6.67430e-8          # cm^3 g^-1 s^-2 (CODATA 2018)
N14_REMNANT_MASS_MSUN = 2.7    # N14 Sect. 2, M_c in Eq. 5
N14_T0_S = 0.010               # N14 Eq. 2: snapshot time of r_max,0
RMAX0_CM = 1.3e8               # N14 Eq. 2 (at t0 = 10 ms); G18 Sect. 2 (at t = 0)
G18_RESC_CM = 4.5e7            # G18 Sect. 2: inner radius of the wind at t = 0 and jet base
GAMMA_JET = {"G18": 80.0, "N14": 100.0}   # terminal Lorentz factors of the papers
PROPAGATION_WINDOW_S = 10.0    # stop a run this long after injection if the head has not broken out
DEFAULT_STEP_S = 1e-3


@dataclass(frozen=True)
class AppendixCase:
    """One Table A.1 configuration.  Times in s (breakout times since injection), lengths in cm, energies in
    erg; L_j is one-sided and ``launch_time_s`` is the lab time since the merger."""

    name: str
    group: str
    mass_msun: float
    luminosity_erg_s: float
    opening_angle_deg: float
    launch_time_s: float
    max_beta: float
    sim_time_s: float | None
    model_time_s: float | None
    sim_radius_cm: float | None = None
    model_radius_cm: float | None = None
    sim_energy_erg: float | None = None
    model_energy_erg: float | None = None


# Published values (Table A.1): time and radius for N14; time and cocoon energy for G18.
CASES = (
    AppendixCase("A", "G18", .01, 2e50, 10, .080, .20, .10, .10, 1.4e9, None, 1.5e49, 1.4e49),
    AppendixCase("B", "G18", .01, 2e50, 10, .240, .20, .20, .18, 3.0e9, None, 2.0e49, 2.2e49),
    AppendixCase("C", "G18", .01, 2e50, 10, .040, .40, .12, .14, 2.5e9, None, 1.5e49, 1.7e49),
    AppendixCase("M-ref", "N14", .01, 2e50, 15, .050, .40, .23, .24, 3.7e9, 3.5e9),
    AppendixCase("M-L4", "N14", .01, 4e50, 15, .050, .40, .20, .18, 3.2e9, 2.7e9),
    AppendixCase("M-th30", "N14", .01, 2e50, 30, .050, .40, .63, .55, 8.9e9, 7.2e9),
    AppendixCase("M-th45", "N14", .01, 2e50, 45, .050, .40, None, 1.01, None, 13e9),
    AppendixCase("M-ti500", "N14", .01, 2e50, 15, .500, .40, .90, .69, 17.5e9, 14e9),
    AppendixCase("M-M3", "N14", .001, 2e50, 15, .050, .40, .11, .10, 2.0e9, 1.7e9),
    AppendixCase("M-M2-2", "N14", .02, 2e50, 15, .050, .40, .32, .32, 5.0e9, 4.4e9),
    AppendixCase("M-M1", "N14", .10, 2e50, 15, .050, .40, .75, .64, 11.0e9, 8.2e9),
)


def n14_outer_radius(time_s: float, max_beta: float) -> float:
    """N14 Eq. 2: r_max(t) = v_max (t - t0) + r_max0, in cm."""
    return max_beta * SPEED_OF_LIGHT * (time_s - N14_T0_S) + RMAX0_CM


def n14_escape_radius(injection_time_s: float, max_beta: float) -> float:
    """N14 Eq. 5: r_esc = (2 G M_c r_max^2 / v_max^2)^(1/3) at t_i, in cm."""
    r_max = n14_outer_radius(injection_time_s, max_beta)
    return (2 * G_NEWTON * N14_REMNANT_MASS_MSUN * SOLAR_MASS * r_max**2
            / (max_beta * SPEED_OF_LIGHT) ** 2) ** (1 / 3)


def ambient_medium(case: AppendixCase) -> HomologousShell:
    """The ambient medium of the simulation paper of `case`."""
    if case.group == "N14":
        r_esc = n14_escape_radius(case.launch_time_s, case.max_beta)
        r_max = n14_outer_radius(case.launch_time_s, case.max_beta)
        return HomologousShell(case.mass_msun, case.max_beta, N14_T0_S, RMAX0_CM, r_esc / r_max, r_esc)
    return HomologousShell(case.mass_msun, case.max_beta, 0.0, RMAX0_CM, G18_RESC_CM / RMAX0_CM, G18_RESC_CM)


def run_case(case: AppendixCase, *, time_step_s: float = DEFAULT_STEP_S, max_time_s: float | None = None):
    """Run one case with the coupled jet/cocoon solver (`max_time_s` is a lab time since the merger)."""
    ejecta = ambient_medium(case)
    # Table A.1 calls this L_j: it is already one-sided, so do not use the
    # isotropic-equivalent constructor here.
    engine = ConstantEngine(
        luminosity_erg_s=case.luminosity_erg_s,
        launch_time_s=case.launch_time_s,
        launch_radius_cm=ejecta.floor_radius_cm,
        opening_angle_rad=np.deg2rad(case.opening_angle_deg),
        lorentz_factor=GAMMA_JET[case.group],
    )
    solver = JetCocoon(engine, ejecta)
    stop = max_time_s or case.launch_time_s + PROPAGATION_WINDOW_S
    trajectory = solver.propagate(max_time_s=stop, time_step_s=time_step_s)
    if not trajectory.broke_out:
        raise RuntimeError(f"{case.name} did not break out by {stop:g} s")
    return trajectory


def _fmt(value: float | None, precision: int = 3) -> str:
    return "—" if value is None else f"{value:.{precision}g}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output-dir", type=Path, default=Path("examples/output/aanda2020"))
    parser.add_argument(
        "--time-step", type=float, default=DEFAULT_STEP_S, help="Maximum integration step [s]"
    )
    parser.add_argument("--max-time", type=float, default=None,
                        help="Common stopping time [s, lab time since the merger]")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | str | None]] = []
    for case in CASES:
        trajectory = run_case(case, time_step_s=args.time_step, max_time_s=args.max_time)
        rows.append({
            "name": case.name, "group": case.group,
            "jetbns_time_s": trajectory.breakout_time_s - case.launch_time_s,   # since injection
            "sim_time_s": case.sim_time_s,
            "model_time_s": case.model_time_s,
            "jetbns_radius_1e9_cm": trajectory.breakout_radius_cm / 1e9,
            "sim_radius_1e9_cm": None if case.sim_radius_cm is None else case.sim_radius_cm / 1e9,
            "model_radius_1e9_cm": None if case.model_radius_cm is None else case.model_radius_cm / 1e9,
            "jetbns_energy_1e49_erg": trajectory.cocoon_energy_erg[-1] / 1e49,
            "sim_energy_1e49_erg": None if case.sim_energy_erg is None else case.sim_energy_erg / 1e49,
            "model_energy_1e49_erg": None if case.model_energy_erg is None else case.model_energy_erg / 1e49,
        })

    columns = tuple(rows[0])
    report = args.output_dir / "table_a1_comparison.tsv"
    with report.open("w", encoding="utf-8") as handle:
        handle.write("\t".join(columns) + "\n")
        for row in rows:
            handle.write("\t".join("" if row[c] is None else str(row[c]) for c in columns) + "\n")

    print("t = t_bo - t_inj [s], r = breakout radius [1e9 cm], E = cocoon energy per jet [1e49 erg]")
    print("case      group  t_jetbns  t_sim  t_model  r_jetbns  r_sim  r_model  E_jetbns  E_sim  E_model")
    for row in rows:
        print(f"{row['name']:8s} {row['group']:>4s}"
              + "".join(f"  {_fmt(row[f'{who}_{q}']):>7s}"
                        for q in ("time_s", "radius_1e9_cm", "energy_1e49_erg")
                        for who in ("jetbns", "sim", "model")))

    names = [str(row["name"]) for row in rows]
    x = np.arange(len(rows))
    nan = lambda key: [np.nan if row[key] is None else row[key] for row in rows]
    figure, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, constrained_layout=True)
    for ax, q, label in ((axes[0], "time_s", r"$t_{\rm bo}-t_{\rm inj}$ [s]"),
                         (axes[1], "radius_1e9_cm", r"breakout radius [$10^9$ cm]")):
        ax.plot(x, nan(f"jetbns_{q}"), "o-", label="jetbns")
        ax.plot(x, nan(f"sim_{q}"), "s--", mfc="none", label="simulation (Table A.1)")
        ax.plot(x, nan(f"model_{q}"), "x:", label="Salafia et al. model (Table A.1)")
        ax.set_ylabel(label)
        ax.legend()
    axes[1].set_xticks(x, names, rotation=45, ha="right")
    figure.savefig(args.output_dir / "table_a1_comparison.png", dpi=180)
    plt.close(figure)
    print(f"\nWrote {report}")


if __name__ == "__main__":
    main()
