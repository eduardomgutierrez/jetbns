"""Export representative breakout models as portable NPC inputs."""

from __future__ import annotations

import shutil
import subprocess
import zipfile
from dataclasses import dataclass, fields
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from plot_numerical_parameter_gallery import solid_angle

from jetbns import (
    BrokenPowerLaw,
    ConstantEngine,
    JetHead,
    NpcConfig,
    NumericalEjecta,
    evaluate_npc_inputs,
)
from jetbns.constants import SPEED_OF_LIGHT
from jetbns.npc import NPC_UNITS

ROOT = Path("/home/emgutierrez/codes")


@dataclass(frozen=True)
class Model:
    name: str
    kind: str
    launch_s: float
    luminosity_iso: float
    profile: str = ""
    width: float = np.nan
    mass_msun: float = np.nan
    tail_exponent: float = np.nan


MODELS = (
    Model(
        "num_dd2_equal",
        "numerical",
        3,
        1e51,
        str(
            ROOT
            / "outflow_files_jet_propagation/Archive/wdir/dd2_analysis"
            / "DD2_M135135_M1_K2_SR/outflow_1_ber_out/lagrangian_profile.h5"
        ),
        0.05,
    ),
    Model(
        "num_dd2_asymmetric",
        "numerical",
        1,
        1e51,
        str(
            ROOT
            / "jet_propagation/jet_propagation/profile_files/new"
            / "DD2_M180_108_45km_M1_LK_SR/outflow_1_ber_out_r/lagrangian_profile.h5"
        ),
        0.05,
    ),
    Model(
        "num_blh",
        "numerical",
        3,
        1e52,
        str(
            ROOT
            / "jet_propagation/jet_propagation/profile_files/new"
            / "BLh_M11461635_M1_K2_SR/outflow_1_ber_out/lagrangian_profile.h5"
        ),
        0.05,
    ),
    Model(
        "num_sfho",
        "numerical",
        3,
        1e51,
        str(
            ROOT
            / "jet_propagation/jet_propagation/profile_files/new"
            / "SFHo_M135-135_M1_SR/outflow_1_ber_hmin_out/lagrangian_profile.h5"
        ),
        0.05,
    ),
    Model(
        "num_sly",
        "numerical",
        1,
        1e50,
        str(
            ROOT
            / "jet_propagation/jet_propagation/profile_files/new"
            / "SLy_M145-125_M1_SR_v2/outflow_1_ber_hmin_out/lagrangian_profile.h5"
        ),
        0.05,
    ),
    Model("ana_low_mass_l50", "analytical", 3, 1e50, mass_msun=1e-5, tail_exponent=16),
    Model("ana_low_mass_l51", "analytical", 3, 1e51, mass_msun=1e-5, tail_exponent=16),
    Model("ana_low_mass_l52", "analytical", 3, 1e52, mass_msun=1e-5, tail_exponent=16),
    Model("ana_broader_tail", "analytical", 3, 1e50, mass_msun=1e-5, tail_exponent=8),
    Model("ana_higher_mass", "analytical", 3, 1e51, mass_msun=1e-4, tail_exponent=16),
)


def build(model: Model):
    if model.kind == "numerical":
        path = Path(model.profile)
        return NumericalEjecta.from_hdf5(
            path,
            bin_name="itheta=00000",
            solid_angle_sr=solid_angle(path, "itheta=00000"),
            beta_width=model.width,
            kernel_shape=2,
            cutoff_mode="smooth",
            integration_samples=128,
        )
    return BrokenPowerLaw(
        mass_msun=model.mass_msun,
        break_beta=0.3,
        max_beta=0.6,
        tail=True,
        tail_exponent=model.tail_exponent,
        tail_extent=1.1,
    )


def solve(model: Model):
    ejecta = build(model)
    engine = ConstantEngine.from_isotropic_equivalent(
        model.luminosity_iso,
        launch_time_s=model.launch_s,
        launch_radius_cm=8.45e7,
        opening_angle_rad=np.deg2rad(10),
        lorentz_factor=100,
    )
    jet = JetHead(engine, ejecta, breakout_optical_depth_samples=256)
    trajectory = jet.propagate(max_time_s=model.launch_s + 20, time_step_s=5e-3)
    if not trajectory.broke_out:
        raise RuntimeError(f"selected model did not break out: {model.name}")
    config = NpcConfig(path_length="radius")
    return ejecta, jet, trajectory, config, evaluate_npc_inputs(
        trajectory, ejecta, config=config
    )


def plot_model(model: Model, ejecta, trajectory, npc, destination: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for time in (
        trajectory.time_s[0],
        trajectory.time_s[len(trajectory.time_s) // 2],
        trajectory.time_s[-1],
    ):
        radius = np.geomspace(
            ejecta.inner_radius(float(time)) * 1.001,
            ejecta.optical_depth_outer_radius(float(time)) * 0.999,
            220,
        )
        density = np.asarray(ejecta.density(radius, float(time)))
        beta = np.asarray(ejecta.velocity(radius, float(time))) / SPEED_OF_LIGHT
        x = radius / ejecta.outer_radius(float(time))
        axes[0, 0].plot(x, density, label=f"t={time:.2g} s")
        axes[0, 1].plot(x, beta, label=f"t={time:.2g} s")
    positive = np.concatenate([line.get_ydata() for line in axes[0, 0].lines])
    top = positive.max()
    axes[0, 0].set(
        xscale="log",
        yscale="log",
        ylim=(top * 1e-8, top * 3),
        xlabel=r"$r/r_{max}$",
        ylabel=r"$\rho$ [g cm$^{-3}$]",
    )
    axes[0, 1].set(xscale="log", xlabel=r"$r/r_{max}$", ylabel=r"$\beta_a$")
    axes[0, 0].legend(fontsize=8)
    time = trajectory.time_s
    axes[1, 0].plot(time, trajectory.radius_cm, label="jet head")
    axes[1, 0].plot(time, [ejecta.outer_radius(float(t)) for t in time], "--", label=r"$r_{max}$")
    axes[1, 0].set(xlabel="time [s]", ylabel="radius [cm]", yscale="log")
    axes[1, 0].legend()
    axes[1, 1].plot(
        npc.time_s, npc.neutron_to_proton_optical_depth, label=r"$\tau_{n\to p}$"
    )
    axes[1, 1].plot(
        npc.time_s, npc.proton_to_neutron_optical_depth, label=r"$\tau_{p\to n}$"
    )
    axes[1, 1].plot(npc.time_s, npc.relative_lorentz_factor, label=r"$\Gamma_{rel}$")
    axes[1, 1].plot(npc.time_s, npc.gyration_parameter, label=r"$\xi(1)$")
    axes[1, 1].axhspan(0.1, 2, color="tab:green", alpha=0.12)
    axes[1, 1].set(xlabel="time [s]", ylabel="NPC quantities", yscale="log")
    axes[1, 1].legend()
    figure.suptitle(
        f"{model.name}: {model.kind}, Liso={model.luminosity_iso:.0e} erg/s, "
        f"tlaunch={model.launch_s:g} s"
    )
    figure.savefig(destination, dpi=170)
    plt.close(figure)


def plot_overview(rows: list[tuple], destination: Path):
    """Compare the pre-breakout species quantities across all models."""
    names = [row[0] for row in rows]
    colors = ["tab:blue" if row[1] == "numerical" else "tab:orange" for row in rows]
    x = np.arange(len(rows))
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    quantities = (
        (14, r"$\Gamma_{rel}$"),
        (15, r"$n_p$ [cm$^{-3}$]"),
        (16, r"$n_n^{free}$ [cm$^{-3}$]"),
        (18, r"$\tau_{p\to n}$"),
    )
    for axis, (column, label) in zip(axes.flat, quantities, strict=True):
        axis.scatter(x, [row[column] for row in rows], c=colors, s=45)
        axis.set(ylabel=label, xticks=x, xticklabels=names)
        axis.tick_params(axis="x", rotation=55, labelsize=8)
        if column == 15:
            axis.set_yscale("log")
        elif column in (16, 18):
            axis.set_ylim(bottom=0)
    axes[0, 0].axhline(2, color="0.4", linestyle="--", linewidth=1)
    axes[1, 1].axhspan(0.1, 2, color="tab:green", alpha=0.12)
    figure.suptitle(
        "NPC quantities immediately before shock breakout\n"
        "blue: numerical; orange: analytical"
    )
    figure.savefig(destination, dpi=180)
    return figure


def _latex_escape(value: str) -> str:
    return value.replace("_", r"\_")


def write_latex_report(rows: list[tuple], output: Path) -> Path:
    """Write and compile the concise data-interface note with LaTeX."""
    table_rows = "\n".join(
        f"{_latex_escape(row[0])} & {_latex_escape(row[1])} & {row[8]:.3g} & "
        f"{row[14]:.3g} & {row[15]:.2e} & {row[16]:.2e} & "
        f"{row[17]:.3g} & {row[18]:.3g} \\\\" for row in rows
    )
    source = rf"""\documentclass[10pt]{{article}}
\usepackage[a4paper,margin=1.6cm]{{geometry}}
\usepackage{{amsmath,graphicx,array}}
\setlength{{\parindent}}{{0pt}}
\begin{{document}}
\begin{{center}}\Large\bfseries Representative jetBNS NPC inputs\end{{center}}

\textbf{{Files to transfer.}} Transfer exactly
\texttt{{npc\_prebreakout\_models.h5}} and this PDF. The HDF5 file is the
machine-readable input. This document defines its content. The PNG files and
Python scripts are optional diagnostics and reproducibility material.

\section*{{Physical definitions}}
The species-resolved estimates are
\begin{{align}}
n_p &= Y_e\rho/m_p,\\
X_{{n,\mathrm{{free}}}} &= \max(0,1-2Y_e)\frac{{2}}{{\pi}}
\tan^{{-1}}\!\left(\frac{{m_n}}{{m_{{\rm above}}}}\right)e^{{-t/\tau_n}},\\
n_n &= X_{{n,\mathrm{{free}}}}\rho/m_p,\\
\tau_{{n\rightarrow p}} &= n_p\sigma_{{pn}}\Delta r/\Gamma_a,\qquad
\tau_{{p\rightarrow n}} = n_n\sigma_{{pn}}\Delta r/\Gamma_a.
\end{{align}}
Defaults are $m_n=10^{{-4}}M_\odot$, $\tau_n=900$ s,
$\Delta r=r$, and $\sigma_{{pn}}=3\times10^{{-26}}$ cm$^2$.
Numerical profiles use local $Y_e$; analytical profiles use $Y_e=0.1$.
These are phenomenological free-nucleon estimates. Bound nuclei and nuclear
reaction-network evolution are not included.

\section*{{How to read the HDF5 file}}
Use \texttt{{/models/<model>/}} for the time series supplied to the Monte
Carlo. The arrays have equal length and each dataset has a \texttt{{unit}}
attribute. The primary arrays are
\texttt{{time\_s}}, \texttt{{radius\_cm}},
\texttt{{relative\_lorentz\_factor}},
\texttt{{proton\_number\_density\_cm3}},
\texttt{{free\_neutron\_number\_density\_cm3}},
\texttt{{neutron\_to\_proton\_optical\_depth}}, and
\texttt{{proton\_to\_neutron\_optical\_depth}}.
The last array element is the last integration point before breakout.
The group \texttt{{/prebreakout\_summary/}} contains one scalar row per model
for rapid model selection; it is not a replacement for the trajectories.

\begin{{verbatim}}
import h5py
with h5py.File("npc_prebreakout_models.h5") as f:
    model = f["models/num_sfho"]
    time = model["time_s"][:]
    gamma_rel = model["relative_lorentz_factor"][:]
    n_p = model["proton_number_density_cm3"][:]
    n_n = model["free_neutron_number_density_cm3"][:]
    tau_np = model["neutron_to_proton_optical_depth"][:]
    tau_pn = model["proton_to_neutron_optical_depth"][:]
\end{{verbatim}}

\section*{{Pre-breakout values}}
\scriptsize
\begin{{center}}
\begin{{tabular}}{{l l r r r r r r}}
Model & Type & $t_{{bo}}$ [s] & $\Gamma_{{rel}}$ & $n_p$ & $n_n$ &
$\tau_{{n\to p}}$ & $\tau_{{p\to n}}$\\ \hline
{table_rows}
\end{{tabular}}
\end{{center}}
\normalsize
Number densities are in cm$^{{-3}}$. Four numerical profiles have
$Y_e\geq0.5$ at the sampled location and hence $n_n=0$ in this prescription.
SFHo has $\tau_{{p\to n}}\simeq1.06$. The analytical cases have
$\tau_{{n\to p}}\simeq0.63$--$1.50$ and
$\tau_{{p\to n}}\simeq5.01$--$11.83$.

\newpage
\begin{{center}}
\includegraphics[width=0.96\textwidth]{{npc_prebreakout_overview.png}}
\end{{center}}
\end{{document}}
"""
    tex_path = output / "npc_model_notes.tex"
    tex_path.write_text(source)
    executable = shutil.which("pdflatex")
    if executable is None:
        raise RuntimeError("pdflatex is required to build npc_model_notes.pdf")
    subprocess.run(
        [executable, "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
        cwd=output,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    return output / "npc_model_notes.pdf"


def write_hdf5(rows: list[tuple], trajectories: list[tuple], destination: Path) -> None:
    names = (
        "name", "kind", "profile", "launch_time_s", "luminosity_iso_erg_s",
        "beta_width", "mass_msun", "tail_exponent", "breakout_time_s",
        "breakout_radius_cm", "breakout_radius_over_nominal", "prebreakout_time_s",
        "prebreakout_radius_cm", "electron_fraction", "relative_lorentz_factor",
        "proton_number_density_cm3", "free_neutron_number_density_cm3",
        "neutron_to_proton_optical_depth", "proton_to_neutron_optical_depth",
        "gyration_parameter", "downstream_temperature_kev",
        "breakout_electron_optical_depth", "shock_beta_ambient_frame",
    )
    units = {
        "launch_time_s": "s", "luminosity_iso_erg_s": "erg s^-1",
        "beta_width": "1", "mass_msun": "Msun", "tail_exponent": "1",
        "breakout_time_s": "s", "breakout_radius_cm": "cm",
        "breakout_radius_over_nominal": "1", "prebreakout_time_s": "s",
        "prebreakout_radius_cm": "cm", "electron_fraction": "1",
        "relative_lorentz_factor": "1", "proton_number_density_cm3": "cm^-3",
        "free_neutron_number_density_cm3": "cm^-3",
        "neutron_to_proton_optical_depth": "1",
        "proton_to_neutron_optical_depth": "1", "gyration_parameter": "1",
        "downstream_temperature_kev": "keV", "breakout_electron_optical_depth": "1",
        "shock_beta_ambient_frame": "1",
    }
    with h5py.File(destination, "w") as handle:
        handle.attrs["schema"] = "jetbns.npc-model-export.v1"
        handle.attrs["description"] = (
            "Five numerical and five analytical models sampled immediately before breakout"
        )
        handle.attrs["caveat"] = "Phenomenological free-nucleon composition; no Monte Carlo"
        summary = handle.create_group("prebreakout_summary")
        strings = h5py.string_dtype("utf-8")
        for index, name in enumerate(names):
            values = [row[index] for row in rows]
            dataset = summary.create_dataset(
                name, data=values, dtype=strings if index < 3 else None
            )
            if name in units:
                dataset.attrs["unit"] = units[name]
        models = handle.create_group("models")
        for model, config, npc in trajectories:
            group = models.create_group(model.name)
            group.attrs["kind"] = model.kind
            group.attrs["launch_time_s"] = model.launch_s
            group.attrs["luminosity_iso_erg_s"] = model.luminosity_iso
            group.attrs["electron_fraction_fallback"] = config.electron_fraction
            for field in fields(npc):
                dataset = group.create_dataset(field.name, data=getattr(npc, field.name))
                dataset.attrs["unit"] = NPC_UNITS[field.name]


def main() -> None:
    output = Path(__file__).parent / "output" / "npc_model_export"
    plots = output / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    rows = []
    trajectories = []
    for model in MODELS:
        ejecta, jet, trajectory, config, npc = solve(model)
        i = -1
        hb, ab, _ = jet.state(trajectory.breakout_radius_cm, trajectory.breakout_time_s)
        shock_beta = jet.shock_beta_in_ambient_frame(hb, ab)
        tau_breakout = ejecta.optical_depth(
            trajectory.breakout_radius_cm, trajectory.breakout_time_s, samples=512
        )
        if hasattr(ejecta, "electron_fraction"):
            electron_fraction = float(ejecta.electron_fraction(npc.radius_cm[i], npc.time_s[i]))
        else:
            electron_fraction = config.electron_fraction
        rows.append(
            (
                model.name,
                model.kind,
                model.profile,
                model.launch_s,
                model.luminosity_iso,
                model.width,
                model.mass_msun,
                model.tail_exponent,
                trajectory.breakout_time_s,
                trajectory.breakout_radius_cm,
                trajectory.breakout_radius_cm / ejecta.outer_radius(trajectory.breakout_time_s),
                npc.time_s[i],
                npc.radius_cm[i],
                electron_fraction,
                npc.relative_lorentz_factor[i],
                npc.proton_number_density_cm3[i],
                npc.free_neutron_number_density_cm3[i],
                npc.neutron_to_proton_optical_depth[i],
                npc.proton_to_neutron_optical_depth[i],
                npc.gyration_parameter[i],
                npc.downstream_temperature_kev[i],
                tau_breakout,
                shock_beta,
            )
        )
        trajectories.append((model, config, npc))
        plot_model(model, ejecta, trajectory, npc, plots / f"{model.name}.png")
        print(f"finished {model.name}", flush=True)
    hdf5_path = output / "npc_prebreakout_models.h5"
    write_hdf5(rows, trajectories, hdf5_path)
    overview = plot_overview(rows, output / "npc_prebreakout_overview.png")
    report = write_latex_report(rows, output)
    archive = output / "npc_monte_carlo_inputs.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.write(hdf5_path, arcname=hdf5_path.name)
        bundle.write(report, arcname=report.name)
    plt.close(overview)
    print(f"wrote {hdf5_path}")
    print(f"wrote {report}")
    print(f"wrote {archive}")


if __name__ == "__main__":
    main()
