"""Export resolution-tested NPC models: analytical by default, numerical opt-in."""
from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from jetbns import (
    Cocoon,
    CocoonPropagationResult,
    ConstantEngine,
    HomologousTail,
    JetCocoon,
    NpcConfig,
    NumericalEjecta,
    PropagationResult,
    evaluate_npc_inputs,
    metzger_free_neutron_fraction,
)
from jetbns.constants import PROTON_MASS, SOLAR_MASS, SPEED_OF_LIGHT
from jetbns.npc import NPC_UNITS

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "examples/output/npc_model_export"
SCHEMA = "jetbns.npc-model-export.v3"
COCOON_FIELDS = {
    "cocoon_radius_cm": "cm", "cocoon_energy_erg": "erg",
    "pressure_energy_erg": "erg", "cocoon_pressure_erg_cm3": "erg cm^-3",
    "cocoon_lateral_beta": "1", "jet_cross_section_cm2": "cm^2",
    "jet_opening_angle_rad": "rad", "cocoon_deposition_rate_erg_s": "erg s^-1",
}
SUMMARY_FIELDS = (
    "time_s", "radius_cm", "relative_lorentz_factor", "electron_fraction",
    "proton_number_density_cm3", "free_neutron_number_density_cm3",
    "neutron_to_proton_optical_depth", "proton_to_neutron_optical_depth",
    "gyration_parameter", "downstream_temperature_kev",
)
CHECK_FIELDS = (
    "upstream_density_g_cm3", "electron_fraction", "relative_lorentz_factor",
    "neutron_to_proton_optical_depth", "proton_to_neutron_optical_depth",
)
COCOON_CHECK_FIELDS = (
    "cocoon_energy_erg", "cocoon_radius_cm",
    "cocoon_pressure_erg_cm3", "jet_cross_section_cm2",
)
NPC_CONVERGENCE_TOLERANCE = .05
COCOON_CONVERGENCE_TOLERANCE = .10


@dataclass(frozen=True)
class Model:
    name: str
    kind: str = "analytical"
    launch_s: float = 1.0
    luminosity_iso_erg_s: float = 1e51
    mass_msun: float = 1e-4
    electron_fraction: float = .3
    tail_exponent: float = 8.0
    profile: str = ""


ANALYTICAL = (
    Model("ana_reference"),
    Model("ana_neutron_rich", electron_fraction=.1),
    Model("ana_steeper_tail", mass_msun=3e-4, tail_exponent=16,
          luminosity_iso_erg_s=1e52, launch_s=2.),
)
NUMERICAL = (
    Model("num_dd2_equal", "numerical", luminosity_iso_erg_s=1e52,
          profile="outflow_files_jet_propagation/Archive/wdir/dd2_analysis/"
                  "DD2_M135135_M1_K2_SR/outflow_1_ber_out/lagrangian_profile.h5"),
    Model("num_sfho", "numerical", luminosity_iso_erg_s=1e53,
          profile="jet_propagation/jet_propagation/profile_files/new/"
                  "SFHo_M135-135_M1_SR/outflow_1_ber_hmin_out/lagrangian_profile.h5"),
    Model("num_sly", "numerical", luminosity_iso_erg_s=1e52,
          profile="jet_propagation/jet_propagation/profile_files/new/"
                  "SLy_M145-125_M1_SR_v2/outflow_1_ber_hmin_out/lagrangian_profile.h5"),
)


def build(model: Model, data_root: Path, *, fine: int = 0):
    if model.kind == "analytical":
        return HomologousTail(mass_msun=model.mass_msun, tail_exponent=model.tail_exponent)
    path = data_root / model.profile
    with h5py.File(path) as handle:
        theta = np.asarray(handle["theta"])
        spacing = theta[1] - theta[0]
        angle = 2*np.pi*(np.cos(theta[0]-spacing/2)-np.cos(theta[0]+spacing/2))
    return NumericalEjecta.from_hdf5(
        path, bin_name="itheta=00000", extraction_radius_cm=4.42e7,
        solid_angle_sr=float(angle), cutoff_mode="smooth", kernel_shape=2., beta_width=.05,
        integration_samples=256*2**fine, history_subsamples=2**fine,
    )


def sample_solution(jet: JetCocoon, trajectory: CocoonPropagationResult, count: int = 201):
    """Select a reproducible physical time, independent of the integration step."""
    start = trajectory.time_s[0]
    stop = start + .99*(trajectory.time_s[-1]-start)
    times = np.linspace(start, stop, count)
    radii = np.interp(times, trajectory.time_s, trajectory.radius_cm)
    width = np.interp(times, trajectory.time_s, trajectory.cocoon_radius_cm)
    energy = np.interp(times, trajectory.time_s, trajectory.cocoon_energy_erg)
    pressure_energy = np.interp(
        times-(radii-trajectory.base_radius_cm)*np.sqrt(3)/(2*SPEED_OF_LIGHT),
        trajectory.time_s, trajectory.cocoon_energy_erg, left=0.)
    if not jet.cocoon.pressure_delay:
        pressure_energy = energy
    states = [jet.state(float(r), float(t), cocoon_radius_cm=w, cocoon_energy_erg=e,
                        pressure_energy_erg=p, base_radius_cm=trajectory.base_radius_cm)
              for r, t, w, e, p in zip(radii, times, width, energy, pressure_energy, strict=True)]
    return PropagationResult(times, radii, np.array([s.head_beta for s in states]),
                             np.array([s.ambient_beta for s in states]),
                             np.array([s.dimensionless_luminosity for s in states]), False)


def solve(model: Model, data_root: Path, *, fine: int = 0):
    ejecta = build(model, data_root, fine=fine)
    launch_radius = max(8.45e7, ejecta.inner_radius(model.launch_s)*1.01)
    engine = ConstantEngine.from_isotropic_equivalent(
        model.luminosity_iso_erg_s, launch_time_s=model.launch_s,
        launch_radius_cm=launch_radius, opening_angle_rad=np.deg2rad(10),
        lorentz_factor=100, duration_s=60.,
    )
    # The ellipsoidal density average has its own convergence test. Refining it
    # inside every trajectory step multiplies the expensive numerical-profile
    # quadrature without changing the accepted solution at useful precision.
    jet = JetCocoon(engine, ejecta, cocoon=Cocoon(density_samples=48),
                    breakout_optical_depth_samples=192*2**fine)
    trajectory = jet.propagate(max_time_s=model.launch_s+60,
                               time_step_s=.01/2**fine)
    if not trajectory.broke_out:
        raise RuntimeError(f"{model.name} did not break out; no transfer archive created")
    config = NpcConfig(electron_fraction=model.electron_fraction,
                       exterior_mass_samples=256*2**fine)
    npc = evaluate_npc_inputs(sample_solution(jet, trajectory), ejecta, config=config)
    return ejecta, jet, trajectory, config, npc


def relative_error(a: float, b: float) -> float:
    return float(abs(a-b)/max(abs(a), abs(b), 1e-100))


def audit_model(coarse: tuple, fine: tuple) -> dict:
    """Refine time, radial, recorded-history and exterior-mass quadratures."""
    ej, jet, tr, _, npc = fine
    errors = {"breakout_time_s": relative_error(coarse[2].breakout_time_s, tr.breakout_time_s)}
    errors.update({name: relative_error(getattr(coarse[4], name)[-1], getattr(npc, name)[-1])
                   for name in CHECK_FIELDS})
    errors.update({name: relative_error(getattr(coarse[2], name)[-1], getattr(tr, name)[-1])
                   for name in COCOON_CHECK_FIELDS})
    beta = jet.shock_beta_in_ambient_frame(tr.head_beta[-1], tr.ambient_beta[-1])
    tau = ej.optical_depth(tr.breakout_radius_cm, tr.breakout_time_s, samples=1536)
    errors["breakout_condition"] = abs(tau*beta-1)
    npc_errors = [value for key, value in errors.items() if key not in COCOON_CHECK_FIELDS]
    cocoon_errors = [errors[key] for key in COCOON_CHECK_FIELDS]
    errors["passed"] = bool(
        max(npc_errors) < NPC_CONVERGENCE_TOLERANCE
        and max(cocoon_errors) < COCOON_CONVERGENCE_TOLERANCE
    )
    if not errors["passed"]:
        raise RuntimeError(f"resolution check failed: {errors}")
    return errors


def write_model(handle, model: Model, solution: tuple, checks: dict, data_root: Path):
    ejecta, jet, trajectory, config, npc = solution
    group = handle.require_group("models").create_group(model.name)
    group.attrs["npc_schema"] = "jetbns.npc-inputs.v4"
    group.attrs["composition"] = "free-proton proxy plus Metzger free-neutron skin"
    group.attrs["sample_fraction_of_propagation"] = .99
    group.attrs["breakout_time_s"] = trajectory.breakout_time_s
    group.attrs["breakout_radius_cm"] = trajectory.breakout_radius_cm
    group.attrs["sample_lag_before_breakout_s"] = trajectory.breakout_time_s-npc.time_s[-1]
    group.attrs["kind"] = model.kind
    group.attrs["propagation_model"] = "JetCocoon"
    group.attrs["npc_convergence_tolerance"] = NPC_CONVERGENCE_TOLERANCE
    group.attrs["cocoon_convergence_tolerance"] = COCOON_CONVERGENCE_TOLERANCE
    group.attrs["base_radius_cm"] = trajectory.base_radius_cm
    dynamics = group.create_group("jet_cocoon")
    for key, unit in COCOON_FIELDS.items():
        ds = dynamics.create_dataset(key, data=np.interp(
            npc.time_s, trajectory.time_s, getattr(trajectory, key)), compression="gzip")
        ds.attrs["unit"] = unit
        ds.attrs["frame"] = "lab / cocoon closure; see PDF"
    for key, value in asdict(jet.cocoon).items():
        dynamics.attrs[key] = value
    dynamics.attrs["calibration"] = jet.calibration
    dynamics.attrs["initial_height_fraction"] = jet.initial_height_fraction
    level = int(np.log2(config.exterior_mass_samples/256))
    group.attrs["resolution_level"] = level
    group.attrs["ode_time_step_s"] = .01/2**level
    group.attrs["breakout_radial_samples"] = jet.breakout_optical_depth_samples
    engine_age = jet.engine.retarded_time(trajectory.breakout_radius_cm,
                                         trajectory.breakout_time_s) - model.launch_s
    group.attrs["engine_active_age_at_breakout_s"] = engine_age
    group.attrs["one_sided_energy_supplied_erg"] = engine_age*jet.engine.luminosity_erg_s
    columns = group.create_group("snapshot_columns")
    for key, value in snapshot_columns(ejecta, config, npc.radius_cm[-1], npc.time_s[-1]).items():
        ds = columns.create_dataset(key, data=value)
        ds.attrs["unit"] = "1"
        ds.attrs["frame"] = "frozen lab-time radial profile; see PDF"
    for field in fields(npc):
        ds = group.create_dataset(field.name, data=getattr(npc, field.name), compression="gzip")
        ds.attrs["unit"] = NPC_UNITS[field.name]
        ds.attrs["frame"] = field_frame(field.name)
    group["pn_optical_depth"].attrs["role"] = "legacy total-baryon reference"
    group["gyration_parameter"].attrs["role"] = "legacy baryon-target estimate"
    group["proton_number_density_cm3"].attrs["role"] = "assumed free fraction times Ye rho/mp"
    for label, values in (("configuration", asdict(config)), ("engine", asdict(jet.engine)),
                          ("model_parameters", asdict(model)), ("convergence", checks)):
        settings = group.create_group(label)
        for key, value in values.items():
            if label == "model_parameters" and model.kind == "numerical" and key in (
                "mass_msun", "tail_exponent",
            ):
                continue
            settings.attrs[key] = value
    if model.kind == "numerical":
        settings = group.create_group("numerical_profile")
        for key in ("extraction_radius_cm", "beta_width", "kernel_shape", "cutoff_mode",
                    "integration_samples", "history_subsamples", "post_simulation_mass_index",
                    "post_simulation_velocity_index"):
            settings.attrs[key] = getattr(ejecta, key)
        settings.attrs["velocity_source"] = "recorded vel; NOT Bernoulli asymptotic velocity"
        settings.attrs["source_sha256"] = hashlib.sha256(
            (data_root/model.profile).read_bytes()).hexdigest()
        settings.attrs["recorded_time_start_s"] = ejecta.history.time_s[0]
        settings.attrs["recorded_time_end_s"] = ejecta.history.time_s[-1]
        settings.attrs["recorded_samples"] = len(ejecta.history.time_s)
        settings.attrs["solid_angle_sr"] = ejecta.history.solid_angle_sr
    else:
        settings = group.create_group("analytical_ejecta")
        for key, value in asdict(ejecta).items():
            settings.attrs[key] = value
    return {name: float(getattr(npc, name)[-1]) for name in SUMMARY_FIELDS}


def snapshot_columns(ejecta, config, radius, time) -> dict[str, float]:
    """Outward ultrarelativistic attenuation through a frozen radial profile."""
    r = np.geomspace(radius, ejecta.optical_depth_outer_radius(time), 1536)
    rho = np.asarray(ejecta.density(r, time))
    lab_density = np.asarray(ejecta.lab_mass_density(r, time))
    beta = np.asarray(ejecta.velocity(r, time))/SPEED_OF_LIGHT
    integrand = 4*np.pi*r**2*lab_density
    increments = .5*(integrand[:-1]+integrand[1:])*np.diff(r)
    mass_above = np.append(np.cumsum(increments[::-1])[::-1], 0)/SOLAR_MASS
    ye = (np.asarray(ejecta.electron_fraction(r, time))
          if isinstance(ejecta, NumericalEjecta) and ejecta.history.electron_fraction is not None
          else np.full_like(r, config.electron_fraction))
    fraction = metzger_free_neutron_fraction(ye, mass_above, time,
        transition_mass_msun=config.free_neutron_transition_mass_msun,
        decay_time_s=config.free_neutron_decay_time_s)
    path_factor = np.sqrt((1-beta)/(1+beta))  # Gamma_a (1-beta_a), radial fast projectile
    return {
        "neutron_on_proton_radial_column_depth": float(np.trapezoid(
            config.proton_free_fraction*ye*rho/PROTON_MASS
            * config.pn_cross_section_cm2*path_factor, r)),
        "proton_on_neutron_radial_column_depth": float(np.trapezoid(
            fraction*rho/PROTON_MASS*config.pn_cross_section_cm2*path_factor, r)),
        "grey_radial_optical_depth": float(np.trapezoid(.16*rho, r)),
    }


def field_frame(name: str) -> str:
    if name == "shock_beta_downstream_frame":
        return "downstream fluid rest"
    if name == "shock_beta_upstream_frame":
        return "upstream fluid rest"
    if name in ("relative_lorentz_factor", "hydrodynamic_compression_ratio"):
        return "dimensionless upstream/downstream relation"
    if name in ("time_s", "radius_cm", "path_length_cm", "head_beta", "ambient_beta",
                "head_lorentz_factor", "ambient_lorentz_factor"):
        return "lab"
    if name.startswith("downstream_") or name.startswith("max_lorentz"):
        return "downstream fluid (jet-head proxy); approximate"
    if name.startswith("max_observer") or name.startswith("observer_"):
        return "observer boost convention, not viewing-angle Doppler factor"
    if name in ("electron_fraction", "free_neutron_mass_fraction", "exterior_mass_msun"):
        return "composition / lab-slice mass"
    return "upstream rest or dimensionless; see PDF"


def plot_model(model, solution, output):
    ej, _, tr, _, npc = solution
    fig, axes = plt.subplots(3, 3, figsize=(13, 10), constrained_layout=True)
    for t in np.linspace(tr.time_s[0], tr.time_s[-1], 3):
        r = np.geomspace(ej.inner_radius(t)*1.01, ej.optical_depth_outer_radius(t)*.999, 250)
        rho = ej.density(r, t)
        keep = rho > max(rho.max()*1e-16, npc.upstream_density_g_cm3[-1]*1e-3)
        axes[0, 0].loglog(r[keep], rho[keep], label=f"{t:.2f} s")
    axes[0, 0].set(xlabel="radius [cm]", ylabel=r"proper density [g cm$^{-3}$]")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].plot(tr.time_s, tr.radius_cm/1e11, label="jet head")
    nominal = getattr(ej, "nominal_outer_radius", ej.outer_radius)
    axes[0, 1].plot(tr.time_s, [nominal(t)/1e11 for t in tr.time_s], "--", label="nominal edge")
    axes[0, 1].set(xlabel="time [s]", ylabel=r"radius [$10^{11}$ cm]")
    axes[0, 1].legend(fontsize=8)
    axes[0, 2].plot(npc.time_s, npc.relative_lorentz_factor)
    axes[0, 2].set(xlabel="time [s]", ylabel=r"$\Gamma_{rel}$")
    near = npc.time_s >= tr.time_s[0]+.9*(tr.time_s[-1]-tr.time_s[0])
    remaining = tr.time_s[-1]-npc.time_s[near]
    for values, label in ((npc.proton_number_density_cm3, r"$n_p$ proxy"),
                           (npc.free_neutron_number_density_cm3, r"$n_n$ free")):
        axes[1, 0].plot(remaining, values[near], label=label)
    axes[1, 0].set(xlabel="time before breakout [s]", ylabel=r"number density [cm$^{-3}$]")
    axes[1, 0].legend(fontsize=8)
    for values, label in ((npc.neutron_to_proton_optical_depth, "n on p"),
                           (npc.proton_to_neutron_optical_depth, "p on n")):
        axes[1, 1].plot(remaining, values[near], label=label)
    axes[1, 1].set(xlabel="time before breakout [s]", ylabel="local collision depth")
    axes[1, 1].legend(fontsize=8)
    axes[1, 2].plot(remaining, npc.downstream_temperature_kev[near])
    axes[1, 2].set(xlabel="time before breakout [s]", ylabel=r"estimated $kT_d$ [keV]")
    axes[2, 0].plot(tr.time_s, tr.cocoon_radius_cm/1e10)
    axes[2, 0].set(xlabel="time [s]", ylabel=r"cocoon width [$10^{10}$ cm]")
    positive = tr.cocoon_pressure_erg_cm3 > 0
    axes[2, 1].semilogy(tr.time_s[positive], tr.cocoon_pressure_erg_cm3[positive])
    axes[2, 1].set(xlabel="time [s]", ylabel=r"cocoon pressure [erg cm$^{-3}$]")
    axes[2, 2].plot(tr.time_s, np.rad2deg(tr.jet_opening_angle_rad))
    axes[2, 2].axhline(10., ls="--", color="grey", label="injection angle")
    axes[2, 2].set(xlabel="time [s]", ylabel="jet opening angle [deg]")
    axes[2, 2].legend(fontsize=8)
    for ax in axes[1]:
        ax.invert_xaxis()
        ax.grid(alpha=.2)
    fig.suptitle(model.name + " (coupled jet and cocoon)")
    fig.savefig(output/f"{model.name}.png", dpi=150)
    plt.close(fig)


def plot_overview(rows, output):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    x = np.arange(len(rows))
    labels = [r["name"] for r in rows]
    for ax, key, title in zip(axes, ("relative_lorentz_factor", "neutron_to_proton_optical_depth",
                                    "proton_to_neutron_optical_depth"),
                              (r"$\Gamma_{rel}$", "n on p: local depth", "p on n: local depth"),
                              strict=True):
        ax.scatter(x, [r[key] for r in rows])
        ax.set(xticks=x, xticklabels=labels, ylabel=title)
        ax.tick_params(axis="x", rotation=60, labelsize=8)
        ax.grid(alpha=.2)
    fig.savefig(output/"npc_prebreakout_overview.png", dpi=160)
    plt.close(fig)


def write_latex_report(rows, output):
    template = ROOT/"docs/npc_model_notes.tex"
    table = "\n".join(
        r"\texttt{"+row["name"].replace("_", r"\_")+"} & " + " & ".join(
            f"{row[k]:.3g}" for k in ("relative_lorentz_factor", "electron_fraction",
                "proton_number_density_cm3", "free_neutron_number_density_cm3",
                "neutron_to_proton_optical_depth", "proton_to_neutron_optical_depth")
        )+r" \\" for row in rows
    )
    tex = output/"npc_model_notes.tex"
    selected = [rows[0]["name"]]
    numerical = [r["name"] for r in rows if r["name"] == "num_sfho"]
    selected += numerical
    figures = "\n".join(
        (r"\newpage" if i else "") + r"\includegraphics[width=\linewidth]{plots/"
        + name + ".png}" for i, name in enumerate(selected))
    tex.write_text(template.read_text().replace("% MODEL_TABLE", table)
                   .replace("% EVOLUTION_PLOTS", figures))
    executable = shutil.which("pdflatex")
    if executable is None:
        raise RuntimeError("Install texlive-latex-base (pdflatex) to generate the PDF")
    for _ in range(2):
        subprocess.run([executable, "-no-shell-escape", "-interaction=nonstopmode",
                        "-halt-on-error", tex.name], cwd=output, check=True,
                       stdout=subprocess.DEVNULL)
    return output/"npc_model_notes.pdf"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-numerical", action="store_true")
    parser.add_argument("--data-root", type=Path, default=ROOT.parent)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    models = ANALYTICAL + NUMERICAL if args.include_numerical else ANALYTICAL
    args.output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="npc-build-", dir=args.output) as temp:
        out = Path(temp)
        plots = out/"plots"
        plots.mkdir()
        rows = []
        hdf5 = out/"npc_prebreakout_models.h5"
        with h5py.File(hdf5, "w") as handle:
            handle.attrs["schema"] = SCHEMA
            handle.attrs["generator_sha256"] = hashlib.sha256(
                Path(__file__).read_bytes()).hexdigest()
            for source in ("ejecta", "npc", "propagation", "cocoon", "engines"):
                handle.attrs[f"{source}_code_sha256"] = hashlib.sha256(
                    (ROOT/f"src/jetbns/{source}.py").read_bytes()).hexdigest()
            handle.attrs["status"] = "resolution-tested scenarios; approximate composition"
            handle.attrs["supersedes"] = (
                "v1: unresolved launch history; v2: conical propagation without cocoon"
            )
            handle.attrs["optical_depth_definition"] = (
                "local n sigma r/Gamma_a, not integrated column"
            )
            handle.attrs["snapshot_fraction"] = .99
            for model in models:
                coarse = solve(model, args.data_root)
                for level in range(1, 4):
                    fine = solve(model, args.data_root, fine=level)
                    try:
                        checks = audit_model(coarse, fine)
                        break
                    except RuntimeError as error:
                        if level == 3:
                            raise
                        print(f"{model.name}: refining to level {level+1}; {error}", flush=True)
                        coarse = fine
                row = write_model(handle, model, fine, checks, args.data_root)
                row["name"] = model.name
                rows.append(row)
                plot_model(model, fine, plots)
                error = max(v for k, v in checks.items() if k != "passed")
                print(f"{model.name}: t_bo={fine[2].breakout_time_s:.4g} s; "
                      f"Gamma_rel={row['relative_lorentz_factor']:.3g}; "
                      f"tau_np={row['neutron_to_proton_optical_depth']:.3g}; "
                      f"tau_pn={row['proton_to_neutron_optical_depth']:.3g}; "
                      f"max error={error:.2%}", flush=True)
            summary = handle.create_group("prebreakout_summary")
            summary.create_dataset("name", data=[r["name"] for r in rows],
                                   dtype=h5py.string_dtype("utf-8"))
            for name in SUMMARY_FIELDS:
                ds = summary.create_dataset(name, data=[r[name] for r in rows])
                ds.attrs["unit"] = NPC_UNITS[name]
        plot_overview(rows, out)
        report = write_latex_report(rows, out)
        reader = Path(__file__).with_name("read_npc_inputs.py")
        shutil.copy2(reader, out/reader.name)
        subprocess.run([sys.executable, str(reader), str(hdf5)], check=True)
        archive = out/"npc_monte_carlo_inputs.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            for path in (hdf5, report):
                bundle.write(path, arcname=path.name)
        for path in out.iterdir():
            destination = args.output/path.name
            if path.is_dir():
                if destination.exists():
                    shutil.rmtree(destination)
                shutil.copytree(path, destination, dirs_exist_ok=True)
            else:
                shutil.copy2(path, destination)
    print(f"Transfer: {args.output/'npc_monte_carlo_inputs.zip'}")


if __name__ == "__main__":
    main()
