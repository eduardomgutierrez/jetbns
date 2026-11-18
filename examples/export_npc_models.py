"""Export resolution-tested NPC models: analytical by default, numerical opt-in."""
from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import asdict, dataclass, fields, replace
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
from jetbns.npc import GREY_OPACITY_CM2_G, NPC_UNITS

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "examples/output/npc_model_export"
SCHEMA = "jetbns.npc-model-export.v4"
NPC_SCHEMA = "jetbns.npc-inputs.v5"
UNIFORM_SAMPLES = 201
UNIFORM_FRACTION = .99  # last uniform sample: fraction of the propagation t_bo - t_0
GEOMETRIC_SAMPLES = 60
GEOMETRIC_REMAINING = (.3, 1e-4)  # (t_bo - t)/(t_bo - t_0), first and last geometric sample
SAMPLING = ("201 uniform to 0.99 plus 60 geometric in (t_bo - t)/(t_bo - t_0) "
            "from 0.3 to 1e-4")
PAIR_REGULATED_TEMPERATURE_KEV = (100., 200.)
RADIATION_STATE_NOTE = (
    "The jet-head shock in merger ejecta is photon-starved (upstream photons per baryon << "
    "threshold); the immediate downstream is expected at 100-200 keV with pair loading; the "
    "thermalised temperature is a lower limit. np-converter decision 2026-11-18.")
FIELD_FRAMES = {
    "exterior_pn_column_lightcone": "light cone of an outgoing particle; all baryons as targets",
    "exterior_pn_column_static": "frozen lab-time radial profile; all baryons as targets",
    "exterior_grey_optical_depth": "frozen lab-time radial profile",
    "lightcone_exit_radius_cm": "lab",
    "lightcone_exited": "light cone of an outgoing particle",
    "downstream_temperature_thermalised_kev": "downstream fluid rest",
    "photon_rich_threshold_photons_per_baryon": "upstream fluid rest",
    "pair_loading_estimate": "immediate downstream (pairs per baryon)",
}
FIELD_ROLES = {
    "exterior_pn_column_lightcone": (
        "primary conversion depth (np-converter decision 2026-11-18); for a steady wind "
        "equals n_u sigma_pn r/(2 Gamma_a)"),
    "exterior_pn_column_static": "frozen-slice diagnostic",
    "exterior_grey_optical_depth": "breakout criterion depth, kappa = 0.16",
    "lightcone_exit_radius_cm": "provenance of exterior_pn_column_lightcone",
    "lightcone_exited": (
        "provenance of exterior_pn_column_lightcone (1: light cone left the ejecta; "
        "0: stopped at 100 r_h)"),
    "pn_optical_depth": (
        "legacy local depth n_b sigma_pn r/Gamma_a; 30-120 x the light-cone column near breakout"),
    "gyration_parameter": "legacy baryon-target estimate",
    "proton_number_density_cm3": "assumed free fraction times Ye rho/mp",
    "downstream_temperature_kev": (
        "legacy estimate (rho c^2 Gamma^2/a)^(1/4) without the jump-condition factor"),
    "downstream_temperature_thermalised_kev": (
        "fully thermalised downstream, lower limit on the immediate downstream temperature"),
    "photon_rich_threshold_photons_per_baryon": (
        "Ito et al. 2018/2020: the shock is photon-rich only above this upstream "
        "photon-to-baryon ratio"),
    "pair_loading_estimate": (
        "Budnik et al. 2010 pairs per baryon in the immediate downstream of a photon-starved shock"),
}
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
    "exterior_pn_column_lightcone", "exterior_grey_optical_depth",
    "downstream_temperature_thermalised_kev",
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
    velocity: str = "recorded"
    continuation: tuple[float, float] = (5/3, 1/4)
    beta_width: float = .05


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
# Gutierrez et al. 2025 settings for the same three profiles: Bernoulli asymptotic launch speed,
# kernel width beta_width = 0.035 (sigma = 0.025 c) and per-history continuation indices
# (post_simulation_mass_index, post_simulation_velocity_index). The indices are the late-time
# slopes (t > 0.3 s) of the paper-era extended files DD2_M135-135_M1_K2_SR.h5,
# SFHo_M135-135_M1_SR.h5 and SLy_M145-125_M1_SR_v2.h5, measured 2026-11-05 (np-converter audit,
# reduced/v3-six-models/audit/ejecta_variants.json).
NUMERICAL_2025 = tuple(
    replace(model, name=f"{model.name}_2025", velocity="asymptotic", beta_width=.035,
            continuation=continuation)
    for model, continuation in zip(NUMERICAL, ((1.50, 0.13), (7.16, 5.23), (9.52, 6.91)))
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
        solid_angle_sr=float(angle), velocity=model.velocity, cutoff_mode="smooth",
        kernel_shape=2., beta_width=model.beta_width,
        post_simulation_mass_index=model.continuation[0],
        post_simulation_velocity_index=model.continuation[1],
        integration_samples=256*2**fine, history_subsamples=2**fine,
    )


def sample_times(start: float, breakout: float) -> np.ndarray:
    """Union of UNIFORM_SAMPLES uniform times to UNIFORM_FRACTION of the propagation and
    GEOMETRIC_SAMPLES times geometric in (t_bo - t)/(t_bo - t_0), deduplicated to 1e-12."""
    span = breakout-start
    uniform = np.linspace(start, start+UNIFORM_FRACTION*span, UNIFORM_SAMPLES)
    geometric = breakout-np.geomspace(*GEOMETRIC_REMAINING, GEOMETRIC_SAMPLES)*span
    times = np.sort(np.concatenate([uniform, geometric]))
    keep = np.concatenate([[True], np.diff(times) > 1e-12*np.abs(times[1:])])
    return times[keep]


def sample_solution(jet: JetCocoon, trajectory: CocoonPropagationResult,
                    count: int | None = None):
    """Select reproducible physical times, independent of the integration step.

    ``count=None`` gives the export sampling (:func:`sample_times`); an integer gives
    ``count`` uniform times to UNIFORM_FRACTION only (the v3 grid, used by the playground).
    """
    start, stop = float(trajectory.time_s[0]), float(trajectory.time_s[-1])
    times = (sample_times(start, stop) if count is None
             else np.linspace(start, start+UNIFORM_FRACTION*(stop-start), count))
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
    group.attrs["npc_schema"] = NPC_SCHEMA
    group.attrs["composition"] = "free-proton proxy plus Metzger free-neutron skin"
    group.attrs["sample_fraction_of_propagation"] = 1-GEOMETRIC_REMAINING[1]
    group.attrs["sampling"] = SAMPLING
    group.attrs["grey_opacity_cm2_g"] = GREY_OPACITY_CM2_G
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
    # Kept at the v3 snapshot (the last uniform sample, UNIFORM_FRACTION of the propagation).
    snapshot_time = trajectory.time_s[0]+UNIFORM_FRACTION*(trajectory.time_s[-1]
                                                           -trajectory.time_s[0])
    index = int(np.argmin(np.abs(npc.time_s-snapshot_time)))
    if not np.isclose(npc.time_s[index], snapshot_time, rtol=1e-12, atol=0):
        raise RuntimeError("snapshot_columns: the 99 % sample is missing")
    columns.attrs["role"] = (
        "legacy frozen-slice convention with Gamma_a (1-beta_a), undercounts by 1.8-5.3; "
        "superseded by exterior_pn_column_lightcone")
    columns.attrs["time_s"] = npc.time_s[index]
    columns.attrs["radius_cm"] = npc.radius_cm[index]
    columns.attrs["sample_index"] = index
    columns.attrs["sample_fraction_of_propagation"] = UNIFORM_FRACTION
    for key, value in snapshot_columns(ejecta, config, npc.radius_cm[index],
                                       npc.time_s[index]).items():
        ds = columns.create_dataset(key, data=value)
        ds.attrs["unit"] = "1"
        ds.attrs["frame"] = "frozen lab-time radial profile; see PDF"
    for field in fields(npc):
        ds = group.create_dataset(field.name, data=getattr(npc, field.name), compression="gzip")
        ds.attrs["unit"] = NPC_UNITS[field.name]
        ds.attrs["frame"] = field_frame(field.name)
    for key, role in FIELD_ROLES.items():
        group[key].attrs["role"] = role
    for label, values in (("configuration", asdict(config)), ("engine", asdict(jet.engine)),
                          ("model_parameters", asdict(model)), ("convergence", checks)):
        settings = group.create_group(label)
        for key, value in values.items():
            if label == "model_parameters" and model.kind == "numerical" and key in (
                "mass_msun", "tail_exponent",
            ):
                continue
            if label == "model_parameters" and model.kind == "analytical" and key in (
                "velocity", "continuation", "beta_width",
            ):
                continue
            settings.attrs[key] = value
    if model.kind == "numerical":
        settings = group.create_group("numerical_profile")
        for key in ("extraction_radius_cm", "beta_width", "kernel_shape", "cutoff_mode",
                    "integration_samples", "history_subsamples", "post_simulation_mass_index",
                    "post_simulation_velocity_index"):
            settings.attrs[key] = getattr(ejecta, key)
        settings.attrs["velocity_source"] = (
            "recorded vel; NOT Bernoulli asymptotic velocity"
            if ejecta.history.velocity_source == "recorded" else
            "Bernoulli asymptotic velocity; mass-loss rate from recorded vel")
        settings.attrs["dropped_non_increasing_samples"] = ejecta.history.dropped_samples
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
    if name in FIELD_FRAMES:
        return FIELD_FRAMES[name]
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
    tex = output/"npc_model_notes.tex"
    # Keep the handoff note short and independent of generated plots.  The
    # HDF5 file remains the authoritative source for model endpoints.
    tex.write_text(template.read_text())
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
    parser.add_argument("--ejecta-2025", action="store_true",
                        help="with --include-numerical, use NUMERICAL_2025 (Gutierrez et al. 2025 "
                             "ejecta settings) instead of NUMERICAL")
    parser.add_argument("--data-root", type=Path, default=ROOT.parent)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.ejecta_2025 and not args.include_numerical:
        parser.error("--ejecta-2025 requires --include-numerical")
    numerical = NUMERICAL_2025 if args.ejecta_2025 else NUMERICAL
    models = ANALYTICAL + numerical if args.include_numerical else ANALYTICAL
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
                "v1: unresolved launch history; v2: conical propagation without cocoon; "
                "v3: 201 uniform samples, local depths only, no radiation state"
            )
            handle.attrs["optical_depth_definition"] = (
                "exterior_pn_column_lightcone: integrated p-n column along the light cone "
                "(primary conversion depth); pn_optical_depth and "
                "neutron_to_proton/proton_to_neutron_optical_depth: local n sigma r/Gamma_a, "
                "not integrated columns"
            )
            handle.attrs["snapshot_fraction"] = UNIFORM_FRACTION
            handle.attrs["sampling"] = SAMPLING
            handle.attrs["pair_regulated_temperature_kev"] = np.array(
                PAIR_REGULATED_TEMPERATURE_KEV)
            handle.attrs["radiation_state_note"] = RADIATION_STATE_NOTE
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
                      f"col_lightcone={row['exterior_pn_column_lightcone']:.4g}; "
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
