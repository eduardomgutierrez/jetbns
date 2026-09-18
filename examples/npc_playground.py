"""Notebook helpers for interactive, exploratory jet/cocoon and NPC calculations.

Uses the same physical classes and sampling convention as the audited exporter.
Single-resolution runs are explicitly exploratory; they cannot certify a transfer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from os.path import relpath
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from export_npc_models import (
    ANALYTICAL,
    COCOON_FIELDS,
    NUMERICAL,
    Model,
    build,
    field_frame,
    sample_solution,
    snapshot_columns,
)
from matplotlib.backends.backend_pdf import PdfPages
from plot_npc_model_comparison import PAGES, _set_scale

from jetbns import (
    CocoonPropagationResult,
    ConstantEngine,
    Ejecta,
    JetCocoon,
    NpcConfig,
    NpcInputs,
    evaluate_npc_inputs,
)
from jetbns.npc import NPC_UNITS


@dataclass(frozen=True)
class Run:
    """Keep model, physical configuration and results together for inspection."""

    model: Model
    ejecta: Ejecta
    jet: JetCocoon
    trajectory: CocoonPropagationResult
    config: NpcConfig
    npc: NpcInputs
    time_step_s: float
    max_duration_s: float


def run_model(
    model: Model,
    data_root: Path,
    *,
    lorentz_factor: float = 100.0,
    opening_angle_deg: float = 10.0,
    time_step_s: float = 0.01,
    max_duration_s: float = 15.0,
    sample_count: int = 201,
    kernel_shape: float = 2.0,
    beta_width: float = 0.05,
    cutoff_mode: str = "smooth",
    npc_config: NpcConfig | None = None,
) -> Run:
    """Run one exploratory model; preserve a non-breakout result as such.

    Numerical reconstruction uses the exporter's polar bin and recorded speeds.
    Kernel shape 1 gives an exponential velocity kernel; 2 gives a Gaussian.
    Time is laboratory time since merger; max_duration_s is the horizon after
    engine launch. Engine duration is 60 s as in the reference suite.
    """
    if model.kind not in ("analytical", "numerical"):
        raise ValueError("kind must be analytical or numerical")
    if sample_count < 2 or max_duration_s <= 0 or time_step_s <= 0:
        raise ValueError("require at least two samples and positive time scales")
    if model.kind == "numerical" and not (Path(data_root) / model.profile).is_file():
        raise FileNotFoundError(f"Missing local profile: {Path(data_root) / model.profile}")
    ejecta = build(model, Path(data_root))
    if model.kind == "numerical":
        ejecta = replace(
            ejecta, kernel_shape=kernel_shape, beta_width=beta_width, cutoff_mode=cutoff_mode
        )
    engine = ConstantEngine.from_isotropic_equivalent(
        model.luminosity_iso_erg_s,
        launch_time_s=model.launch_s,
        launch_radius_cm=max(8.45e7, ejecta.inner_radius(model.launch_s) * 1.01),
        opening_angle_rad=np.deg2rad(opening_angle_deg),
        lorentz_factor=lorentz_factor,
        duration_s=60.0,
    )
    jet = JetCocoon(engine, ejecta, breakout_optical_depth_samples=192)
    trajectory = jet.propagate(max_time_s=model.launch_s + max_duration_s, time_step_s=time_step_s)
    if len(trajectory.time_s) < 2:
        raise ValueError("The launch state is already transparent; choose optically thick ejecta")
    config = npc_config or NpcConfig(electron_fraction=model.electron_fraction)
    sampled = sample_solution(jet, trajectory, count=sample_count)
    npc = evaluate_npc_inputs(sampled, ejecta, config=config)
    return Run(model, ejecta, jet, trajectory, config, npc, time_step_s, max_duration_s)


def table(run: Run) -> pd.DataFrame:
    """All exported NPC quantities, with units attached in DataFrame.attrs."""
    result = pd.DataFrame({field.name: getattr(run.npc, field.name) for field in fields(run.npc)})
    result.attrs["units"] = NPC_UNITS.copy()
    return result


def summary(runs: list[Run]) -> pd.DataFrame:
    """Endpoint summary; NaN breakout times indicate the horizon was reached."""
    return pd.DataFrame(
        [
            dict(
                model=run.model.name,
                broke_out=run.trajectory.broke_out,
                breakout_time_s=run.trajectory.breakout_time_s,
                sample_time_s=run.npc.time_s[-1],
                gamma_rel=run.npc.relative_lorentz_factor[-1],
                n_p_cm3=run.npc.proton_number_density_cm3[-1],
                n_n_cm3=run.npc.free_neutron_number_density_cm3[-1],
                tau_n_on_p=run.npc.neutron_to_proton_optical_depth[-1],
                tau_p_on_n=run.npc.proton_to_neutron_optical_depth[-1],
            )
            for run in runs
        ]
    )


def plot_run(run: Run) -> plt.Figure:
    """Ejecta density/speed/opacity at three times, plus jet and cocoon history."""
    fig, axes = plt.subplots(3, 3, figsize=(13, 10), constrained_layout=True)
    tr, ej = run.trajectory, run.ejecta
    for time in np.linspace(tr.time_s[0], tr.time_s[-1], 3):
        radius = np.geomspace(
            ej.inner_radius(time) * 1.001, ej.optical_depth_outer_radius(time) * 0.999, 80
        )
        density = ej.density(radius, time)
        keep = density > max(density.max() * 1e-12, run.npc.upstream_density_g_cm3[-1] * 1e-3)
        axes[0, 0].loglog(radius[keep], density[keep], label=f"{time:.2f} s")
        axes[0, 1].plot(radius[keep], np.asarray(ej.velocity(radius[keep], time)) / 2.99792458e10)
        # This is grey exterior opacity, distinct from local nucleon depths.
        optical = [ej.optical_depth(float(r), time, samples=128) for r in radius[keep]]
        axes[0, 2].loglog(radius[keep], optical)
    for ax, label in zip(
        axes[0],
        (r"density [g cm$^{-3}$]", r"ambient $\beta$", r"grey exterior $\tau$"),
        strict=True,
    ):
        ax.set(xlabel="radius [cm]", ylabel=label)
    axes[0, 0].legend(fontsize=8)
    axes[0, 2].axhline(1.0, color="gray", linestyle=":")
    nominal = getattr(ej, "nominal_outer_radius", ej.outer_radius)
    axes[1, 0].plot(tr.time_s, tr.radius_cm, label="head")
    axes[1, 0].plot(tr.time_s, [nominal(t) for t in tr.time_s], "--", label="nominal edge")
    axes[1, 0].legend(fontsize=8)
    histories = (
        (axes[1, 1], tr.head_beta, "head speed beta", False),
        (axes[1, 2], tr.cocoon_radius_cm, "cocoon radius [cm]", True),
        (axes[2, 0], tr.cocoon_energy_erg, "cocoon energy [erg]", True),
        (axes[2, 1], tr.cocoon_pressure_erg_cm3, r"cocoon pressure [erg cm$^{-3}$]", True),
        (axes[2, 2], np.rad2deg(tr.jet_opening_angle_rad), "jet opening angle [deg]", False),
    )
    axes[1, 0].set(ylabel="radius [cm]")
    for ax, values, label, logarithmic in histories:
        values = np.asarray(values)
        mask = values > 0 if logarithmic else np.ones(values.shape, dtype=bool)
        ax.plot(tr.time_s[mask], values[mask])
        ax.set(ylabel=label, yscale="log" if logarithmic and mask.any() else "linear")
    for ax in axes[1:].ravel():
        ax.set(xlabel="lab time since merger [s]")
    for ax in axes.ravel():
        ax.grid(alpha=0.2)
    status = "breakout" if tr.broke_out else "horizon reached; no breakout"
    fig.suptitle(f"{run.model.name}: {status} — exploratory resolution")
    return fig


def plot_comparisons(
    runs: list[Run], normalized: bool = True, near_breakout: bool = False
) -> list[plt.Figure]:
    """All existing comparison panels, refreshed directly from in-memory runs.

    Normalization uses (t-t0)/(t_bo-t0); when any model has not broken out,
    show laboratory seconds for all models. Near-breakout view retains the
    final 20% of each completed trajectory and recomputes vertical limits.
    """
    if not runs:
        return []
    normalized = normalized and all(run.trajectory.broke_out for run in runs)
    figures = []
    for title, panels in PAGES.items():
        rows = (len(panels) + 2) // 3
        fig, axes = plt.subplots(
            rows, 3, figsize=(13, rows * 3.2 + 1), constrained_layout=True, squeeze=False
        )
        for ax, panel in zip(axes.ravel(), panels, strict=False):
            all_values = []
            for i, run in enumerate(runs):
                time = run.npc.time_s
                start = run.trajectory.time_s[0]
                end = run.trajectory.time_s[-1]
                fraction = (time - start) / (end - start)
                keep = fraction >= 0.8 if near_breakout else np.ones(time.shape, dtype=bool)
                x = fraction if normalized else time
                values = np.asarray(getattr(run.npc, panel.field))[keep]
                all_values.append(values)
                ax.plot(
                    x[keep],
                    values,
                    color=f"C{i % 10}",
                    linestyle="--" if run.model.kind == "numerical" else "-",
                    label=run.model.name,
                )
            scale = panel.scale
            if scale == "log" and any(np.any(values == 0) for values in all_values):
                scale = "symlog"  # Ye=0 or a zero free-proton fraction is physically allowed.
            _set_scale(ax, scale, all_values)
            if scale == "symlog":
                ax.set_ylim(bottom=0)
            ax.set(
                xlabel=r"$(t-t_0)/(t_{\rm bo}-t_0)$" if normalized else "lab time [s]",
                ylabel=panel.label,
            )
            ax.grid(alpha=0.2)
        for ax in axes.ravel()[len(panels) :]:
            ax.set_visible(False)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=3)
        fig.suptitle(title.split("_", 1)[1].replace("_", " ").title())
        figures.append(fig)
    return figures


def save_run(run: Run, output_dir: Path) -> list[Path]:
    """Save exploratory HDF5 and plots in a fresh directory; never overwrite.

    This is not the resolution-tested multi-model transfer format. Reproduction
    metadata includes model, engine, NPC, cocoon and numerical-kernel settings.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    npc_path = output_dir / "npc_inputs.h5"
    run.npc.to_hdf5(
        npc_path,
        config=run.config,
        metadata={"status": "exploratory; no convergence certification"},
    )
    dynamics_path = output_dir / "jet_cocoon.h5"
    with h5py.File(dynamics_path, "w") as handle:
        handle.attrs["status"] = "exploratory; no convergence certification"
        handle.attrs["broke_out"] = run.trajectory.broke_out
        handle.attrs["breakout_time_s"] = run.trajectory.breakout_time_s or np.nan
        handle.attrs["time_step_s"] = run.time_step_s
        handle.attrs["max_duration_s"] = run.max_duration_s
        handle.attrs["sample_count"] = len(run.npc.time_s)
        handle.attrs["sample_fraction_of_trajectory"] = 0.99
        for name, values in (
            ("model", asdict(run.model)),
            ("engine", asdict(run.jet.engine)),
            ("npc_configuration", asdict(run.config)),
            ("cocoon_configuration", asdict(run.jet.cocoon)),
        ):
            group = handle.create_group(name)
            for key, value in values.items():
                if value is not None:
                    group.attrs[key] = value
        for name in (
            "kernel_shape",
            "beta_width",
            "cutoff_mode",
            "integration_samples",
            "history_subsamples",
        ):
            if hasattr(run.ejecta, name):
                handle.attrs[name] = getattr(run.ejecta, name)
        for name, unit in {
            "time_s": "s",
            "radius_cm": "cm",
            "head_beta": "1",
            "ambient_beta": "1",
            **COCOON_FIELDS,
        }.items():
            ds = handle.create_dataset(name, data=getattr(run.trajectory, name))
            ds.attrs["unit"] = unit
        group = handle.create_group("endpoint_radial_columns")
        for key, value in snapshot_columns(
            run.ejecta, run.config, run.npc.radius_cm[-1], run.npc.time_s[-1]
        ).items():
            group[key] = value
    with h5py.File(npc_path, "r+") as handle:
        for name in NPC_UNITS:
            handle[name].attrs["frame"] = field_frame(name)
    png = output_dir / "ejecta_jet_cocoon.png"
    figure = plot_run(run)
    figure.savefig(png, dpi=150)
    plt.close(figure)
    pdf = output_dir / "npc_quantities.pdf"
    with PdfPages(pdf) as pages:
        for figure in plot_comparisons([run]):
            pages.savefig(figure)
            plt.close(figure)
    return [npc_path, dynamics_path, png, pdf]


def launch_dashboard(data_root: Path):
    """Widget controls that solve and refresh after a slider is released.

    Auto-update replaces the latest run so parameter exploration does not fill
    the comparison tabs with every intermediate trial.  The explicit button
    retains the latest run and adds a newly named model for comparisons.
    """
    from time import perf_counter

    import ipywidgets as widgets
    from IPython.display import FileLink, clear_output, display

    presets = {model.name: model for model in (*ANALYTICAL, *NUMERICAL)}
    preset = widgets.Dropdown(options=list(presets), description="Preset")
    controls = {
        "name": widgets.Text(value="trial_1", description="Name"),
        "mass": widgets.FloatLogSlider(
            value=1e-4, min=-6, max=-1, description="Mass [Msun]", continuous_update=False
        ),
        "power": widgets.FloatLogSlider(
            value=1e51, min=48, max=53, description="L_iso [erg/s]", continuous_update=False
        ),
        "launch": widgets.FloatSlider(
            value=1.0, min=0.1, max=10.0, description="Launch [s]", continuous_update=False
        ),
        "tail": widgets.FloatSlider(
            value=8.0,
            min=2.0,
            max=24.0,
            description="Tail exponent",
            continuous_update=False,
        ),
        "ye": widgets.FloatSlider(
            value=0.3,
            min=0.0,
            max=0.5,
            step=0.01,
            description="Ye fallback",
            continuous_update=False,
        ),
        "gamma": widgets.FloatSlider(
            value=100.0,
            min=2.0,
            max=200.0,
            description="Jet Gamma",
            continuous_update=False,
        ),
        "theta": widgets.FloatSlider(
            value=10.0,
            min=1.0,
            max=30.0,
            description="Angle [deg]",
            continuous_update=False,
        ),
        "kernel": widgets.Dropdown(
            options=[("Gaussian", 2.0), ("Exponential", 1.0)], description="Kernel"
        ),
        "width": widgets.FloatSlider(
            value=0.05,
            min=0.01,
            max=0.1,
            step=0.005,
            description="Beta width",
            continuous_update=False,
        ),
        "cutoff": widgets.Dropdown(options=["smooth", "sharp"], description="Cutoff"),
        "dt": widgets.FloatText(value=0.01, description="Max dt [s]"),
        "horizon": widgets.FloatText(value=15.0, description="Horizon [s]"),
    }
    near = widgets.Checkbox(value=False, description="Final 20% only")
    auto_update = widgets.Checkbox(value=True, description="Update after slider release")
    run_button = widgets.Button(description="Run and add model", button_style="primary")
    clear_button = widgets.Button(description="Clear comparisons")
    save_button = widgets.Button(description="Save latest run")
    activity = widgets.HTML(
        value="<b>Ready.</b> Numerical models can take about 20 seconds per update."
    )
    status = widgets.Output()
    output = widgets.Output()
    runs: list[Run] = []
    choosing_preset = False

    def choose(change):
        nonlocal choosing_preset
        choosing_preset = True
        model = presets[change["new"]]
        try:
            for key, value in (
                ("mass", model.mass_msun),
                ("power", model.luminosity_iso_erg_s),
                ("launch", model.launch_s),
                ("tail", model.tail_exponent),
                ("ye", model.electron_fraction),
            ):
                controls[key].value = value
            controls["name"].value = model.name + "_trial"
            for key in ("mass", "tail", "ye"):
                controls[key].disabled = model.kind == "numerical"
            for key in ("kernel", "width", "cutoff"):
                controls[key].disabled = model.kind != "numerical"
        finally:
            choosing_preset = False
        if runs and auto_update.value:
            execute(None, replace_latest=True)

    def refresh(*_):
        with output:
            clear_output(wait=True)
            if not runs:
                print("Run a model to see results.")
                return
            display(summary(runs))
            tabs = widgets.Tab()
            boxes = [widgets.Output() for _ in range(6)]
            tabs.children = boxes
            for i, title in enumerate(
                (
                    "Ejecta + cocoon",
                    "Kinematics",
                    "Composition",
                    "Interactions",
                    "Energy limits",
                    "Data table",
                )
            ):
                tabs.set_title(i, title)
            display(tabs)
            with boxes[0]:
                figure = plot_run(runs[-1])
                display(figure)
                plt.close(figure)
            figures = plot_comparisons(runs, near_breakout=near.value)
            for box, figure in zip(boxes[1:5], figures, strict=True):
                with box:
                    display(figure)
                plt.close(figure)
            with boxes[-1]:
                display(table(runs[-1]))

    def execute(_, *, replace_latest: bool = False):
        run_button.disabled = True
        original_button_description = run_button.description
        run_button.description = "Solving..."
        started = perf_counter()
        activity.value = (
            "<b>Solving jet + cocoon…</b> The existing plots remain visible until "
            "the new numerical result is complete."
        )
        with status:
            clear_output()
            print(
                "Solving jet + cocoon; plots refresh after completion. "
                "Numerical profiles are slower.",
                flush=True,
            )
            try:
                model = replace(
                    presets[preset.value],
                    name=controls["name"].value,
                    mass_msun=controls["mass"].value,
                    luminosity_iso_erg_s=controls["power"].value,
                    launch_s=controls["launch"].value,
                    tail_exponent=controls["tail"].value,
                    electron_fraction=controls["ye"].value,
                )
                if not replace_latest and any(run.model.name == model.name for run in runs):
                    raise ValueError("Use a different name for each comparison model")
                result = run_model(
                    model,
                    data_root,
                    lorentz_factor=controls["gamma"].value,
                    opening_angle_deg=controls["theta"].value,
                    time_step_s=controls["dt"].value,
                    max_duration_s=controls["horizon"].value,
                    kernel_shape=controls["kernel"].value,
                    beta_width=controls["width"].value,
                    cutoff_mode=controls["cutoff"].value,
                )
                if replace_latest and runs:
                    runs[-1] = result
                else:
                    runs.append(result)
                refresh()
                elapsed = perf_counter() - started
                activity.value = f"<b>Updated plots in {elapsed:.1f} s.</b>"
                print("Finished. Exploratory resolution; inspect the breakout flag above.")
            except Exception as error:
                activity.value = "<b>Update failed.</b> See the message below."
                print(f"{type(error).__name__}: {error}")
                if runs:
                    print("Plots still show the previous successful run.")
            finally:
                run_button.disabled = False
                run_button.description = original_button_description

    def update_from_control(_):
        if choosing_preset or not auto_update.value or not runs:
            return
        execute(None, replace_latest=True)

    def clear(_):
        runs.clear()
        refresh()
        activity.value = "<b>Ready.</b> Run a model to create the first plot."

    def save(_):
        from datetime import datetime, timezone

        with status:
            if not runs:
                print("Run a model first.")
                return
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
            destination = Path(__file__).parent / "output/npc_playground" / stamp
            for path in save_run(runs[-1], destination):
                display(FileLink(relpath(path, Path.cwd())))

    preset.observe(choose, names="value")
    choose({"new": preset.value})
    for key, control in controls.items():
        if key != "name":
            control.observe(update_from_control, names="value")
    near.observe(refresh, names="value")
    run_button.on_click(execute)
    clear_button.on_click(clear)
    save_button.on_click(save)
    return widgets.VBox(
        [
            preset,
            widgets.GridBox(
                list(controls.values()),
                layout=widgets.Layout(grid_template_columns="repeat(2, 1fr)"),
            ),
            widgets.HBox([auto_update, near]),
            widgets.HBox([run_button, clear_button, save_button]),
            activity,
            status,
            output,
        ]
    )
