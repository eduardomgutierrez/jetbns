"""Compare all exported NPC model time series on a normalized time axis.

The input is the HDF5 file written by ``export_npc_models.py``. Each panel
shows one physical quantity, with every exported model overlaid. Time is
normalized between the first retained jet-head state and shock breakout so
models with different launch and breakout times can be compared directly.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "examples/output/npc_model_export/npc_prebreakout_models.h5"
DEFAULT_OUTPUT = ROOT / "examples/output/npc_model_export/comparison"


@dataclass(frozen=True)
class Panel:
    """Plotting metadata for one exported time series."""

    field: str
    label: str
    scale: str = "linear"


PAGES = {
    "01_kinematics": (
        Panel("radius_cm", r"head radius $r_h$ [cm]", "log"),
        Panel("head_beta", r"head speed $\beta_h$"),
        Panel("ambient_beta", r"ambient speed $\beta_a$"),
        Panel("head_lorentz_factor", r"head Lorentz factor $\Gamma_h$", "log"),
        Panel("ambient_lorentz_factor", r"ambient Lorentz factor $\Gamma_a$"),
        Panel("relative_lorentz_factor", r"relative Lorentz factor $\Gamma_{\rm rel}$", "log"),
        Panel("shock_beta_upstream_frame", r"shock speed $\beta'_{s,u}$"),
        Panel("shock_beta_downstream_frame", r"shock speed $\beta'_{s,d}$"),
        Panel("hydrodynamic_compression_ratio", r"compression $n_d/n_u$", "log"),
    ),
    "02_upstream_composition": (
        Panel("upstream_density_g_cm3", r"upstream density $\rho_u$ [g cm$^{-3}$]", "log"),
        Panel("upstream_baryon_number_density_cm3", r"baryon density $n_b$ [cm$^{-3}$]", "log"),
        Panel("electron_fraction", r"electron fraction $Y_e$"),
        Panel("free_neutron_mass_fraction", r"free-neutron fraction $X_{n,\rm free}$", "symlog"),
        Panel("proton_number_density_cm3", r"proton target density $n_p$ [cm$^{-3}$]", "log"),
        Panel(
            "free_neutron_number_density_cm3",
            r"neutron target density $n_n$ [cm$^{-3}$]",
            "symlog",
        ),
        Panel("upstream_magnetic_field_g", r"upstream field $B_u$ [G]", "log"),
        Panel("exterior_mass_msun", r"exterior mass $M_{>r}$ [$M_\odot$]", "log"),
        Panel("effective_path_length_upstream_cm", r"upstream path $\ell_u$ [cm]", "log"),
    ),
    "03_interactions": (
        Panel("pn_optical_depth", r"total-baryon reference $\tau_{pn}$", "log"),
        Panel("neutron_to_proton_optical_depth", r"neutron on proton $\tau_{n:p}$", "log"),
        Panel("proton_to_neutron_optical_depth", r"proton on neutron $\tau_{p:n}$", "symlog"),
        Panel(
            "neutron_on_proton_collision_rate_s1",
            r"neutron--proton rate $\nu_{n:p}$ [s$^{-1}$]",
            "log",
        ),
        Panel(
            "proton_on_neutron_collision_rate_s1",
            r"proton--neutron rate $\nu_{p:n}$ [s$^{-1}$]",
            "symlog",
        ),
        Panel(
            "proton_gyrofrequency_at_gamma_one_s1",
            r"proton gyrofrequency $\omega_{g,1}$ [s$^{-1}$]",
            "log",
        ),
    ),
    "04_acceleration_limits": (
        Panel("downstream_temperature_kev", r"downstream $k_B T_d$ [keV]", "log"),
        Panel("gyration_parameter", r"gyration parameter $\xi(1)$", "log"),
        Panel("max_lorentz_factor_bethe_heitler", r"Bethe--Heitler limit $\gamma_{\rm BH}$", "log"),
        Panel("max_lorentz_factor_gyration", r"gyration limit $\gamma_{\rm gyr}$", "log"),
        Panel("max_lorentz_factor", r"adopted maximum $\gamma_{\max}$", "log"),
        Panel("observer_lorentz_factor", r"observer boost $\Gamma_{\rm obs}$", "log"),
        Panel("max_observer_energy_erg", r"maximum observer energy [erg]", "log"),
    ),
}


def _normalized_time(group: h5py.Group) -> np.ndarray:
    """Return elapsed time normalized from the first sample to breakout."""
    time = np.asarray(group["time_s"][:])
    breakout = float(group.attrs["breakout_time_s"])
    duration = breakout - time[0]
    if duration <= 0:
        raise ValueError(f"{group.name}: breakout must follow the first sample")
    return (time - time[0]) / duration


def _set_scale(axis: plt.Axes, scale: str, all_values: list[np.ndarray]) -> None:
    if scale == "log":
        axis.set_yscale("log")
    elif scale == "symlog":
        positive = np.concatenate([values[values > 0] for values in all_values])
        threshold = float(positive.min() / 10) if positive.size else 1.0
        axis.set_yscale("symlog", linthresh=threshold)


def plot_comparison(input_path: Path, output_dir: Path) -> list[Path]:
    """Write four PNG pages and a matching multipage PDF."""
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    pdf_path = output_dir / "npc_all_models_comparison.pdf"
    with h5py.File(input_path) as handle, PdfPages(pdf_path) as pdf:
        if handle.attrs.get("schema") != "jetbns.npc-model-export.v3":
            raise ValueError("Expected a jetbns.npc-model-export.v3 file")
        names = list(handle["prebreakout_summary/name"].asstr()[:])
        groups = {name: handle[f"models/{name}"] for name in names}
        colors = plt.get_cmap("tab10")(np.linspace(0, 0.8, len(names)))

        for page_name, panels in PAGES.items():
            rows = (len(panels) + 2) // 3
            fig, axes = plt.subplots(
                rows,
                3,
                figsize=(13, 3.2 * rows + 1),
                constrained_layout=True,
                squeeze=False,
            )
            flat_axes = axes.ravel()
            for axis, panel in zip(flat_axes, panels, strict=False):
                values_for_scale = []
                for color, (name, group) in zip(colors, groups.items(), strict=True):
                    values = np.asarray(group[panel.field][:])
                    values_for_scale.append(values)
                    linestyle = "--" if group.attrs["kind"] == "numerical" else "-"
                    axis.plot(
                        _normalized_time(group),
                        values,
                        color=color,
                        linestyle=linestyle,
                        linewidth=1.7,
                        label=name,
                    )
                _set_scale(axis, panel.scale, values_for_scale)
                axis.set(
                    xlabel=r"normalized time $(t-t_0)/(t_{\rm bo}-t_0)$",
                    ylabel=panel.label,
                    xlim=(0, 1),
                )
                axis.grid(alpha=0.25, which="both")
            for axis in flat_axes[len(panels) :]:
                axis.set_visible(False)
            handles, labels = flat_axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=9)
            fig.suptitle(page_name.split("_", 1)[1].replace("_", " ").title())
            png_path = output_dir / f"{page_name}.png"
            fig.savefig(png_path, dpi=180)
            pdf.savefig(fig)
            plt.close(fig)
            written.append(png_path)
    written.append(pdf_path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    for path in plot_comparison(args.input, args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
