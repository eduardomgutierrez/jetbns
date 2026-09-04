"""Deterministic inputs for an external neutron--proton converter Monte Carlo.

This module evaluates the shock and upstream quantities in the local NPC notes
from an already solved :class:`~jetbns.PropagationResult`.  It deliberately
does not simulate particles, collisions, conversions, or radiation spectra.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
from numpy.typing import ArrayLike, NDArray

from .constants import (
    BOLTZMANN_CONSTANT,
    ELECTRON_MASS,
    ELEMENTARY_CHARGE,
    PROTON_MASS,
    RADIATION_CONSTANT,
    SOLAR_MASS,
    SPEED_OF_LIGHT,
)
from .ejecta import Ejecta, lorentz_factor
from .propagation import PropagationResult

FloatArray = NDArray[np.float64]
PathLength = Literal["radius", "remaining_ejecta"]

PN_CROSS_SECTION = 3.0e-26  # cm^2
PN_INELASTICITY = 0.5
BETHE_HEITLER_WIEN_FACTOR = 6.0
DEFAULT_MAGNETIC_FIELD_G = 1.0e13
DEFAULT_MAGNETIC_RADIUS_CM = 1.0e6
ERG_PER_KEV = 1.602176634e-9


@dataclass(frozen=True)
class NpcConfig:
    """Physical choices used to derive NPC inputs.

    ``path_length="radius"`` and ``target_nucleon_fraction=1`` reproduce the
    total-baryon reference in the notes and legacy implementation. The
    species-resolved estimate instead uses ejecta ``Y_e`` when available (or
    ``electron_fraction`` otherwise) and the Metzger et al. neutron-skin model.
    The alternative path is the radial distance to the ejecta outer boundary.
    """

    magnetic_field_at_reference_g: float = DEFAULT_MAGNETIC_FIELD_G
    magnetic_reference_radius_cm: float = DEFAULT_MAGNETIC_RADIUS_CM
    pn_cross_section_cm2: float = PN_CROSS_SECTION
    pn_inelasticity: float = PN_INELASTICITY
    target_nucleon_fraction: float = 1.0
    path_length: PathLength = "radius"
    bethe_heitler_wien_factor: float = BETHE_HEITLER_WIEN_FACTOR
    electron_fraction: float = 0.1
    free_neutron_transition_mass_msun: float = 1.0e-4
    free_neutron_decay_time_s: float = 900.0

    def __post_init__(self) -> None:
        positive = (
            self.magnetic_field_at_reference_g,
            self.magnetic_reference_radius_cm,
            self.pn_cross_section_cm2,
            self.bethe_heitler_wien_factor,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("magnetic, cross-section, radius, and Wien values must be positive")
        if not 0 < self.pn_inelasticity <= 1:
            raise ValueError("pn_inelasticity must lie in (0, 1]")
        if not 0 < self.target_nucleon_fraction <= 1:
            raise ValueError("target_nucleon_fraction must lie in (0, 1]")
        if self.path_length not in ("radius", "remaining_ejecta"):
            raise ValueError("path_length must be 'radius' or 'remaining_ejecta'")
        if not 0 <= self.electron_fraction <= 0.5:
            raise ValueError("electron_fraction must lie in [0, 0.5]")
        if self.free_neutron_transition_mass_msun <= 0 or self.free_neutron_decay_time_s <= 0:
            raise ValueError("free-neutron mass and decay time must be positive")


NPC_UNITS = {
    "time_s": "s",
    "radius_cm": "cm",
    "head_beta": "1",
    "ambient_beta": "1",
    "head_lorentz_factor": "1",
    "ambient_lorentz_factor": "1",
    "relative_lorentz_factor": "1",
    "upstream_density_g_cm3": "g cm^-3",
    "upstream_number_density_cm3": "cm^-3",
    "proton_number_density_cm3": "cm^-3",
    "free_neutron_number_density_cm3": "cm^-3",
    "upstream_magnetic_field_g": "G",
    "path_length_cm": "cm",
    "pn_optical_depth": "1",
    "neutron_to_proton_optical_depth": "1",
    "proton_to_neutron_optical_depth": "1",
    "gyration_parameter": "1",
    "downstream_temperature_k": "K",
    "downstream_temperature_kev": "keV",
    "max_lorentz_factor_bethe_heitler": "1",
    "max_lorentz_factor_gyration": "1",
    "max_lorentz_factor": "1",
    "observer_lorentz_factor": "1",
    "max_observer_energy_erg": "erg",
}


@dataclass(frozen=True)
class NpcInputs:
    """Vectorized NPC input table; field units are given by :data:`NPC_UNITS`."""

    time_s: FloatArray
    radius_cm: FloatArray
    head_beta: FloatArray
    ambient_beta: FloatArray
    head_lorentz_factor: FloatArray
    ambient_lorentz_factor: FloatArray
    relative_lorentz_factor: FloatArray
    upstream_density_g_cm3: FloatArray
    upstream_number_density_cm3: FloatArray
    proton_number_density_cm3: FloatArray
    free_neutron_number_density_cm3: FloatArray
    upstream_magnetic_field_g: FloatArray
    path_length_cm: FloatArray
    pn_optical_depth: FloatArray
    neutron_to_proton_optical_depth: FloatArray
    proton_to_neutron_optical_depth: FloatArray
    gyration_parameter: FloatArray
    downstream_temperature_k: FloatArray
    downstream_temperature_kev: FloatArray
    max_lorentz_factor_bethe_heitler: FloatArray
    max_lorentz_factor_gyration: FloatArray
    max_lorentz_factor: FloatArray
    observer_lorentz_factor: FloatArray
    max_observer_energy_erg: FloatArray

    def to_hdf5(
        self,
        path: str | Path,
        *,
        config: NpcConfig,
        metadata: Mapping[str, str | float | int | bool] | None = None,
    ) -> None:
        """Write a portable HDF5 table with per-dataset units and assumptions."""
        with h5py.File(path, "w") as handle:
            handle.attrs["description"] = "Deterministic inputs for an external NPC Monte Carlo"
            handle.attrs["schema"] = "jetbns.npc-inputs.v3"
            handle.attrs["source_equations"] = (
                "Kashiyama, Murase & Meszaros (2013), equation 7; "
                "Metzger et al. (2015), free-neutron skin prescription; "
                "local NPC notes, equations 31, 33, 38-43"
            )
            for field in fields(self):
                dataset = handle.create_dataset(field.name, data=getattr(self, field.name))
                dataset.attrs["unit"] = NPC_UNITS[field.name]
            group = handle.create_group("configuration")
            for field in fields(config):
                group.attrs[field.name] = getattr(config, field.name)
            if metadata:
                group = handle.create_group("metadata")
                for key, value in metadata.items():
                    group.attrs[key] = value


def relative_lorentz_factor(head_beta: ArrayLike, ambient_beta: ArrayLike) -> FloatArray:
    r"""Return ``Gamma_h Gamma_e (1 - beta_h beta_e)`` (notes equation 31)."""
    head = np.asarray(head_beta, dtype=float)
    ambient = np.asarray(ambient_beta, dtype=float)
    return np.asarray(lorentz_factor(head) * lorentz_factor(ambient) * (1.0 - head * ambient))


def metzger_free_neutron_fraction(
    electron_fraction: ArrayLike,
    exterior_mass_msun: ArrayLike,
    time_s: ArrayLike,
    *,
    transition_mass_msun: float = 1.0e-4,
    decay_time_s: float = 900.0,
) -> FloatArray:
    r"""Estimate the surviving free-neutron mass fraction in the outer skin.

    This implements the schematic Metzger et al. (2015) mass-coordinate
    prescription, including beta decay. It is not a reaction-network result.
    """
    ye = np.asarray(electron_fraction, dtype=float)
    mass = np.asarray(exterior_mass_msun, dtype=float)
    time = np.asarray(time_s, dtype=float)
    skin = (2.0 / np.pi) * np.arctan(transition_mass_msun / np.maximum(mass, np.finfo(float).tiny))
    return np.maximum(0.0, 1.0 - 2.0 * ye) * skin * np.exp(-time / decay_time_s)


def evaluate_npc_inputs(
    result: PropagationResult,
    ejecta: Ejecta,
    *,
    config: NpcConfig | None = None,
    observer_lorentz_factor: ArrayLike | None = None,
    include_breakout: bool = False,
) -> NpcInputs:
    r"""Derive NPC quantities along a solved jet-head trajectory.

    This evaluates notes equations 31, 33, 38--43: ``Gamma_rel`` from the
    collinear relative motion, ``tau_pn = n sigma Delta-r / Gamma_e``,
    ``B = B0 (r0/r)^2``, the published ``xi(1) = e B / (sigma_pn m_p c^2 n)``,
    a radiation-dominated shock temperature, and the Bethe--Heitler/gyration
    energy limits.  The original paper's equation 7 is used instead of the
    extra factor of ``c`` accidentally present in the legacy code and notes.

    The breakout sample is excluded by default because the upstream column ends
    there.  When no breakout occurred, all samples are retained.  A supplied
    observer Lorentz factor may be scalar or broadcastable to the retained
    trajectory; otherwise the jet-head Lorentz factor is used.
    """
    if config is None:
        config = NpcConfig()
    stop = -1 if result.broke_out and not include_breakout else None
    time = np.asarray(result.time_s[:stop], dtype=float)
    radius = np.asarray(result.radius_cm[:stop], dtype=float)
    head_beta = np.asarray(result.head_beta[:stop], dtype=float)
    ambient_beta = np.asarray(result.ambient_beta[:stop], dtype=float)
    if time.size == 0:
        raise ValueError("propagation result contains no pre-breakout samples")
    if not (time.shape == radius.shape == head_beta.shape == ambient_beta.shape):
        raise ValueError("propagation arrays must have identical shapes")

    density = np.asarray(
        [ejecta.density(float(r), float(t)) for r, t in zip(radius, time, strict=True)],
        dtype=float,
    )
    if np.any(density <= 0):
        raise ValueError("trajectory must remain inside positive-density ejecta")
    head_gamma = np.asarray(lorentz_factor(head_beta))
    ambient_gamma = np.asarray(lorentz_factor(ambient_beta))
    relative_gamma = relative_lorentz_factor(head_beta, ambient_beta)
    number_density = density * config.target_nucleon_fraction / PROTON_MASS
    try:
        electron_fraction = np.asarray(
            [
                ejecta.electron_fraction(float(r), float(t))
                for r, t in zip(radius, time, strict=True)
            ]
        )
    except (AttributeError, ValueError):
        electron_fraction = np.full_like(radius, config.electron_fraction)
    magnetic_field = (
        config.magnetic_field_at_reference_g * (config.magnetic_reference_radius_cm / radius) ** 2
    )
    if config.path_length == "radius":
        path_length = radius.copy()
    else:
        path_length = np.asarray(
            [
                ejecta.optical_depth_outer_radius(float(t)) - r
                for r, t in zip(radius, time, strict=True)
            ]
        )
        if np.any(path_length < 0):
            raise ValueError("trajectory extends beyond the ejecta outer boundary")

    optical_depth = number_density * config.pn_cross_section_cm2 * path_length / ambient_gamma
    exterior_mass = np.asarray(
        [
            ejecta.mass_above(float(r), float(t), samples=128) / SOLAR_MASS
            for r, t in zip(radius, time, strict=True)
        ]
    )
    free_neutron_fraction = metzger_free_neutron_fraction(
        electron_fraction,
        exterior_mass,
        time,
        transition_mass_msun=config.free_neutron_transition_mass_msun,
        decay_time_s=config.free_neutron_decay_time_s,
    )
    proton_density = electron_fraction * density / PROTON_MASS
    neutron_density = free_neutron_fraction * density / PROTON_MASS
    neutron_to_proton_depth = (
        proton_density * config.pn_cross_section_cm2 * path_length / ambient_gamma
    )
    proton_to_neutron_depth = (
        neutron_density * config.pn_cross_section_cm2 * path_length / ambient_gamma
    )
    gyration = (
        ELEMENTARY_CHARGE
        * magnetic_field
        / (PROTON_MASS * SPEED_OF_LIGHT**2 * number_density * config.pn_cross_section_cm2)
    )
    temperature = (density * SPEED_OF_LIGHT**2 * relative_gamma**2 / RADIATION_CONSTANT) ** 0.25
    temperature_kev = BOLTZMANN_CONSTANT * temperature / ERG_PER_KEV
    max_bh = (
        2.0
        * ELECTRON_MASS
        * SPEED_OF_LIGHT**2
        / (config.bethe_heitler_wien_factor * BOLTZMANN_CONSTANT * temperature)
    )
    max_gyration = gyration.copy()
    maximum = np.minimum(max_bh, max_gyration)

    if observer_lorentz_factor is None:
        observer_gamma = head_gamma.copy()
    else:
        try:
            observer_gamma = np.broadcast_to(
                np.asarray(observer_lorentz_factor, dtype=float), time.shape
            ).copy()
        except ValueError as error:
            raise ValueError(
                "observer_lorentz_factor is not broadcastable to trajectory"
            ) from error
        if np.any(observer_gamma < 1):
            raise ValueError("observer_lorentz_factor must be at least one")
    observer_energy = observer_gamma * maximum * PROTON_MASS * SPEED_OF_LIGHT**2

    return NpcInputs(
        time,
        radius,
        head_beta,
        ambient_beta,
        head_gamma,
        ambient_gamma,
        relative_gamma,
        density,
        number_density,
        proton_density,
        neutron_density,
        magnetic_field,
        path_length,
        optical_depth,
        neutron_to_proton_depth,
        proton_to_neutron_depth,
        gyration,
        temperature,
        temperature_kev,
        max_bh,
        max_gyration,
        maximum,
        observer_gamma,
        observer_energy,
    )
