"""Ejecta profiles used by the jet-propagation model.

All public methods use CGS units. Input velocities are dimensionless ``beta =
v/c``. Masses supplied to constructors are in solar masses for convenience.
The density is an isotropic-equivalent density: integrating over ``4 pi`` gives
the isotropic-equivalent mass in the selected angular direction.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import gamma
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .constants import SOLAR_MASS, SPEED_OF_LIGHT

FloatArray = NDArray[np.float64]


def _array(value: ArrayLike) -> FloatArray:
    return np.asarray(value, dtype=float)


def _return_like_input(value: FloatArray, original: ArrayLike) -> float | FloatArray:
    return float(value) if np.ndim(original) == 0 else value


def lorentz_factor(beta: ArrayLike) -> float | FloatArray:
    """Return the Lorentz factor for a dimensionless velocity ``beta``."""
    beta_array = _array(beta)
    if np.any(np.abs(beta_array) >= 1):
        raise ValueError("beta must satisfy |beta| < 1")
    result = 1.0 / np.sqrt(1.0 - beta_array**2)
    return _return_like_input(result, beta)


class Ejecta(ABC):
    """Common interface for spherically represented ejecta profiles."""

    @abstractmethod
    def inner_radius(self, time: float) -> float:
        """Return the inner profile radius in cm."""

    @abstractmethod
    def outer_radius(self, time: float) -> float:
        """Return the nominal outer profile radius in cm."""

    @abstractmethod
    def density(self, radius: ArrayLike, time: float) -> float | FloatArray:
        """Return rest-mass density in g cm^-3."""

    @abstractmethod
    def velocity(self, radius: ArrayLike, time: float) -> float | FloatArray:
        """Return radial velocity in cm s^-1."""

    def mass(self, time: float, *, samples: int = 4096) -> float:
        """Numerically integrate the isotropic-equivalent ejecta mass in g."""
        if samples < 2:
            raise ValueError("samples must be at least 2")
        inner = self.inner_radius(time)
        outer = self.outer_radius(time)
        if outer <= inner:
            raise ValueError("time precedes formation of a valid ejecta shell")
        radius = np.geomspace(inner, outer, samples)
        integrand = 4.0 * np.pi * radius**2 * self.density(radius, time)
        return float(np.trapezoid(integrand, radius))

    def optical_depth(
        self, radius: float, time: float, *, opacity: float = 0.16, samples: int = 2048
    ) -> float:
        """Integrate grey optical depth from ``radius`` to the outer boundary.

        ``opacity`` is in cm^2 g^-1. The default retains the legacy electron-
        scattering value used in the published calculations.
        """
        outer = self.outer_radius(time)
        if radius >= outer:
            return 0.0
        if radius < self.inner_radius(time):
            raise ValueError("radius lies below the ejecta inner boundary")
        grid = np.geomspace(radius, outer, samples)
        return float(np.trapezoid(opacity * self.density(grid, time), grid))


@dataclass(frozen=True)
class HomologousPowerLaw(Ejecta):
    """Homologously expanding, single-power-law ejecta.

    This is the cleaned equivalent of ``EjectaAnalytical`` in ``jetBNS3`` and
    the Hamidani--Ioka profile used as a simple jet-propagation benchmark.
    """

    mass_msun: float = 0.002
    density_index: float = 2.0
    max_beta: float = 0.34641
    inner_radius_cm: float = 1.0e6
    initial_outer_radius_cm: float = 1.67e9
    reference_time_s: float = 0.0

    def __post_init__(self) -> None:
        if self.mass_msun <= 0:
            raise ValueError("mass_msun must be positive")
        if not 0 < self.max_beta < 1:
            raise ValueError("max_beta must lie between 0 and 1")
        if self.density_index >= 3:
            raise ValueError("density_index must be below 3 for this normalization")
        if not 0 < self.inner_radius_cm < self.initial_outer_radius_cm:
            raise ValueError("radii must satisfy 0 < inner < initial outer")

    @property
    def mass_g(self) -> float:
        return self.mass_msun * SOLAR_MASS

    def inner_radius(self, time: float) -> float:
        return self.inner_radius_cm

    def outer_radius(self, time: float) -> float:
        result = self.initial_outer_radius_cm + self.max_beta * SPEED_OF_LIGHT * (
            time - self.reference_time_s
        )
        if result <= self.inner_radius_cm:
            raise ValueError("time precedes formation of a valid ejecta shell")
        return result

    def density(self, radius: ArrayLike, time: float) -> float | FloatArray:
        radius_array = _array(radius)
        outer = self.outer_radius(time)
        n = self.density_index
        normalization = (
            self.mass_g
            * (3.0 - n)
            / (
                4.0
                * np.pi
                * self.inner_radius_cm**n
                * (outer ** (3.0 - n) - self.inner_radius_cm ** (3.0 - n))
            )
        )
        density = normalization * (self.inner_radius_cm / radius_array) ** n
        inside = (radius_array >= self.inner_radius_cm) & (radius_array <= outer)
        result = np.where(inside, density, 0.0)
        return _return_like_input(result, radius)

    def velocity(self, radius: ArrayLike, time: float) -> float | FloatArray:
        radius_array = _array(radius)
        outer = self.outer_radius(time)
        result = self.max_beta * SPEED_OF_LIGHT * radius_array / outer
        result = np.where(
            (radius_array >= self.inner_radius_cm) & (radius_array <= outer), result, 0.0
        )
        return _return_like_input(result, radius)


@dataclass(frozen=True)
class BrokenPowerLaw(Ejecta):
    """Inner/outer power-law ejecta with an optional exponential fast tail.

    The sharp model is normalized exactly between the inner and outer radii.
    With ``tail=True``, the same core normalization is retained and the tail
    adds a small amount of mass beyond the nominal outer radius, matching the
    convention of the paper implementation. ``tail_extent`` controls only the
    finite numerical boundary, not the exponential scale.
    """

    mass_msun: float = 0.002
    inner_index: float = 2.0
    outer_index: float = 6.0
    break_beta: float = 0.3
    max_beta: float = 0.6
    inner_radius_cm: float = 1.0e6
    break_launch_time_s: float = 0.0
    max_launch_time_s: float = 0.0
    tail: bool = False
    tail_exponent: float = 4.0
    tail_extent: float = 3.0

    def __post_init__(self) -> None:
        if self.mass_msun <= 0:
            raise ValueError("mass_msun must be positive")
        if not 0 < self.break_beta < self.max_beta < 1:
            raise ValueError("velocities must satisfy 0 < break_beta < max_beta < 1")
        if self.inner_index >= 3 or self.outer_index <= 3:
            raise ValueError("indices must satisfy inner_index < 3 < outer_index")
        if self.inner_radius_cm <= 0 or self.tail_exponent <= 0 or self.tail_extent <= 1:
            raise ValueError("radii, tail_exponent, and tail_extent must be positive")

    @property
    def mass_g(self) -> float:
        return self.mass_msun * SOLAR_MASS

    def inner_radius(self, time: float) -> float:
        return self.inner_radius_cm

    def break_radius(self, time: float) -> float:
        return self.inner_radius_cm + self.break_beta * SPEED_OF_LIGHT * (
            time - self.break_launch_time_s
        )

    def nominal_outer_radius(self, time: float) -> float:
        return self.inner_radius_cm + self.max_beta * SPEED_OF_LIGHT * (
            time - self.max_launch_time_s
        )

    def outer_radius(self, time: float) -> float:
        nominal = self.nominal_outer_radius(time)
        if not self.inner_radius_cm < self.break_radius(time) < nominal:
            raise ValueError("time does not produce inner < break < outer radii")
        return self.tail_extent * nominal if self.tail else nominal

    def _normalization(self, time: float) -> float:
        break_radius = self.break_radius(time)
        nominal_outer = self.nominal_outer_radius(time)
        inner_term = (1.0 - (self.inner_radius_cm / break_radius) ** (3 - self.inner_index)) / (
            3 - self.inner_index
        )
        outer_term = (1.0 - (nominal_outer / break_radius) ** (3 - self.outer_index)) / (
            self.outer_index - 3
        )
        return self.mass_g / (4.0 * np.pi * break_radius**3 * (inner_term + outer_term))

    def density(self, radius: ArrayLike, time: float) -> float | FloatArray:
        radius_array = _array(radius)
        break_radius = self.break_radius(time)
        nominal_outer = self.nominal_outer_radius(time)
        outer = self.outer_radius(time)
        rho0 = self._normalization(time)
        inner_density = rho0 * (radius_array / break_radius) ** (-self.inner_index)
        outer_density = rho0 * (radius_array / break_radius) ** (-self.outer_index)
        if self.tail:
            outer_density *= np.where(
                radius_array > nominal_outer,
                np.exp(1.0 - (radius_array / nominal_outer) ** self.tail_exponent),
                1.0,
            )
        result = np.where(radius_array <= break_radius, inner_density, outer_density)
        result = np.where(
            (radius_array >= self.inner_radius_cm) & (radius_array <= outer), result, 0.0
        )
        return _return_like_input(result, radius)

    def velocity(self, radius: ArrayLike, time: float) -> float | FloatArray:
        radius_array = _array(radius)
        nominal_outer = self.nominal_outer_radius(time)
        # Retain the small non-homologous floor used in the paper code.
        delta = 0.05 * (time / 0.2) ** -0.5
        if not 0 <= delta < 1:
            raise ValueError("time is too early for the calibrated velocity prescription")
        beta = self.max_beta * np.sqrt(delta + (1.0 - delta) * (radius_array / nominal_outer) ** 2)
        result = np.minimum(beta, self.max_beta) * SPEED_OF_LIGHT
        result = np.where(
            (radius_array >= self.inner_radius_cm) & (radius_array <= self.outer_radius(time)),
            result,
            0.0,
        )
        return _return_like_input(result, radius)


@dataclass(frozen=True)
class OutflowHistory:
    """One angular bin of numerical outflow recorded at a fixed radius."""

    time_s: FloatArray
    velocity_c: FloatArray
    mass_loss_rate_g_s: FloatArray
    electron_fraction: FloatArray | None = None
    solid_angle_sr: float = 4.0 * np.pi

    def __post_init__(self) -> None:
        time = _array(self.time_s)
        velocity = _array(self.velocity_c)
        mass_rate = _array(self.mass_loss_rate_g_s)
        if time.ndim != 1 or len(time) < 2:
            raise ValueError("outflow arrays must be one-dimensional with at least two samples")
        if velocity.shape != time.shape or mass_rate.shape != time.shape:
            raise ValueError("time, velocity, and mass-loss-rate arrays must have equal shapes")
        if np.any(np.diff(time) <= 0):
            raise ValueError("time_s must be strictly increasing")
        if np.any((velocity <= 0) | (velocity >= 1)):
            raise ValueError("velocity_c values must lie strictly between 0 and 1")
        if np.any(mass_rate < 0):
            raise ValueError("mass_loss_rate_g_s cannot be negative")
        if not 0 < self.solid_angle_sr <= 4.0 * np.pi:
            raise ValueError("solid_angle_sr must lie in (0, 4*pi]")
        electron_fraction = self.electron_fraction
        if electron_fraction is not None:
            ye = _array(electron_fraction)
            if ye.shape != time.shape or np.any((ye < 0) | (ye > 1)):
                raise ValueError("electron_fraction must match time_s and lie in [0, 1]")
            object.__setattr__(self, "electron_fraction", ye)
        object.__setattr__(self, "time_s", time)
        object.__setattr__(self, "velocity_c", velocity)
        object.__setattr__(self, "mass_loss_rate_g_s", mass_rate)

    @classmethod
    def from_hdf5(
        cls,
        path: str | Path,
        *,
        bin_name: str,
        extraction_radius_cm: float,
        solid_angle_sr: float = 4.0 * np.pi,
    ) -> OutflowHistory:
        """Load one angular bin from a WhiskyTHC-style HDF5 file.

        The file stores local density and velocity rather than angular-bin mass
        flux. The latter is reconstructed as ``rho Gamma v r^2 solid_angle``
        at the supplied extraction radius. HDF5 is the standard persisted
        numerical-input format for this project.
        """
        import h5py

        with h5py.File(path, "r") as handle:
            group = handle[bin_name]
            time = np.asarray(group["time"], dtype=float)
            velocity = np.asarray(group["vel"], dtype=float) / SPEED_OF_LIGHT
            density = np.asarray(group["rho"], dtype=float)
            gamma = _array(lorentz_factor(velocity))
            mass_rate = (
                density
                * gamma
                * velocity
                * SPEED_OF_LIGHT
                * extraction_radius_cm**2
                * solid_angle_sr
            )
            ye = np.asarray(group["ye"], dtype=float) if "ye" in group else None
        return cls(time, velocity, mass_rate, ye, solid_angle_sr)


@dataclass(frozen=True)
class NumericalEjecta(Ejecta):
    """Ballistic reconstruction of a numerical outflow time series.

    Each recorded launch epoch contributes a generalized-Gaussian distribution
    in beta. ``cutoff_mode="sharp"`` truncates it at the fastest recorded
    shell; ``cutoff_mode="smooth"`` retains the exponential-like high-velocity
    tail. ``outer_radius`` remains the nominal fastest-shell radius used by the
    legacy jet-breakout calculation in both modes.
    The local density follows from mass conservation after free expansion. The
    history is isotropic-equivalent by default; a partial angular bin is scaled
    by ``4*pi/solid_angle_sr``.
    """

    history: OutflowHistory
    extraction_radius_cm: float = 4.42e7
    beta_width: float = 0.035
    kernel_shape: float = 2.0
    cutoff_mode: Literal["sharp", "smooth"] = "sharp"
    post_simulation_mass_index: float = 5.0 / 3.0
    post_simulation_velocity_index: float = 0.25
    integration_samples: int = 512

    def __post_init__(self) -> None:
        if self.extraction_radius_cm <= 0 or self.beta_width <= 0 or self.kernel_shape <= 0:
            raise ValueError("extraction_radius_cm, beta_width, and kernel_shape must be positive")
        if self.cutoff_mode not in ("sharp", "smooth"):
            raise ValueError("cutoff_mode must be 'sharp' or 'smooth'")
        if self.integration_samples < 16:
            raise ValueError("integration_samples must be at least 16")

    @classmethod
    def from_hdf5(
        cls,
        path: str | Path,
        *,
        bin_name: str = "itheta=00000",
        extraction_radius_cm: float = 4.42e7,
        solid_angle_sr: float = 4.0 * np.pi,
        **kwargs: float,
    ) -> NumericalEjecta:
        """Construct a numerical ejecta model directly from an HDF5 bin."""
        history = OutflowHistory.from_hdf5(
            path,
            bin_name=bin_name,
            extraction_radius_cm=extraction_radius_cm,
            solid_angle_sr=solid_angle_sr,
        )
        return cls(history, extraction_radius_cm=extraction_radius_cm, **kwargs)

    def inner_radius(self, time: float) -> float:
        if time <= self.history.time_s[0]:
            raise ValueError("time must be later than the first outflow sample")
        return self.extraction_radius_cm

    def outer_radius(self, time: float) -> float:
        """Return the nominal boundary set by the fastest recorded parcel."""
        if time <= self.history.time_s[0]:
            raise ValueError("time must be later than the first outflow sample")
        launched = self.history.time_s < time
        radii = self.extraction_radius_cm + self.history.velocity_c[launched] * SPEED_OF_LIGHT * (
            time - self.history.time_s[launched]
        )
        return float(np.max(radii))

    def _launch_grid(self, time: float) -> tuple[FloatArray, FloatArray, FloatArray]:
        upper = min(time * (1.0 - 1e-6), max(time, self.history.time_s[-1]))
        lower = self.history.time_s[0]
        if upper <= lower:
            raise ValueError("time must be later than the first outflow sample")
        launch_time = np.linspace(lower, upper, self.integration_samples)
        last_time = self.history.time_s[-1]
        mass_rate = np.interp(
            np.minimum(launch_time, last_time),
            self.history.time_s,
            self.history.mass_loss_rate_g_s,
        )
        launch_beta = np.interp(
            np.minimum(launch_time, last_time), self.history.time_s, self.history.velocity_c
        )
        after = launch_time > last_time
        mass_rate[after] *= (launch_time[after] / last_time) ** -self.post_simulation_mass_index
        launch_beta[after] *= (
            launch_time[after] / last_time
        ) ** -self.post_simulation_velocity_index
        angular_scale = 4.0 * np.pi / self.history.solid_angle_sr
        return launch_time, launch_beta, mass_rate * angular_scale

    def _moments(self, radius: float, time: float) -> tuple[float, float]:
        if radius < self.extraction_radius_cm:
            return 0.0, 0.0
        if self.cutoff_mode == "sharp" and radius > self.outer_radius(time):
            return 0.0, 0.0
        launch_time, launch_beta, mass_rate = self._launch_grid(time)
        flight_time = time - launch_time
        beta = (radius - self.extraction_radius_cm) / (SPEED_OF_LIGHT * flight_time)
        valid = (beta > 0) & (beta < 1)
        scaled_offset = np.abs((beta - launch_beta) / self.beta_width)
        normalization = self.kernel_shape / (2.0 * self.beta_width * gamma(1.0 / self.kernel_shape))
        kernel = normalization * np.exp(-(scaled_offset**self.kernel_shape))
        gamma_inverse = np.sqrt(np.maximum(1.0 - beta**2, 0.0))
        weights = np.where(valid, mass_rate * kernel * gamma_inverse / flight_time, 0.0)
        denominator = float(np.trapezoid(weights, launch_time))
        numerator = float(np.trapezoid(weights * beta * SPEED_OF_LIGHT, launch_time))
        return numerator, denominator

    def density(self, radius: ArrayLike, time: float) -> float | FloatArray:
        radius_array = _array(radius)
        flat = radius_array.reshape(-1)
        result = np.empty_like(flat)
        for index, current_radius in enumerate(flat):
            _, mass_per_time = self._moments(float(current_radius), time)
            result[index] = mass_per_time / (4.0 * np.pi * current_radius**2 * SPEED_OF_LIGHT)
        result = result.reshape(radius_array.shape)
        return _return_like_input(result, radius)

    def velocity(self, radius: ArrayLike, time: float) -> float | FloatArray:
        radius_array = _array(radius)
        flat = radius_array.reshape(-1)
        result = np.empty_like(flat)
        for index, current_radius in enumerate(flat):
            numerator, denominator = self._moments(float(current_radius), time)
            result[index] = numerator / denominator if denominator > 0 else 0.0
        result = result.reshape(radius_array.shape)
        return _return_like_input(result, radius)

    def electron_fraction(self, radius: ArrayLike, time: float) -> float | FloatArray:
        """Return a local mass-weighted electron fraction when present."""
        if self.history.electron_fraction is None:
            raise ValueError("the outflow history has no electron-fraction data")
        radius_array = _array(radius)
        flat = radius_array.reshape(-1)
        result = np.zeros_like(flat)
        for index, current_radius in enumerate(flat):
            outside = current_radius < self.extraction_radius_cm or (
                self.cutoff_mode == "sharp" and current_radius > self.outer_radius(time)
            )
            if outside:
                continue
            launch_time, launch_beta, mass_rate = self._launch_grid(time)
            flight_time = time - launch_time
            beta = (current_radius - self.extraction_radius_cm) / (SPEED_OF_LIGHT * flight_time)
            valid = (beta > 0) & (beta < 1)
            kernel = np.exp(-(np.abs((beta - launch_beta) / self.beta_width) ** self.kernel_shape))
            weights = np.where(valid, mass_rate * kernel / flight_time, 0.0)
            ye = np.interp(
                np.minimum(launch_time, self.history.time_s[-1]),
                self.history.time_s,
                self.history.electron_fraction,
            )
            denominator = np.trapezoid(weights, launch_time)
            if denominator > 0:
                result[index] = np.trapezoid(weights * ye, launch_time) / denominator
        result = result.reshape(radius_array.shape)
        return _return_like_input(result, radius)
