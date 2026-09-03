"""Central-engine luminosity models for relativistic jets."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from .constants import SPEED_OF_LIGHT
from .ejecta import FloatArray, _array, _return_like_input


@dataclass(frozen=True)
class Engine(ABC):
    """Base class for a one-sided jet engine.

    ``luminosity`` is the true luminosity in one jet, not an isotropic-
    equivalent luminosity. Radiation reaches radius ``r`` at the retarded time
    ``t - (r-r_launch)/c``.
    """

    launch_time_s: float = 0.05
    launch_radius_cm: float = 8.45e7
    opening_angle_rad: float = np.deg2rad(6.8)
    lorentz_factor: float = 10.0

    def __post_init__(self) -> None:
        if self.launch_time_s < 0 or self.launch_radius_cm <= 0:
            raise ValueError("launch time cannot be negative and radius must be positive")
        if not 0 < self.opening_angle_rad < np.pi / 2:
            raise ValueError("opening_angle_rad must lie between 0 and pi/2")
        if self.lorentz_factor <= 1:
            raise ValueError("lorentz_factor must exceed 1")

    @property
    def beta(self) -> float:
        """Dimensionless bulk speed of the unshocked jet."""
        return float(np.sqrt(1.0 - self.lorentz_factor**-2))

    def retarded_time(self, radius: ArrayLike, time: float) -> float | FloatArray:
        """Return source time corresponding to radiation at ``(radius, time)``."""
        radius_array = _array(radius)
        result = time - (radius_array - self.launch_radius_cm) / SPEED_OF_LIGHT
        return _return_like_input(result, radius)

    @abstractmethod
    def source_luminosity(self, time: ArrayLike) -> float | FloatArray:
        """Return one-sided luminosity at the engine in erg s^-1."""

    def luminosity(self, radius: ArrayLike, time: float) -> float | FloatArray:
        """Return retarded one-sided luminosity at radius in erg s^-1."""
        source_time = self.retarded_time(radius, time)
        return self.source_luminosity(source_time)


@dataclass(frozen=True)
class ConstantEngine(Engine):
    """Engine with constant luminosity over an optional finite duration."""

    luminosity_erg_s: float = 1.0e49
    duration_s: float | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.luminosity_erg_s <= 0:
            raise ValueError("luminosity_erg_s must be positive")
        if self.duration_s is not None and self.duration_s <= 0:
            raise ValueError("duration_s must be positive when supplied")

    @classmethod
    def from_isotropic_equivalent(
        cls, luminosity_iso_erg_s: float, **kwargs: float
    ) -> ConstantEngine:
        """Construct from isotropic-equivalent luminosity for one top-hat jet."""
        angle = float(kwargs.get("opening_angle_rad", cls.opening_angle_rad))
        one_sided = luminosity_iso_erg_s * (1.0 - np.cos(angle)) / 2.0
        return cls(luminosity_erg_s=one_sided, **kwargs)

    def source_luminosity(self, time: ArrayLike) -> float | FloatArray:
        time_array = _array(time)
        active = time_array >= self.launch_time_s
        if self.duration_s is not None:
            active &= time_array <= self.launch_time_s + self.duration_s
        result = np.where(active, self.luminosity_erg_s, 0.0)
        return _return_like_input(result, time)


@dataclass(frozen=True)
class PowerLawEngine(Engine):
    """Plateau luminosity followed by a power-law decline.

    This captures the stable-then-declining form used by the simplified
    black-hole engine in ``jetBNS3`` without importing its unvalidated baryon-
    loading and magnetic-acceleration branches.
    """

    luminosity_erg_s: float = 1.0e49
    plateau_duration_s: float = 0.1
    decay_index: float = 2.0
    duration_s: float | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.luminosity_erg_s <= 0 or self.plateau_duration_s <= 0:
            raise ValueError("luminosity and plateau duration must be positive")
        if self.decay_index < 0:
            raise ValueError("decay_index cannot be negative")
        if self.duration_s is not None and self.duration_s <= 0:
            raise ValueError("duration_s must be positive when supplied")

    def source_luminosity(self, time: ArrayLike) -> float | FloatArray:
        time_array = _array(time)
        age = time_array - self.launch_time_s
        scale = np.maximum(age / self.plateau_duration_s, 1.0)
        result = self.luminosity_erg_s * scale**-self.decay_index
        active = age >= 0
        if self.duration_s is not None:
            active &= age <= self.duration_s
        result = np.where(active, result, 0.0)
        return _return_like_input(result, time)
