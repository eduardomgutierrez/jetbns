"""One-dimensional relativistic jet-head propagation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .constants import SPEED_OF_LIGHT
from .ejecta import Ejecta, lorentz_factor
from .engines import Engine


@dataclass(frozen=True)
class PropagationResult:
    """Time history produced by :meth:`JetHead.propagate`."""

    time_s: NDArray[np.float64]
    radius_cm: NDArray[np.float64]
    head_beta: NDArray[np.float64]
    ambient_beta: NDArray[np.float64]
    dimensionless_luminosity: NDArray[np.float64]
    broke_out: bool

    @property
    def breakout_time_s(self) -> float | None:
        return float(self.time_s[-1]) if self.broke_out else None

    @property
    def breakout_radius_cm(self) -> float | None:
        return float(self.radius_cm[-1]) if self.broke_out else None


@dataclass(frozen=True)
class JetHead:
    r"""Propagate an uncollimated conical jet through moving ejecta.

    Momentum-flux balance follows the cleaned core of ``jetBNS3``:

    .. math::
       \tilde L = \frac{L_j}{\Sigma_j\,\beta_j\,\rho_a c^3\Gamma_a^2},
       \quad
       \beta_h = \beta_a + \frac{\beta_j-\beta_a}
       {1 + (N_s^2\tilde L)^{-1/2}}.

    The cross-section is ``pi * (r tan(theta))^2``. Cocoon collimation is not
    included in this first validated propagator and is recorded as future work.
    """

    engine: Engine
    ejecta: Ejecta
    calibration: float = 0.65
    breakout_opacity_cm2_g: float = 0.16
    breakout_optical_depth_samples: int = 256

    def __post_init__(self) -> None:
        if self.calibration <= 0:
            raise ValueError("calibration must be positive")
        if self.breakout_opacity_cm2_g <= 0:
            raise ValueError("breakout opacity must be positive")
        if self.breakout_optical_depth_samples < 16:
            raise ValueError("breakout_optical_depth_samples must be at least 16")

    def cross_section(self, radius: float) -> float:
        """Return conical jet cross-section in cm^2."""
        if radius <= 0:
            raise ValueError("radius must be positive")
        return float(np.pi * (radius * np.tan(self.engine.opening_angle_rad)) ** 2)

    def state(self, radius: float, time: float) -> tuple[float, float, float]:
        """Return ``(head_beta, ambient_beta, effective_tilde_l)``."""
        density = float(self.ejecta.density(radius, time))
        ambient_beta = float(self.ejecta.velocity(radius, time)) / SPEED_OF_LIGHT
        luminosity = float(self.engine.luminosity(radius, time))
        if density <= 0 or luminosity <= 0:
            return ambient_beta, ambient_beta, 0.0
        ambient_gamma = float(lorentz_factor(ambient_beta))
        tilde_l = luminosity / (
            self.cross_section(radius)
            * self.engine.beta
            * density
            * SPEED_OF_LIGHT**3
            * ambient_gamma**2
        )
        effective = self.calibration**2 * tilde_l
        head_beta = ambient_beta + (self.engine.beta - ambient_beta) / (1.0 + effective**-0.5)
        return float(head_beta), ambient_beta, float(effective)

    @staticmethod
    def shock_beta_in_ambient_frame(head_beta: float, ambient_beta: float) -> float:
        r"""Return legacy shock speed :math:`\beta'_s` in the ambient frame."""
        relative_beta = (head_beta - ambient_beta) / (1.0 - head_beta * ambient_beta)
        if relative_beta <= 0:
            return 0.0
        relative_gamma = float(lorentz_factor(relative_beta))
        adiabatic_index = 4.0 / 3.0
        shock_four_velocity_squared = (
            (relative_gamma - 1.0)
            * (adiabatic_index * relative_gamma + 1.0) ** 2
            / (2.0 + adiabatic_index * (2.0 - adiabatic_index) * (relative_gamma - 1.0))
        )
        return float(np.sqrt(shock_four_velocity_squared / (1.0 + shock_four_velocity_squared)))

    def breakout_residual(self, radius: float, time: float) -> float:
        r"""Return positive before breakout and zero at the breakout surface."""
        if radius >= self.ejecta.optical_depth_outer_radius(time):
            return -1.0
        head_beta, ambient_beta, _ = self.state(radius, time)
        shock_beta = self.shock_beta_in_ambient_frame(head_beta, ambient_beta)
        if shock_beta <= 0:
            return np.inf
        optical_depth = self.ejecta.optical_depth(
            radius,
            time,
            opacity=self.breakout_opacity_cm2_g,
            samples=self.breakout_optical_depth_samples,
        )
        return optical_depth - 1.0 / shock_beta

    def propagate(
        self,
        *,
        start_time_s: float | None = None,
        max_time_s: float = 10.0,
        time_step_s: float = 1.0e-4,
    ) -> PropagationResult:
        """Integrate the jet-head trajectory with fourth-order Runge--Kutta.

        Integration ends when the upstream optical depth falls below
        ``1 / beta_s'``. Smooth numerical profiles include their high-velocity
        tail in that optical-depth integral. The fixed step is explicit so
        convergence can be checked by rerunning with a smaller value.
        """
        start = self.engine.launch_time_s if start_time_s is None else start_time_s
        if start < self.engine.launch_time_s:
            raise ValueError("start_time_s cannot precede engine launch")
        if max_time_s <= start or time_step_s <= 0:
            raise ValueError("max_time_s must exceed start and time_step_s must be positive")
        radius = max(self.engine.launch_radius_cm, self.ejecta.inner_radius(start) * 1.001)
        if radius >= self.ejecta.outer_radius(start):
            raise ValueError("jet launch radius must lie inside the ejecta")

        times = [float(start)]
        radii = [float(radius)]
        states = [self.state(radius, start)]
        broke_out = False

        def derivative(current_radius: float, current_time: float) -> float:
            return self.state(current_radius, current_time)[0] * SPEED_OF_LIGHT

        time = float(start)
        while time < max_time_s:
            step = min(time_step_s, max_time_s - time)
            k1 = derivative(radius, time)
            k2 = derivative(radius + 0.5 * step * k1, time + 0.5 * step)
            k3 = derivative(radius + 0.5 * step * k2, time + 0.5 * step)
            k4 = derivative(radius + step * k3, time + step)
            next_radius = radius + step * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
            next_time = time + step
            current_breakout_residual = self.breakout_residual(next_radius, next_time)
            if current_breakout_residual <= 0:
                lower_fraction = 0.0
                upper_fraction = 1.0
                radius_increment = next_radius - radius
                for _ in range(32):
                    fraction = 0.5 * (lower_fraction + upper_fraction)
                    trial_time = time + fraction * step
                    trial_radius = radius + fraction * radius_increment
                    if self.breakout_residual(trial_radius, trial_time) > 0:
                        lower_fraction = fraction
                    else:
                        upper_fraction = fraction
                fraction = 0.5 * (lower_fraction + upper_fraction)
                next_time = time + fraction * step
                next_radius = radius + fraction * radius_increment
                broke_out = True
            time, radius = next_time, next_radius
            times.append(time)
            radii.append(radius)
            states.append(self.state(radius * (1.0 - 1e-12), time))
            if broke_out:
                break

        state_array = np.asarray(states)
        return PropagationResult(
            time_s=np.asarray(times),
            radius_cm=np.asarray(radii),
            head_beta=state_array[:, 0],
            ambient_beta=state_array[:, 1],
            dimensionless_luminosity=state_array[:, 2],
            broke_out=broke_out,
        )
