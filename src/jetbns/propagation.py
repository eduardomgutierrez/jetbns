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

    def __post_init__(self) -> None:
        if self.calibration <= 0:
            raise ValueError("calibration must be positive")

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

    def propagate(
        self,
        *,
        start_time_s: float | None = None,
        max_time_s: float = 10.0,
        time_step_s: float = 1.0e-4,
    ) -> PropagationResult:
        """Integrate the jet-head trajectory with fourth-order Runge--Kutta.

        Integration ends at the ejecta outer boundary or ``max_time_s``. The
        fixed step is explicit so convergence can be checked by rerunning with a
        smaller value.
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
            outer = self.ejecta.outer_radius(next_time)
            if next_radius >= outer:
                previous_gap = self.ejecta.outer_radius(time) - radius
                current_gap = outer - next_radius
                fraction = previous_gap / (previous_gap - current_gap)
                next_time = time + fraction * step
                next_radius = self.ejecta.outer_radius(next_time)
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
