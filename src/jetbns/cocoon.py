"""Coupled jet/cocoon evolution through head breakout.

Port of jetBNS3/modules/propagation.py, Jet and Cocoon (GPL-3.0-only),
following Gutiérrez et al. (2025), arXiv:2408.15973, equations 10--19.
The conical JetHead is retained as a comparison model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .constants import SPEED_OF_LIGHT as C
from .ejecta import Ejecta, lorentz_factor
from .engines import Engine
from .propagation import JetHead, PropagationResult


@dataclass(frozen=True)
class Cocoon:
    """Radiation dominated ellipsoid between the injection plane and head.

    Pressure uses the published volume 2*pi*(z_h-z_0)*r_c^2/3. Density is
    averaged with ellipsoidal axial area weights, using the axial ejecta
    profile as in the legacy closure (not a two-dimensional volume integral).
    Lateral pressure and ambient speeds are added relativistically as in
    jetBNS3, rather than using the low-speed sum printed in equation 14.
    """

    density_samples: int = 48
    pressure_delay: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.density_samples, int) or self.density_samples < 16:
            raise ValueError("density_samples must be an integer of at least 16")

    @staticmethod
    def volume(head_radius_cm: float, lateral_radius_cm: float, base_radius_cm: float) -> float:
        """Ellipsoid volume in cm^3; all three arguments are laboratory lengths."""
        if not head_radius_cm > base_radius_cm > 0 or lateral_radius_cm <= 0:
            raise ValueError("require head > base > 0 and positive lateral radius")
        return 2*np.pi/3 * (head_radius_cm-base_radius_cm)*lateral_radius_cm**2

    @staticmethod
    def causal_fraction(head_beta: float, opening_angle_rad: float) -> float:
        """Fraction eta of the head feeding the cocoon, equation 12."""
        if not 0 <= head_beta < 1 or not 0 <= opening_angle_rad < np.pi/2:
            raise ValueError("invalid speed or opening angle")
        mu = np.sqrt(3)*head_beta*lorentz_factor(head_beta)*opening_angle_rad
        return float(1.0 if mu <= 1 else 2/mu-1/mu**2)

    def mean_density(self, ejecta: Ejecta, head: float, base: float, time: float) -> float:
        """Ellipsoidal axial average of proper density, in g cm^-3."""
        z = np.geomspace(base, head, self.density_samples)
        x = (z-base)/(head-base)
        weights = 4*x*(1-x)
        return float(np.trapezoid(ejecta.density(z, time)*weights, z)
                     / np.trapezoid(weights, z))


@dataclass(frozen=True)
class CocoonState:
    """Instantaneous coupled state; pressure is in erg cm^-3, area in cm^2."""

    head_beta: float
    ambient_beta: float
    dimensionless_luminosity: float
    lateral_beta: float
    pressure_erg_cm3: float
    jet_cross_section_cm2: float
    jet_opening_angle_rad: float
    causal_fraction: float
    deposition_rate_erg_s: float


@dataclass(frozen=True)
class CocoonPropagationResult(PropagationResult):
    """Head trajectory with the cocoon variables that determined its motion."""

    cocoon_radius_cm: NDArray[np.float64]
    cocoon_energy_erg: NDArray[np.float64]
    pressure_energy_erg: NDArray[np.float64]
    cocoon_pressure_erg_cm3: NDArray[np.float64]
    cocoon_lateral_beta: NDArray[np.float64]
    jet_cross_section_cm2: NDArray[np.float64]
    jet_opening_angle_rad: NDArray[np.float64]
    cocoon_deposition_rate_erg_s: NDArray[np.float64]
    base_radius_cm: float


@dataclass(frozen=True)
class JetCocoon:
    """Evolve head position, cocoon width and energy with pressure collimation.

    dE_c/dt = eta L_j(t_eng) (beta_j-beta_h), with no additional PdV sink:
    this is the published pre-breakout energy closure. Pressure uses
    E_c(t-(z_h-z_0)/(2*c/sqrt(3))) by default, interpolated from accepted
    history. Energy before injection is zero. The optional instantaneous
    closure is exposed through Cocoon(pressure_delay=False).

    Reconfinement solves z_hat=z_0+(beta_j-beta_a)*sqrt(L_j/(pi beta_j c P_c)).
    The ambient speed is evaluated at z_hat/cos(theta_0); luminosity is the
    power reaching the head. This local luminosity closure avoids evaluating
    a time-dependent engine outside the head's causal domain. For steady
    powered jets it agrees with the legacy reconfinement equation.
    """

    engine: Engine
    ejecta: Ejecta
    cocoon: Cocoon = Cocoon()
    calibration: float = .65
    breakout_opacity_cm2_g: float = .16
    breakout_optical_depth_samples: int = 256
    initial_height_fraction: float = 1e-4

    def __post_init__(self) -> None:
        if not np.isfinite(self.calibration) or self.calibration <= 0:
            raise ValueError("calibration must be finite and positive")
        if not np.isfinite(self.breakout_opacity_cm2_g) or self.breakout_opacity_cm2_g <= 0:
            raise ValueError("opacity must be finite and positive")
        if self.breakout_optical_depth_samples < 16:
            raise ValueError("breakout_optical_depth_samples must be at least 16")
        if not 0 < self.initial_height_fraction < .1:
            raise ValueError("initial_height_fraction must lie between 0 and 0.1")

    shock_beta_in_ambient_frame = staticmethod(JetHead.shock_beta_in_ambient_frame)

    def cross_section(self, radius: float, time: float, pressure: float, base: float) -> float:
        """Pressure-confined jet area, or conical area when reconfinement is above the head."""
        angle = self.engine.opening_angle_rad
        luminosity = float(self.engine.luminosity(radius, time))
        if pressure <= 0 or luminosity <= 0:
            return float(np.pi*(radius*np.tan(angle))**2)
        scale = np.sqrt(luminosity/(np.pi*self.engine.beta*C*pressure))

        def residual(height: float) -> float:
            beta = float(self.ejecta.velocity(height/np.cos(angle), time))/C
            return height-base-max(self.engine.beta-beta, 0)*scale

        # Only roots with (z_hat+z_0)/2 < z_h can collimate this head.
        lower, upper = base, 2*radius-base
        if residual(upper) <= 0:
            collimation_height = radius
        else:
            for _ in range(28):
                middle = (lower+upper)/2
                if residual(middle) > 0:
                    upper = middle
                else:
                    lower = middle
            collimation_height = .5*(base+.5*(lower+upper))
        return float(np.pi*(collimation_height*np.tan(angle))**2)

    def state(self, radius: float, time: float, *, cocoon_radius_cm: float,
              cocoon_energy_erg: float, base_radius_cm: float,
              pressure_energy_erg: float | None = None) -> CocoonState:
        """Evaluate coupling for a supplied cocoon state (all quantities CGS).

        A delayed pressure energy must be supplied when pressure_delay=True;
        a local state cannot reconstruct the earlier history by itself.
        """
        if pressure_energy_erg is None:
            if self.cocoon.pressure_delay:
                raise ValueError("pressure_energy_erg is required for delayed pressure")
            pressure_energy_erg = cocoon_energy_erg
        if min(cocoon_energy_erg, pressure_energy_erg) < 0:
            raise ValueError("cocoon energies must be nonnegative")
        pressure = pressure_energy_erg/(3*self.cocoon.volume(
            radius, cocoon_radius_cm, base_radius_cm))
        area = self.cross_section(radius, time, pressure, base_radius_cm)
        if hasattr(self.ejecta, "density_and_velocity"):
            density, velocity = self.ejecta.density_and_velocity(radius, time)
        else:
            density = float(self.ejecta.density(radius, time))
            velocity = float(self.ejecta.velocity(radius, time))
        ambient = velocity/C
        luminosity = float(self.engine.luminosity(radius, time))
        if luminosity <= 0:
            head, effective = ambient, 0.
        elif density <= 0:
            head, effective = self.engine.beta, 0.
        else:
            effective = self.calibration**2*luminosity*(1-ambient**2)/(
                area*self.engine.beta*density*C**3)
            head = ambient+(self.engine.beta-ambient)/(1+effective**-.5)
        angle = float(np.arctan(np.sqrt(area/np.pi)/radius))
        eta = self.cocoon.causal_fraction(head, angle)
        deposition = eta*luminosity*max(self.engine.beta-head, 0)
        mean_density = self.cocoon.mean_density(self.ejecta, radius, base_radius_cm, time)
        distance = np.hypot(cocoon_radius_cm, .5*(radius+base_radius_cm))
        ambient_lateral = float(self.ejecta.velocity(distance, time))/C*cocoon_radius_cm/distance
        lateral_shock = np.sqrt(pressure/(pressure+mean_density*C**2)) if pressure > 0 else 0.
        lateral = (lateral_shock+ambient_lateral)/(1+lateral_shock*ambient_lateral)
        return CocoonState(head, ambient, effective, lateral, pressure, area, angle,
                           eta, deposition)

    def propagate(self, *, start_time_s: float | None = None, max_time_s: float = 10.,
                  time_step_s: float = 1e-3) -> CocoonPropagationResult:
        """Integrate the coupled system with RK4 and accepted-history pressure delay.

        time_step_s is a maximum step; early steps also resolve the light
        crossing time of the head and lateral radius. Stage values interpolate
        the most recent history interval when a delay falls within that step.
        Refining the maximum step tests that interpolation as well as the ODE.
        The endpoint satisfies the optical-depth shock breakout criterion.
        """
        launch = self.engine.launch_time_s if start_time_s is None else start_time_s
        if not np.isfinite([launch, max_time_s, time_step_s]).all():
            raise ValueError("times must be finite")
        if launch < self.engine.launch_time_s or time_step_s <= 0:
            raise ValueError("invalid start time or time step")
        base = max(self.engine.launch_radius_cm, self.ejecta.inner_radius(launch)*1.001)
        radius = base*(1+self.initial_height_fraction)
        time = launch+(radius-self.engine.launch_radius_cm)/C
        if max_time_s <= time or radius >= self.ejecta.outer_radius(time):
            raise ValueError("require max time after injection and launch within ejecta")
        y = np.array([radius, base*np.tan(self.engine.opening_angle_rad), 0.])
        times, values = [time], [y.copy()]
        energies = [0.]

        def evaluate(t: float, value: NDArray[np.float64]) -> tuple[CocoonState, float]:
            if self.cocoon.pressure_delay:
                delayed = t-(value[0]-base)*np.sqrt(3)/(2*C)
                if delayed <= times[-1]:
                    energy = float(np.interp(delayed, times, energies, left=0.))
                else:
                    energy = energies[-1]+(value[2]-energies[-1])*(delayed-times[-1])/(t-times[-1])
            else:
                energy = value[2]
            return self.state(value[0], t, cocoon_radius_cm=value[1], cocoon_energy_erg=value[2],
                              base_radius_cm=base, pressure_energy_erg=energy), energy

        def rhs(t: float, value: NDArray[np.float64]) -> NDArray[np.float64]:
            state, _ = evaluate(t, value)
            return np.array([state.head_beta*C, state.lateral_beta*C, state.deposition_rate_erg_s])

        def advance(t: float, value: NDArray[np.float64], step: float) -> NDArray[np.float64]:
            k1 = rhs(t, value)
            k2 = rhs(t+step/2, value+step*k1/2)
            k3 = rhs(t+step/2, value+step*k2/2)
            k4 = rhs(t+step, value+step*k3)
            return value+step*(k1+2*k2+2*k3+k4)/6

        def residual(t: float, value: NDArray[np.float64], state: CocoonState) -> float:
            beta = self.shock_beta_in_ambient_frame(state.head_beta, state.ambient_beta)
            if beta <= 0:
                return np.inf
            return self.ejecta.optical_depth(value[0], t, opacity=self.breakout_opacity_cm2_g,
                       samples=self.breakout_optical_depth_samples)*beta-1

        state, energy = evaluate(time, y)
        states, pressure_energies = [state], [energy]
        broke_out = residual(time, y, state) <= 0
        while time < max_time_s and not broke_out:
            step = min(time_step_s, .1*min(y[:2])/C, max_time_s-time)
            next_y = advance(time, y, step)
            next_state, energy = evaluate(time+step, next_y)
            if residual(time+step, next_y, next_state) <= 0:
                lower, upper = 0., 1.
                for _ in range(28):
                    fraction = .5*(lower+upper)
                    trial = y+fraction*(next_y-y)
                    trial_state, _ = evaluate(time+fraction*step, trial)
                    if residual(time+fraction*step, trial, trial_state) > 0:
                        lower = fraction
                    else:
                        upper = fraction
                fraction = .5*(lower+upper)
                next_y = y+fraction*(next_y-y)
                step *= fraction
                next_state, energy = evaluate(time+step, next_y)
                broke_out = True
            time += step
            y = next_y
            if not np.isfinite(y).all() or np.any(y < 0):
                raise RuntimeError("nonfinite or negative jet-cocoon solution")
            times.append(time)
            values.append(y.copy())
            energies.append(y[2])
            states.append(next_state)
            pressure_energies.append(energy)

        data = np.asarray(values)
        def array(name: str) -> NDArray[np.float64]:
            return np.asarray([getattr(s, name) for s in states])
        return CocoonPropagationResult(
            np.asarray(times), data[:, 0], array("head_beta"), array("ambient_beta"),
            array("dimensionless_luminosity"), broke_out, data[:, 1], data[:, 2],
            np.asarray(pressure_energies), array("pressure_erg_cm3"), array("lateral_beta"),
            array("jet_cross_section_cm2"), array("jet_opening_angle_rad"),
            array("deposition_rate_erg_s"), base)
