"""Independent limiting equations, legacy closure, causality and convergence."""
from dataclasses import dataclass

import numpy as np
import pytest

from jetbns import Cocoon, ConstantEngine, Ejecta, HomologousTail, JetCocoon, JetHead
from jetbns.constants import SPEED_OF_LIGHT as C


@dataclass
class UniformEjecta(Ejecta):
    rho: float = 1e-7
    beta: float = 0.

    def inner_radius(self, time):
        return 1e7

    def outer_radius(self, time):
        return 1e12

    def density(self, radius, time):
        return np.full_like(np.asarray(radius, dtype=float), self.rho)

    def velocity(self, radius, time):
        return np.full_like(np.asarray(radius, dtype=float), self.beta*C)


def test_uniform_medium_reconfinement_and_legacy_pressure_balance():
    engine = ConstantEngine(luminosity_erg_s=1e46, launch_time_s=0.,
                            launch_radius_cm=1e8, lorentz_factor=10.)
    ejecta = UniformEjecta(beta=.2)
    cocoon = Cocoon(pressure_delay=False)
    jet = JetCocoon(engine, ejecta, cocoon)
    z, radius, base, energy = 1e10, 1e9, 1e8, 1e46
    # Published geometry; the legacy implementation instead used z, not z-z0.
    volume = 2*np.pi/3*(z-base)*radius**2
    pressure = energy/(3*volume)
    zhat = base+(engine.beta-ejecta.beta)*np.sqrt(
        engine.luminosity_erg_s/(np.pi*engine.beta*C*pressure))
    expected_area = np.pi*(min(z, (zhat+base)/2)*np.tan(engine.opening_angle_rad))**2
    state = jet.state(z, 1., cocoon_radius_cm=radius, cocoon_energy_erg=energy,
                      base_radius_cm=base)
    assert state.jet_cross_section_cm2 == pytest.approx(expected_area, rel=1e-6)
    assert state.pressure_erg_cm3 == pytest.approx(pressure)
    assert cocoon.mean_density(ejecta, z, base, 1.) == pytest.approx(ejecta.rho)
    d = np.hypot(radius, .5*(z+base))
    ambient = ejecta.beta*radius/d
    shock = np.sqrt(pressure/(pressure+ejecta.rho*C**2))
    assert state.lateral_beta == pytest.approx((shock+ambient)/(1+shock*ambient))
    # No pressure recovers exactly the separately implemented conical solution.
    cold = jet.state(z, 1., cocoon_radius_cm=radius, cocoon_energy_erg=0,
                     base_radius_cm=base)
    assert cold.head_beta == pytest.approx(JetHead(engine, ejecta).state(z, 1.)[0])
    assert state.head_beta > cold.head_beta


def test_causal_fraction_limits_and_delay_requires_history():
    assert Cocoon.causal_fraction(.1, .1) == 1.
    assert 0 < Cocoon.causal_fraction(.9999, .2) < 1
    jet = JetCocoon(ConstantEngine(), UniformEjecta())
    with pytest.raises(ValueError, match="pressure_energy"):
        jet.state(1e10, 1., cocoon_radius_cm=1e9, cocoon_energy_erg=1e40, base_radius_cm=1e8)


def reference(step, **kwargs):
    ejecta = HomologousTail()
    engine = ConstantEngine.from_isotropic_equivalent(
        1e51, launch_time_s=1., launch_radius_cm=ejecta.inner_radius(1)*1.01,
        opening_angle_rad=np.deg2rad(10), lorentz_factor=100.)
    return JetCocoon(engine, ejecta, **kwargs).propagate(max_time_s=6., time_step_s=step)


def test_coupled_energy_budget_pressure_delay_and_step_convergence():
    coarse, fine = reference(.02), reference(.01)
    assert coarse.broke_out and fine.broke_out
    assert fine.breakout_time_s == pytest.approx(coarse.breakout_time_s, rel=.015)
    assert fine.cocoon_energy_erg[-1] == pytest.approx(coarse.cocoon_energy_erg[-1], rel=.015)
    assert np.all(np.diff(fine.cocoon_energy_erg) >= 0)
    assert np.all((fine.cocoon_lateral_beta >= 0) & (fine.cocoon_lateral_beta < 1))
    assert np.all(fine.pressure_energy_erg <= fine.cocoon_energy_erg*(1+1e-12))
    deposited = np.trapezoid(fine.cocoon_deposition_rate_erg_s, fine.time_s)
    assert fine.cocoon_energy_erg[-1] == pytest.approx(deposited, rel=.005)
    delayed = fine.time_s-(fine.radius_cm-fine.base_radius_cm)*np.sqrt(3)/(2*C)
    assert np.allclose(fine.pressure_energy_erg,
                       np.interp(delayed, fine.time_s, fine.cocoon_energy_erg, left=0.),
                       rtol=.01, atol=1e30)
    angle = np.deg2rad(10)
    conical = np.pi*(fine.radius_cm*np.tan(angle))**2
    assert np.all(fine.jet_cross_section_cm2 <= conical*(1+1e-12))
    assert np.min(fine.jet_cross_section_cm2/conical) < .5
    # Both the sound-crossing delay and collimation must affect the trajectory.
    instantaneous = reference(.01, cocoon=Cocoon(pressure_delay=False))
    assert instantaneous.breakout_time_s != pytest.approx(fine.breakout_time_s, rel=.001)


def test_initial_height_regularization_converges():
    a = reference(.02)
    b = reference(.02, initial_height_fraction=5e-5)
    assert a.breakout_time_s == pytest.approx(b.breakout_time_s, rel=.002)
    assert a.cocoon_energy_erg[-1] == pytest.approx(b.cocoon_energy_erg[-1], rel=.005)


def test_ellipsoidal_density_quadrature_converges_independently():
    ejecta = HomologousTail()
    base = ejecta.inner_radius(1.)*1.01
    head = .35*C
    coarse = Cocoon(density_samples=48).mean_density(ejecta, head, base, 1.)
    fine = Cocoon(density_samples=192).mean_density(ejecta, head, base, 1.)
    assert coarse == pytest.approx(fine, rel=.005)


def test_no_injection_after_engine_switches_off():
    engine = ConstantEngine(launch_time_s=0., duration_s=.01)
    jet = JetCocoon(engine, UniformEjecta(), Cocoon(pressure_delay=False))
    state = jet.state(1e9, 1., cocoon_radius_cm=1e8, cocoon_energy_erg=1e40,
                      base_radius_cm=engine.launch_radius_cm)
    assert state.deposition_rate_erg_s == 0.
    assert state.lateral_beta > 0.
