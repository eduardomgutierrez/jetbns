# Scientific and numerical audit

This audit supersedes the pre-audit numerical export and its model rankings.
The package is a deterministic reference setup for transport calculations.
Numerical convergence is tested separately from the physical assumptions.

## Confirmed problems and changes

1. **Recorded outflow was underresolved.** A uniform grid of 128 epochs from
   the first measurement to a query at about 12 s retained almost none of a
   roughly 0.1 s simulation. For the old DD2 equal-mass endpoint, proper density
   changed from `2.84e-9` to about `1.31e-4 g cm^-3` when all measured epochs
   were retained. Local Ye changed from 0.537 to about 0.463 at that *same*
   spacetime point. This invalidates the old trajectory and its zero-neutron
   interpretation; it does not prescribe the composition at the new breakout.
   Native epochs are now preserved, with geometric sampling of extrapolated
   flight times and optional refinement between measured epochs. A synthetic
   short-pulse regression is checked against independent dense quadrature.

2. **The Ye weighting differed from the density weighting.** The old Ye moment
   omitted the parcel `1/Gamma` factor present in proper density. Both now use
   the same proper-density weights. Missing composition can use an explicit
   fallback; an invalid provided composition now raises rather than silently
   replacing the whole trajectory with Ye=0.1.

3. **Exterior mass used the wrong volume density.** Proper density was
   integrated over lab volume without Gamma. The neutron-skin mass coordinate
   now integrates lab-frame baryon density, summing the parcel lab moments for
   numerical ejecta. This remains an isotropic-equivalent angular convention.

4. **The old analytical tail was kinematically inconsistent.** Its density
   surfaces could expand faster than light while its material velocity was
   capped at the nominal speed. `HomologousTail` supplies a separate physical
   reference: all surfaces follow r=beta*c*t, beta<1, density scales as t^-3,
   and the requested baryon mass includes the exponential tail. The older
   classes remain available for legacy comparisons; they are not used in the
   audited transfer suite. Their `mass()` is the legacy proper-density volume
   integral, which should not be mistaken for a relativistic conserved mass.

5. **The previous export did not establish convergence.** Tests of shape and
   the same equations implemented in the code did not catch the launch-grid
   defect. New regressions check late-time recovery of a short ejection pulse,
   mass conservation, shock-frame transformations, and invalid inputs. Each
   exported model now has refined time/radial/history/mass quadratures and an
   independent high-resolution breakout residual; 5% is the maximum permitted
   endpoint discrepancy, not a claim of 5% physical accuracy. SFHo required
   additional refinement beyond the first comparison.

6. **The old transfer lacked enough context.** The new file records complete
   engine/ejecta/composition settings, profile hashes and measured durations,
   frames, rates, effective lengths, cold hydrodynamic shock speeds, and
   convergence errors. It includes a standalone validation/reading script.
   The snapshot is fixed at 99% of propagation duration rather than an
   arbitrary last ODE step. Failed builds preserve the previous archive.

7. **Published dependency bounds were too loose.** `numpy.trapezoid` requires
   NumPy 2.0; the declared minimum was 1.24. The minimum is corrected.

## Physics interpretation

The conical momentum-balance propagator implements the appropriate cold-jet
limit of the propagation paper, but not its cocoon collimation and radiation
model. Numerical inputs here use recorded `vel`, whereas the paper uses the
Bernoulli-derived asymptotic velocity. Reproduction of the paper therefore
requires a separate comparison with validated input metadata. The new
quadrature corrects the implementation without claiming that late-time
power-law extrapolation is measured data.
[Gutiérrez et al.](https://arxiv.org/abs/2408.15973)

The neutron-skin expression `(2/pi)*atan(M_n/M_above)` is confirmed in Eq. (7)
of the arXiv PDF. The publisher HTML rendering reverses the ratio, so the PDF
and its shallow/deep limits are the appropriate reference. This remains a
phenomenological abundance. It is applied to locally averaged Ye; mixing and
capture history are not resolved. The legacy `t_ee ~ 1e-8 M^(-5/4)` comparison
is an electron/positron capture estimate from the hot NS collision discussion,
not a beta-decay time or a direct neutron-capture freeze-out prescription.
Legacy docstrings claiming otherwise should not be ported.
[Metzger et al.](https://arxiv.org/pdf/1409.0544)

`n_p=Ye*rho/mp` includes proton content bound in nuclei. It is exported as a
free-proton *proxy*, with a configurable fraction of that content assumed free.
`n_n` is the separate free-neutron estimate. The radial free-proton fraction
and a nuclear-target interaction model cannot be inferred from Ye alone.
Beta-decay daughters, weak evolution, and particle backreaction are absent.

The correction of the spurious extra factor of c in the gyration diagnostic
was valid. However, its retained baryon density is not automatically the target
density of every channel. The export now supplies the gyrofrequency and the
two explicit target encounter rates. Charge-conversion probabilities and
other target channels remain part of transport. The paper's test-particle
injection approximation does not become valid merely because one local
collision depth is near unity. The new optional cold shock closure uses
compression `4*Gamma_rel+3`, rather than the printed `4*(Gamma_rel+3)` in the
notes; the former follows the gamma-hat=4/3 conservation relations and passes
the velocity-transformation check.
[Kashiyama et al.](https://arxiv.org/pdf/1304.1945)

`n*sigma*r/Gamma_a` is a local dynamical-length convention borrowed from the
internal-shock setup. It is not the integrated exterior column of the BNS
ejecta. The export adds endpoint frozen-profile radial columns with the
outward relativistic attenuation factor `Gamma_a*(1-beta_a)`. These do not
replace evolving-medium transport along a particle's trajectory. For the
legacy grey snapshot opacity, `sigma_pn/(kappa*mp)=0.112`; the integrated
hadronic column near photon breakout is thus naturally smaller than a local
depth evaluated across the entire radius. This distinction materially affects
return and conversion probabilities.

## Checks on the rough estimates in the local notes

These are direct substitutions into the notes' own equations, independent of
the solver:

- `Mej=1e-3 Msun, r=1e11 cm, v=0.3c, t=1 s` gives approximately
  `n=1.05e21 cm^-3` in `M/(4*pi*mp*r^2*v*t)`, not `1e14--1e16`.
- `B0=1e13 G, r0=1e6 cm, r=1e10 cm` gives `B=1e5 G` for B proportional
  to r^-2. The quoted `1e7--1e10 G` is inconsistent with those inputs.
- `n=1e15 cm^-3, Gamma_rel=3` gives `kT=0.560 keV` from
  `(rho*c^2*Gamma_rel^2/a)^(1/4)`, rather than tens of keV.

These errors explain part of the earlier apparent physical tension. The
temperature and Bethe-Heitler estimates are still approximate diagnostics,
not a solved radiation field. An observer-energy boost using Gamma_h is not
a viewing-angle-dependent Doppler transformation.

## How to use the result

Use `examples/export_npc_models.py` and share only its final ZIP. The HDF5
schema is `jetbns.npc-model-export.v2` with `jetbns.npc-inputs.v4` per-model
arrays. The default suite runs without private data. Add `--include-numerical`
to include measured profiles. The LaTeX report defines the contract; the
included reader validates it and prints any model's endpoint as JSON. The
earlier plots and sweeps are historical diagnostic products, not additional
files needed for the transport setup.
