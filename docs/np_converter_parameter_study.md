# NPC parameter screening study

This screening tests whether lower ejecta masses, later jet launches, different
isotropic-equivalent jet powers, and steeper exponential tails reconcile the
current jet/ejecta solution with the parameter regime proposed in the local NPC
notes. It is a model diagnostic, not a population prediction.

The Cartesian grid contains 320 scenarios:

- isotropic-equivalent ejecta mass: `1e-5`, `1e-4`, `1e-3`, `1e-2` solar masses;
- engine launch time: `0.1`, `0.3`, `1`, `3` seconds;
- isotropic-equivalent jet luminosity: `1e49` through `1e53` erg/s by decades;
- exponential-tail power: `2`, `4`, `8`, `16`.

All models use a 10-degree jet, engine Lorentz factor 100, `B0 = 1e13 G` at
10 km, and the notes' `Delta r = r` convention. The tail extent is 1.1 times
the nominal edge. This modest cutoff is necessary because the current default
extent of 3 combined with `max_beta = 0.6` creates a computational boundary
moving at 1.8 times the speed of light. Results therefore test the tail shape
only over this explicitly finite interval.

## Correction to the legacy gyration parameter

The first version of this study inherited an extra factor of `c` in the
denominator of `xi(1)` from the local notes and `jetBNS3`. Equation 7 of
Kashiyama, Murase & Meszaros (2013) instead gives

```text
xi(1) = e B_u / (sigma_pn m_p c^2 n_u).
```

Removing the extra factor raises every `xi(1)` by the speed of light,
approximately `3e10`. HDF5 exports using the corrected definition have schema
`jetbns.npc-inputs.v2`; version 1 values must not be used.

## Corrected result

Seven of the 320 scenarios simultaneously reach `Gamma_rel > 2`,
`0.1 <= tau_pn <= 2`, and `xi(1) > 1`. One representative successful sample
has approximately:

- ejecta mass `1e-5` solar masses;
- launch time 3 seconds;
- isotropic-equivalent luminosity `1e50` erg/s;
- tail exponent 16;
- `Gamma_rel = 5.1`, `tau_pn = 0.67`, and `xi(1) = 1.5e7`.

Four of the seven compatible trajectories also reach the configured finite
tail edge within the integration interval. They all have `1e-5` solar masses,
a 3-second launch, tail exponent 16, and luminosities from `1e50` through
`1e53` erg/s. Three weaker-jet cases enter the target parameter region but do
not reach that edge within 12 seconds after launch, so they must not be labeled
successful jet breakouts.

Thus the lower mass, later launch, and steep tail bring the optical depth into
the desired range, sufficient jet power raises the relative Lorentz factor,
and the published gyration condition is easily satisfied. In the entire grid,
172 scenarios reach `Gamma_rel > 2`, seven enter the desired optical-depth
interval, and all 320 reach `xi(1) > 1` at some point.

The parameter trends are physically coupled:

- Lower mass reduces `tau_pn` and raises `xi(1)` through the lower density.
- A later launch allows expansion to reduce density, but it also moves the
  interaction outward, where the flux-frozen magnetic field is weaker.
- Increasing jet power raises `Gamma_rel` and produces earlier breakout, which
  can leave a denser upstream medium and therefore worsen `tau_pn` and `xi(1)`.
- A steeper tail sharply lowers the terminal density. For the representative
  `1e-5` solar-mass, 1-second, `1e52` erg/s case, increasing the exponent from
  2 to 16 changes the near-breakout optical depth from about 331 to 10.5 and
  raises `xi(1)` from about `1.2e5` to `3.7e6`.

The tension is visible directly from the adopted equations. Their product is

```text
tau_pn * xi(1) = e B(r) Delta-r / (m_p c^2 Gamma_e).
```

Density and the p-n cross-section cancel. With `Delta r = r` and
`B(r) = B0 (r0/r)^2`, this product decreases as `1/r`; nevertheless, the
corrected normalization is large enough that gyration is not restrictive in
this grid. The main screening constraint is now obtaining `tau_pn` near unity,
followed by the Bethe--Heitler maximum energy. The field normalization and
radial scaling should still be varied because flux freezing is an uncertain
BNS-ejecta assumption, not an input required by the original NPC paper.

Run the reproducible screening with:

```bash
python examples/explore_npc_parameter_space.py
```

It writes a diagnostic plot and HDF5 summary under `examples/output/`.

## Numerical-ejecta screening

A separate screening used the polar bin (`itheta=00000`) from six locally
available, physically valid WhiskyTHC-style outflow histories: DD2 equal-mass,
DD2 asymmetric, BLh, two SFHo configurations, and SLy. The grid varied launch
time (`0.1`, `0.3`, `1`, `3` s), isotropic-equivalent luminosity (`1e49` through
`1e52` erg/s), and generalized-Gaussian velocity-kernel width (`0.02`, `0.035`,
`0.05`). The original 288-trajectory run inadvertently applied a sharp cutoff
in every case, even though varying the kernel width was described as changing
the fast tail. That result is retained below as historical context but should
not be interpreted as a comparison of cutoff modes.

The HDF5 loader was corrected before this run: mass flux stored for an angular
bin is now calculated with that bin's solid angle and converted to an
isotropic-equivalent value exactly once during reconstruction. The previous
implementation applied `4*pi/solid_angle` twice and overestimated the polar
density by about three orders of magnitude.

Of the 288 numerical trajectories, 131 enter the adopted NPC region and 70 both
enter it and reach the ejecta edge within the 12-second integration window.
Across compatible samples:

- `Gamma_rel` ranges from 3.95 to 46.1;
- `tau_pn` ranges from 0.107 to 1.999;
- `xi(1)` ranges from `6.9e6` to `2.1e9`;
- downstream `k_B T_d` ranges from 0.53 to 2.60 keV.

Compatibility counts are 2, 22, 53, and 54 for luminosities `1e49`, `1e50`,
`1e51`, and `1e52` erg/s, respectively. Kernel widths `0.02`, `0.035`, and
`0.05` give 57, 45, and 29 compatible trajectories. This width dependence
occurs within the nominal ejecta boundary; it is not evidence for the effect of
retaining material above the recorded maximum velocity.

The cleaned numerical model now explicitly supports `cutoff_mode="sharp"` and
`cutoff_mode="smooth"`, corresponding to legacy `abrupt_cutoff=1` and `0`.
It also exposes `kernel_shape`, corresponding to legacy `alpha`; shape 1 is an
exponential and shape 2 is the paper-era Gaussian choice. A corrected run on
the exact archived `DD2_M135135_M1_K2_SR/outflow_1_ber_out` profile evaluated 96
trajectories (48 per cutoff mode). Jet propagation now uses the legacy physical
breakout condition: the electron-scattering optical depth ahead of the shock is
integrated to `r_max` for a sharp profile and `3*r_max` for a smooth profile,
and breakout occurs at `tau = 1/beta_s'`, where `beta_s'` is obtained from the
relative head/ambient Lorentz factor and the relativistic jump conditions.

With that corrected criterion, 32 of 48 trajectories break out in each mode,
but none reaches the adopted NPC region before breakout; the minimum retained
pre-breakout `tau_pn` is 5.58. Sharp and smooth results are identical for this
profile and grid because their electron-scattering breakout surfaces lie inside
the nominal fastest-shell radius. The propagator does continue beyond `r_max`
when a smooth tail remains optically thick there, and this behavior has a
dedicated regression test.

These are screening results, not validated predictions. The local files are
not the exact `outflow_data/more` paths referenced by the paper-era notebooks;
their redistribution status is unknown. The clean loader currently uses the
recorded velocity rather than a Bernoulli-derived asymptotic velocity and
power-law extrapolates the outflow after the recorded history. Those choices
must be tested before publication use.

Run the numerical screening by passing one or more source files:

```bash
python examples/explore_npc_numerical.py path/to/lagrangian_profile.h5
```
