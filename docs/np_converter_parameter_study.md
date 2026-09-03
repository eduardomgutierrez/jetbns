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

## Result

None of the 320 scenarios simultaneously reaches
`Gamma_rel > 2`, `0.1 <= tau_pn <= 2`, and `xi(1) > 1`. The most favorable
sample has approximately:

- ejecta mass `1e-5` solar masses;
- launch time 3 seconds;
- isotropic-equivalent luminosity `1e50` erg/s;
- tail exponent 16;
- `Gamma_rel = 5.1`, `tau_pn = 0.67`, and `xi(1) = 4.9e-4`.

Thus the lower mass, later launch, and steep tail bring the optical depth into
the desired range and a sufficient jet power raises the relative Lorentz
factor, but the gyration condition remains short by about three orders of
magnitude. In the entire grid, 172 scenarios reach `Gamma_rel > 2`, seven enter
the desired optical-depth interval, and none reaches `xi(1) > 1`.

The parameter trends are physically coupled:

- Lower mass reduces `tau_pn` and raises `xi(1)` through the lower density.
- A later launch allows expansion to reduce density, but it also moves the
  interaction outward, where the flux-frozen magnetic field is weaker.
- Increasing jet power raises `Gamma_rel` and produces earlier breakout, which
  can leave a denser upstream medium and therefore worsen `tau_pn` and `xi(1)`.
- A steeper tail sharply lowers the terminal density. For the representative
  `1e-5` solar-mass, 1-second, `1e52` erg/s case, increasing the exponent from
  2 to 16 changes the near-breakout optical depth from about 331 to 10.5 and
  raises `xi(1)` from `4.0e-6` to `1.2e-4`.

The tension is visible directly from the adopted equations. Their product is

```text
tau_pn * xi(1) = e B(r) Delta-r / (m_p c^3 Gamma_e).
```

Density and the p-n cross-section cancel. With `Delta r = r` and
`B(r) = B0 (r0/r)^2`, this product decreases as `1/r`. Reducing mass can move a
trajectory along this relation, but cannot raise the product. At radii of order
`1e10` cm with the fiducial field normalization, `tau_pn` in the desired range
therefore implies `xi(1) << 1`.

For confirmation, mathematically compatible solutions appear only for ejecta
masses around `1e-14` to `5e-14` solar masses at the launch radius in this
model, far below plausible merger ejecta masses. A larger magnetic
normalization, a different radial field scaling, a shorter-radius interaction,
or a revision of the assumed path/field equations is required to remove the
tension. Those changes were not included in this sweep.

Run the reproducible screening with:

```bash
python examples/explore_npc_parameter_space.py
```

It writes a diagnostic plot and HDF5 summary under `examples/output/`.
