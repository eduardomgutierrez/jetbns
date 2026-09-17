# Coupled jet and cocoon propagation

`JetCocoon(engine, ejecta).propagate(...)` is the default physical model in the
propagation example and NPC export. `JetHead` is the conical comparison model.
Both follow the head until its forward shock meets the opacity criterion.

The implementation ports the pre-breakout `Jet` and `Cocoon` routines from
`jetBNS3/modules/propagation.py` (also compared with `jetBNS2`). It follows
[Gutiérrez et al., equations 10--19](https://arxiv.org/html/2408.15973v2#S2.SS3):

\[
\dot z_h=\beta_h c,\qquad \dot r_c=\beta_{c,\perp}c,\qquad
\dot E_c=\eta L_j(t_{\rm eng})(\beta_j-\beta_h).
\]

The one-sided luminosity and the moving ambient medium enter the usual
momentum balance, with the existing calibration `N_s^2`. The crucial additional
coupling is that the jet area is determined by cocoon pressure:

\[
P_c=\frac{E_c(t-\Delta t_c)}{3V_c},\quad
V_c=\frac{2\pi}{3}(z_h-z_0)r_c^2,\quad
\Delta t_c=\frac{\sqrt3(z_h-z_0)}{2c},
\]
\[
\hat z=z_0+(\beta_j-\beta_a)\sqrt{\frac{L_j}{\pi\beta_j cP_c}},\qquad
\Sigma_j=\pi\tan^2\theta_0\min[z_h,(\hat z+z_0)/2]^2.
\]

The energy fraction is `eta=1` for `mu<=1`, otherwise `2/mu-1/mu^2`, with
`mu=sqrt(3)*Gamma_h*beta_h*theta_j`. Pressure drives a lateral speed
`sqrt(P_c/(P_c+rho_mean*c^2))` relative to the ejecta. As in the legacy code,
this is combined with ambient lateral expansion using relativistic speed
addition. Thus the head deposits energy, pressure changes the cocoon width and
jet area, and the changed area changes head speed and subsequent deposition.

## Explicit choices and differences from the legacy implementation

- The paper's ellipsoid height is `z_h-z_0`. The legacy volume used `z_h`,
  although its transverse contact point used `(z_h+z_0)/2`. The new volume and
  axial area weights consistently use an ellipsoid between base and head.
- As in the legacy model, density is averaged along the axis with ellipsoidal
  area weights. It assumes each transverse slice has the axial density;
  it is not an integration of a two-dimensional ejecta distribution.
- Both collimation and lateral expansion use the same delayed pressure by
  default. Delay is evaluated directly as printed in equation 13. The legacy
  code mixed instantaneous and delayed pressure and estimated its delay by a
  fixed-point iteration on a retarded head position. The accepted solution
  history is now used, with zero energy before injection. Setting
  `Cocoon(pressure_delay=False)` selects instantaneous pressure explicitly.
- The reconfinement equation is solved by bracketed bisection, evaluating
  ambient speed at `z_hat/cos(theta_0)` as in the legacy code. Jet luminosity is
  evaluated at the head; this is a stated local approximation for variable
  engines. It agrees with the legacy equation for steady powered jets while
  avoiding a luminosity query ahead of the causal jet front.
- Equation 14 prints a low-speed sum. The legacy relativistic addition is
  retained so the lateral speed remains subluminal.
- The initial head is `z_0*(1+1e-4)`, its lateral radius is
  `z_0*tan(theta_0)`, and energy starts at zero. Engine flight time to the
  initial head is included. Halving the initial height offset is tested.
- No extra `P dV` sink is appended: energy follows the published equation 19.
  Engine shutoff stops deposition; the inherited head momentum-balance closure
  then assumes ambient advection. An inertially coasting, choked head requires
  a different dynamical model. The reference suite is powered through breakout.

## Tests and outputs

Tests check the closed-form uniform-medium reconfinement solution against
the legacy pressure-balance equation, the conical zero-pressure limit, the
ellipsoidal density average, relativistic lateral speed, energy integral,
causal delay, pressure feedback, initial-height sensitivity, and resolution.
The export additionally refines the cocoon density quadrature with the head,
opacity, outflow-history and exterior-mass resolutions. It requires less than
5% endpoint changes in cocoon width, energy, pressure and jet area, as well as
the existing NPC quantities.

`CocoonPropagationResult` retains head quantities and adds the cocoon width,
deposited energy, delayed energy used for pressure, pressure, lateral speed,
jet area, opening angle, and deposition rate. The NPC exporter stores these
under each model's `jet_cocoon` group on the same time grid as the NPC arrays.
Schema `jetbns.npc-model-export.v3` supersedes conical v2 results. These are
inputs at the head's forward shock before head breakout. Detached-cocoon
evolution and its later radiation are outside this pre-breakout solver.
