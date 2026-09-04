# jetbns

`jetbns` is a clean, tested Python implementation of semi-analytic models for
relativistic jets propagating through binary-neutron-star merger ejecta. It
currently provides ejecta profiles, one-sided luminosity engines, an
uncollimated relativistic jet-head propagator, and deterministic inputs for an
external neutron--proton converter (NPC) Monte Carlo. Cocoon collimation and
radiation will be added only with reproducible regression tests.

The physical context is Gutiérrez et al., [*Cocoon shock breakout emission from
binary neutron star mergers*](https://arxiv.org/abs/2408.15973), Phys. Rev. D
111, 063031 (2025).

## Installation

Create a virtual environment and install the package with its development tools:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

The base package requires NumPy and h5py. Matplotlib is needed for examples.

## Ejecta models

All methods use CGS units. Constructor masses are in solar masses and speeds
are supplied as `beta = v/c`. Densities and masses are isotropic-equivalent,
which is the convention used for a selected polar angular bin in the legacy
code.

```python
from jetbns import BrokenPowerLaw, HomologousPowerLaw

ejecta = HomologousPowerLaw(mass_msun=0.002, density_index=2, max_beta=0.35)
rho = ejecta.density(radius=1e9, time=0.2)  # g cm^-3

ejecta_with_tail = BrokenPowerLaw(
    mass_msun=0.002,
    inner_index=2,
    outer_index=6,
    break_beta=0.3,
    max_beta=0.6,
    tail=True,
)
```

Persisted numerical profiles use the WhiskyTHC-style HDF5 format. Select the
angular-bin group and state its extraction radius and solid angle explicitly:

```python
from jetbns import NumericalEjecta

ejecta = NumericalEjecta.from_hdf5(
    "path/to/outflow.h5",
    bin_name="itheta=00000",
    extraction_radius_cm=4.42e7,
    solid_angle_sr=0.2,
)
rho = ejecta.density(radius=1e9, time=0.2)
```

`OutflowHistory` can still be constructed directly from in-memory arrays for
synthetic tests and generated workflows. CSV file loading is intentionally not
part of the API.

## Engine and jet-head propagation

Engine luminosities are true one-sided jet powers. The convenience constructor
converts a top-hat isotropic-equivalent luminosity using its opening solid angle:

```python
import numpy as np
from jetbns import ConstantEngine, HomologousPowerLaw, JetHead

ejecta = HomologousPowerLaw()
engine = ConstantEngine.from_isotropic_equivalent(
    5e51,
    launch_time_s=0.1,
    opening_angle_rad=np.deg2rad(6.8),
)
result = JetHead(engine, ejecta).propagate(max_time_s=3, time_step_s=2e-4)
print(result.broke_out, result.breakout_time_s)
```

Breakout is located where the optical depth ahead of the shock falls to
`1 / beta_s'`. For smooth numerical ejecta, propagation and the optical-depth
integral include the retained high-velocity material beyond nominal `r_max`.

`JetHead` implements momentum-flux balance for a conical, uncollimated jet and
includes the ambient ejecta velocity and retarded engine luminosity. The fixed
integration step is explicit; convergence should be checked by halving it.

## Neutron--proton converter inputs

On the `project/np-converter` branch, a solved trajectory can be converted into
the deterministic shock quantities required by a separate NPC Monte Carlo:

```python
from jetbns import NpcConfig, evaluate_npc_inputs

config = NpcConfig(
    path_length="radius",          # notes/legacy default: Delta r = r
    target_nucleon_fraction=1.0,   # replace when a composition model is known
)
inputs = evaluate_npc_inputs(result, ejecta, config=config)
inputs.to_hdf5("npc_inputs.h5", config=config, metadata={"run": "example"})
```

The table includes relative Lorentz factor, hadronuclear optical depth,
gyration parameter, upstream density and magnetic field, downstream
temperature, both maximum-energy limits, and observer-frame maximum energy.
The exact breakout sample is omitted by default. Every HDF5 dataset records its
unit, and the configuration is stored with the output. The default observer
boost is the jet-head Lorentz factor; a scalar or array can be supplied
explicitly. This module does not perform particle injection, collision
sampling, conversion cycles, transport, or spectral synthesis.

`xi(1)` follows equation 7 of Kashiyama, Murase & Meszaros (2013),
`e B / (sigma_pn m_p c^2 n)`. This corrects an extra factor of `c` in the local
notes and legacy implementation. Corrected HDF5 files use schema version 2;
schema-version-1 NPC files should be regenerated.

## Examples and tests

```bash
python examples/plot_ejecta.py
python examples/plot_propagation.py
python examples/plot_npc_inputs.py
python examples/explore_npc_parameter_space.py
pytest
ruff check .
```

The parameter-space example screens 320 combinations of mass, launch time,
jet power, and tail exponent. Its assumptions, results, and the current finite-
tail boundary caveat are documented in
[`docs/np_converter_parameter_study.md`](docs/np_converter_parameter_study.md).

Numerical ejecta profiles can be screened with the same NPC criteria by passing
one or more WhiskyTHC-style HDF5 files. The angular-bin solid angle is inferred
from the file's `theta` grid:

```bash
python examples/explore_npc_numerical.py path/to/lagrangian_profile.h5
```

This writes `npc_numerical_sweep.png` and `npc_numerical_sweep.h5` under
`examples/output/`. Numerical data are not bundled with the repository.

Plots are written under `examples/output/`, which is ignored by Git.
