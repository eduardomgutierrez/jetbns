# jetbns

`jetbns` is a clean, tested Python implementation of semi-analytic models for
relativistic jets propagating through binary-neutron-star merger ejecta. It
currently provides ejecta profiles, one-sided luminosity engines, and an
uncollimated relativistic jet-head propagator. Cocoon collimation and radiation
will be added only with reproducible regression tests.

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

`JetHead` implements momentum-flux balance for a conical, uncollimated jet and
includes the ambient ejecta velocity and retarded engine luminosity. The fixed
integration step is explicit; convergence should be checked by halving it.

## Examples and tests

```bash
python examples/plot_ejecta.py
python examples/plot_propagation.py
pytest
ruff check .
```

Plots are written under `examples/output/`, which is ignored by Git.
