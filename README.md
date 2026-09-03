# jetbns

`jetbns` is a clean, tested Python implementation of semi-analytic models for
relativistic jets propagating through binary-neutron-star merger ejecta. The
first milestone provides ejecta profiles; engine, jet/cocoon propagation, and
radiation will be added only after their legacy results have reproducible
regression tests.

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

The base package requires only NumPy. Matplotlib is needed for examples, and
h5py is needed only to import legacy WhiskyTHC HDF5 data.

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

Numerical profiles are constructed from an outflow history. The portable CSV
schema is:

```text
time_s,velocity_c,mass_loss_rate_g_s,electron_fraction
```

```python
from jetbns import NumericalEjecta, OutflowHistory

history = OutflowHistory.from_csv("data/example_outflow.csv")
ejecta = NumericalEjecta(history)
rho = ejecta.density(radius=1e9, time=0.2)
```

The example dataset is synthetic and exists to demonstrate the interface; it
must not be interpreted as simulation output. For the older WhiskyTHC layout,
use `OutflowHistory.from_legacy_hdf5(...)` and specify the angular-bin group and
extraction radius explicitly.

## Examples and tests

```bash
python examples/plot_ejecta.py
pytest
ruff check .
```

The plot is written to `examples/output/ejecta_profiles.png`, which is ignored
by Git. See `PROJECT_CONTEXT.md` for model assumptions and `NEXT_STEPS.md` for
the exact handoff state.

