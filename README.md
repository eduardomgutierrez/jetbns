# jetbns

`jetbns` is a clean, tested Python implementation of semi-analytic models for
relativistic jets propagating through binary-neutron-star merger ejecta. It
currently provides ejecta profiles, one-sided luminosity engines, coupled
jet--cocoon propagation with pressure collimation, and deterministic inputs for
an external neutron--proton converter (NPC) Monte Carlo. Evolution is followed
through jet-head breakout; detached-cocoon radiation is a separate future layer.

The physical context is Gutiérrez et al., [*Cocoon shock breakout emission from
binary neutron star mergers*](https://arxiv.org/abs/2408.15973), Phys. Rev. D
111, 063031 (2025).

Start with the audited transfer workflow below. The [scientific audit](docs/scientific_audit.md)
records the corrected reconstruction and the remaining model assumptions.
Numerical exports made before this audit must be regenerated.

## NPC transfer workflow

After installation, run the self-contained analytical reference suite:

```bash
python examples/export_npc_models.py
```

This generates **one file to transfer**:
`examples/output/npc_model_export/npc_monte_carlo_inputs.zip`.
It contains the HDF5 input table and a LaTeX PDF with equations, definitions,
and representative evolution plots.
PDF generation requires `pdflatex`
(`texlive-latex-base` on Debian/Ubuntu). Every model is run at two or more
resolutions before a new archive can be produced.

The repository also provides an optional reader/validator; after copying it
alongside the HDF5 file, the receiving system needs only NumPy and h5py:

```bash
python -m pip install numpy h5py
python read_npc_inputs.py
python read_npc_inputs.py --model ana_reference
```

Three measured numerical profiles can be included when their local files are available:

```bash
python examples/export_npc_models.py --include-numerical --data-root /path/to/codes
```

The model list and input paths are explicit in `examples/export_npc_models.py`.
The numerical files are not bundled. Plots and build files beside the archive
are optional diagnostics. The reference suite uses a mass-conserving
`HomologousTail`; the earlier `BrokenPowerLaw` retains legacy behavior.

## Installation

Create a virtual environment and install the package with its development tools:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

The base package requires NumPy 2+ and h5py. Matplotlib is needed for examples.

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
from jetbns import ConstantEngine, HomologousPowerLaw, JetCocoon

ejecta = HomologousPowerLaw()
engine = ConstantEngine.from_isotropic_equivalent(
    5e51,
    launch_time_s=0.1,
    opening_angle_rad=np.deg2rad(6.8),
)
result = JetCocoon(engine, ejecta).propagate(max_time_s=3, time_step_s=2e-4)
print(result.broke_out, result.breakout_time_s)
```

Breakout is located where the optical depth ahead of the shock falls to
`1 / beta_s'`. For smooth numerical ejecta, propagation and the optical-depth
integral include the retained high-velocity material beyond nominal `r_max`.

`JetCocoon` evolves head position, cocoon width and energy together. Delayed
cocoon pressure changes the jet cross-section and therefore its head speed.
The returned arrays include `cocoon_energy_erg`, `cocoon_pressure_erg_cm3`, and
`jet_opening_angle_rad`. `time_step_s` is a maximum step; convergence should be
checked by halving it. See [equations and legacy correspondence](docs/jet_cocoon.md).
`JetHead` remains available for an explicitly conical comparison.

## Neutron--proton converter inputs

On the `project/np-converter` branch, a solved trajectory can be converted into
the deterministic shock quantities required by a separate NPC Monte Carlo:

```python
from jetbns import NpcConfig, evaluate_npc_inputs

config = NpcConfig(
    path_length="radius",       # notes/legacy default: Delta r = r
    electron_fraction=0.1,       # fallback when the ejecta has no Ye profile
    free_neutron_transition_mass_msun=1e-4,
)
inputs = evaluate_npc_inputs(result, ejecta, config=config)
inputs.to_hdf5("npc_inputs.h5", config=config, metadata={"run": "example"})
```

The table includes relative Lorentz factor, total baryon density, approximate
proton and surviving free-neutron densities, both directional nucleon optical
depths, gyration parameter, upstream magnetic field, downstream temperature,
both maximum-energy limits, and observer-frame maximum energy.
The exact breakout sample is omitted by default. Every HDF5 dataset records its
unit, and the configuration is stored with the output. The default observer
boost is the jet-head Lorentz factor; a scalar or array can be supplied
explicitly. This module does not perform particle injection, collision
sampling, conversion cycles, transport, or spectral synthesis.

The free-neutron abundance is the schematic outer-skin prescription of Metzger
et al. (2015), not a reaction-network result. The exported names are explicit:
`neutron_to_proton_optical_depth` uses the proton density as its target and
`proton_to_neutron_optical_depth` uses the free-neutron density. `xi(1)` follows
equation 7 of Kashiyama, Murase & Meszaros (2013),
`e B / (sigma_pn m_p c^2 n)`. This corrects an extra factor of `c` in the local
notes and legacy implementation. Individual NPC tables use schema version 4;
the multi-model transfer uses `jetbns.npc-model-export.v3`. Its `jet_cocoon`
subgroup records the coupled dynamics on the NPC time grid. It also records
effective upstream lengths, collision rates, shock speeds in both fluid frames,
and the cold hydrodynamic compression ratio. Proton densities remain an
explicit free-proton proxy; bound nuclei require a separate treatment.

## Examples and tests

```bash
python examples/plot_ejecta.py
python examples/plot_propagation.py
python examples/plot_npc_inputs.py
python examples/explore_npc_parameter_space.py
python examples/export_npc_models.py
pytest
ruff check .
```

The older parameter-space example screens 320 combinations using a legacy
profile and baryon-reference optical depth. Its historical results are documented in
[`docs/np_converter_parameter_study.md`](docs/np_converter_parameter_study.md).

Numerical ejecta profiles can be screened with the same NPC criteria by passing
one or more WhiskyTHC-style HDF5 files. The angular-bin solid angle is inferred
from the file's `theta` grid:

```bash
python examples/explore_npc_numerical.py path/to/lagrangian_profile.h5
python examples/plot_numerical_parameter_gallery.py path/to/lagrangian_profile.h5
```

This writes `npc_numerical_sweep.png` and `npc_numerical_sweep.h5` under
`examples/output/`. Numerical data are not bundled with the repository.
The gallery command writes ten representative six-panel diagnostics under
`examples/output/numerical_gallery/`.

Plots are written under `examples/output/`, which is ignored by Git.

Use the transfer workflow at the top of this README for new NPC calculations.
