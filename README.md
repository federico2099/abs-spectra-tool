# spectra-tool

Photoabsorption cross sections (**spectra**) and spectral overlap metrics (**spec_overlap**) as a friendly CLI and Python API.

## Install
```bash
pip install -e .
# with optional extras:
pip install -e .[dev,plots]
```



## Installation options

By default, installing with:

```bash
pip install -e .
```

will give you the core functionality (CLI and API).

- If you also want **plotting support** (PNG figures), install with:

```bash
pip install -e .[plots]
```

- If you are a **developer** and want testing tools (`pytest`, `coverage`) as well as plotting:

```bash
pip install -e .[dev,plots]
```

These extras are defined in `pyproject.toml` under `[project.optional-dependencies]`.

## Command line
```bash
# Cross section (spectra)
spectra-tool spectra  --energies energies.dat --osc osc.dat  --nstates 5 --nsamp 100  --temp 298.15 --ref-index 1.2  --bro-fac 0.05 --lshape gau --lspoints 1000  --set-min 0.1 --set-max 6.0  --out-prefix abs --plot

# Spectral overlap (spec_overlap)
spectra-tool spec_overlap  --exp spec_exp.dat --calc spec_calc.dat   --lam-min 400 --lam-max 600 --dlam 0.1   --out-prefix overlap_job --plot
```

## JSON config
```jsonc
// examples/spectra_job.json
{
  "command": "spectra",
  "out_prefix": "WS_MC4H",
  "plot": true,
  "spectra": {
    "energies": "energies.csv",
    "osc": "osc.csv",
    "nstates": 25,
    "nsamp": 100,
    "temp": 298.15,
    "ref_index": 1.477,
    "bro_fac": 0.075,
    "lshape": "gau",
    "lspoints": 4000,
    "set_min": 0.3,
    "set_max": 8.0
  }
}
```

Run with:
```bash
spectra-tool --config examples/spectra_job.json
```

Another example:
```jsonc
// examples/spec_overlap_job.json
{
  "command": "spec_overlap",
  "out_prefix": "overlap_job",
  "plot": true,
  "spec_overlap": {
    "exp": "spec_exp.csv",
    "calc": "spec_calc.csv",
    "lam_min": 420.0,
    "lam_max": 560.0,
    "dlam": 0.1
  }
}
```

## Python API
```python
from spectra_tool import CrossSectionJob, OverlapJob
import numpy as np

# spectra
job1 = CrossSectionJob(
    energies=np.loadtxt('energies.csv', delimiter=','),
    osc=np.loadtxt('osc.csv', delimiter=','),
    nstates=25, nsamp=100,
    temp=298.15, ref_index=1.477, bro_fac=0.075,
    lshape='gau', lspoints=4000, set_min=0.3, set_max=8.0,
)
res1 = job1.compute(); res1.save('WS_MC4H', plot=True)

# spec_overlap
job2 = OverlapJob(
    exp=np.loadtxt('spec_exp.csv', delimiter=','),
    calc=np.loadtxt('spec_calc.csv', delimiter=','),
    lam_min=420.0, lam_max=560.0, dlam=0.1,
)
res2 = job2.compute(); res2.save('overlap_job', plot=True)
```

## Tests
```bash
pip install -e .[dev,plots]
pytest -q
```

## CI
A GitHub Actions workflow is provided in `.github/workflows/ci.yml` to run tests on each push/PR.
