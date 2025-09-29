# Absorption Spectra Tool

Photoabsorption cross sections (**spectra**) and spectral overlap metrics (**spec_overlap**) as a friendly CLI and Python API.

- **spectra**: compute photoabsorption cross sections from excitation energies and oscillator strengths.
- **spec_overlap**: compute shape/intensity agreement metrics between experimental and calculated spectra.

---

## Installation
Clone the repository and install locally:

```bash
git clone https://github.com/yourusername/abs-spectra-tool.git
cd abs-spectra-tool
pip install -e .
# with optional extras:
pip install -e .[dev,plots]
```

This will install the `spectra-tool` command.

---

## Usage

### CLI example (spectra)
```bash
spectra-tool spectra  --energies energies.dat --osc osc.dat  --nstates 5 --nsamp 100  --temp 298.15 --ref-index 1.2  --bro-fac 0.05 --lshape gau --lspoints 1000  --set-min 0.1 --set-max 6.0  --out-prefix abs --plot --plot-xmin 430 --plot-xmax 560
```

### CLI example (spectra overlap)
```bash
spectra-tool spec_overlap  --exp spec_exp.dat --calc spec_calc.dat   --lam-min 400 --lam-max 600 --dlam 0.1   --out-prefix overlap_job --plot --plot-xmin 430 --plot-xmax 560
```

### JSON config example
```bash
spectra-tool --config config.json
```
---

# Command-line Options and Config Keywords

Every option can be provided either:

- **via CLI**, e.g. `--energies energies.dat`
- **via JSON config file**, e.g. `"energies": "energies.dat"`

Both interfaces accept the same keywords. In JSON, use `snake_case` (e.g. `lam_min`) rather than dashes.

---

## Global Options

| Keyword / CLI flag | Description |
|--------------------|-------------|
| `--config` | Path to JSON config file (recommended for non-experts). |
| `--verbose` | Enable verbose logging. |
| `--out-prefix` / `out_prefix` | Prefix for all output files (CSV/NPZ/plots). |
| `--plot` / `plot` | If set, save PNG plots of spectra/overlap results. |

---

## `spectra` (Cross Section) Options

| Keyword / CLI flag | Description |
|--------------------|-------------|
| `--energies` / `energies` | Path to file with excitation energies (eV). Formats: CSV, TXT, DAT, NPY, NPZ. Each column = state, each row = conformation. |
| `--osc` / `osc` | Path to file with oscillator strengths. Same shape as `energies`. |
| `--nstates` / `nstates` | Number of electronic states to include (columns). |
| `--nsamp` / `nsamp` | Number of sampled conformations (rows). Default: all rows. |
| `--temp` / `temp` | Temperature in Kelvin. If >0, applies thermal factor to intensities. |
| `--ref-index` / `ref_index` | Refractive index of medium (default 1.0). |
| `--bro-fac` / `bro_fac` | Broadening factor (in eV). Width of Gaussian/Lorentzian lines. |
| `--lshape` / `lshape` | Lineshape type: `gau`, `lor`, or `voi`. |
| `--lspoints` / `lspoints` | Number of points in the lineshape grid. |
| `--set-min` / `set_min` | Minimum energy (eV) of the grid (defaults to min(energies)-0.5). |
| `--set-max` / `set_max` | Maximum energy (eV) of the grid (defaults to max(energies)+0.5). |
| `--plot-xmin` / `plot_xmin` | Minimum wavelength (nm) shown in cross section plot (x-axis only). |
| `--plot-xmax` / `plot_xmax` | Maximum wavelength (nm) shown in cross section plot (x-axis only). |

---

## `spec_overlap` (Spectral Overlap) Options

| Keyword / CLI flag | Description |
|--------------------|-------------|
| `--exp` / `exp` | Experimental/reference spectrum file. Two columns: [wavelength_nm, intensity]. Formats: CSV, TXT, DAT, NPY, NPZ. |
| `--calc` / `calc` | Calculated spectrum file. Same format as `--exp`. |
| `--lam-min` / `lam_min` | Minimum wavelength (nm) for comparison domain. |
| `--lam-max` / `lam_max` | Maximum wavelength (nm) for comparison domain. |
| `--dlam` / `dlam` | Step size (nm) for uniform comparison grid. Ignored if using `grid_mode=union`. |
| `--grid-mode` / `grid_mode` | Grid construction: `uniform` (regular spacing via `dlam`) or `union`. |
| `--plot-xmin` / `plot_xmin` | Minimum wavelength (nm) shown in overlap plot (x-axis only). |
| `--plot-xmax` / `plot_xmax` | Maximum wavelength (nm) shown in overlap plot (x-axis only). |

---

## File Formats

- `.csv`, `.txt`, `.dat` (comma or whitespace separated).
- `.npy` → NumPy binary array.
- `.npz` → NumPy compressed archive (first array inside is used).

---

## Example JSON config

### Cross Section
```json
// examples/spectra_job.json
{
  "command": "spectra",
  "out_prefix": "abs",
  "plot": true,
  "spectra": {
    "energies": "energies.dat",
    "osc": "osc.dat",
    "nstates": 5,
    "nsamp": 100,
    "temp": 298.15,
    "ref_index": 1.2,
    "bro_fac": 0.05,
    "lshape": "gau",
    "lspoints": 1000,
    "set_min": 0.1,
    "set_max": 6.0
    "plot_xmin": 430.0,
    "plot_xmax": 560.0
  }
}
```

### Spectral Overlap
```json
{
  "command": "spec_overlap",
  "out_prefix": "overlap",
  "plot": true,
  "spec_overlap": {
    "exp": "Exp_spec.txt",
    "calc": "Calc_spec.dat",
    "lam_min": 430.0,
    "lam_max": 560.0,
    "dlam": 0.1,
    "grid_mode": "uniform",
    "plot_xmin": 430,
    "plot_xmax": 550
  }
}
```

---

## Expected Outputs

### Cross Section (`spectra`)
- `<prefix>_grid.csv` — energy grid, wavelength grid, total and error sigma.
- `<prefix>_sigma_per_state.csv` — wavelength grid and per-state cross sections.
- `<prefix>_sigma_per_state.npz` — compressed binary version (NumPy).
- `<prefix>_total.png` — plot of total cross section (if `--plot` used).

### Spectral Overlap (`spec_overlap`)
- `<prefix>_overlap_metrics.json` — metrics (OA_norm, OA_raw, RMSE, etc).
- `<prefix>_common_grid.csv` — common wavelength grid with interpolated spectra.
- `<prefix>_overlap.png` — overlap plot (if `--plot` used).
- `<prefix>_summary.txt` — human-readable summary of match quality.

---

  
## Python API
You can also use this package directly from Python. The public entry points are:

- `CrossSectionJob` → computes photoabsorption cross sections.
- `OverlapJob` → computes spectral overlap metrics.

### 1) Cross section from NumPy arrays

```python
from spectra_tool import CrossSectionJob
import numpy as np

# Example input arrays: shape (npoints, nstates)
energies = np.loadtxt("energies.dat")  # eV
osc = np.loadtxt("osc.dat")

# spectra
calc = CrossSectionJob(
    energies=energies,
    osc=osc,
    nstates=energies.shape[1],
    nsamp=None,          # or an int to subsample first N rows
    temp=298.15,
    ref_index=1.2,
    bro_fac=0.05,
    lshape="gau",
    lspoints=2000,
    set_min=0.3,         # energy grid min (eV)
    set_max=8.0,         # energy grid max (eV)

spec = calc.compute()

# Access arrays
lam = spec.lambda_grid_nm          # (n_grid,)
sigma_total = spec.total_sigma     # (n_grid,)
sigma_states = spec.sigma_per_state  # (n_grid, nstates)

# Save outputs (optionally with plot limits for the figure)
spec.save("Calc_spectrum",
         plot=True,
         plot_xmin=430,  # nm (figure only)
         plot_xmax=560)  # nm (figure only)
```

### 2) Spectral overlap from NumPy arrays

```python
import numpy as np
from spectra_tool import OverlapJob

# Two-column arrays: [wavelength_nm, intensity]
exp = np.loadtxt("Exp_spectrum.txt")
cal = np.loadtxt("Calc_spectrum.txt")

calc = OverlapJob(
    exp=exp[:, :2],
    calc=cal[:, :2],
    lam_min=430.0,
    lam_max=560.0,
    dlam=0.1,         
    grid_mode="uniform"
)

overlap = calc.compute()  # OverlapResult

# Metrics (as numbers)
print(overlap.OA_norm, overlap.OA_raw, overlap.RMSE_raw, overlap.MAE_raw, overlap.s_opt, overlap.RMSE_scaled)

# Human-readable summary
print(overlap.summarize())

# Save outputs (optionally with plot limits for the figure)
overlap.save("overlap",
         plot=True,
         plot_xmin=430,  # nm (figure only)
         plot_xmax=550)  # nm (figure only)
```

### 3) Programmatic run with a config dict

This mirrors the JSON file structure

```python
from spectra_tool import SpectraTool  # adjust import to your package/module name

cfg = {
    "command": "spectra",
    "out_prefix": "Calc_spectrum",
    "plot": True,
    "spectra": {
        "energies": "energies.dat",
        "osc": "osc.dat",
        "nstates": 5,
        "nsamp": 100,
        "temp": 298.15,
        "ref_index": 1.2,
        "bro_fac": 0.05,
        "lshape": "gau",
        "lspoints": 2000,
        "set_min": 0.3,
        "set_max": 8.0,
        "plot_xmin": 430,
        "plot_xmax": 560
    }
}

SpectraTool.run_config(cfg, verbose=True)
```

### Data shape expectations

- **Cross section**: `energies` and `osc` must be the **same shape** `(npoints, nstates)`. Each **column** is a state; each **row** is a sampled conformation/point.
- **Overlap**: input arrays are two-column `[wavelength_nm, intensity]`. Grids may be irregular; the tool interpolates to the requested domain.

### Plot limits vs. calculation windows

- `plot_xmin`, `plot_xmax` control only the **figure x-axis** (visualisation).
- Cross section computation window is set by **energy** bounds `set_min`, `set_max` (in eV).
- Overlap comparison window is set by **wavelength** bounds `lam_min`, `lam_max` (in nm), with grid built by `dlam`.

### Saving and reloading

- CSV outputs include a `#` header so they can be safely read with `numpy.loadtxt` (header ignored).
- For large per-state matrices, use the NPZ (`*_sigma_per_state.npz`) for efficient reload:
  ```python
  data = np.load("spec_sigma_per_state.npz")
  lam = data["lambda_grid_nm"]
  sigma = data["sigma_matrix"]
  ```


A GitHub Actions workflow is provided in `.github/workflows/ci.yml` to run tests on each push/PR.
