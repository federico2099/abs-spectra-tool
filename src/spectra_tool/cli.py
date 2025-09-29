#!/usr/bin/env python3
"""
Absorption Spectra Tool
Subcommands:
  - spectra       -> photoabsorption cross section
  - spec_overlap  -> spectral overlap metrics

Also supports JSON config with: { "command": "spectra" | "spec_overlap", ... }

************************** Author ****************************
!                                                            !
!                   Federico J. Hernandez                    !
!                         Sep 2025                           !
!                                                            !
**************************************************************

"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from typing import Optional, Dict

import numpy as np

# Optional plotting
try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover
    plt = None

# Logging
def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format='[%(levelname)s] %(message)s')

# Config utils
def normalize_keys(obj):
    """Recursively convert all dict keys from hyphen-case to underscore_case."""
    if isinstance(obj, dict):
        return {str(k).replace('-', '_'): normalize_keys(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [normalize_keys(x) for x in obj]
    else:
        return obj

# I/O utils
def load_array(path: str) -> np.ndarray:
    """Load CSV/NPY/NPZ to 2D numpy array."""
    ext = os.path.splitext(path)[1].lower()
    if ext == '.npy':
        arr = np.load(path)
    elif ext == '.npz':
        data = np.load(path)
        key = list(data.keys())[0]
        arr = data[key]
    else:
        try:
            arr = np.loadtxt(path, delimiter=',')
        except Exception:
            try:
                arr = np.loadtxt(path, delimiter=',', skiprows=1)
        except Exception:
            try:
                arr = np.loadtxt(path)
            except Exception:
                arr = np.loadtxt(path, skiprows=1)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr

def save_csv(path: str, array: np.ndarray, header: Optional[str] = None) -> None:
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    np.savetxt(path, array, delimiter=',', header=header or '')

# Auxiliary Physical functions
def eV2nm(energies_eV: np.ndarray) -> np.ndarray:
    """
      Convert an array of energies in eV to wavelengths in nanometers

      Parameters:
      energies_eV (np.array): 1D array of wavelengths in nanometers

      Returns:
      np.array: 1D array of wavelengths in nanometers
    """
    from scipy.constants import h, c, eV
    energies_joules = np.asarray(energies_eV) * eV
    wavelengths_m = (h * c) / energies_joules
    return wavelengths_m / 1e-9

def nm2eV(wavelengths_nm: np.ndarray) -> np.ndarray:
    """
      Convert an array of wavelengths in nanometers to energies in electronvolts

      Parameters:
      wavelengths_nm (np.array): 1D array of wavelengths in nanometers

      Returns:
      np.array: 1D array of energies in electronvolts
    """
    from scipy.constants import h, c, eV
    wavelengths_m = wavelengths_nm * 1e-9
    energies_joules = (h * c) / wavelengths_m
    energies_eV = energies_joules / eV
    return energies_eV

def ls_func(shape: str, grid: np.ndarray, broad, ener: float) -> np.ndarray:
    '''
      Lineshape function: Gaussian, Lorentzian or Voigt

      Parameters:
          shape (str): 'gau' for Gaussian, 'lor' for Lorentzian, 'voi' for Voigt.
          grid (ndarray): Array representing the grid.
          broad (float): Broadening factor. Sigma for Gaussian, gamma for Lorentzian
                         and np.array[sigma,gamma] for Voigt
          ener (float): Energy value.

      Returns:
          ndarray: Calculated lineshape function values.
    '''
    from scipy.special import wofz  # Voigt
    hbar = 1.0
    if shape == 'gau':
        pre_fac = hbar / broad * (2.0 / np.pi) ** 0.5
        return pre_fac * np.exp(-2.0 * (grid - ener) ** 2 / broad ** 2)
    elif shape == 'lor':
        pre_fac = hbar * broad / (2.0 * np.pi)
        return pre_fac / ((grid - ener) ** 2 + (broad / 2.0) ** 2)
    elif shape == 'voi':
        pre_fac = ((grid - ener) + 1j * broad[1]) / (broad[0] * np.sqrt(2))
        return np.real(wofz(pre_fac)) / (broad[0] * np.sqrt(2 * np.pi))
    else:
        raise ValueError("Unknown line shape; use 'gau', 'lor', or 'voi'")

# Results dataclasses
@dataclass
class CrossSectionResult:
    energy_grid_eV: np.ndarray
    lambda_grid_nm: np.ndarray
    total_sigma: np.ndarray
    total_err_sigma: np.ndarray
    sigma_per_state: np.ndarray  # (n_grid, nstates)

    def save(self, out_prefix: str, plot: bool = False,
             plot_xmin: Optional[float] = None,
             plot_xmax: Optional[float] = None) -> None:
        grid_mat = np.column_stack([
            self.energy_grid_eV,
            self.lambda_grid_nm,
            self.total_sigma,
            self.total_err_sigma,
        ])
        save_csv(
            f"{out_prefix}_grid.csv",
            grid_mat,
            header='energy_eV,         wavelength_nm,         total_sigma_cm2,          total_err_sigma_cm2',
        )
        per_state = np.column_stack([self.lambda_grid_nm, self.sigma_per_state])
        headers = ['wavelength_nm'] + [f'sigma_state_{i+1}_cm2' for i in range(self.sigma_per_state.shape[1])]
        save_csv(f"{out_prefix}_sigma_per_state.csv", per_state, header=','.join(headers))
        np.savez_compressed(
            f"{out_prefix}_sigma_per_state.npz",
            lambda_grid_nm=self.lambda_grid_nm,
            energy_grid_eV=self.energy_grid_eV,
            sigma_matrix=self.sigma_per_state,
        )
        if plot and plt is not None:
            fig = plt.figure()
            plt.plot(self.lambda_grid_nm, self.total_sigma, label='Total σ')
            plt.xlabel('Wavelength (nm)'); plt.ylabel('Cross section (cm$^2$)')
            plt.title('Photoabsorption cross section ($\sigma$)'); plt.legend(); plt.tight_layout()
            if plot_xmin is not None or plot_xmax is not None:
                plt.xlim(plot_xmin, plot_xmax)
            fig.savefig(f"{out_prefix}_total.png", dpi=400)
            plt.close(fig)

@dataclass
class OverlapResult:
    # Data
    lambda_grid_nm: np.ndarray
    y_exp_interp: np.ndarray # Experimental spec
    y_calc_interp: np.ndarray # Calculated spec
    # Metrics (shape & intensity)
    OA_norm: float
    OA_raw: float
    RMSE_raw: float
    MAE_raw: float
    area_exp: float
    area_cal: float
    area_ratio: float
    s_opt: float
    RMSE_scaled: float

    def metrics_dict(self) -> Dict[str, float]:
        """Return metrics as a plain dict."""
        return {
            'OA_norm': float(self.OA_norm),
            'OA_raw': float(self.OA_raw),
            'RMSE_raw': float(self.RMSE_raw),
            'MAE_raw': float(self.MAE_raw),
            'area_exp': float(self.area_exp),
            'area_cal': float(self.area_cal),
            'area_ratio': float(self.area_ratio),
            's_opt': float(self.s_opt),
            'RMSE_scaled': float(self.RMSE_scaled),
        }

    def summarize(self) -> str:
        """Human-readable verdict combining shape & intensity agreement."""
        lam = np.asarray(self.lambda_grid_nm)
        L = float(lam[-1] - lam[0]) if lam.size >= 2 else np.nan
        OA = float(self.OA_norm)

        if OA >= 0.95:
            shape_verdict = "excellent"
        elif OA >= 0.9:
            shape_verdict = "very good"
        elif OA >= 0.8:
            shape_verdict = "good"
        elif OA >= 0.7:
            shape_verdict = "fair"
        else:
            shape_verdict = "poor"

        area_ratio = float(self.area_ratio)
        rmse_raw = float(self.RMSE_raw)
        mae_raw = float(self.MAE_raw)

        mean_exp = (self.area_exp / L) if (np.isfinite(self.area_exp) and np.isfinite(L) and L > 0) else np.nan
        rmse_pct = (rmse_raw / mean_exp * 100.0) if (np.isfinite(mean_exp) and mean_exp > 0) else np.nan
        mae_pct  = (mae_raw  / mean_exp * 100.0) if (np.isfinite(mean_exp) and mean_exp > 0) else np.nan
        bias_pct = ((area_ratio - 1.0) * 100.0) if np.isfinite(area_ratio) else np.nan

        if np.isfinite(rmse_pct) and np.isfinite(bias_pct):
            if abs(bias_pct) <= 0 and rmse_pct <= 5:
                intensity_verdict = "excellent"
            elif abs(bias_pct) <= 5 and rmse_pct <= 15:
                intensity_verdict = "very good"
            elif abs(bias_pct) <= 15 and rmse_pct <= 25:
                intensity_verdict = "good"
            elif abs(bias_pct) <= 30 and rmse_pct <= 40:
                intensity_verdict = "fair"
            else:
                intensity_verdict = "poor"
        else:
            intensity_verdict = "undetermined"

        lines = []
        lines.append("— Shape comparison (normalised): " + shape_verdict)
        lines.append(f"   OA_norm={OA:.3f}")
        lines.append("— Intensity comparison (raw ε): " + intensity_verdict)
        if np.isfinite(bias_pct):
            lines.append(f"   area_ratio={area_ratio:.3f} ({bias_pct:+.1f}% bias), "
                         f"RMSE_raw={rmse_raw:.4g} (≈{rmse_pct:.1f}% of mean ε), "
                         f"MAE_raw={mae_raw:.4g} (≈{mae_pct:.1f}% of mean ε)")
        else:
            lines.append(f"   RMSE_raw={rmse_raw:.4g}, MAE_raw={mae_raw:.4g}")

        if shape_verdict in {"excellent", "very good", "good"} and intensity_verdict in {"excellent", "very good", "good"}:
            overall = "Overall: strong match in both shape and intensities."
        elif shape_verdict in {"excellent", "very good", "good"} and intensity_verdict in {"fair", "poor"}:
            overall = "Overall: good shape agreement but intensity mismatch."
        elif shape_verdict in {"fair"} and intensity_verdict in {"excellent", "very good", "good"}:
            overall = "Overall: intensities align, but shape agreement could improve."
        else:
            overall = "Overall: limited agreement; both shape and intensity need work."

        lines.append(overall)
        return "\n".join(lines)

    def save(self, out_prefix: str, plot: bool = False,
             plot_xmin: Optional[float] = None,
             plot_xmax: Optional[float] = None) -> None:
        # Metrics to JSON
        with open(f"{out_prefix}_overlap_metrics.json", 'w') as f:
            json.dump(self.metrics_dict(), f, indent=2)

        # Save the interpolated intensities using a common grid
        grid = np.column_stack([self.lambda_grid_nm, self.y_exp_interp, self.y_calc_interp])
        save_csv(f"{out_prefix}_common_grid.csv", grid, header='wavelength_nm,exp_interp,calc_interp')

        # Save scaled experimental intensity (using s_opt) and residuals
        scaled_exp = self.s_opt * self.y_exp_interp
        residuals = self.y_calc_interp - scaled_exp
        save_csv(f"{out_prefix}_scaled_exp.csv",
                 np.column_stack([self.lambda_grid_nm, scaled_exp]),
                 header='wavelength_nm,scaled_exp_intensity')
        save_csv(f"{out_prefix}_residuals.csv",
                 np.column_stack([self.lambda_grid_nm, residuals]),
                 header='wavelength_nm,residual_calc_minus_scaled_exp')

        # Save the summary
        with open(f"{out_prefix}_summary.txt", "w") as f:
            f.write(self.summarize() + "\n")

        # (Optional) if plot, generate the figure with the two spectra
        if plot and plt is not None:
            fig = plt.figure()
            plt.plot(self.lambda_grid_nm, self.y_exp_interp, label='EXP (interp)')
            plt.plot(self.lambda_grid_nm, self.y_calc_interp, label='CALC (interp)')
            plt.xlabel('Wavelength (nm)'); plt.ylabel('Intensity')
            plt.title('Spectral overlap window'); plt.legend(); plt.tight_layout()
            if (plot_xmin is not None) or (plot_xmax is not None):
                plt.xlim(plot_xmin, plot_xmax)
            fig.savefig(f"{out_prefix}_overlap.png", dpi=400)
            plt.close(fig)

# ------------- Job classes -------------
@dataclass
class CrossSectionJob:
    energies: np.ndarray  # Array of energies in eV. Each column corresponds to one state (npoints, nstates)
    osc: np.ndarray       # Array of oscillator strengths. Each column corresponds to one state (npoints, nstates)
    nstates: int          # Number of electronic states consdiered
    nsamp: Optional[int] = None # Number of sampled conformations
    temp: float = 0.0   # Temperature in Kelvin
    ref_index: float = 1.0  # Refractive index
    bro_fac: float = 0.05 # Broadening factor
    lshape: str = 'gau' # Lineshape function ('gau' for Gaussian, 'lor' for Lorentzian)
    lspoints: int = 2000 # Number of points in lineshape grid
    set_min: Optional[float] = None # Minimum energy considered for the grid 
    set_max: Optional[float] = None # Maximum energy considered for the grid

    def compute(self) -> CrossSectionResult:
        """
        Returns:
        tuple: (ls_grid, total_sigma, sigma)
            ls_grid (ndarray): Lineshape grid.
            total_sigma (ndarray): Total photoabsorption cross-section.
            sigma (ndarray): Photoabsorption cross-section for each state
        """
        from scipy.constants import physical_constants, m_e, c, e, epsilon_0
        energies = np.asarray(self.energies)
        osc = np.asarray(self.osc)
        if energies.shape != osc.shape:
            raise ValueError('energies and osc must have the same shape (npoints, nstates)')
        if energies.shape[1] < self.nstates:
            raise ValueError('nstates exceeds number of columns in input arrays')
        npoints = self.nsamp if self.nsamp is not None else energies.shape[0]

        h_ev_s = physical_constants['Planck constant in eV s'][0]
        kb = physical_constants['Boltzmann constant in eV/K'][0]
        hbar = h_ev_s / (2.0 * np.pi)
        coeff = hbar * 1.0e4 * np.pi * e * e / (2.0 * m_e * epsilon_0 * c * self.ref_index)

        data_min = float(energies[:, :self.nstates].min())
        data_max = float(energies[:, :self.nstates].max())
        ls_min = data_min - 0.5 if self.set_min is None else float(self.set_min)
        ls_max = data_max + 0.5 if self.set_max is None else float(self.set_max)
        if ls_min == ls_max:
            ls_min -= 1.0; ls_max += 1.0

        ener_grid = np.linspace(ls_min, ls_max, int(self.lspoints))
        wl_grid = eV2nm(ener_grid)

        sigma = np.zeros((ener_grid.size, self.nstates), dtype=float)
        err_sigma = np.zeros_like(sigma)

        for i in range(self.nstates):
            Ei = energies[:npoints, i]
            fi = osc[:npoints, i]
            for Eij, fij in zip(Ei, fi):
                sigma[:, i] += fij * Eij * ls_func(self.lshape, ener_grid, self.bro_fac, float(Eij)) / ener_grid
            sigma[:, i] /= npoints
            for Eij, fij in zip(Ei, fi):
                err_sigma[:, i] += (fij * Eij * ls_func(self.lshape, ener_grid, self.bro_fac, float(Eij)) / ener_grid - sigma[:, i])**2
            err_sigma[:, i] = np.sqrt(err_sigma[:, i])

        total_sigma = sigma.sum(axis=1) * coeff
        sigma *= coeff

        if npoints > 9:
            total_err_sigma = err_sigma.sum(axis=1) * coeff / (np.sqrt(npoints) * np.sqrt(npoints - 1))
        else:
            total_err_sigma = np.zeros_like(total_sigma)

        if self.temp != 0.0:
            gamma = 1.0 - np.exp(-ener_grid / (kb * self.temp))
            sigma *= gamma[:, None]
            total_sigma *= gamma
            total_err_sigma *= gamma

        return CrossSectionResult(
            energy_grid_eV=ener_grid,
            lambda_grid_nm=wl_grid,
            total_sigma=total_sigma,
            total_err_sigma=total_err_sigma,
            sigma_per_state=sigma,
        )
# Spectral Overlap helpers

def _make_grid(exp_arr, cal_arr, lam_min, lam_max, dlam, grid_mode="uniform"):
    """Build comparison grid on [lam_min, lam_max]."""
    if grid_mode == "uniform":
        if dlam is None or dlam <= 0:
            raise ValueError("For grid_mode='uniform', provide a positive dlam (nm).")
        lam = np.arange(lam_min, lam_max + dlam * 0.5, dlam)
#    elif grid_mode == "union":
#        wl_e = np.asarray(exp_arr[:, 0])
#        wl_c = np.asarray(cal_arr[:, 0])
#        mask_e = (wl_e >= lam_min) & (wl_e <= lam_max)
#        mask_c = (wl_c >= lam_min) & (wl_c <= lam_max)
#        lam = np.unique(np.concatenate([wl_e[mask_e], wl_c[mask_c]]))
#        if lam.size < 4:  # fallback if too sparse
#            all_wl = np.sort(np.concatenate([wl_e, wl_c]))
#            diffs = np.diff(all_wl)
#            diffs = diffs[diffs > 0]
#            median_spacing = np.median(diffs) if diffs.size else (dlam if dlam else 0.5)
#            if not np.isfinite(median_spacing) or median_spacing <= 0:
#                median_spacing = dlam if (dlam and dlam > 0) else 0.5
#            lam = np.arange(lam_min, lam_max + median_spacing * 0.5, median_spacing)
    else:
        raise ValueError("grid_mode must be 'uniform'")# or 'union'")
    return lam

def _interp_nonneg(arr, lam):
    """Interpolate [wl, I] onto lam, clipping negatives to zero and zeroing outside range."""
    wl = np.asarray(arr[:, 0], dtype=float)
    I = np.asarray(arr[:, 1], dtype=float)
    order = np.argsort(wl)
    wl, I = wl[order], np.maximum(I[order], 0.0)
    y = np.interp(lam, wl, I, left=0.0, right=0.0)
    return y

def _integral(y, x):
    """Numerical integral via trapezoid (NumPy >= 2.0)."""
    return np.trapezoid(y, x)


def _normalise_area(y, x):
    area = _integral(y, x)
    if area > 0:
        return y / area, area
    return y, 0.0

@dataclass
class OverlapJob:
    exp: np.ndarray   # Reference spectrum (can be experimental) - columns: wl_nm, intensity
    calc: np.ndarray  # Simulated spectrum - columns: wl_nm, intensity
    lam_min: float = 300.0
    lam_max: float = 500.0
    dlam: Optional[float] = 0.1
    grid_mode: str = "uniform" 

    def compute(self) -> OverlapResult:
        # Build grid and interpolate
        lam = _make_grid(self.exp, self.calc, self.lam_min, self.lam_max, self.dlam, self.grid_mode)
        y_exp = _interp_nonneg(self.exp, lam)
        y_cal = _interp_nonneg(self.calc, lam)

        # Shape metrics (obtained normalising the are to 1 on [lam_min, lam_max])
        p, area_cal = _normalise_area(y_cal, lam)
        q, area_exp = _normalise_area(y_exp, lam)
        OA_norm = _integral(np.minimum(p, q), lam)

        # Intensity metrics (raw epsilon in the original units)
        OA_raw = _integral(np.minimum(y_cal, y_exp), lam)

        # RMSE_raw and MAE_raw
        diff = y_cal - y_exp
        L = float(lam[-1] - lam[0]) if lam.size >= 2 else np.nan
        RMSE_raw = float(np.sqrt(_integral(diff**2, lam) / L)) if (np.isfinite(L) and L > 0) else float('nan')
        MAE_raw  = float(_integral(np.abs(diff), lam) / L) if (np.isfinite(L) and L > 0) else float('nan')

        # (Optional) best-scale RMSE between raw spectra
        num = _integral(y_cal * y_exp, lam)
        den = _integral(y_exp * y_exp, lam)
        s_opt = (num / den) if den > 0 else 0.0
        RMSE_scaled = float(np.sqrt(_integral((y_cal - s_opt * y_exp) ** 2, lam) / L)) if (np.isfinite(L) and L > 0) else float('nan')

        area_ratio = float(area_cal / area_exp) if area_exp > 0 else float('nan')

        return OverlapResult(
            lambda_grid_nm=lam,
            y_exp_interp=y_exp,
            y_calc_interp=y_cal,
            OA_norm=float(OA_norm),
            OA_raw=float(OA_raw),
            RMSE_raw=float(RMSE_raw),
            MAE_raw=float(MAE_raw),
            area_exp=float(area_exp),
            area_cal=float(area_cal),
            area_ratio=area_ratio,
            s_opt=float(s_opt),
            RMSE_scaled=float(RMSE_scaled),
        )

# Facade + CLI
class SpectraTool:
    @staticmethod
    def run_config(cfg: Dict, verbose: bool = False) -> None:
        """Run from dict/JSON config.
        Top-level keys:
          command: "spectra" | "spec_overlap"
          out_prefix: str
          plot: bool
          spectra: {...}        # params for spectra
          spec_overlap: {...}   # params for overlap

        Compat with Old nomenclature: 
                                     if 'mode' is 1 or 2, map to the respective command.
        """
        cfg = normalize_keys(cfg)
        setup_logging(verbose)

        out_prefix = cfg.get('out_prefix', 'out')
        plot = bool(cfg.get('plot', False))

        command = cfg.get('command')

        if command == 'spectra':
            m1 = cfg.get('spectra', {})
            energies = load_array(m1['energies']) if isinstance(m1.get('energies'), str) else np.asarray(m1['energies'])
            osc = load_array(m1['osc']) if isinstance(m1.get('osc'), str) else np.asarray(m1['osc'])
            job = CrossSectionJob(
                energies=energies,
                osc=osc,
                nstates=int(m1['nstates']),
                nsamp=m1.get('nsamp'),
                temp=float(m1.get('temp', 0.0)),
                ref_index=float(m1.get('ref_index', 1.0)),
                bro_fac=m1.get('bro_fac', 0.05),
                lshape=m1.get('lshape', 'gau'),
                lspoints=int(m1.get('lspoints', 2000)),
                set_min=m1.get('set_min'),
                set_max=m1.get('set_max'),
            )
            res = job.compute()
            res.save(out_prefix, plot=plot,
                     plot_xmin=m1.get('plot_xmin'),
                     plot_xmax=m1.get('plot_xmax')))

        elif command == 'spec_overlap':
            m2 = cfg.get('spec_overlap', {})
            exp = load_array(m2['exp']) if isinstance(m2.get('exp'), str) else np.asarray(m2['exp'])
            calc = load_array(m2['calc']) if isinstance(m2.get('calc'), str) else np.asarray(m2['calc'])
            job = OverlapJob(
                exp=exp[:, :2],
                calc=calc[:, :2],
                lam_min=float(m2.get('lam_min', 300.0)),
                lam_max=float(m2.get('lam_max', 500.0)),
                dlam=float(m2.get('dlam', 0.1)),
                grid_mode=str(m2.get('grid_mode', 'uniform')),
            )
            res = job.compute(); res.save(out_prefix, plot=plot,
                                          plot_xmin=m2.get('plot_xmin'),
                                          plot_xmax=m2.get('plot_xmax')))
        else:
            raise ValueError("Config must include 'command': 'spectra' or 'spec_overlap'")

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='Absorption Spectra Tool: spectra (cross section), spec_overlap (overlap)')

    # Global-only flags
    p.add_argument('--config', help='JSON config file')
    p.add_argument('--verbose', action='store_true', help='Verbose logging')

    # Common flags
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--out-prefix', default='out', help='Prefix path for outputs')
    common.add_argument('--plot', action='store_true', help='Write PNG plots')

    sub = p.add_subparsers(dest='subcmd')

    # spectra (absorption cross section)
    m1 = sub.add_parser('spectra', parents=[common], help='Photoabsorption cross section')
    m1.add_argument('--energies', help='Energies array (CSV/TXT/DAT/NPY/NPZ)')
    m1.add_argument('--osc', help='Oscillator strengths array (CSV/TXT/DAT/NPY/NPZ)')
    m1.add_argument('--nstates', type=int, help='Number of electronic states (columns)')
    m1.add_argument('--nsamp', type=int, help='Number of samples (rows)')
    m1.add_argument('--temp', type=float, default=0.0, help='Temperature (K); 0 disables thermal factor')
    m1.add_argument('--ref-index', dest='ref_index', type=float, default=1.0, help='Refractive index')
    m1.add_argument('--bro-fac', dest='bro_fac', type=float, default=0.05, help='Broadening factor (eV)')
    m1.add_argument('--lshape', choices=['gau', 'lor', 'voi'], default='gau', help='Line shape')
    m1.add_argument('--lspoints', type=int, default=2000, help='Lineshape grid points')
    m1.add_argument('--set-min', type=float, help='Energy min (eV)')
    m1.add_argument('--set-max', type=float, help='Energy max (eV)')
    m1.add_argument('--plot-xmin', type=float, help='Minimum wavelength (nm) for plotting (x-axis only)')
    m1.add_argument('--plot-xmax', type=float, help='Maximum wavelength (nm) for plotting (x-axis only)')

    # spec_overlap (spectra overlap and metrics)
    m2 = sub.add_parser('spec_overlap', parents=[common], help='Spectral overlap metrics')
    m2.add_argument('--exp', help='Experimental spectrum (CSV/TXT/DAT/NPY/NPZ): wl_nm,intensity')
    m2.add_argument('--calc', help='Calculated spectrum (CSV/TXT/DAT/NPY/NPZ): wl_nm,intensity')
    m2.add_argument('--lam-min', type=float, default=300.0, help='Wavelength min (nm) or energy min (eV)')
    m2.add_argument('--lam-max', type=float, default=500.0, help='Wavelength max (nm) or energy max (eV)')
    m2.add_argument('--dlam', type=float, default=0.1, help='Grid spacing (nm or eV)')
    m2.add_argument('--grid-mode', choices=['uniform', 'union'], default='uniform',
                    help="Grid construction: 'uniform' uses --dlam; 'union' uses union of input grids")
    m2.add_argument('--plot-xmin', type=float, help='Minimum wavelength (nm) for overlap plot (x-axis only)')
    m2.add_argument('--plot-xmax', type=float, help='Maximum wavelength (nm) for overlap plot (x-axis only)')

    return p

def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.config:
        with open(args.config, 'r') as f:
            cfg = json.load(f)
        cfg = normalize_keys(cfg)
        SpectraTool.run_config(cfg, verbose=args.verbose)
        return 0

    setup_logging(args.verbose)

    if args.subcmd == 'spectra':
        missing = [x for x in (args.energies, args.osc, args.nstates) if x is None]
        if missing:
            parser.error("'spectra' requires --energies, --osc, and --nstates (or provide them in --config)")
        energies = load_array(args.energies)
        osc = load_array(args.osc)
        job = CrossSectionJob(
            energies=energies,
            osc=osc,
            nstates=int(args.nstates),
            nsamp=args.nsamp,
            temp=float(args.temp),
            ref_index=float(args.ref_index),
            bro_fac=float(args.bro_fac),
            lshape=args.lshape,
            lspoints=int(args.lspoints),
            set_min=args.set_min,
            set_max=args.set_max,
        )
        res = job.compute()
        res.save(args.out_prefix, plot=args.plot, plot_xmin=args.plot_xmin, plot_xmax=args.plot_xmax)
        return 0

    elif args.subcmd == 'spec_overlap':
        missing = [x for x in (args.exp, args.calc) if x is None]
        if missing:
            parser.error("'spec_overlap' requires --exp and --calc (or provide them in --config)")
        exp = load_array(args.exp)
        calc = load_array(args.calc)
        job = OverlapJob(
            exp=exp[:, :2],
            calc=calc[:, :2],
            lam_min=float(args.lam_min),
            lam_max=float(args.lam_max),
            dlam=float(args.dlam),
            grid_mode=str(args.grid_mode),
        )
        res = job.compute() 
        res.save(args.out_prefix, plot=args.plot,
                 plot_xmin=args.plot_xmin,
                 plot_xmax=args.plot_xmax)
        return 0

    else:
        parser.print_help(); return 1

if __name__ == '__main__':
    sys.exit(main())
