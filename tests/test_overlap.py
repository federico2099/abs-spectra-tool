import pytest
import numpy as np
from spectra_tool import OverlapJob

def gaussian_spectrum(center, width, lam_min=400.0, lam_max=600.0, step=0.5):
    lam = np.arange(lam_min, lam_max + step/2, step)
    I = np.exp(-0.5 * ((lam - center)/width)**2)
    return np.column_stack([lam, I])

def test_overlap_identical_spectra():
    exp = gaussian_spectrum(500.0, 10.0)
    calc = exp.copy()
    job = OverlapJob(exp=exp, calc=calc, lam_min=420, lam_max=560, dlam=0.5)
    res = job.compute()
    assert res.OA == pytest.approx(1.0, rel=1e-3, abs=1e-3)
    assert res.rmse_scaled == pytest.approx(0.0, abs=1e-8)
    assert res.s_opt == pytest.approx(1.0, rel=1e-3)

def test_overlap_shifted_spectra():
    exp = gaussian_spectrum(500.0, 10.0)
    calc = gaussian_spectrum(520.0, 10.0)
    job = OverlapJob(exp=exp, calc=calc, lam_min=420, lam_max=560, dlam=0.5)
    res = job.compute()
    assert 0.0 < res.OA < 1.0
    assert res.rmse_scaled > 0.0
