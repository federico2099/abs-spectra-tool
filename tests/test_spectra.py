import numpy as np
from spectra_tool import CrossSectionJob

def test_cross_section_shapes_and_positivity():
    # small synthetic dataset: 3 samples x 2 states
    energies = np.array([[2.0, 3.0],
                         [2.05, 2.95],
                         [1.95, 3.05]])  # eV
    osc = np.array([[0.5, 0.3],
                    [0.55, 0.28],
                    [0.52, 0.31]])
    job = CrossSectionJob(
        energies=energies,
        osc=osc,
        nstates=2,
        nsamp=3,
        temp=0.0,
        ref_index=1.0,
        bro_fac=0.05,
        lshape='gau',
        lspoints=500,
        set_min=1.0,
        set_max=4.0,
    )
    res = job.compute()
    assert res.energy_grid_eV.shape == (500,)
    assert res.lambda_grid_nm.shape == (500,)
    assert res.sigma_per_state.shape == (500, 2)
    assert res.total_sigma.shape == (500,)
    # non-negative and finite
    assert np.all(np.isfinite(res.total_sigma))
    assert np.all(res.total_sigma >= 0.0)
