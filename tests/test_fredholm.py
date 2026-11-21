"""Tests for Section V: Fredholm-module compression and Schatten acceleration."""

import math
import numpy as np

from primorial_feedback_bridge.fredholm import (
    fredholm_commutator_blocks,
    fredholm_defect_asymptotic,
    fredholm_iterate_diagnostics,
    iterate_fredholm_eigs,
)


def test_doubly_exponential_schatten_exponents() -> None:
    tau = 0.1
    lam = np.array([20.0, 40.0])
    for n, expected_slope in ((0, 2.0), (1, 4.0), (2, 8.0)):
        fn = iterate_fredholm_eigs(lam, tau=tau, n=n)
        defect = 1.0 - fn * fn
        measured_slope = math.log(defect[0] / defect[1]) / math.log(lam[1] / lam[0])
        assert abs(measured_slope - expected_slope) < 0.05

        asymp = fredholm_defect_asymptotic(lam, tau=tau, n=n)
        assert np.allclose(defect, asymp, rtol=0.05)


def test_fredholm_defect_convergence_and_commutator_compression() -> None:
    diag = fredholm_iterate_diagnostics(k=3, tau=0.1, eps0=0.35, m_max=150, n_iters=10)
    assert diag["exact_defect_rank"] == 42
    _, op_err_10, s1_err_10 = diag["history"][-1]
    assert op_err_10 < 1e-12
    assert s1_err_10 < 1e-12

    rng = np.random.default_rng(7)
    m_mat = rng.standard_normal((30, 30))
    c0 = fredholm_commutator_blocks(k=3, r=2, m_mat=m_mat, tau=0.1, eps0=0.35, m_max=80, n_iters=0)
    c5 = fredholm_commutator_blocks(k=3, r=2, m_mat=m_mat, tau=0.1, eps0=0.35, m_max=80, n_iters=5)
    c_inf = fredholm_commutator_blocks(
        k=3, r=2, m_mat=m_mat, tau=0.1, eps0=0.35, m_max=80, n_iters=None
    )

    assert c5["numerical_rank"] < c0["numerical_rank"]
    assert c_inf["numerical_rank"] <= c5["numerical_rank"]
    assert c_inf["numerical_rank"] == 102
