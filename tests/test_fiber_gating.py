"""Tests for Section III: Prime-channel fixed points and bandpass gating."""

import numpy as np

from primorial_feedback_bridge.feedback import (
    X_MAX,
    iterate_omega_g_odd,
    omega_g_odd_matrix,
)
from primorial_feedback_bridge.fiber_gating import (
    daleckii_krein_prime_channel,
    fiber_bandpass_diagnostics,
    finite_diff_frechet,
    principal_bandpass_diagnostics,
)
from primorial_feedback_bridge.primorial import (
    build_T_herm,
    build_prime_channel,
    build_principal_block_kronecker,
    primorial,
    principal_subspace_basis,
)


def test_prime_channels_are_exact_fixed_points() -> None:
    for k in (2, 3, 4):
        _, primes = primorial(k)
        for p in primes:
            c_p = build_prime_channel(k, p)
            err = np.max(np.abs(omega_g_odd_matrix(c_p) - c_p))
            assert err < 1e-14


def test_daleckii_krein_derivative_and_quadratic_suppression() -> None:
    rng = np.random.default_rng(42)
    c_3 = build_prime_channel(k=3, p=3)
    raw = rng.standard_normal(c_3.shape)
    h_sym = 0.5 * (raw + raw.T)
    h_sym /= np.linalg.norm(h_sym, ord=2)

    exact_df = daleckii_krein_prime_channel(c_3, h_sym)
    fd_df = finite_diff_frechet(c_3, h_sym, eps=1e-5)
    assert np.max(np.abs(exact_df - fd_df)) < 1e-10

    # Project h_sym onto the commutant of C_tilde_3: D Omega_G^odd(C_3)[H_comm] = 0.
    h_comm = h_sym - exact_df
    assert np.max(np.abs(daleckii_krein_prime_channel(c_3, h_comm))) < 1e-13

    err_eps = np.linalg.norm(omega_g_odd_matrix(c_3 + 1e-3 * h_comm) - c_3, ord=2)
    err_half = np.linalg.norm(omega_g_odd_matrix(c_3 + 5e-4 * h_comm) - c_3, ord=2)
    ratio = err_eps / err_half
    assert abs(ratio - 4.0) < 0.05


def test_principal_kronecker_matches_dense_projection() -> None:
    for k in (1, 2, 3, 4):
        t_herm, _, _ = build_T_herm(k)
        basis = principal_subspace_basis(k)
        dense_proj = basis.T @ t_herm @ basis
        kron_block = build_principal_block_kronecker(k)
        assert np.max(np.abs(dense_proj - kron_block)) < 1e-14


def test_table_1_and_bandpass_convergence() -> None:
    expected = {
        1: (1, 0, 1, 0, 0, 1),
        2: (3, 0, 1, 2, 0, -1),
        3: (12, 2, 5, 5, 0, 0),
        4: (82, 20, 31, 30, 1, 1),
    }
    for k, (act, ab, bp, bn, ej, tr) in expected.items():
        stats = fiber_bandpass_diagnostics(k)
        assert stats["active_rank"] == act
        assert stats["absorbed"] == ab
        assert stats["basin_pos"] == bp
        assert stats["basin_neg"] == bn
        assert stats["ejected_total"] == ej
        assert stats["trace_limit"] == tr
        assert stats["max_ug_intertwine_err"] < 1e-14

    # Verify convergence to partial symmetry S_k^3 = S_k at k = 3.
    t3, _, _ = build_T_herm(3)
    s3 = iterate_omega_g_odd(t3, n=15)
    assert np.max(np.abs(s3 @ s3 @ s3 - s3)) < 1e-12
    assert abs(X_MAX - 2.88747485) < 1e-7


def test_table_2_principal_block_counts() -> None:
    s5 = principal_bandpass_diagnostics(5)
    assert s5["active_rank"] == 31
    assert s5["absorbed"] == 4
    assert s5["basin_pos"] == 11
    assert s5["basin_neg"] == 14
    assert s5["ejected_pos"] == 1
    assert s5["ejected_neg"] == 1
    assert s5["trace_limit"] == -3
