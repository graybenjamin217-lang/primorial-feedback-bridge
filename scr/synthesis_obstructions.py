"""Section VI: Prime-channel synthesis, V_{chi_0} rigidity, and 1D gauge obstruction."""

from __future__ import annotations

import math
from typing import Callable
import numpy as np

from .primorial import (
    build_T_herm,
    build_prime_channel,
    principal_subspace_basis,
    primorial,
)


def channel_exp_closed_form(c_p: np.ndarray, alpha: float) -> np.ndarray:
    """Closed-form exp(alpha * C_tilde_p) = I + sinh(a) C_p + (cosh(a) - 1) C_p^2."""
    eye = np.eye(c_p.shape[0], dtype=np.float64)
    return eye + math.sinh(alpha) * c_p + (math.cosh(alpha) - 1.0) * (c_p @ c_p)


def commutator(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Matrix commutator [A, B] = AB - BA."""
    return a @ b - b @ a


def bch_fourth_order_approx(
    c_p: np.ndarray, c_q: np.ndarray, alpha: float
) -> np.ndarray:
    """Proposition 6.1(2) fourth-order expansion of Omega_G(e^(X/2) e^Y e^(X/2))."""
    x_hat = alpha * c_p
    y_hat = alpha * c_q
    s_hat = x_hat + y_hat
    xy = commutator(x_hat, y_hat)
    c3 = (-1.0 / 24.0) * commutator(x_hat, xy) - (1.0 / 12.0) * commutator(y_hat, xy)

    eye = np.eye(c_p.shape[0], dtype=np.float64)
    s2 = s_hat @ s_hat
    s3 = s2 @ s_hat
    s4 = s2 @ s2
    return eye - s2 + 0.5 * s3 + (1.0 / 3.0) * s4 - (s_hat @ c3 + c3 @ s_hat)


def symmetric_multiprime_product(
    channels: list[np.ndarray], alpha: float
) -> np.ndarray:
    """Construct A_sym(alpha) = E_{p_1} ... E_{p_k}^2 ... E_{p_1} with E_p = exp((a/2) C_p)."""
    n = channels[0].shape[0]
    half = np.eye(n, dtype=np.float64)
    for c_p in channels:
        half = half @ channel_exp_closed_form(c_p, 0.5 * alpha)
    return half @ half.T


def principal_subspace_leak(mat: np.ndarray, k: int) -> float:
    """Compute ||(I - P_{chi_0}) M P_{chi_0}||_2 measuring leakage out of V_{chi_0}."""
    basis = principal_subspace_basis(k)
    m_on_v = mat @ basis
    projected = basis @ (basis.T @ m_on_v)
    return float(np.linalg.norm(m_on_v - projected, ord=2))


def bulk_spacing_ratio(
    eigs: np.ndarray, active_tol: float = 1e-10, bulk_fraction: float = 0.6
) -> float:
    """Compute unfolding-free consecutive level-spacing ratio <r>_bulk."""
    active = np.sort(eigs[np.abs(eigs) > active_tol])
    n = active.size
    if n < 4:
        return float("nan")
    trim = int(round(0.5 * (1.0 - bulk_fraction) * n))
    bulk = active[trim : n - trim] if trim > 0 else active
    gaps = np.diff(bulk)
    if gaps.size < 2:
        return float("nan")
    s1 = gaps[:-1]
    s2 = gaps[1:]
    denom = np.maximum(s1, s2)
    valid = denom > 1e-15
    ratios = np.minimum(s1[valid], s2[valid]) / denom[valid]
    return float(np.mean(ratios))


def gauge_equivalence_check(
    k: int,
    eps_func: Callable[[np.ndarray], np.ndarray],
    m_max: int = 40,
    n_grid: int = 4096,
) -> dict[str, float]:
    """Compare truncated Fourier spectrum of D_k[eps_0(y)] against D_k[eps_bar]."""
    t_herm, nk, _ = build_T_herm(k)
    l_k = math.log(nk)
    mu = np.linalg.eigvalsh(t_herm)

    y = (np.arange(n_grid, dtype=np.float64) + 0.5) * (l_k / n_grid)
    coeffs = np.fft.fft(eps_func(y)) / n_grid
    eps_bar = float(np.real(coeffs[0]))

    m_vals = np.arange(-m_max, m_max + 1, dtype=np.int64)
    size = m_vals.size
    diff_idx = (m_vals[:, None] - m_vals[None, :]) % n_grid
    toeplitz_eps = coeffs[diff_idx]
    diag_momentum = np.diag((2.0 * np.pi / l_k) * m_vals)

    # Compare the inner half of modes to exclude Fourier-cutoff boundary modes.
    inner_lo = size // 4
    inner_hi = size - inner_lo
    max_spec_diff = 0.0

    for mu_a in mu:
        h_var = diag_momentum + mu_a * toeplitz_eps
        eigs_var = np.linalg.eigvalsh(h_var)
        eigs_const = np.sort((2.0 * np.pi / l_k) * m_vals + eps_bar * mu_a)
        err = float(
            np.max(np.abs(eigs_var[inner_lo:inner_hi] - eigs_const[inner_lo:inner_hi]))
        )
        if err > max_spec_diff:
            max_spec_diff = err

    return {
        "eps_bar": eps_bar,
        "max_bulk_spectral_diff": max_spec_diff,
    }


def _hermitian_expm_imag(h_sym: np.ndarray) -> np.ndarray:
    """Compute exp(i * H) for a real-symmetric matrix H via eigendecomposition."""
    w, v = np.linalg.eigh(h_sym)
    return (v * np.exp(1j * w)[None, :]) @ v.T


def non_abelian_wilson_loop(
    k: int,
    eps_funcs: dict[int, Callable[[float], float]],
    n_steps: int = 300,
) -> dict[str, float]:
    """Compute path-ordered non-Abelian Wilson loop W_NA vs Abelianized W_Ab."""
    nk, primes = primorial(k)
    l_k = math.log(nk)
    dy = l_k / n_steps
    channels = {p: build_prime_channel(k, p) for p in primes}

    w_na = np.eye(nk, dtype=np.complex128)
    integrated_conn = np.zeros((nk, nk), dtype=np.float64)

    for step in range(n_steps):
        y_mid = (step + 0.5) * dy
        a_y = np.zeros((nk, nk), dtype=np.float64)
        for p in primes:
            a_y += eps_funcs[p](y_mid) * channels[p]
        integrated_conn += dy * a_y
        w_na = _hermitian_expm_imag(dy * a_y) @ w_na

    w_ab = _hermitian_expm_imag(integrated_conn)
    leak = principal_subspace_leak(w_na, k)
    diff_norm = float(np.linalg.norm(w_na - w_ab, ord=2))

    return {
        "w_na_minus_w_ab_norm": diff_norm,
        "v_chi0_leak": leak,
    }
