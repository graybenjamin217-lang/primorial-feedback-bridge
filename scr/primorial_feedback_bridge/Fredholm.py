"""Section V: Dynamical Fredholm-module compression of (A_k, H_k, D_k)."""

from __future__ import annotations

import math
import numpy as np

from .feedback import f_odd
from .primorial import build_T_herm, hybrid_eigenvalues


def odd_bounded_transform_eigs(lam: np.ndarray, tau: float) -> np.ndarray:
    """Eigenvalues of F_tau(D_k) = sqrt(tau) * D_k * (I + tau * D_k^2)^(-1/2)."""
    return (math.sqrt(tau) * lam) / np.sqrt(1.0 + tau * (lam * lam))


def iterate_fredholm_eigs(lam: np.ndarray, tau: float, n: int) -> np.ndarray:
    """Evaluate F_n(lam) = (f_odd)^n(F_tau(lam))."""
    vals = odd_bounded_transform_eigs(lam, tau)
    for _ in range(n):
        vals = f_odd(vals)
    return vals


def fredholm_defect_asymptotic(lam: np.ndarray, tau: float, n: int) -> np.ndarray:
    """Leading asymptotic 2 * (2*tau)^(-2^n) * |lam|^(-2^(n+1)) (Theorem 5.1(1))."""
    power_two = 1 << n
    return 2.0 * ((2.0 * tau) ** (-power_two)) * (np.abs(lam) ** (-2 * power_two))


def fredholm_iterate_diagnostics(
    k: int,
    tau: float = 0.1,
    eps0: float = 0.35,
    m_max: int = 150,
    n_iters: int = 10,
) -> dict[str, object]:
    """Track operator-norm convergence F_n -> F_inf and S_1 defect convergence."""
    lam, _ = hybrid_eigenvalues(k, eps0=eps0, m_max=m_max)
    odd_cutoff = (3.0 * tau) ** (-0.5)

    f_inf = np.where(lam > odd_cutoff, 1.0, np.where(lam < -odd_cutoff, -1.0, 0.0))
    pi_def = (np.abs(lam) <= odd_cutoff).astype(np.float64)

    fn = odd_bounded_transform_eigs(lam, tau)
    history: list[tuple[int, float, float]] = []
    for n in range(n_iters + 1):
        op_err = float(np.max(np.abs(fn - f_inf)))
        s1_def_err = float(np.sum(np.abs((1.0 - fn * fn) - pi_def)))
        history.append((n, op_err, s1_def_err))
        if n < n_iters:
            fn = f_odd(fn)

    return {
        "k": k,
        "odd_cutoff": odd_cutoff,
        "exact_defect_rank": int(np.sum(pi_def)),
        "history": history,
    }


def fredholm_commutator_blocks(
    k: int,
    r: int,
    m_mat: np.ndarray,
    tau: float = 0.1,
    eps0: float = 0.35,
    m_max: int = 80,
    n_iters: int | None = None,
    rank_tol: float = 1e-10,
) -> dict[str, float | int]:
    """Compute S_1 norm and numerical rank of [F_n, e_r x M] via Fourier blocks.

    Pass n_iters=None to evaluate the exact limiting Fredholm module operator F_{inf,tau}.
    """
    t_herm, nk, _ = build_T_herm(k)
    if m_mat.shape != (nk, nk):
        raise ValueError(f"Expected fiber operator of shape ({nk}, {nk}).")

    mu, eigvecs = np.linalg.eigh(t_herm)
    m_tilde = eigvecs.T @ m_mat @ eigvecs
    l_k = math.log(nk)
    odd_cutoff = (3.0 * tau) ** (-0.5)

    def eval_symbol(lam_vec: np.ndarray) -> np.ndarray:
        if n_iters is None:
            return np.where(
                lam_vec > odd_cutoff, 1.0, np.where(lam_vec < -odd_cutoff, -1.0, 0.0)
            )
        return iterate_fredholm_eigs(lam_vec, tau=tau, n=n_iters)

    total_s1 = 0.0
    total_rank = 0
    for m in range(-m_max, m_max + 1):
        lam_m = (2.0 * np.pi * m / l_k) + eps0 * mu
        lam_mr = (2.0 * np.pi * (m + r) / l_k) + eps0 * mu
        diff = eval_symbol(lam_mr)[:, None] - eval_symbol(lam_m)[None, :]
        block = diff * m_tilde
        svals = np.linalg.svd(block, compute_uv=False)
        total_s1 += float(np.sum(svals))
        total_rank += int(np.sum(svals > rank_tol))

    return {
        "k": k,
        "r": r,
        "n_iters": -1 if n_iters is None else n_iters,
        "s1_norm": total_s1,
        "numerical_rank": total_rank,
    }
