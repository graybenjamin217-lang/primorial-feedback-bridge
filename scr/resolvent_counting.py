"""Section IV: Super-Gaussian resolvent lifting and trace-norm spectral counting."""

from __future__ import annotations

import numpy as np

from .feedback import f_pos
from .primorial import hybrid_eigenvalues, primorial


def positive_resolvent_eigs(lam: np.ndarray, tau: float) -> np.ndarray:
    """Eigenvalues of R_tau(D_k) = (I + tau * D_k^2)^(-1)."""
    return 1.0 / (1.0 + tau * (lam * lam))


def super_gaussian_lift_eigs(lam: np.ndarray, tau: float) -> np.ndarray:
    """Closed-form eigenvalues exp(-tau * lam^2 * ln(1 + tau * lam^2)) (Theorem 4.1)."""
    u = tau * (lam * lam)
    return np.exp(-u * np.log1p(u))


def heat_kernel_eigs(lam: np.ndarray, tau: float) -> np.ndarray:
    """Eigenvalues of the heat kernel exp(-tau * D_k^2)."""
    return np.exp(-tau * (lam * lam))


def trace_norm_spectral_counter(
    k: int,
    tau: float = 0.1,
    eps0: float = 0.35,
    m_max: int = 150,
    n_iters: int = 14,
    tail_m_cutoff: int = 10,
) -> dict[str, object]:
    """Iterate Omega_G on R_tau(D_k) and record S_1 errors and tail suppression."""
    nk, _ = primorial(k)
    lam, m_idx = hybrid_eigenvalues(k, eps0=eps0, m_max=m_max)
    cutoff = tau ** (-0.5)
    target = (np.abs(lam) < cutoff).astype(np.float64)
    min_gap = float(np.min(np.abs(np.abs(lam) - cutoff)))

    r_eigs = positive_resolvent_eigs(lam, tau)
    sg_eigs = super_gaussian_lift_eigs(lam, tau)
    heat_eigs = heat_kernel_eigs(lam, tau)

    tail_mask = np.abs(m_idx) > tail_m_cutoff
    xk = r_eigs.copy()
    history: list[tuple[int, float, float]] = []
    iters_to_1e12: int | None = None

    for n in range(1, n_iters + 1):
        xk = f_pos(xk)
        trace_est = float(np.sum(xk))
        s1_err = float(np.sum(np.abs(xk - target)))
        history.append((n, trace_est, s1_err))
        if iters_to_1e12 is None and s1_err < 1e-12:
            iters_to_1e12 = n

    return {
        "k": k,
        "nk": nk,
        "cutoff": cutoff,
        "exact_count": int(np.sum(target)),
        "min_gap": min_gap,
        "lift_identity_err": float(np.max(np.abs(f_pos(r_eigs) - sg_eigs))),
        "iters_to_1e12": iters_to_1e12,
        "tail_raw_resolvent": float(np.sum(r_eigs[tail_mask])),
        "tail_heat_kernel": float(np.sum(heat_eigs[tail_mask])),
        "tail_super_gaussian": float(np.sum(sg_eigs[tail_mask])),
        "history": history,
    }
