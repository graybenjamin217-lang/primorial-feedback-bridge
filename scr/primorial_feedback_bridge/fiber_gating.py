"""Section III: Prime-channel fixed points, Fréchet derivative, and bandpass gating."""

from __future__ import annotations

import numpy as np

from .feedback import X_MAX, omega_g_odd_matrix
from .primorial import (
    apply_unit_intertwining,
    build_T_herm,
    build_principal_block_kronecker,
    unit_group_elements,
)


def daleckii_krein_prime_channel(
    c_p: np.ndarray, h: np.ndarray, tol: float = 1e-10
) -> np.ndarray:
    """Exact Fréchet derivative D Omega_G^odd(C_tilde_p)[H] = H - sum_s P_s H P_s."""
    eigvals, eigvecs = np.linalg.eigh(c_p)
    p_plus = eigvecs[:, eigvals > 1.0 - tol]
    p_minus = eigvecs[:, eigvals < -1.0 + tol]
    p_zero = eigvecs[:, np.abs(eigvals) < tol]

    diag_blocks = np.zeros_like(h, dtype=np.float64)
    for basis in (p_plus, p_zero, p_minus):
        if basis.shape[1] > 0:
            proj = basis @ basis.T
            diag_blocks += proj @ h @ proj
    return h - diag_blocks


def finite_diff_frechet(
    a: np.ndarray, h: np.ndarray, eps: float = 1e-5
) -> np.ndarray:
    """Central finite-difference approximation to D Omega_G^odd(A)[H]."""
    return (omega_g_odd_matrix(a + eps * h) - omega_g_odd_matrix(a - eps * h)) / (
        2.0 * eps
    )


def partition_spectrum_basins(
    eigs: np.ndarray, active_tol: float = 1e-10
) -> dict[str, int | float]:
    """Partition active eigenvalues into Omega_G^odd dynamical basins."""
    active = eigs[np.abs(eigs) > active_tol]
    absorbed = int(np.sum(np.abs(active) < 0.5))
    basin_pos = int(np.sum((active > 0.5) & (active < X_MAX)))
    basin_neg = int(np.sum((active < -0.5) & (active > -X_MAX)))
    ejected_pos = int(np.sum(active > X_MAX))
    ejected_neg = int(np.sum(active < -X_MAX))

    boundaries = np.array([-X_MAX, -0.5, 0.5, X_MAX], dtype=np.float64)
    min_dist = (
        float(np.min(np.abs(active[:, None] - boundaries[None, :])))
        if active.size > 0
        else float("inf")
    )

    return {
        "active_rank": int(active.size),
        "absorbed": absorbed,
        "basin_pos": basin_pos,
        "basin_neg": basin_neg,
        "ejected_pos": ejected_pos,
        "ejected_neg": ejected_neg,
        "ejected_total": ejected_pos + ejected_neg,
        "trace_limit": basin_pos - basin_neg,
        "min_dist_boundary": min_dist,
    }


def fiber_bandpass_diagnostics(
    k: int, max_units_to_check: int = 8
) -> dict[str, int | float]:
    """Compute Table 1 diagnostics for T_k^Herm under Omega_G^odd."""
    t_herm, nk, _ = build_T_herm(k)
    eigs = np.linalg.eigvalsh(t_herm)
    stats = partition_spectrum_basins(eigs)

    t_once = omega_g_odd_matrix(t_herm)
    units = unit_group_elements(k)
    # Prefer non-involutory units when available (g^2 != 1 mod N_k).
    non_involutions = [g for g in units if (g * g) % nk != 1]
    sample_units = (non_involutions or units)[:max_units_to_check]

    max_err = 0.0
    for g in sample_units:
        intertwined = apply_unit_intertwining(t_once, g, nk)
        err = float(np.max(np.abs(intertwined - t_once)))
        if err > max_err:
            max_err = err

    stats["k"] = k
    stats["nk"] = nk
    stats["max_ug_intertwine_err"] = max_err
    return stats


def principal_bandpass_diagnostics(k: int) -> dict[str, int | float]:
    """Compute Table 2 basin diagnostics on the 2^k-dimensional block V_{chi_0}."""
    t_block = build_principal_block_kronecker(k)
    eigs = np.linalg.eigvalsh(t_block)
    stats = partition_spectrum_basins(eigs)
    stats["k"] = k
    stats["dim_v_chi0"] = 1 << k
    stats["surviving_pct"] = (
        100.0 * (stats["basin_pos"] + stats["basin_neg"]) / stats["active_rank"]
    )
    return stats
