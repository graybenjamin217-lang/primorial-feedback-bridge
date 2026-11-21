"""Variable-exponent feedback operators Omega_G and Omega_G^odd."""

from __future__ import annotations

import math
import numpy as np


def f_pos(x: np.ndarray | float) -> np.ndarray:
    """Evaluate f(x) = x^(1/x - 1) for x > 0 and 0 for x <= 0."""
    arr = np.asarray(x, dtype=np.float64)
    out = np.zeros_like(arr, dtype=np.float64)
    # Below 1e-3, ((1 - x) / x) * ln(x) < -6900, which underflows cleanly to 0 in float64.
    mask = arr > 1e-3
    xm = arr[mask]
    out[mask] = np.exp(((1.0 - xm) / xm) * np.log(xm))
    return out


def f_odd(x: np.ndarray | float) -> np.ndarray:
    """Evaluate the C^infty odd extension f_odd(x) = sign(x) * f(|x|)."""
    arr = np.asarray(x, dtype=np.float64)
    return np.sign(arr) * f_pos(np.abs(arr))


def f_prime(x: float) -> float:
    """First derivative f'(x) = f(x) * (1 - x - ln(x)) / x^2 for x > 0."""
    if x <= 1e-3:
        return 0.0
    fx = float(f_pos(x))
    return fx * (1.0 - x - math.log(x)) / (x * x)


def f_double_prime(x: float) -> float:
    """Second derivative f''(x) for x > 0."""
    if x <= 1e-3:
        return 0.0
    fx = float(f_pos(x))
    lnx = math.log(x)
    term1 = ((1.0 - x - lnx) / (x * x)) ** 2
    term2 = (x - 3.0 + 2.0 * lnx) / (x**3)
    return fx * (term1 + term2)


def compute_x_max(tol: float = 1e-15, max_iter: int = 50) -> float:
    """Compute the unique root x_max > 1 of f(x) = 1/2 via Newton's method."""
    x = 2.88747485
    log2 = math.log(2.0)
    for _ in range(max_iter):
        g = ((1.0 - x) / x) * math.log(x) + log2
        dg = (1.0 - x - math.log(x)) / (x * x)
        step = g / dg
        x -= step
        if abs(step) < tol:
            break
    return x


def compute_x_crit(tol: float = 1e-15, max_iter: int = 50) -> float:
    """Compute the unique root x_crit in (0, 1) of f'(x) = 1 via Newton's method."""
    x = 0.72582784
    for _ in range(max_iter):
        g = f_prime(x) - 1.0
        dg = f_double_prime(x)
        step = g / dg
        x -= step
        if abs(step) < tol:
            break
    return x


X_MAX: float = compute_x_max()
X_CRIT: float = compute_x_crit()
DELTA: float = 1.0 - X_CRIT
C_DELTA: float = 0.5 * abs(f_double_prime(X_CRIT))
K_DELTA: float = (1.0 - float(f_pos(X_CRIT))) / (DELTA * DELTA)


def omega_g_matrix(a: np.ndarray) -> np.ndarray:
    """Apply positive functional-calculus operator Omega_G to a positive Hermitian matrix."""
    a_sym = 0.5 * (a + a.conj().T)
    eigvals, eigvecs = np.linalg.eigh(a_sym)
    mapped = f_pos(np.maximum(eigvals, 0.0))
    return (eigvecs * mapped[None, :]) @ eigvecs.conj().T


def omega_g_odd_matrix(a: np.ndarray) -> np.ndarray:
    """Apply odd functional-calculus operator Omega_G^odd to a Hermitian matrix."""
    a_sym = 0.5 * (a + a.conj().T)
    eigvals, eigvecs = np.linalg.eigh(a_sym)
    mapped = f_odd(eigvals)
    return (eigvecs * mapped[None, :]) @ eigvecs.conj().T


def iterate_omega_g_odd(a: np.ndarray, n: int) -> np.ndarray:
    """Compute (Omega_G^odd)^n(A) via a single spectral decomposition."""
    if n < 0:
        raise ValueError("Iteration count n must be non-negative.")
    a_sym = 0.5 * (a + a.conj().T)
    if n == 0:
        return a_sym.copy()
    eigvals, eigvecs = np.linalg.eigh(a_sym)
    mapped = eigvals.copy()
    for _ in range(n):
        mapped = f_odd(mapped)
    return (eigvecs * mapped[None, :]) @ eigvecs.conj().T
