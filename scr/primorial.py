"""Primorial-modulated dilation fiber matrices and exact k-qubit reductions."""

from __future__ import annotations

import math
import numpy as np

PRIMES: tuple[int, ...] = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)


def primorial(k: int) -> tuple[int, tuple[int, ...]]:
    """Return the k-th primorial N_k and its prime factors (p_1, ..., p_k)."""
    if not 1 <= k <= len(PRIMES):
        raise ValueError(f"Stage k must satisfy 1 <= k <= {len(PRIMES)}.")
    chosen = PRIMES[:k]
    return math.prod(chosen), chosen


def euler_totient_primorial(k: int) -> int:
    """Return Euler's totient phi(N_k) = prod_{j=1}^k (p_j - 1)."""
    _, primes = primorial(k)
    return math.prod(p - 1 for p in primes)


def build_prime_channel(k: int, p: int) -> np.ndarray:
    """Build normalized prime channel matrix C_tilde_p of shape (N_k, N_k)."""
    nk, primes = primorial(k)
    if p not in primes:
        raise ValueError(f"Prime {p} does not divide N_{k} = {nk}.")
    idx = np.arange(nk, dtype=np.int64)
    ab = (np.outer(idx, idx) * p) % nk
    return np.cos((2.0 * np.pi / nk) * ab) / math.sqrt(p * nk)


def build_T_herm(k: int) -> tuple[np.ndarray, int, tuple[int, ...]]:
    """Build the real-symmetric phase matrix T_k^Herm = sum_{p | N_k} C_tilde_p."""
    nk, primes = primorial(k)
    idx = np.arange(nk, dtype=np.int64)
    ab = np.outer(idx, idx) % nk
    t_mat = np.zeros((nk, nk), dtype=np.float64)
    for p in primes:
        t_mat += (1.0 / math.sqrt(p)) * np.cos((2.0 * np.pi * ((ab * p) % nk)) / nk)
    return t_mat / math.sqrt(nk), nk, primes


def build_principal_channel_kronecker(k: int, prime_index: int) -> np.ndarray:
    """Exact 2^k x 2^k Kronecker matrix of C_tilde_{p_j} restricted to V_{chi_0}."""
    _, primes = primorial(k)
    if not 0 <= prime_index < k:
        raise ValueError(f"prime_index must satisfy 0 <= prime_index < {k}.")
    mat = np.array([[1.0]], dtype=np.float64)
    for i, p in enumerate(primes):
        s = math.sqrt(p - 1.0)
        if i == prime_index:
            local = np.array([[p - 1.0, s], [s, 1.0]], dtype=np.float64) / p
        else:
            local = np.array([[-1.0, s], [s, 1.0]], dtype=np.float64) / math.sqrt(p)
        mat = np.kron(mat, local)
    return mat


def build_principal_block_kronecker(k: int) -> np.ndarray:
    """Exact 2^k x 2^k matrix of T_k^Herm restricted to V_{chi_0}."""
    n = 1 << k
    total = np.zeros((n, n), dtype=np.float64)
    for j in range(k):
        total += build_principal_channel_kronecker(k, j)
    return total


def principal_subspace_basis(k: int) -> np.ndarray:
    """Build orthonormal divisor-stratum basis E (shape N_k x 2^k) for V_{chi_0}.

    Column index matches binary prime divisibility (MSB = p_1 = 2, LSB = p_k):
    bit i = 0 if p_i does not divide d, bit i = 1 if p_i divides d.
    """
    nk, primes = primorial(k)
    n_cols = 1 << k
    basis = np.zeros((nk, n_cols), dtype=np.float64)
    gcds = np.array([math.gcd(a, nk) for a in range(nk)], dtype=np.int64)

    for col in range(n_cols):
        d = 1
        for i, p in enumerate(primes):
            if (col >> (k - 1 - i)) & 1:
                d *= p
        mask = gcds == d
        basis[mask, col] = 1.0 / math.sqrt(int(np.sum(mask)))
    return basis


def unit_group_elements(k: int) -> list[int]:
    """Return all units g in (Z / N_k Z)^x."""
    nk, _ = primorial(k)
    return [g for g in range(1, nk) if math.gcd(g, nk) == 1]


def apply_unit_intertwining(mat: np.ndarray, g: int, nk: int) -> np.ndarray:
    """Compute U_g @ mat @ U_g efficiently using residue permutations."""
    idx = np.arange(nk, dtype=np.int64)
    perm = (g * idx) % nk
    inv_perm = (pow(g, -1, nk) * idx) % nk
    return mat[np.ix_(perm, inv_perm)]


def g2_generators(k: int) -> tuple[list[int], int]:
    """Return the k - 1 CRT generators g_j of the 2-torsion group G_2(N_k)."""
    nk, primes = primorial(k)
    gens: list[int] = []
    for p in primes[1:]:
        m = nk // p
        g = (1 - 2 * m * pow(m, -1, p)) % nk
        gens.append(g)
    return gens, nk


def hybrid_eigenvalues(k: int, eps0: float, m_max: int) -> tuple[np.ndarray, np.ndarray]:
    """Return flattened eigenvalues lambda_{m,a} and matching momentum indices m."""
    t_herm, nk, _ = build_T_herm(k)
    mu = np.linalg.eigvalsh(t_herm)
    m_vals = np.arange(-m_max, m_max + 1, dtype=np.int64)
    lam_grid = (2.0 * np.pi * m_vals[:, None] / math.log(nk)) + eps0 * mu[None, :]
    m_grid = np.broadcast_to(m_vals[:, None], lam_grid.shape)
    return lam_grid.ravel(), m_grid.ravel()
