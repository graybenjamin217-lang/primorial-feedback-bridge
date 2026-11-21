"""Tests for Section VI: Prime-channel synthesis, rigidity, and 1D gauge obstruction."""

import math
import numpy as np

from primorial_feedback_bridge.feedback import f_pos, omega_g_matrix
from primorial_feedback_bridge.primorial import build_prime_channel, primorial
from primorial_feedback_bridge.synthesis_obstructions import (
    bch_fourth_order_approx,
    channel_exp_closed_form,
    gauge_equivalence_check,
    non_abelian_wilson_loop,
    principal_subspace_leak,
    symmetric_multiprime_product,
)


def test_closed_form_channel_exponential() -> None:
    for k in (2, 3, 4):
        _, primes = primorial(k)
        for p in primes:
            c_p = build_prime_channel(k, p)
            alpha = 0.65
            closed = channel_exp_closed_form(c_p, alpha)
            w, v = np.linalg.eigh(c_p)
            dense = (v * np.exp(alpha * w)[None, :]) @ v.T
            assert np.max(np.abs(closed - dense)) < 1e-14


def test_fourth_order_symmetric_bch_scaling() -> None:
    c2 = build_prime_channel(k=3, p=2)
    c3 = build_prime_channel(k=3, p=3)

    def bch_residual(alpha: float) -> float:
        e_half = channel_exp_closed_form(c2, 0.5 * alpha)
        e_mid = channel_exp_closed_form(c3, alpha)
        exact = omega_g_matrix(e_half @ e_mid @ e_half)
        approx = bch_fourth_order_approx(c2, c3, alpha)
        return float(np.linalg.norm(exact - approx, ord=2))

    r1 = bch_residual(0.04)
    r2 = bch_residual(0.02)
    # Halving alpha reduces O(alpha^5) error by ~2^5 = 32.
    assert abs((r1 / r2) - 32.0) < 1.5


def test_v_chi0_rigidity_and_1d_gauge_obstruction() -> None:
    for k in (3, 4):
        _, primes = primorial(k)
        channels = [build_prime_channel(k, p) for p in primes]
        a_sym = symmetric_multiprime_product(channels, alpha=0.5)
        assert principal_subspace_leak(a_sym, k) < 1e-13

    gauge = gauge_equivalence_check(
        k=2, eps_func=lambda y: 0.8 * f_pos(np.exp(y)), m_max=40, n_grid=4096
    )
    assert abs(gauge["eps_bar"] - 0.486339) < 1e-5
    assert gauge["max_bulk_spectral_diff"] < 1e-4

    _, primes_3 = primorial(3)
    eps_funcs = {
        p: (lambda y, lp=math.log(p): float(1.2 * f_pos(math.exp(y - lp))))
        for p in primes_3
    }
    wilson = non_abelian_wilson_loop(k=3, eps_funcs=eps_funcs, n_steps=150)
    assert wilson["w_na_minus_w_ab_norm"] > 0.1
    assert wilson["v_chi0_leak"] < 1e-13
