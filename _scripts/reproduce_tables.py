"""Reproduce Tables 1–4 of the Primorial–Feedback Bridge paper."""

from __future__ import annotations

import math
import numpy as np

from primorial_feedback_bridge.feedback import f_pos, omega_g_matrix
from primorial_feedback_bridge.fiber_gating import (
    fiber_bandpass_diagnostics,
    principal_bandpass_diagnostics,
)
from primorial_feedback_bridge.primorial import (
    build_prime_channel,
    build_principal_channel_kronecker,
    primorial,
)
from primorial_feedback_bridge.resolvent_counting import trace_norm_spectral_counter
from primorial_feedback_bridge.synthesis_obstructions import (
    bulk_spacing_ratio,
    gauge_equivalence_check,
    non_abelian_wilson_loop,
    principal_subspace_leak,
    symmetric_multiprime_product,
)


def print_table_1() -> None:
    print("\n=== Table 1: Exact Basin Partition of sigma(T_k^Herm) (k = 1..5) ===")
    header = (
        f"{'k':>2} {'N_k':>5} {'Act':>4} {'Abs':>4} {'B(+1)':>5} {'B(-1)':>5} "
        f"{'Ejec':>4} {'Tr(S_k)':>7} {'MinDist':>10} {'Max U_g Err':>12}"
    )
    print(header)
    for k in range(1, 6):
        s = fiber_bandpass_diagnostics(k)
        print(
            f"{s['k']:>2} {s['nk']:>5} {s['active_rank']:>4} {s['absorbed']:>4} "
            f"{s['basin_pos']:>5} {s['basin_neg']:>5} {s['ejected_total']:>4} "
            f"{s['trace_limit']:>+7} {s['min_dist_boundary']:>10.3e} "
            f"{s['max_ug_intertwine_err']:>12.2e}"
        )


def print_table_2() -> None:
    print("\n=== Table 2: Principal-Character Block V_{chi_0} Basins (k = 1..11) ===")
    header = (
        f"{'k':>2} {'Act':>5} {'Abs':>4} {'B(+1)':>5} {'B(-1)':>5} "
        f"{'Surv%':>7} {'Ej+':>4} {'Ej-':>4} {'Tr(S_k)':>7}"
    )
    print(header)
    for k in range(1, 12):
        s = principal_bandpass_diagnostics(k)
        print(
            f"{s['k']:>2} {s['active_rank']:>5} {s['absorbed']:>4} "
            f"{s['basin_pos']:>5} {s['basin_neg']:>5} {s['surviving_pct']:>6.1f}% "
            f"{s['ejected_pos']:>4} {s['ejected_neg']:>4} {s['trace_limit']:>+7}"
        )


def print_table_3() -> None:
    print("\n=== Table 3: Super-Gaussian Tail & Trace-Norm Counting (eps0=0.35, tau=0.1) ===")
    header = (
        f"{'k':>2} {'N_k':>4} {'Count':>6} {'MinGap':>10} {'Iters':>5} "
        f"{'Tail R_tau':>11} {'Tail Heat':>11} {'Tail SG':>11}"
    )
    print(header)
    for k in (2, 3, 4):
        s = trace_norm_spectral_counter(k, tau=0.1, eps0=0.35, m_max=150, n_iters=14)
        print(
            f"{s['k']:>2} {s['nk']:>4} {s['exact_count']:>6} {s['min_gap']:>10.3e} "
            f"{str(s['iters_to_1e12']):>5} {s['tail_raw_resolvent']:>11.3e} "
            f"{s['tail_heat_kernel']:>11.3e} {s['tail_super_gaussian']:>11.3e}"
        )


def print_table_4() -> None:
    print("\n=== Table 4: V_{chi_0} Rigidity & 1D Gauge Obstruction ===")
    for k in (3, 4):
        _, primes = primorial(k)
        channels = [build_prime_channel(k, p) for p in primes]
        a_sym = symmetric_multiprime_product(channels, alpha=0.5)
        leak = principal_subspace_leak(a_sym, k)
        print(f"V_{{chi_0}} leak of A_sym(0.5) at k={k}: {leak:.3e}")

    for k in (8, 9, 10):
        ch_v0 = [build_principal_channel_kronecker(k, j) for j in range(k)]
        a_sym_v0 = symmetric_multiprime_product(ch_v0, alpha=0.5)
        w = np.linalg.eigvalsh(a_sym_v0)
        r_log = bulk_spacing_ratio(np.log(w))
        r_om = bulk_spacing_ratio(np.linalg.eigvalsh(omega_g_matrix(a_sym_v0)))
        print(f"k={k} (V_chi0): <r>_bulk log A_sym={r_log:.4f}, Omega_G(A_sym)={r_om:.4f}")

    gauge = gauge_equivalence_check(k=2, eps_func=lambda y: 0.8 * f_pos(np.exp(y)), m_max=40)
    print(
        f"1D gauge check on T_{{L_2}} (eps_bar={gauge['eps_bar']:.6f}): "
        f"max diff = {gauge['max_bulk_spectral_diff']:.3e}"
    )

    _, primes_4 = primorial(4)
    eps_funcs = {
        p: (lambda y, lp=math.log(p): float(1.2 * f_pos(math.exp(y - lp))))
        for p in primes_4
    }
    wilson = non_abelian_wilson_loop(k=4, eps_funcs=eps_funcs, n_steps=300)
    print(
        f"Non-Abelian Wilson loop at k=4: ||W_NA - W_Ab||_2 = {wilson['w_na_minus_w_ab_norm']:.3f}, "
        f"V_{{chi_0}} leak = {wilson['v_chi0_leak']:.3e}"
    )


if __name__ == "__main__":
    print_table_1()
    print_table_2()
    print_table_3()
    print_table_4()
