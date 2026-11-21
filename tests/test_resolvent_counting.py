"""Tests for Section IV: Super-Gaussian resolvent lifting and spectral counting."""

import math
import numpy as np

from primorial_feedback_bridge.feedback import f_pos
from primorial_feedback_bridge.resolvent_counting import (
    heat_kernel_eigs,
    positive_resolvent_eigs,
    super_gaussian_lift_eigs,
    trace_norm_spectral_counter,
)


def test_super_gaussian_identity_and_infrared_flatness() -> None:
    tau = 0.1
    lam = np.linspace(-25.0, 25.0, 2001)
    lhs = f_pos(positive_resolvent_eigs(lam, tau))
    rhs = super_gaussian_lift_eigs(lam, tau)
    assert np.max(np.abs(lhs - rhs)) < 1e-15

    # Strictly faster decay than heat kernel when tau * lam^2 > e - 1.
    high = np.abs(lam) > math.sqrt((math.e - 1.0) / tau) + 0.1
    assert np.all(rhs[high] < heat_kernel_eigs(lam[high], tau))

    # Fourth-order infrared flatness: 1 - f((1 + tau*lam^2)^-1) ~ tau^2 * lam^4.
    small_lam = np.array([1e-2, 5e-3])
    dev = 1.0 - super_gaussian_lift_eigs(small_lam, tau)
    lead = (tau**2) * (small_lam**4)
    assert np.allclose(dev, lead, rtol=1e-3)


def test_trace_norm_spectral_counting_table_3() -> None:
    expected_counts = {2: 8, 3: 93, 4: 1100}
    for k, count in expected_counts.items():
        res = trace_norm_spectral_counter(k, tau=0.1, eps0=0.35, m_max=150, n_iters=14)
        assert res["exact_count"] == count
        assert res["lift_identity_err"] < 1e-15
        assert res["iters_to_1e12"] is not None
        assert res["tail_super_gaussian"] < res["tail_heat_kernel"]
        assert res["tail_heat_kernel"] < res["tail_raw_resolvent"]
        _, final_trace, final_s1 = res["history"][-1]
        assert abs(final_trace - count) < 1e-12
        assert final_s1 < 1e-12
