"""Phase-5 tests."""

from __future__ import annotations

import numpy as np

from cubic_reg.problems import Rosenbrock
from cubic_reg.solvers import krylov


def test_theta_policies_smoke():
    p = Rosenbrock(n=4)
    x0 = np.array([-1.2, 1.0, -1.2, 1.0])
    for power, adaptive in [(0.0, False), (1.0, True), (2.0, True)]:
        r = krylov.minimize(
            p,
            x0=x0.copy(),
            M=10.0,
            eps=1e-4,
            max_iter=30,
            krylov_dim=8,
            m_max=20,
            adaptive_tol=adaptive,
            inexact_tol=0.5,
            tol_power=power,
        )
        assert np.isfinite(r.f)
        assert r.n_hvp > 0
