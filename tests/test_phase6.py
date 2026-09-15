"""Phase-6 smoke tests."""

from __future__ import annotations

import numpy as np

from cubic_reg.problems import Quadratic
from cubic_reg.solvers import krylov


def test_forced_small_krylov_grows():
    p = Quadratic(n=50, condition=200.0, seed=0)
    r = krylov.minimize(
        p,
        x0=np.ones(p.dim),
        M=1.0,
        eps=1e-5,
        max_iter=25,
        krylov_dim=3,
        m_max=30,
        adaptive_tol=True,
        inexact_tol=0.1,
        tol_power=1.0,
    )
    assert hasattr(r, "history_m")
    assert max(r.history_m) >= 3
    assert r.grad_norm <= 1e-4 or r.nit >= 5
