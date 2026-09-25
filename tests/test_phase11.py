"""Phase-11 tests supporting Theorem mf-qn (matrix-free QN-CR)."""

from __future__ import annotations

import numpy as np

from cubic_reg.problems import Quadratic
from cubic_reg.solvers import quasi_newton
from cubic_reg.solvers.quasi_newton import LBFGSHessian


def test_compact_matvec_matches_dense():
    rng = np.random.default_rng(42)
    n, mem = 60, 8
    B = LBFGSHessian(n, memory=mem)
    for _ in range(mem):
        s = rng.standard_normal(n)
        y = s + 0.2 * rng.standard_normal(n)
        if y @ s <= 0:
            y = s.copy()
        B.update(s, y)
    for _ in range(10):
        v = rng.standard_normal(n)
        assert np.allclose(B.matvec(v), B.to_dense() @ v, rtol=1e-9, atol=1e-9)


def test_mf_and_dense_qn_agree_on_quadratic():
    p = Quadratic(n=30, condition=25.0, seed=0)
    x0 = np.ones(p.dim)
    r_mf = quasi_newton.minimize(
        p, x0=x0.copy(), M=1.0, eps=1e-6, memory=8, matrix_free=True, krylov_dim=20, m_max=30
    )
    r_d = quasi_newton.minimize(p, x0=x0.copy(), M=1.0, eps=1e-6, memory=8, matrix_free=False)
    assert r_mf.grad_norm <= 1e-5
    assert r_d.grad_norm <= 1e-5
    assert abs(r_mf.f - r_d.f) <= 1e-4
