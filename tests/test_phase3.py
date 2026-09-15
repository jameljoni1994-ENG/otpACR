"""Tests for matrix-free problems and first-order baselines."""

from __future__ import annotations

import numpy as np

from cubic_reg.problems import MatrixFreeQuadratic, LogSumExp
from cubic_reg.solvers import krylov, first_order


def test_matrix_free_quadratic_small():
    p = MatrixFreeQuadratic(n=80, condition=20.0, seed=0)
    g = p.grad(p.x_star)
    assert np.linalg.norm(g) < 1e-6
    v = np.ones(p.dim)
    Hv = p.hvp(np.zeros(p.dim), v)
    assert Hv.shape == (p.dim,)


def test_krylov_matrix_free():
    p = MatrixFreeQuadratic(n=120, condition=30.0, seed=1)
    r = krylov.minimize(p, x0=np.zeros(p.dim), M=1.0, eps=1e-5, krylov_dim=25, max_iter=40)
    assert r.grad_norm <= 1e-4
    assert r.n_hvp > 0


def test_nesterov_reduces_f():
    p = LogSumExp(n=10, m=20, mu=0.1, seed=0)
    x0 = np.ones(p.dim)
    f0 = p.f(x0)
    H = p.hess(x0)
    step = 1.0 / (float(np.linalg.norm(H, 2)) + 1e-8)
    r = first_order.minimize_nesterov(p, x0=x0, step=step, eps=1e-5, max_iter=200)
    assert r.f <= f0 + 1e-8
    assert min(r.history_f) < f0
