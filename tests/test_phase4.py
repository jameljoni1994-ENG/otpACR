"""Phase-4 unit tests."""

from __future__ import annotations

import numpy as np

from cubic_reg.problems import Quadratic, make_mnist_like_logistic, MatrixFreeQuadratic
from cubic_reg.solvers import krylov, sketch_krylov


def test_adaptive_inexact_krylov():
    p = Quadratic(n=60, condition=40.0, seed=0)
    r = krylov.minimize(
        p, x0=np.ones(p.dim), M=1.0, eps=1e-6, krylov_dim=8, adaptive_tol=True, m_max=40
    )
    assert r.grad_norm <= 1e-5
    assert hasattr(r, "history_m")
    assert len(r.history_m) >= 1


def test_sketch_krylov_mf():
    p = MatrixFreeQuadratic(n=100, condition=20.0, seed=0)
    r = sketch_krylov.minimize(
        p, x0=np.zeros(p.dim), M0=1.0, eps=1e-4, max_iter=30, sketch_rank=15, krylov_dim=12
    )
    assert r.grad_norm <= 5e-3
    assert r.n_hvp > 0


def test_mnist_like_shapes():
    p = make_mnist_like_logistic(n_samples=100, n_features=784, seed=0)
    assert p.dim == 784
    assert p.X.shape == (100, 784)
    assert set(np.unique(p.y)).issubset({-1.0, 1.0})
