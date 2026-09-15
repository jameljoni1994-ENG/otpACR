"""Softmax regression tests."""

from __future__ import annotations

import numpy as np

from cubic_reg.problems import SoftmaxRegression
from cubic_reg.solvers import krylov


def test_softmax_grad_hvp_shapes():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 8))
    y = rng.integers(0, 3, size=40)
    p = SoftmaxRegression(X, y, n_classes=3, l2=1e-3)
    theta = 0.01 * rng.standard_normal(p.dim)
    assert np.isfinite(p.f(theta))
    g = p.grad(theta)
    Hv = p.hvp(theta, np.ones(p.dim))
    assert g.shape == (p.dim,)
    assert Hv.shape == (p.dim,)
    assert 0.0 <= p.accuracy(theta) <= 1.0


def test_krylov_softmax_small():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((60, 6))
    y = rng.integers(0, 4, size=60)
    p = SoftmaxRegression(X, y, n_classes=4, l2=1e-3)
    r = krylov.minimize(
        p, x0=np.zeros(p.dim), M=1.0, eps=1e-3, max_iter=25, krylov_dim=15, adaptive_tol=True
    )
    assert r.f < p.f(np.zeros(p.dim))
    assert r.n_hvp > 0
