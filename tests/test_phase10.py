"""Phase-10 unit tests (no full MNIST download required)."""

from __future__ import annotations

import numpy as np

from cubic_reg.problems import SoftmaxRegression, MatrixFreeQuadratic, make_synthetic_logistic
from cubic_reg.solvers import quasi_newton, sketch_krylov, stochastic
from cubic_reg.training import l2_path_early_stop


def _tiny_softmax(seed=0):
    rng = np.random.default_rng(seed)
    N, d, C = 80, 8, 3
    X = rng.standard_normal((N, d))
    y = rng.integers(0, C, size=N)
    return SoftmaxRegression(X, y, n_classes=C, l2=1e-3), X[:20], y[:20]


def test_l2_path_early_stop_smoke():
    prob, Xv, yv = _tiny_softmax()
    res = l2_path_early_stop(
        prob,
        Xv,
        yv,
        l2_grid=[1e-2, 1e-3],
        solver="arc",
        max_iter=8,
        patience=3,
        krylov_dim=8,
        m_max=12,
    )
    assert res.best_val_acc >= 0.0
    assert "l2_path" in res.message
    assert res.x.shape == (prob.dim,)


def test_qn_matrix_free_largeish():
    p = MatrixFreeQuadratic(n=500, condition=20.0, seed=0)
    r = quasi_newton.minimize(
        p,
        x0=np.zeros(p.dim),
        M=1.0,
        eps=1e-4,
        max_iter=30,
        memory=10,
        matrix_free=True,
        krylov_dim=20,
        m_max=40,
    )
    assert r.grad_norm <= 5e-3
    assert len(r.history_m) == len(r.history_f)


def test_sketch_scaling_smoke():
    p = MatrixFreeQuadratic(n=800, condition=30.0, seed=1)
    r = sketch_krylov.minimize(
        p, x0=np.zeros(p.dim), M0=1.0, eps=1e-3, max_iter=15, sketch_rank=15, krylov_dim=12
    )
    assert np.isfinite(r.f)
    assert r.n_hvp > 0


def test_stochastic_arc_ratio():
    prob = make_synthetic_logistic(n_samples=250, n_features=12, seed=2)
    r = stochastic.minimize(
        prob,
        max_iter=25,
        batch_grad=48,
        batch_hess=24,
        adaptive_M=True,
        eval_every=5,
        grow_batches=True,
        eps=5e-2,
    )
    assert np.isfinite(r.f)
    assert len(r.history_M) == len(r.history_f)
