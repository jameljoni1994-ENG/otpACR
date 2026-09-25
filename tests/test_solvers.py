"""Unit tests for cubic regularization package."""

from __future__ import annotations

import numpy as np
import pytest

from cubic_reg.problems import Quadratic, LogSumExp, Rosenbrock, Rastrigin
from cubic_reg.subproblem import solve_cubic_subproblem, model_decrease
from cubic_reg.solvers import cr, krylov, arc, quasi_newton, accelerated, stochastic, tensor, distributed


def test_quadratic_optimum():
    p = Quadratic(n=20, condition=50.0, seed=1)
    assert abs(p.f(p.x_star) - p.f_star) < 1e-10
    assert np.linalg.norm(p.grad(p.x_star)) < 1e-8


def test_subproblem_decreases_model():
    rng = np.random.default_rng(0)
    n = 15
    g = rng.standard_normal(n)
    A = rng.standard_normal((n, n))
    H = A.T @ A + 0.1 * np.eye(n)
    M = 2.0
    sol = solve_cubic_subproblem(g, H, M)
    assert sol.success
    assert sol.model_decrease >= -1e-10
    assert model_decrease(g, H, sol.s, M) >= -1e-8


def test_cr_converges_quadratic():
    p = Quadratic(n=30, condition=100.0, seed=0)
    x0 = np.ones(p.dim)
    res = cr.minimize(p, x0=x0, M=1.0, eps=1e-6, max_iter=100)
    assert res.success
    assert res.grad_norm <= 1e-6
    assert res.f <= p.f(x0) + 1e-8


def test_krylov_converges_quadratic():
    p = Quadratic(n=40, condition=50.0, seed=2)
    res = krylov.minimize(p, x0=np.ones(p.dim), M=1.0, eps=1e-5, krylov_dim=15, max_iter=80)
    assert res.grad_norm <= 1e-4
    assert res.n_hvp > 0


def test_arc_adaptive():
    p = Quadratic(n=25, condition=100.0, seed=3)
    res = arc.minimize(p, x0=np.ones(p.dim), M0=1e-3, eps=1e-6, max_iter=150)
    assert res.success
    assert res.grad_norm <= 1e-6


def test_arc_krylov_mode():
    p = Quadratic(n=30, condition=20.0, seed=4)
    res = arc.minimize(
        p,
        x0=np.ones(p.dim),
        M0=1.0,
        mode="krylov",
        krylov_dim=12,
        m_max=40,
        adaptive_tol=True,
        inexact_tol=0.5,
        eps=1e-5,
    )
    assert res.grad_norm <= 1e-4
    assert res.n_hvp > 0
    assert len(res.history_theta) == len(res.history_f)
    assert len(res.history_m) == len(res.history_f)
    assert len(res.history_resid) == len(res.history_f)
    # θ_k decreases under adaptive schedule
    if len(res.history_theta) >= 2:
        assert res.history_theta[1] <= res.history_theta[0] + 1e-12


def test_qn_cr():
    p = Quadratic(n=20, condition=30.0, seed=5)
    res = quasi_newton.minimize(p, x0=np.ones(p.dim), M=1.0, eps=1e-5, memory=8)
    assert res.grad_norm <= 1e-4


def test_qn_cr_matrix_free_matches_dense_shape():
    p = Quadratic(n=25, condition=20.0, seed=11)
    x0 = np.ones(p.dim)
    r_mf = quasi_newton.minimize(
        p, x0=x0.copy(), M=1.0, eps=1e-5, memory=8, matrix_free=True, krylov_dim=15, m_max=40
    )
    r_d = quasi_newton.minimize(
        p, x0=x0.copy(), M=1.0, eps=1e-5, memory=8, matrix_free=False
    )
    assert r_mf.grad_norm <= 1e-4
    assert r_d.grad_norm <= 1e-4
    assert len(r_mf.history_m) == len(r_mf.history_f)


def test_lbfgs_matvec_agrees_with_dense():
    rng = np.random.default_rng(0)
    n, m = 40, 6
    B = quasi_newton.LBFGSHessian(n, memory=m)
    for _ in range(m):
        s = rng.standard_normal(n)
        y = rng.standard_normal(n)
        # ensure curvature
        if y @ s <= 0:
            y = s + 0.1 * y
        B.update(s, y)
    v = rng.standard_normal(n)
    assert np.allclose(B.matvec(v), B.to_dense() @ v, rtol=1e-8, atol=1e-8)


def test_accelerated_reduces_f():
    p = Quadratic(n=20, condition=10.0, seed=6)
    x0 = np.ones(p.dim)
    f0 = p.f(x0)
    res = accelerated.minimize(p, x0=x0, M=1.0, eps=1e-5, max_iter=100)
    assert res.f < f0


def test_stochastic_logistic():
    prob = stochastic.make_synthetic_logistic(n_samples=400, n_features=20, seed=0)
    res = stochastic.minimize(
        prob,
        max_iter=40,
        batch_grad=64,
        batch_hess=32,
        eps=1e-2,
        M=1.0,
        adaptive_M=True,
        eval_every=4,
    )
    assert res.history_f[-1] <= res.history_f[0] + 1e-6
    assert res.rejected_steps >= 0


def test_stochastic_growing_batches():
    prob = stochastic.make_synthetic_logistic(n_samples=300, n_features=15, seed=1)
    res = stochastic.minimize(
        prob,
        max_iter=30,
        batch_grad=32,
        batch_hess=16,
        eps=1e-2,
        M=1.0,
        grow_batches=True,
        batch_grow_factor=1.5,
        eval_every=5,
    )
    assert np.isfinite(res.f)


def test_tensor_step():
    p = Quadratic(n=10, condition=5.0, seed=7)
    x0 = np.ones(p.dim)
    res = tensor.minimize(p, x0=x0, L=1.0, eps=1e-4, max_iter=30)
    assert res.f < p.f(x0)


def test_distributed_sketch():
    p = Quadratic(n=25, condition=20.0, seed=8)
    res = distributed.minimize(p, x0=np.ones(p.dim), n_workers=4, sketch_rank=8, eps=1e-4, max_iter=80)
    assert res.grad_norm <= 5e-3


def test_analytic_problems_smoke():
    for p in [LogSumExp(n=8, m=20), Rosenbrock(n=4), Rastrigin(n=5)]:
        x = np.zeros(p.dim)
        f = p.f(x)
        g = p.grad(x)
        H = p.hess(x)
        assert np.isfinite(f)
        assert g.shape == (p.dim,)
        assert H.shape == (p.dim, p.dim)


def test_mlp_and_logistic():
    from cubic_reg.problems import make_synthetic_logistic, make_synthetic_mlp

    logreg = make_synthetic_logistic(n_samples=100, n_features=10, seed=0)
    assert np.isfinite(logreg.f(np.zeros(logreg.dim)))
    mlp = make_synthetic_mlp(n_samples=40, d_in=4, hidden=3, seed=0)
    theta = np.zeros(mlp.dim)
    g = mlp.grad(theta)
    Hv = mlp.hvp(theta, np.ones(mlp.dim))
    assert g.shape == (mlp.dim,)
    assert Hv.shape == (mlp.dim,)
    res = quasi_newton.minimize(mlp, x0=theta, eps=1e-3, max_iter=25, memory=5)
    assert res.f <= mlp.f(theta) + 1e-8
