"""Stochastic / subsampled cubic regularization for finite-sum problems."""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from cubic_reg.metrics import OptimizationResult
from cubic_reg.problems.ml import FiniteSumLogistic, make_synthetic_logistic
from cubic_reg.subproblem import solve_cubic_subproblem

# Re-export for notebooks that import from solvers.stochastic
__all__ = [
    "FiniteSumLogistic",
    "make_synthetic_logistic",
    "minimize",
    "minimize_sgd",
    "minimize_adam",
]


def minimize(
    problem: FiniteSumLogistic,
    x0: Optional[np.ndarray] = None,
    M: float = 1.0,
    eps: float = 1e-4,
    max_iter: int = 200,
    batch_grad: int = 256,
    batch_hess: int = 64,
    seed: int = 0,
    verbose: bool = False,
) -> OptimizationResult:
    """Subsampled cubic regularization with independent grad/hess mini-batches."""
    problem.reset_counters()
    rng = np.random.default_rng(seed)
    x = np.zeros(problem.dim) if x0 is None else np.asarray(x0, dtype=float).copy()
    history_f: list[float] = []
    history_g: list[float] = []
    history_M: list[float] = []

    t0 = time.perf_counter()
    success = False
    message = "max_iter reached"
    gnorm = np.inf

    for k in range(max_iter):
        problem.set_batch(None)
        fx = problem.f(x)
        g_full = problem.grad(x)
        gnorm = float(np.linalg.norm(g_full))
        history_f.append(fx)
        history_g.append(gnorm)
        history_M.append(M)

        if verbose:
            print(f"Stoch-CR iter {k}: f={fx:.6e} ||g||={gnorm:.6e}")

        if gnorm <= eps:
            success = True
            message = "gradient norm below eps"
            break

        bg = min(batch_grad, problem.N)
        idx_g = rng.choice(problem.N, size=bg, replace=False)
        problem.set_batch(idx_g)
        g = problem.grad(x)

        bh = min(batch_hess, problem.N)
        idx_h = rng.choice(problem.N, size=bh, replace=False)
        problem.set_batch(idx_h)
        H = problem.hess(x)

        sol = solve_cubic_subproblem(g, H, M)
        if not sol.success:
            message = sol.message
            break
        x = x + sol.s

    problem.set_batch(None)
    elapsed = time.perf_counter() - t0
    c = problem.counters
    return OptimizationResult(
        x=x,
        f=problem.f(x),
        grad_norm=gnorm,
        nit=len(history_f),
        success=success,
        message=message,
        time_sec=elapsed,
        n_f=c.f,
        n_grad=c.grad,
        n_hess=c.hess,
        n_hvp=c.hvp,
        history_f=history_f,
        history_grad_norm=history_g,
        history_M=history_M,
        f_star=problem.f_star,
    )


def minimize_sgd(
    problem: FiniteSumLogistic,
    x0: Optional[np.ndarray] = None,
    lr: float = 0.1,
    batch: int = 64,
    max_iter: int = 500,
    seed: int = 0,
    eps: float = 1e-4,
) -> OptimizationResult:
    """Simple SGD baseline."""
    problem.reset_counters()
    rng = np.random.default_rng(seed)
    x = np.zeros(problem.dim) if x0 is None else np.asarray(x0, dtype=float).copy()
    history_f: list[float] = []
    history_g: list[float] = []
    t0 = time.perf_counter()
    success = False
    gnorm = np.inf

    for k in range(max_iter):
        problem.set_batch(None)
        fx = problem.f(x)
        gnorm = float(np.linalg.norm(problem.grad(x)))
        history_f.append(fx)
        history_g.append(gnorm)
        if gnorm <= eps:
            success = True
            break
        idx = rng.choice(problem.N, size=min(batch, problem.N), replace=False)
        problem.set_batch(idx)
        g = problem.grad(x)
        x = x - lr * g

    problem.set_batch(None)
    elapsed = time.perf_counter() - t0
    c = problem.counters
    return OptimizationResult(
        x=x,
        f=problem.f(x),
        grad_norm=gnorm,
        nit=len(history_f),
        success=success,
        message="done",
        time_sec=elapsed,
        n_f=c.f,
        n_grad=c.grad,
        history_f=history_f,
        history_grad_norm=history_g,
    )


def minimize_adam(
    problem: FiniteSumLogistic,
    x0: Optional[np.ndarray] = None,
    lr: float = 0.01,
    batch: int = 64,
    max_iter: int = 500,
    beta1: float = 0.9,
    beta2: float = 0.999,
    seed: int = 0,
    eps: float = 1e-4,
) -> OptimizationResult:
    """Adam baseline."""
    problem.reset_counters()
    rng = np.random.default_rng(seed)
    x = np.zeros(problem.dim) if x0 is None else np.asarray(x0, dtype=float).copy()
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    history_f: list[float] = []
    history_g: list[float] = []
    t0 = time.perf_counter()
    success = False
    gnorm = np.inf

    for k in range(1, max_iter + 1):
        problem.set_batch(None)
        fx = problem.f(x)
        gnorm = float(np.linalg.norm(problem.grad(x)))
        history_f.append(fx)
        history_g.append(gnorm)
        if gnorm <= eps:
            success = True
            break
        idx = rng.choice(problem.N, size=min(batch, problem.N), replace=False)
        problem.set_batch(idx)
        g = problem.grad(x)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g * g)
        mhat = m / (1 - beta1**k)
        vhat = v / (1 - beta2**k)
        x = x - lr * mhat / (np.sqrt(vhat) + 1e-8)

    problem.set_batch(None)
    elapsed = time.perf_counter() - t0
    c = problem.counters
    return OptimizationResult(
        x=x,
        f=problem.f(x),
        grad_norm=gnorm,
        nit=len(history_f),
        success=success,
        message="done",
        time_sec=elapsed,
        n_f=c.f,
        n_grad=c.grad,
        history_f=history_f,
        history_grad_norm=history_g,
    )
