"""Baseline Nesterov Cubic Regularization (exact Hessian)."""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from cubic_reg.metrics import OptimizationResult
from cubic_reg.problems.base import Problem
from cubic_reg.subproblem import model_decrease, solve_cubic_subproblem


def minimize(
    problem: Problem,
    x0: Optional[np.ndarray] = None,
    M: float = 1.0,
    eps: float = 1e-6,
    max_iter: int = 200,
    verbose: bool = False,
) -> OptimizationResult:
    """Exact cubic regularization with dense Hessian subproblem solves."""
    problem.reset_counters()
    x = np.zeros(problem.dim) if x0 is None else np.asarray(x0, dtype=float).copy()
    history_f: list[float] = []
    history_g: list[float] = []
    history_M: list[float] = []

    t0 = time.perf_counter()
    success = False
    message = "max_iter reached"
    gnorm = np.inf
    fx = problem.f(x)

    for k in range(max_iter):
        g = problem.grad(x)
        gnorm = float(np.linalg.norm(g))
        history_f.append(fx)
        history_g.append(gnorm)
        history_M.append(M)

        if verbose:
            print(f"CR iter {k}: f={fx:.6e} ||g||={gnorm:.6e}")

        if gnorm <= eps:
            success = True
            message = "gradient norm below eps"
            break

        H = problem.hess(x)
        sol = solve_cubic_subproblem(g, H, M)
        if not sol.success:
            message = sol.message
            break

        s = sol.s
        x = x + s
        fx = problem.f(x)

        # Optional sanity: ensure model predicted decrease
        _ = model_decrease(g, H, s, M)

    elapsed = time.perf_counter() - t0
    c = problem.counters
    return OptimizationResult(
        x=x,
        f=fx,
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
