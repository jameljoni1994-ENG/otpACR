"""Accelerated cubic regularization (Nesterov-style momentum)."""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from cubic_reg.metrics import OptimizationResult
from cubic_reg.problems.base import Problem
from cubic_reg.subproblem import solve_cubic_subproblem


def minimize(
    problem: Problem,
    x0: Optional[np.ndarray] = None,
    M: float = 1.0,
    eps: float = 1e-6,
    max_iter: int = 300,
    verbose: bool = False,
) -> OptimizationResult:
    """Accelerated CR with auxiliary extrapolation and cubic step at y_k.

    Practical scheme targeting faster gap decay on smooth convex problems:
      A_{k+1} = A_k + a_{k+1} with a_{k+1} ~ (k+1)^2
      y_k = (A_k x_k + a_{k+1} v_k) / A_{k+1}
      x_{k+1} = y_k + s_k  where s_k solves the cubic model at y_k
      v_{k+1} = x_{k+1} + (A_k / A_{k+1}) (x_{k+1} - x_k)
    """
    problem.reset_counters()
    x = np.zeros(problem.dim) if x0 is None else np.asarray(x0, dtype=float).copy()
    v = x.copy()
    A = 0.0
    history_f: list[float] = []
    history_g: list[float] = []
    history_M: list[float] = []

    t0 = time.perf_counter()
    success = False
    message = "max_iter reached"
    gnorm = np.inf
    fx = problem.f(x)

    for k in range(max_iter):
        a = 1.0 if A <= 0 else (k + 1) ** 2 / 9.0
        A_next = A + a
        y = (A * x + a * v) / A_next

        g_x = problem.grad(x)
        gnorm = float(np.linalg.norm(g_x))
        fx = problem.f(x)
        history_f.append(fx)
        history_g.append(gnorm)
        history_M.append(M)

        if verbose:
            print(f"Acc-CR iter {k}: f={fx:.6e} ||g||={gnorm:.6e}")

        if gnorm <= eps:
            success = True
            message = "gradient norm below eps"
            break

        g = problem.grad(y)
        H = problem.hess(y)
        sol = solve_cubic_subproblem(g, H, M)
        if not sol.success:
            message = sol.message
            break

        x_new = y + sol.s
        v = x_new + (A / A_next) * (x_new - x)
        x = x_new
        A = A_next

    elapsed = time.perf_counter() - t0
    c = problem.counters
    return OptimizationResult(
        x=x,
        f=fx if success else problem.f(x),
        grad_norm=gnorm if success else float(np.linalg.norm(problem.grad(x))),
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
