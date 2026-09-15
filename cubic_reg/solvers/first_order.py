"""First-order baselines: gradient descent and Nesterov accelerated gradient."""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from cubic_reg.metrics import OptimizationResult
from cubic_reg.problems.base import Problem


def _backtrack(problem: Problem, x: np.ndarray, g: np.ndarray, fx: float, alpha: float):
    gnorm2 = float(g @ g)
    while True:
        x_trial = x - alpha * g
        f_trial = problem.f(x_trial)
        if f_trial <= fx - 1e-4 * alpha * gnorm2 or alpha < 1e-18:
            return x_trial, f_trial, alpha
        alpha *= 0.5


def minimize_gd(
    problem: Problem,
    x0: Optional[np.ndarray] = None,
    step: float = 0.01,
    eps: float = 1e-6,
    max_iter: int = 1000,
    verbose: bool = False,
) -> OptimizationResult:
    """Plain gradient descent with Armijo backtracking."""
    problem.reset_counters()
    x = np.zeros(problem.dim) if x0 is None else np.asarray(x0, dtype=float).copy()
    history_f: list[float] = []
    history_g: list[float] = []
    t0 = time.perf_counter()
    success = False
    message = "max_iter reached"
    gnorm = np.inf
    fx = problem.f(x)
    alpha = step

    for k in range(max_iter):
        g = problem.grad(x)
        gnorm = float(np.linalg.norm(g))
        history_f.append(fx)
        history_g.append(gnorm)
        if verbose:
            print(f"GD {k}: f={fx:.6e} ||g||={gnorm:.6e}")
        if gnorm <= eps:
            success = True
            message = "gradient norm below eps"
            break
        x, fx, alpha = _backtrack(problem, x, g, fx, alpha)
        alpha = min(step, alpha * 1.5)

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
        history_f=history_f,
        history_grad_norm=history_g,
        f_star=problem.f_star,
    )


def minimize_nesterov(
    problem: Problem,
    x0: Optional[np.ndarray] = None,
    step: float = 0.01,
    eps: float = 1e-6,
    max_iter: int = 1000,
    momentum: Optional[float] = None,
    verbose: bool = False,
) -> OptimizationResult:
    """Nesterov AG with monotone restart: accept only non-increasing f(x)."""
    problem.reset_counters()
    x = np.zeros(problem.dim) if x0 is None else np.asarray(x0, dtype=float).copy()
    y = x.copy()
    history_f: list[float] = []
    history_g: list[float] = []
    t0 = time.perf_counter()
    success = False
    message = "max_iter reached"
    gnorm = np.inf
    fx = problem.f(x)
    alpha = step
    t_nes = 1.0

    for k in range(max_iter):
        g = problem.grad(y)
        gnorm = float(np.linalg.norm(g))
        history_f.append(fx)
        history_g.append(gnorm)
        if verbose:
            print(f"NAG {k}: f={fx:.6e} ||g||={gnorm:.6e}")
        if gnorm <= eps:
            success = True
            message = "gradient norm below eps"
            break

        x_trial, f_trial, alpha = _backtrack(problem, y, g, problem.f(y), alpha)

        # Monotone restart if the iterate does not improve f(x)
        if f_trial > fx:
            g = problem.grad(x)
            gnorm = float(np.linalg.norm(g))
            x_trial, f_trial, alpha = _backtrack(problem, x, g, fx, alpha)
            y = x_trial
            x = x_trial
            fx = f_trial
            t_nes = 1.0
            alpha = min(step, alpha * 1.5)
            continue

        if momentum is None:
            t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t_nes**2))
            beta = (t_nes - 1.0) / t_next
            t_nes = t_next
        else:
            beta = momentum

        y = x_trial + beta * (x_trial - x)
        x = x_trial
        fx = f_trial
        alpha = min(step, alpha * 1.5)

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
        history_f=history_f,
        history_grad_norm=history_g,
        f_star=problem.f_star,
    )
