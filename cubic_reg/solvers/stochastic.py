"""Stochastic / subsampled cubic regularization for finite-sum problems."""

from __future__ import annotations

import time
from typing import List, Optional

import numpy as np

from cubic_reg.metrics import OptimizationResult
from cubic_reg.problems.ml import FiniteSumLogistic, make_synthetic_logistic
from cubic_reg.subproblem import model_decrease, solve_cubic_subproblem

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
    adaptive_M: bool = True,
    eta1: float = 0.1,
    eta2: float = 0.9,
    gamma1: float = 0.5,
    gamma2: float = 2.0,
    M_min: float = 1e-16,
    M_max: float = 1e16,
    eval_every: int = 5,
    grow_batches: bool = False,
    batch_grow_factor: float = 1.5,
    batch_grad_max: Optional[int] = None,
    batch_hess_max: Optional[int] = None,
    verbose: bool = False,
) -> OptimizationResult:
    """Subsampled cubic regularization with optional ARC-style ratio test.

    Full-batch ``f`` / ``grad`` are evaluated only every ``eval_every`` iterations
    (and on the final iterate) for logging / stopping, while steps use mini-batches.
    """
    problem.reset_counters()
    rng = np.random.default_rng(seed)
    x = np.zeros(problem.dim) if x0 is None else np.asarray(x0, dtype=float).copy()
    history_f: List[float] = []
    history_g: List[float] = []
    history_M: List[float] = []
    rejected = 0

    bg = min(batch_grad, problem.N)
    bh = min(batch_hess, problem.N)
    bg_max = batch_grad_max if batch_grad_max is not None else problem.N
    bh_max = batch_hess_max if batch_hess_max is not None else problem.N

    t0 = time.perf_counter()
    success = False
    message = "max_iter reached"
    gnorm = np.inf
    fx = np.nan
    M_cur = float(M)

    def _full_eval(xk: np.ndarray):
        problem.set_batch(None)
        f_val = problem.f(xk)
        g_full = problem.grad(xk)
        return float(f_val), float(np.linalg.norm(g_full))

    for k in range(max_iter):
        do_eval = (k % max(1, eval_every) == 0) or (k == max_iter - 1)
        if do_eval or not history_f:
            fx, gnorm = _full_eval(x)
            history_f.append(fx)
            history_g.append(gnorm)
            history_M.append(M_cur)
            if verbose:
                print(f"Stoch-CR iter {k}: f={fx:.6e} ||g||={gnorm:.6e} M={M_cur:.3e}")
            if gnorm <= eps:
                success = True
                message = "gradient norm below eps"
                break
        else:
            # keep histories aligned with accepted outer iterations
            history_f.append(history_f[-1] if history_f else fx)
            history_g.append(history_g[-1] if history_g else gnorm)
            history_M.append(M_cur)

        idx_g = rng.choice(problem.N, size=min(bg, problem.N), replace=False)
        problem.set_batch(idx_g)
        g = problem.grad(x)

        idx_h = rng.choice(problem.N, size=min(bh, problem.N), replace=False)
        problem.set_batch(idx_h)
        H = problem.hess(x)

        sol = solve_cubic_subproblem(g, H, M_cur)
        if not sol.success:
            if adaptive_M:
                M_cur = min(M_max, M_cur * gamma2)
                rejected += 1
                continue
            message = sol.message
            break

        s = sol.s
        md = model_decrease(g, H, s, M_cur)

        if adaptive_M:
            if md <= 0:
                M_cur = min(M_max, M_cur * gamma2)
                rejected += 1
                continue
            # Ratio uses batch model vs batch function decrease for cheap accept/reject
            problem.set_batch(idx_g)
            f_batch = float(problem.f(x))
            f_trial_batch = float(problem.f(x + s))
            actual = f_batch - f_trial_batch
            rho = actual / md if md > 0 else -np.inf
            if rho < eta1:
                rejected += 1
                M_cur = min(M_max, M_cur * gamma2)
                continue
            if rho >= eta2:
                M_cur = max(M_min, gamma1 * M_cur)

        x = x + s

        if grow_batches:
            bg = min(bg_max, max(bg, int(bg * batch_grow_factor)))
            bh = min(bh_max, max(bh, int(bh * batch_grow_factor)))

    problem.set_batch(None)
    fx_final, gnorm_final = _full_eval(x)
    if history_f:
        history_f[-1] = fx_final
        history_g[-1] = gnorm_final
    elapsed = time.perf_counter() - t0
    c = problem.counters
    return OptimizationResult(
        x=x,
        f=fx_final,
        grad_norm=gnorm_final,
        nit=len(history_f),
        success=success,
        message=message,
        time_sec=elapsed,
        n_f=c.f,
        n_grad=c.grad,
        n_hess=c.hess,
        n_hvp=c.hvp,
        rejected_steps=rejected,
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
    eval_every: int = 5,
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
        if k % max(1, eval_every) == 0:
            problem.set_batch(None)
            fx = problem.f(x)
            gnorm = float(np.linalg.norm(problem.grad(x)))
            history_f.append(fx)
            history_g.append(gnorm)
            if gnorm <= eps:
                success = True
                break
        else:
            if history_f:
                history_f.append(history_f[-1])
                history_g.append(history_g[-1])
        idx = rng.choice(problem.N, size=min(batch, problem.N), replace=False)
        problem.set_batch(idx)
        g = problem.grad(x)
        x = x - lr * g

    problem.set_batch(None)
    fx = problem.f(x)
    gnorm = float(np.linalg.norm(problem.grad(x)))
    if history_f:
        history_f[-1] = fx
        history_g[-1] = gnorm
    elapsed = time.perf_counter() - t0
    c = problem.counters
    return OptimizationResult(
        x=x,
        f=fx,
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
    eval_every: int = 5,
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
        if (k - 1) % max(1, eval_every) == 0:
            problem.set_batch(None)
            fx = problem.f(x)
            gnorm = float(np.linalg.norm(problem.grad(x)))
            history_f.append(fx)
            history_g.append(gnorm)
            if gnorm <= eps:
                success = True
                break
        else:
            if history_f:
                history_f.append(history_f[-1])
                history_g.append(history_g[-1])
        idx = rng.choice(problem.N, size=min(batch, problem.N), replace=False)
        problem.set_batch(idx)
        g = problem.grad(x)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g * g)
        mhat = m / (1 - beta1**k)
        vhat = v / (1 - beta2**k)
        x = x - lr * mhat / (np.sqrt(vhat) + 1e-8)

    problem.set_batch(None)
    fx = problem.f(x)
    gnorm = float(np.linalg.norm(problem.grad(x)))
    if history_f:
        history_f[-1] = fx
        history_g[-1] = gnorm
    elapsed = time.perf_counter() - t0
    c = problem.counters
    return OptimizationResult(
        x=x,
        f=fx,
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
