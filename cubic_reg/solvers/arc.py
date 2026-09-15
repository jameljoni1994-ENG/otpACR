"""Adaptive Regularization using Cubics (ARC)."""

from __future__ import annotations

import time
from typing import Literal, Optional

import numpy as np

from cubic_reg.metrics import OptimizationResult
from cubic_reg.problems.base import Problem
from cubic_reg.subproblem import model_decrease, solve_cubic_subproblem
from cubic_reg.solvers.krylov import lanczos_tridiag
from cubic_reg.subproblem import solve_cubic_subproblem_reduced


def minimize(
    problem: Problem,
    x0: Optional[np.ndarray] = None,
    M0: float = 1.0,
    eps: float = 1e-6,
    max_iter: int = 500,
    eta1: float = 0.1,
    eta2: float = 0.9,
    gamma1: float = 0.5,
    gamma2: float = 2.0,
    M_min: float = 1e-16,
    M_max: float = 1e16,
    mode: Literal["exact", "krylov"] = "exact",
    krylov_dim: int = 20,
    verbose: bool = False,
) -> OptimizationResult:
    """ARC with ratio-test adaptation of M.

    rho = (f_k - f_{k+1}) / (f_k - m_k(s_k))
    Accept if rho >= eta1; shrink M if rho >= eta2, grow if rejected/small rho.
    """
    problem.reset_counters()
    x = np.zeros(problem.dim) if x0 is None else np.asarray(x0, dtype=float).copy()
    M = float(M0)
    history_f: list[float] = []
    history_g: list[float] = []
    history_M: list[float] = []
    rejected = 0

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
            print(f"ARC iter {k}: f={fx:.6e} ||g||={gnorm:.6e} M={M:.3e}")

        if gnorm <= eps:
            success = True
            message = "gradient norm below eps"
            break

        # Solve cubic subproblem
        if mode == "exact":
            H = problem.hess(x)
            sol = solve_cubic_subproblem(g, H, M)
            pred = sol.model_decrease
            # recompute with stored H for ratio
            if not sol.success:
                M = min(M_max, M * gamma2)
                rejected += 1
                continue
            s = sol.s
            md = model_decrease(g, H, s, M)
        else:
            Q, T, g_red = lanczos_tridiag(problem.hvp, x, g, krylov_dim)
            if Q.shape[1] == 0:
                success = True
                message = "zero gradient"
                break
            sol, s_red = solve_cubic_subproblem_reduced(g_red, T, M, Q=Q)
            if not sol.success:
                M = min(M_max, M * gamma2)
                rejected += 1
                continue
            s = sol.s
            # predicted decrease in reduced model
            md = sol.model_decrease
            # Also estimate with full HVP for better ratio
            Hs = problem.hvp(x, s)
            sn = float(np.linalg.norm(s))
            md = float(-(g @ s + 0.5 * s @ Hs + (M / 6.0) * sn**3))

        if md <= 0:
            M = min(M_max, M * gamma2)
            rejected += 1
            continue

        x_trial = x + s
        f_trial = problem.f(x_trial)
        actual = fx - f_trial
        rho = actual / md if md > 0 else -np.inf

        if rho >= eta1:
            x = x_trial
            fx = f_trial
            if rho >= eta2:
                M = max(M_min, gamma1 * M)
        else:
            rejected += 1
            M = min(M_max, gamma2 * M)

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
        rejected_steps=rejected,
        history_f=history_f,
        history_grad_norm=history_g,
        history_M=history_M,
        f_star=problem.f_star,
    )
