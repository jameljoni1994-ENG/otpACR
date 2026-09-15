"""Third-order tensor method with quartic regularization (exploratory)."""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from cubic_reg.metrics import OptimizationResult
from cubic_reg.problems.base import Problem
from cubic_reg.subproblem import solve_cubic_subproblem


def finite_diff_tensor_directional(
    hess_fn,
    x: np.ndarray,
    s: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    """Approximate T[x](s,s) via finite differences on the Hessian: (H(x+h s)-H(x-h s))/(2h) @ s."""
    nrm = np.linalg.norm(s)
    if nrm < 1e-16:
        return np.zeros_like(s)
    h = eps / nrm
    Hp = hess_fn(x + h * s)
    Hm = hess_fn(x - h * s)
    return ((Hp - Hm) / (2.0 * h)) @ s


def solve_tensor_subproblem_approx(
    g: np.ndarray,
    H: np.ndarray,
    x: np.ndarray,
    hess_fn,
    L: float,
    max_inner: int = 20,
) -> np.ndarray:
    """Approximate p=3 step by nested cubic regularization / fixed-point.

    Model: g^T s + 1/2 s^T H s + 1/6 T[s,s,s] + (L/24)||s||^4
    We approximate T[s,s] via FD and solve a cubic-like model iteratively:
      (H + 0.5 * T_dir(s) + (L/6)||s||^2 I) s = -g
    using cubic regularization with M_eff = L * ||s|| as a proxy.
    """
    s = solve_cubic_subproblem(g, H, M=max(L, 1e-6)).s
    for _ in range(max_inner):
        Tdir = finite_diff_tensor_directional(hess_fn, x, s)
        # Effective Hessian: H + 0.5 * symmetrized directional tensor action
        # Use cubic with M = L * ||s|| (since quartic ~ cubic * ||s||)
        sn = float(np.linalg.norm(s))
        M_eff = max(L * max(sn, 1e-8), 1e-8)
        H_eff = H + 0.5 * (np.outer(Tdir, s) + np.outer(s, Tdir)) / max(sn**2, 1e-16)
        # Symmetrize
        H_eff = 0.5 * (H_eff + H_eff.T)
        s_new = solve_cubic_subproblem(g, H_eff, M=M_eff).s
        if np.linalg.norm(s_new - s) < 1e-10 * (1 + np.linalg.norm(s)):
            return s_new
        s = s_new
    return s


def minimize(
    problem: Problem,
    x0: Optional[np.ndarray] = None,
    L: float = 1.0,
    eps: float = 1e-6,
    max_iter: int = 100,
    verbose: bool = False,
) -> OptimizationResult:
    """Exploratory third-order method; compare cost vs CR via hess/FD calls."""
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

    def hess_fn(z: np.ndarray) -> np.ndarray:
        return problem.hess(z)

    for k in range(max_iter):
        g = problem.grad(x)
        gnorm = float(np.linalg.norm(g))
        history_f.append(fx)
        history_g.append(gnorm)
        history_M.append(L)

        if verbose:
            print(f"Tensor iter {k}: f={fx:.6e} ||g||={gnorm:.6e}")

        if gnorm <= eps:
            success = True
            message = "gradient norm below eps"
            break

        H = problem.hess(x)
        s = solve_tensor_subproblem_approx(g, H, x, hess_fn, L)
        x = x + s
        fx = problem.f(x)

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
