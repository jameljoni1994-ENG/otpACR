"""L-BFGS quasi-Newton cubic regularization."""

from __future__ import annotations

import time
from collections import deque
from typing import Deque, Optional, Tuple

import numpy as np

from cubic_reg.metrics import OptimizationResult
from cubic_reg.problems.base import Problem
from cubic_reg.subproblem import solve_cubic_subproblem


class LBFGSHessian:
    """Compact limited-memory BFGS Hessian approximation (dense for small m*n).

    Stores pairs (s, y) and can form a dense matrix H_k for the cubic subproblem
    when dimension is moderate, or apply H_k @ v via two-loop recursion style
    for the Hessian (forward BFGS).
    """

    def __init__(self, n: int, memory: int = 10):
        self.n = n
        self.memory = memory
        self.S: Deque[np.ndarray] = deque(maxlen=memory)
        self.Y: Deque[np.ndarray] = deque(maxlen=memory)
        self.gamma = 1.0

    def update(self, s: np.ndarray, y: np.ndarray) -> None:
        ys = float(y @ s)
        if ys <= 1e-12:
            return
        self.S.append(s.copy())
        self.Y.append(y.copy())
        self.gamma = ys / float(y @ y)

    def matvec(self, v: np.ndarray) -> np.ndarray:
        """Apply approximate Hessian B_k to v (compact BFGS)."""
        if not self.S:
            return v / self.gamma if self.gamma > 0 else v

        # Compact representation: B = gamma^{-1} I - W M W^T  (standard L-BFGS Hessian form)
        # Use recursive formula for Hessian-vector:
        # Start from B0 = I/gamma, apply BFGS updates sequentially.
        Bv = v / self.gamma
        # Rebuild via successive BFGS Hessian updates (O(m^2 n) ok for small m)
        # Better: use explicit dense if n small; else recursive
        pairs = list(zip(self.S, self.Y))
        # Reset and apply updates to vector via Sherman-style accumulation is messy;
        # form dense B when n is manageable else use loop of BFGS updates on vector.
        return self._bfgs_matvec_from_scratch(v)

    def _bfgs_matvec_from_scratch(self, v: np.ndarray) -> np.ndarray:
        # Apply sequence of BFGS Hessian updates to vector starting from B0=I/gamma
        # B+ = B + yy^T/(y^T s) - (Bs)(Bs)^T / (s^T B s)
        # We maintain action on v by also tracking... actually need full matrix for cubic.
        return self.to_dense() @ v

    def to_dense(self) -> np.ndarray:
        B = np.eye(self.n) / max(self.gamma, 1e-16)
        for s, y in zip(self.S, self.Y):
            Bs = B @ s
            ys = float(y @ s)
            sBs = float(s @ Bs)
            if ys <= 1e-16 or sBs <= 1e-16:
                continue
            B = B + np.outer(y, y) / ys - np.outer(Bs, Bs) / sBs
        return B


def minimize(
    problem: Problem,
    x0: Optional[np.ndarray] = None,
    M: float = 1.0,
    eps: float = 1e-6,
    max_iter: int = 300,
    memory: int = 10,
    adaptive_M: bool = True,
    eta1: float = 0.1,
    gamma1: float = 0.5,
    gamma2: float = 2.0,
    verbose: bool = False,
) -> OptimizationResult:
    """Cubic regularization with L-BFGS Hessian approximation."""
    problem.reset_counters()
    x = np.zeros(problem.dim) if x0 is None else np.asarray(x0, dtype=float).copy()
    lbfgs = LBFGSHessian(problem.dim, memory=memory)
    history_f: list[float] = []
    history_g: list[float] = []
    history_M: list[float] = []
    rejected = 0

    t0 = time.perf_counter()
    success = False
    message = "max_iter reached"
    gnorm = np.inf
    fx = problem.f(x)
    g = problem.grad(x)

    for k in range(max_iter):
        gnorm = float(np.linalg.norm(g))
        history_f.append(fx)
        history_g.append(gnorm)
        history_M.append(M)

        if verbose:
            print(f"QN-CR iter {k}: f={fx:.6e} ||g||={gnorm:.6e} M={M:.3e}")

        if gnorm <= eps:
            success = True
            message = "gradient norm below eps"
            break

        B = lbfgs.to_dense()
        sol = solve_cubic_subproblem(g, B, M)
        if not sol.success:
            message = sol.message
            break

        s = sol.s
        md = sol.model_decrease
        x_trial = x + s
        f_trial = problem.f(x_trial)
        g_trial = problem.grad(x_trial)

        if adaptive_M and md > 0:
            rho = (fx - f_trial) / md
            if rho < eta1:
                rejected += 1
                M = M * gamma2
                continue
            if rho > 0.9:
                M = max(1e-16, M * gamma1)

        # L-BFGS update
        y = g_trial - g
        lbfgs.update(s, y)

        x = x_trial
        fx = f_trial
        g = g_trial

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


def minimize_lbfgs_plain(
    problem: Problem,
    x0: Optional[np.ndarray] = None,
    eps: float = 1e-6,
    max_iter: int = 300,
    memory: int = 10,
    verbose: bool = False,
) -> OptimizationResult:
    """Standard L-BFGS with Armijo line search (baseline without cubic term)."""
    from scipy.optimize import minimize as sp_minimize

    problem.reset_counters()
    x0 = np.zeros(problem.dim) if x0 is None else np.asarray(x0, dtype=float).copy()
    history_f: list[float] = []
    history_g: list[float] = []

    def fun(x):
        return problem.f(x)

    def jac(x):
        return problem.grad(x)

    t0 = time.perf_counter()
    xs = [x0.copy()]

    def callback(xk):
        history_f.append(problem.f(xk))
        history_g.append(float(np.linalg.norm(problem.grad(xk))))
        xs[0] = xk.copy()

    # seed history
    history_f.append(problem.f(x0))
    history_g.append(float(np.linalg.norm(problem.grad(x0))))

    res = sp_minimize(
        fun,
        x0,
        jac=jac,
        method="L-BFGS-B",
        options={"maxiter": max_iter, "gtol": eps},
        callback=callback,
    )
    elapsed = time.perf_counter() - t0
    c = problem.counters
    return OptimizationResult(
        x=res.x,
        f=float(res.fun),
        grad_norm=float(np.linalg.norm(problem.grad(res.x))),
        nit=res.nit,
        success=bool(res.success),
        message=str(res.message),
        time_sec=elapsed,
        n_f=c.f,
        n_grad=c.grad,
        n_hess=c.hess,
        n_hvp=c.hvp,
        history_f=history_f,
        history_grad_norm=history_g,
        f_star=problem.f_star,
    )
