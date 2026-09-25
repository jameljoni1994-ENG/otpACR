"""L-BFGS quasi-Newton cubic regularization."""

from __future__ import annotations

import time
from collections import deque
from typing import Deque, List, Optional

import numpy as np

from cubic_reg.metrics import OptimizationResult
from cubic_reg.problems.base import Problem
from cubic_reg.subproblem import solve_cubic_subproblem
from cubic_reg.solvers.krylov import solve_inexact_step, theta_schedule


class LBFGSHessian:
    """Limited-memory BFGS Hessian approximation.

    Supports matrix-free ``matvec`` via the compact Byrd–Nocedal–Schnabel form
    and an optional dense ``to_dense`` path for small-dimension reference solves.
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
        """Apply approximate Hessian B_k to v (compact BFGS, O(m n + m²))."""
        v = np.asarray(v, dtype=float)
        gamma = max(self.gamma, 1e-16)
        if not self.S:
            return v / gamma

        S = np.column_stack(list(self.S))
        Y = np.column_stack(list(self.Y))
        m = S.shape[1]
        B0S = S / gamma
        STY = S.T @ Y
        D = np.diag(np.diag(STY))
        L = np.tril(STY, k=-1)
        STS = (S.T @ S) / gamma
        Mid = np.block([[STS, L], [L.T, -D]])
        rhs = np.concatenate([B0S.T @ v, Y.T @ v])
        try:
            alpha = np.linalg.solve(Mid, rhs)
        except np.linalg.LinAlgError:
            alpha = np.linalg.lstsq(Mid, rhs, rcond=None)[0]
        return v / gamma - (B0S @ alpha[:m] + Y @ alpha[m:])

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
    matrix_free: bool = True,
    krylov_dim: int = 20,
    m_max: Optional[int] = None,
    inexact_tol: float = 0.5,
    adaptive_tol: bool = True,
    tol_power: float = 1.0,
    verbose: bool = False,
) -> OptimizationResult:
    """Cubic regularization with L-BFGS Hessian approximation.

    By default solves the cubic subproblem in a matrix-free way via Krylov on
    ``LBFGSHessian.matvec``. Set ``matrix_free=False`` to form dense B_k
    (suitable only for small n).
    """
    problem.reset_counters()
    x = np.zeros(problem.dim) if x0 is None else np.asarray(x0, dtype=float).copy()
    lbfgs = LBFGSHessian(problem.dim, memory=memory)
    history_f: List[float] = []
    history_g: List[float] = []
    history_M: List[float] = []
    history_resid: List[float] = []
    history_m: List[int] = []
    history_theta: List[float] = []
    rejected = 0

    t0 = time.perf_counter()
    success = False
    message = "max_iter reached"
    gnorm = np.inf
    fx = problem.f(x)
    g = problem.grad(x)
    m_cap = m_max if m_max is not None else min(problem.dim, max(krylov_dim * 4, 80))

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
            history_theta.append(0.0)
            history_m.append(0)
            history_resid.append(0.0)
            break

        if matrix_free:
            theta = theta_schedule(k, inexact_tol, tol_power) if adaptive_tol else inexact_tol
            history_theta.append(theta)
            hvp_B = lambda _x, v, _lbfgs=lbfgs: _lbfgs.matvec(v)
            s, lam, m_used, resid, ok = solve_inexact_step(
                problem, x, g, M, krylov_dim, m_cap, theta, hvp=hvp_B
            )
            history_m.append(m_used)
            history_resid.append(resid)
            if not ok and np.linalg.norm(s) == 0:
                message = "inexact subproblem failed"
                break
            Bs = lbfgs.matvec(s)
            sn = float(np.linalg.norm(s))
            md = float(-(g @ s + 0.5 * s @ Bs + (M / 6.0) * sn**3))
        else:
            B = lbfgs.to_dense()
            sol = solve_cubic_subproblem(g, B, M)
            if not sol.success:
                message = sol.message
                history_theta.append(0.0)
                history_m.append(0)
                history_resid.append(0.0)
                break
            s = sol.s
            md = sol.model_decrease
            history_theta.append(0.0)
            history_m.append(problem.dim)
            history_resid.append(0.0)

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
        elif adaptive_M and md <= 0:
            rejected += 1
            M = M * gamma2
            continue

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
        history_resid=history_resid,
        history_m=history_m,
        history_theta=history_theta,
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
