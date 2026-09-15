"""Krylov / Lanczos inexact cubic regularization."""

from __future__ import annotations

import time
from typing import List, Optional, Tuple

import numpy as np

from cubic_reg.metrics import OptimizationResult
from cubic_reg.problems.base import Problem
from cubic_reg.subproblem import solve_cubic_subproblem_reduced


def lanczos_tridiag(
    hvp,
    x: np.ndarray,
    g: np.ndarray,
    m: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build m-dimensional Krylov basis for H starting from g.

    Returns
    -------
    Q : (n, k) orthonormal basis (k <= m)
    T : (k, k) tridiagonal projected Hessian
    g_red : (k,) gradient in the basis (= ||g|| e_1)
    """
    n = g.shape[0]
    beta0 = float(np.linalg.norm(g))
    if beta0 < 1e-16:
        return np.zeros((n, 0)), np.zeros((0, 0)), np.zeros(0)

    Q = np.zeros((n, m))
    alpha = np.zeros(m)
    beta = np.zeros(m)

    q = g / beta0
    Q[:, 0] = q
    v = hvp(x, q)
    alpha[0] = float(q @ v)
    v = v - alpha[0] * q
    v = v - Q[:, :1] @ (Q[:, :1].T @ v)

    k_eff = 1
    for i in range(1, m):
        b = float(np.linalg.norm(v))
        beta[i - 1] = b
        if b < 1e-12:
            break
        q = v / b
        Q[:, i] = q
        v = hvp(x, q)
        alpha[i] = float(q @ v)
        v = v - alpha[i] * q - b * Q[:, i - 1]
        v = v - Q[:, : i + 1] @ (Q[:, : i + 1].T @ v)
        k_eff = i + 1

    Q = Q[:, :k_eff]
    T = np.diag(alpha[:k_eff])
    for i in range(k_eff - 1):
        T[i, i + 1] = beta[i]
        T[i + 1, i] = beta[i]
    g_red = np.zeros(k_eff)
    g_red[0] = beta0
    return Q, T, g_red


def stationarity_residual(hvp, x: np.ndarray, g: np.ndarray, s: np.ndarray, lam: float) -> float:
    """||(H + λ I) s + g|| via HVP."""
    return float(np.linalg.norm(hvp(x, s) + lam * s + g))


def theta_schedule(k: int, theta0: float = 0.5, power: float = 1.0) -> float:
    """Decreasing inexactness tolerance θ_k = θ0 / (k+1)^p (cartis-style flavor)."""
    return float(theta0 / ((k + 1) ** power))


def solve_inexact_step(
    problem: Problem,
    x: np.ndarray,
    g: np.ndarray,
    M: float,
    m_start: int,
    m_max: int,
    theta: float,
) -> Tuple[np.ndarray, float, int, float, bool]:
    """Grow Krylov dimension until residual ≤ θ ||g|| or m_max reached.

    Returns s, lam, m_used, residual, satisfied.
    """
    gnorm = float(np.linalg.norm(g))
    m = max(1, min(m_start, problem.dim, m_max))
    m_cap = min(m_max, problem.dim)
    s = np.zeros_like(g)
    lam = 0.0
    resid = gnorm
    satisfied = False

    while True:
        Q, T, g_red = lanczos_tridiag(problem.hvp, x, g, m)
        if Q.shape[1] == 0:
            return s, lam, m, 0.0, True
        sol, _ = solve_cubic_subproblem_reduced(g_red, T, M, Q=Q)
        if not sol.success:
            return s, lam, m, resid, False
        s, lam = sol.s, sol.lam
        resid = stationarity_residual(problem.hvp, x, g, s, lam)
        satisfied = resid <= theta * gnorm + 1e-14
        if satisfied or m >= m_cap:
            break
        m = min(m_cap, max(m + 5, int(m * 1.5)))

    return s, lam, m, resid, satisfied


def minimize(
    problem: Problem,
    x0: Optional[np.ndarray] = None,
    M: float = 1.0,
    eps: float = 1e-6,
    max_iter: int = 200,
    krylov_dim: int = 20,
    m_max: Optional[int] = None,
    inexact_tol: float = 0.5,
    adaptive_tol: bool = True,
    tol_power: float = 1.0,
    verbose: bool = False,
) -> OptimizationResult:
    """Cubic regularization with adaptive inexact Lanczos subproblem solves.

    Accepts s when ||(H+λI)s + g|| ≤ θ_k ||g||, growing the Krylov dimension
    until the tolerance is met (or m_max is hit).
    """
    problem.reset_counters()
    x = np.zeros(problem.dim) if x0 is None else np.asarray(x0, dtype=float).copy()
    history_f: List[float] = []
    history_g: List[float] = []
    history_M: List[float] = []
    history_resid: List[float] = []
    history_m: List[int] = []
    history_theta: List[float] = []

    t0 = time.perf_counter()
    success = False
    message = "max_iter reached"
    gnorm = np.inf
    fx = problem.f(x)
    m_cap = m_max if m_max is not None else min(problem.dim, max(krylov_dim * 4, 80))

    for k in range(max_iter):
        g = problem.grad(x)
        gnorm = float(np.linalg.norm(g))
        history_f.append(fx)
        history_g.append(gnorm)
        history_M.append(M)

        theta = theta_schedule(k, inexact_tol, tol_power) if adaptive_tol else inexact_tol
        history_theta.append(theta)

        if verbose:
            print(f"Krylov-CR iter {k}: f={fx:.6e} ||g||={gnorm:.6e} θ={theta:.3e}")

        if gnorm <= eps:
            success = True
            message = "gradient norm below eps"
            break

        s, lam, m_used, resid, ok = solve_inexact_step(
            problem, x, g, M, krylov_dim, m_cap, theta
        )
        history_m.append(m_used)
        history_resid.append(resid)
        if not ok and np.linalg.norm(s) == 0:
            message = "inexact subproblem failed"
            break

        x = x + s
        fx = problem.f(x)

    elapsed = time.perf_counter() - t0
    c = problem.counters
    res = OptimizationResult(
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
    # Attach extra diagnostics without breaking dataclass consumers
    res.__dict__["history_resid"] = history_resid
    res.__dict__["history_m"] = history_m
    res.__dict__["history_theta"] = history_theta
    return res
