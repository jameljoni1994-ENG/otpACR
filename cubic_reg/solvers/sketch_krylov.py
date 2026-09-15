"""Hybrid sketched + Krylov cubic / ARC (matrix-free communication-friendly)."""

from __future__ import annotations

import time
from typing import Optional, Tuple

import numpy as np

from cubic_reg.metrics import OptimizationResult
from cubic_reg.problems.base import Problem
from cubic_reg.subproblem import model_decrease, solve_cubic_subproblem_reduced
from cubic_reg.solvers.krylov import lanczos_tridiag, solve_inexact_step, theta_schedule


def hvp_sketch_basis(
    hvp,
    x: np.ndarray,
    rank: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Build orthonormal sketch basis from random HVP probes (no dense H).

    Returns Q (n,r), B ≈ Q^T H Q, and bytes proxy for communicating Q and B.
    """
    n = x.shape[0]
    r = min(max(rank, 1), n)
    Omega = rng.standard_normal((n, r))
    Y = np.column_stack([hvp(x, Omega[:, j]) for j in range(r)])
    # one power iteration
    Y = np.column_stack([hvp(x, Y[:, j]) for j in range(r)])
    Q, _ = np.linalg.qr(Y)
    B = np.zeros((r, r))
    for j in range(r):
        Hj = hvp(x, Q[:, j])
        B[:, j] = Q.T @ Hj
    B = 0.5 * (B + B.T)
    bytes_proxy = (n * r + r * r) * 8
    return Q, B, bytes_proxy


def minimize(
    problem: Problem,
    x0: Optional[np.ndarray] = None,
    M0: float = 1.0,
    eps: float = 1e-6,
    max_iter: int = 200,
    sketch_rank: int = 20,
    krylov_dim: int = 15,
    use_krylov_refine: bool = True,
    seed: int = 0,
    eta1: float = 0.1,
    eta2: float = 0.9,
    gamma1: float = 0.5,
    gamma2: float = 2.0,
    verbose: bool = False,
) -> OptimizationResult:
    """ARC with HVP sketch; optional Krylov refinement of the cubic step.

    Pipeline per iteration:
      1) Build sketch basis Q via HVPs (communicable factors).
      2) Solve cubic subproblem in span(Q).
      3) Optionally refine with inexact Krylov in the full space starting from g.
    """
    problem.reset_counters()
    rng = np.random.default_rng(seed)
    x = np.zeros(problem.dim) if x0 is None else np.asarray(x0, dtype=float).copy()
    M = float(M0)
    history_f = []
    history_g = []
    history_M = []
    rejected = 0
    total_bytes = 0

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
            print(f"Sketch+Krylov ARC {k}: f={fx:.6e} ||g||={gnorm:.6e} M={M:.2e}")

        if gnorm <= eps:
            success = True
            message = f"converged; bytes~{total_bytes}"
            break

        Qs, B, nbytes = hvp_sketch_basis(problem.hvp, x, sketch_rank, rng)
        total_bytes += nbytes
        g_red = Qs.T @ g
        sol, _ = solve_cubic_subproblem_reduced(g_red, B, M, Q=Qs)
        if not sol.success:
            M = min(1e16, M * gamma2)
            rejected += 1
            continue
        s = sol.s

        if use_krylov_refine:
            theta = theta_schedule(k, 0.5, 1.0)
            s_ref, _, _, _, ok = solve_inexact_step(
                problem, x, g, M, krylov_dim, min(problem.dim, krylov_dim * 3), theta
            )
            if ok and np.linalg.norm(s_ref) > 0:
                s = s_ref

        # Model decrease via HVP
        Hs = problem.hvp(x, s)
        sn = float(np.linalg.norm(s))
        md = float(-(g @ s + 0.5 * s @ Hs + (M / 6.0) * sn**3))
        if md <= 0:
            M = min(1e16, M * gamma2)
            rejected += 1
            continue

        x_trial = x + s
        f_trial = problem.f(x_trial)
        rho = (fx - f_trial) / md
        if rho >= eta1:
            x = x_trial
            fx = f_trial
            if rho >= eta2:
                M = max(1e-16, gamma1 * M)
        else:
            rejected += 1
            M = min(1e16, gamma2 * M)

    elapsed = time.perf_counter() - t0
    if not success:
        message = f"{message}; bytes~{total_bytes}"
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
