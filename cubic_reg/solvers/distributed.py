"""Simulated distributed ARC with low-rank Hessian sketching."""

from __future__ import annotations

import time
from typing import List, Optional, Tuple

import numpy as np

from cubic_reg.metrics import OptimizationResult
from cubic_reg.problems.base import Problem
from cubic_reg.subproblem import model_decrease, solve_cubic_subproblem
from cubic_reg.solvers.arc import minimize as arc_minimize


def sketch_hessian(H: np.ndarray, rank: int, rng: np.random.Generator) -> Tuple[np.ndarray, int]:
    """Randomized range-finder sketch of H with a ridge fallback on the orthogonal complement.

    Pure low-rank QBQ^T can destroy curvature and stall ARC. We keep the sketched
    action on range(Q) and a scaled identity on the complement:
        H ≈ Q B Q^T + σ (I - Q Q^T)
    where σ is the median eigenvalue of B (robust curvature scale).
    """
    n = H.shape[0]
    r = min(max(rank, 1), n)
    Omega = rng.standard_normal((n, r))
    Y = H @ Omega
    # one power iteration improves spectrum capture
    Y = H @ (H @ Y)
    Q, _ = np.linalg.qr(Y)
    B = Q.T @ H @ Q
    evals = np.linalg.eigvalsh(0.5 * (B + B.T))
    sigma = float(np.median(np.abs(evals))) + 1e-8
    H_approx = Q @ B @ Q.T + sigma * (np.eye(n) - Q @ Q.T)
    bytes_proxy = (n * r + r * r) * 8
    return H_approx, bytes_proxy


def split_data_indices(N: int, n_workers: int) -> List[np.ndarray]:
    idx = np.arange(N)
    return [chunk for chunk in np.array_split(idx, n_workers)]


def minimize(
    problem: Problem,
    x0: Optional[np.ndarray] = None,
    M0: float = 1.0,
    eps: float = 1e-6,
    max_iter: int = 200,
    n_workers: int = 4,
    sketch_rank: int = 10,
    network_latency_ms: float = 0.0,
    seed: int = 0,
    verbose: bool = False,
) -> OptimizationResult:
    """Centralized ARC but Hessian is replaced by sketched aggregate (simulates distributed).

    For generic problems without data shards, we sketch the full Hessian and
    optionally sleep to emulate latency. Bytes transferred are tracked in message.
    """
    problem.reset_counters()
    rng = np.random.default_rng(seed)
    x = np.zeros(problem.dim) if x0 is None else np.asarray(x0, dtype=float).copy()
    M = float(M0)
    history_f: list[float] = []
    history_g: list[float] = []
    history_M: list[float] = []
    rejected = 0
    total_bytes = 0
    # Simulated latency cost accumulated into wall time via sleep optional
    latency_sec = network_latency_ms / 1000.0

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
            print(f"Dist-ARC iter {k}: f={fx:.6e} ||g||={gnorm:.6e} workers={n_workers}")

        if gnorm <= eps:
            success = True
            message = f"converged; bytes~{total_bytes}"
            break

        H = problem.hess(x)
        # Simulate per-worker local sketch then average (here: one sketch of full H)
        # Multiplying communication by n_workers models all-reduce of sketches
        H_sk, nbytes = sketch_hessian(H, sketch_rank, rng)
        total_bytes += nbytes * n_workers
        if latency_sec > 0:
            time.sleep(latency_sec)

        sol = solve_cubic_subproblem(g, H_sk, M)
        if not sol.success:
            M *= 2.0
            rejected += 1
            continue

        s = sol.s
        md = model_decrease(g, H_sk, s, M)
        if md <= 0:
            M *= 2.0
            rejected += 1
            continue

        x_trial = x + s
        f_trial = problem.f(x_trial)
        rho = (fx - f_trial) / md
        if rho >= 0.1:
            x = x_trial
            fx = f_trial
            if rho >= 0.9:
                M = max(1e-16, 0.5 * M)
        else:
            rejected += 1
            M *= 2.0

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


def speedup_experiment(
    problem: Problem,
    worker_counts: Optional[List[int]] = None,
    **kwargs,
) -> List[OptimizationResult]:
    """Run distributed ARC for several worker counts (for scalability plots)."""
    if worker_counts is None:
        worker_counts = [1, 2, 4, 8]
    results = []
    for w in worker_counts:
        # Fresh copy of starting point each run
        res = minimize(problem, n_workers=w, **kwargs)
        results.append(res)
    return results
