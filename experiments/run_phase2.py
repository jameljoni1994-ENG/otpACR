"""Phase-2 experiments: high-dim Krylov, convergence rates, multiprocess distributed."""

from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from cubic_reg.problems import Quadratic, LogSumExp
from cubic_reg.problems.ml import FiniteSumLogistic, make_synthetic_logistic
from cubic_reg.solvers import cr, krylov, accelerated, arc
from cubic_reg.subproblem import solve_cubic_subproblem, model_decrease


OUT = Path(__file__).resolve().parent / "results"


def high_dim_krylov(
    ns: Optional[List[int]] = None,
    krylov_dim: int = 30,
    max_iter: int = 12,
) -> dict:
    """Time/iteration vs n for Krylov; exact CR only up to a cutoff."""
    if ns is None:
        ns = [200, 500, 1000, 2000]
    rows = []
    for n in ns:
        p = Quadratic(n=n, condition=50.0, seed=0)
        x0 = np.ones(n)
        row = {"n": n}

        t0 = time.perf_counter()
        rk = krylov.minimize(
            p, x0=x0, M=1.0, eps=1e-5, krylov_dim=min(krylov_dim, n), max_iter=max_iter
        )
        row["krylov"] = {
            "time_sec": time.perf_counter() - t0,
            "nit": rk.nit,
            "grad_norm": rk.grad_norm,
            "n_hvp": rk.n_hvp,
            "sec_per_iter": rk.time_sec / max(rk.nit, 1),
        }

        if n <= 800:
            p2 = Quadratic(n=n, condition=50.0, seed=0)
            re = cr.minimize(p2, x0=x0, M=1.0, eps=1e-5, max_iter=max_iter)
            row["exact"] = {
                "time_sec": re.time_sec,
                "nit": re.nit,
                "grad_norm": re.grad_norm,
                "n_hess": re.n_hess,
                "sec_per_iter": re.time_sec / max(re.nit, 1),
            }
        rows.append(row)
        print(f"n={n}: krylov {row['krylov']['sec_per_iter']:.4f}s/it", flush=True)
    return {"rows": rows}


def rate_comparison(max_iter: int = 50) -> dict:
    """Compare CR vs Acc-CR gaps on LogSumExp; estimate empirical slopes."""
    p = LogSumExp(n=30, m=120, mu=0.01, seed=1)
    x0 = np.ones(p.dim)

    # Approximate f* by long ARC run
    warm = arc.minimize(p, x0=x0, M0=1.0, eps=1e-14, max_iter=400)
    f_star = warm.f
    x_ref = warm.x

    # eps < 0 disables early stop so we keep a full iteration budget for slope fits
    r_cr = cr.minimize(p, x0=x0, M=8.0, eps=-1.0, max_iter=max_iter)
    r_acc = accelerated.minimize(p, x0=x0, M=8.0, eps=-1.0, max_iter=max_iter)

    def pack(res):
        gap = np.maximum(np.asarray(res.history_f) - f_star, 1e-16)
        ks = np.arange(1, len(gap) + 1, dtype=float)
        valid = np.where(gap > 1e-12)[0]
        if valid.size >= 10:
            sl = valid[valid.size // 2 :]
            slope = float(np.polyfit(np.log(ks[sl]), np.log(gap[sl]), 1)[0])
        else:
            # fallback: all points above tiny floor
            mask = gap > 1e-14
            if mask.sum() >= 5:
                slope = float(np.polyfit(np.log(ks[mask]), np.log(gap[mask]), 1)[0])
            else:
                slope = float("nan")
        return {
            "nit": res.nit,
            "time_sec": res.time_sec,
            "grad_norm": res.grad_norm,
            "f_final": res.f,
            "f_star_est": f_star,
            "gap": gap.tolist(),
            "gap_k2": (gap * ks**2).tolist(),
            "gap_k3": (gap * ks**3).tolist(),
            "loglog_slope": slope,
            "dist_to_ref": float(np.linalg.norm(res.x - x_ref)),
        }

    out = {"cr": pack(r_cr), "accelerated": pack(r_acc)}
    print(
        f"rates: CR slope={out['cr']['loglog_slope']:.3f}, "
        f"Acc slope={out['accelerated']['loglog_slope']:.3f}",
        flush=True,
    )
    return out


def _local_grad_hess(payload: Tuple[np.ndarray, np.ndarray, np.ndarray, float]):
    """Worker: compute grad and hess on a data shard."""
    X, y, w, l2 = payload
    # local logistic oracles (no shared Problem object — picklable)
    z = y * (X @ w)
    sig = 1.0 / (1.0 + np.exp(np.clip(z, -50, 50)))
    nloc = len(y)
    g = -(X.T @ (y * sig)) / nloc + l2 * w
    d = sig * (1.0 - sig)
    H = (X.T @ (X * d[:, None])) / nloc + l2 * np.eye(w.shape[0])
    return g, H, nloc


def multiprocess_distributed_logistic(
    n_workers_list: Optional[List[int]] = None,
    n_samples: int = 8000,
    n_features: int = 80,
    max_iter: int = 20,
    M0: float = 1.0,
    eps: float = 1e-3,
    backend: str = "thread",
) -> dict:
    """Data-parallel ARC-like loop: shard data, aggregate grad/hess in parallel.

    backend='thread' is usually better on Windows for NumPy BLAS work (low spawn cost).
    backend='process' isolates CPU work but pays pickling/spawn overhead.
    """
    from concurrent.futures import ThreadPoolExecutor

    if n_workers_list is None:
        n_workers_list = [1, 2, 4]
    Pool = ProcessPoolExecutor if backend == "process" else ThreadPoolExecutor
    base = make_synthetic_logistic(n_samples=n_samples, n_features=n_features, seed=0)
    X, y, l2 = base.X, base.y, base.l2
    results = {}

    for nw in n_workers_list:
        shards = np.array_split(np.arange(n_samples), nw)
        w = np.zeros(n_features)
        M = M0
        hist_f, hist_g = [], []
        rejected = 0
        t0 = time.perf_counter()
        success = False

        with Pool(max_workers=nw) as pool:
            for k in range(max_iter):
                # full objective for monitor (serial, cheap relative to hess)
                base.set_batch(None)
                fx = base.f(w)
                gnorm = float(np.linalg.norm(base.grad(w)))
                hist_f.append(fx)
                hist_g.append(gnorm)
                if gnorm <= eps:
                    success = True
                    break

                payloads = [(X[idx], y[idx], w, l2) for idx in shards if len(idx) > 0]
                parts = list(pool.map(_local_grad_hess, payloads))
                # weighted average by local sample counts
                Ntot = sum(p[2] for p in parts)
                g = sum(p[0] * p[2] for p in parts) / Ntot
                H = sum(p[1] * p[2] for p in parts) / Ntot

                sol = solve_cubic_subproblem(g, H, M)
                if not sol.success:
                    M *= 2.0
                    rejected += 1
                    continue
                md = model_decrease(g, H, sol.s, M)
                if md <= 0:
                    M *= 2.0
                    rejected += 1
                    continue
                w_trial = w + sol.s
                f_trial = float(
                    np.mean(np.logaddexp(0.0, -(y * (X @ w_trial))))
                    + 0.5 * l2 * np.dot(w_trial, w_trial)
                )
                rho = (fx - f_trial) / md
                if rho >= 0.1:
                    w = w_trial
                    if rho >= 0.9:
                        M = max(1e-16, 0.5 * M)
                else:
                    rejected += 1
                    M *= 2.0

        elapsed = time.perf_counter() - t0
        results[f"workers_{nw}"] = {
            "nit": len(hist_f),
            "time_sec": elapsed,
            "grad_norm": hist_g[-1] if hist_g else None,
            "f": hist_f[-1] if hist_f else None,
            "success": success,
            "rejected_steps": rejected,
            "history_f": hist_f,
            "history_grad_norm": hist_g,
            "sec_per_iter": elapsed / max(len(hist_f), 1),
            "backend": backend,
        }
        print(f"{backend} workers={nw}: {elapsed:.3f}s, ||g||={hist_g[-1]:.3e}", flush=True)

    # speedup vs 1 worker
    t1 = results["workers_1"]["time_sec"]
    for key, val in results.items():
        val["speedup_vs_1"] = t1 / max(val["time_sec"], 1e-12)
    return results


def run_phase2() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    report = {
        "high_dim_krylov": high_dim_krylov(),
        "rate_comparison": rate_comparison(),
        "multiprocess_distributed": multiprocess_distributed_logistic(),
    }
    path = OUT / "phase2.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {path}")
    return report


if __name__ == "__main__":
    run_phase2()
