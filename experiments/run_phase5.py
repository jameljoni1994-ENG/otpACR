"""Phase-5: theta_k policies on nonconvex problems + sketched communication timing."""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cubic_reg.problems import Rosenbrock, Rastrigin, MatrixFreeQuadratic
from cubic_reg.solvers import krylov, sketch_krylov
from cubic_reg.solvers.sketch_krylov import hvp_sketch_basis

FIGDIR = Path(__file__).resolve().parent / "figures"
RESDIR = Path(__file__).resolve().parent / "results"


def _save(fig, name: str) -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    path = FIGDIR / name
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


def theta_policy_study() -> dict:
    """Compare fixed / 1/k / 1/k^2 inexactness on nonconvex problems."""
    problems = {
        "rosenbrock": (Rosenbrock(n=10), np.full(10, -1.2)),
        "rastrigin": (Rastrigin(n=15), 0.5 * np.ones(15)),
    }
    policies = {
        "fixed": dict(adaptive_tol=False, inexact_tol=0.5, tol_power=0.0),
        "1_over_k": dict(adaptive_tol=True, inexact_tol=0.5, tol_power=1.0),
        "1_over_k2": dict(adaptive_tol=True, inexact_tol=0.5, tol_power=2.0),
    }
    out = {}
    for pname, (prob, x0) in problems.items():
        out[pname] = {}
        for pol, kwargs in policies.items():
            r = krylov.minimize(
                prob,
                x0=x0.copy(),
                M=10.0 if pname == "rosenbrock" else 1.0,
                eps=1e-5,
                max_iter=80,
                krylov_dim=12,
                m_max=40,
                **kwargs,
            )
            out[pname][pol] = {
                "nit": r.nit,
                "f": r.f,
                "grad_norm": r.grad_norm,
                "n_hvp": r.n_hvp,
                "time_sec": r.time_sec,
                "history_f": r.history_f,
                "history_grad_norm": r.history_grad_norm,
                "history_m": getattr(r, "history_m", []),
                "success": r.success,
            }
            print(
                f"{pname}/{pol}: nit={r.nit} f={r.f:.4e} ||g||={r.grad_norm:.2e} hvp={r.n_hvp}",
                flush=True,
            )
    return out


def fig_theta_policies(data: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, pname in zip(axes, data.keys()):
        for pol, res in data[pname].items():
            ax.semilogy(np.maximum(res["history_grad_norm"], 1e-16), label=pol)
        ax.set_title(pname)
        ax.set_xlabel("iteration")
        ax.set_ylabel(r"$\|\nabla f\|$")
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.3)
    fig.suptitle("Inexactness policies on nonconvex problems")
    _save(fig, "16_theta_policies.png")


def _worker_local_sketch(payload):
    """Process worker: build local HVP sketch factors from a diagonal+rank1 shard proxy.

    payload = (n, seed_offset, rank, x, diag, u, rho)
    Simulates a worker holding a local operator and returning (Q,B) factors.
    """
    n, seed_offset, rank, x, diag, u, rho = payload

    def hvp(_x, v):
        return diag * v + rho * u * (u @ v)

    rng = np.random.default_rng(seed_offset)
    Q, B, nbytes = hvp_sketch_basis(hvp, x, rank, rng)
    return Q, B, nbytes


def sketch_comm_timing(
    n: int = 2000,
    ranks=(10, 20, 40),
    workers=(1, 2, 4),
) -> dict:
    """Measure wall time to build/aggregate sketches across processes."""
    rng = np.random.default_rng(0)
    diag = np.linspace(1.0, 50.0, n)
    u = rng.standard_normal(n)
    u /= np.linalg.norm(u)
    rho = 10.0
    x = np.zeros(n)
    rows = []

    for rank in ranks:
        for nw in workers:
            payloads = [
                (n, 1000 + w, rank, x, diag, u * (1.0 + 0.01 * w), rho) for w in range(nw)
            ]
            t0 = time.perf_counter()
            if nw == 1:
                parts = [_worker_local_sketch(payloads[0])]
            else:
                with ProcessPoolExecutor(max_workers=nw) as pool:
                    parts = list(pool.map(_worker_local_sketch, payloads))
            # Aggregate: average B in a common Q from worker 0 (proxy allreduce)
            Q0 = parts[0][0]
            B_avg = sum(Q0.T @ (p[0] @ p[1] @ p[0].T) @ Q0 for p in parts) / nw
            elapsed = time.perf_counter() - t0
            bytes_total = sum(p[2] for p in parts)
            rows.append(
                {
                    "n": n,
                    "rank": rank,
                    "workers": nw,
                    "time_sec": elapsed,
                    "bytes_total": bytes_total,
                    "B_frob": float(np.linalg.norm(B_avg)),
                }
            )
            print(
                f"comm n={n} r={rank} w={nw}: {elapsed:.3f}s bytes={bytes_total}",
                flush=True,
            )
    return {"rows": rows}


def fig_comm(data: dict) -> None:
    rows = data["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for rank in sorted({r["rank"] for r in rows}):
        sub = [r for r in rows if r["rank"] == rank]
        ws = [r["workers"] for r in sub]
        ts = [r["time_sec"] for r in sub]
        bs = [r["bytes_total"] for r in sub]
        axes[0].plot(ws, ts, "o-", label=f"rank={rank}")
        axes[1].plot(ws, bs, "s-", label=f"rank={rank}")
    axes[0].set_xlabel("workers")
    axes[0].set_ylabel("wall time (s)")
    axes[0].set_title("Sketch build+aggregate time")
    axes[1].set_xlabel("workers")
    axes[1].set_ylabel("bytes proxy")
    axes[1].set_title("Communication volume")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    _save(fig, "17_sketch_comm.png")


def main() -> None:
    RESDIR.mkdir(parents=True, exist_ok=True)
    theta = theta_policy_study()
    comm = sketch_comm_timing()
    payload = {"theta_policies": theta, "sketch_comm": comm}
    path = RESDIR / "phase5.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {path}")
    fig_theta_policies(theta)
    fig_comm(comm)
    print("Phase-5 done.")


if __name__ == "__main__":
    main()
