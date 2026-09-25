"""Phase-11: matrix-free L-BFGS cubic vs dense B_k (Theorem mf-qn support)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cubic_reg.problems import Quadratic, MatrixFreeQuadratic
from cubic_reg.solvers import quasi_newton
from cubic_reg.solvers.quasi_newton import LBFGSHessian

FIGDIR = Path(__file__).resolve().parent / "figures"
RESDIR = Path(__file__).resolve().parent / "results"


def _save(fig, name: str) -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    path = FIGDIR / name
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


def matvec_agreement(n: int = 80, memory: int = 8, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    B = LBFGSHessian(n, memory=memory)
    for _ in range(memory):
        s = rng.standard_normal(n)
        y = s + 0.05 * rng.standard_normal(n)
        if y @ s <= 0:
            y = np.abs(y) * np.sign(s) + s
        B.update(s, y)
    errs = []
    for _ in range(20):
        v = rng.standard_normal(n)
        errs.append(float(np.linalg.norm(B.matvec(v) - B.to_dense() @ v)))
    return {"n": n, "memory": memory, "max_abs_err": max(errs), "mean_abs_err": float(np.mean(errs))}


def dense_vs_mf_solve(n: int = 40, seed: int = 1) -> dict:
    p = Quadratic(n=n, condition=50.0, seed=seed)
    x0 = np.ones(p.dim)
    r_mf = quasi_newton.minimize(
        p, x0=x0.copy(), M=1.0, eps=1e-6, memory=10, matrix_free=True, krylov_dim=20, m_max=n
    )
    r_d = quasi_newton.minimize(
        p, x0=x0.copy(), M=1.0, eps=1e-6, memory=10, matrix_free=False
    )
    return {
        "n": n,
        "mf": {"f": r_mf.f, "grad_norm": r_mf.grad_norm, "nit": r_mf.nit, "time_sec": r_mf.time_sec},
        "dense": {"f": r_d.f, "grad_norm": r_d.grad_norm, "nit": r_d.nit, "time_sec": r_d.time_sec},
        "f_gap": abs(r_mf.f - r_d.f),
        "x_gap": float(np.linalg.norm(r_mf.x - r_d.x)),
    }


def scaling(dims: list[int] | None = None, memory: int = 10) -> dict:
    if dims is None:
        dims = [200, 500, 1000, 2000, 4000]
    rows = []
    for n in dims:
        p = MatrixFreeQuadratic(n=n, condition=30.0, seed=0)
        x0 = np.zeros(p.dim)
        # matrix-free only (dense O(n^2) infeasible at large n)
        t0 = time.perf_counter()
        r = quasi_newton.minimize(
            p,
            x0=x0,
            M=1.0,
            eps=1e-4,
            max_iter=40,
            memory=memory,
            matrix_free=True,
            krylov_dim=min(25, n // 10),
            m_max=min(80, n),
        )
        elapsed = time.perf_counter() - t0
        rows.append(
            {
                "n": n,
                "time_sec": r.time_sec,
                "wall_sec": elapsed,
                "grad_norm": r.grad_norm,
                "nit": r.nit,
                "f": r.f,
            }
        )
        print(f"[scale] n={n} time={r.time_sec:.3f}s ||g||={r.grad_norm:.2e}", flush=True)
    return {"memory": memory, "rows": rows}


def dense_time_ceiling(ns: list[int] | None = None) -> dict:
    """Time one dense to_dense() build vs matvec for growing n (memory fixed)."""
    if ns is None:
        ns = [100, 200, 400, 800]
    rng = np.random.default_rng(0)
    rows = []
    for n in ns:
        B = LBFGSHessian(n, memory=10)
        for _ in range(10):
            s = rng.standard_normal(n)
            y = s + 0.1 * rng.standard_normal(n)
            B.update(s, y)
        v = rng.standard_normal(n)
        t0 = time.perf_counter()
        for _ in range(5):
            _ = B.to_dense()
        t_dense = (time.perf_counter() - t0) / 5
        t0 = time.perf_counter()
        for _ in range(50):
            _ = B.matvec(v)
        t_mv = (time.perf_counter() - t0) / 50
        rows.append({"n": n, "to_dense_sec": t_dense, "matvec_sec": t_mv, "ratio": t_dense / max(t_mv, 1e-16)})
        print(f"[ops] n={n} dense={t_dense:.4e}s matvec={t_mv:.4e}s ratio={t_dense/max(t_mv,1e-16):.1f}", flush=True)
    return {"rows": rows}


def make_figures(report: dict) -> None:
    if "scaling" in report:
        rows = report["scaling"]["rows"]
        ns = [r["n"] for r in rows]
        ts = [r["time_sec"] for r in rows]
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        ax.loglog(ns, ts, "o-", color="#2c7fb8", label="QN-CR matrix-free")
        ax.set_xlabel("dimension n")
        ax.set_ylabel("time (s)")
        ax.set_title("Matrix-free QN-CR scaling (Thm. mf-qn)")
        ax.legend()
        fig.tight_layout()
        _save(fig, "28_phase11_qn_scale.png")

    if "ops" in report:
        rows = report["ops"]["rows"]
        ns = [r["n"] for r in rows]
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        ax.semilogy(ns, [r["to_dense_sec"] for r in rows], "s-", label="to_dense", color="#e34a33")
        ax.semilogy(ns, [r["matvec_sec"] for r in rows], "o-", label="matvec", color="#2c7fb8")
        ax.set_xlabel("dimension n")
        ax.set_ylabel("seconds / call")
        ax.set_title("Compact BFGS: dense build vs matvec")
        ax.legend()
        fig.tight_layout()
        _save(fig, "29_phase11_bfgs_ops.png")


def main() -> None:
    RESDIR.mkdir(parents=True, exist_ok=True)
    report = {
        "matvec_agreement": matvec_agreement(),
        "dense_vs_mf": dense_vs_mf_solve(),
        "ops": dense_time_ceiling(),
        "scaling": scaling(),
    }
    make_figures(report)
    out = RESDIR / "phase11.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    print("agreement max_err", report["matvec_agreement"]["max_abs_err"])
    print("solve f_gap", report["dense_vs_mf"]["f_gap"], "x_gap", report["dense_vs_mf"]["x_gap"])


if __name__ == "__main__":
    main()
