"""Run all research-axis experiment suites and write JSON summaries."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cubic_reg.problems import Quadratic, LogSumExp, make_synthetic_logistic, make_synthetic_mlp
from cubic_reg.solvers import (
    cr,
    krylov,
    arc,
    quasi_newton,
    accelerated,
    stochastic,
    tensor,
    distributed,
)


def _summ(res, include_history: bool = False) -> dict:
    out = {
        "nit": res.nit,
        "time_sec": res.time_sec,
        "grad_norm": res.grad_norm,
        "f": res.f,
        "success": res.success,
        "n_f": res.n_f,
        "n_grad": res.n_grad,
        "n_hess": res.n_hess,
        "n_hvp": res.n_hvp,
        "rejected_steps": res.rejected_steps,
        "message": res.message,
    }
    if include_history:
        out["history_f"] = [float(v) for v in res.history_f]
        out["history_grad_norm"] = [float(v) for v in res.history_grad_norm]
        out["history_M"] = [float(v) for v in res.history_M]
    return out


def run_all(out_dir: str | Path = "experiments/results") -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report: dict = {}

    # 1) Baseline + Krylov + ARC
    p = Quadratic(n=50, condition=100.0, seed=0)
    x0 = np.ones(p.dim)
    report["quadratic_core"] = {
        "cr": _summ(cr.minimize(p, x0=x0, M=1.0, eps=1e-8), include_history=True),
        "krylov_m20": _summ(krylov.minimize(p, x0=x0, M=1.0, eps=1e-6, krylov_dim=20), include_history=True),
        "arc": _summ(arc.minimize(p, x0=x0, M0=1.0, eps=1e-8), include_history=True),
        "arc_krylov": _summ(arc.minimize(p, x0=x0, M0=1.0, mode="krylov", krylov_dim=15, eps=1e-6), include_history=True),
    }

    # 2) Quasi-Newton
    report["quasi_newton"] = {
        "cr": _summ(cr.minimize(p, x0=x0, M=1.0, eps=1e-6)),
        "qn_m10": _summ(quasi_newton.minimize(p, x0=x0, M=1.0, eps=1e-6, memory=10)),
        "lbfgs_plain": _summ(quasi_newton.minimize_lbfgs_plain(p, x0=x0, eps=1e-6)),
    }
    mlp = make_synthetic_mlp(n_samples=120, d_in=5, hidden=5, seed=0)
    report["mlp_qn"] = {
        "qn": _summ(quasi_newton.minimize(mlp, x0=np.zeros(mlp.dim), eps=1e-4, max_iter=60, memory=8)),
    }

    # 3) Accelerated
    report["accelerated"] = {}
    for kappa in [10, 100, 1000]:
        q = Quadratic(n=40, condition=float(kappa), seed=0)
        x = np.ones(q.dim)
        report["accelerated"][f"kappa_{kappa}"] = {
            "cr": _summ(cr.minimize(q, x0=x, M=1.0, eps=1e-8, max_iter=100)),
            "acc": _summ(accelerated.minimize(q, x0=x, M=1.0, eps=1e-8, max_iter=100)),
        }

    # 4) Stochastic
    logreg = make_synthetic_logistic(n_samples=800, n_features=40, seed=0)
    z0 = np.zeros(logreg.dim)
    report["stochastic"] = {
        "full_cr": _summ(cr.minimize(logreg, x0=z0, M=1.0, eps=1e-3, max_iter=30)),
        "sub_cr": _summ(stochastic.minimize(logreg, x0=z0, max_iter=50, batch_grad=128, batch_hess=32, eps=1e-3)),
        "sgd": _summ(stochastic.minimize_sgd(logreg, x0=z0, max_iter=200, batch=64, lr=0.15, eps=1e-3)),
        "adam": _summ(stochastic.minimize_adam(logreg, x0=z0, max_iter=200, batch=64, lr=0.05, eps=1e-3)),
    }

    # 5) Tensor
    ls = LogSumExp(n=12, m=30, seed=0)
    xls = 0.1 * np.ones(ls.dim)
    report["tensor"] = {
        "cr": _summ(cr.minimize(ls, x0=xls, M=1.0, eps=1e-5, max_iter=30)),
        "tensor": _summ(tensor.minimize(ls, x0=xls, L=1.0, eps=1e-5, max_iter=20)),
    }

    # 6) Distributed
    report["distributed"] = {}
    for w in [1, 2, 4, 8]:
        report["distributed"][f"workers_{w}"] = _summ(
            distributed.minimize(
                p,
                x0=x0,
                n_workers=w,
                sketch_rank=25,
                eps=1e-4,
                max_iter=80,
                network_latency_ms=0.2,
            )
        )

    path = out / "all_axes.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {path}")
    return report


if __name__ == "__main__":
    run_all()
