"""Batch experiment helpers (optional CLI)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cubic_reg.problems import Quadratic
from cubic_reg.solvers import cr, krylov, arc


def run_baseline_suite(out_dir: str | Path = "experiments/results") -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for kappa in [10, 100, 1000]:
        p = Quadratic(n=50, condition=float(kappa), seed=0)
        x0 = np.ones(p.dim)
        r_cr = cr.minimize(p, x0=x0, M=1.0, eps=1e-8)
        r_arc = arc.minimize(p, x0=x0, M0=1.0, eps=1e-8)
        r_kry = krylov.minimize(p, x0=x0, M=1.0, eps=1e-6, krylov_dim=20)
        summary[f"kappa_{kappa}"] = {
            "cr": {"nit": r_cr.nit, "time": r_cr.time_sec, "grad": r_cr.grad_norm},
            "arc": {"nit": r_arc.nit, "time": r_arc.time_sec, "grad": r_arc.grad_norm},
            "krylov": {"nit": r_kry.nit, "time": r_kry.time_sec, "grad": r_kry.grad_norm, "hvp": r_kry.n_hvp},
        }
    path = out / "baseline_suite.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(run_baseline_suite(), indent=2))
