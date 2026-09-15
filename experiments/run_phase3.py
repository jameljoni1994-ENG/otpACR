"""Phase-3: matrix-free n=1e4 Krylov + Acc-CR vs Nesterov AGD."""

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

from cubic_reg.problems import MatrixFreeQuadratic, LogSumExp
from cubic_reg.solvers import krylov, arc, accelerated, first_order

FIGDIR = Path(__file__).resolve().parent / "figures"
RESDIR = Path(__file__).resolve().parent / "results"


def _save(fig, name: str) -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    path = FIGDIR / name
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


def matrix_free_scaling(ns=(2000, 5000, 10000), krylov_dim=40, max_iter=15) -> dict:
    rows = []
    for n in ns:
        print(f"MF quadratic n={n} ...", flush=True)
        p = MatrixFreeQuadratic(n=n, condition=100.0, seed=0)
        x0 = np.zeros(n)
        t0 = time.perf_counter()
        r = krylov.minimize(
            p,
            x0=x0,
            M=1.0,
            eps=1e-5,
            krylov_dim=min(krylov_dim, 80),
            max_iter=max_iter,
        )
        elapsed = time.perf_counter() - t0
        rows.append(
            {
                "n": n,
                "nit": r.nit,
                "time_sec": elapsed,
                "sec_per_iter": elapsed / max(r.nit, 1),
                "grad_norm": r.grad_norm,
                "n_hvp": r.n_hvp,
                "success": r.success,
                "f": r.f,
                "f_star": p.f_star,
            }
        )
        print(
            f"  nit={r.nit} ||g||={r.grad_norm:.2e} "
            f"{rows[-1]['sec_per_iter']:.4f}s/it hvp={r.n_hvp}",
            flush=True,
        )
    return {"rows": rows}


def fig_mf_scaling(data: dict) -> None:
    ns = [r["n"] for r in data["rows"]]
    tpi = [r["sec_per_iter"] for r in data["rows"]]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.loglog(ns, tpi, "o-")
    ax.set_xlabel("n")
    ax.set_ylabel("sec / iteration")
    ax.set_title("Matrix-free Krylov scalability (HVP only)")
    ax.grid(True, which="both", alpha=0.3)
    _save(fig, "11_matrix_free_krylov.png")


def acc_vs_nesterov(max_iter: int = 80) -> dict:
    p = LogSumExp(n=30, m=120, mu=0.01, seed=1)
    x0 = np.ones(p.dim)
    # Lipschitz-ish step: use small conservative step for FO methods
    # Rough L estimate via Hessian eigenvalues at x0
    H0 = p.hess(x0)
    L = float(np.linalg.norm(H0, 2))
    step = 1.0 / L

    warm = arc.minimize(p, x0=x0, M0=1.0, eps=1e-14, max_iter=400)
    f_star = warm.f

    # Fixed budget for fair curve comparison
    r_acc = accelerated.minimize(p, x0=x0, M=8.0, eps=-1.0, max_iter=max_iter)
    r_nag = first_order.minimize_nesterov(p, x0=x0, step=step, eps=-1.0, max_iter=max_iter)
    r_gd = first_order.minimize_gd(p, x0=x0, step=step, eps=-1.0, max_iter=max_iter)

    def pack(res, name):
        gap = np.maximum(np.asarray(res.history_f) - f_star, 1e-16)
        return {
            "name": name,
            "nit": res.nit,
            "time_sec": res.time_sec,
            "grad_norm": res.grad_norm,
            "f": res.f,
            "gap": gap.tolist(),
            "n_grad": res.n_grad,
            "n_hess": res.n_hess,
        }

    out = {
        "f_star": f_star,
        "L_est": L,
        "step": step,
        "accelerated_cr": pack(r_acc, "Acc-CR"),
        "nesterov_ag": pack(r_nag, "Nesterov AG"),
        "gd": pack(r_gd, "GD"),
    }
    print(
        f"Acc-CR gap_end={out['accelerated_cr']['gap'][-1]:.2e} "
        f"NAG={out['nesterov_ag']['gap'][-1]:.2e} GD={out['gd']['gap'][-1]:.2e}",
        flush=True,
    )
    return out


def fig_acc_vs_nag(data: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for key, style in [
        ("accelerated_cr", "-"),
        ("nesterov_ag", "--"),
        ("gd", ":"),
    ]:
        gap = np.asarray(data[key]["gap"])
        axes[0].semilogy(gap, style, label=data[key]["name"])
        # vs gradient evaluations (proxy for cost for FO; Acc-CR also uses Hess)
        n = len(gap)
        if key == "accelerated_cr":
            # each iter: ~2 grad + 1 hess — plot vs iter still on left; right uses wall time spacing
            t = np.linspace(0, data[key]["time_sec"], n)
        else:
            t = np.linspace(0, data[key]["time_sec"], n)
        axes[1].semilogy(t, gap, style, label=data[key]["name"])
    axes[0].set_xlabel("iteration")
    axes[0].set_ylabel(r"$f-f^*$")
    axes[0].set_title("Gap vs iterations")
    axes[1].set_xlabel("wall time (s)")
    axes[1].set_ylabel(r"$f-f^*$")
    axes[1].set_title("Gap vs time")
    for ax in axes:
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)
    fig.suptitle("Acc-CR vs Nesterov AG vs GD (LogSumExp)")
    _save(fig, "12_acc_vs_nesterov.png")


def main() -> None:
    RESDIR.mkdir(parents=True, exist_ok=True)
    mf = matrix_free_scaling()
    rates = acc_vs_nesterov()
    payload = {"matrix_free_krylov": mf, "acc_vs_nesterov": rates}
    path = RESDIR / "phase3.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {path}")
    fig_mf_scaling(mf)
    fig_acc_vs_nag(rates)
    print("Phase-3 done.")


if __name__ == "__main__":
    main()
