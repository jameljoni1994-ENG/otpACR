"""Phase-7: MNIST hyperparameter sweep + ARC-Krylov fair comparison."""

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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_phase6 import load_mnist_binary
from cubic_reg.problems import make_mnist_like_logistic
from cubic_reg.solvers import krylov, arc, stochastic

FIGDIR = Path(__file__).resolve().parent / "figures"
RESDIR = Path(__file__).resolve().parent / "results"


def _save(fig, name: str) -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    path = FIGDIR / name
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


def _pack(r, extra=None):
    d = {
        "nit": r.nit,
        "time_sec": r.time_sec,
        "grad_norm": r.grad_norm,
        "f": r.f,
        "n_hvp": r.n_hvp,
        "n_hess": r.n_hess,
        "n_grad": r.n_grad,
        "rejected_steps": r.rejected_steps,
        "history_f": r.history_f,
        "history_grad_norm": r.history_grad_norm,
        "success": r.success,
    }
    if extra:
        d.update(extra)
    return d


def get_problem():
    prob, source = load_mnist_binary(digits=(3, 8), max_per_class=1500, seed=0)
    if prob is None:
        prob = make_mnist_like_logistic(n_samples=3000, n_features=784, seed=0)
        source = "mnist_like_fallback"
    return prob, source


def hyperparam_sweep(prob) -> dict:
    x0 = np.zeros(prob.dim)
    sweep = {"adam": [], "sgd": [], "subcr": []}

    for lr in [0.001, 0.005, 0.01, 0.05, 0.1]:
        r = stochastic.minimize_adam(
            prob, x0=x0.copy(), lr=lr, batch=128, max_iter=200, eps=1e-4, seed=0
        )
        sweep["adam"].append(_pack(r, {"lr": lr, "batch": 128}))
        print(f"Adam lr={lr}: f={r.f:.5f} ||g||={r.grad_norm:.3e} t={r.time_sec:.3f}", flush=True)

    for lr in [0.01, 0.05, 0.1, 0.2]:
        r = stochastic.minimize_sgd(
            prob, x0=x0.copy(), lr=lr, batch=128, max_iter=200, eps=1e-4, seed=0
        )
        sweep["sgd"].append(_pack(r, {"lr": lr, "batch": 128}))
        print(f"SGD  lr={lr}: f={r.f:.5f} ||g||={r.grad_norm:.3e} t={r.time_sec:.3f}", flush=True)

    for bg, bh in [(128, 32), (256, 64), (512, 128)]:
        r = stochastic.minimize(
            prob,
            x0=x0.copy(),
            M=1.0,
            eps=1e-4,
            max_iter=40,
            batch_grad=bg,
            batch_hess=bh,
            seed=0,
        )
        sweep["subcr"].append(_pack(r, {"batch_grad": bg, "batch_hess": bh}))
        print(
            f"SubCR bg={bg} bh={bh}: f={r.f:.5f} ||g||={r.grad_norm:.3e} t={r.time_sec:.3f}",
            flush=True,
        )

    def best(rows, key="f"):
        return min(rows, key=lambda d: d[key])

    return {
        "sweep": sweep,
        "best_adam": best(sweep["adam"]),
        "best_sgd": best(sweep["sgd"]),
        "best_subcr": best(sweep["subcr"]),
    }


def core_comparison(prob, time_budget: float | None = None) -> dict:
    """Compare Krylov-CR, ARC-Krylov, and best-tuned FO methods.

    If time_budget is set, FO methods get more iterations until wall time roughly matches.
    """
    x0 = np.zeros(prob.dim)

    r_kry = krylov.minimize(
        prob,
        x0=x0.copy(),
        M=1.0,
        eps=1e-4,
        max_iter=30,
        krylov_dim=30,
        adaptive_tol=True,
        m_max=60,
    )
    print(f"Krylov-CR: f={r_kry.f:.5f} t={r_kry.time_sec:.3f}", flush=True)

    r_arc = arc.minimize(
        prob,
        x0=x0.copy(),
        M0=1.0,
        eps=1e-4,
        max_iter=40,
        mode="krylov",
        krylov_dim=30,
    )
    print(
        f"ARC-Krylov: f={r_arc.f:.5f} t={r_arc.time_sec:.3f} rej={r_arc.rejected_steps}",
        flush=True,
    )

    budget = time_budget if time_budget is not None else max(r_kry.time_sec, r_arc.time_sec)

    # Give Adam/SGD a matched wall-time budget via generous max_iter + early stop by time
    def run_fo_timed(name, fn):
        t0 = time.perf_counter()
        # Estimate iters: previous ~0.2s for 120 iters => ~600/sec for adam-ish; use large max
        r = fn()
        # If finished too fast, continue from solution with more iters
        while time.perf_counter() - t0 < budget:
            more = fn(x0=r.x)
            # merge histories
            r.history_f = list(r.history_f) + list(more.history_f)
            r.history_grad_norm = list(r.history_grad_norm) + list(more.history_grad_norm)
            r.x, r.f, r.grad_norm = more.x, more.f, more.grad_norm
            r.nit = len(r.history_f)
            r.n_grad += more.n_grad
            r.n_f += more.n_f
            r.time_sec = time.perf_counter() - t0
            if more.grad_norm <= 1e-4:
                break
            if more.nit <= 1:
                break
        r.time_sec = time.perf_counter() - t0
        print(f"{name} (budget~{budget:.2f}s): f={r.f:.5f} t={r.time_sec:.3f}", flush=True)
        return r

    # First get good lrs from a quick mini-sweep already done externally; use strong defaults
    r_adam = run_fo_timed(
        "Adam",
        lambda x0=x0.copy(): stochastic.minimize_adam(
            prob, x0=x0, lr=0.01, batch=128, max_iter=80, eps=1e-4, seed=0
        ),
    )
    r_sgd = run_fo_timed(
        "SGD",
        lambda x0=x0.copy(): stochastic.minimize_sgd(
            prob, x0=x0, lr=0.05, batch=128, max_iter=80, eps=1e-4, seed=0
        ),
    )

    return {
        "time_budget_sec": budget,
        "Krylov-CR": _pack(r_kry),
        "ARC-Krylov": _pack(r_arc),
        "Adam_matched_time": _pack(r_adam),
        "SGD_matched_time": _pack(r_sgd),
    }


def fig_sweep(hs: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    adam = hs["sweep"]["adam"]
    axes[0].semilogy([d["lr"] for d in adam], [d["f"] for d in adam], "o-")
    axes[0].set_xlabel("lr")
    axes[0].set_ylabel("final f")
    axes[0].set_title("Adam sweep")
    axes[0].grid(True, alpha=0.3)

    sgd = hs["sweep"]["sgd"]
    axes[1].semilogy([d["lr"] for d in sgd], [d["f"] for d in sgd], "s-")
    axes[1].set_xlabel("lr")
    axes[1].set_title("SGD sweep")
    axes[1].grid(True, alpha=0.3)

    sub = hs["sweep"]["subcr"]
    labels = [f"{d['batch_grad']}/{d['batch_hess']}" for d in sub]
    axes[2].semilogy(range(len(sub)), [d["f"] for d in sub], "^-")
    axes[2].set_xticks(range(len(sub)), labels)
    axes[2].set_xlabel("bg/bh")
    axes[2].set_title("SubCR sweep")
    axes[2].grid(True, alpha=0.3)
    fig.suptitle("MNIST hyperparameter sweeps (lower f better)")
    plt.tight_layout()
    _save(fig, "21_mnist_hp_sweep.png")


def fig_fair(comp: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for name in ["Krylov-CR", "ARC-Krylov", "Adam_matched_time", "SGD_matched_time"]:
        r = comp[name]
        t = np.linspace(0, r["time_sec"], num=len(r["history_grad_norm"]))
        axes[0].semilogy(t, np.maximum(r["history_grad_norm"], 1e-16), label=name)
        axes[1].semilogy(
            np.linspace(0, r["time_sec"], num=len(r["history_f"])),
            np.maximum(r["history_f"], 1e-16),
            label=name,
        )
    axes[0].set_title(r"$\|\nabla f\|$ vs time")
    axes[1].set_title(r"$f$ vs time")
    for ax in axes:
        ax.set_xlabel("wall time (s)")
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.3)
    fig.suptitle(f"Fair-ish comparison (budget~{comp['time_budget_sec']:.2f}s)")
    plt.tight_layout()
    _save(fig, "22_mnist_fair_compare.png")


def main() -> None:
    RESDIR.mkdir(parents=True, exist_ok=True)
    prob, source = get_problem()
    print(f"Problem source={source} N={prob.N} d={prob.dim}", flush=True)
    hs = hyperparam_sweep(prob)
    # Use best Adam/SGD lrs inside comparison by monkey patching defaults via closure
    best_adam_lr = hs["best_adam"]["lr"]
    best_sgd_lr = hs["best_sgd"]["lr"]

    # Re-bind comparison with best lrs
    global stochastic

    def core_with_best():
        x0 = np.zeros(prob.dim)
        r_kry = krylov.minimize(
            prob, x0=x0.copy(), M=1.0, eps=1e-4, max_iter=30, krylov_dim=30, adaptive_tol=True, m_max=60
        )
        r_arc = arc.minimize(
            prob, x0=x0.copy(), M0=1.0, eps=1e-4, max_iter=40, mode="krylov", krylov_dim=30
        )
        budget = max(r_kry.time_sec, r_arc.time_sec)

        def fo_loop(make_fn, name):
            t0 = time.perf_counter()
            x = x0.copy()
            hist_f, hist_g = [], []
            n_grad = n_f = 0
            f = gnorm = None
            while time.perf_counter() - t0 < budget:
                r = make_fn(x)
                hist_f.extend(r.history_f)
                hist_g.extend(r.history_grad_norm)
                x, f, gnorm = r.x, r.f, r.grad_norm
                n_grad += r.n_grad
                n_f += r.n_f
                if gnorm <= 1e-4:
                    break
            elapsed = time.perf_counter() - t0
            print(f"{name}: f={f:.5f} ||g||={gnorm:.3e} t={elapsed:.3f}", flush=True)
            return {
                "nit": len(hist_f),
                "time_sec": elapsed,
                "grad_norm": gnorm,
                "f": f,
                "n_grad": n_grad,
                "n_f": n_f,
                "n_hvp": 0,
                "n_hess": 0,
                "rejected_steps": 0,
                "history_f": hist_f,
                "history_grad_norm": hist_g,
                "success": gnorm <= 1e-4,
            }

        adam = fo_loop(
            lambda x: stochastic.minimize_adam(
                prob, x0=x, lr=best_adam_lr, batch=128, max_iter=60, eps=1e-4, seed=0
            ),
            f"Adam(lr={best_adam_lr})",
        )
        sgd = fo_loop(
            lambda x: stochastic.minimize_sgd(
                prob, x0=x, lr=best_sgd_lr, batch=128, max_iter=60, eps=1e-4, seed=0
            ),
            f"SGD(lr={best_sgd_lr})",
        )
        # best SubCR single run (may exceed budget — still report)
        b = hs["best_subcr"]
        r_sub = stochastic.minimize(
            prob,
            x0=x0.copy(),
            M=1.0,
            eps=1e-4,
            max_iter=50,
            batch_grad=b["batch_grad"],
            batch_hess=b["batch_hess"],
            seed=0,
        )
        print(f"Best SubCR: f={r_sub.f:.5f} t={r_sub.time_sec:.3f}", flush=True)

        return {
            "time_budget_sec": budget,
            "best_adam_lr": best_adam_lr,
            "best_sgd_lr": best_sgd_lr,
            "Krylov-CR": _pack(r_kry),
            "ARC-Krylov": _pack(r_arc),
            "Adam_matched_time": adam,
            "SGD_matched_time": sgd,
            "Best_SubCR": _pack(r_sub, {"batch_grad": b["batch_grad"], "batch_hess": b["batch_hess"]}),
        }

    comp = core_with_best()
    payload = {"source": source, "n": prob.N, "dim": prob.dim, "hyperparams": hs, "comparison": comp}
    path = RESDIR / "phase7.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {path}")
    fig_sweep(hs)
    fig_fair(comp)
    print("Phase-7 done.")


if __name__ == "__main__":
    main()
