"""Phase-4: adaptive inexact Krylov, sketch+Krylov hybrid, MNIST-like stochastic."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cubic_reg.problems import Quadratic, MatrixFreeQuadratic, make_mnist_like_logistic
from cubic_reg.solvers import krylov, sketch_krylov, stochastic, arc

FIGDIR = Path(__file__).resolve().parent / "figures"
RESDIR = Path(__file__).resolve().parent / "results"


def _save(fig, name: str) -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    path = FIGDIR / name
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


def inexact_krylov_study() -> dict:
    p = Quadratic(n=200, condition=100.0, seed=0)
    x0 = np.ones(p.dim)
    fixed = krylov.minimize(
        p, x0=x0, M=1.0, eps=1e-8, krylov_dim=10, adaptive_tol=False, inexact_tol=0.5, m_max=10
    )
    adapt = krylov.minimize(
        p, x0=x0, M=1.0, eps=1e-8, krylov_dim=10, adaptive_tol=True, inexact_tol=0.5, m_max=60
    )
    out = {
        "fixed_m10": {
            "nit": fixed.nit,
            "grad_norm": fixed.grad_norm,
            "n_hvp": fixed.n_hvp,
            "time_sec": fixed.time_sec,
            "history_m": getattr(fixed, "history_m", []),
            "history_resid": getattr(fixed, "history_resid", []),
            "history_theta": getattr(fixed, "history_theta", []),
            "history_grad_norm": fixed.history_grad_norm,
        },
        "adaptive": {
            "nit": adapt.nit,
            "grad_norm": adapt.grad_norm,
            "n_hvp": adapt.n_hvp,
            "time_sec": adapt.time_sec,
            "history_m": getattr(adapt, "history_m", []),
            "history_resid": getattr(adapt, "history_resid", []),
            "history_theta": getattr(adapt, "history_theta", []),
            "history_grad_norm": adapt.history_grad_norm,
        },
    }
    print(
        f"inexact: fixed m HVP={fixed.n_hvp} ||g||={fixed.grad_norm:.2e}; "
        f"adapt HVP={adapt.n_hvp} ||g||={adapt.grad_norm:.2e}",
        flush=True,
    )
    return out


def fig_inexact(data: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for key, style in [("fixed_m10", "-"), ("adaptive", "--")]:
        axes[0].semilogy(data[key]["history_grad_norm"], style, label=key)
        if data[key]["history_m"]:
            axes[1].plot(data[key]["history_m"], style + "o", label=key, markersize=3)
    axes[0].set_title(r"$\|\nabla f\|$")
    axes[0].set_xlabel("iteration")
    axes[1].set_title("Krylov dim used")
    axes[1].set_xlabel("iteration")
    for ax in axes:
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.suptitle("Adaptive inexact Krylov vs fixed m")
    _save(fig, "13_inexact_krylov.png")


def sketch_krylov_study() -> dict:
    p = MatrixFreeQuadratic(n=1500, condition=80.0, seed=0)
    x0 = np.zeros(p.dim)
    r_arc = arc.minimize(p, x0=x0, M0=1.0, eps=1e-5, max_iter=40, mode="krylov", krylov_dim=25)
    # MatrixFreeQuadratic has no dense hess — arc exact mode would fail; use sketch hybrid
    r_sk = sketch_krylov.minimize(
        p, x0=x0, M0=1.0, eps=1e-5, max_iter=40, sketch_rank=25, krylov_dim=20
    )
    out = {
        "arc_krylov": {
            "nit": r_arc.nit,
            "grad_norm": r_arc.grad_norm,
            "n_hvp": r_arc.n_hvp,
            "time_sec": r_arc.time_sec,
            "rejected": r_arc.rejected_steps,
            "history_grad_norm": r_arc.history_grad_norm,
        },
        "sketch_krylov": {
            "nit": r_sk.nit,
            "grad_norm": r_sk.grad_norm,
            "n_hvp": r_sk.n_hvp,
            "time_sec": r_sk.time_sec,
            "rejected": r_sk.rejected_steps,
            "message": r_sk.message,
            "history_grad_norm": r_sk.history_grad_norm,
        },
    }
    print(
        f"sketch: ARC-K ||g||={r_arc.grad_norm:.2e} t={r_arc.time_sec:.3f}; "
        f"hybrid ||g||={r_sk.grad_norm:.2e} t={r_sk.time_sec:.3f} {r_sk.message}",
        flush=True,
    )
    return out


def fig_sketch(data: dict) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for key in data:
        ax.semilogy(data[key]["history_grad_norm"], label=key)
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"$\|\nabla f\|$")
    ax.set_title("ARC-Krylov vs Sketch+Krylov hybrid")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    _save(fig, "14_sketch_krylov.png")


def mnist_like_stochastic() -> dict:
    # 28x28 logistic; avoid dense exact CR (eigh O(n^3) is prohibitive at n=784)
    prob = make_mnist_like_logistic(n_samples=3000, n_features=784, seed=0)
    x0 = np.zeros(prob.dim)
    r_sub = stochastic.minimize(
        prob, x0=x0, M=1.0, eps=1e-3, max_iter=40, batch_grad=256, batch_hess=64
    )
    r_kry = krylov.minimize(
        prob, x0=x0, M=1.0, eps=1e-3, max_iter=25, krylov_dim=30, adaptive_tol=True, m_max=50
    )
    r_sgd = stochastic.minimize_sgd(prob, x0=x0, lr=0.05, batch=128, max_iter=150, eps=1e-3)
    r_adam = stochastic.minimize_adam(prob, x0=x0, lr=0.01, batch=128, max_iter=150, eps=1e-3)
    methods = {
        "SubCR": r_sub,
        "Krylov-CR": r_kry,
        "SGD": r_sgd,
        "Adam": r_adam,
    }
    out = {}
    for name, r in methods.items():
        out[name] = {
            "nit": r.nit,
            "time_sec": r.time_sec,
            "grad_norm": r.grad_norm,
            "f": r.f,
            "history_grad_norm": r.history_grad_norm,
            "n_hess": r.n_hess,
            "n_grad": r.n_grad,
            "n_hvp": r.n_hvp,
        }
        print(f"{name:10} t={r.time_sec:.3f}s ||g||={r.grad_norm:.3e} f={r.f:.5f}", flush=True)
    return out


def fig_mnist(data: dict) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for name, r in data.items():
        t = np.linspace(0, r["time_sec"], num=len(r["history_grad_norm"]))
        ax.semilogy(t, np.maximum(r["history_grad_norm"], 1e-16), label=name)
    ax.set_xlabel("wall time (s)")
    ax.set_ylabel(r"$\|\nabla f\|$")
    ax.set_title("MNIST-like logistic (784-d): stochastic comparison")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    _save(fig, "15_mnist_like_stochastic.png")


def main() -> None:
    RESDIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "inexact_krylov": inexact_krylov_study(),
        "sketch_krylov": sketch_krylov_study(),
        "mnist_like": mnist_like_stochastic(),
    }
    path = RESDIR / "phase4.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {path}")
    fig_inexact(payload["inexact_krylov"])
    fig_sketch(payload["sketch_krylov"])
    fig_mnist(payload["mnist_like"])
    print("Phase-4 done.")


if __name__ == "__main__":
    main()
