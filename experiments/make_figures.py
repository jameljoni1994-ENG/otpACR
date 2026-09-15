"""Generate publication-style figures for all research axes."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cubic_reg.problems import Quadratic, LogSumExp, make_synthetic_logistic
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
from cubic_reg.plotting import plot_optimality_gap, plot_grad_norm, plot_time_vs_n, plot_M_history


FIGDIR = Path(__file__).resolve().parent / "figures"


def _save(fig, name: str) -> Path:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    path = FIGDIR / name
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")
    return path


def fig_baseline_condition() -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for kappa in [10, 100, 1000]:
        p = Quadratic(n=50, condition=float(kappa), seed=0)
        res = cr.minimize(p, x0=np.ones(p.dim), M=1.0, eps=1e-12, max_iter=80)
        gap = np.maximum(np.asarray(res.history_f) - p.f_star, 1e-16)
        ax.semilogy(gap, label=f"κ={kappa}, nit={res.nit}")
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"$f(x_k)-f^*$")
    ax.set_title("Baseline CR — effect of condition number")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    _save(fig, "00_baseline_condition.png")


def fig_krylov_vs_exact() -> None:
    p = Quadratic(n=80, condition=100.0, seed=0)
    x0 = np.ones(p.dim)
    results = {"exact CR": cr.minimize(p, x0=x0, M=1.0, eps=1e-10)}
    for m in [5, 10, 20, 50]:
        results[f"Krylov m={m}"] = krylov.minimize(
            p, x0=x0, M=1.0, eps=1e-10, krylov_dim=m, max_iter=100
        )
    fig, ax = plt.subplots(figsize=(7, 4))
    plot_optimality_gap(results, f_star=p.f_star, ax=ax, title="Exact CR vs Krylov")
    _save(fig, "01_krylov_gap.png")

    ns = [50, 100, 200, 400]
    t_exact, t_kry = [], []
    for n in ns:
        q = Quadratic(n=n, condition=50.0, seed=0)
        x = np.ones(n)
        re = cr.minimize(q, x0=x, M=1.0, eps=1e-6, max_iter=15)
        rk = krylov.minimize(q, x0=x, M=1.0, eps=1e-6, krylov_dim=20, max_iter=15)
        t_exact.append(re.time_sec / max(re.nit, 1))
        t_kry.append(rk.time_sec / max(rk.nit, 1))
    fig, ax = plt.subplots(figsize=(7, 4))
    plot_time_vs_n(ns, {"exact CR": t_exact, "Krylov m=20": t_kry}, ax=ax)
    _save(fig, "01_krylov_scalability.png")


def fig_arc_sensitivity() -> None:
    p = Quadratic(n=40, condition=100.0, seed=1)
    x0 = np.ones(p.dim)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for M0 in [1e-4, 1e-2, 1.0, 1e2]:
        r = arc.minimize(p, x0=x0, M0=M0, eps=1e-10, max_iter=120)
        gap = np.maximum(np.asarray(r.history_f) - p.f_star, 1e-16)
        axes[0].semilogy(gap, label=f"M0={M0:g}")
        axes[1].semilogy(r.history_M, label=f"M0={M0:g}")
    axes[0].set_title("gap"); axes[1].set_title("M_k")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("iteration")
    fig.suptitle("ARC sensitivity to M0")
    _save(fig, "02_arc_sensitivity.png")


def fig_quasi_newton() -> None:
    p = Quadratic(n=80, condition=200.0, seed=0)
    x0 = np.ones(p.dim)
    results = {
        "CR": cr.minimize(p, x0=x0, M=1.0, eps=1e-8),
        "QN-CR m=10": quasi_newton.minimize(p, x0=x0, M=1.0, eps=1e-6, memory=10),
        "L-BFGS": quasi_newton.minimize_lbfgs_plain(p, x0=x0, eps=1e-6),
    }
    fig, ax = plt.subplots(figsize=(7, 4))
    plot_optimality_gap(results, f_star=p.f_star, ax=ax, title="Quasi-Newton cubic")
    _save(fig, "03_quasi_newton.png")


def fig_accelerated() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
    for ax, kappa in zip(axes, [10, 100, 1000]):
        p = Quadratic(n=40, condition=float(kappa), seed=0)
        x0 = np.ones(p.dim)
        results = {
            "CR": cr.minimize(p, x0=x0, M=1.0, eps=1e-10, max_iter=100),
            "Acc-CR": accelerated.minimize(p, x0=x0, M=1.0, eps=1e-10, max_iter=100),
        }
        plot_optimality_gap(results, f_star=p.f_star, ax=ax, title=f"κ={kappa}")
    fig.suptitle("Accelerated vs baseline CR")
    _save(fig, "04_accelerated.png")


def fig_stochastic() -> None:
    prob = make_synthetic_logistic(n_samples=1000, n_features=50, seed=0)
    x0 = np.zeros(prob.dim)
    methods = {
        "Full CR": cr.minimize(prob, x0=x0, M=1.0, eps=1e-3, max_iter=35),
        "SubCR": stochastic.minimize(prob, x0=x0, max_iter=50, batch_grad=128, batch_hess=32, eps=1e-3),
        "SGD": stochastic.minimize_sgd(prob, x0=x0, max_iter=250, batch=64, lr=0.15, eps=1e-3),
        "Adam": stochastic.minimize_adam(prob, x0=x0, max_iter=250, batch=64, lr=0.05, eps=1e-3),
    }
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for name, r in methods.items():
        t = np.linspace(0, r.time_sec, num=len(r.history_grad_norm))
        ax.semilogy(t, np.maximum(r.history_grad_norm, 1e-16), label=f"{name}")
    ax.set_xlabel("wall time (s)")
    ax.set_ylabel(r"$\|\nabla f\|$")
    ax.set_title("Stochastic methods vs time")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    _save(fig, "05_stochastic_time.png")


def fig_tensor() -> None:
    p = LogSumExp(n=12, m=30, seed=0)
    x0 = 0.1 * np.ones(p.dim)
    results = {
        "CR": cr.minimize(p, x0=x0, M=1.0, eps=1e-6, max_iter=40),
        "Tensor": tensor.minimize(p, x0=x0, L=1.0, eps=1e-6, max_iter=25),
    }
    fig, ax = plt.subplots(figsize=(7, 4))
    plot_grad_norm(results, ax=ax, title="Tensor vs CR (LogSumExp)")
    _save(fig, "06_tensor.png")


def fig_distributed() -> None:
    p = Quadratic(n=60, condition=80.0, seed=0)
    workers = [1, 2, 4, 8, 16]
    results = distributed.speedup_experiment(
        p, worker_counts=workers, sketch_rank=20, eps=1e-4, max_iter=60, network_latency_ms=0.8
    )
    t1 = results[0].time_sec
    speedups = [t1 / max(r.time_sec, 1e-12) for r in results]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(workers, speedups, "o-")
    ax.set_xlabel("workers")
    ax.set_ylabel("speedup vs 1 worker")
    ax.set_title("Simulated distributed ARC speedup")
    ax.grid(True, alpha=0.3)
    _save(fig, "07_distributed_speedup.png")


def write_manifest(paths: list[str]) -> None:
    manifest = {"figures": paths}
    path = FIGDIR / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    generators = [
        fig_baseline_condition,
        fig_krylov_vs_exact,
        fig_arc_sensitivity,
        fig_quasi_newton,
        fig_accelerated,
        fig_stochastic,
        fig_tensor,
        fig_distributed,
    ]
    saved = []
    for gen in generators:
        gen()
        saved.append(gen.__name__)
    write_manifest(saved)
    print(f"Done. Figures in {FIGDIR}")


if __name__ == "__main__":
    main()
