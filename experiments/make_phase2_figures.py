"""Generate phase-2 figures from fresh runs / phase2.json."""

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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_phase2 import high_dim_krylov, rate_comparison, multiprocess_distributed_logistic

FIGDIR = Path(__file__).resolve().parent / "figures"
RESDIR = Path(__file__).resolve().parent / "results"


def _save(fig, name: str) -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    path = FIGDIR / name
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


def fig_high_dim(data: dict) -> None:
    rows = data["rows"]
    ns_k = [r["n"] for r in rows]
    t_k = [r["krylov"]["sec_per_iter"] for r in rows]
    ns_e, t_e = [], []
    for r in rows:
        if "exact" in r:
            ns_e.append(r["n"])
            t_e.append(r["exact"]["sec_per_iter"])

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.loglog(ns_k, t_k, "s-", label="Krylov m≤30")
    if ns_e:
        ax.loglog(ns_e, t_e, "o-", label="Exact CR")
    ax.set_xlabel("n")
    ax.set_ylabel("sec / iteration")
    ax.set_title("High-dimensional scalability")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    _save(fig, "08_high_dim_krylov.png")


def fig_rates(data: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for name, style in [("cr", "-"), ("accelerated", "--")]:
        gap = np.asarray(data[name]["gap"])
        k = np.arange(1, len(gap) + 1)
        axes[0].semilogy(k, gap, style, label=f"{name} slope={data[name]['loglog_slope']:.2f}")
        axes[1].semilogy(k, gap * k**2, style, label=name)
        axes[2].semilogy(k, gap * k**3, style, label=name)
    axes[0].set_title(r"$f-f^*$")
    axes[1].set_title(r"$k^2(f-f^*)$")
    axes[2].set_title(r"$k^3(f-f^*)$")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("k")
    fig.suptitle("Rate diagnostics on LogSumExp")
    _save(fig, "09_rate_logsumexp.png")


def fig_mp(data: dict) -> None:
    workers, times, speedups, gnorms = [], [], [], []
    for key in sorted(data.keys(), key=lambda s: int(s.split("_")[1])):
        w = int(key.split("_")[1])
        workers.append(w)
        times.append(data[key]["time_sec"])
        speedups.append(data[key]["speedup_vs_1"])
        gnorms.append(data[key]["grad_norm"])

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].plot(workers, times, "o-")
    axes[0].set_xlabel("workers")
    axes[0].set_ylabel("wall time (s)")
    axes[0].set_title("Multiprocess data-parallel ARC")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(workers, speedups, "s-")
    axes[1].axhline(1.0, color="gray", ls=":")
    axes[1].set_xlabel("workers")
    axes[1].set_ylabel("speedup vs 1")
    axes[1].set_title("Measured speedup")
    axes[1].grid(True, alpha=0.3)
    _save(fig, "10_multiprocess_speedup.png")


def main() -> None:
    # Prefer recompute for figures so curves match this machine
    print("Running high-dim Krylov...")
    hd = high_dim_krylov()
    print("Running rate comparison...")
    rates = rate_comparison()
    print("Running multiprocess distributed...")
    mp = multiprocess_distributed_logistic()

    payload = {
        "high_dim_krylov": hd,
        "rate_comparison": rates,
        "multiprocess_distributed": mp,
    }
    RESDIR.mkdir(parents=True, exist_ok=True)
    (RESDIR / "phase2.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    fig_high_dim(hd)
    fig_rates(rates)
    fig_mp(mp)
    print("Phase-2 figures done.")


if __name__ == "__main__":
    main()
