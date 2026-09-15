"""Shared plotting helpers for notebooks and experiment scripts."""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np

from cubic_reg.metrics import OptimizationResult


def plot_optimality_gap(
    results: Dict[str, OptimizationResult],
    f_star: Optional[float] = None,
    ax=None,
    title: str = "Optimality gap",
):
    """Semilogy of f(x_k) - f* (or f if f* unknown)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    for name, res in results.items():
        fs = f_star if f_star is not None else res.f_star
        y = np.asarray(res.history_f, dtype=float)
        if fs is not None:
            y = np.maximum(y - fs, 1e-16)
        else:
            y = np.maximum(y - np.min(y) + 1e-16, 1e-16)
        ax.semilogy(y, label=f"{name} (nit={res.nit})")
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"$f(x_k)-f^*$" if f_star is not None or any(r.f_star is not None for r in results.values()) else r"$f(x_k)$")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    return ax


def plot_grad_norm(
    results: Dict[str, OptimizationResult],
    ax=None,
    title: str = "Gradient norm",
):
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    for name, res in results.items():
        ax.semilogy(np.maximum(res.history_grad_norm, 1e-16), label=name)
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"$\|\nabla f\|$")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    return ax


def plot_time_vs_n(
    ns: Iterable[int],
    series: Dict[str, Iterable[float]],
    ax=None,
    title: str = "Time per iteration vs n",
    ylabel: str = "sec / iteration",
):
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    ns = list(ns)
    for name, vals in series.items():
        ax.loglog(ns, list(vals), "o-", label=name)
    ax.set_xlabel("n")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    return ax


def plot_M_history(res: OptimizationResult, ax=None, title: str = "Adaptive M"):
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3.5))
    ax.semilogy(res.history_M)
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"$M_k$")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    return ax
