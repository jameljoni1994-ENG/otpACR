"""Training helpers: validation early-stopping wrappers around existing solvers."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from cubic_reg.metrics import OptimizationResult
from cubic_reg.problems.base import Problem
from cubic_reg.problems.ml import SoftmaxRegression
from cubic_reg.solvers.krylov import solve_inexact_step, theta_schedule
from cubic_reg.subproblem import model_decrease


@dataclass
class EarlyStopResult:
    x: np.ndarray
    f_train: float
    grad_norm: float
    nit: int
    time_sec: float
    best_val_acc: float
    best_iter: int
    history_f: List[float] = field(default_factory=list)
    history_grad_norm: List[float] = field(default_factory=list)
    history_val_acc: List[float] = field(default_factory=list)
    history_val_loss: List[float] = field(default_factory=list)
    history_M: List[float] = field(default_factory=list)
    n_hvp: int = 0
    n_grad: int = 0
    n_f: int = 0
    rejected_steps: int = 0
    message: str = ""


def _val_metrics(prob: SoftmaxRegression, x: np.ndarray, Xv: np.ndarray, yv: np.ndarray):
    hold = SoftmaxRegression(Xv, yv, n_classes=prob.C, l2=0.0)
    # loss without touching training counters much — create fresh
    loss = hold.f(x) - 0.0  # l2=0
    acc = hold.accuracy(x)
    return float(loss), float(acc)


def arc_krylov_early_stop(
    prob: SoftmaxRegression,
    Xv: np.ndarray,
    yv: np.ndarray,
    x0: Optional[np.ndarray] = None,
    M0: float = 1.0,
    max_iter: int = 60,
    krylov_dim: int = 40,
    m_max: int = 80,
    patience: int = 8,
    eta1: float = 0.1,
    eta2: float = 0.9,
    gamma1: float = 0.5,
    gamma2: float = 2.0,
    eps: float = 1e-5,
    monitor: str = "acc",  # "acc" or "loss"
) -> EarlyStopResult:
    """ARC with Krylov steps + restore weights of best validation metric."""
    prob.reset_counters()
    x = np.zeros(prob.dim) if x0 is None else np.asarray(x0, dtype=float).copy()
    M = float(M0)
    best_x = x.copy()
    best_score = -np.inf if monitor == "acc" else np.inf
    best_iter = 0
    bad = 0
    rejected = 0
    hist_f, hist_g, hist_va, hist_vl, hist_M = [], [], [], [], []

    t0 = time.perf_counter()
    fx = prob.f(x)

    for k in range(max_iter):
        g = prob.grad(x)
        gnorm = float(np.linalg.norm(g))
        vl, va = _val_metrics(prob, x, Xv, yv)
        hist_f.append(fx)
        hist_g.append(gnorm)
        hist_va.append(va)
        hist_vl.append(vl)
        hist_M.append(M)

        score = va if monitor == "acc" else -vl
        if score > best_score + 1e-8:
            best_score = score
            best_x = x.copy()
            best_iter = k
            bad = 0
        else:
            bad += 1

        if gnorm <= eps:
            break
        if bad >= patience:
            break

        theta = theta_schedule(k, 0.5, 1.0)
        s, lam, m_used, resid, ok = solve_inexact_step(
            prob, x, g, M, krylov_dim, min(prob.dim, m_max), theta
        )
        if not ok and np.linalg.norm(s) == 0:
            M = min(1e16, M * gamma2)
            rejected += 1
            continue

        Hs = prob.hvp(x, s)
        sn = float(np.linalg.norm(s))
        md = float(-(g @ s + 0.5 * s @ Hs + (M / 6.0) * sn**3))
        if md <= 0:
            M = min(1e16, M * gamma2)
            rejected += 1
            continue

        x_trial = x + s
        f_trial = prob.f(x_trial)
        rho = (fx - f_trial) / md
        if rho >= eta1:
            x = x_trial
            fx = f_trial
            if rho >= eta2:
                M = max(1e-16, gamma1 * M)
        else:
            rejected += 1
            M = min(1e16, gamma2 * M)

    # final metrics at best_x
    fx_b = prob.f(best_x)
    g_b = float(np.linalg.norm(prob.grad(best_x)))
    vl_b, va_b = _val_metrics(prob, best_x, Xv, yv)
    c = prob.counters
    return EarlyStopResult(
        x=best_x,
        f_train=fx_b,
        grad_norm=g_b,
        nit=len(hist_f),
        time_sec=time.perf_counter() - t0,
        best_val_acc=va_b if monitor == "acc" else va_b,
        best_iter=best_iter,
        history_f=hist_f,
        history_grad_norm=hist_g,
        history_val_acc=hist_va,
        history_val_loss=hist_vl,
        history_M=hist_M,
        n_hvp=c.hvp,
        n_grad=c.grad,
        n_f=c.f,
        rejected_steps=rejected,
        message=f"early_stop@{best_iter} patience={patience} monitor={monitor}",
    )


def lbfgs_early_stop(
    prob: SoftmaxRegression,
    Xv: np.ndarray,
    yv: np.ndarray,
    x0: Optional[np.ndarray] = None,
    max_iter: int = 80,
    patience: int = 8,
    eps: float = 1e-5,
    memory: int = 20,
) -> EarlyStopResult:
    """Scipy L-BFGS-B with periodic validation checks via callback."""
    from scipy.optimize import minimize as sp_minimize

    prob.reset_counters()
    x0 = np.zeros(prob.dim) if x0 is None else np.asarray(x0, dtype=float).copy()
    hist_f, hist_g, hist_va, hist_vl = [], [], [], []
    state = {"best_x": x0.copy(), "best_score": -np.inf, "best_iter": 0, "bad": 0, "k": 0}
    t0 = time.perf_counter()

    def fun(x):
        return prob.f(x)

    def jac(x):
        return prob.grad(x)

    def callback(xk):
        k = state["k"]
        state["k"] = k + 1
        fx = prob.f(xk)
        gnorm = float(np.linalg.norm(prob.grad(xk)))
        vl, va = _val_metrics(prob, xk, Xv, yv)
        hist_f.append(fx)
        hist_g.append(gnorm)
        hist_va.append(va)
        hist_vl.append(vl)
        if va > state["best_score"] + 1e-8:
            state["best_score"] = va
            state["best_x"] = np.asarray(xk, dtype=float).copy()
            state["best_iter"] = k
            state["bad"] = 0
        else:
            state["bad"] += 1
        if state["bad"] >= patience or gnorm <= eps:
            raise StopIteration

    try:
        sp_minimize(
            fun,
            x0,
            jac=jac,
            method="L-BFGS-B",
            callback=callback,
            options={"maxiter": max_iter, "gtol": eps},
        )
    except StopIteration:
        pass

    best_x = state["best_x"]
    fx_b = prob.f(best_x)
    g_b = float(np.linalg.norm(prob.grad(best_x)))
    _, va_b = _val_metrics(prob, best_x, Xv, yv)
    c = prob.counters
    return EarlyStopResult(
        x=best_x,
        f_train=fx_b,
        grad_norm=g_b,
        nit=len(hist_f),
        time_sec=time.perf_counter() - t0,
        best_val_acc=va_b,
        best_iter=state["best_iter"],
        history_f=hist_f,
        history_grad_norm=hist_g,
        history_val_acc=hist_va,
        history_val_loss=hist_vl,
        n_hvp=c.hvp,
        n_grad=c.grad,
        n_f=c.f,
        message=f"early_stop@{state['best_iter']} patience={patience}",
    )
