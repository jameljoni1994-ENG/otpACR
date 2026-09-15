"""Phase-9: validation early-stopping for ARC-Krylov vs L-BFGS on MNIST softmax."""

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

from run_phase8 import load_mnist_multiclass
from cubic_reg.problems import SoftmaxRegression
from cubic_reg.training import arc_krylov_early_stop, lbfgs_early_stop
from cubic_reg.solvers import arc, quasi_newton

FIGDIR = Path(__file__).resolve().parent / "figures"
RESDIR = Path(__file__).resolve().parent / "results"


def _save(fig, name: str) -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    path = FIGDIR / name
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


def split_train_val(X, y, val_frac=0.2, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    n_val = max(200, int(len(y) * val_frac))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    return X[tr_idx], y[tr_idx], X[val_idx], y[val_idx]


def _pack(res, Xte, yte, C=10):
    te = SoftmaxRegression(Xte, yte, n_classes=C, l2=0.0)
    return {
        "f_train": res.f_train,
        "grad_norm": res.grad_norm,
        "nit": res.nit,
        "time_sec": res.time_sec,
        "best_iter": res.best_iter,
        "val_acc_best": res.best_val_acc,
        "test_acc": te.accuracy(res.x),
        "test_loss": te.f(res.x),
        "n_hvp": res.n_hvp,
        "n_grad": res.n_grad,
        "rejected_steps": res.rejected_steps,
        "message": res.message,
        "history_f": res.history_f,
        "history_grad_norm": res.history_grad_norm,
        "history_val_acc": res.history_val_acc,
        "history_M": res.history_M,
    }


def main() -> None:
    RESDIR.mkdir(parents=True, exist_ok=True)
    # Larger sample; then split train/val from training portion of phase8 loader
    full, Xte, yte = load_mnist_multiclass(max_samples=10000, seed=0)
    Xtr, ytr, Xv, yv = split_train_val(full.X, full.y, val_frac=0.2, seed=1)
    prob = SoftmaxRegression(Xtr, ytr, n_classes=10, l2=1e-4, name="mnist_softmax_es")
    print(f"N_train={prob.N} N_val={len(yv)} N_test={len(yte)} dim={prob.dim}", flush=True)

    x0 = np.zeros(prob.dim)

    r_arc_es = arc_krylov_early_stop(
        prob, Xv, yv, x0=x0.copy(), max_iter=50, patience=6, krylov_dim=40, m_max=80
    )
    print(
        f"ARC-ES: val_acc={r_arc_es.best_val_acc:.3f} "
        f"test={SoftmaxRegression(Xte,yte,10,l2=0).accuracy(r_arc_es.x):.3f} "
        f"f={r_arc_es.f_train:.4f} t={r_arc_es.time_sec:.2f} best_it={r_arc_es.best_iter}",
        flush=True,
    )

    r_lbfgs_es = lbfgs_early_stop(
        SoftmaxRegression(Xtr, ytr, n_classes=10, l2=1e-4),
        Xv,
        yv,
        x0=x0.copy(),
        max_iter=80,
        patience=6,
    )
    print(
        f"LBFGS-ES: val_acc={r_lbfgs_es.best_val_acc:.3f} "
        f"test={SoftmaxRegression(Xte,yte,10,l2=0).accuracy(r_lbfgs_es.x):.3f} "
        f"f={r_lbfgs_es.f_train:.4f} t={r_lbfgs_es.time_sec:.2f} best_it={r_lbfgs_es.best_iter}",
        flush=True,
    )

    # Baselines without ES (full run)
    prob2 = SoftmaxRegression(Xtr, ytr, n_classes=10, l2=1e-4)
    r_arc_full = arc.minimize(
        prob2, x0=x0.copy(), M0=1.0, eps=1e-4, max_iter=50, mode="krylov", krylov_dim=40
    )
    acc_arc_full = SoftmaxRegression(Xte, yte, 10, l2=0).accuracy(r_arc_full.x)
    print(
        f"ARC-full: test={acc_arc_full:.3f} f={r_arc_full.f:.4f} t={r_arc_full.time_sec:.2f}",
        flush=True,
    )

    prob3 = SoftmaxRegression(Xtr, ytr, n_classes=10, l2=1e-4)
    r_lbfgs_full = quasi_newton.minimize_lbfgs_plain(
        prob3, x0=x0.copy(), eps=1e-4, max_iter=80, memory=20
    )
    acc_lbfgs_full = SoftmaxRegression(Xte, yte, 10, l2=0).accuracy(r_lbfgs_full.x)
    print(
        f"LBFGS-full: test={acc_lbfgs_full:.3f} f={r_lbfgs_full.f:.4f} t={r_lbfgs_full.time_sec:.2f}",
        flush=True,
    )

    payload = {
        "N_train": prob.N,
        "N_val": len(yv),
        "N_test": len(yte),
        "dim": prob.dim,
        "ARC_Krylov_ES": _pack(r_arc_es, Xte, yte),
        "LBFGS_ES": _pack(r_lbfgs_es, Xte, yte),
        "ARC_full": {
            "f_train": r_arc_full.f,
            "grad_norm": r_arc_full.grad_norm,
            "nit": r_arc_full.nit,
            "time_sec": r_arc_full.time_sec,
            "test_acc": acc_arc_full,
            "history_f": r_arc_full.history_f,
            "history_grad_norm": r_arc_full.history_grad_norm,
        },
        "LBFGS_full": {
            "f_train": r_lbfgs_full.f,
            "grad_norm": r_lbfgs_full.grad_norm,
            "nit": r_lbfgs_full.nit,
            "time_sec": r_lbfgs_full.time_sec,
            "test_acc": acc_lbfgs_full,
            "history_f": r_lbfgs_full.history_f,
            "history_grad_norm": r_lbfgs_full.history_grad_norm,
        },
    }
    path = RESDIR / "phase9.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {path}")

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for name, key in [("ARC-ES", "ARC_Krylov_ES"), ("LBFGS-ES", "LBFGS_ES")]:
        r = payload[key]
        axes[0].plot(r["history_val_acc"], label=name)
        t = np.linspace(0, r["time_sec"], num=len(r["history_f"]))
        axes[1].semilogy(t, np.maximum(r["history_f"], 1e-16), label=name)
    axes[0].set_title("validation accuracy")
    axes[0].set_xlabel("iteration")
    axes[1].set_title("train f vs time")
    axes[1].set_xlabel("time (s)")
    names = ["ARC-ES", "LBFGS-ES", "ARC-full", "LBFGS-full"]
    accs = [
        payload["ARC_Krylov_ES"]["test_acc"],
        payload["LBFGS_ES"]["test_acc"],
        payload["ARC_full"]["test_acc"],
        payload["LBFGS_full"]["test_acc"],
    ]
    axes[2].bar(names, accs, color="#55A868")
    axes[2].tick_params(axis="x", rotation=25)
    axes[2].set_ylim(0.8, 1.0)
    axes[2].set_title("test accuracy")
    for ax in axes[:2]:
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[2].grid(True, axis="y", alpha=0.3)
    fig.suptitle("Early-stopping on MNIST softmax")
    plt.tight_layout()
    _save(fig, "24_early_stopping.png")

    print("\n=== Summary ===", flush=True)
    for name in names:
        key = {
            "ARC-ES": "ARC_Krylov_ES",
            "LBFGS-ES": "LBFGS_ES",
            "ARC-full": "ARC_full",
            "LBFGS-full": "LBFGS_full",
        }[name]
        r = payload[key]
        print(
            f"{name:12} test={r['test_acc']:.3f} f={r['f_train']:.4f} t={r['time_sec']:.2f}",
            flush=True,
        )
    print("Phase-9 done.")


if __name__ == "__main__":
    main()
