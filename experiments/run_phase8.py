"""Phase-8: multiclass MNIST softmax + L-BFGS / ARC-Krylov / Adam comparison."""

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

from run_phase6 import _load_idx_images, _load_idx_labels, DATADIR, _download
from cubic_reg.problems import SoftmaxRegression
from cubic_reg.solvers import krylov, arc, quasi_newton, first_order

FIGDIR = Path(__file__).resolve().parent / "figures"
RESDIR = Path(__file__).resolve().parent / "results"


def _save(fig, name: str) -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    path = FIGDIR / name
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


def load_mnist_multiclass(max_samples: int = 8000, seed: int = 0):
    base = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    files = {
        "train-images-idx3-ubyte.gz": base + "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz": base + "train-labels-idx1-ubyte.gz",
    }
    paths = {}
    for name, url in files.items():
        dest = DATADIR / name
        if dest.exists() or _download(url, dest):
            paths[name] = dest
        else:
            raise RuntimeError("MNIST download failed")

    X = _load_idx_images(paths["train-images-idx3-ubyte.gz"])
    y = _load_idx_labels(paths["train-labels-idx1-ubyte.gz"])
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y), size=min(max_samples, len(y)), replace=False)
    X, y = X[idx], y[idx]
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    # hold out 20% for accuracy
    n_test = max(500, len(y) // 5)
    Xte, yte = X[:n_test], y[:n_test]
    Xtr, ytr = X[n_test:], y[n_test:]
    prob = SoftmaxRegression(Xtr, ytr, n_classes=10, l2=1e-4, name="mnist_softmax")
    return prob, Xte, yte


def _pack(r, prob, Xte, yte, extra=None):
    acc_tr = prob.accuracy(r.x)
    acc_te = SoftmaxRegression(Xte, yte, n_classes=10).accuracy(r.x)
    # don't pollute counters of training problem further
    d = {
        "nit": r.nit,
        "time_sec": r.time_sec,
        "grad_norm": r.grad_norm,
        "f": r.f,
        "n_hvp": r.n_hvp,
        "n_grad": r.n_grad,
        "n_f": r.n_f,
        "acc_train": acc_tr,
        "acc_test": acc_te,
        "history_f": r.history_f,
        "history_grad_norm": r.history_grad_norm,
        "success": r.success,
        "message": r.message,
    }
    if extra:
        d.update(extra)
    return d


def run_compare(prob, Xte, yte) -> dict:
    x0 = np.zeros(prob.dim)
    print(f"Softmax dim={prob.dim} N={prob.N}", flush=True)

    r_kry = krylov.minimize(
        prob,
        x0=x0.copy(),
        M=1.0,
        eps=1e-4,
        max_iter=40,
        krylov_dim=40,
        adaptive_tol=True,
        m_max=80,
    )
    print(
        f"Krylov-CR: f={r_kry.f:.4f} ||g||={r_kry.grad_norm:.2e} "
        f"acc_te={SoftmaxRegression(Xte,yte,10).accuracy(r_kry.x):.3f} t={r_kry.time_sec:.2f}",
        flush=True,
    )

    r_arc = arc.minimize(
        prob,
        x0=x0.copy(),
        M0=1.0,
        eps=1e-4,
        max_iter=50,
        mode="krylov",
        krylov_dim=40,
    )
    print(
        f"ARC-Krylov: f={r_arc.f:.4f} ||g||={r_arc.grad_norm:.2e} "
        f"acc_te={SoftmaxRegression(Xte,yte,10).accuracy(r_arc.x):.3f} t={r_arc.time_sec:.2f}",
        flush=True,
    )

    r_lbfgs = quasi_newton.minimize_lbfgs_plain(
        prob, x0=x0.copy(), eps=1e-4, max_iter=80, memory=20
    )
    print(
        f"L-BFGS: f={r_lbfgs.f:.4f} ||g||={r_lbfgs.grad_norm:.2e} "
        f"acc_te={SoftmaxRegression(Xte,yte,10).accuracy(r_lbfgs.x):.3f} t={r_lbfgs.time_sec:.2f}",
        flush=True,
    )

    # Adam-style via GD with small steps is weak; use Nesterov FO with heuristic step
    # Estimate Lipschitz roughly via one HVP power iteration
    v = np.random.default_rng(0).standard_normal(prob.dim)
    v /= np.linalg.norm(v)
    Hv = prob.hvp(x0, v)
    L_est = float(np.linalg.norm(Hv)) + 1e-6
    step = 1.0 / L_est
    r_nag = first_order.minimize_nesterov(
        prob, x0=x0.copy(), step=step, eps=1e-4, max_iter=300
    )
    print(
        f"Nesterov: f={r_nag.f:.4f} ||g||={r_nag.grad_norm:.2e} "
        f"acc_te={SoftmaxRegression(Xte,yte,10).accuracy(r_nag.x):.3f} t={r_nag.time_sec:.2f} step={step:.2e}",
        flush=True,
    )

    return {
        "Krylov-CR": _pack(r_kry, prob, Xte, yte),
        "ARC-Krylov": _pack(r_arc, prob, Xte, yte),
        "L-BFGS": _pack(r_lbfgs, prob, Xte, yte),
        "Nesterov": _pack(r_nag, prob, Xte, yte, {"step": step, "L_est": L_est}),
    }


def fig_compare(results: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for name, r in results.items():
        t = np.linspace(0, r["time_sec"], num=max(len(r["history_f"]), 1))
        axes[0].semilogy(t, np.maximum(r["history_f"], 1e-16), label=name)
        axes[1].semilogy(t, np.maximum(r["history_grad_norm"], 1e-16), label=name)
    names = list(results.keys())
    axes[2].bar(names, [results[n]["acc_test"] for n in names], color="#4C72B0")
    axes[2].tick_params(axis="x", rotation=30)
    axes[2].set_ylim(0, 1)
    axes[2].set_title("test accuracy")
    axes[0].set_title("f vs time")
    axes[1].set_title(r"$\|\nabla f\|$ vs time")
    for ax in axes[:2]:
        ax.set_xlabel("wall time (s)")
        ax.legend(fontsize=7)
        ax.grid(True, which="both", alpha=0.3)
    axes[2].grid(True, axis="y", alpha=0.3)
    fig.suptitle("MNIST multiclass softmax (10 classes)")
    plt.tight_layout()
    _save(fig, "23_mnist_multiclass.png")


def main() -> None:
    RESDIR.mkdir(parents=True, exist_ok=True)
    prob, Xte, yte = load_mnist_multiclass(max_samples=8000, seed=0)
    results = run_compare(prob, Xte, yte)
    payload = {
        "N_train": prob.N,
        "N_test": len(yte),
        "dim": prob.dim,
        "n_classes": 10,
        "methods": results,
    }
    path = RESDIR / "phase8.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {path}")
    fig_compare(results)
    # summary table
    print("\n=== Summary ===", flush=True)
    print(f"{'method':12} {'f':>10} {'||g||':>10} {'acc_te':>8} {'time':>8}", flush=True)
    for name, r in results.items():
        print(
            f"{name:12} {r['f']:10.4f} {r['grad_norm']:10.2e} {r['acc_test']:8.3f} {r['time_sec']:8.2f}",
            flush=True,
        )
    print("Phase-8 done.")


if __name__ == "__main__":
    main()
