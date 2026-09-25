"""Phase-10: full MNIST scale-up, l2 path + ES, and sketch scaling at large n."""

from __future__ import annotations

import argparse
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
from run_phase9 import split_train_val, _pack
from cubic_reg.problems import SoftmaxRegression, MatrixFreeQuadratic
from cubic_reg.training import arc_krylov_early_stop, lbfgs_early_stop, l2_path_early_stop
from cubic_reg.solvers import sketch_krylov, quasi_newton

FIGDIR = Path(__file__).resolve().parent / "figures"
RESDIR = Path(__file__).resolve().parent / "results"


def _save(fig, name: str) -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    path = FIGDIR / name
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


def mnist_scale(max_samples: int, max_iter: int = 40, patience: int = 6) -> dict:
    full, Xte, yte = load_mnist_multiclass(max_samples=max_samples, seed=0)
    Xtr, ytr, Xv, yv = split_train_val(full.X, full.y, val_frac=0.2, seed=1)
    # use same l2 as prior phases for the primary run
    prob = SoftmaxRegression(Xtr, ytr, n_classes=10, l2=1e-4, name="mnist_softmax_p10")
    print(
        f"[mnist] N_train={prob.N} N_val={len(yv)} N_test={len(yte)} dim={prob.dim}",
        flush=True,
    )
    x0 = np.zeros(prob.dim)
    r_arc = arc_krylov_early_stop(
        prob, Xv, yv, x0=x0.copy(), max_iter=max_iter, patience=patience, krylov_dim=40, m_max=80
    )
    r_lbfgs = lbfgs_early_stop(
        prob, Xv, yv, x0=x0.copy(), max_iter=max_iter + 20, patience=patience
    )
    # matrix-free QN-CR short smoke on a subset of iters (no ES wrapper)
    r_qn = quasi_newton.minimize(
        SoftmaxRegression(Xtr, ytr, n_classes=10, l2=1e-4),
        x0=x0.copy(),
        M=1.0,
        eps=1e-4,
        max_iter=min(25, max_iter),
        memory=20,
        matrix_free=True,
        krylov_dim=30,
        m_max=60,
    )
    out = {
        "N_train": int(prob.N),
        "N_val": int(len(yv)),
        "N_test": int(len(yte)),
        "dim": int(prob.dim),
        "max_samples": int(max_samples),
        "arc_es": _pack(r_arc, Xte, yte),
        "lbfgs_es": _pack(r_lbfgs, Xte, yte),
        "qn_mf": {
            "f": r_qn.f,
            "grad_norm": r_qn.grad_norm,
            "nit": r_qn.nit,
            "time_sec": r_qn.time_sec,
            "test_acc": SoftmaxRegression(Xte, yte, n_classes=10, l2=0.0).accuracy(r_qn.x),
        },
    }
    return out


def l2_path_experiment(max_samples: int = 8000) -> dict:
    full, Xte, yte = load_mnist_multiclass(max_samples=max_samples, seed=0)
    Xtr, ytr, Xv, yv = split_train_val(full.X, full.y, val_frac=0.2, seed=2)
    prob = SoftmaxRegression(Xtr, ytr, n_classes=10, l2=1e-3, name="mnist_l2_path")
    grid = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4]
    print(f"[l2_path] grid={grid} N_train={prob.N}", flush=True)
    r_arc = l2_path_early_stop(
        prob, Xv, yv, l2_grid=grid, solver="arc", max_iter=30, patience=5, krylov_dim=30, m_max=60
    )
    r_lbfgs = l2_path_early_stop(
        SoftmaxRegression(Xtr, ytr, n_classes=10, l2=1e-3),
        Xv,
        yv,
        l2_grid=grid,
        solver="lbfgs",
        max_iter=40,
        patience=5,
    )
    return {
        "l2_grid": grid,
        "arc_path": _pack(r_arc, Xte, yte),
        "lbfgs_path": _pack(r_lbfgs, Xte, yte),
    }


def sketch_scaling(dims: list[int] | None = None, sketch_rank: int = 20) -> dict:
    if dims is None:
        dims = [2_000, 8_000, 20_000]
    rows = []
    for n in dims:
        p = MatrixFreeQuadratic(n=n, condition=50.0, seed=0)
        x0 = np.zeros(p.dim)
        r = sketch_krylov.minimize(
            p,
            x0=x0,
            M0=1.0,
            eps=1e-4,
            max_iter=25,
            sketch_rank=min(sketch_rank, n // 10),
            krylov_dim=min(25, n // 20),
            use_krylov_refine=True,
        )
        rows.append(
            {
                "n": n,
                "nit": r.nit,
                "time_sec": r.time_sec,
                "grad_norm": r.grad_norm,
                "f": r.f,
                "n_hvp": r.n_hvp,
                "message": r.message,
            }
        )
        print(f"[sketch] n={n} time={r.time_sec:.3f}s ||g||={r.grad_norm:.3e}", flush=True)
    return {"ranks": sketch_rank, "rows": rows}


def make_figures(report: dict) -> None:
    # MNIST summary bars
    if "mnist" in report:
        m = report["mnist"]
        labels = ["ARC-ES", "L-BFGS-ES", "QN-MF"]
        accs = [
            m["arc_es"]["test_acc"],
            m["lbfgs_es"]["test_acc"],
            m["qn_mf"]["test_acc"],
        ]
        times = [
            m["arc_es"]["time_sec"],
            m["lbfgs_es"]["time_sec"],
            m["qn_mf"]["time_sec"],
        ]
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
        ax[0].bar(labels, accs, color=["#2c7fb8", "#41b6c4", "#a1dab4"])
        ax[0].set_ylabel("test accuracy")
        ax[0].set_title(f"MNIST softmax (N≈{m['N_train']})")
        ax[1].bar(labels, times, color=["#2c7fb8", "#41b6c4", "#a1dab4"])
        ax[1].set_ylabel("time (s)")
        ax[1].set_title("wall time")
        fig.tight_layout()
        _save(fig, "25_phase10_mnist.png")

    if "l2_path" in report:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        for key, color in [("arc_path", "#2c7fb8"), ("lbfgs_path", "#41b6c4")]:
            h = report["l2_path"][key].get("history_val_acc") or []
            if h:
                ax.plot(h, label=key.replace("_", "-"), color=color)
        ax.set_xlabel("outer iteration (path)")
        ax.set_ylabel("val accuracy")
        ax.set_title("l2 path + early stopping")
        ax.legend()
        fig.tight_layout()
        _save(fig, "26_phase10_l2_path.png")

    if "sketch" in report:
        rows = report["sketch"]["rows"]
        ns = [r["n"] for r in rows]
        ts = [r["time_sec"] for r in rows]
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        ax.loglog(ns, ts, "o-", color="#2c7fb8")
        ax.set_xlabel("dimension n")
        ax.set_ylabel("time (s)")
        ax.set_title("sketch–Krylov scaling")
        fig.tight_layout()
        _save(fig, "27_phase10_sketch_scale.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase-10 scale / l2-path / sketch experiments")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Use full MNIST train set (~60k) for the scale experiment",
    )
    parser.add_argument("--skip-mnist", action="store_true")
    parser.add_argument("--skip-l2", action="store_true")
    parser.add_argument("--skip-sketch", action="store_true")
    parser.add_argument("--quick", action="store_true", help="Smaller dims / fewer iters")
    args = parser.parse_args()

    RESDIR.mkdir(parents=True, exist_ok=True)
    out = RESDIR / "phase10.json"
    report: dict = {}
    if out.exists():
        try:
            report = json.loads(out.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            report = {}

    if not args.skip_mnist:
        n = 60000 if args.full else (4000 if args.quick else 12000)
        iters = 20 if args.quick else 40
        report["mnist"] = mnist_scale(max_samples=n, max_iter=iters, patience=5 if args.quick else 6)

    if not args.skip_l2:
        n_l2 = 3000 if args.quick else 8000
        report["l2_path"] = l2_path_experiment(max_samples=n_l2)

    if not args.skip_sketch:
        dims = [1_000, 3_000] if args.quick else [2_000, 8_000, 20_000]
        report["sketch"] = sketch_scaling(dims=dims, sketch_rank=16 if args.quick else 20)

    make_figures(report)
    # histories can be large; store compact summary for JSON
    compact = json.loads(
        json.dumps(
            report,
            default=lambda o: list(o)
            if hasattr(o, "__iter__") and not isinstance(o, (str, dict))
            else o,
        )
    )
    out.write_text(json.dumps(compact, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
