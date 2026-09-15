"""Phase-6: forced Krylov growth, persistent worker pool, optional real MNIST."""

from __future__ import annotations

import gzip
import json
import struct
import sys
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cubic_reg.problems import Quadratic, Rosenbrock, make_mnist_like_logistic
from cubic_reg.problems.ml import FiniteSumLogistic
from cubic_reg.solvers import krylov, stochastic
from cubic_reg.solvers.sketch_krylov import hvp_sketch_basis

FIGDIR = Path(__file__).resolve().parent / "figures"
RESDIR = Path(__file__).resolve().parent / "results"
DATADIR = Path(__file__).resolve().parent / "data"


def _save(fig, name: str) -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    path = FIGDIR / name
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


def forced_theta_growth() -> dict:
    """Start with krylov_dim=3 so adaptive policies must grow m."""
    cases = {
        "quadratic_kappa1000": (Quadratic(n=120, condition=1000.0, seed=0), np.ones(120), 1.0),
        "rosenbrock": (Rosenbrock(n=8), np.full(8, -1.2), 20.0),
    }
    policies = {
        "fixed_m3": dict(adaptive_tol=False, inexact_tol=0.5, krylov_dim=3, m_max=3),
        "fixed_m20": dict(adaptive_tol=False, inexact_tol=0.5, krylov_dim=20, m_max=20),
        "adapt_1_over_k": dict(adaptive_tol=True, inexact_tol=0.5, tol_power=1.0, krylov_dim=3, m_max=40),
        "adapt_1_over_k2": dict(adaptive_tol=True, inexact_tol=0.5, tol_power=2.0, krylov_dim=3, m_max=40),
    }
    out = {}
    for cname, (prob, x0, M) in cases.items():
        out[cname] = {}
        for pol, kwargs in policies.items():
            r = krylov.minimize(
                prob,
                x0=x0.copy(),
                M=M,
                eps=1e-6,
                max_iter=60,
                **kwargs,
            )
            ms = getattr(r, "history_m", [])
            out[cname][pol] = {
                "nit": r.nit,
                "f": r.f,
                "grad_norm": r.grad_norm,
                "n_hvp": r.n_hvp,
                "time_sec": r.time_sec,
                "mean_m": float(np.mean(ms)) if ms else None,
                "max_m": int(np.max(ms)) if ms else None,
                "history_m": ms,
                "history_grad_norm": r.history_grad_norm,
                "success": r.success,
            }
            print(
                f"{cname}/{pol}: nit={r.nit} ||g||={r.grad_norm:.2e} "
                f"hvp={r.n_hvp} mean_m={out[cname][pol]['mean_m']}",
                flush=True,
            )
    return out


def fig_forced_theta(data: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax_row, cname in zip(axes, data.keys()):
        for pol, res in data[cname].items():
            ax_row[0].semilogy(np.maximum(res["history_grad_norm"], 1e-16), label=pol)
            if res["history_m"]:
                ax_row[1].plot(res["history_m"], label=pol)
        ax_row[0].set_title(f"{cname}: grad")
        ax_row[1].set_title(f"{cname}: Krylov m")
        for ax in ax_row:
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
            ax.set_xlabel("iteration")
    fig.suptitle("Forced growth: krylov_dim starts at 3")
    plt.tight_layout()
    _save(fig, "18_forced_theta_growth.png")


def _sketch_worker(payload):
    n, seed, rank, x, diag, u, rho = payload

    def hvp(_x, v):
        return diag * v + rho * u * (u @ v)

    rng = np.random.default_rng(seed)
    return hvp_sketch_basis(hvp, x, rank, rng)


def persistent_pool_comm(
    n: int = 4000,
    rank: int = 30,
    workers_list=(1, 2, 4),
    rounds: int = 8,
) -> dict:
    """Reuse one ProcessPool across rounds to amortize spawn cost."""
    rng = np.random.default_rng(0)
    diag = np.linspace(1.0, 100.0, n)
    u = rng.standard_normal(n)
    u /= np.linalg.norm(u)
    rho = 20.0
    x = np.zeros(n)
    rows = []

    for nw in workers_list:
        payloads = [
            (n, 2000 + w, rank, x, diag, u * (1.0 + 0.01 * w), rho) for w in range(nw)
        ]
        # cold start (includes spawn)
        t_cold0 = time.perf_counter()
        with ProcessPoolExecutor(max_workers=nw) as pool:
            # warm round
            list(pool.map(_sketch_worker, payloads))
            t_warm0 = time.perf_counter()
            bytes_acc = 0
            for _ in range(rounds):
                parts = list(pool.map(_sketch_worker, payloads))
                bytes_acc += sum(p[2] for p in parts)
                # cheap aggregate proxy
                Q0, B0, _ = parts[0]
                _ = B0
            t_end = time.perf_counter()
        cold = t_warm0 - t_cold0
        steady = (t_end - t_warm0) / rounds
        rows.append(
            {
                "workers": nw,
                "rank": rank,
                "n": n,
                "rounds": rounds,
                "cold_start_sec": cold,
                "steady_sec_per_round": steady,
                "bytes_per_round": bytes_acc / rounds,
                "speedup_steady_vs_1": None,
            }
        )
        print(
            f"pool w={nw}: cold={cold:.3f}s steady={steady:.3f}s/round bytes~{bytes_acc/rounds:.0f}",
            flush=True,
        )

    t1 = next(r["steady_sec_per_round"] for r in rows if r["workers"] == 1)
    for r in rows:
        r["speedup_steady_vs_1"] = t1 / max(r["steady_sec_per_round"], 1e-12)
    return {"rows": rows}


def fig_persistent_pool(data: dict) -> None:
    rows = data["rows"]
    ws = [r["workers"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].plot(ws, [r["cold_start_sec"] for r in rows], "o-", label="cold start")
    axes[0].plot(ws, [r["steady_sec_per_round"] for r in rows], "s-", label="steady / round")
    axes[0].set_xlabel("workers")
    axes[0].set_ylabel("seconds")
    axes[0].set_title("Persistent pool timing")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(ws, [r["speedup_steady_vs_1"] for r in rows], "o-")
    axes[1].axhline(1.0, color="gray", ls=":")
    axes[1].set_xlabel("workers")
    axes[1].set_ylabel("speedup vs 1 (steady)")
    axes[1].set_title("Steady-state sketch speedup")
    axes[1].grid(True, alpha=0.3)
    _save(fig, "19_persistent_pool.png")


def _download(url: str, dest: Path, timeout: int = 60) -> bool:
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {url} ...", flush=True)
        urllib.request.urlretrieve(url, dest)  # noqa: S310
        return True
    except Exception as exc:  # network may be unavailable
        print(f"download failed: {exc}", flush=True)
        return False


def _load_idx_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(n, rows * cols).astype(np.float64) / 255.0


def _load_idx_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8).astype(np.int64)


def load_mnist_binary(
    digits=(3, 8),
    max_per_class: int = 1500,
    seed: int = 0,
) -> tuple[FiniteSumLogistic | None, str]:
    """Try real MNIST; fallback note if unavailable."""
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
            return None, "download_failed"

    X = _load_idx_images(paths["train-images-idx3-ubyte.gz"])
    y = _load_idx_labels(paths["train-labels-idx1-ubyte.gz"])
    d0, d1 = digits
    m0 = np.where(y == d0)[0][:max_per_class]
    m1 = np.where(y == d1)[0][:max_per_class]
    idx = np.concatenate([m0, m1])
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    Xb = X[idx]
    yb = np.where(y[idx] == d0, -1.0, 1.0)
    Xb = (Xb - Xb.mean(axis=0)) / (Xb.std(axis=0) + 1e-8)
    prob = FiniteSumLogistic(Xb, yb, l2=1e-4, name=f"mnist_{d0}v{d1}")
    return prob, "real_mnist"


def mnist_experiment() -> dict:
    prob, source = load_mnist_binary()
    if prob is None:
        print("Using MNIST-like fallback", flush=True)
        prob = make_mnist_like_logistic(n_samples=3000, n_features=784, seed=0)
        source = "mnist_like_fallback"

    x0 = np.zeros(prob.dim)
    methods = {
        "SubCR": stochastic.minimize(
            prob, x0=x0, M=1.0, eps=2e-3, max_iter=35, batch_grad=256, batch_hess=64
        ),
        "Krylov-CR": krylov.minimize(
            prob, x0=x0, M=1.0, eps=2e-3, max_iter=20, krylov_dim=25, adaptive_tol=True, m_max=50
        ),
        "SGD": stochastic.minimize_sgd(prob, x0=x0, lr=0.05, batch=128, max_iter=120, eps=2e-3),
        "Adam": stochastic.minimize_adam(prob, x0=x0, lr=0.01, batch=128, max_iter=120, eps=2e-3),
    }
    out = {"source": source, "n_samples": prob.N, "dim": prob.dim, "methods": {}}
    for name, r in methods.items():
        out["methods"][name] = {
            "nit": r.nit,
            "time_sec": r.time_sec,
            "grad_norm": r.grad_norm,
            "f": r.f,
            "history_grad_norm": r.history_grad_norm,
            "n_hvp": r.n_hvp,
            "n_hess": r.n_hess,
        }
        print(f"[{source}] {name:10} t={r.time_sec:.3f} ||g||={r.grad_norm:.3e} f={r.f:.5f}", flush=True)
    return out


def fig_mnist(data: dict) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for name, r in data["methods"].items():
        t = np.linspace(0, r["time_sec"], num=len(r["history_grad_norm"]))
        ax.semilogy(t, np.maximum(r["history_grad_norm"], 1e-16), label=name)
    ax.set_xlabel("wall time (s)")
    ax.set_ylabel(r"$\|\nabla f\|$")
    ax.set_title(f"MNIST binary ({data['source']}, d={data['dim']})")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    _save(fig, "20_mnist_real_or_fallback.png")


def main() -> None:
    RESDIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "forced_theta": forced_theta_growth(),
        "persistent_pool": persistent_pool_comm(),
        "mnist": mnist_experiment(),
    }
    path = RESDIR / "phase6.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {path}")
    fig_forced_theta(payload["forced_theta"])
    fig_persistent_pool(payload["persistent_pool"])
    fig_mnist(payload["mnist"])
    print("Phase-6 done.")


if __name__ == "__main__":
    main()
