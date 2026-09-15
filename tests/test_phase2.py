"""Smoke tests for phase-2 experiment helpers (small sizes)."""

from __future__ import annotations

import numpy as np

from experiments.run_phase2 import high_dim_krylov, rate_comparison, multiprocess_distributed_logistic


def test_high_dim_krylov_small():
    data = high_dim_krylov(ns=[60, 80], krylov_dim=10, max_iter=5)
    assert len(data["rows"]) == 2
    assert data["rows"][0]["krylov"]["sec_per_iter"] > 0


def test_rate_comparison_small():
    data = rate_comparison(max_iter=15)
    assert "gap" in data["cr"] and len(data["cr"]["gap"]) >= 1
    assert np.isfinite(data["cr"]["f_star_est"])


def test_multiprocess_one_worker():
    data = multiprocess_distributed_logistic(
        n_workers_list=[1],
        n_samples=200,
        n_features=10,
        max_iter=5,
        eps=1e-2,
    )
    assert "workers_1" in data
    assert data["workers_1"]["nit"] >= 1
