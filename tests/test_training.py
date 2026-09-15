"""Early-stopping helper tests."""

from __future__ import annotations

import numpy as np

from cubic_reg.problems import SoftmaxRegression
from cubic_reg.training import arc_krylov_early_stop


def test_arc_early_stop_smoke():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((80, 6))
    y = rng.integers(0, 3, size=80)
    Xtr, ytr = X[:60], y[:60]
    Xv, yv = X[60:], y[60:]
    prob = SoftmaxRegression(Xtr, ytr, n_classes=3, l2=1e-3)
    r = arc_krylov_early_stop(
        prob, Xv, yv, x0=np.zeros(prob.dim), max_iter=15, patience=4, krylov_dim=10, m_max=20
    )
    assert r.nit >= 1
    assert 0.0 <= r.best_val_acc <= 1.0
    assert len(r.history_val_acc) == r.nit
