"""Hessian-vector product utilities."""

from __future__ import annotations

from typing import Callable

import numpy as np

from cubic_reg.problems.base import Problem


def finite_diff_hvp(
    f_grad: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    v: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """Central finite-difference Hessian-vector product."""
    nrm = np.linalg.norm(v)
    if nrm == 0:
        return np.zeros_like(v)
    h = eps / nrm
    return (f_grad(x + h * v) - f_grad(x - h * v)) / (2.0 * h)


def make_hvp(problem: Problem, use_finite_diff: bool = False) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Return an HVP callable that increments problem counters appropriately."""

    if use_finite_diff:

        def hvp_fd(x: np.ndarray, v: np.ndarray) -> np.ndarray:
            problem.counters.hvp += 1

            def g(z: np.ndarray) -> np.ndarray:
                # finite-diff HVP uses two gradient evaluations
                return problem.grad(z)

            return finite_diff_hvp(g, x, v)

        return hvp_fd

    return problem.hvp
