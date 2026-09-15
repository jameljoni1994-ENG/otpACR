"""Matrix-free analytic problems for large-scale Krylov experiments."""

from __future__ import annotations

from typing import Optional

import numpy as np

from cubic_reg.problems.base import Problem


class MatrixFreeQuadratic(Problem):
    """f(x) = 0.5 x^T A x - b^T x with A = Q diag(λ) Q^T applied matrix-free.

    For large n we avoid forming A. Q is implicit Haar-like via a fixed random
    orthogonal transform implemented as: signed permutation + DCT-style mixing
    is heavy; instead we use A = D + u u^T (diagonal + rank-one) which needs
    only O(n) storage and gives nontrivial Krylov behavior.
    """

    def __init__(
        self,
        n: int = 10_000,
        condition: float = 100.0,
        seed: int = 0,
        name: Optional[str] = None,
    ):
        super().__init__(name or f"mf_quadratic_n{n}_kappa{condition:g}")
        rng = np.random.default_rng(seed)
        self._n = n
        self.condition = condition
        # Diagonal spectrum from 1 to kappa
        self.diag = np.linspace(1.0, condition, n)
        # Rank-one bump
        self.u = rng.standard_normal(n)
        self.u /= np.linalg.norm(self.u)
        self.rho = 0.5 * (condition - 1.0)  # strength of rank-one
        self.b = rng.standard_normal(n)
        # x* solves A x = b; use CG once for reference (matrix-free)
        self._x_star = self._solve_cg(self.b, tol=1e-12, maxiter=min(5 * n, 20000))
        self._f_star = float(0.5 * self._x_star @ self._Ax(self._x_star) - self.b @ self._x_star)

    @property
    def dim(self) -> int:
        return self._n

    def _Ax(self, v: np.ndarray) -> np.ndarray:
        return self.diag * v + self.rho * self.u * (self.u @ v)

    def _solve_cg(self, b: np.ndarray, tol: float = 1e-10, maxiter: int = 10000) -> np.ndarray:
        x = np.zeros_like(b)
        r = b - self._Ax(x)
        p = r.copy()
        rs_old = float(r @ r)
        if rs_old == 0.0:
            return x
        for _ in range(maxiter):
            Ap = self._Ax(p)
            alpha = rs_old / float(p @ Ap)
            x = x + alpha * p
            r = r - alpha * Ap
            rs_new = float(r @ r)
            if np.sqrt(rs_new) < tol:
                break
            p = r + (rs_new / rs_old) * p
            rs_old = rs_new
        return x

    def f(self, x: np.ndarray) -> float:
        self.counters.f += 1
        return float(0.5 * x @ self._Ax(x) - self.b @ x)

    def grad(self, x: np.ndarray) -> np.ndarray:
        self.counters.grad += 1
        return self._Ax(x) - self.b

    def hess(self, x: np.ndarray) -> np.ndarray:
        """Dense Hessian — only for small n debugging."""
        if self._n > 2000:
            raise NotImplementedError("Dense hess unavailable for large matrix-free quadratic")
        self.counters.hess += 1
        return np.diag(self.diag) + self.rho * np.outer(self.u, self.u)

    def hvp(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        self.counters.hvp += 1
        return self._Ax(v)

    @property
    def f_star(self) -> float:
        return self._f_star

    @property
    def x_star(self) -> np.ndarray:
        return self._x_star.copy()
