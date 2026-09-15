"""Analytic test problems for cubic regularization experiments."""

from __future__ import annotations

from typing import Optional

import numpy as np

from cubic_reg.problems.base import Problem


class Quadratic(Problem):
    """f(x) = 0.5 x^T A x - b^T x, with controllable condition number."""

    def __init__(
        self,
        n: int = 50,
        condition: float = 100.0,
        seed: int = 0,
        name: Optional[str] = None,
    ):
        super().__init__(name or f"quadratic_n{n}_kappa{condition:g}")
        rng = np.random.default_rng(seed)
        eigvals = np.linspace(1.0, condition, n)
        Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
        self.A = (Q * eigvals) @ Q.T
        self.b = rng.standard_normal(n)
        self._x_star = np.linalg.solve(self.A, self.b)
        self._f_star = float(0.5 * self._x_star @ self.A @ self._x_star - self.b @ self._x_star)
        self._n = n
        self.condition = condition

    @property
    def dim(self) -> int:
        return self._n

    def f(self, x: np.ndarray) -> float:
        self.counters.f += 1
        return float(0.5 * x @ self.A @ x - self.b @ x)

    def grad(self, x: np.ndarray) -> np.ndarray:
        self.counters.grad += 1
        return self.A @ x - self.b

    def hess(self, x: np.ndarray) -> np.ndarray:
        self.counters.hess += 1
        return self.A.copy()

    def hvp(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        self.counters.hvp += 1
        return self.A @ v

    @property
    def f_star(self) -> float:
        return self._f_star

    @property
    def x_star(self) -> np.ndarray:
        return self._x_star.copy()


class LogSumExp(Problem):
    """f(x) = log(sum_i exp(a_i^T x + b_i)) + (mu/2)||x||^2."""

    def __init__(self, n: int = 20, m: int = 50, mu: float = 0.1, seed: int = 0):
        super().__init__(f"log_sum_exp_n{n}_m{m}")
        rng = np.random.default_rng(seed)
        self.A = rng.standard_normal((m, n))
        self.b = rng.standard_normal(m)
        self.mu = mu
        self._n = n

    @property
    def dim(self) -> int:
        return self._n

    def _softmax_weights(self, x: np.ndarray) -> np.ndarray:
        z = self.A @ x + self.b
        z = z - np.max(z)
        e = np.exp(z)
        return e / e.sum()

    def f(self, x: np.ndarray) -> float:
        self.counters.f += 1
        z = self.A @ x + self.b
        zmax = np.max(z)
        return float(zmax + np.log(np.sum(np.exp(z - zmax))) + 0.5 * self.mu * np.dot(x, x))

    def grad(self, x: np.ndarray) -> np.ndarray:
        self.counters.grad += 1
        w = self._softmax_weights(x)
        return self.A.T @ w + self.mu * x

    def hess(self, x: np.ndarray) -> np.ndarray:
        self.counters.hess += 1
        w = self._softmax_weights(x)
        AW = self.A * w[:, None]
        H = self.A.T @ AW - np.outer(self.A.T @ w, self.A.T @ w)
        H += self.mu * np.eye(self._n)
        return H

    def hvp(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        self.counters.hvp += 1
        w = self._softmax_weights(x)
        Av = self.A @ v
        return self.A.T @ (w * Av) - (self.A.T @ w) * (w @ Av) + self.mu * v


class Rosenbrock(Problem):
    """Classic Rosenbrock: f(x,y)=100(y-x^2)^2 + (1-x)^2, extended to n dims."""

    def __init__(self, n: int = 2):
        if n < 2:
            raise ValueError("Rosenbrock needs n >= 2")
        super().__init__(f"rosenbrock_n{n}")
        self._n = n

    @property
    def dim(self) -> int:
        return self._n

    def f(self, x: np.ndarray) -> float:
        self.counters.f += 1
        return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))

    def grad(self, x: np.ndarray) -> np.ndarray:
        self.counters.grad += 1
        n = self._n
        g = np.zeros(n)
        # d/dx_i terms from (1-x_i)^2 and 100(x_{i+1}-x_i^2)^2
        for i in range(n - 1):
            g[i] += -2.0 * (1.0 - x[i]) - 400.0 * x[i] * (x[i + 1] - x[i] ** 2)
            g[i + 1] += 200.0 * (x[i + 1] - x[i] ** 2)
        return g

    def hess(self, x: np.ndarray) -> np.ndarray:
        self.counters.hess += 1
        n = self._n
        H = np.zeros((n, n))
        for i in range(n - 1):
            # second derivatives
            H[i, i] += 2.0 - 400.0 * (x[i + 1] - x[i] ** 2) + 800.0 * x[i] ** 2
            H[i, i + 1] += -400.0 * x[i]
            H[i + 1, i] += -400.0 * x[i]
            H[i + 1, i + 1] += 200.0
        return H

    @property
    def f_star(self) -> float:
        return 0.0


class Rastrigin(Problem):
    """Non-convex multimodal: A*n + sum(x_i^2 - A cos(2 pi x_i))."""

    def __init__(self, n: int = 10, A: float = 10.0):
        super().__init__(f"rastrigin_n{n}")
        self._n = n
        self.A = A

    @property
    def dim(self) -> int:
        return self._n

    def f(self, x: np.ndarray) -> float:
        self.counters.f += 1
        return float(self.A * self._n + np.sum(x**2 - self.A * np.cos(2 * np.pi * x)))

    def grad(self, x: np.ndarray) -> np.ndarray:
        self.counters.grad += 1
        return 2.0 * x + 2.0 * np.pi * self.A * np.sin(2 * np.pi * x)

    def hess(self, x: np.ndarray) -> np.ndarray:
        self.counters.hess += 1
        diag = 2.0 + (2.0 * np.pi) ** 2 * self.A * np.cos(2 * np.pi * x)
        return np.diag(diag)

    def hvp(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        self.counters.hvp += 1
        diag = 2.0 + (2.0 * np.pi) ** 2 * self.A * np.cos(2 * np.pi * x)
        return diag * v

    @property
    def f_star(self) -> float:
        return 0.0
