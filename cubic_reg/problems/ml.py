"""Machine-learning style finite-sum / neural problems."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from cubic_reg.problems.base import Problem


class FiniteSumLogistic(Problem):
    """Regularized logistic regression on a finite dataset."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        l2: float = 1e-4,
        name: str = "logistic",
    ):
        super().__init__(name)
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.l2 = l2
        self.N, self._n = self.X.shape
        self._batch_idx: Optional[np.ndarray] = None

    @property
    def dim(self) -> int:
        return self._n

    def set_batch(self, idx: Optional[np.ndarray]) -> None:
        self._batch_idx = idx

    def _data(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._batch_idx is None:
            return self.X, self.y
        return self.X[self._batch_idx], self.y[self._batch_idx]

    def f(self, w: np.ndarray) -> float:
        self.counters.f += 1
        X, y = self._data()
        z = y * (X @ w)
        loss = np.mean(np.logaddexp(0.0, -z))
        return float(loss + 0.5 * self.l2 * np.dot(w, w))

    def grad(self, w: np.ndarray) -> np.ndarray:
        self.counters.grad += 1
        X, y = self._data()
        z = y * (X @ w)
        sig = 1.0 / (1.0 + np.exp(np.clip(z, -50, 50)))
        return -(X.T @ (y * sig)) / len(y) + self.l2 * w

    def hess(self, w: np.ndarray) -> np.ndarray:
        self.counters.hess += 1
        X, y = self._data()
        z = y * (X @ w)
        sig = 1.0 / (1.0 + np.exp(np.clip(z, -50, 50)))
        d = sig * (1.0 - sig)
        Xd = X * d[:, None]
        return (X.T @ Xd) / len(y) + self.l2 * np.eye(self._n)

    def hvp(self, w: np.ndarray, v: np.ndarray) -> np.ndarray:
        self.counters.hvp += 1
        X, y = self._data()
        z = y * (X @ w)
        sig = 1.0 / (1.0 + np.exp(np.clip(z, -50, 50)))
        d = sig * (1.0 - sig)
        return (X.T @ (d * (X @ v))) / len(y) + self.l2 * v


def make_synthetic_logistic(
    n_samples: int = 1000,
    n_features: int = 50,
    seed: int = 0,
    l2: float = 1e-4,
) -> FiniteSumLogistic:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features))
    w_true = rng.standard_normal(n_features)
    logits = X @ w_true
    y = np.sign(logits + 0.1 * rng.standard_normal(n_samples))
    y[y == 0] = 1.0
    return FiniteSumLogistic(X, y, l2=l2, name="synth_logistic")


class TinyMLP(Problem):
    """Small 1-hidden-layer MLP regression with analytic NumPy oracles.

    Architecture: R^{d_in} -> hidden (tanh) -> R^1, trained on squared loss + l2.
    Parameters flattened as [W1(h,d), b1(h), W2(1,h), b2(1)].
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        hidden: int = 8,
        l2: float = 1e-4,
        name: str = "tiny_mlp",
    ):
        super().__init__(name)
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y, dtype=float).reshape(-1)
        self.N, self.d_in = self.X.shape
        self.hidden = hidden
        self.l2 = l2
        self._n = hidden * self.d_in + hidden + hidden + 1

    @property
    def dim(self) -> int:
        return self._n

    def _unpack(self, theta: np.ndarray):
        h, d = self.hidden, self.d_in
        i0 = 0
        i1 = h * d
        i2 = i1 + h
        i3 = i2 + h
        W1 = theta[i0:i1].reshape(h, d)
        b1 = theta[i1:i2]
        W2 = theta[i2:i3].reshape(1, h)
        b2 = theta[i3:]
        return W1, b1, W2, b2

    def _forward(self, theta: np.ndarray):
        W1, b1, W2, b2 = self._unpack(theta)
        Z1 = self.X @ W1.T + b1  # (N,h)
        A1 = np.tanh(Z1)
        pred = (A1 @ W2.T).reshape(-1) + b2[0]
        return pred, A1, Z1, W1, b1, W2, b2

    def f(self, theta: np.ndarray) -> float:
        self.counters.f += 1
        pred, *_ = self._forward(theta)
        loss = 0.5 * np.mean((pred - self.y) ** 2)
        return float(loss + 0.5 * self.l2 * np.dot(theta, theta))

    def grad(self, theta: np.ndarray) -> np.ndarray:
        self.counters.grad += 1
        pred, A1, Z1, W1, b1, W2, b2 = self._forward(theta)
        N = self.N
        resid = (pred - self.y) / N  # (N,)
        # dL/dW2, db2
        gW2 = A1.T @ resid  # (h,)
        gb2 = np.array([np.sum(resid)])
        # backprop to hidden
        dA1 = np.outer(resid, W2.reshape(-1))  # (N,h)
        dZ1 = dA1 * (1.0 - np.tanh(Z1) ** 2)
        gW1 = dZ1.T @ self.X  # (h,d)
        gb1 = np.sum(dZ1, axis=0)
        g = np.concatenate([gW1.ravel(), gb1, gW2.ravel(), gb2])
        return g + self.l2 * theta

    def hess(self, theta: np.ndarray) -> np.ndarray:
        """Dense Hessian via finite-difference HVP columns (small networks only)."""
        self.counters.hess += 1
        n = self.dim
        H = np.zeros((n, n))
        e = np.zeros(n)
        for i in range(n):
            e[i] = 1.0
            H[:, i] = self.hvp(theta, e)
            e[i] = 0.0
        # hvp already counted n times; hess counter also incremented once above
        return 0.5 * (H + H.T)

    def hvp(self, theta: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Central finite-difference Hessian-vector product on grad."""
        self.counters.hvp += 1
        # Temporarily avoid double-counting grad in FD
        eps = 1e-5
        nrm = np.linalg.norm(v)
        if nrm < 1e-16:
            return np.zeros_like(v)
        h = eps / nrm
        # manual grad without incrementing? use internal
        g_plus = self._grad_no_count(theta + h * v)
        g_minus = self._grad_no_count(theta - h * v)
        return (g_plus - g_minus) / (2.0 * h) + self.l2 * 0.0  # l2 already in grad

    def _grad_no_count(self, theta: np.ndarray) -> np.ndarray:
        pred, A1, Z1, W1, b1, W2, b2 = self._forward(theta)
        N = self.N
        resid = (pred - self.y) / N
        gW2 = A1.T @ resid
        gb2 = np.array([np.sum(resid)])
        dA1 = np.outer(resid, W2.reshape(-1))
        dZ1 = dA1 * (1.0 - np.tanh(Z1) ** 2)
        gW1 = dZ1.T @ self.X
        gb1 = np.sum(dZ1, axis=0)
        g = np.concatenate([gW1.ravel(), gb1, gW2.ravel(), gb2])
        return g + self.l2 * theta


def make_synthetic_mlp(
    n_samples: int = 200,
    d_in: int = 6,
    hidden: int = 8,
    seed: int = 0,
    l2: float = 1e-4,
) -> TinyMLP:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, d_in))
    # nonlinear target
    y = np.sin(X[:, 0]) + 0.5 * X[:, 1] ** 2 + 0.1 * rng.standard_normal(n_samples)
    return TinyMLP(X, y, hidden=hidden, l2=l2, name="synth_mlp")


def make_mnist_like_logistic(
    n_samples: int = 4000,
    n_features: int = 784,
    n_classes: int = 10,
    binary_digits: tuple[int, int] = (3, 8),
    seed: int = 0,
    l2: float = 1e-4,
    noise: float = 0.35,
) -> FiniteSumLogistic:
    """Synthetic MNIST-scale binary logistic (no download).

    Builds class prototypes on a 28x28 grid (stroke-like blobs), samples around
    two chosen digit classes, and returns ±1 labels for logistic regression.
    """
    rng = np.random.default_rng(seed)
    side = int(np.sqrt(n_features))
    if side * side != n_features:
        raise ValueError("n_features must be a perfect square (e.g. 784)")

    yy, xx = np.mgrid[0:side, 0:side]
    prototypes = []
    for c in range(n_classes):
        # place a few Gaussian ink blobs depending on class id
        img = np.zeros((side, side))
        for b in range(3):
            cy = 6 + (c * 3 + b * 5) % (side - 8)
            cx = 6 + (c * 5 + b * 3) % (side - 8)
            amp = 0.8 + 0.2 * ((c + b) % 3)
            img += amp * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * (2.5 + (b % 3)) ** 2))
        prototypes.append(img.ravel())
    prototypes = np.asarray(prototypes)

    c0, c1 = binary_digits
    half = n_samples // 2
    X0 = prototypes[c0] + noise * rng.standard_normal((half, n_features))
    X1 = prototypes[c1] + noise * rng.standard_normal((n_samples - half, n_features))
    X = np.vstack([X0, X1])
    y = np.concatenate([np.full(half, -1.0), np.full(n_samples - half, 1.0)])
    # shuffle
    perm = rng.permutation(n_samples)
    X, y = X[perm], y[perm]
    # normalize features
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    return FiniteSumLogistic(X, y, l2=l2, name=f"mnist_like_{c0}v{c1}")


class SoftmaxRegression(Problem):
    """Multiclass softmax / multinomial logistic regression.

    Parameters W are flattened as (n_features * n_classes,).
    Labels y are integer class ids in {0,...,C-1}.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_classes: int,
        l2: float = 1e-4,
        name: str = "softmax",
    ):
        super().__init__(name)
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y, dtype=int)
        self.C = int(n_classes)
        self.l2 = l2
        self.N, self.d = self.X.shape
        self._n = self.d * self.C
        self._batch_idx: Optional[np.ndarray] = None

    @property
    def dim(self) -> int:
        return self._n

    def set_batch(self, idx: Optional[np.ndarray]) -> None:
        self._batch_idx = idx

    def _data(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._batch_idx is None:
            return self.X, self.y
        return self.X[self._batch_idx], self.y[self._batch_idx]

    def _W(self, theta: np.ndarray) -> np.ndarray:
        return theta.reshape(self.d, self.C)

    def _probs(self, X: np.ndarray, W: np.ndarray) -> np.ndarray:
        logits = X @ W
        logits = logits - logits.max(axis=1, keepdims=True)
        e = np.exp(logits)
        return e / e.sum(axis=1, keepdims=True)

    def f(self, theta: np.ndarray) -> float:
        self.counters.f += 1
        X, y = self._data()
        W = self._W(theta)
        logits = X @ W
        logits = logits - logits.max(axis=1, keepdims=True)
        log_sum = np.log(np.exp(logits).sum(axis=1))
        loss = -np.mean(logits[np.arange(len(y)), y] - log_sum)
        return float(loss + 0.5 * self.l2 * np.dot(theta, theta))

    def grad(self, theta: np.ndarray) -> np.ndarray:
        self.counters.grad += 1
        X, y = self._data()
        W = self._W(theta)
        P = self._probs(X, W)
        P[np.arange(len(y)), y] -= 1.0
        G = (X.T @ P) / len(y)
        return G.ravel() + self.l2 * theta

    def hess(self, theta: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Dense Hessian unavailable for softmax; use HVP/Krylov")

    def hvp(self, theta: np.ndarray, v: np.ndarray) -> np.ndarray:
        """HVP of softmax cross-entropy: X^T (diag(p)-pp^T) (X V) / N + l2 v."""
        self.counters.hvp += 1
        X, y = self._data()
        W = self._W(theta)
        V = v.reshape(self.d, self.C)
        P = self._probs(X, W)
        XV = X @ V  # (N, C)
        # (diag(p)-pp^T) u = p * u - p * (p·u)
        dot = np.sum(P * XV, axis=1, keepdims=True)
        S = P * XV - P * dot
        HV = (X.T @ S) / len(y)
        return HV.ravel() + self.l2 * v

    def accuracy(self, theta: np.ndarray, X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None) -> float:
        if X is None or y is None:
            X, y = self.X, self.y
        W = self._W(theta)
        pred = np.argmax(X @ W, axis=1)
        return float(np.mean(pred == y))
