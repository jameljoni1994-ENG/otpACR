"""Unified optimization problem interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from cubic_reg.metrics import CallCounters


class Problem(ABC):
    """Base class: f, grad, hess/hvp, optional f_star."""

    def __init__(self, name: str = "problem"):
        self.name = name
        self.counters = CallCounters()

    @property
    @abstractmethod
    def dim(self) -> int:
        ...

    @abstractmethod
    def f(self, x: np.ndarray) -> float:
        ...

    @abstractmethod
    def grad(self, x: np.ndarray) -> np.ndarray:
        ...

    def hess(self, x: np.ndarray) -> np.ndarray:
        """Dense Hessian. Override when available."""
        raise NotImplementedError(f"{self.name} does not implement hess()")

    def hvp(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Hessian-vector product. Default: H @ v via dense Hessian."""
        self.counters.hvp += 1
        return self.hess(x) @ v

    @property
    def f_star(self) -> Optional[float]:
        return None

    def reset_counters(self) -> None:
        self.counters.reset()
