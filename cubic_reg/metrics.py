"""Shared metrics and result containers for all solvers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class CallCounters:
    """Track oracle calls during optimization."""

    f: int = 0
    grad: int = 0
    hess: int = 0
    hvp: int = 0

    def reset(self) -> None:
        self.f = self.grad = self.hess = self.hvp = 0


@dataclass
class OptimizationResult:
    """Unified solver return type."""

    x: np.ndarray
    f: float
    grad_norm: float
    nit: int
    success: bool
    message: str = ""
    time_sec: float = 0.0
    n_f: int = 0
    n_grad: int = 0
    n_hess: int = 0
    n_hvp: int = 0
    rejected_steps: int = 0
    history_f: List[float] = field(default_factory=list)
    history_grad_norm: List[float] = field(default_factory=list)
    history_M: List[float] = field(default_factory=list)
    # Inexact Krylov / ARC–Krylov diagnostics
    history_resid: List[float] = field(default_factory=list)
    history_m: List[int] = field(default_factory=list)
    history_theta: List[float] = field(default_factory=list)
    f_star: Optional[float] = None

    @property
    def optimality_gap(self) -> Optional[np.ndarray]:
        if self.f_star is None:
            return None
        return np.asarray(self.history_f) - self.f_star
