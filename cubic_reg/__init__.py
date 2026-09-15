"""Cubic regularization research package (Nesterov CR and extensions)."""

from cubic_reg.metrics import OptimizationResult, CallCounters
from cubic_reg.problems.base import Problem

__all__ = ["Problem", "OptimizationResult", "CallCounters"]
__version__ = "0.2.0"
