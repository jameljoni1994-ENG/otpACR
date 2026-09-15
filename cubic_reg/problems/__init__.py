"""Problem package exports."""

from cubic_reg.problems.base import Problem
from cubic_reg.problems.analytic import Quadratic, LogSumExp, Rosenbrock, Rastrigin
from cubic_reg.problems.ml import (
    FiniteSumLogistic,
    TinyMLP,
    SoftmaxRegression,
    make_synthetic_logistic,
    make_synthetic_mlp,
    make_mnist_like_logistic,
)
from cubic_reg.problems.matrix_free import MatrixFreeQuadratic

__all__ = [
    "Problem",
    "Quadratic",
    "LogSumExp",
    "Rosenbrock",
    "Rastrigin",
    "FiniteSumLogistic",
    "TinyMLP",
    "SoftmaxRegression",
    "make_synthetic_logistic",
    "make_synthetic_mlp",
    "make_mnist_like_logistic",
    "MatrixFreeQuadratic",
]
