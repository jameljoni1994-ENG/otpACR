"""Cubic subproblem solver via Lagrange multiplier root-finding.

Minimize  m(s) = g^T s + (1/2) s^T H s + (M/6) ||s||^3

Stationarity: (H + (M/2) ||s|| I) s = -g
Let lambda = (M/2) ||s|| >= 0. Then (H + lambda I) s = -g and
lambda = (M/2) ||s||.

We solve phi(lambda) = lambda - (M/2) ||(H+lambda I)^{-1} g|| = 0
for lambda >= max(0, -lambda_min(H)) with Newton on the scalar phi.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.linalg import LinAlgError


@dataclass
class SubproblemSolution:
    s: np.ndarray
    model_decrease: float
    lam: float
    success: bool
    message: str = ""


def model_value(g: np.ndarray, H: np.ndarray, s: np.ndarray, M: float) -> float:
    """Evaluate cubic model m(s) - f (i.e. without the constant f)."""
    sn = np.linalg.norm(s)
    return float(g @ s + 0.5 * s @ H @ s + (M / 6.0) * sn**3)


def model_decrease(g: np.ndarray, H: np.ndarray, s: np.ndarray, M: float) -> float:
    """Predicted decrease f - m(s) = -model_value."""
    return -model_value(g, H, s, M)


def solve_cubic_subproblem(
    g: np.ndarray,
    H: np.ndarray,
    M: float,
    max_newton: int = 50,
    tol: float = 1e-12,
) -> SubproblemSolution:
    """Solve the cubic regularized Newton step for dense H."""
    n = g.shape[0]
    g = np.asarray(g, dtype=float)
    H = np.asarray(H, dtype=float)
    if M <= 0:
        raise ValueError("M must be positive")

    # Eigen-decomposition for robust scalar Newton (works for indefinite H)
    try:
        eigvals, eigvecs = np.linalg.eigh(H)
    except LinAlgError as exc:
        return SubproblemSolution(
            s=np.zeros(n),
            model_decrease=0.0,
            lam=0.0,
            success=False,
            message=f"eigh failed: {exc}",
        )

    # Coefficients of g in eigenbasis
    g_hat = eigvecs.T @ g
    lam_min = float(eigvals[0])
    lam_floor = max(0.0, -lam_min) + 1e-12

    def s_norm(lam: float) -> float:
        denom = eigvals + lam
        # guard tiny denominators
        denom = np.where(np.abs(denom) < 1e-16, np.sign(denom) * 1e-16 + 1e-16, denom)
        return float(np.linalg.norm(g_hat / denom))

    def phi(lam: float) -> float:
        return lam - (M / 2.0) * s_norm(lam)

    def phi_prime(lam: float) -> float:
        denom = eigvals + lam
        denom = np.where(np.abs(denom) < 1e-16, np.sign(denom) * 1e-16 + 1e-16, denom)
        sn = s_norm(lam)
        if sn < 1e-16:
            return 1.0
        # d/dλ ||(H+λI)^{-1}g|| = -||u||^{-1} u^T (H+λI)^{-2} g  where u=(H+λI)^{-1}g
        # in eigenbasis: sum (g_i^2 / d_i^3) / sn
        deriv_sn = -float(np.sum(g_hat**2 / denom**3)) / sn
        return 1.0 - (M / 2.0) * deriv_sn

    # Hard case: g nearly orthogonal to leftmost eigenvector and lam_min < 0
    if abs(g_hat[0]) < 1e-10 * (np.linalg.norm(g_hat) + 1e-16) and lam_min < 0:
        lam = -lam_min
        # Particular solution on range, plus null-space component for ||s||
        denom = eigvals + lam
        s_hat = np.zeros(n)
        for i in range(n):
            if abs(denom[i]) > 1e-10:
                s_hat[i] = -g_hat[i] / denom[i]
        target = (2.0 / M) * lam  # ||s|| = 2λ/M
        curr = float(np.linalg.norm(s_hat))
        if curr < target:
            # add component along first eigenvector
            s_hat[0] = np.sqrt(max(target**2 - curr**2, 0.0))
        s = eigvecs @ s_hat
        md = model_decrease(g, H, s, M)
        return SubproblemSolution(s=s, model_decrease=md, lam=lam, success=True, message="hard case")

    # Bracket / initialize lambda
    lam = lam_floor
    # If phi(lam_floor) > 0 already, solution is at boundary (rare for M>0)
    # Grow until phi changes sign or is near zero
    if phi(lam) > 0:
        # lambda too large initially — try smaller isn't possible; take Newton from floor
        pass
    else:
        # phi(lam_floor) <= 0: increase until phi > 0
        upper = max(lam_floor + 1.0, (M / 2.0) * np.linalg.norm(g) + 1.0)
        for _ in range(60):
            if phi(upper) >= 0:
                break
            upper *= 2.0
        lam = upper

    for _ in range(max_newton):
        sn = s_norm(lam)
        p = lam - (M / 2.0) * sn
        if abs(p) < tol * (1.0 + abs(lam)):
            break
        pp = phi_prime(lam)
        if abs(pp) < 1e-16:
            break
        lam_new = lam - p / pp
        if lam_new < lam_floor:
            lam_new = 0.5 * (lam + lam_floor)
        if not np.isfinite(lam_new):
            break
        lam = lam_new

    denom = eigvals + lam
    denom = np.where(np.abs(denom) < 1e-16, np.sign(denom) * 1e-16 + 1e-16, denom)
    s_hat = -g_hat / denom
    s = eigvecs @ s_hat
    md = model_decrease(g, H, s, M)
    return SubproblemSolution(
        s=s,
        model_decrease=md,
        lam=float(lam),
        success=True,
        message="ok",
    )


def solve_cubic_subproblem_reduced(
    g_red: np.ndarray,
    T: np.ndarray,
    M: float,
    Q: Optional[np.ndarray] = None,
) -> Tuple[SubproblemSolution, np.ndarray]:
    """Solve cubic subproblem in a reduced Krylov basis.

    Parameters
    ----------
    g_red : coefficients of g in the orthonormal basis Q (usually e1 * ||g||)
    T : tridiagonal (or dense) projected Hessian Q^T H Q
    M : cubic regularization parameter
    Q : optional basis; if given, full-space s = Q @ s_red is returned in solution

    Returns
    -------
    solution (with s in reduced space if Q is None, else full space), s_red
    """
    sol = solve_cubic_subproblem(g_red, T, M)
    s_red = sol.s
    if Q is not None:
        sol = SubproblemSolution(
            s=Q @ s_red,
            model_decrease=sol.model_decrease,
            lam=sol.lam,
            success=sol.success,
            message=sol.message,
        )
    return sol, s_red
