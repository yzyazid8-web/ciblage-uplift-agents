"""Solveurs de RÉFÉRENCE (HiGHS, via scipy) : relaxation linéaire et programme en nombres entiers."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

from ciblage.optim.problem import Problem


@dataclass
class ExactResult:
    assignment: np.ndarray  # affectation entière (0 = rien, j = action j)
    objective: float  # valeur de l'affectation entière renvoyée
    bound: float  # borne supérieure prouvée de l'optimum entier
    status: str
    seconds: float
    n_fractional: int = 0  # (relaxation) clients non entiers


def _matrices(problem: Problem):
    rows, cols = np.nonzero(problem.candidates())
    m = len(rows)
    blocks = [sp.csr_matrix((np.ones(m), (rows, np.arange(m))), shape=(problem.n, m))]
    upper = [np.ones(problem.n)]
    if np.isfinite(problem.budget):
        blocks.append(sp.csr_matrix(problem.cost[cols][None, :]))
        upper.append(np.array([problem.budget]))
    assert problem.capacity is not None
    for j in range(problem.k):
        if np.isfinite(problem.capacity[j]):
            blocks.append(sp.csr_matrix((cols == j).astype(np.float64)[None, :]))
            upper.append(np.array([problem.capacity[j]]))
    return rows, cols, sp.vstack(blocks).tocsr(), np.concatenate(upper)


def _to_assignment(problem: Problem, rows, cols, x: np.ndarray) -> np.ndarray:
    a = np.zeros(problem.n, dtype=np.int64)
    take = x > 0.5
    a[rows[take]] = cols[take] + 1
    return a


def solve_lp(problem: Problem) -> ExactResult:
    """Relaxation linéaire. L'affectation renvoyée ne garde que les x = 1 (arrondi par défaut),
    donc toujours faisable ; sa valeur est très proche de la borne."""
    start = time.perf_counter()
    rows, cols, A, ub = _matrices(problem)
    if len(rows) == 0:
        return ExactResult(np.zeros(problem.n, dtype=np.int64), 0.0, 0.0, "vide", 0.0)
    res = linprog(-problem.score[rows, cols], A_ub=A, b_ub=ub, bounds=(0, 1), method="highs")
    if res.status != 0:
        raise RuntimeError(f"Relaxation linéaire non résolue : {res.message}")
    x = res.x
    integral = np.where(x > 1 - 1e-6, 1.0, 0.0)
    a = _to_assignment(problem, rows, cols, integral)
    return ExactResult(
        assignment=a,
        objective=float(problem.score[rows, cols] @ integral),
        bound=float(-res.fun),
        status="optimal (relaxation)",
        seconds=time.perf_counter() - start,
        n_fractional=int(((x > 1e-6) & (x < 1 - 1e-6)).sum()),
    )


def solve_milp(problem: Problem, time_limit: float = 60.0) -> ExactResult:
    """Optimum entier exact (ou meilleure solution trouvée dans le temps imparti)."""
    start = time.perf_counter()
    rows, cols, A, ub = _matrices(problem)
    if len(rows) == 0:
        return ExactResult(np.zeros(problem.n, dtype=np.int64), 0.0, 0.0, "vide", 0.0)
    res = milp(
        -problem.score[rows, cols],
        constraints=LinearConstraint(A, -np.inf, ub),
        integrality=np.ones(len(rows)),
        bounds=Bounds(0, 1),
        options={"time_limit": time_limit, "mip_rel_gap": 1e-9},
    )
    if res.x is None:
        raise RuntimeError(f"Aucune solution entière trouvée : {res.message}")
    a = _to_assignment(problem, rows, cols, res.x)
    bound = getattr(res, "mip_dual_bound", None)
    return ExactResult(
        assignment=a,
        objective=float(problem.score[rows, cols] @ (res.x > 0.5)),
        bound=float(-bound) if bound is not None else float("nan"),
        status="optimal" if res.status == 0 else res.message,
        seconds=time.perf_counter() - start,
    )
