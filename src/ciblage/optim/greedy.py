"""Heuristique gloutonne : on sert d'abord les couples (client, action) au meilleur rendement."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ciblage.optim.problem import Problem, evaluate


@dataclass
class HeuristicResult:
    assignment: np.ndarray
    objective: float
    seconds: float
    bound: float = float("nan")  # borne supérieure prouvée (si la méthode en fournit une)
    budget_price: float = float(
        "nan"
    )  # prix marginal du budget (lagrangien) : objectif gagné par euro en plus

    @property
    def gap(self) -> float:
        """Écart relatif maximal à l'optimum, garanti par la borne (0 = optimal prouvé)."""
        if not np.isfinite(self.bound) or self.bound <= 0:
            return float("nan")
        return max(0.0, (self.bound - self.objective) / self.bound)


def fill(problem: Problem, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Affecte les couples (rows[k], cols[k]) dans l'ordre donné, si c'est encore possible."""
    assert problem.capacity is not None
    a = np.zeros(problem.n, dtype=np.int64)
    budget = float(problem.budget)
    capacity = problem.capacity.astype(np.float64).copy()
    cost = problem.cost
    cheapest = float(cost.min()) if len(cost) else 0.0
    if budget < cheapest:  # rien n'est abordable : inutile de parcourir les candidats
        return a
    for i, j in zip(rows.tolist(), cols.tolist(), strict=True):
        if a[i] or capacity[j] < 1 or cost[j] > budget:
            continue
        a[i] = j + 1
        budget -= cost[j]
        capacity[j] -= 1
        if budget < cheapest:
            break
    return a


def solve_greedy(problem: Problem) -> HeuristicResult:
    """Tri par rendement score / cout (score seul pour les actions gratuites, servies en premier)."""
    start = time.perf_counter()
    rows, cols = np.nonzero(problem.candidates())
    score = problem.score[rows, cols]
    cost = problem.cost[cols]
    ratio = np.where(cost > 0, score / np.where(cost > 0, cost, 1.0), np.inf)
    order = np.lexsort((-score, -ratio))  # rendement décroissant, puis score décroissant
    a = fill(problem, rows[order], cols[order])
    return HeuristicResult(a, evaluate(problem, a)["objective"], time.perf_counter() - start)
