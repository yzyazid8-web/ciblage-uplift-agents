"""Relaxation lagrangienne : une solution quasi optimale et une preuve de sa qualité."""

from __future__ import annotations

import time

import numpy as np

from ciblage.optim.greedy import HeuristicResult, fill
from ciblage.optim.problem import Problem, evaluate


def _reduced(problem: Problem, lam: float, mu: np.ndarray, mask: np.ndarray) -> np.ndarray:
    r = problem.score - lam * problem.cost[None, :] - mu[None, :]
    return np.where(mask, r, -np.inf)


def dual(problem: Problem, lam: float, mu: np.ndarray, mask: np.ndarray):
    """Valeur de L, choix de chaque client, dépense et effectifs induits par les prix donnés."""
    assert problem.capacity is not None
    r = _reduced(problem, lam, mu, mask)
    best = r.argmax(axis=1)
    best_val = r[np.arange(problem.n), best]
    chosen = best_val > 0
    value = float(best_val[chosen].sum())
    if np.isfinite(problem.budget):
        value += lam * problem.budget
    finite_cap = np.isfinite(problem.capacity)
    value += float((mu[finite_cap] * problem.capacity[finite_cap]).sum())
    spend = float(problem.cost[best[chosen]].sum())
    counts = np.bincount(best[chosen], minlength=problem.k).astype(np.float64)
    return value, spend, counts


def _bisect_budget(problem: Problem, mask: np.ndarray, iters: int = 60) -> tuple[float, float]:
    """Dichotomie sur lambda (budget seul) : renvoie (lambda*, L(lambda*))."""
    mu = np.zeros(problem.k)
    value0, spend0, _ = dual(problem, 0.0, mu, mask)
    if spend0 <= problem.budget:  # budget non contraignant : lambda = 0 est optimal
        return 0.0, value0
    pos = problem.cost > 0
    hi = (
        float((problem.score[:, pos] / problem.cost[pos]).max()) + 1.0
    )  # au-delà, rien n'est choisi
    lo = 0.0
    best_val, best_lam = np.inf, 0.0
    for _ in range(iters):
        mid = (lo + hi) / 2
        value, spend, _ = dual(problem, mid, mu, mask)
        if value < best_val:
            best_val, best_lam = value, mid
        if spend > problem.budget:
            lo = mid
        else:
            hi = mid
    for lam in (lo, hi):  # L est linéaire par morceaux : on évalue aussi les bornes finales
        value, _, _ = dual(problem, lam, mu, mask)
        if value < best_val:
            best_val, best_lam = value, lam
    return best_lam, best_val


def _subgradient(
    problem: Problem, mask: np.ndarray, lower: float, iters: int = 400
) -> tuple[float, np.ndarray, float]:
    """Sous-gradient projeté (pas de Polyak) sur (lambda, mu) ; renvoie les meilleurs prix et L."""
    assert problem.capacity is not None
    has_budget = np.isfinite(problem.budget)
    finite_cap = np.isfinite(problem.capacity)
    lam, mu = 0.0, np.zeros(problem.k)
    best = (np.inf, lam, mu.copy())
    theta = 1.0
    stall = 0
    for _ in range(iters):
        value, spend, counts = dual(problem, lam, mu, mask)
        if value < best[0] - 1e-9:
            best, stall = (value, lam, mu.copy()), 0
        else:
            stall += 1
            if stall >= 20:  # pas trop grands : on les réduit
                theta, stall = theta / 2, 0
        g_lam = (problem.budget - spend) if has_budget else 0.0
        g_mu = np.where(finite_cap, problem.capacity - counts, 0.0)
        norm = g_lam**2 + float((g_mu**2).sum())
        if norm < 1e-12 or theta < 1e-6:
            break
        step = theta * max(value - lower, 1e-9) / norm
        lam = max(0.0, lam - step * g_lam) if has_budget else 0.0
        mu = np.maximum(0.0, mu - step * g_mu)
    return best[1], best[2], best[0]


def solve_lagrange(problem: Problem) -> HeuristicResult:
    """Heuristique lagrangienne avec borne d'optimalité garantie."""
    start = time.perf_counter()
    assert problem.capacity is not None
    mask = problem.candidates()
    mu = np.zeros(problem.k)
    if not np.isfinite(problem.capacity).any():
        lam, bound = _bisect_budget(problem, mask)
    else:
        # une première solution faisable sert de borne inférieure au pas de Polyak
        rows, cols = np.nonzero(mask)
        first = fill(
            problem,
            rows[np.argsort(-problem.score[rows, cols])],
            cols[np.argsort(-problem.score[rows, cols])],
        )
        lower = evaluate(problem, first)["objective"]
        lam, mu, bound = _subgradient(problem, mask, lower)

    # solution primale : couples servis par score réduit décroissant, à prix optimaux
    rows, cols = np.nonzero(mask)
    reduced = problem.score[rows, cols] - lam * problem.cost[cols] - mu[cols]
    order = np.lexsort((-problem.score[rows, cols], -reduced))
    a = fill(problem, rows[order], cols[order])
    return HeuristicResult(
        a, evaluate(problem, a)["objective"], time.perf_counter() - start, bound=bound
    )
