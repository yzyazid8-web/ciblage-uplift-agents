"""Formulation du problème d'allocation : qui reçoit quelle action, sous budget et capacités."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

Objective = Literal["outcome", "profit"]


@dataclass
class Problem:
    score: np.ndarray  # (n, K)
    cost: np.ndarray  # (K,)
    budget: float = np.inf
    capacity: np.ndarray | None = None  # (K,), np.inf = pas de limite
    allowed: np.ndarray | None = None  # (n, K) booléens
    action_names: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.score = np.asarray(self.score, dtype=np.float64)
        if self.score.ndim != 2:
            raise ValueError("score doit être une matrice n × K")
        n, k = self.score.shape
        self.cost = np.asarray(self.cost, dtype=np.float64).reshape(-1)
        if self.cost.shape != (k,):
            raise ValueError(f"cost doit contenir {k} coûts (un par action)")
        if (self.cost < 0).any():
            raise ValueError("Les coûts doivent être positifs ou nuls")
        if self.budget < 0:
            raise ValueError("Le budget doit être positif ou nul")
        cap = (
            np.full(k, np.inf)
            if self.capacity is None
            else np.asarray(self.capacity, dtype=np.float64)
        )
        if cap.shape != (k,) or (cap < 0).any():
            raise ValueError(f"capacity doit contenir {k} valeurs positives")
        self.capacity = cap
        allowed = (
            np.ones((n, k), dtype=bool)
            if self.allowed is None
            else np.asarray(self.allowed, dtype=bool)
        )
        if allowed.shape != (n, k):
            raise ValueError("allowed doit avoir la même forme que score")
        self.allowed = allowed
        if not self.action_names:
            self.action_names = [f"action_{j + 1}" for j in range(k)]

    @property
    def n(self) -> int:
        return self.score.shape[0]

    @property
    def k(self) -> int:
        return self.score.shape[1]

    def candidates(self) -> np.ndarray:
        """Couples (client, action) qui valent la peine d'être envisagés : autorisés et à score > 0."""
        assert self.allowed is not None
        return self.allowed & (self.score > 0)


def from_uplift(
    uplift: np.ndarray,
    cost: np.ndarray | list[float],
    objective: Objective = "outcome",
    value_per_unit: float = 1.0,
    budget: float = np.inf,
    capacity: np.ndarray | list[float] | None = None,
    allowed: np.ndarray | None = None,
    action_names: list[str] | None = None,
) -> Problem:
    """Construit le problème à partir de l'uplift prédit (n x K)."""
    u = np.asarray(uplift, dtype=np.float64)
    if u.ndim == 1:
        u = u[:, None]
    c = np.asarray(cost, dtype=np.float64)
    if objective == "outcome":
        score = u.copy()
    elif objective == "profit":
        score = value_per_unit * u - c[None, :]
    else:
        raise ValueError("objective doit valoir 'outcome' ou 'profit'")
    return Problem(
        score=score,
        cost=c,
        budget=budget,
        capacity=None if capacity is None else np.asarray(capacity, dtype=np.float64),
        allowed=allowed,
        action_names=action_names or [],
    )


def evaluate(problem: Problem, assignment: np.ndarray) -> dict[str, float]:
    """Objectif, dépense et nombre de clients par action pour une affectation."""
    a = np.asarray(assignment)
    chosen = a > 0
    rows, cols = np.flatnonzero(chosen), a[chosen] - 1
    out = {
        "objective": float(problem.score[rows, cols].sum()),
        "spend": float(problem.cost[cols].sum()),
        "n_contacted": int(chosen.sum()),
    }
    for j, name in enumerate(problem.action_names):
        out[f"n_{name}"] = int((a == j + 1).sum())
    return out


def violations(problem: Problem, assignment: np.ndarray, tol: float = 1e-6) -> list[str]:
    """Liste lisible des contraintes violées (vide = affectation faisable)."""
    a = np.asarray(assignment)
    issues = []
    if a.shape != (problem.n,):
        return [f"L'affectation doit contenir {problem.n} entrées"]
    if ((a < 0) | (a > problem.k)).any():
        return [f"Codes d'action hors de 0..{problem.k}"]
    chosen = a > 0
    rows, cols = np.flatnonzero(chosen), a[chosen] - 1
    assert problem.allowed is not None and problem.capacity is not None
    forbidden = int((~problem.allowed[rows, cols]).sum())
    if forbidden:
        issues.append(f"{forbidden} client(s) reçoivent une action non autorisée")
    spend = float(problem.cost[cols].sum())
    if spend > problem.budget + tol:
        issues.append(f"Budget dépassé : {spend:.2f} > {problem.budget:.2f}")
    for j, name in enumerate(problem.action_names):
        count = int((a == j + 1).sum())
        if count > problem.capacity[j] + tol:
            issues.append(f"Capacité dépassée pour {name} : {count} > {problem.capacity[j]:.0f}")
    return issues
