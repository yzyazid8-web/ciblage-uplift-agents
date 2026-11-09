"""Réglage des learners d'uplift, avec un protocole qui empêche de « tricher »."""

from __future__ import annotations

import itertools
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ciblage import config
from ciblage.uplift import metrics
from ciblage.uplift.learners import BaseUplift

# Modèles de RÉPONSE (S- et T-learner, étape 1 du X- et du DR-learner)
RESPONSE_GRID: dict[str, list[Any]] = {
    "max_depth": [2, 3, 4],
    "n_estimators": [100, 300],
    "min_child_weight": [5, 100],
}
# Modèles d'EFFET (X- et DR-learner) : cibles bruitées, on explore des modèles plus simples
EFFECT_GRID: dict[str, list[Any]] = {
    "max_depth": [1, 2, 3],
    "n_estimators": [50, 150],
    "min_child_weight": [50, 300],
}


def holdout_split(
    t: np.ndarray, y: np.ndarray, test_size: float = 0.3, seed: int = config.SEED
) -> tuple[np.ndarray, np.ndarray]:
    """Indices (entraînement, test), stratifiés sur traitement × résultat."""
    t, y = np.asarray(t), np.asarray(y)
    strata = t.astype(np.int64) * 2 + (y > np.median(y)).astype(np.int64)
    idx = np.arange(len(t))
    train, test = train_test_split(idx, test_size=test_size, stratify=strata, random_state=seed)
    return np.sort(train), np.sort(test)


def grid(space: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Toutes les combinaisons d'une grille {paramètre: [valeurs]}."""
    keys = list(space)
    return [dict(zip(keys, values, strict=True)) for values in itertools.product(*space.values())]


def qini_score(y, t, uplift: np.ndarray) -> float:
    """Critère de sélection : Qini moyen sur toutes les actions (fixé avant le réglage)."""
    return float(np.mean(list(metrics.per_treatment(metrics.qini, y, uplift, t).values())))


@dataclass
class TuningResult:
    learner: str
    table: pd.DataFrame  # une ligne par configuration essayée
    best_params: dict[str, Any] = field(default_factory=dict)
    best_effect_params: dict[str, Any] = field(default_factory=dict)
    best_score: float = float("nan")
    default_score: float = float("nan")
    seconds: float = 0.0


def tune(
    cls: type[BaseUplift],
    X,
    t,
    y,
    cross_val: Callable[..., np.ndarray],
    propensity: dict[int, float] | None = None,
    response_grid: dict[str, list[Any]] = RESPONSE_GRID,
    effect_grid: dict[str, list[Any]] = EFFECT_GRID,
    n_folds: int = 5,
    seed: int = config.SEED,
) -> TuningResult:
    """Réglage en deux temps : d'abord les modèles de réponse, puis (si le learner en a)
    les modèles d'effet, à réponse fixée. Évite d'explorer le produit des deux grilles.

    `cross_val(make, X, t, y, n_folds, seed)` renvoie l'uplift hors échantillon.
    """
    start = time.perf_counter()
    has_effect = cls.__name__ in {"XLearner", "DRLearner"}
    rows = []

    def evaluate(params: dict[str, Any], effect: dict[str, Any], stage: str) -> float:
        def make() -> BaseUplift:
            return cls(params=params, effect_params=effect, propensity=propensity, seed=seed)

        score = qini_score(y, t, cross_val(make, X, t, y, n_folds=n_folds, seed=seed))
        rows.append(
            {
                "etape": stage,
                **{f"rep_{k}": v for k, v in params.items()},
                **{f"eff_{k}": v for k, v in effect.items()},
                "qini_cv": score,
            }
        )
        return score

    default = evaluate({}, {}, "défaut")
    best_params, best_score = {}, default
    for params in grid(response_grid):
        score = evaluate(params, {}, "réponse")
        if score > best_score:
            best_params, best_score = params, score

    best_effect: dict[str, Any] = {}
    if has_effect:
        for effect in grid(effect_grid):
            score = evaluate(best_params, effect, "effet")
            if score > best_score:
                best_effect, best_score = effect, score

    return TuningResult(
        learner=cls.name,
        table=pd.DataFrame(rows),
        best_params=best_params,
        best_effect_params=best_effect,
        best_score=best_score,
        default_score=default,
        seconds=time.perf_counter() - start,
    )
