"""Optimisation prudente : tenir compte de l'incertitude de l'uplift prédit."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.model_selection import KFold

from ciblage import config
from ciblage.uplift.learners import BaseUplift


def uplift_with_uncertainty(
    cls: type[BaseUplift],
    params: dict[str, Any],
    X_train: np.ndarray,
    t_train: np.ndarray,
    y_train: np.ndarray,
    X_pred: np.ndarray,
    propensity: dict[int, float] | None = None,
    n_models: int = 5,
    seed: int = config.SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Moyenne et écart-type de l'uplift prédit par `n_models` modèles (chacun sur (n-1)/n des données)."""
    if n_models < 2:
        raise ValueError("Il faut au moins 2 modèles pour estimer une dispersion")
    preds = []
    for k, (fit_idx, _) in enumerate(
        KFold(n_models, shuffle=True, random_state=seed).split(X_train)
    ):
        model = cls(params=params, propensity=propensity, seed=seed + k)
        preds.append(
            model.fit(X_train[fit_idx], t_train[fit_idx], y_train[fit_idx]).predict(X_pred)
        )
    stack = np.stack(preds)  # (n_models, n, K)
    return stack.mean(axis=0), stack.std(axis=0, ddof=1)


def prudent_score(mean: np.ndarray, std: np.ndarray, z: float = 1.0) -> np.ndarray:
    """Valeur prudente de l'uplift : moyenne - z × écart-type (z = 0 : pas de prudence)."""
    if z < 0:
        raise ValueError("z doit être positif ou nul")
    return np.asarray(mean) - z * np.asarray(std)
