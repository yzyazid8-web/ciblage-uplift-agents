"""Effet moyen hors essai randomisé : estimateurs naïf, IPW et AIPW (doublement robuste)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from ciblage import config
from ciblage.causal.propensity import clip_propensity

# Modèles de réponse volontairement légers : l'AIPW tourne 5 folds x 2 bras.
OUTCOME_PARAMS: dict[str, Any] = {
    "n_estimators": 200,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "tree_method": "hist",
    "n_jobs": -1,
}


@dataclass
class Estimate:
    method: str
    ate: float
    se: float
    ci_low: float
    ci_high: float

    @property
    def significant(self) -> bool:
        return self.ci_low > 0 or self.ci_high < 0


def _ci(ate: float, se: float, level: float = 0.95) -> tuple[float, float]:
    q = float(stats.norm.ppf(0.5 + level / 2))
    return ate - q * se, ate + q * se


def _as_binary(y: np.ndarray) -> np.ndarray:
    ya = np.asarray(y, dtype=np.float64)
    if not np.isin(ya, [0, 1]).all():
        raise ValueError("Ce module suppose un résultat binaire (0/1)")
    return ya


def naive_ate(y: np.ndarray, w: np.ndarray) -> Estimate:
    """Différence de moyennes traités - témoins (ignore tout biais d'affectation)."""
    ya = np.asarray(y, dtype=np.float64)
    wa = np.asarray(w).astype(int)
    yt, yc = ya[wa == 1], ya[wa == 0]
    ate = float(yt.mean() - yc.mean())
    se = float(np.sqrt(yt.var(ddof=1) / len(yt) + yc.var(ddof=1) / len(yc)))
    lo, hi = _ci(ate, se)
    return Estimate("naïf", ate, se, lo, hi)


def ipw_ate(y: np.ndarray, w: np.ndarray, e: np.ndarray) -> Estimate:
    """Effet par pondération inverse de la propension (Hajek, poids normalisés)."""
    ya = np.asarray(y, dtype=np.float64)
    wa = np.asarray(w).astype(int)
    ea = clip_propensity(e)
    wt = wa / ea
    wc = (1 - wa) / (1 - ea)
    sum_t, sum_c = float(wt.sum()), float(wc.sum())
    mu1 = float(np.sum(wt * ya) / sum_t)
    mu0 = float(np.sum(wc * ya) / sum_c)
    n = len(ya)
    # Fonction d'influence approchée de l'estimateur de Hajek (poids moyens normalisés à 1).
    inf = wt * (ya - mu1) / (sum_t / n) - wc * (ya - mu0) / (sum_c / n)
    ate = mu1 - mu0
    se = float(np.sqrt(np.var(inf, ddof=1) / n))
    lo, hi = _ci(ate, se)
    return Estimate("IPW", ate, se, lo, hi)


def _outcome_cv(
    Xa: np.ndarray, wa: np.ndarray, ya: np.ndarray, n_folds: int, seed: int, params: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    """mu1(x) = E[Y|x, traité] et mu0(x) = E[Y|x, témoin], prédits hors-échantillon."""
    mu1, mu0 = np.zeros(len(ya)), np.zeros(len(ya))
    strata = wa * 2 + ya.astype(int)  # stratifie par bras ET résultat (bras rares conservés)
    folds = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
    for train, test in folds.split(Xa, strata):
        for code, mu in ((1, mu1), (0, mu0)):
            rows = train[wa[train] == code]
            model = XGBClassifier(**params, random_state=seed, eval_metric="logloss")
            model.fit(Xa[rows], ya[rows])
            mu[test] = model.predict_proba(Xa[test])[:, 1]
    return mu1, mu0


def aipw_ate(
    X: pd.DataFrame | np.ndarray,
    w: np.ndarray,
    y: np.ndarray,
    e: np.ndarray,
    n_folds: int = 5,
    seed: int = config.SEED,
    params: dict[str, Any] | None = None,
) -> Estimate:
    """Effet doublement robuste (AIPW) en cross-fitting ; IC par la fonction d'influence.

    psi_i = mu1 - mu0 + W(Y - mu1)/e - (1-W)(Y - mu0)/(1-e) ; l'ATE est la moyenne des psi,
    et son erreur-type l'écart-type empirique des psi divisé par racine de n.
    """
    Xa = np.asarray(X, dtype=np.float64)
    wa = np.asarray(w).astype(int)
    ya = _as_binary(y)
    ea = clip_propensity(e)
    mu1, mu0 = _outcome_cv(Xa, wa, ya, n_folds, seed, {**OUTCOME_PARAMS, **(params or {})})
    psi = mu1 - mu0 + wa * (ya - mu1) / ea - (1 - wa) * (ya - mu0) / (1 - ea)
    ate = float(psi.mean())
    se = float(np.sqrt(np.var(psi, ddof=1) / len(psi)))
    lo, hi = _ci(ate, se)
    return Estimate("AIPW", ate, se, lo, hi)
