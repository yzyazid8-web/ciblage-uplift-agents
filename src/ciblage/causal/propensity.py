"""Score de propension et diagnostics de recouvrement (cas observationnel)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ciblage import config


def _model() -> Pipeline:
    # Régression logistique régularisée sur variables standardisées : estimateur de propension
    # stable, peu enclin à prédire 0 ou 1 (ce qui ferait exploser les poids 1/e).
    return Pipeline(
        [("scale", StandardScaler()), ("logit", LogisticRegression(max_iter=1000, C=1.0))]
    )


def propensity_cv(
    X: pd.DataFrame | np.ndarray,
    w: np.ndarray,
    n_folds: int = 5,
    seed: int = config.SEED,
) -> np.ndarray:
    """Propension e(x) estimée hors-échantillon par validation croisée (cross-fitting).

    Chaque client reçoit une propension prédite par un modèle qui ne l'a pas vu : on évite
    ainsi le biais d'ajustement quand on l'injecte ensuite dans un estimateur AIPW.
    """
    Xa = np.asarray(X, dtype=np.float64)
    wa = np.asarray(w).astype(int)
    if not np.isin(wa, [0, 1]).all():
        raise ValueError("w doit être binaire (0 = témoin, 1 = traité)")
    e = np.zeros(len(wa))
    folds = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
    for train, test in folds.split(Xa, wa):
        model = _model().fit(Xa[train], wa[train])
        e[test] = model.predict_proba(Xa[test])[:, 1]
    return e


def clip_propensity(e: np.ndarray, epsilon: float = 0.01) -> np.ndarray:
    """Tronque la propension à [epsilon, 1 - epsilon] pour borner les poids inverses 1/e."""
    return np.clip(np.asarray(e, dtype=np.float64), epsilon, 1.0 - epsilon)


@dataclass
class Overlap:
    e_min: float
    e_max: float
    share_extreme: float  # part des clients avec e < seuil ou e > 1 - seuil
    ess_treated: float  # taille d'échantillon efficace des poids 1/e chez les traités
    ess_control: float  # idem avec 1/(1-e) chez les témoins
    n_treated: int
    n_control: int

    @property
    def ess_ratio_treated(self) -> float:
        return self.ess_treated / self.n_treated if self.n_treated else float("nan")

    @property
    def ess_ratio_control(self) -> float:
        return self.ess_control / self.n_control if self.n_control else float("nan")

    @property
    def is_good(self) -> bool:
        # Recouvrement acceptable : pas de propension extrême ET peu de perte d'effectif efficace.
        worst_ratio = min(self.ess_ratio_treated, self.ess_ratio_control)
        return self.share_extreme < 0.05 and worst_ratio > 0.5


def _ess(weights: np.ndarray) -> float:
    """Taille d'échantillon efficace de Kish : (somme des poids)^2 / somme des poids^2."""
    s = float(weights.sum())
    return s * s / float(np.square(weights).sum()) if len(weights) else 0.0


def overlap(e: np.ndarray, w: np.ndarray, extreme: float = 0.05) -> Overlap:
    """Diagnostics de recouvrement (positivité) à partir des propensions estimées."""
    ea = clip_propensity(e)
    wa = np.asarray(w).astype(int)
    treated, control = wa == 1, wa == 0
    return Overlap(
        e_min=float(ea.min()),
        e_max=float(ea.max()),
        share_extreme=float(np.mean((ea < extreme) | (ea > 1 - extreme))),
        ess_treated=_ess(1.0 / ea[treated]),
        ess_control=_ess(1.0 / (1.0 - ea[control])),
        n_treated=int(treated.sum()),
        n_control=int(control.sum()),
    )


def _weighted_mean(x: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(weights * x) / np.sum(weights))


def smd_max(X: pd.DataFrame, w: np.ndarray, e: np.ndarray | None = None) -> float:
    """|SMD| maximale entre traités et témoins, brute (e=None) ou pondérée par 1/propension.

    Sert à montrer que la repondération par la propension RÉÉQUILIBRE les groupes : la |SMD|
    maximale, élevée sur l'échantillon biaisé, redescend sous le seuil après pondération.
    """
    wa = np.asarray(w).astype(int)
    treated, control = wa == 1, wa == 0
    if e is None:
        wt = np.ones(int(treated.sum()))
        wc = np.ones(int(control.sum()))
    else:
        ea = clip_propensity(e)
        wt = 1.0 / ea[treated]
        wc = 1.0 / (1.0 - ea[control])
    worst = 0.0
    for col in X.columns:
        x = X[col].to_numpy(dtype=np.float64)
        xt, xc = x[treated], x[control]
        pooled = np.sqrt((xt.var(ddof=1) + xc.var(ddof=1)) / 2)
        if pooled == 0:
            continue
        diff = _weighted_mean(xt, wt) - _weighted_mean(xc, wc)
        worst = max(worst, abs(diff / pooled))
    return float(worst)
