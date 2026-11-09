"""Métriques d'uplift : un modèle classe-t-il bien les clients selon l'effet de l'action ?"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def _check(y: np.ndarray, treatment: np.ndarray, score: np.ndarray | None = None) -> None:
    if score is not None and not (len(y) == len(treatment) == len(score)):
        raise ValueError("y, treatment et score doivent avoir la même longueur")
    if not np.isin(treatment, [0, 1]).all():
        raise ValueError("treatment doit être binaire (0 = témoin, 1 = traité)")
    if not ((treatment == 0).any() and (treatment == 1).any()):
        raise ValueError("Il faut des traités ET des témoins")


def _cumulative(y: np.ndarray, score: np.ndarray, treatment: np.ndarray):
    """Cumuls après tri par score décroissant, à chaque changement de valeur du score."""
    order = np.argsort(score, kind="mergesort")[::-1]
    y, score, t = y[order], score[order], treatment[order]
    cut = np.r_[np.flatnonzero(np.diff(score)), len(score) - 1]
    n_treated = np.cumsum(t)[cut].astype(np.float64)
    y_treated = np.cumsum(y * t)[cut].astype(np.float64)
    y_control = np.cumsum(y * (1 - t))[cut].astype(np.float64)
    n_all = (cut + 1).astype(np.float64)
    return n_all, n_treated, n_all - n_treated, y_treated, y_control


def _start_at_origin(x: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if x.size == 0 or x[0] != 0 or v[0] != 0:
        return np.r_[0.0, x], np.r_[0.0, v]
    return x, v


def _ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.divide(a, b, out=np.zeros_like(a), where=b != 0)


def uplift_curve(y, score, treatment) -> tuple[np.ndarray, np.ndarray]:
    """Points (k, valeur) de la courbe d'uplift."""
    y, score, treatment = np.asarray(y), np.asarray(score), np.asarray(treatment)
    _check(y, treatment, score)
    n, nt, nc, yt, yc = _cumulative(y, score, treatment)
    return _start_at_origin(n, (_ratio(yt, nt) - _ratio(yc, nc)) * n)


def qini_curve(y, score, treatment) -> tuple[np.ndarray, np.ndarray]:
    """Points (k, valeur) de la courbe de Qini."""
    y, score, treatment = np.asarray(y), np.asarray(score), np.asarray(treatment)
    _check(y, treatment, score)
    n, nt, nc, yt, yc = _cumulative(y, score, treatment)
    return _start_at_origin(n, yt - yc * _ratio(nt, nc))


def _perfect_uplift_score(y: np.ndarray, treatment: np.ndarray) -> np.ndarray:
    """Score d'un « oracle » qui connaîtrait les réponses : il place d'abord les traités qui
    répondent et les témoins qui ne répondent pas (ou l'inverse selon le cas le plus favorable)."""
    control_responders = np.sum((y == 1) & (treatment == 0))
    treated_non_responders = np.sum((y == 0) & (treatment == 1))
    summand = y if control_responders > treated_non_responders else treatment
    return 2 * (y == treatment) + summand


def _normalized_area(
    curve: Callable[..., tuple[np.ndarray, np.ndarray]],
    y: np.ndarray,
    score: np.ndarray,
    treatment: np.ndarray,
    perfect: np.ndarray,
) -> float:
    x_model, v_model = curve(y, score, treatment)
    x_best, v_best = curve(y, perfect, treatment)
    random_area = np.trapezoid([0.0, v_best[-1]], [0.0, x_best[-1]])  # diagonale = hasard
    best = np.trapezoid(v_best, x_best) - random_area
    model = np.trapezoid(v_model, x_model) - random_area
    return float(model / best) if best != 0 else float("nan")


def auuc(y, score, treatment) -> float:
    """Aire sous la courbe d'uplift, normalisée : 0 = hasard, 1 = oracle."""
    y, score, treatment = np.asarray(y), np.asarray(score), np.asarray(treatment)
    _check(y, treatment, score)
    return _normalized_area(uplift_curve, y, score, treatment, _perfect_uplift_score(y, treatment))


def qini(y, score, treatment) -> float:
    """Coefficient de Qini normalisé : 0 = hasard, 1 = oracle (effets négatifs autorisés)."""
    y, score, treatment = np.asarray(y), np.asarray(score), np.asarray(treatment)
    _check(y, treatment, score)
    perfect = y * treatment - y * (1 - treatment)
    return _normalized_area(qini_curve, y, score, treatment, perfect)


def uplift_at_k(y, score, treatment, k: float = 0.3) -> float:
    """Uplift observé (taux traités - taux témoins) parmi la fraction `k` de clients la mieux notée."""
    y, score, treatment = np.asarray(y), np.asarray(score), np.asarray(treatment)
    _check(y, treatment, score)
    if not 0 < k < 1:
        raise ValueError("k doit être dans ]0, 1[")
    top = np.argsort(score, kind="mergesort")[::-1][: int(len(y) * k)]
    yt, tt = y[top], treatment[top]
    if not ((tt == 0).any() and (tt == 1).any()):
        raise ValueError("Le haut du classement ne contient pas les deux groupes")
    return float(yt[tt == 1].mean() - yt[tt == 0].mean())


def uplift_by_decile(y, score, treatment, bins: int = 10) -> np.ndarray:
    """Uplift observé dans chaque tranche du classement (tranche 1 = scores les plus élevés).

    Un bon modèle donne des valeurs décroissantes de la première à la dernière tranche.
    """
    y, score, treatment = np.asarray(y), np.asarray(score), np.asarray(treatment)
    _check(y, treatment, score)
    order = np.argsort(score, kind="mergesort")[::-1]
    out = []
    for idx in np.array_split(order, bins):
        yb, tb = y[idx], treatment[idx]
        has_both = (tb == 1).any() and (tb == 0).any()
        out.append(yb[tb == 1].mean() - yb[tb == 0].mean() if has_both else np.nan)
    return np.array(out)


def per_treatment(
    metric: Callable[..., float],
    y,
    uplift: np.ndarray,
    treatment,
    control: int = 0,
    **kwargs,
) -> dict[int, float]:
    """Applique `metric` à chaque action contre le témoin (uplift : matrice n x K, colonne a-1)."""
    y, treatment, uplift = np.asarray(y), np.asarray(treatment), np.asarray(uplift)
    if uplift.ndim == 1:
        uplift = uplift[:, None]
    actions = [a for a in np.unique(treatment) if a != control]
    if len(actions) != uplift.shape[1]:
        raise ValueError("Une colonne d'uplift par action est attendue")
    res = {}
    for j, a in enumerate(actions):
        mask = (treatment == a) | (treatment == control)
        res[int(a)] = metric(y[mask], uplift[mask, j], (treatment[mask] == a).astype(int), **kwargs)
    return res
