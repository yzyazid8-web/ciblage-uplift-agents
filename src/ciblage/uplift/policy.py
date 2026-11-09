"""Valeur d'une politique de ciblage, estimée sans biais grâce à l'essai randomisé.

Une politique pi associe à chaque client une action (0 = rien, 1 = e-mail homme, ...).
Sa valeur V(pi) = E[Y(pi(x))] est le résultat moyen qu'on obtiendrait en l'appliquant à tous.

On ne peut pas la mesurer directement : chaque client n'a reçu qu'UNE action, tirée au hasard.
Mais le tirage au sort donne la solution (pondération inverse, IPW) : on garde les clients dont
l'action reçue coïncide avec celle que la politique aurait choisie, et on les repondère par
1 / probabilité d'avoir reçu cette action. En moyenne, on retrouve exactement V(pi).

    V_IPW(pi)   = mean( 1{T = pi(x)} * Y / e(T) )
    V_Hajek(pi) = sum( 1{T = pi(x)} * Y / e(T) ) / sum( 1{T = pi(x)} / e(T) )   (plus stable)

Les intervalles de confiance sont obtenus par bootstrap. Pour comparer deux politiques, le
bootstrap est APPARIÉ (mêmes rééchantillons pour les deux) : l'écart est beaucoup plus précis
que la différence de deux intervalles séparés.
"""

from __future__ import annotations

import numpy as np

from ciblage import config


def _weights(treatment: np.ndarray, policy: np.ndarray, propensity: dict[int, float]) -> np.ndarray:
    t = np.asarray(treatment)
    pi = np.asarray(policy)
    if len(t) != len(pi):
        raise ValueError("treatment et policy doivent avoir la même longueur")
    missing = set(np.unique(pi)) - set(propensity)
    if missing:
        raise ValueError(f"Actions sans propension connue : {sorted(missing)}")
    e = np.array([propensity[int(a)] for a in t], dtype=np.float64)
    return (t == pi) / e


def ipw_value(y, treatment, policy, propensity: dict[int, float], normalized: bool = True) -> float:
    """Valeur estimée de la politique (Hajek si `normalized`, IPW pur sinon)."""
    w = _weights(np.asarray(treatment), np.asarray(policy), propensity)
    yy = np.asarray(y, dtype=np.float64)
    if normalized:
        return float((w * yy).sum() / w.sum()) if w.sum() > 0 else float("nan")
    return float((w * yy).mean())


def compare_policies(
    y,
    treatment,
    policies: dict[str, np.ndarray],
    propensity: dict[int, float],
    baseline: str,
    n_boot: int = 1_000,
    level: float = 0.95,
    seed: int = config.SEED,
) -> dict[str, dict[str, float]]:
    """Valeur de chaque politique et gain vs `baseline`, avec IC bootstrap apparié."""
    if baseline not in policies:
        raise ValueError(f"Politique de référence inconnue : {baseline}")
    yy = np.asarray(y, dtype=np.float64)
    t = np.asarray(treatment)
    n = len(yy)
    weights = {k: _weights(t, p, propensity) for k, p in policies.items()}

    def value(w: np.ndarray, idx: np.ndarray | None = None) -> float:
        if idx is not None:
            w, y_ = w[idx], yy[idx]
        else:
            y_ = yy
        return float((w * y_).sum() / w.sum())

    rng = np.random.default_rng(seed)
    boots = {k: np.empty(n_boot) for k in policies}
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        for k, w in weights.items():
            boots[k][b] = value(w, idx)

    alpha = (1 - level) / 2
    out: dict[str, dict[str, float]] = {}
    for k, w in weights.items():
        diff = boots[k] - boots[baseline]
        out[k] = {
            "value": value(w),
            "value_low": float(np.quantile(boots[k], alpha)),
            "value_high": float(np.quantile(boots[k], 1 - alpha)),
            "gain": value(w) - value(weights[baseline]),
            "gain_low": float(np.quantile(diff, alpha)),
            "gain_high": float(np.quantile(diff, 1 - alpha)),
            "share_treated": float(np.mean(np.asarray(policies[k]) != 0)),
        }
    return out


def best_action_policy(uplift: np.ndarray, actions: list[int], frac: float = 1.0) -> np.ndarray:
    """Politique simple : à chacun sa meilleure action si son uplift est positif, en ne gardant
    que la fraction `frac` des clients aux plus forts uplifts (budget d'envois limité)."""
    u = np.asarray(uplift, dtype=np.float64)
    if u.ndim == 1:
        u = u[:, None]
    best = u.argmax(axis=1)
    best_gain = u[np.arange(len(u)), best]
    pi = np.where(best_gain > 0, np.asarray(actions)[best], 0)
    if frac < 1.0:
        n_keep = int(len(u) * frac)
        keep = np.zeros(len(u), dtype=bool)
        keep[np.argsort(-best_gain, kind="mergesort")[:n_keep]] = True
        pi = np.where(keep, pi, 0)
    return pi.astype(np.int64)


def top_score_policy(score: np.ndarray, action: int, frac: float) -> np.ndarray:
    """Envoie `action` à la fraction `frac` des clients au score le plus élevé (ciblage classique)."""
    s = np.asarray(score, dtype=np.float64)
    pi = np.zeros(len(s), dtype=np.int64)
    pi[np.argsort(-s, kind="mergesort")[: int(len(s) * frac)]] = action
    return pi
