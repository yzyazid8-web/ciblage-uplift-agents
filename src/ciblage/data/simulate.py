"""Simulateur de campagne à uplift connu."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ciblage import config

TYPES = ["fidele", "persuadable", "perdu", "ne_pas_deranger"]
FEATURES = ["recency", "history", "mens", "womens", "newbie", "bruit_1", "bruit_2"]
N_ACTIONS = 2  # 1 = e-mail « homme », 2 = e-mail « femme » (comme Hillstrom)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)  # stabilité numérique
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _features(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Variables inspirées de Hillstrom (mêmes noms), plus deux variables de pur bruit."""
    return pd.DataFrame(
        {
            "recency": rng.integers(1, 13, n).astype(np.float64),  # mois depuis le dernier achat
            "history": np.round(rng.lognormal(5.0, 0.8, n), 2),  # montant dépensé l'an passé
            "mens": rng.binomial(1, 0.55, n).astype(np.float64),
            "womens": rng.binomial(1, 0.55, n).astype(np.float64),
            "newbie": rng.binomial(1, 0.5, n).astype(np.float64),
            "bruit_1": rng.normal(size=n),  # sans aucun lien avec les réponses :
            "bruit_2": rng.normal(size=n),  # un bon learner doit les ignorer
        }
    )


def type_probabilities(X: pd.DataFrame) -> np.ndarray:
    """P(type | x), matrice n × 4 dans l'ordre de `TYPES`."""
    rec = (X["recency"].to_numpy() - 6.5) / 3.5  # centré-réduit (approximativement)
    hist = (np.log(X["history"].to_numpy()) - 5.0) / 0.8
    newbie = X["newbie"].to_numpy()
    logits = np.column_stack(
        [
            -0.5 + 1.2 * hist - 0.8 * rec,  # fidèle : gros acheteur, achat récent
            0.3 + 0.4 * newbie - 0.3 * np.abs(hist),  # persuadable : profil « moyen », nouveau
            0.4 + 0.9 * rec - 0.6 * hist,  # perdu : inactif depuis longtemps
            -1.6 + 0.7 * rec * (1 - newbie),  # ne pas déranger : ancien client, inactif
        ]
    )
    return _softmax(logits)


def responsiveness(X: pd.DataFrame) -> np.ndarray:
    """P(sensible à l'action a | x) pour un persuadable, matrice n × N_ACTIONS."""
    mens = X["mens"].to_numpy()
    womens = X["womens"].to_numpy()
    return np.column_stack(
        [
            _sigmoid(-1.0 + 2.2 * mens - 0.4 * womens),  # e-mail homme : acheteurs « homme »
            _sigmoid(-1.2 + 2.0 * womens),  # e-mail femme : acheteurs « femme »
        ]
    )


def true_uplift(X: pd.DataFrame) -> np.ndarray:
    """Uplift vrai tau_a(x) = P(visite | x, action a) - P(visite | x, aucune action), n × N_ACTIONS."""
    p = type_probabilities(X)
    return p[:, [1]] * responsiveness(X) - p[:, [3]]


def baseline_rate(X: pd.DataFrame) -> np.ndarray:
    """P(visite | x, aucune action) = P(fidèle) + P(ne pas déranger)."""
    p = type_probabilities(X)
    return p[:, 0] + p[:, 3]


def simulate(
    n: int,
    propensity: tuple[float, ...] | None = None,
    seed: int = config.SEED,
) -> pd.DataFrame:
    """Simule un essai randomisé de `n` clients.

    `propensity` = probabilités d'affectation (témoin, action 1, action 2) ; par défaut 1/3 chacune.
    Elles ne dépendent pas de x : l'affectation est aléatoire, comme dans un vrai essai.

    Colonnes renvoyées :
      - variables explicatives (`FEATURES`), `treatment` (0, 1, 2) et `y` (résultat observé) ;
      - vérité terrain, INTERDITE aux modèles : `type`, `y0`, `y1`, `y2` (réponses potentielles),
        `tau_1`, `tau_2` (uplift vrai), `mu_0` (taux de base vrai).
    """
    if propensity is None:
        propensity = (1 / 3,) * (N_ACTIONS + 1)
    prop = np.asarray(propensity, dtype=np.float64)
    if prop.shape != (N_ACTIONS + 1,) or (prop < 0).any() or not np.isclose(prop.sum(), 1.0):
        raise ValueError(f"propensity doit contenir {N_ACTIONS + 1} probabilités de somme 1")

    rng = np.random.default_rng(seed)
    X = _features(n, rng)
    p_type = type_probabilities(X)
    p_resp = responsiveness(X)

    # Tirage du type latent (inverse de la fonction de répartition, ligne par ligne)
    u = rng.random((n, 1))
    types = (u > np.cumsum(p_type, axis=1)).sum(axis=1)
    types = np.minimum(types, len(TYPES) - 1)  # garde-fou contre les arrondis
    sensitive = rng.random((n, N_ACTIONS)) < p_resp

    fidele, persuadable, _, sleeping = (types == k for k in range(len(TYPES)))
    y0 = (fidele | sleeping).astype(np.int8)
    y_actions = (fidele[:, None] | (persuadable[:, None] & sensitive)).astype(np.int8)

    treatment = rng.choice(N_ACTIONS + 1, size=n, p=prop).astype(np.int8)
    potential = np.column_stack([y0, y_actions])
    y = potential[np.arange(n), treatment]

    out = X.copy()
    out["treatment"] = treatment
    out["y"] = y
    out["type"] = pd.Categorical.from_codes(types, categories=TYPES)
    out["y0"] = y0
    for a in range(N_ACTIONS):
        out[f"y{a + 1}"] = y_actions[:, a]
    tau = p_type[:, [1]] * p_resp - p_type[:, [3]]
    for a in range(N_ACTIONS):
        out[f"tau_{a + 1}"] = tau[:, a]
    out["mu_0"] = p_type[:, 0] + p_type[:, 3]
    return out
