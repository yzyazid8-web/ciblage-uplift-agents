"""Fabrique un biais de confusion CONNU à partir d'un essai randomisé."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ciblage import config


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def make_confounded(
    df: pd.DataFrame,
    action: int,
    covariate: str,
    strength: float = 1.5,
    control: int = 0,
    treatment_col: str = "treatment",
    seed: int = config.SEED,
) -> pd.DataFrame:
    """Sous-échantillon biaisé des groupes {control, action}, confondu par `covariate`."""
    sub = df[df[treatment_col].isin([control, action])].copy()
    if covariate not in sub.columns:
        raise ValueError(f"Covariable inconnue : {covariate}")
    rng = np.random.default_rng(seed)
    z = sub[covariate].to_numpy(dtype=np.float64)
    z = (z - z.mean()) / (z.std() + 1e-12)  # standardisée
    is_action = (sub[treatment_col] == action).to_numpy()
    # Probabilité de conserver le client, opposée selon le groupe -> corrélation traitement/covariable.
    keep_p = np.where(is_action, _sigmoid(strength * z), _sigmoid(-strength * z))
    keep = rng.random(len(sub)) < keep_p
    return sub.loc[keep].reset_index(drop=True)
