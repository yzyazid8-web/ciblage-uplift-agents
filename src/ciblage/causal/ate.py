"""Effet moyen du traitement (ATE) dans un essai randomisé."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import stats

from ciblage import config


@dataclass
class Effect:
    treatment: int
    outcome: str
    n_treated: int
    n_control: int
    mean_treated: float
    mean_control: float
    ate: float
    lift: float  # effet relatif : ate / moyenne témoin
    se: float
    ci_low: float  # IC de Welch
    ci_high: float
    boot_low: float  # IC bootstrap percentile
    boot_high: float
    p_value: float


def welch(y_treated: np.ndarray, y_control: np.ndarray, level: float = 0.95) -> dict[str, float]:
    """Différence de moyennes, erreur-type, IC et p-valeur de Welch (bilatérale)."""
    yt = np.asarray(y_treated, dtype=np.float64)
    yc = np.asarray(y_control, dtype=np.float64)
    nt, nc = len(yt), len(yc)
    if nt < 2 or nc < 2:
        raise ValueError("Il faut au moins 2 observations par groupe")
    vt, vc = yt.var(ddof=1) / nt, yc.var(ddof=1) / nc
    diff = yt.mean() - yc.mean()
    se = np.sqrt(vt + vc)
    if se == 0:
        return {
            "ate": diff,
            "se": 0.0,
            "ci_low": diff,
            "ci_high": diff,
            "p_value": float(diff == 0),
        }
    # Degrés de liberté de Welch-Satterthwaite
    dof = (vt + vc) ** 2 / (vt**2 / (nt - 1) + vc**2 / (nc - 1))
    q = stats.t.ppf(0.5 + level / 2, dof)
    p = 2 * stats.t.sf(abs(diff) / se, dof)
    return {
        "ate": float(diff),
        "se": float(se),
        "ci_low": float(diff - q * se),
        "ci_high": float(diff + q * se),
        "p_value": float(p),
    }


def bootstrap_ci(
    y_treated: np.ndarray,
    y_control: np.ndarray,
    n_boot: int = 2_000,
    level: float = 0.95,
    seed: int = config.SEED,
) -> tuple[float, float]:
    """IC bootstrap percentile de la différence de moyennes (rééchantillonnage par groupe)."""
    yt = np.asarray(y_treated, dtype=np.float64)
    yc = np.asarray(y_control, dtype=np.float64)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    batch = max(1, 2_000_000 // max(len(yt), len(yc)))  # borne la mémoire 
    for start in range(0, n_boot, batch):
        b = min(batch, n_boot - start)
        mt = yt[rng.integers(0, len(yt), (b, len(yt)))].mean(axis=1)
        mc = yc[rng.integers(0, len(yc), (b, len(yc)))].mean(axis=1)
        diffs[start : start + b] = mt - mc
    alpha = 1 - level
    low, high = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return float(low), float(high)


def holm(p_values: np.ndarray) -> np.ndarray:
    """p-valeurs ajustées de Holm-Bonferroni (dans l'ordre d'entrée)."""
    p = np.asarray(p_values, dtype=np.float64)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.maximum.accumulate(np.minimum(1.0, p[order] * (m - np.arange(m))))
    out = np.empty(m)
    out[order] = adjusted
    return out


def estimate_effects(
    df: pd.DataFrame,
    outcomes: list[str],
    treatment_col: str = "treatment",
    control: int = 0,
    n_boot: int = 2_000,
    seed: int = config.SEED,
) -> pd.DataFrame:
    """ATE de chaque action sur chaque résultat, avec IC et p-valeurs ajustées (Holm)."""
    t = df[treatment_col].to_numpy()
    treatments = [g for g in np.unique(t) if g != control]
    effects = []
    for g in treatments:
        for outcome in outcomes:
            yt = df.loc[t == g, outcome].to_numpy()
            yc = df.loc[t == control, outcome].to_numpy()
            w = welch(yt, yc)
            lo, hi = bootstrap_ci(yt, yc, n_boot=n_boot, seed=seed)
            mc = float(yc.mean())
            effects.append(
                Effect(
                    treatment=int(g),
                    outcome=outcome,
                    n_treated=len(yt),
                    n_control=len(yc),
                    mean_treated=float(yt.mean()),
                    mean_control=mc,
                    ate=w["ate"],
                    lift=w["ate"] / mc if mc != 0 else float("nan"),
                    se=w["se"],
                    ci_low=w["ci_low"],
                    ci_high=w["ci_high"],
                    boot_low=lo,
                    boot_high=hi,
                    p_value=w["p_value"],
                )
            )
    table = pd.DataFrame([asdict(e) for e in effects])
    table["p_holm"] = holm(table["p_value"].to_numpy())
    return table
