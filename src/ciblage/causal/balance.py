"""Vérification de la randomisation : les groupes sont-ils comparables avant la campagne ?"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from ciblage import config

SMD_THRESHOLD = 0.1


def smd(x_treated: np.ndarray, x_control: np.ndarray) -> float:

    xt = np.asarray(x_treated, dtype=np.float64)
    xc = np.asarray(x_control, dtype=np.float64)
    pooled = np.sqrt((xt.var(ddof=1) + xc.var(ddof=1)) / 2)
    if pooled == 0:
        return 0.0 if xt.mean() == xc.mean() else float(np.inf)
    return float((xt.mean() - xc.mean()) / pooled)


def smd_table(X: pd.DataFrame, treatment: np.ndarray, control: int = 0) -> pd.DataFrame:
    """SMD de chaque variable (lignes) pour chaque groupe traité (colonnes) face au témoin."""
    t = np.asarray(treatment)
    groups = [g for g in np.unique(t) if g != control]
    is_control = t == control
    rows = {
        col: {
            f"smd_{g}": smd(X.loc[t == g, col].to_numpy(), X.loc[is_control, col].to_numpy())
            for g in groups
        }
        for col in X.columns
    }
    return pd.DataFrame.from_dict(rows, orient="index")


def _is_binary(x: pd.Series) -> bool:
    return bool(x.dropna().isin([0, 1]).all())


TEST_MAX_N = 200_000


def distribution_tests(
    X: pd.DataFrame,
    treatment: np.ndarray,
    max_n: int = TEST_MAX_N,
    seed: int = config.SEED,
) -> pd.DataFrame:
    """p-valeur d'un test d'égalité des distributions entre tous les groupes, par variable."""
    t = np.asarray(treatment)
    if len(t) > max_n:
        idx = np.random.default_rng(seed).choice(len(t), size=max_n, replace=False)
        X, t = X.iloc[idx], t[idx]
    groups = np.unique(t)
    out = {}
    for col in X.columns:
        x = X[col]
        if _is_binary(x):
            table = pd.crosstab(t, x.to_numpy())
            p = float(stats.chi2_contingency(table)[1])
            test = "chi2"
        else:
            samples = [x.to_numpy()[t == g] for g in groups]
            p = float(stats.kruskal(*samples).pvalue)
            test = "kruskal"
        out[col] = {"test": test, "p_value": p}
    return pd.DataFrame.from_dict(out, orient="index")


def sample_ratio_test(treatment: np.ndarray, expected: dict[int, float]) -> dict[str, float]:
    """Test du chi-2 d'adéquation : effectifs observés vs proportions prévues par le protocole."""
    t = np.asarray(treatment)
    keys = sorted(expected)
    observed = np.array([(t == k).sum() for k in keys], dtype=np.float64)
    probs = np.array([expected[k] for k in keys], dtype=np.float64)
    if not np.isclose(probs.sum(), 1.0):
        raise ValueError("Les proportions attendues doivent sommer à 1")
    if observed.sum() != len(t):
        raise ValueError("Des groupes observés sont absents des proportions attendues")
    res = stats.chisquare(observed, f_exp=probs * observed.sum())
    return {"chi2": float(res.statistic), "p_value": float(res.pvalue)}


@dataclass
class BalanceReport:
    smd: pd.DataFrame
    tests: pd.DataFrame
    sample_ratio: dict[str, float] | None  # None : protocole d'affectation inconnu

    @property
    def max_abs_smd(self) -> float:
        return float(self.smd.abs().to_numpy().max())

    @property
    def is_balanced(self) -> bool:
        """Toutes les |SMD| sous le seuil ET, si le protocole est connu, effectifs conformes (p > 0,001)."""
        ratio_ok = self.sample_ratio is None or self.sample_ratio["p_value"] > 1e-3
        return self.max_abs_smd < SMD_THRESHOLD and ratio_ok


def check_balance(
    X: pd.DataFrame,
    treatment: np.ndarray,
    expected: dict[int, float] | None,
    control: int = 0,
) -> BalanceReport:
    """Rapport complet d'équilibre (SMD, tests par variable, effectifs si `expected` est connu)."""
    return BalanceReport(
        smd=smd_table(X, treatment, control),
        tests=distribution_tests(X, treatment),
        sample_ratio=None if expected is None else sample_ratio_test(treatment, expected),
    )
