"""Estimation causale observationnelle : propension, AIPW et fabrique de confounding."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ciblage.causal import confounding
from ciblage.causal.aipw import aipw_ate, ipw_ate, naive_ate
from ciblage.causal.propensity import overlap, propensity_cv, smd_max


def _sim(n: int = 4000, tau: float = 0.1, seed: int = 0):
    """Données confondues à effet CONNU : la propension et le résultat dépendent tous deux de x0."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    e_true = 1.0 / (1.0 + np.exp(-(1.5 * x[:, 0])))
    w = (rng.random(n) < e_true).astype(int)
    base = 1.0 / (1.0 + np.exp(-(-0.5 + 1.0 * x[:, 0] + 0.5 * x[:, 1])))
    p = np.clip(base + tau * w, 0.01, 0.99)
    y = (rng.random(n) < p).astype(np.float64)
    X = pd.DataFrame(x, columns=["x0", "x1", "x2"])
    return X, w, y


def test_propensity_cv_returns_probabilities() -> None:
    X, w, _ = _sim(seed=1)
    e = propensity_cv(X, w)
    assert e.shape == (len(w),)
    assert (e > 0).all() and (e < 1).all()
    # la propension doit croître avec x0 (qui pilote l'affectation)
    assert np.corrcoef(e, X["x0"])[0, 1] > 0.5


def test_aipw_corrects_confounding_bias() -> None:
    X, w, y = _sim(tau=0.1, seed=0)
    e = propensity_cv(X, w)
    naive = naive_ate(y, w)
    ipw = ipw_ate(y, w, e)
    aipw = aipw_ate(X, w, y, e)
    # le naïf est biaisé vers le haut (traités = x0 élevé = base plus forte)
    assert naive.ate > 0.14
    # IPW et AIPW se rapprochent de la vérité 0,10
    assert abs(aipw.ate - 0.10) < 0.04
    assert abs(aipw.ate - 0.10) < abs(naive.ate - 0.10)
    assert abs(ipw.ate - 0.10) < abs(naive.ate - 0.10)


def test_overlap_diagnostics() -> None:
    X, w, _ = _sim(seed=2)
    e = propensity_cv(X, w)
    ov = overlap(e, w)
    assert 0.0 < ov.e_min <= ov.e_max < 1.0
    assert 0.0 < ov.ess_ratio_treated <= 1.0
    assert 0.0 < ov.ess_ratio_control <= 1.0


def test_weighting_reduces_imbalance() -> None:
    X, w, _ = _sim(seed=3)
    e = propensity_cv(X, w)
    assert smd_max(X, w, e=e) < smd_max(X, w, e=None)


def test_make_confounded_induces_imbalance() -> None:
    rng = np.random.default_rng(0)
    n = 6000
    cov = rng.normal(size=n)
    df = pd.DataFrame({"treatment": rng.integers(0, 2, n), "cov": cov, "y": rng.random(n)})
    before = df.loc[df.treatment == 1, "cov"].mean() - df.loc[df.treatment == 0, "cov"].mean()
    conf = confounding.make_confounded(df, action=1, covariate="cov", strength=2.0)
    after = (
        conf.loc[conf.treatment == 1, "cov"].mean() - conf.loc[conf.treatment == 0, "cov"].mean()
    )
    assert abs(before) < 0.1  # randomisé : covariable équilibrée
    assert after > 0.5  # confondu : les traités ont une covariable nettement plus élevée


def test_aipw_biased_when_confounder_dropped() -> None:
    # (b) retirer le confondant (x0) de l'ajustement fait revenir le biais : il n'est plus "mesuré".
    X, w, y = _sim(tau=0.1, seed=0)
    full = aipw_ate(X, w, y, propensity_cv(X, w))
    X_drop = X.drop(columns=["x0"])
    dropped = aipw_ate(X_drop, w, y, propensity_cv(X_drop, w))
    assert abs(dropped.ate - 0.10) > abs(full.ate - 0.10)  # plus loin de la vérité
    assert dropped.ate > full.ate  # biais vers le haut, comme le naïf


def test_confounding_strength_monotone() -> None:
    # (a) plus l'intensité est forte, plus le déséquilibre induit est grand.
    rng = np.random.default_rng(0)
    n = 8000
    df = pd.DataFrame({"treatment": rng.integers(0, 2, n), "cov": rng.normal(size=n)})

    def imbalance(s: float) -> float:
        c = confounding.make_confounded(df, action=1, covariate="cov", strength=s)
        return c.loc[c.treatment == 1, "cov"].mean() - c.loc[c.treatment == 0, "cov"].mean()

    assert imbalance(0.0) < imbalance(1.0) < imbalance(2.0)
