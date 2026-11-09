"""Validation des learners sur le simulateur, où l'uplift vrai est connu."""

import numpy as np
import pytest

from ciblage.data import simulate as sim
from ciblage.uplift import learners as L

FAST = {"n_estimators": 150}


@pytest.fixture(scope="module")
def split():
    df = sim.simulate(40_000, seed=31)
    train, test = df.iloc[:30_000], df.iloc[30_000:]
    return train, test


def _rmse(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - truth) ** 2)))


@pytest.fixture(scope="module")
def fitted(split):
    train, _ = split
    return {
        name: cls(params=FAST).fit(train[sim.FEATURES], train["treatment"], train["y"])
        for name, cls in L.LEARNERS.items()
    }


@pytest.mark.parametrize("name", list(L.LEARNERS))
def test_output_shape_one_column_per_action(fitted, split, name):
    _, test = split
    pred = fitted[name].predict(test[sim.FEATURES])
    assert pred.shape == (len(test), 2)
    assert fitted[name].actions_ == [1, 2]


@pytest.mark.parametrize("name", ["T-learner", "X-learner", "DR-learner"])
def test_learner_beats_constant_effect(fitted, split, name):
    """Erreur (PEHE) plus faible qu'en prédisant le même effet moyen pour tout le monde."""
    train, test = split
    pred = fitted[name].predict(test[sim.FEATURES])
    for j, col in enumerate(["tau_1", "tau_2"]):
        truth = test[col].to_numpy()
        constant = np.full_like(truth, train[col].mean())
        assert _rmse(pred[:, j], truth) < 0.8 * _rmse(constant, truth)
        assert np.corrcoef(pred[:, j], truth)[0, 1] > 0.6


def test_learners_detect_clients_not_to_disturb(fitted, split):
    """Les clients à uplift vrai négatif doivent recevoir un uplift prédit plus faible."""
    _, test = split
    pred = fitted["DR-learner"].predict(test[sim.FEATURES])[:, 0]
    negative = test["tau_1"].to_numpy() < 0
    assert pred[negative].mean() < pred[~negative].mean() - 0.05


def test_dr_pseudo_outcomes_are_unbiased_on_average(split):
    """La moyenne des pseudo-réponses DR estime l'effet moyen vrai (double robustesse)."""
    train, _ = split
    dr = L.DRLearner(params=FAST)
    Xa, t, ya = dr._prepare(train[sim.FEATURES], train["treatment"], train["y"])
    phi = dr.pseudo_outcomes(Xa, t, ya)
    for j, col in enumerate(["tau_1", "tau_2"]):
        assert phi[:, j].mean() == pytest.approx(train[col].mean(), abs=0.015)


def test_continuous_outcome_is_supported():
    df = sim.simulate(6_000, seed=2)
    y = df["y"] * np.random.default_rng(0).lognormal(4, 0.5, len(df))  # une « dépense »
    pred = L.TLearner(outcome="continuous", params=FAST).fit(df[sim.FEATURES], df["treatment"], y)
    assert pred.predict(df[sim.FEATURES]).shape == (len(df), 2)


def test_known_propensity_is_used_by_x_learner():
    df = sim.simulate(3_000, seed=5)
    x = L.XLearner(params=FAST, propensity={0: 0.5, 1: 0.25, 2: 0.25})
    x.fit(df[sim.FEATURES], df["treatment"], df["y"])
    assert x.propensity_[1] == 0.25


@pytest.mark.parametrize(
    "y, t, message",
    [
        ([0, 2, 1, 0], [0, 1, 0, 1], "binaire"),
        ([0, 1, 1, 0], [1, 1, 1, 1], "témoin"),
        ([0, 1, 1, 0], [0, 0, 0, 0], "traité"),
    ],
)
def test_input_validation(y, t, message):
    X = np.zeros((4, 2))
    with pytest.raises(ValueError, match=message):
        L.TLearner().fit(X, t, y)
