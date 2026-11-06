import numpy as np
import pandas as pd
import pytest

from ciblage.causal import balance
from ciblage.data import simulate as sim


def test_smd_matches_hand_computation():
    xt = np.array([1.0, 2.0, 3.0, 4.0])
    xc = np.array([0.0, 1.0, 2.0, 3.0])
    pooled = np.sqrt((np.var(xt, ddof=1) + np.var(xc, ddof=1)) / 2)
    assert balance.smd(xt, xc) == pytest.approx(1.0 / pooled)


def test_smd_handles_constant_columns():
    assert balance.smd(np.ones(5), np.ones(5)) == 0.0
    assert balance.smd(np.ones(5), np.zeros(5)) == np.inf


def test_randomized_trial_is_balanced():
    df = sim.simulate(30_000, seed=11)
    rep = balance.check_balance(
        df[sim.FEATURES], df["treatment"].to_numpy(), {0: 1 / 3, 1: 1 / 3, 2: 1 / 3}
    )
    assert rep.is_balanced
    assert list(rep.smd.columns) == ["smd_1", "smd_2"]


def test_biased_assignment_is_detected():
    """Si les gros acheteurs reçoivent plus souvent l'e-mail, l'équilibre doit être rejeté."""
    df = sim.simulate(30_000, seed=12)
    big = df["history"] > df["history"].median()
    rng = np.random.default_rng(0)
    t = np.where(big & (rng.random(len(df)) < 0.4), 1, df["treatment"])
    rep = balance.check_balance(df[sim.FEATURES], t, {0: 1 / 3, 1: 1 / 3, 2: 1 / 3})
    assert not rep.is_balanced
    assert rep.smd.loc["history", "smd_1"] > balance.SMD_THRESHOLD
    assert rep.tests.loc["history", "p_value"] < 1e-6


def test_sample_ratio_mismatch_is_detected():
    t = np.array([0] * 4_000 + [1] * 6_000)
    assert balance.sample_ratio_test(t, {0: 0.5, 1: 0.5})["p_value"] < 1e-10
    assert balance.sample_ratio_test(t, {0: 0.4, 1: 0.6})["p_value"] > 0.5


def test_sample_ratio_rejects_bad_expected():
    with pytest.raises(ValueError, match="sommer"):
        balance.sample_ratio_test(np.array([0, 1]), {0: 0.5, 1: 0.6})
    with pytest.raises(ValueError, match="absents"):
        balance.sample_ratio_test(np.array([0, 1, 2]), {0: 0.5, 1: 0.5})


def test_distribution_tests_choose_test_by_variable_type():
    X = pd.DataFrame({"binaire": [0, 1] * 50, "continue": np.linspace(0, 1, 100)})
    res = balance.distribution_tests(X, np.array([0, 1] * 50))
    assert res.loc["binaire", "test"] == "chi2"
    assert res.loc["continue", "test"] == "kruskal"


def test_tests_subsample_large_data_and_survive_massive_ties():
    """Sur de très gros volumes avec une valeur répétée des millions de fois : pas de NaN."""
    n = 2_000_000
    rng = np.random.default_rng(3)
    X = pd.DataFrame({"tres_repetee": np.where(rng.random(n) < 0.9, 8.2, rng.normal(8.5, 0.1, n))})
    res = balance.distribution_tests(X, rng.integers(0, 2, n))
    assert np.isfinite(res.loc["tres_repetee", "p_value"])


def test_unknown_protocol_skips_sample_ratio_test():
    df = sim.simulate(5_000, seed=4)
    rep = balance.check_balance(df[sim.FEATURES], df["treatment"].to_numpy(), expected=None)
    assert rep.sample_ratio is None
    assert rep.is_balanced
