import numpy as np
import pytest
from scipy import stats

from ciblage.causal import ate
from ciblage.data import simulate as sim


def test_welch_matches_scipy():
    rng = np.random.default_rng(0)
    yt, yc = rng.normal(1.2, 2.0, 300), rng.normal(1.0, 1.0, 500)
    res = ate.welch(yt, yc)
    ref = stats.ttest_ind(yt, yc, equal_var=False)
    ci = ref.confidence_interval(0.95)
    assert res["p_value"] == pytest.approx(ref.pvalue, rel=1e-9)
    assert res["ci_low"] == pytest.approx(ci.low, rel=1e-9)
    assert res["ci_high"] == pytest.approx(ci.high, rel=1e-9)


def test_welch_requires_two_observations_per_group():
    with pytest.raises(ValueError, match="2 observations"):
        ate.welch(np.array([1.0]), np.array([0.0, 1.0]))


def test_bootstrap_ci_is_close_to_welch_on_large_samples():
    rng = np.random.default_rng(1)
    yt, yc = rng.binomial(1, 0.18, 20_000), rng.binomial(1, 0.11, 20_000)
    lo, hi = ate.bootstrap_ci(yt, yc, n_boot=1_000)
    w = ate.welch(yt, yc)
    assert lo == pytest.approx(w["ci_low"], abs=0.002)
    assert hi == pytest.approx(w["ci_high"], abs=0.002)


def test_holm_matches_textbook_example():
    # Exemple : p = (0.01, 0.04, 0.03, 0.005) -> ajustées (0.03, 0.06, 0.06, 0.02)
    adjusted = ate.holm(np.array([0.01, 0.04, 0.03, 0.005]))
    np.testing.assert_allclose(adjusted, [0.03, 0.06, 0.06, 0.02])


def test_holm_caps_at_one():
    assert ate.holm(np.array([0.6, 0.7])).max() == 1.0


def test_confidence_intervals_cover_true_effect_at_nominal_rate():
    """Sur 200 essais simulés, l'IC à 95 % doit contenir l'effet vrai environ 95 % du temps."""
    covered = 0
    for seed in range(200):
        df = sim.simulate(3_000, seed=seed)
        truth = (df["y1"] - df["y0"]).mean()  # effet moyen vrai dans cet échantillon
        w = ate.welch(df.loc[df["treatment"] == 1, "y"], df.loc[df["treatment"] == 0, "y"])
        covered += w["ci_low"] <= truth <= w["ci_high"]
    assert 0.90 <= covered / 200 <= 0.99


def test_estimate_effects_table_on_simulated_trial():
    df = sim.simulate(30_000, seed=5)
    table = ate.estimate_effects(df, outcomes=["y"], n_boot=300)
    assert set(table["treatment"]) == {1, 2}
    for _, row in table.iterrows():
        assert row["ci_low"] <= df[f"tau_{row['treatment']}"].mean() <= row["ci_high"]
        assert row["p_holm"] >= row["p_value"]
