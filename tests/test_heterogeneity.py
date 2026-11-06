import pandas as pd
import pytest

from ciblage.causal import heterogeneity as het
from ciblage.data import simulate as sim


@pytest.fixture(scope="module")
def trial():
    return sim.simulate(120_000, seed=21)


def test_segment_uplift_recovers_true_segment_average(trial):
    """Dans chaque segment, l'IC doit contenir l'uplift moyen vrai de ce segment."""
    res = het.uplift_by_segment(trial, "mens", "y")
    assert len(res) == 4  # 2 modalités x 2 actions
    for _, r in res.iterrows():
        truth = trial.loc[trial["mens"] == r["niveau"], f"tau_{r['treatment']}"].mean()
        assert r["ci_low"] <= truth <= r["ci_high"]


def test_mens_email_works_better_on_mens_buyers(trial):
    res = het.uplift_by_segment(trial, "mens", "y").set_index(["niveau", "treatment"])
    assert res.loc[(1.0, 1), "uplift"] > res.loc[(0.0, 1), "uplift"]


def test_small_cells_are_dropped():
    df = pd.DataFrame({"seg": ["a"] * 100 + ["b"] * 10, "treatment": [0, 1] * 55, "y": [0, 1] * 55})
    res = het.uplift_by_segment(df, "seg", "y", min_per_group=30)
    assert set(res["niveau"]) == {"a"}


def test_holm_adjustment_is_applied(trial):
    res = het.uplift_by_segment(trial, "newbie", "y")
    assert (res["p_holm"] >= res["p_value"]).all()


def test_bin_quantiles_balanced_labels():
    bins = het.bin_quantiles(pd.Series(range(100)), q=4)
    assert bins.value_counts().sort_index().tolist() == [25, 25, 25, 25]
    assert list(bins.cat.categories) == ["Q1", "Q2", "Q3", "Q4"]
