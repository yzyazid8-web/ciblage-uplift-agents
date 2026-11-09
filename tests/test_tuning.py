import numpy as np

from ciblage.data import simulate as sim
from ciblage.uplift import learners as L
from ciblage.uplift import tuning
from ciblage.uplift.select import cross_val_uplift


def test_holdout_split_is_disjoint_stratified_and_reproducible():
    df = sim.simulate(10_000, seed=1)
    t, y = df["treatment"].to_numpy(), df["y"].to_numpy()
    tr, te = tuning.holdout_split(t, y, test_size=0.3)
    assert len(np.intersect1d(tr, te)) == 0 and len(tr) + len(te) == len(df)
    assert len(te) == 3_000
    for g in (0, 1, 2):  # mêmes proportions de groupes des deux côtés
        assert abs((t[tr] == g).mean() - (t[te] == g).mean()) < 0.01
    tr2, te2 = tuning.holdout_split(t, y, test_size=0.3)
    np.testing.assert_array_equal(te, te2)


def test_grid_enumerates_all_combinations():
    combos = tuning.grid({"a": [1, 2], "b": ["x", "y", "z"]})
    assert len(combos) == 6
    assert {"a": 2, "b": "z"} in combos


def test_effect_params_override_only_effect_models():
    x = L.XLearner(params={"max_depth": 4}, effect_params={"max_depth": 1})
    assert x.params["max_depth"] == 4
    assert x.effect_params["max_depth"] == 1
    assert x._effect_model().get_params()["max_depth"] == 1
    assert x._response_model().get_params()["max_depth"] == 4


def test_tune_explores_effect_grid_only_for_x_and_dr():
    df = sim.simulate(3_000, seed=2)
    small_r = {"max_depth": [2], "n_estimators": [30]}
    small_e = {"max_depth": [1, 2], "n_estimators": [30]}
    args = (df[sim.FEATURES], df["treatment"].to_numpy(), df["y"].to_numpy(), cross_val_uplift)
    t_res = tuning.tune(L.TLearner, *args, response_grid=small_r, effect_grid=small_e, n_folds=2)
    x_res = tuning.tune(L.XLearner, *args, response_grid=small_r, effect_grid=small_e, n_folds=2)
    assert list(t_res.table["etape"]) == ["défaut", "réponse"]
    assert list(x_res.table["etape"]) == ["défaut", "réponse", "effet", "effet"]
    assert x_res.best_score >= x_res.default_score
    assert t_res.best_effect_params == {}
