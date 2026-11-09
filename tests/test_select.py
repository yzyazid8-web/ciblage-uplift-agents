import numpy as np

from ciblage.data import simulate as sim
from ciblage.uplift import learners as L
from ciblage.uplift import select


def test_cross_val_uplift_is_out_of_fold_and_shaped():
    df = sim.simulate(4_000, seed=9)
    oof = select.cross_val_uplift(
        lambda: L.TLearner(params={"n_estimators": 50}),
        df[sim.FEATURES],
        df["treatment"],
        df["y"],
        n_folds=3,
    )
    assert oof.shape == (len(df), 2)
    assert np.isfinite(oof).all()
    # chaque client a reçu une prédiction (aucune ligne laissée à zéro)
    assert (np.abs(oof).sum(axis=1) > 0).mean() > 0.99


def test_response_score_and_ranking_metrics():
    df = sim.simulate(4_000, seed=10)
    t, y = df["treatment"].to_numpy(), df["y"].to_numpy()
    score = select.cross_val_response(df[sim.FEATURES], t, y, n_folds=3)
    assert ((score >= 0) & (score <= 1)).all()
    rk = select.ranking_metrics(y, t, df[["tau_1", "tau_2"]].to_numpy())
    assert set(rk) == {1, 2}
    assert set(rk[1]) == {"qini", "auuc", "uplift_at_30"}
