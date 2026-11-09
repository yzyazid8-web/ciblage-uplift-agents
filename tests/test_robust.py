import numpy as np
import pytest

from ciblage.data import simulate as sim
from ciblage.optim import robust
from ciblage.uplift.learners import TLearner


def test_uncertainty_shapes_and_positive_spread():
    df = sim.simulate(6_000, seed=3)
    X = df[sim.FEATURES].to_numpy()
    t, y = df["treatment"].to_numpy(), df["y"].to_numpy()
    mean, std = robust.uplift_with_uncertainty(
        TLearner,
        {"n_estimators": 40, "max_depth": 2},
        X[:5_000],
        t[:5_000],
        y[:5_000],
        X[5_000:],
        n_models=3,
    )
    assert mean.shape == std.shape == (1_000, 2)
    assert (std >= 0).all() and std.mean() > 0


def test_prudent_score_penalizes_uncertain_clients():
    mean = np.array([[0.10], [0.10]])
    std = np.array([[0.01], [0.08]])
    s = robust.prudent_score(mean, std, z=1.0)
    assert s[0, 0] > s[1, 0]
    np.testing.assert_allclose(robust.prudent_score(mean, std, z=0.0), mean)


def test_invalid_arguments():
    with pytest.raises(ValueError, match="z doit"):
        robust.prudent_score(np.zeros(1), np.zeros(1), z=-1)
    with pytest.raises(ValueError, match="2 modèles"):
        robust.uplift_with_uncertainty(
            TLearner, {}, np.zeros((4, 1)), np.zeros(4), np.zeros(4), np.zeros((1, 1)), n_models=1
        )
