"""Nos métriques sont implémentées indépendamment ; scikit-uplift sert uniquement de référence."""

import numpy as np
import pytest
from sklift import metrics as ref

from ciblage.data import simulate as sim
from ciblage.uplift import metrics as m


@pytest.fixture(scope="module")
def data():
    df = sim.simulate(6_000, seed=3)
    df = df[df["treatment"] <= 1]
    y, t = df["y"].to_numpy(), df["treatment"].to_numpy()
    rng = np.random.default_rng(0)
    scores = {
        "vrai": df["tau_1"].to_numpy(),  # le classement idéal (inconnu en pratique)
        "bruite": df["tau_1"].to_numpy() + rng.normal(0, 0.1, len(df)),
        "hasard": rng.random(len(df)),
        "ex_aequo": np.round(df["tau_1"].to_numpy(), 1),  # beaucoup de scores identiques
    }
    return y, t, scores


@pytest.mark.parametrize("name", ["vrai", "bruite", "hasard", "ex_aequo"])
def test_curves_match_reference(data, name):
    y, t, scores = data
    for ours, theirs in [(m.uplift_curve, ref.uplift_curve), (m.qini_curve, ref.qini_curve)]:
        x1, v1 = ours(y, scores[name], t)
        x2, v2 = theirs(y, scores[name], t)
        np.testing.assert_allclose(x1, x2)
        np.testing.assert_allclose(v1, v2, atol=1e-9)


@pytest.mark.parametrize("name", ["vrai", "bruite", "hasard", "ex_aequo"])
def test_normalized_scores_match_reference(data, name):
    y, t, scores = data
    assert m.auuc(y, scores[name], t) == pytest.approx(ref.uplift_auc_score(y, scores[name], t))
    assert m.qini(y, scores[name], t) == pytest.approx(ref.qini_auc_score(y, scores[name], t))


def test_uplift_at_k_matches_reference(data):
    y, t, scores = data
    for k in (0.1, 0.3, 0.5):
        expected = ref.uplift_at_k(y, scores["bruite"], t, strategy="overall", k=k)
        assert m.uplift_at_k(y, scores["bruite"], t, k=k) == pytest.approx(expected)


def test_true_uplift_ranks_better_than_random():
    """Même l'uplift VRAI a un Qini normalisé faible : l'« oracle » de normalisation connaît les
    réponses observées, ce qu'aucun modèle ne peut égaler. Les Qini se lisent donc en relatif."""
    df = sim.simulate(60_000, seed=8)
    df = df[df["treatment"] <= 1]
    y, t, tau = df["y"].to_numpy(), df["treatment"].to_numpy(), df["tau_1"].to_numpy()
    noise = np.random.default_rng(1).random(len(df))
    q_true, q_random = m.qini(y, tau, t), m.qini(y, noise, t)
    assert q_true > 0.04
    assert abs(q_random) < 0.02
    deciles = m.uplift_by_decile(y, tau, t, bins=5)
    assert deciles[0] > deciles[2] > deciles[-1]


def test_per_treatment_evaluates_each_action_against_control():
    df = sim.simulate(9_000, seed=4)
    tau = df[["tau_1", "tau_2"]].to_numpy()
    res = m.per_treatment(m.qini, df["y"], tau, df["treatment"])
    assert set(res) == {1, 2}
    with pytest.raises(ValueError, match="colonne"):
        m.per_treatment(m.qini, df["y"], tau[:, :1], df["treatment"])


def test_input_validation():
    with pytest.raises(ValueError, match="binaire"):
        m.qini([0, 1], [0.1, 0.2], [0, 2])
    with pytest.raises(ValueError, match="traités ET"):
        m.qini([0, 1], [0.1, 0.2], [1, 1])
    with pytest.raises(ValueError, match="longueur"):
        m.qini([0, 1, 1], [0.1, 0.2], [0, 1, 1])
    with pytest.raises(ValueError, match="k doit"):
        m.uplift_at_k([0, 1], [0.1, 0.2], [0, 1], k=1.0)
