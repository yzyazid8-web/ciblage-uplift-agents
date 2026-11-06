import numpy as np
import pytest

from ciblage.data import simulate as sim


@pytest.fixture(scope="module")
def big():
    return sim.simulate(200_000, seed=7)


def test_observed_outcome_is_potential_outcome_of_assigned_treatment(big):
    pot = big[["y0", "y1", "y2"]].to_numpy()
    assert (big["y"].to_numpy() == pot[np.arange(len(big)), big["treatment"].to_numpy()]).all()


def test_potential_outcomes_follow_the_four_types(big):
    by = {t: big[big["type"] == t] for t in sim.TYPES}
    assert (by["fidele"][["y0", "y1", "y2"]] == 1).all().all()
    assert (by["perdu"][["y0", "y1", "y2"]] == 0).all().all()
    assert (by["persuadable"]["y0"] == 0).all()
    sleeping = by["ne_pas_deranger"]
    assert (sleeping["y0"] == 1).all() and (sleeping[["y1", "y2"]] == 0).all().all()


def test_all_four_types_are_present(big):
    shares = big["type"].value_counts(normalize=True)
    assert all(shares[t] > 0.03 for t in sim.TYPES)


def test_randomized_difference_in_means_recovers_true_average_uplift(big):
    """Grâce au tirage au sort, la différence de moyennes estime sans biais l'uplift moyen vrai."""
    control = big.loc[big["treatment"] == 0, "y"].mean()
    for a in (1, 2):
        observed = big.loc[big["treatment"] == a, "y"].mean() - control
        assert observed == pytest.approx(big[f"tau_{a}"].mean(), abs=0.01)


def test_true_uplift_matches_empirical_potential_outcomes(big):
    """L'uplift « formule » coïncide avec la moyenne des différences de réponses potentielles."""
    for a in (1, 2):
        empirical = (big[f"y{a}"] - big["y0"]).mean()
        assert empirical == pytest.approx(big[f"tau_{a}"].mean(), abs=0.005)
    assert big["y0"].mean() == pytest.approx(big["mu_0"].mean(), abs=0.005)


def test_uplift_is_heterogeneous_with_sign_changes(big):
    """Il existe des clients à cibler (uplift > 0) et d'autres à éviter (uplift < 0)."""
    for a in (1, 2):
        tau = big[f"tau_{a}"]
        assert (tau > 0.2).mean() > 0.05
        assert (tau < 0).mean() > 0.05


def test_each_email_works_best_on_its_audience(big):
    assert big.loc[big["mens"] == 1, "tau_1"].mean() > big.loc[big["mens"] == 0, "tau_1"].mean()
    assert big.loc[big["womens"] == 1, "tau_2"].mean() > big.loc[big["womens"] == 0, "tau_2"].mean()


def test_noise_features_do_not_enter_true_uplift():
    df = sim.simulate(1_000, seed=1)
    shifted = df[sim.FEATURES].copy()
    shifted[["bruit_1", "bruit_2"]] += 100.0
    np.testing.assert_allclose(sim.true_uplift(df[sim.FEATURES]), sim.true_uplift(shifted))


def test_simulation_is_reproducible():
    a = sim.simulate(500, seed=3)
    b = sim.simulate(500, seed=3)
    assert a.equals(b)


def test_custom_propensity_is_respected():
    df = sim.simulate(60_000, propensity=(0.15, 0.85, 0.0), seed=2)
    shares = df["treatment"].value_counts(normalize=True)
    assert shares[0] == pytest.approx(0.15, abs=0.01)
    assert 2 not in shares.index


@pytest.mark.parametrize("prop", [(0.5, 0.5), (0.6, 0.6, -0.2), (0.2, 0.2, 0.2)])
def test_invalid_propensity_is_rejected(prop):
    with pytest.raises(ValueError, match="propensity"):
        sim.simulate(10, propensity=prop)
