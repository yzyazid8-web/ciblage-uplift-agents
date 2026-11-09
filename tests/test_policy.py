import numpy as np
import pytest

from ciblage.data import simulate as sim
from ciblage.uplift import calibration, policy

PROP = {0: 1 / 3, 1: 1 / 3, 2: 1 / 3}


def _true_value(df, pi):
    """Valeur VRAIE d'une politique : connue dans le simulateur (réponses potentielles)."""
    pot = df[["y0", "y1", "y2"]].to_numpy()
    return pot[np.arange(len(df)), pi].mean()


@pytest.fixture(scope="module")
def trial():
    return sim.simulate(90_000, seed=41)


def test_ipw_recovers_true_policy_value(trial):
    tau = trial[["tau_1", "tau_2"]].to_numpy()
    pi = policy.best_action_policy(tau, [1, 2], frac=0.4)
    for normalized in (True, False):
        est = policy.ipw_value(trial["y"], trial["treatment"], pi, PROP, normalized=normalized)
        assert est == pytest.approx(_true_value(trial, pi), abs=0.01)


def test_ipw_of_treat_nobody_is_control_mean(trial):
    pi = np.zeros(len(trial), dtype=int)
    est = policy.ipw_value(trial["y"], trial["treatment"], pi, PROP)
    assert est == pytest.approx(trial.loc[trial["treatment"] == 0, "y"].mean())


def test_compare_policies_ci_contains_true_gain(trial):
    tau = trial[["tau_1", "tau_2"]].to_numpy()
    smart = policy.best_action_policy(tau, [1, 2], frac=0.3)
    blind = policy.top_score_policy(np.random.default_rng(0).random(len(trial)), 1, 0.3)
    res = policy.compare_policies(
        trial["y"],
        trial["treatment"],
        {"uplift": smart, "hasard": blind},
        PROP,
        baseline="hasard",
        n_boot=300,
    )
    true_gain = _true_value(trial, smart) - _true_value(trial, blind)
    assert res["uplift"]["gain_low"] <= true_gain <= res["uplift"]["gain_high"]
    assert res["uplift"]["gain_low"] > 0  # cibler selon le vrai uplift bat le hasard
    assert res["hasard"]["gain"] == 0.0
    assert res["uplift"]["share_treated"] == pytest.approx(0.3, abs=0.01)


def test_best_action_policy_never_treats_negative_uplift():
    uplift = np.array([[0.2, 0.1], [-0.1, -0.3], [0.05, 0.3]])
    np.testing.assert_array_equal(policy.best_action_policy(uplift, [1, 2]), [1, 0, 2])
    np.testing.assert_array_equal(policy.best_action_policy(uplift, [1, 2], frac=0.34), [0, 0, 2])


def test_policy_input_validation():
    with pytest.raises(ValueError, match="propension"):
        policy.ipw_value([1, 0], [0, 1], [3, 0], PROP)
    with pytest.raises(ValueError, match="longueur"):
        policy.ipw_value([1, 0], [0, 1], [0], PROP)
    with pytest.raises(ValueError, match="référence"):
        policy.compare_policies([1, 0], [0, 1], {"a": np.array([0, 0])}, PROP, baseline="b")


def test_calibration_of_true_uplift_is_good(trial):
    two = trial[trial["treatment"] <= 1]
    table = calibration.calibration_table(two["y"], two["tau_1"], two["treatment"], bins=5)
    assert len(table) == 5
    assert calibration.calibration_error(table) < 0.02
    covered = ((table["ci_low"] <= table["predit"]) & (table["predit"] <= table["ci_high"])).mean()
    assert covered >= 0.8


def test_miscalibrated_prediction_is_detected(trial):
    two = trial[trial["treatment"] <= 1]
    table = calibration.calibration_table(two["y"], 3 * two["tau_1"], two["treatment"], bins=5)
    assert calibration.calibration_error(table) > 0.1
