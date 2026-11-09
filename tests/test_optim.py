import numpy as np
import pytest

from ciblage.optim import exact, greedy, lagrange
from ciblage.optim.problem import Problem, evaluate, from_uplift, violations


def _instance(
    n: int, seed: int, budget: float, capacity=None, allowed_share: float = 1.0
) -> Problem:
    rng = np.random.default_rng(seed)
    score = np.clip(rng.normal(0.02, 0.05, (n, 2)), -0.1, None)
    allowed = rng.random((n, 2)) < allowed_share
    return Problem(
        score=score, cost=[0.12, 0.10], budget=budget, capacity=capacity, allowed=allowed
    )


SOLVERS = {"glouton": greedy.solve_greedy, "lagrange": lagrange.solve_lagrange}
CASES = {
    "budget seul": dict(budget=8.0),
    "budget + capacité": dict(budget=8.0, capacity=[np.inf, 25]),
    "capacités seules": dict(budget=np.inf, capacity=[30, 20]),
    "exclusions": dict(budget=8.0, allowed_share=0.6),
}


@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("solver", list(SOLVERS))
def test_heuristics_are_feasible(case, solver):
    p = _instance(400, seed=1, **CASES[case])
    res = SOLVERS[solver](p)
    assert violations(p, res.assignment) == []


@pytest.mark.parametrize("case", list(CASES))
def test_lagrange_is_near_optimal_and_its_bound_is_valid(case):
    for seed in range(3):
        p = _instance(400, seed=seed, **CASES[case])
        opt = exact.solve_milp(p, time_limit=30)
        lag = lagrange.solve_lagrange(p)
        assert opt.status == "optimal"
        assert lag.bound >= opt.objective - 1e-6  # la borne majore bien l'optimum
        assert lag.objective >= 0.98 * opt.objective
        assert lag.objective <= opt.objective + 1e-9


def test_lagrange_bound_equals_lp_bound_with_budget_only():
    """Dualité forte : à prix optimal, L égale l'optimum de la relaxation linéaire."""
    p = _instance(5_000, seed=4, budget=60.0)
    lp = exact.solve_lp(p)
    lag = lagrange.solve_lagrange(p)
    assert lag.bound == pytest.approx(lp.bound, rel=1e-4)
    assert lag.gap < 0.001


def test_lp_relaxation_is_almost_integral_and_feasible():
    p = _instance(3_000, seed=5, budget=30.0, capacity=[np.inf, 100])
    lp = exact.solve_lp(p)
    assert lp.n_fractional <= 2  # au plus autant que de contraintes communes
    assert violations(p, lp.assignment) == []
    assert lp.objective <= lp.bound + 1e-9


def test_unconstrained_problem_gives_each_client_its_best_positive_action():
    score = np.array([[0.3, 0.1], [-0.2, 0.4], [-0.1, -0.3]])
    p = Problem(score=score, cost=[1.0, 1.0])
    for solve in SOLVERS.values():
        np.testing.assert_array_equal(solve(p).assignment, [1, 2, 0])


def test_from_uplift_profit_never_selects_unprofitable_actions():
    uplift = np.array([[0.05, 0.01], [0.002, 0.0]])
    p = from_uplift(uplift, cost=[0.1, 0.1], objective="profit", value_per_unit=10.0)
    # client 1 : 10*0.05-0.1 = 0.4 (rentable) ; client 2 : 10*0.002-0.1 < 0 (pas rentable)
    np.testing.assert_allclose(p.score[:, 0], [0.4, -0.08])
    assert greedy.solve_greedy(p).assignment.tolist() == [1, 0]


def test_evaluate_and_violations_report_clearly():
    p = Problem(
        score=np.ones((4, 2)),
        cost=[2.0, 1.0],
        budget=3.0,
        capacity=[1, 5],
        action_names=["homme", "femme"],
    )
    a = np.array([1, 1, 2, 0])
    ev = evaluate(p, a)
    assert ev["spend"] == 5.0 and ev["n_homme"] == 2 and ev["n_contacted"] == 3
    issues = violations(p, a)
    assert any("Budget dépassé" in s for s in issues)
    assert any("Capacité dépassée pour homme" in s for s in issues)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(score=np.ones(3), cost=[1.0]), "matrice"),
        (dict(score=np.ones((3, 2)), cost=[1.0]), "coûts"),
        (dict(score=np.ones((3, 2)), cost=[1.0, -1.0]), "positifs"),
        (dict(score=np.ones((3, 2)), cost=[1.0, 1.0], budget=-1), "budget"),
        (dict(score=np.ones((3, 2)), cost=[1.0, 1.0], allowed=np.ones((2, 2))), "forme"),
    ],
)
def test_problem_validation(kwargs, message):
    with pytest.raises(ValueError, match=message):
        Problem(**kwargs)
