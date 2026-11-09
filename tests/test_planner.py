import numpy as np
import pytest

from ciblage.serve.planner import Planner, Scenario


@pytest.fixture(scope="module")
def planner():
    return Planner()


def test_plan_respects_budget_and_reports_clearly(planner):
    plan = planner.plan(Scenario(budget=500.0), n_boot=50)
    s = plan.summary
    assert s["depense"] <= 500.0 + 1e-9
    assert s["clients_contactes"] == s["email_homme"] + s["email_femme"]
    assert s["visites_en_plus_predites"] > 0
    assert plan.bound_gap < 0.01  # optimalité prouvée à 1 % près
    assert plan.by_segment["clients"].sum() == planner.n_clients
    assert set(plan.comparison["politique"]) == {
        "personne",
        "ciblage classique",
        "plan optimisé",
        "tout le monde (e-mail homme)",
    }


def test_capacity_is_respected(planner):
    plan = planner.plan(Scenario(budget=2_000.0, max_homme=1_000), compare=False)
    assert plan.summary["email_homme"] <= 1_000


def test_zero_budget_contacts_nobody(planner):
    plan = planner.plan(Scenario(budget=0.0), compare=False)
    assert plan.summary["clients_contactes"] == 0


def test_profit_objective_skips_unprofitable_emails(planner):
    """Si une visite ne rapporte presque rien, aucun e-mail n'est rentable."""
    cheap = planner.plan(
        Scenario(budget=10_000.0, objective="profit", value_per_visit=0.5), compare=False
    )
    assert cheap.summary["clients_contactes"] == 0
    rich = planner.plan(
        Scenario(budget=10_000.0, objective="profit", value_per_visit=20.0), compare=False
    )
    assert rich.summary["clients_contactes"] > 0
    assert rich.summary["profit_predit"] > 0


def test_plan_is_serializable(planner):
    d = planner.plan(Scenario(budget=300.0), n_boot=30).to_dict()
    assert d["scenario"]["budget"] == 300.0
    assert isinstance(d["by_segment"], list) and isinstance(d["comparison"], list)


@pytest.mark.parametrize(
    "scenario, message",
    [
        (Scenario(budget=-1.0), "budget"),
        (Scenario(budget=100.0, cost_homme=-0.1), "coûts"),
        (Scenario(budget=100.0, max_femme=-5), "capacité"),
        (Scenario(budget=100.0, objective="profit", value_per_visit=0.0), "valeur d'une visite"),
    ],
)
def test_invalid_scenarios_are_rejected_in_plain_language(planner, scenario, message):
    assert any(message in e for e in scenario.errors())
    with pytest.raises(ValueError):
        planner.plan(scenario)


def test_assignment_codes(planner):
    a = planner.plan(Scenario(budget=400.0), compare=False).assignment
    assert set(np.unique(a)) <= {0, 1, 2}
