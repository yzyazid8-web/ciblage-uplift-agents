"""Outils de l'agent analyste et contrôles de fidélité renforcés (constats I1 à I3 de l'audit)."""

import pytest

from ciblage.agents import faithfulness as F
from ciblage.agents import tools
from ciblage.serve.planner import Planner

BASE = {"budget": 700.0, "cost_homme": 0.12, "cost_femme": 0.10}


@pytest.fixture(scope="module")
def planner():
    return Planner()


def test_every_tool_has_a_function_calling_schema():
    names = {s["function"]["name"] for s in tools.schemas()}
    assert names == set(tools.TOOLS)
    for s in tools.schemas():
        assert s["type"] == "function" and s["function"]["description"]
        assert s["function"]["parameters"]["type"] == "object"


def test_plans_for_several_budgets_start_from_the_displayed_scenario(planner):
    out = tools.run_tool(planner, BASE, "calculer_plans", {"budgets": [300, 700]})
    assert [p["budget"] for p in out["plans"]] == [300, 700]
    assert out["couts_utilises"] == {"email_homme": 0.12, "email_femme": 0.10}
    low, high = out["plans"]
    assert low["depense_eur"] <= 300 and high["depense_eur"] <= 700
    assert low["visites_en_plus_predites"] < high["visites_en_plus_predites"]
    assert {"gain_demontre", "gain_ic95_bas", "par_segment"} <= set(high)


def test_tool_arguments_override_the_displayed_scenario(planner):
    out = tools.run_tool(planner, BASE, "calculer_plans", {"budgets": [500], "cout_femme": 0.5})
    assert out["couts_utilises"]["email_femme"] == 0.5


def test_tool_errors_are_returned_to_the_llm_not_raised(planner):
    assert "Outil inconnu" in tools.run_tool(planner, BASE, "supprimer_tout", {})["erreur"]
    bad = tools.run_tool(planner, BASE, "expliquer_client", {"client_id": "douze"})
    assert "Arguments invalides" in bad["erreur"]
    missing = tools.run_tool(planner, None, "calculer_plans", {"budgets": [500]})
    assert "Il manque" in missing["erreur"]
    absurd = tools.run_tool(planner, BASE, "calculer_plans", {"budgets": [500], "cout_homme": 12})
    assert "plausible" in absurd["erreur"]
    unknown_client = tools.run_tool(planner, BASE, "expliquer_client", {"client_id": 10**7})
    assert "numéro de client" in unknown_client["erreur"]


def test_other_tools_return_facts(planner):
    for name, args in {
        "effet_variation_budget": {"variation_eur": 100},
        "expliquer_segments": {},
        "expliquer_client": {"client_id": 12},
        "profil_des_ecartes": {},
        "comparer_politiques": {},
        "portrait_portefeuille": {"segment": "homme et femme"},
    }.items():
        out = tools.run_tool(planner, BASE, name, args)
        assert "erreur" not in out and out, name


# contrôles de fidélité (audit I1 à I3)
FACTS = {
    "clients": 5947,
    "depense": 699.92,
    "gain_vs_classique_points": -0.23,
    "gain_demontre": False,
}


@pytest.mark.parametrize(
    "text",
    ["contacte ~6200 clients", "rapporte €1500", "=4321 visites", "x15 visites", "«42 %»"],
)
def test_numbers_glued_to_symbols_are_checked(text):
    """Avant : [+-−] était une plage qui avalait ces symboles, et le nombre échappait au contrôle."""
    assert not F.check(text, FACTS).ok


def test_points_are_not_turned_into_percent_and_written_signs_must_match():
    assert not F.check("Le gain est de 23 %.", FACTS).ok  # -0,23 point n'est pas « 23 % »
    assert not F.check("Un gain de +0,23 point.", FACTS).ok  # signe faux
    assert F.check("Un écart de -0,23 point.", FACTS).ok
    assert F.check("Une baisse de 0,23 point.", FACTS).ok  # signe omis : toléré
    assert F.check("Taux de visite de 13,0 %.", {"taux": 0.1304}).ok  # vraie proportion


@pytest.mark.parametrize(
    "text",
    [
        "Ces résultats démontrent que le plan bat nettement le ciblage classique.",
        "Le plan optimisé fait mieux que le ciblage classique, c’est prouvé.",
    ],
)
def test_a_claimed_gain_is_rejected_when_it_is_not_demonstrated(text):
    report = F.check(text, FACTS)
    assert report.verdict and not report.ok


@pytest.mark.parametrize(
    "text",
    [
        "La différence avec le ciblage classique n’est pas démontrée pour le plan.",
        "Le plan optimisé n'est pas statistiquement supérieur au ciblage classique.",
        "Le plan envoie l’e-mail aux clients dont le rendement est supérieur au seuil.",
        "La vérification démontre que l’uplift mesuré de l’e-mail homme est positif.",
        "Tout le monde avec l’e-mail homme bat nettement le ciblage classique.",
    ],
)
def test_legitimate_sentences_pass_the_verdict_check(text):
    assert F.check(text, FACTS).verdict == []


def test_verdict_check_applies_only_when_the_gain_is_not_demonstrated():
    shown = {**FACTS, "gain_demontre": True, "gain_vs_classique_points": 1.2}
    assert F.check("Le plan bat le ciblage classique, c'est démontré.", shown).ok
    policies = {"politiques": [{"politique": "plan optimisé", "difference_demontree": False}]}
    assert F.plan_gain_shown(policies) is False
    assert F.plan_gain_shown({"autre": 1}) is None
