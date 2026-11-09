"""Tests du graphe de l'agent avec un FAUX LLM : déterministes, sans clé ni réseau.

Le faux « compréhenseur » associe des messages connus à une compréhension fixée ; le faux
« rédacteur » produit un texte à partir du dossier de faits (fidèle) ou invente des chiffres.
"""

import json

import pytest

from ciblage.agents.copilot import Copilot
from ciblage.agents.graph import MAX_CLARIFY, REFUSAL, Deps, merge
from ciblage.agents.schema import Understanding
from ciblage.serve.planner import Planner

U = {
    "plan complet": Understanding(intent="plan", budget=600, cost_homme=0.12, cost_femme=0.10),
    "plan sans coût femme": Understanding(intent="plan", budget=600, cost_homme=0.12),
    "0,10 €": Understanding(intent="plan", cost_femme=0.10),
    "moitié": Understanding(intent="modifier", budget_factor=0.5),
    "100 de plus": Understanding(intent="modifier", budget_delta=100),
    "météo": Understanding(intent="hors_sujet"),
    "compare": Understanding(intent="comparer"),
    "pourquoi": Understanding(intent="expliquer"),
    "conseil +100": Understanding(intent="conseil", advice_topic="budget", budget_delta=100),
    "conseil courbe": Understanding(intent="conseil", advice_topic="budget"),
    "conseil femme": Understanding(intent="conseil", advice_topic="cout_femme"),
    "flou": Understanding(
        intent="plan", budget=600, ambiguities=["Le coût de 10 € est-il pour 100 e-mails ?"]
    ),
}


def fake_understand(messages):
    return U[messages[-1][1]]


def faithful_writer(messages):
    facts = json.loads(messages[1][1].split("\n", 1)[1])
    if "plan" not in facts:  # conseil, comparaison : réponse sans chiffre (toujours fidèle)
        return "Analyse effectuée."
    p = facts["plan"]
    return f"Envoie {p['email_homme']} e-mails homme et {p['email_femme']} e-mails femme pour {p['depense_eur']} €."


def lying_writer(messages):
    return "Envoie 123 456 e-mails pour 9 999 € : +42 % de visites garanties."


@pytest.fixture(scope="module")
def planner():
    return Planner()


def copilot(planner, writer=faithful_writer, db_path=None):
    return Copilot(Deps(planner=planner, understand=fake_understand, write=writer), db_path)


def test_complete_request_gives_verified_answer(planner):
    c = copilot(planner)
    r = c.send("t1", "plan complet")
    assert r.kind == "reponse"
    assert r.scenario["budget"] == 600 and r.scenario["cost_femme"] == 0.10
    assert r.result["summary"]["depense"] <= 600
    assert "vérification : conforme" in r.trace
    assert any("chiffres vérifiés" in step for step in r.trace)
    assert r.llm_calls == 2  # compréhension + explication


def test_missing_information_triggers_clarification_then_resumes(planner):
    c = copilot(planner)
    r1 = c.send("t2", "plan sans coût femme")
    assert r1.kind == "question" and "coût d'un e-mail femme" in r1.text
    assert c.waiting_for_answer("t2")
    r2 = c.send("t2", "0,10 €")  # la réponse reprend le graphe là où il s'était arrêté
    assert r2.kind == "reponse"
    assert r2.scenario == {**r2.scenario, "budget": 600, "cost_homme": 0.12, "cost_femme": 0.10}
    assert not c.waiting_for_answer("t2")


def test_ambiguity_reported_by_llm_is_asked(planner):
    r = copilot(planner).send("t3", "flou")
    assert r.kind == "question" and "pour 100 e-mails" in r.text


def test_what_if_modifies_only_the_budget_and_compares(planner):
    c = copilot(planner)
    c.send("t4", "plan complet")
    r = c.send("t4", "moitié")
    assert r.kind == "reponse"
    assert r.scenario["budget"] == 300 and r.scenario["cost_homme"] == 0.12
    assert r.result["summary"]["depense"] <= 300
    assert len(c.state("t4")["history"]) == 2


def test_what_if_without_reference_budget_asks(planner):
    r = copilot(planner).send("t5", "moitié")
    assert r.kind == "question" and "budget de référence" in r.text


def test_off_topic_is_refused_with_a_single_llm_call(planner):
    r = copilot(planner).send("t6", "météo")
    assert r.text == REFUSAL and r.llm_calls == 1


def test_not_yet_supported_intent_is_explained(planner):
    r = copilot(planner).send("t7", "pourquoi")
    assert "bientôt" in r.text


def test_compare_needs_two_scenarios(planner):
    r = copilot(planner).send("t7b", "compare")
    assert "au moins deux scénarios" in r.text


def test_invented_numbers_fall_back_to_safe_template(planner):
    r = copilot(planner, writer=lying_writer).send("t8", "plan complet")
    assert "123" not in r.text and "Plan recommandé" in r.text
    assert any("repli sûr" in step for step in r.trace)
    assert r.llm_calls == 3  # compréhension + 2 rédactions refusées


def test_clarification_loop_is_bounded(planner):
    c = copilot(planner)
    r = c.send("t9", "flou")
    for _ in range(MAX_CLARIFY):
        assert r.kind == "question"
        r = c.send("t9", "flou")
    assert r.kind == "reponse" and "reformuler" in r.text


def test_sessions_survive_a_restart(planner, tmp_path):
    db = tmp_path / "sessions.db"
    copilot(planner, db_path=db).send("t10", "plan complet")
    r = copilot(planner, db_path=db).send("t10", "100 de plus")  # nouvelle instance, même base
    assert r.scenario["budget"] == 700


def test_merge_applies_relative_and_absolute_changes():
    base = {"budget": 400.0, "cost_homme": 0.12, "max_femme": 50}
    draft, issues = merge(
        base,
        Understanding(intent="modifier", budget_factor=1.5, budget_delta=-100, remove_limits=True),
    )
    assert draft["budget"] == 500.0 and draft["max_femme"] is None and issues == []


def test_confidence_level_is_part_of_the_facts(planner):
    """« IC à 95 % » ne doit pas être pris pour un chiffre inventé (faux positif vu en test réel)."""
    from ciblage.agents import faithfulness
    from ciblage.agents.graph import build_facts
    from ciblage.serve.planner import Scenario

    facts = build_facts(planner.plan(Scenario(budget=300.0), n_boot=30).to_dict())
    assert faithfulness.check("Intervalle de confiance à 95 %.", facts).ok


def test_advice_on_extra_budget_does_not_change_the_scenario(planner):
    c = copilot(planner)
    c.send("t11", "plan complet")
    r = c.send("t11", "conseil +100")
    assert r.kind == "reponse"
    assert r.scenario["budget"] == 600  # la question « +100 € » ne modifie pas le plan
    facts = c.state("t11")["facts"]
    assert (
        facts["budget_envisage"] == 700 and facts["visites_envisagees"] > facts["visites_actuelles"]
    )
    assert c.state("t11")["facts_kind"] == "conseil_budget_delta"
    assert len(c.state("t11")["history"]) == 1  # aucun nouveau plan enregistré


def test_advice_on_cost_and_budget_curve(planner):
    c = copilot(planner)
    c.send("t12", "plan complet")
    c.send("t12", "conseil femme")
    facts = c.state("t12")["facts"]
    assert facts["seuil_cout_au_dela_duquel_plus_aucun_envoi"] > 0.10
    c.send("t12", "conseil courbe")
    assert c.state("t12")["facts_kind"] == "conseil_budget_courbe"
    assert c.state("t12")["facts"]["courbe"][-1]["visites_par_euro_supplementaire"] == 0.0


def test_advice_without_scenario_first_collects_it(planner):
    c = copilot(planner)
    r = c.send("t13", "conseil +100")
    assert r.kind == "question" and "budget total" in r.text
    r = c.send("t13", "plan complet")  # la réponse fournit le scénario de référence
    assert r.kind == "reponse" and c.state("t13")["facts"]["budget_envisage"] == 700


def test_compare_two_computed_scenarios(planner):
    c = copilot(planner)
    c.send("t14", "plan complet")
    c.send("t14", "moitié")
    r = c.send("t14", "compare")
    assert r.kind == "reponse"
    rows = c.state("t14")["facts"]["scenarios"]
    assert [row["budget"] for row in rows] == [600, 300]


def test_fallback_templates_cover_every_answer_type(planner):
    from ciblage.agents.graph import fallback
    from ciblage.serve import advisor
    from ciblage.serve.planner import Scenario

    sc = Scenario(budget=400.0)
    cases = {
        "conseil_budget_delta": advisor.marginal_value(planner, sc, 50.0),
        "conseil_budget_courbe": advisor.budget_curve(planner, sc, points=4),
        "conseil_cout": advisor.cost_threshold(planner, sc, "femme"),
    }
    from ciblage.agents import faithfulness

    for kind, facts in cases.items():
        text = fallback(kind, facts)
        assert faithfulness.check(text, facts).ok, (kind, text)
