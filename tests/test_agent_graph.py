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
    "pourquoi client 5": Understanding(intent="expliquer", client_id=5),
    "pourquoi client inconnu": Understanding(intent="expliquer", client_id=999_999),
    "pourquoi homme seulement": Understanding(intent="expliquer", segment="homme seulement"),
    "qui est écarté": Understanding(intent="expliquer", about_excluded=True),
    "portefeuille": Understanding(intent="donnees"),
    "portefeuille deux rayons": Understanding(intent="donnees", segment="homme et femme"),
    "exporte": Understanding(intent="exporter"),
    "conseil +100": Understanding(intent="conseil", advice_topic="budget", budget_delta=100),
    "conseil courbe": Understanding(intent="conseil", advice_topic="budget"),
    "conseil femme": Understanding(intent="conseil", advice_topic="cout_femme"),
    "Interprète le graphique 2": Understanding(intent="expliquer", element="graphique_segments"),
    "Explique les indicateurs": Understanding(intent="expliquer", element="indicateurs"),
    "Que montre le graphique 1 ?": Understanding(intent="expliquer"),  # élément non extrait
    "Que fait ce marimo ?": Understanding(intent="hors_sujet"),  # erreur vue en test réel
    "C'est quoi l'uplift ?": Understanding(intent="application"),
    "Explique-moi tous les résultats": Understanding(
        intent="expliquer", ambiguities=["Quel élément de l'écran faut-il interpréter ?"]
    ),
    "Compare 500, 700 et 1 000 € : lequel est le plus rentable ?": Understanding(
        intent="plan", budget=500
    ),  # le LLM n'a retenu qu'un budget : la règle des budgets multiples corrige
    "analyse libre": Understanding(intent="analyse"),
    "Avec 1 000 €, limite à 3 000 envois, l'e-mail homme 0,12 € et l'e-mail femme 0,10 €.": (
        Understanding(intent="plan", budget=1000, cost_homme=0.12, cost_femme=0.10)
    ),  # vu en test réel (audit I4) : la limite était ignorée sans question
    "l'e-mail homme": Understanding(intent="plan"),
    "aucune": Understanding(intent="plan"),
    "flou": Understanding(
        intent="plan", budget=600, ambiguities=["Le coût de 10 € est-il pour 100 e-mails ?"]
    ),
}


def fake_understand(messages):
    return U[messages[-1][1]]


def faithful_writer(messages):
    dossier = next(c for _, c in messages if c.startswith("DOSSIER DE FAITS"))
    facts = json.loads(dossier.split("\n", 1)[1])
    if "plan" not in facts:  # conseil, comparaison : réponse sans chiffre (toujours fidèle)
        return "Analyse effectuée."
    p = facts["plan"]
    return f"Envoie {p['email_homme']} e-mails homme et {p['email_femme']} e-mails femme pour {p['depense_eur']} €."


def lying_writer(messages):
    return "Envoie 123 456 e-mails pour 9 999 € : +42 % de visites garanties."


@pytest.fixture(scope="module")
def planner():
    return Planner()


def copilot(planner, writer=faithful_writer, db_path=None, exports_dir=None, analyze=None):
    deps = Deps(planner=planner, understand=fake_understand, write=writer, analyze=analyze)
    if exports_dir is not None:
        deps.exports_dir = exports_dir
    return Copilot(deps, db_path)


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


def test_explain_a_client_uses_the_profitability_threshold(planner):
    c = copilot(planner)
    c.send("t20", "plan complet")
    r = c.send("t20", "pourquoi client 5")
    assert r.kind == "reponse"
    facts = c.state("t20")["facts"]
    assert facts["client"] == 5 and "seuil_de_rentabilite" in facts and facts["raison"]
    assert c.state("t20")["facts_kind"] == "explication_client"


def test_explain_invalid_client_is_answered_clearly(planner):
    c = copilot(planner)
    c.send("t21", "plan complet")
    r = c.send("t21", "pourquoi client inconnu")
    assert "numéro de client" in r.text and r.llm_calls == 1


def test_explain_segment_excluded_and_all_segments(planner):
    c = copilot(planner)
    c.send("t22", "plan complet")
    c.send("t22", "pourquoi homme seulement")
    assert c.state("t22")["facts"]["segment"] == "homme seulement"
    c.send("t22", "qui est écarté")
    assert c.state("t22")["facts_kind"] == "explication_ecartes"
    c.send("t22", "pourquoi")  # sans précision : tout l'écran
    assert c.state("t22")["facts_kind"] == "explication_resultats"
    c.send("t22", "Interprète le graphique 2")
    assert c.state("t22")["facts_kind"] == "explication_segments"
    assert len(c.state("t22")["facts"]["segments"]) == 3


def test_every_displayed_result_can_be_explained(planner):
    c = copilot(planner)
    c.send("t30", "plan complet")
    c.send("t30", "Explique les indicateurs")
    assert c.state("t30")["facts_kind"] == "explication_indicateurs"
    assert len(c.state("t30")["facts"]["indicateurs"]) == 5
    # le LLM n'a pas extrait l'élément : la règle déterministe le repère dans la question
    r = c.send("t30", "Que montre le graphique 1 ?")
    assert c.state("t30")["facts_kind"] == "explication_politiques"
    assert any("élément de l'écran repéré" in step for step in r.trace)


def test_questions_about_the_app_are_answered_not_refused(planner):
    """Vu en test réel : « Que fait ce marimo ? » était refusé comme hors sujet."""
    c = copilot(planner)
    r = c.send("t31", "Que fait ce marimo ?")
    assert r.kind == "reponse" and r.text != REFUSAL
    assert c.state("t31")["facts_kind"] == "application"
    assert any("application" in step for step in r.trace)
    c.send("t31", "C'est quoi l'uplift ?")  # aucun scénario nécessaire
    assert c.state("t31")["facts_kind"] == "application"
    assert c.send("t31", "météo").text == REFUSAL  # le vrai hors sujet reste refusé


def test_empty_answer_is_rejected_then_replaced_by_the_safe_text(planner):
    """Vu en test réel : un texte vide (aucun chiffre) passait le contrôle de fidélité."""
    r = copilot(planner, writer=lambda messages: "  ").send("t34", "plan complet")
    assert r.text.startswith("Plan recommandé") and r.llm_calls == 3
    assert r.trace.count("rédaction : réponse vide") == 2


def test_explanation_requests_never_ask_a_useless_question(planner):
    """Vu en test réel : « Explique-moi tous les résultats » renvoyait « quel élément ? »."""
    c = copilot(planner)
    c.send("t33", "plan complet")
    r = c.send("t33", "Explique-moi tous les résultats")
    assert r.kind == "reponse" and c.state("t33")["facts_kind"] == "explication_resultats"


def test_screen_question_without_a_plan_falls_back_to_the_guide(planner):
    c = copilot(planner)
    c.send("t32", "Que montre le graphique 1 ?")
    assert c.state("t32")["facts_kind"] == "application"


def test_data_questions_need_no_scenario(planner):
    c = copilot(planner)
    r = c.send("t23", "portefeuille")
    assert (
        r.kind == "reponse" and c.state("t23")["facts"]["clients_du_perimetre"] == planner.n_clients
    )
    c.send("t23", "portefeuille deux rayons")
    assert c.state("t23")["facts"]["perimetre"] == "homme et femme"


def test_export_requires_explicit_confirmation(planner, tmp_path):
    c = copilot(planner, exports_dir=tmp_path)
    c.send("t24", "plan complet")
    r = c.send("t24", "exporte")
    assert r.kind == "question" and "Confirmes-tu" in r.text
    assert not list(tmp_path.iterdir())  # rien n'est écrit avant la confirmation
    r = c.send("t24", "non merci")
    assert "annulé" in r.text and not list(tmp_path.iterdir())
    c.send("t24", "exporte")
    r = c.send("t24", "oui")
    assert "Fichier de campagne écrit" in r.text
    assert len(list(tmp_path.glob("campagne_*.csv"))) == 1


def test_too_long_message_is_rejected_without_calling_the_llm(planner):
    r = copilot(planner).send("t25", "x" * 5_000)
    assert "raccourcir" in r.text and r.llm_calls == 0


def test_generic_fallback_is_faithful(planner):
    from ciblage.agents import faithfulness
    from ciblage.agents.graph import fallback
    from ciblage.serve import explainer, guide
    from ciblage.serve.planner import Scenario

    sc = Scenario(budget=500.0)
    for kind, facts in {
        "explication_client": explainer.client_explanation(planner, sc, 3),
        "explication_ecartes": explainer.excluded_profile(planner, sc),
        "explication_segments": explainer.segments_explanation(planner, sc),
        "application": guide.app_guide(planner),
        "donnees": explainer.portfolio_overview(planner),
    }.items():
        text = fallback(kind, facts)
        assert faithfulness.check(text, facts).ok, (kind, text)


def test_off_topic_with_campaign_parameters_is_corrected(planner):
    """Vu en test réel : le LLM classe parfois « hors sujet » un message qui donne budget et coûts."""
    U["incohérent"] = Understanding(
        intent="hors_sujet", budget=400, cost_homme=0.12, cost_femme=0.10
    )
    r = copilot(planner).send("t26", "incohérent")
    assert r.kind == "reponse" and r.scenario["budget"] == 400
    assert any("corrigé" in step for step in r.trace)


def test_question_about_excluded_clients_is_routed_to_the_plan(planner):
    """Vu en test réel : « Qui sont les clients écartés ? » était compris comme une question de données."""
    U["Qui sont les clients écartés ?"] = Understanding(intent="donnees")
    c = copilot(planner)
    c.send("t27", "plan complet")
    c.send("t27", "Qui sont les clients écartés ?")
    assert c.state("t27")["facts_kind"] == "explication_ecartes"


def test_stream_emits_steps_then_final_answer(planner):
    c = copilot(planner)
    events = list(c.stream("t28", "plan complet"))
    steps = [e["texte"] for e in events if e["type"] == "etape"]
    assert "calcul du plan" in steps and "vérification : conforme" in steps
    assert events[-1]["type"] == "reponse" and events[-1]["scenario"]["budget"] == 600
    question = list(c.stream("t29", "plan sans coût femme"))[-1]
    assert question["type"] == "question" and "coût d'un e-mail femme" in question["texte"]
    answer = list(c.stream("t29", "0,10 €"))[-1]  # reprise du graphe interrompu
    assert answer["type"] == "reponse"


def test_dashboard_scenario_is_synced_into_the_agent(planner):
    """Le scénario affiché (formulaire) est transmis à l'agent : « la moitié » s'applique dessus."""
    c = copilot(planner)
    shown = {
        "budget": 800.0,
        "cost_homme": 0.12,
        "cost_femme": 0.10,
        "max_homme": None,
        "max_femme": None,
        "objective": "visites",
        "value_per_visit": 1.0,
    }
    assert c.sync_scenario("t30", shown)
    assert not c.sync_scenario("t30", shown)  # déjà aligné : rien à faire
    r = c.send("t30", "moitié")
    assert r.kind == "reponse" and r.scenario["budget"] == 400


def test_sync_never_breaks_a_pending_question(planner):
    c = copilot(planner)
    c.send("t31", "plan sans coût femme")
    assert not c.sync_scenario("t31", {"budget": 50.0})
    assert c.send("t31", "0,10 €").scenario["budget"] == 600


def test_llm_failure_does_not_crash_the_turn(planner):
    """Vu en évaluation : une réponse invalide du LLM (booléen à null) faisait planter le tour."""

    def broken(messages):
        raise ValueError("Tool call validation failed")

    c = Copilot(Deps(planner=planner, understand=broken, write=faithful_writer))
    r = c.send("t32", "plan complet")
    assert r.kind == "reponse" and "reformuler" in r.text
    assert any("échec" in step for step in r.trace)


def test_implausible_cost_triggers_a_question(planner):
    """Vu en évaluation : « l'e-mail homme à 12 » (sans unité) était lu comme 12 € par e-mail."""
    U["sans unité"] = Understanding(intent="plan", budget=400, cost_homme=12, cost_femme=10)
    r = copilot(planner).send("t33", "sans unité")
    assert r.kind == "question" and "centimes" in r.text


def test_null_booleans_are_accepted():
    u = Understanding.model_validate(
        {"intent": "expliquer", "about_excluded": None, "remove_limits": None}
    )
    draft, issues = merge({"budget": 100.0, "max_femme": 10}, u)
    assert draft["max_femme"] == 10 and issues == []  # null = « non demandé »


# agent analyste (boucle LLM <-> outils) avec un faux LLM à outils
def scripted_analyst(messages):
    """1re décision : calculer trois plans d'un coup ; 2e : répondre avec les chiffres reçus."""
    tool_results = [m for m in messages if m["role"] == "tool"]
    if not tool_results:
        return {
            "content": "",
            "tool_calls": [
                {"id": "a1", "name": "calculer_plans", "args": {"budgets": [500, 700, 1000]}}
            ],
        }
    plans = json.loads(tool_results[-1]["content"])["plans"]
    text = " ; ".join(
        f"{p['budget']:.0f} € : {p['visites_en_plus_predites']} visites" for p in plans
    )
    return {"content": f"Visites en plus prédites : {text}.", "tool_calls": []}


def test_the_analyst_chooses_its_tools_then_answers_with_checked_numbers(planner):
    c = copilot(planner, analyze=scripted_analyst)
    c.send("t40", "plan complet")
    shown = c.state("t40")["scenario"]
    r = c.send("t40", "Compare 500, 700 et 1 000 € : lequel est le plus rentable ?")
    assert r.kind == "reponse" and r.text.startswith("Visites en plus prédites")
    assert any("budgets cités -> analyse" in step for step in r.trace)
    assert any("calculer_plans" in step for step in r.trace)
    assert "rédaction : chiffres vérifiés (essai 1)" in r.trace
    assert r.llm_calls == 3  # compréhension + 2 décisions de l'analyste ; aucune rédaction de plus
    state = c.state("t40")
    assert state["facts_kind"] == "analyse"
    assert len(state["facts"]["resultats_des_outils"]) == 1
    assert state["scenario"] == shown  # une analyse ne change pas le plan affiché


def test_an_analyst_answer_with_invented_numbers_is_rewritten(planner):
    def inventive(messages):
        if not any(m["role"] == "tool" for m in messages):
            return {
                "content": "",
                "tool_calls": [{"id": "x", "name": "calculer_plans", "args": {"budgets": [500]}}],
            }
        return {"content": "Le plan rapporte 9 999 visites.", "tool_calls": []}

    c = copilot(planner, analyze=inventive)
    c.send("t41", "plan complet")
    r = c.send("t41", "analyse libre")
    assert any("chiffres non vérifiés" in step for step in r.trace)
    assert "9 999" not in r.text


def test_the_analyst_is_stopped_after_the_step_limit(planner):
    from ciblage.agents.graph import MAX_ANALYST_STEPS

    def endless(messages):
        return {"content": "", "tool_calls": [{"id": "z", "name": "courbe_budget", "args": {}}]}

    c = copilot(planner, analyze=endless)
    c.send("t42", "plan complet")
    r = c.send("t42", "analyse libre")
    assert r.kind == "reponse"
    assert any("limite d'étapes atteinte" in step for step in r.trace)
    tool_steps = [step for step in r.trace if step.startswith("analyste : étape")]
    assert len(tool_steps) == MAX_ANALYST_STEPS - 1


def test_analyst_failure_does_not_break_the_conversation(planner):
    def broken(messages):
        raise RuntimeError("quota")

    c = copilot(planner, analyze=broken)
    c.send("t43", "plan complet")
    r = c.send("t43", "analyse libre")
    assert r.kind == "reponse" and any("analyste : échec" in step for step in r.trace)


def test_without_an_analyst_the_request_is_answered_clearly(planner):
    c = copilot(planner)  # pas d'agent à outils configuré
    r = c.send("t44", "analyse libre")
    assert r.kind == "reponse" and r.llm_calls == 1


def test_a_writer_claiming_an_undemonstrated_gain_is_corrected(planner):
    """Audit I1 : une injection avait fait écrire « c'est prouvé » malgré un gain non démontré."""

    def boastful(messages):
        return "Le plan optimisé bat nettement le ciblage classique, c'est prouvé."

    c = copilot(planner, writer=boastful)
    r = c.send("t45", "plan complet")
    if c.state("t45")["facts"]["mesure_sur_la_campagne"]["gain_demontre"]:
        pytest.skip("gain démontré sur ce scénario : rien à corriger")
    assert "rédaction : verdict contraire aux mesures" in r.trace
    assert "prouvé" not in r.text and r.text.startswith("Plan recommandé")


def test_writer_failure_falls_back_to_the_safe_text(planner):
    """Vu en test réel : quota du jour atteint pendant la rédaction -> le tour plantait."""

    def out_of_quota(messages):
        raise RuntimeError("Rate limit reached (tokens per day)")

    r = copilot(planner, writer=out_of_quota).send("t46", "plan complet")
    assert r.kind == "reponse" and r.text.startswith("Plan recommandé")
    assert "rédaction : échec (RuntimeError)" in r.trace


LIMIT_MSG = "Avec 1 000 €, limite à 3 000 envois, l'e-mail homme 0,12 € et l'e-mail femme 0,10 €."


def test_a_limit_without_its_email_triggers_a_question_then_is_applied(planner):
    c = copilot(planner)
    r = c.send("t50", LIMIT_MSG)
    assert r.kind == "question" and "3 000 envois" in r.text and "e-mail homme" in r.text
    r = c.send("t50", "l'e-mail homme")  # le LLM n'extrait rien : la règle lit la réponse
    assert r.kind == "reponse"
    assert r.scenario["max_homme"] == 3000 and r.scenario["max_femme"] is None
    assert r.result["summary"]["email_homme"] <= 3000


def test_a_limit_can_be_explicitly_ignored(planner):
    c = copilot(planner)
    c.send("t51", LIMIT_MSG)
    r = c.send("t51", "aucune")
    assert r.kind == "reponse" and r.scenario["max_homme"] is None


def test_export_needs_an_unambiguous_yes():
    from ciblage.agents.graph import is_yes

    assert is_yes("oui") and is_yes("Oui, vas-y !") and is_yes("je confirme")
    for reply in ("oui mais avec 500 € plutôt", "ok non attends", "yes please no", "non", ""):
        assert not is_yes(reply), reply


def test_demo_limits_stop_before_any_llm_call(planner):
    calls = []

    def counting_understand(messages):
        calls.append(1)
        return fake_understand(messages)

    deps = Deps(planner=planner, understand=counting_understand, write=faithful_writer)
    c = Copilot(deps, max_per_session=2, max_per_day=3)
    c.send("t52", "météo")
    c.send("t52", "météo")
    r = c.send("t52", "météo")  # 3e message de la session : refusé sans appel
    assert "limite de la démo" in r.text and len(calls) == 2
    c.send("t53", "météo")  # autre session : 3e message du jour, accepté
    r = c.send("t54", "météo")  # 4e message du jour : refusé
    assert "aujourd'hui" in r.text and len(calls) == 3
    events = list(c.stream("t55", "météo"))
    assert len(events) == 1 and "aujourd'hui" in events[0]["texte"]


def test_quota_errors_get_a_clear_message(planner):
    from ciblage.agents.graph import QUOTA_MESSAGE, is_quota_error

    class RateLimitError(Exception):
        pass

    def no_quota(messages):
        raise RateLimitError("Error code: 429 - Rate limit reached (tokens per day)")

    assert is_quota_error(RateLimitError("429")) and not is_quota_error(ValueError("bug"))
    deps = Deps(planner=planner, understand=no_quota, write=faithful_writer)
    r = Copilot(deps).send("t56", "plan complet")
    assert r.text == QUOTA_MESSAGE

    def analyst_no_quota(messages):
        raise RateLimitError("429")

    c = copilot(planner, analyze=analyst_no_quota)
    r = c.send("t57", "analyse libre")
    assert r.text == QUOTA_MESSAGE
