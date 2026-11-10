import numpy as np
import pytest

from ciblage.serve import explainer
from ciblage.serve.planner import Planner, Scenario

SC = Scenario(budget=700.0)


@pytest.fixture(scope="module")
def planner():
    return Planner()


def test_decisions_follow_the_profitability_threshold(planner):
    """Règle d'or : un client contacté a un rendement au-dessus du seuil, un client écarté non
    (à l'exception de quelques clients départagés au bord du budget)."""
    plan = planner.plan(SC, compare=False)
    lam = plan.summary["prix_marginal_budget"]
    best_yield = (planner.uplift / np.array([SC.cost_homme, SC.cost_femme])).max(axis=1)
    contacted = plan.assignment > 0
    assert (best_yield[contacted] >= lam - 1e-6).mean() > 0.999
    assert (best_yield[~contacted] <= lam + 1e-6).mean() > 0.999


def test_client_explanation_is_consistent_with_the_plan(planner):
    plan = planner.plan(SC, compare=False)
    contacted = int(np.flatnonzero(plan.assignment > 0)[0])
    skipped = int(np.flatnonzero(plan.assignment == 0)[0])
    a = explainer.client_explanation(planner, SC, contacted)
    b = explainer.client_explanation(planner, SC, skipped)
    assert a["decision"] != "rien" and a["meilleur_rendement_au_dessus_du_seuil"]
    assert a["raison"] == explainer.BEST
    assert b["decision"] == "rien"
    assert ("seuil" in b["raison"]) or ("uplift" in b["raison"])
    with pytest.raises(ValueError, match="numéro de client"):
        explainer.client_explanation(planner, SC, 10**6)


def test_segment_explanation_counts_reasons_and_keeps_measure_as_a_check(planner):
    f = explainer.segment_explanation(planner, SC, "homme seulement")
    assert f["email_homme"] + f["email_femme"] + f["rien"] == f["clients"]
    assert f["email_femme"] == 0  # l'e-mail femme ne marche presque pas sur ce segment
    # groupe vide (personne ne reçoit l'e-mail femme) : absent, et non une valeur manquante
    assert "clients_qui_recoivent_l_email_femme" not in f
    for group in ("clients_qui_recoivent_l_email_homme", "clients_sans_email"):
        assert sum(f[group]["raisons"].values()) == f[group]["clients"]
    check = f["verification_sur_la_campagne"]
    assert "jamais dans la décision" in check["role"]
    assert (
        check["uplift_mesure_email_homme"]["points"] > check["uplift_mesure_email_femme"]["points"]
    )
    assert f["niveau_de_confiance_pct"] == 95
    with pytest.raises(ValueError, match="Segment inconnu"):
        explainer.segment_explanation(planner, SC, "enfants")


def test_excluded_profile_contrasts_both_groups(planner):
    f = explainer.excluded_profile(planner, SC)
    assert f["contactes"]["clients"] + f["ecartes"]["clients"] == planner.n_clients
    assert (
        f["contactes"]["uplift_predit_moyen_homme_points"]
        > f["ecartes"]["uplift_predit_moyen_homme_points"]
    )


def test_portfolio_overview(planner):
    all_ = explainer.portfolio_overview(planner)
    assert all_["clients_du_perimetre"] == planner.n_clients
    assert abs(sum(all_["repartition_segments_pct"].values()) - 100) < 0.5
    seg = explainer.portfolio_overview(planner, "homme et femme")
    assert seg["clients_du_perimetre"] < seg["clients_du_portefeuille_entier"] == planner.n_clients
    # filtré sur un segment : pas de répartition à 100 % trompeuse (vu en test réel)
    assert "repartition_segments_pct" not in seg and 0 < seg["part_du_portefeuille_pct"] < 100


def test_reasons_follow_the_optimizer_rule_net_gain_not_yield(planner):
    """Le choix entre les deux e-mails suit le gain net (gain - seuil × coût), comme l'optimiseur."""
    plan = planner.plan(SC, compare=False)
    lam = plan.summary["prix_marginal_budget"]
    reasons = explainer.decision_reasons(planner, SC, plan, lam)
    net = planner.uplift - lam * np.array([SC.cost_homme, SC.cost_femme])
    best = reasons == explainer.BEST
    assert best.sum() > 0.99 * (plan.assignment > 0).sum()
    assert (plan.assignment[best] == net[best].argmax(axis=1) + 1).all()


def test_binding_send_limit_is_named_as_a_reason(planner):
    sc = Scenario(budget=700.0, max_homme=2000)
    plan = planner.plan(sc, compare=False)
    assert plan.action_prices[0] > 0  # la limite de l'e-mail homme pèse sur la décision
    counts = explainer.excluded_profile(planner, sc)["ecartes"]["raisons"]
    assert counts.get(explainer.LIMIT, 0) > 0


def test_screen_elements_are_all_explained(planner):
    ind = explainer.indicators_explanation(planner, SC)
    assert [i["indicateur"] for i in ind["indicateurs"]][0] == "Clients contactés"
    assert len(ind["indicateurs"]) == 5
    pol = explainer.policies_explanation(planner, SC)
    names = {p["politique"] for p in pol["politiques"]}
    assert names == {
        "personne",
        "ciblage classique",
        "plan optimisé",
        "tout le monde (e-mail homme)",
    }
    assert all(p["description"] for p in pol["politiques"])
    seg = explainer.segments_explanation(planner, SC)
    assert [s["segment"] for s in seg["segments"]] == list(explainer.SEGMENTS)
    overview = explainer.results_overview(planner, SC)
    assert {
        "indicateurs",
        "graphique_1_taux_de_visite_mesure",
        "graphique_2_qui_recoit_quoi",
    } <= set(overview)


def test_app_guide_numbers_are_numeric_fields(planner):
    """Le contrôle de fidélité ne lit que les champs numériques : les chiffres clés en sont."""
    from ciblage.agents import faithfulness
    from ciblage.serve.guide import app_guide

    g = app_guide(planner)
    text = (
        f"Le portefeuille compte {g['donnees']['clients_du_portefeuille_affiche']:,} clients, "
        "soit 30 % des 64 000 clients de la campagne, suivis 2 semaines ; intervalles à 95 %."
    ).replace(",", " ")
    assert faithfulness.check(text, g).ok
