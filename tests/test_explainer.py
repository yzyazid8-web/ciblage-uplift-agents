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
    assert b["decision"] == "rien"
    assert ("seuil" in b["raison"]) or ("uplift" in b["raison"])
    with pytest.raises(ValueError, match="numéro de client"):
        explainer.client_explanation(planner, SC, 10**6)


def test_segment_explanation_adds_measured_uplift(planner):
    f = explainer.segment_explanation(planner, SC, "homme seulement")
    assert f["email_homme"] + f["email_femme"] + f["rien"] == f["clients"]
    assert f["email_femme"] == 0  # l'e-mail femme ne marche presque pas sur ce segment
    assert f["uplift_mesure_homme"]["points"] > f["uplift_mesure_femme"]["points"]
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
