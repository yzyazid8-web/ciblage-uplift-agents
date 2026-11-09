import pytest

from ciblage.serve import advisor
from ciblage.serve.planner import Planner, Scenario


@pytest.fixture(scope="module")
def planner():
    return Planner()


def test_shadow_price_matches_real_budget_change(planner):
    """Le prix marginal théorique (lambda) coïncide avec l'effet réel d'un petit ajout de budget."""
    res = advisor.marginal_value(planner, Scenario(budget=700.0), delta=20.0)
    assert res["visites_par_euro_de_variation"] == pytest.approx(
        res["prix_marginal_actuel_visites_par_euro"], rel=0.05
    )
    assert res["visites_envisagees"] > res["visites_actuelles"]


def test_returns_are_diminishing_then_saturate(planner):
    curve = advisor.budget_curve(planner, Scenario(budget=700.0), points=6)
    rates = [r["visites_par_euro_supplementaire"] for r in curve["courbe"]]
    visits = [r["visites_en_plus"] for r in curve["courbe"]]
    assert visits == sorted(visits)  # plus de budget, jamais moins de visites
    assert rates[1] >= rates[-2] >= rates[-1]  # rendements décroissants
    assert rates[-1] == 0.0  # au budget de saturation, un euro de plus ne rapporte rien
    assert 700.0 in [r["budget"] for r in curve["courbe"]]


def test_budget_beyond_saturation_brings_nothing(planner):
    sat = advisor.saturation_budget(planner, Scenario(budget=100.0))
    res = advisor.marginal_value(planner, Scenario(budget=sat), delta=500.0)
    assert res["visites_en_plus_ou_moins"] == 0


def test_cost_threshold_is_where_the_email_stops_being_used(planner):
    sc = Scenario(budget=700.0)
    res = advisor.cost_threshold(planner, sc, "femme")
    seuil = res["seuil_cout_au_dela_duquel_plus_aucun_envoi"]
    assert res["envois_actuels"] > 0 and seuil > sc.cost_femme
    above = planner.plan(Scenario(budget=700.0, cost_femme=seuil * 1.01), compare=False)
    assert above.summary["email_femme"] == 0


def test_compare_scenarios_table():
    h = {
        "scenario": {"budget": 700.0, "cost_homme": 0.12, "cost_femme": 0.1},
        "summary": {
            "clients_contactes": 10,
            "email_homme": 6,
            "email_femme": 4,
            "depense": 1.12,
            "visites_en_plus_predites": 3.4,
            "cout_par_visite_en_plus": 0.33,
        },
        "mesure": {"gain": -0.24},
    }
    rows = advisor.compare_scenarios([h] * 7, last=5)
    assert [r["scenario"] for r in rows] == [3, 4, 5, 6, 7]
    assert rows[0]["gain_mesure_vs_classique_points"] == -0.24
