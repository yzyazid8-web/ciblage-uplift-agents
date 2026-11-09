import numpy as np
import pytest

from ciblage.agents import faithfulness as F
from ciblage.agents.verify import verify_plan
from ciblage.serve.planner import Plan, Scenario


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Contacter 5 947 clients", [5947.0]),
        ("Contacter 5 947 clients pour 699,92 €", [5947.0, 699.92]),
        ("un gain de -0,24 point, soit +7,5 %", [-0.24, 7.5]),
        ("coût de 0.12 € par e-mail", [0.12]),
        ("aucun chiffre ici", []),
    ],
)
def test_parse_numbers_handles_french_formats(text, expected):
    assert [v for _, v, _ in F.parse_numbers(text)] == pytest.approx(expected)


def test_check_accepts_rounded_and_percent_versions():
    facts = {"clients": 5947, "depense": 699.92, "gain": -0.0024, "taux": 0.1304}
    text = "Le plan contacte 5 947 clients pour 700 €, taux de visite 13,0 %, écart de -0,24 point."
    report = F.check(text, facts)
    assert report.ok, report.unsupported


def test_check_flags_invented_numbers():
    facts = {"clients": 5947, "depense": 699.92}
    report = F.check("Le plan contacte 6 200 clients et rapporte 1 500 €.", facts)
    assert not report.ok
    assert report.unsupported == ["6 200", "1 500"]


def test_trivial_numbers_and_nested_facts():
    facts = {"segments": [{"uplift": 0.0752}, {"uplift": 0.0226}]}
    assert F.check("Les 2 e-mails : +7,52 et +2,26 points.", facts).ok
    assert F.flatten({"a": [1, {"b": 2.5}], "c": True, "d": "x"}) == [1.0, 2.5]


def _plan(assignment, budget=1.0, **summary):
    sc = Scenario(budget=budget, cost_homme=0.5, cost_femme=0.25, max_femme=1)
    a = np.asarray(assignment)
    base = {
        "email_homme": int((a == 1).sum()),
        "email_femme": int((a == 2).sum()),
        "depense": float(0.5 * (a == 1).sum() + 0.25 * (a == 2).sum()),
    }
    return Plan(
        scenario=sc,
        assignment=a,
        summary={**base, **summary},
        bound_gap=0.0,
        seconds=0.0,
        by_segment=None,
    )


def test_verify_accepts_conforming_plan():
    assert verify_plan(_plan([1, 2, 0, 0])) == []


def test_verify_detects_each_violation():
    assert any("budget" in s for s in verify_plan(_plan([1, 1, 1])))
    assert any("limite" in s for s in verify_plan(_plan([2, 2, 0])))
    assert any("résumé" in s for s in verify_plan(_plan([1, 0], email_homme=5)))
    assert any("inconnus" in s for s in verify_plan(_plan([3, 0])))
