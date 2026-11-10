"""Tests de l'évaluation elle-même (sans LLM) : un évaluateur faux donnerait des chiffres faux."""

from ciblage.eval import cases as C
from ciblage.eval.report import wilson
from ciblage.eval.run import check_conversation, evaluate_understanding


def test_case_counts_and_unique_ids():
    l1, l2 = C.understanding_cases(), C.conversation_cases()
    assert len(l1) == 120 and len(l2) >= 25
    assert len({c.id for c in l1}) == len(l1) and len({c.id for c in l2}) == len(l2)


def test_generated_messages_contain_their_ground_truth():
    """Le budget attendu doit figurer dans le message (sinon l'attendu serait faux)."""
    for c in C.understanding_cases():
        if c.category == "plan":
            assert str(c.expect["budget"]) in c.message.replace(" ", "")


def test_generation_is_reproducible():
    assert [c.message for c in C.understanding_cases()] == [
        c.message for c in C.understanding_cases()
    ]


def _u(**kw):
    base = {k: None for k in C.SCENARIO_KEYS}
    base.update(intent="plan", ambiguities=[], about_excluded=False)
    base.update(kw)
    return base


def test_evaluate_understanding_detects_errors_and_inventions():
    case = C.UnderstandingCase(
        "x",
        "plan_incomplet",
        "msg",
        {"intent": "plan", "budget": 500, "cost_homme": 0.12},
        check_no_invention=True,
    )
    good = evaluate_understanding(case, _u(budget=500.0, cost_homme=0.12))
    assert (
        good["parametres_ok"] and good["inventes"] == [] and good["question_posee"]
    )  # coût femme manquant
    bad = evaluate_understanding(case, _u(budget=50, cost_homme=0.12, cost_femme=0.10))
    assert not bad["parametres_ok"] and bad["inventes"] == ["cost_femme"]


def test_evaluate_understanding_applies_deterministic_rules():
    case = C.UnderstandingCase(
        "y",
        "expliquer",
        "Qui sont les clients écartés ?",
        {"intent": "expliquer"},
        context={"budget": 1},
    )
    res = evaluate_understanding(case, _u(intent="donnees"))
    assert not res["intent_brut_ok"] and res["intent_corrige_ok"]


def test_check_conversation():
    case = C.ConversationCase(
        "z", "export_refuse", ["exporte", "non"], {"first": "question", "export_written": False}
    )
    turns = [
        {"type": "question", "reponse": "Confirmes-tu ?", "appels_llm": 1},
        {"type": "reponse", "reponse": "Export annulé", "appels_llm": 0},
    ]
    assert check_conversation(case, turns, {}, exports_written=False)["succes"]
    assert not check_conversation(case, turns, {}, exports_written=True)["succes"]


def test_wilson_interval():
    lo, hi = wilson(45, 50)
    assert 0.78 < lo < 0.80 and 0.95 < hi < 0.97
    assert wilson(0, 10)[0] == 0.0 and wilson(10, 10)[1] == 1.0
    assert wilson(0, 0) != wilson(0, 0)  # NaN quand aucun cas
