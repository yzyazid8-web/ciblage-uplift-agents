"""Graphe de l'agent (LangGraph) : superviseur, planificateur, clarification, vérification, explication."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from ciblage.agents import faithfulness, prompts
from ciblage.agents.llm import Understander, Writer, understand_messages
from ciblage.agents.schema import AgentState, Understanding
from ciblage.agents.verify import verify_plan
from ciblage.serve.planner import Planner, Scenario

REQUIRED = {
    "budget": "le budget total (en €)",
    "cost_homme": "le coût d'un e-mail homme (en €)",
    "cost_femme": "le coût d'un e-mail femme (en €)",
}
MAX_CLARIFY = 3
SCENARIO_FIELDS = {f for f in Scenario.__dataclass_fields__}

REFUSAL = (
    "Je suis un copilote de campagnes : je calcule qui contacter, avec quel e-mail et pour quel "
    "budget, et j'explique ce que le plan rapporte. Je ne peux pas traiter cette demande."
)


@dataclass
class Deps:
    planner: Planner
    understand: Understander
    write: Writer


# outils de mise en forme
def _context(state: AgentState) -> str:
    current = state.get("draft") or state.get("scenario")
    lines = [
        "Scénario en cours : " + (json.dumps(current, ensure_ascii=False) if current else "aucun")
    ]
    if state.get("pending_intent"):
        lines.append(
            f"Tu viens de poser cette question à l'utilisateur : « {state.get('question')} ». "
            "Son message est la RÉPONSE : extrais les valeurs qu'il donne."
        )
    return "\n".join(lines)


def merge(base: dict[str, Any] | None, u: Understanding) -> tuple[dict[str, Any], list[str]]:
    """Applique ce que le message demande au scénario de base. Renvoie (scénario, problèmes)."""
    draft = dict(base or {})
    issues: list[str] = []
    if u.budget is not None:
        draft["budget"] = u.budget
    if u.budget_factor is not None or u.budget_delta is not None:
        if draft.get("budget") is None:
            issues.append(
                "Il n'y a pas encore de budget de référence à modifier : quel budget veux-tu ?"
            )
        else:
            if u.budget_factor is not None:
                draft["budget"] = draft["budget"] * u.budget_factor
            if u.budget_delta is not None:
                draft["budget"] = draft["budget"] + u.budget_delta
    for name in (
        "cost_homme",
        "cost_femme",
        "max_homme",
        "max_femme",
        "objective",
        "value_per_visit",
    ):
        value = getattr(u, name)
        if value is not None:
            draft[name] = value
    if u.remove_limits:
        draft["max_homme"] = draft["max_femme"] = None
    return draft, issues


def build_facts(result: dict[str, Any], previous: dict[str, Any] | None = None) -> dict[str, Any]:
    """Dossier de faits compact et arrondi, seule source de chiffres pour l'explication."""
    s = result["summary"]
    cmp = {r["politique"]: r for r in result["comparison"]}
    mine, classic = cmp["plan optimisé"], cmp["ciblage classique"]
    facts: dict[str, Any] = {
        "scenario": result["scenario"],
        "plan": {
            "clients_contactes": s["clients_contactes"],
            "email_homme": s["email_homme"],
            "email_femme": s["email_femme"],
            "depense_eur": round(s["depense"], 2),
            "budget_inutilise_eur": round(s["budget_inutilise"], 2),
            "visites_en_plus_predites": round(s["visites_en_plus_predites"]),
            "cout_par_visite_gagnee_eur": round(s["cout_par_visite_en_plus"], 2),
            "ecart_prouve_a_l_optimum_pct": round(100 * result["bound_gap"], 3),
        },
        "mesure_sur_la_campagne": {
            "taux_visite_plan_pct": round(100 * mine["taux de visite"], 2),
            "taux_visite_ciblage_classique_pct": round(100 * classic["taux de visite"], 2),
            "gain_vs_classique_points": round(mine["gain vs classique (pts)"], 2),
            "ic95_gain_bas": round(mine["gain_bas"], 2),
            "ic95_gain_haut": round(mine["gain_haut"], 2),
            "gain_demontre": bool(mine["gain_bas"] > 0),
            "niveau_de_confiance_pct": 95,
        },
        "segments": [
            {
                "segment": r["segment"],
                "clients": r["clients"],
                "email_homme": r["e-mail homme"],
                "email_femme": r["e-mail femme"],
                "rien": r["rien"],
                "uplift_homme_points": r["uplift prédit homme (pts)"],
                "uplift_femme_points": r["uplift prédit femme (pts)"],
            }
            for r in result["by_segment"]
        ],
    }
    if previous:
        p = previous["summary"]
        facts["scenario_precedent"] = {
            "budget": previous["scenario"]["budget"],
            "clients_contactes": p["clients_contactes"],
            "depense_eur": round(p["depense"], 2),
            "visites_en_plus_predites": round(p["visites_en_plus_predites"]),
        }
    return facts


def _fr(x: float, decimals: int = 2, sign: bool = False) -> str:
    """Nombre au format français (virgule décimale, espace fine pour les milliers)."""
    text = f"{x:+,.{decimals}f}" if sign else f"{x:,.{decimals}f}"
    return text.replace(",", " ").replace(".", ",")


def template_answer(facts: dict[str, Any]) -> str:
    """Réponse standard, sans LLM : utilisée si l'explication rédigée n'est pas fiable."""
    p, m = facts["plan"], facts["mesure_sur_la_campagne"]
    gain = (
        f"Mesuré sur la campagne, le plan fait {_fr(m['gain_vs_classique_points'], sign=True)} point de "
        f"visite par rapport au ciblage classique (IC 95 % : {_fr(m['ic95_gain_bas'], sign=True)} à "
        f"{_fr(m['ic95_gain_haut'], sign=True)})"
    )
    verdict = "." if m["gain_demontre"] else " : pas de gain démontré sur ce budget."
    return (
        f"Plan recommandé : {_fr(p['email_homme'], 0)} e-mails homme et {_fr(p['email_femme'], 0)} "
        f"e-mails femme, soit {_fr(p['clients_contactes'], 0)} clients pour {_fr(p['depense_eur'])} €. "
        f"Visites en plus attendues : environ {_fr(p['visites_en_plus_predites'], 0)} "
        f"({_fr(p['cout_par_visite_gagnee_eur'])} € par visite gagnée). {gain}{verdict}"
    )


# construction du graphe 
    def start(state: AgentState) -> dict[str, Any]:
        return {"trace": ["début"], "llm_calls": 0, "answer": "", "clarify_count": 0, "route": ""}

    def understand(state: AgentState) -> dict[str, Any]:
        msgs = understand_messages(state["user_input"], _context(state))
        u = deps.understand(msgs)
        return {
            "understanding": u.model_dump(),
            "llm_calls": state.get("llm_calls", 0) + 1,
            "trace": state.get("trace", []) + [f"compréhension : {u.intent}"],
        }

    def supervisor(state: AgentState) -> dict[str, Any]:
        intent = state.get("pending_intent") or state["understanding"]["intent"]
        if intent in ("plan", "modifier"):
            route = "assemble"
        elif intent == "hors_sujet":
            route = "refuse"
        else:
            route = "not_yet"
        return {"route": route, "trace": state["trace"] + [f"superviseur -> {route}"]}

    def assemble(state: AgentState) -> dict[str, Any]:
        u = Understanding.model_validate(state["understanding"])
        base = state.get("draft") if state.get("pending_intent") else state.get("scenario")
        draft, issues = merge(base, u)
        missing = [label for key, label in REQUIRED.items() if draft.get(key) is None]
        if draft.get("objective") == "profit" and draft.get("value_per_visit") is None:
            missing.append("ce que rapporte une visite (en €), pour l'objectif profit")
        if not missing and not issues:
            errors = Scenario(**{k: v for k, v in draft.items() if k in SCENARIO_FIELDS}).errors()
            issues += errors
        problems = (
            ([f"Il me manque : {', '.join(missing)}."] if missing else []) + issues + u.ambiguities
        )
        intent = state.get("pending_intent") or u.intent
        if problems:
            return {
                "draft": draft,
                "pending_intent": intent,
                "question": " ".join(problems),
                "route": "clarify",
                "trace": state["trace"] + ["assemblage : informations manquantes"],
            }
        return {
            "draft": draft,
            "route": "compute",
            "trace": state["trace"] + ["assemblage : scénario complet"],
        }

    def clarify(state: AgentState) -> dict[str, Any]:
        if state.get("clarify_count", 0) >= MAX_CLARIFY:
            return {
                "answer": "Je n'arrive pas à réunir les informations nécessaires. Peux-tu reformuler ta demande "
                "en précisant le budget et le coût de chaque e-mail ?",
                "pending_intent": None,
                "draft": None,
                "route": "end",
                "trace": state["trace"] + ["clarification : abandon"],
            }
        reply = interrupt({"question": state["question"]})
        return {
            "user_input": str(reply),
            "clarify_count": state.get("clarify_count", 0) + 1,
            "route": "understand",
            "trace": state["trace"] + ["clarification : réponse reçue"],
        }

    def compute(state: AgentState) -> dict[str, Any]:
        draft = state["draft"] or {}
        sc = Scenario(**{k: v for k, v in draft.items() if k in SCENARIO_FIELDS})
        plan = deps.planner.plan(sc, n_boot=200)
        issues = verify_plan(plan)
        if issues:  # l'optimiseur est déterministe : un nouvel essai donnerait le même plan
            return {
                "answer": "Le plan calculé ne respecte pas tes contraintes ("
                + " ".join(issues)
                + ") : je préfère ne pas te le proposer.",
                "route": "end",
                "trace": state["trace"] + ["vérification : échec"],
            }
        result = plan.to_dict()
        previous = state.get("result")
        return {
            "turn_intent": state.get("pending_intent") or state["understanding"]["intent"],
            "scenario": asdict(sc),
            "result": result,
            "history": [{"scenario": asdict(sc), "summary": result["summary"]}],
            "draft": None,
            "pending_intent": None,
            "question": None,
            "route": "explain",
            "trace": state["trace"] + ["calcul du plan", "vérification : conforme"],
            "previous_result": previous,
        }

    def explain(state: AgentState) -> dict[str, Any]:
        previous = state.get("previous_result")
        result = state.get("result")
        assert result is not None, "explain est toujours précédé d'un calcul réussi"
        facts = build_facts(result, previous if state.get("turn_intent") == "modifier" else None)
        dossier = json.dumps(facts, ensure_ascii=False)
        msgs = [("system", prompts.EXPLAIN), ("user", "DOSSIER DE FAITS :\n" + dossier)]
        calls, trace = state.get("llm_calls", 0), list(state["trace"])
        for attempt in range(2):
            text = deps.write(msgs)
            calls += 1
            report = faithfulness.check(text, facts)
            if report.ok:
                trace.append(f"explication : chiffres vérifiés (essai {attempt + 1})")
                return {"answer": text, "llm_calls": calls, "trace": trace}
            trace.append(f"explication : chiffres non vérifiés {report.unsupported}")
            msgs = msgs + [
                ("user", prompts.REGENERATE.format(numbers=", ".join(report.unsupported)))
            ]
        trace.append("explication : texte standard (repli sûr)")
        return {"answer": template_answer(facts), "llm_calls": calls, "trace": trace}

    def refuse(state: AgentState) -> dict[str, Any]:
        return {"answer": REFUSAL, "trace": state["trace"] + ["refus (hors sujet)"]}

    def not_yet(state: AgentState) -> dict[str, Any]:
        return {
            "answer": "Cette demande sera bientôt prise en charge. Pour l'instant, je peux calculer un plan "
            "(« j'ai 700 €, l'e-mail homme coûte 0,12 € et l'e-mail femme 0,10 € ») et le modifier "
            "(« et avec la moitié du budget ? »).",
            "trace": state["trace"] + ["intention pas encore prise en charge"],
        }

    g = StateGraph(AgentState)
    for name, fn in [
        ("start", start),
        ("understand", understand),
        ("supervisor", supervisor),
        ("assemble", assemble),
        ("clarify", clarify),
        ("compute", compute),
        ("explain", explain),
        ("refuse", refuse),
        ("not_yet", not_yet),
    ]:
        g.add_node(name, fn)
    g.add_edge(START, "start")
    g.add_edge("start", "understand")
    g.add_edge("understand", "supervisor")
    g.add_conditional_edges("supervisor", lambda s: s["route"], ["assemble", "refuse", "not_yet"])
    g.add_conditional_edges("assemble", lambda s: s["route"], ["clarify", "compute"])
    g.add_conditional_edges(
        "clarify", lambda s: s["route"], {"understand": "understand", "end": END}
    )
    g.add_conditional_edges("compute", lambda s: s["route"], {"explain": "explain", "end": END})
    for name in ("explain", "refuse", "not_yet"):
        g.add_edge(name, END)
    return g.compile(checkpointer=checkpointer)
