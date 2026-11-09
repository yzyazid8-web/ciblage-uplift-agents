"""Graphe de l'agent (LangGraph) : superviseur, planificateur, clarification, vérification, explication."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from ciblage.agents import faithfulness, prompts
from ciblage.agents.llm import Understander, Writer, understand_messages
from ciblage.agents.schema import AgentState, Understanding
from ciblage.agents.verify import verify_plan
from ciblage.serve import advisor, explainer, export
from ciblage.serve.planner import Planner, Scenario

REQUIRED = {
    "budget": "le budget total (en €)",
    "cost_homme": "le coût d'un e-mail homme (en €)",
    "cost_femme": "le coût d'un e-mail femme (en €)",
}
MAX_CLARIFY = 3
MAX_MESSAGE_CHARS = 2_000  # protège le quota : un message plus long n'est pas envoyé au LLM
EXCLUDED_WORDS = re.compile(
    r"(écart|exclu|non[- ]contact|pas contact|laiss\w* de côté)", re.IGNORECASE
)
MARKDOWN = re.compile(r"\*\*|__")  # gras markdown, retiré malgré la consigne si le LLM en met
NEXT_STEP = {"conseil": "advise", "expliquer": "explain_decision", "exporter": "confirm_export"}
YES = re.compile(
    r"^\s*(oui|o|ok|yes|y|vas-y|go|confirme|je confirme|d'accord|c'est bon)\b", re.IGNORECASE
)
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
    exports_dir: Path = export.EXPORTS_DIR


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


def merge(
    base: dict[str, Any] | None, u: Understanding, apply_relative: bool = True
) -> tuple[dict[str, Any], list[str]]:
    """Applique ce que le message demande au scénario de base. Renvoie (scénario, problèmes)."""
    draft = dict(base or {})
    issues: list[str] = []
    if u.budget is not None:
        draft["budget"] = u.budget
    if apply_relative and (u.budget_factor is not None or u.budget_delta is not None):
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


def template_advice(kind: str, facts: dict[str, Any]) -> str:
    """Réponses standard (sans LLM) pour le conseil et la comparaison."""
    if kind == "conseil_budget_delta":
        sens = "de plus" if facts["variation_budget"] >= 0 else "de moins"
        return (
            f"Avec {_fr(abs(facts['variation_budget']))} € {sens} (budget de "
            f"{_fr(facts['budget_envisage'])} €), le plan prévoit {_fr(facts['visites_envisagees'], 0)} "
            f"visites en plus au lieu de {_fr(facts['visites_actuelles'], 0)}, soit "
            f"{_fr(facts['visites_par_euro_de_variation'], 3)} visite par euro de variation. Au budget "
            f"actuel, un euro supplémentaire rapporte {_fr(facts['prix_marginal_actuel_visites_par_euro'], 3)} "
            f"visite ; au-delà de {_fr(facts['budget_de_saturation'])} €, un euro de plus ne rapporte "
            "plus rien. Ces visites sont prédites par le modèle."
        )
    if kind == "conseil_budget_courbe":
        points = "; ".join(
            f"{_fr(r['budget'], 0)} € → {_fr(r['visites_en_plus'], 0)} visites"
            for r in facts["courbe"]
        )
        return (
            f"Visites en plus prédites selon le budget : {points}. Au-delà de "
            f"{_fr(facts['budget_de_saturation'])} €, un euro de plus ne rapporte plus de visite."
        )
    if kind == "conseil_cout":
        return (
            f"Au coût actuel de {_fr(facts['cout_actuel'])} €, le plan envoie "
            f"{_fr(facts['envois_actuels'], 0)} e-mails {facts['email']}. Au-delà de "
            f"{_fr(facts['seuil_cout_au_dela_duquel_plus_aucun_envoi'], 3)} € par envoi, il n'en "
            "envoie plus aucun avec ce budget."
        )
    if kind == "comparaison":
        rows = "; ".join(
            f"scénario {r['scenario']} : {_fr(r['budget'], 0)} €, {_fr(r['clients_contactes'], 0)} "
            f"clients, {_fr(r['visites_en_plus_predites'], 0)} visites en plus "
            f"({_fr(r['cout_par_visite_gagnee'])} € par visite)"
            for r in facts["scenarios"]
        )
        return f"Comparaison des scénarios calculés : {rows}."
    raise ValueError(f"Type de réponse inconnu : {kind}")


def _flat(facts: Any, prefix: str = "") -> list[str]:
    """Faits aplatis en « libellé : valeur » lisibles (pour le texte de repli générique)."""
    if isinstance(facts, dict):
        out: list[str] = []
        for k, v in facts.items():
            out += _flat(
                v, f"{prefix}{k.replace('_', ' ')} › " if isinstance(v, dict | list) else ""
            )
            if not isinstance(v, dict | list) and v is not None:
                value = _fr(v, 3).rstrip("0").rstrip(",") if isinstance(v, float) else str(v)
                out.append(f"{prefix}{k.replace('_', ' ')} : {value}")
        return out
    if isinstance(facts, list):
        return [line for item in facts for line in _flat(item, prefix)]
    return []


def generic_template(facts: dict[str, Any]) -> str:
    """Repli sur pour les explications et les données : les faits calculés, tels quels."""
    return "Voici les éléments calculés : " + " ; ".join(_flat(facts)) + "."


def fallback(kind: str, facts: dict[str, Any]) -> str:
    if kind == "plan":
        return template_answer(facts)
    if kind.startswith(("conseil", "comparaison")):
        return template_advice(kind, facts)
    return generic_template(facts)


def _has_campaign_params(understanding: dict[str, Any]) -> bool:
    """Le message contient-il au moins un paramètre de campagne (budget, coût, limite...) ?"""
    keys = (
        "budget",
        "budget_factor",
        "budget_delta",
        "cost_homme",
        "cost_femme",
        "max_homme",
        "max_femme",
    )
    return any(understanding.get(k) is not None for k in keys)


def _scenario(draft: dict[str, Any] | None) -> Scenario:
    return Scenario(**{k: v for k, v in (draft or {}).items() if k in SCENARIO_FIELDS})


# construction du graphe
def build_graph(deps: Deps, checkpointer: Any = None):
    def start(state: AgentState) -> dict[str, Any]:
        too_long = len(state.get("user_input", "")) > MAX_MESSAGE_CHARS
        return {
            "trace": ["début"],
            "llm_calls": 0,
            "answer": "",
            "clarify_count": 0,
            "route": "too_long" if too_long else "understand",
            "facts": None,
            "facts_kind": "",
            "turn_question": state.get("user_input", ""),
        }

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
        trace = list(state["trace"])
        understanding = dict(state["understanding"])
        if intent == "hors_sujet" and _has_campaign_params(understanding):
            # incohérence du LLM (observée en test réel) : des paramètres de campagne ont été
            # extraits, le message n'est donc pas hors sujet -> règle déterministe
            intent = "modifier" if state.get("scenario") else "plan"
            trace.append(
                f"superviseur : « hors sujet » corrigé en « {intent} » (paramètres présents)"
            )
        if intent == "donnees" and EXCLUDED_WORDS.search(state.get("turn_question", "")):
            # « qui sont les clients écartés ? » porte sur le PLAN, pas sur le portefeuille
            intent = "expliquer"
            understanding["about_excluded"] = True
            trace.append("superviseur : question sur les clients écartés -> explication du plan")
        understanding["intent"] = intent  # les étapes suivantes lisent l'intention corrigée
        routes = {
            "plan": "assemble",
            "modifier": "assemble",
            "conseil": "assemble",
            "expliquer": "assemble",
            "exporter": "assemble",
            "comparer": "compare",
            "donnees": "data",
            "hors_sujet": "refuse",
        }
        route = routes.get(intent, "not_yet")
        return {
            "understanding": understanding,
            "turn_intent": intent,
            "route": route,
            "trace": trace + [f"superviseur -> {route}"],
        }

    def assemble(state: AgentState) -> dict[str, Any]:
        u = Understanding.model_validate(state["understanding"])
        intent = state.get("pending_intent") or u.intent
        update: dict[str, Any] = {}
        if not state.get("pending_intent"):
            # demande d'origine (sujet du conseil, client, segment...), gardée pendant la clarification
            update["request"] = u.model_dump()
        base = state.get("draft") if state.get("pending_intent") else state.get("scenario")
        # seules les demandes de plan modifient le budget relativement ; pour un conseil,
        # « 100 € de plus » est la question posée, pas une modification du scénario
        draft, issues = merge(base, u, apply_relative=intent in ("plan", "modifier"))
        missing = [label for key, label in REQUIRED.items() if draft.get(key) is None]
        if draft.get("objective") == "profit" and draft.get("value_per_visit") is None:
            missing.append("ce que rapporte une visite (en €), pour l'objectif profit")
        if not missing and not issues:
            issues += _scenario(draft).errors()
        problems = (
            ([f"Il me manque : {', '.join(missing)}."] if missing else []) + issues + u.ambiguities
        )
        if problems:
            return {
                **update,
                "draft": draft,
                "pending_intent": intent,
                "question": " ".join(problems),
                "route": "clarify",
                "trace": state["trace"] + ["assemblage : informations manquantes"],
            }
        return {
            **update,
            "draft": draft,
            "route": NEXT_STEP.get(intent, "compute"),
            "trace": state["trace"] + ["assemblage : scénario complet"],
        }

    def clarify(state: AgentState) -> dict[str, Any]:
        if state.get("clarify_count", 0) >= MAX_CLARIFY:
            return {
                "answer": "Je n'arrive pas à réunir les informations nécessaires. Peux-tu reformuler ta "
                "demande en précisant le budget et le coût de chaque e-mail ?",
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
        sc = _scenario(state["draft"])
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
        mine = next(r for r in result["comparison"] if r["politique"] == "plan optimisé")
        facts = build_facts(result, previous if state.get("turn_intent") == "modifier" else None)
        return {
            "scenario": asdict(sc),
            "result": result,
            "previous_result": previous,
            "history": [
                {
                    "scenario": asdict(sc),
                    "summary": result["summary"],
                    "mesure": {
                        "gain": round(mine["gain vs classique (pts)"], 2),
                        "ic95_bas": round(mine["gain_bas"], 2),
                        "ic95_haut": round(mine["gain_haut"], 2),
                    },
                }
            ],
            "draft": None,
            "pending_intent": None,
            "question": None,
            "facts": facts,
            "facts_kind": "plan",
            "route": "write",
            "trace": state["trace"] + ["calcul du plan", "vérification : conforme"],
        }

    def advise(state: AgentState) -> dict[str, Any]:
        sc = _scenario(state["draft"])
        request = state.get("request") or {}
        topic, delta = request.get("advice_topic"), request.get("budget_delta")
        if topic in ("cout_homme", "cout_femme"):
            email: Literal["homme", "femme"] = "homme" if topic == "cout_homme" else "femme"
            facts, kind = advisor.cost_threshold(deps.planner, sc, email), "conseil_cout"
        elif delta:
            facts, kind = (
                advisor.marginal_value(deps.planner, sc, float(delta)),
                "conseil_budget_delta",
            )
        else:
            facts, kind = advisor.budget_curve(deps.planner, sc), "conseil_budget_courbe"
        return {
            "scenario": asdict(sc),  # le scénario de référence est conservé, sans nouveau plan
            "draft": None,
            "pending_intent": None,
            "question": None,
            "request": None,
            "facts": facts,
            "facts_kind": kind,
            "route": "write",
            "trace": state["trace"] + [f"conseil : {kind}"],
        }

    def compare(state: AgentState) -> dict[str, Any]:
        history = state.get("history", [])
        if len(history) < 2:
            return {
                "answer": "Il faut au moins deux scénarios calculés pour les comparer. Calcule un plan, "
                "puis demande par exemple « et avec 1 000 € ? ».",
                "route": "end",
                "trace": state["trace"] + ["comparaison : pas assez de scénarios"],
            }
        return {
            "facts": {"scenarios": advisor.compare_scenarios(history)},
            "facts_kind": "comparaison",
            "route": "write",
            "trace": state["trace"] + ["comparaison des scénarios"],
        }

    def write(state: AgentState) -> dict[str, Any]:
        """Rédaction par le LLM, contrôle des chiffres, nouvelle rédaction puis repli sûr."""
        facts, kind = state.get("facts") or {}, state["facts_kind"]
        msgs = [
            ("system", prompts.PROMPTS[kind]),
            ("user", f"QUESTION : {state.get('turn_question', '')}"),
            ("user", "DOSSIER DE FAITS :\n" + json.dumps(facts, ensure_ascii=False)),
        ]
        calls, trace = state.get("llm_calls", 0), list(state["trace"])
        for attempt in range(2):
            text = deps.write(msgs)
            calls += 1
            report = faithfulness.check(text, facts)
            if report.ok:
                trace.append(f"rédaction : chiffres vérifiés (essai {attempt + 1})")
                return {"answer": MARKDOWN.sub("", text), "llm_calls": calls, "trace": trace}
            trace.append(f"rédaction : chiffres non vérifiés {report.unsupported}")
            msgs = msgs + [
                ("user", prompts.REGENERATE.format(numbers=", ".join(report.unsupported)))
            ]
        trace.append("rédaction : texte standard (repli sûr)")
        return {"answer": fallback(kind, facts), "llm_calls": calls, "trace": trace}

    def explain_decision(state: AgentState) -> dict[str, Any]:
        sc = _scenario(state["draft"])
        req = state.get("request") or {}
        try:
            if req.get("client_id") is not None:
                facts = explainer.client_explanation(deps.planner, sc, int(req["client_id"]))
                kind = "explication_client"
            elif req.get("segment"):
                facts = explainer.segment_explanation(deps.planner, sc, req["segment"])
                kind = "explication_segment"
            elif req.get("about_excluded"):
                facts = explainer.excluded_profile(deps.planner, sc)
                kind = "explication_ecartes"
            else:
                facts = {
                    "segments": [
                        explainer.segment_explanation(deps.planner, sc, seg)
                        for seg in explainer.SEGMENTS
                    ]
                }
                kind = "explication_segments"
        except ValueError as err:  # numéro de client hors portefeuille, segment inconnu
            return {
                "answer": f"{err}.",
                "route": "end",
                "trace": state["trace"] + ["explication : demande invalide"],
            }
        return {
            "scenario": asdict(sc),
            "draft": None,
            "pending_intent": None,
            "question": None,
            "request": None,
            "facts": facts,
            "facts_kind": kind,
            "route": "write",
            "trace": state["trace"] + [f"explication : {kind}"],
        }

    def data(state: AgentState) -> dict[str, Any]:
        segment = (state.get("understanding") or {}).get("segment")
        return {
            "facts": explainer.portfolio_overview(deps.planner, segment),
            "facts_kind": "donnees",
            "trace": state["trace"] + ["données : portrait du portefeuille"],
        }

    def confirm_export(state: AgentState) -> dict[str, Any]:
        """Seule action à effet réel : on montre ce qui sera écrit et on attend un oui explicite."""
        sc = _scenario(state["draft"])
        p = export.preview(deps.planner, sc)
        question = (
            f"Je vais créer le fichier de campagne : {_fr(p['lignes'], 0)} clients "
            f"({_fr(p['email_homme'], 0)} e-mails homme, {_fr(p['email_femme'], 0)} e-mails femme), "
            f"pour une dépense de {_fr(p['depense_eur'])} €. Confirmes-tu l'export ? (oui / non)"
        )
        reply = str(interrupt({"question": question}))
        base = {
            "scenario": asdict(sc),
            "draft": None,
            "pending_intent": None,
            "question": None,
            "request": None,
        }
        if not YES.match(reply):
            return {
                **base,
                "answer": "Export annulé : aucun fichier n'a été écrit.",
                "trace": state["trace"] + ["export : refusé par l'utilisateur"],
            }
        res = export.export_campaign(deps.planner, sc, directory=deps.exports_dir)
        return {
            **base,
            "answer": f"Fichier de campagne écrit : {res['fichier']} ({_fr(res['lignes'], 0)} lignes).",
            "trace": state["trace"] + ["export : confirmé et écrit"],
        }

    def too_long(state: AgentState) -> dict[str, Any]:
        return {
            "answer": f"Ton message dépasse {MAX_MESSAGE_CHARS} caractères : peux-tu le raccourcir ?",
            "trace": state["trace"] + ["garde-fou : message trop long (aucun appel au LLM)"],
        }

    def refuse(state: AgentState) -> dict[str, Any]:
        return {"answer": REFUSAL, "trace": state["trace"] + ["refus (hors sujet)"]}

    def not_yet(state: AgentState) -> dict[str, Any]:
        return {
            "answer": "Cette demande sera bientôt prise en charge. Pour l'instant, je peux calculer un "
            "plan (« j'ai 700 €, l'e-mail homme coûte 0,12 € et l'e-mail femme 0,10 € »), le modifier "
            "(« et avec la moitié du budget ? »), le comparer aux précédents et te conseiller "
            "(« 100 € de plus valent-ils le coup ? »).",
            "trace": state["trace"] + ["intention pas encore prise en charge"],
        }

    g = StateGraph(AgentState)
    nodes = {
        "start": start,
        "understand": understand,
        "supervisor": supervisor,
        "assemble": assemble,
        "clarify": clarify,
        "compute": compute,
        "advise": advise,
        "compare": compare,
        "explain_decision": explain_decision,
        "data": data,
        "confirm_export": confirm_export,
        "too_long": too_long,
        "write": write,
        "refuse": refuse,
        "not_yet": not_yet,
    }
    for name, fn in nodes.items():
        g.add_node(name, fn)
    g.add_edge(START, "start")
    g.add_conditional_edges("start", lambda s: s["route"], ["understand", "too_long"])
    g.add_edge("understand", "supervisor")
    g.add_conditional_edges(
        "supervisor", lambda s: s["route"], ["assemble", "compare", "data", "refuse", "not_yet"]
    )
    g.add_conditional_edges(
        "assemble",
        lambda s: s["route"],
        ["clarify", "compute", "advise", "explain_decision", "confirm_export"],
    )
    g.add_conditional_edges(
        "explain_decision", lambda s: s["route"], {"write": "write", "end": END}
    )
    g.add_edge("data", "write")
    g.add_conditional_edges(
        "clarify", lambda s: s["route"], {"understand": "understand", "end": END}
    )
    g.add_conditional_edges("compute", lambda s: s["route"], {"write": "write", "end": END})
    g.add_conditional_edges("compare", lambda s: s["route"], {"write": "write", "end": END})
    g.add_edge("advise", "write")
    for name in ("write", "refuse", "not_yet", "confirm_export", "too_long"):
        g.add_edge(name, END)
    return g.compile(checkpointer=checkpointer)
