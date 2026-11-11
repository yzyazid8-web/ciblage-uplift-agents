"""Graphe de l'agent (LangGraph)."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from ciblage.agents import faithfulness, prompts, tools
from ciblage.agents.llm import Analyst, Understander, Writer, understand_messages
from ciblage.agents.schema import AgentState, Understanding
from ciblage.agents.verify import verify_plan
from ciblage.serve import advisor, explainer, export, guide
from ciblage.serve.planner import Planner, Scenario

REQUIRED = {
    "budget": "le budget total (en €)",
    "cost_homme": "le coût d'un e-mail homme (en €)",
    "cost_femme": "le coût d'un e-mail femme (en €)",
}
MAX_CLARIFY = 3
MAX_ANALYST_STEPS = 5  # décisions du LLM à outils par tour (quota : environ 1 appel par étape)
MAX_TOOL_CALLS = 6  # appels d'outils par décision

MONEY_WORDS = re.compile(r"€|\beuros?\b|\bbudgets?\b", re.IGNORECASE)
NOT_A_BUDGET_AFTER = re.compile(
    r"\s*(e-?mails?|mails?|envois?|clients?|visites?|%|pts?|points?)", re.IGNORECASE
)
NOT_A_BUDGET_BEFORE = re.compile(r"(client|n°|numéro)\s*$", re.IGNORECASE)
MAX_PLAUSIBLE_COST = 2.0  # € par e-mail : au-delà, on soupçonne une erreur d'unité
MAX_MESSAGE_CHARS = 2_000  # protège le quota : un message plus long n'est pas envoyé au LLM
EXCLUDED_WORDS = re.compile(
    r"(écart|exclu|non[- ]contact|pas contact|laiss\w* de côté)", re.IGNORECASE
)
# éléments de l'écran, repérés dans la question si le LLM ne les a pas extraits
ELEMENTS = {
    "graphique_segments": re.compile(
        r"(graph\w*\s*(n°\s*|numéro\s*)?2\b|(deuxième|second) graph|qui re[çc]oi\w* quoi"
        r"|d[ée]j[àa] achet|d[ée]tail par segment|tableau (par|des) segments?)",
        re.IGNORECASE,
    ),
    "graphique_taux_de_visite": re.compile(
        r"(graph\w*\s*(n°\s*|numéro\s*)?1\b|premier graph|taux de visite mesur)", re.IGNORECASE
    ),
    "indicateurs": re.compile(r"(indicateurs?|kpi|cases? du haut|chiffres du haut)", re.IGNORECASE),
}
APP_WORDS = re.compile(
    r"(marimo|\bappli\w*|tableau de bord|dashboard|\bcet outil|\bce site|cette page|interface"
    r"|comment (ça|ca) marche|comment l'utiliser)",
    re.IGNORECASE,
)
CAUSAL_WORDS = re.compile(
    r"causal\w*|corr[ée]lation\w*|a/?b[- ]?test|randomis\w*|confond\w*|confusion|observationnel\w*"
    r"|sans (?:essai|a/?b|test)",
    re.IGNORECASE,
)
MARKDOWN = re.compile(r"\*\*|__")  # gras markdown, retiré malgré la consigne si le LLM en met
NEXT_STEP = {"conseil": "advise", "expliquer": "explain_decision", "exporter": "confirm_export"}

YES_WORDS = {
    "oui", "o", "ok", "okay", "yes", "y", "vas-y", "go", "confirme", "confirmé", "je",
    "d'accord", "c'est", "bon", "valide", "validé", "parfait", "exporte", "merci", "allez",
}  # fmt: skip


def is_yes(reply: str) -> bool:
    words = re.findall(r"[\w'’-]+", (reply or "").lower().replace("’", "'"))
    return bool(words) and all(w in YES_WORDS for w in words)


SCENARIO_FIELDS = {f for f in Scenario.__dataclass_fields__}

QUOTA_MESSAGE = (
    "Le quota gratuit du service de langage est atteint pour le moment. Le formulaire et les "
    "graphiques restent utilisables ; réessaie l'agent dans quelques heures."
)


def is_quota_error(err: BaseException) -> bool:
    """Limite de débit ou quota du fournisseur de LLM (et non une erreur de programmation)."""
    text = f"{type(err).__name__} {err}".lower()
    return "ratelimit" in text or "rate limit" in text or "429" in text


REFUSAL = (
    "Je suis un copilote de campagnes : je calcule qui contacter, avec quel e-mail et pour quel "
    "budget, j'explique les résultats affichés et je réponds aux questions sur cette application. "
    "Je ne peux pas traiter cette demande."
)


@dataclass
class Deps:
    planner: Planner
    understand: Understander
    write: Writer
    exports_dir: Path = export.EXPORTS_DIR
    analyze: Analyst | None = (
        None  # agent à outils ; absent = analyses en plusieurs étapes indisponibles
    )


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
    if kind == "application":
        return (
            f"{facts['role']} {facts['idee_cle']} Le panneau de droite : "
            f"{facts['ecran']['panneau_de_droite']}. À gauche : "
            + " ; ".join(facts["ecran"]["zone_de_gauche_de_haut_en_bas"])
            + "."
        )
    if kind == "plan":
        return template_answer(facts)
    if kind.startswith(("conseil", "comparaison")):
        return template_advice(kind, facts)
    return generic_template(facts)


# provenance : le code sait quels calculs ont produit les chiffres (pas le LLM, qui l'inventerait)
SOURCES = {
    "plan": "plan optimisé sous budget, vérifié sur le jeu de test (IC 95 %)",
    "conseil_budget_delta": "analyse de sensibilité au budget (prix marginal, saturation)",
    "conseil_budget_courbe": "courbe budget → visites du plan",
    "conseil_cout": "seuil de rentabilité de l'e-mail",
    "comparaison": "comparaison des scénarios déjà calculés",
    "explication_client": "décision du plan pour ce client (uplift prédit, seuil de rentabilité)",
    "explication_segment": "répartition du plan par segment",
    "explication_segments": "répartition du plan par segment",
    "explication_ecartes": "profil des clients écartés par le plan",
    "explication_indicateurs": "indicateurs du plan affiché",
    "explication_politiques": "taux de visite mesurés des politiques (IC 95 %)",
    "explication_resultats": "résultats du plan affiché",
    "donnees": "portrait du portefeuille (données de la campagne)",
    "causalite": "inférence causale (essai randomisé) et analyse de robustesse (observationnel)",
}


def provenance(kind: str, facts: dict[str, Any]) -> str:
    """Ligne « Source » déterministe, annexée à la réponse pour rendre les chiffres auditables."""
    if kind == "analyse":
        outils = facts.get("resultats_des_outils") or []
        noms = list(dict.fromkeys(r.get("outil", "") for r in outils if r.get("outil")))
        propre = ", ".join(n.replace("_", " ") for n in noms)
        return f"Sources : outils — {propre}." if propre else ""
    label = SOURCES.get(kind)
    return f"Source : {label}." if label else ""


def _with_source(text: str, kind: str, facts: dict[str, Any]) -> str:
    src = provenance(kind, facts)
    return f"{text}\n\n{src}" if src else text


# limite d'envois citée sans dire à quel e-mail elle s'applique (audit I4)
_LIMIT = re.compile(
    r"(?:limit\w*|max(?:imum)?\b|au plus|pas plus de|plafon\w*|envoy\w*[^\d.,;]{0,15}plus de)"
    r"\s*(?:à|de|:)?\s*"
    r"(\d{1,3}(?:[   ]\d{3})+|\d+)",
    re.IGNORECASE,
)
_ASSIGNED_AFTER = re.compile(
    r"\s*(?:e-?mails?|envois?|mails?)?\s*(?:de l'e-?mail|d'e-?mails?|pour l'e-?mail|pour les e-?mails"
    r"|e-?mails?)?\s*(homme|femme)\b",
    re.IGNORECASE,
)
_ASSIGNED_BEFORE = re.compile(r"(homme|femme)\s*[:,]?\s*$", re.IGNORECASE)
_MONEY_AFTER = re.compile(r"\s*(€|euros?\b)", re.IGNORECASE)


def unassigned_limit(text: str) -> int | None:
    """Nombre d'envois maximal cité sans préciser l'e-mail (« limite à 3 000 envois »), sinon None."""
    for m in _LIMIT.finditer(text or ""):
        after = text[m.end() : m.end() + 45]
        before = text[max(0, m.start() - 30) : m.start()]
        if _MONEY_AFTER.match(after):  # « budget max 1 000 € » : un budget, pas une limite d'envois
            continue
        if _ASSIGNED_AFTER.match(after) or _ASSIGNED_BEFORE.search(before):
            continue
        return int(re.sub(r"\D", "", m.group(1)))
    return None


def limit_choice(reply: str) -> str | None:
    """Réponse à « à quel e-mail s'applique la limite ? » : homme, femme, les_deux, aucune ou None."""
    text = (reply or "").lower()
    if re.search(r"\baucune\b|sans limite|pas de limite|ignore", text):
        return "aucune"
    both = re.search(r"chacun|les deux|tous les deux|deux e-?mails", text)
    homme, femme = "homme" in text, "femme" in text
    if both or (homme and femme):
        return "les_deux"
    return "homme" if homme else "femme" if femme else None


def plausibility_issues(draft: dict[str, Any]) -> list[str]:
    """Controle de vraisemblance deterministe (ne dépend pas du LLM)."""
    issues = []
    for key, name in (("cost_homme", "homme"), ("cost_femme", "femme")):
        cost = draft.get(key)
        if cost is not None and cost > MAX_PLAUSIBLE_COST:
            issues.append(
                f"Le coût de {cost:g} € par e-mail {name} semble élevé : s'agit-il de centimes, ou d'un "
                "prix pour plusieurs e-mails ? Indique le coût d'UN e-mail en euros."
            )
    return issues


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


def correct_intent(
    understanding: dict[str, Any], question: str, has_scenario: bool
) -> tuple[dict[str, Any], list[str]]:
    """Règles deterministes qui corrigent des erreurs du LLM observées en test réel."""
    u, notes = dict(understanding), []
    if u["intent"] == "hors_sujet" and _has_campaign_params(u):
        # des paramètres de campagne ont été extraits : le message n'est pas hors sujet
        u["intent"] = "modifier" if has_scenario else "plan"
        notes.append(f"routeur : « hors sujet » corrigé en « {u['intent']} » (paramètres présents)")
    if u["intent"] == "donnees" and EXCLUDED_WORDS.search(question or ""):
        # « qui sont les clients écartés ? » porte sur le PLAN, pas sur le portefeuille
        u["intent"], u["about_excluded"] = "expliquer", True
        notes.append("routeur : question sur les clients écartés -> explication du plan")
    if not u.get("element") and u["intent"] in (
        "expliquer",
        "application",
        "hors_sujet",
        "donnees",
    ):
        element = next((k for k, rx in ELEMENTS.items() if rx.search(question or "")), None)
        if element:
            u["element"] = element
            notes.append(f"routeur : élément de l'écran repéré ({element})")
    if u.get("element") and u["intent"] in ("expliquer", "application", "hors_sujet", "donnees"):
        # interpréter un résultat affiché demande le plan à l'écran ; sans plan, le guide suffit
        target = "expliquer" if has_scenario else "application"
        if u["intent"] != target:
            u["intent"] = target
            notes.append(f"routeur : question sur l'écran -> {target}")
    if u["intent"] in ("expliquer", "application") and u.get("ambiguities"):
        # une explication n'oblige jamais à deviner (sans précision : tout l'écran) ; vu en test
        # réel : « Explique-moi tous les résultats » renvoyait une question inutile
        u["ambiguities"] = []
        notes.append("routeur : ambiguïté ignorée (demande d'explication)")
    amounts = budget_amounts(question)
    alternatives = re.search(r"compar|\bou\b|\bvs\b|versus", question or "", re.IGNORECASE)
    several = len(amounts) >= 3 or (len(amounts) >= 2 and alternatives is not None)
    if u["intent"] in ("plan", "modifier", "comparer", "conseil") and several:
        # plusieurs budgets dans un même message : le workflow n'en traite qu'un
        u["intent"] = "analyse"
        notes.append(f"routeur : {len(amounts)} budgets cités -> analyse en plusieurs étapes")
    if u["intent"] == "hors_sujet" and APP_WORDS.search(question or ""):
        u["intent"] = "application"
        notes.append("routeur : « hors sujet » corrigé en « application » (question sur l'outil)")
    if u["intent"] in ("hors_sujet", "application", "expliquer", "donnees") and CAUSAL_WORDS.search(
        question or ""
    ):
        u["intent"] = "causalite"
        notes.append("routeur : question sur la causalité -> causalite")
    return u, notes


def budget_amounts(question: str) -> set[float]:
    """Budgets cités dans la question (« compare 500, 700 et 1 000 € » -> {500, 700, 1000})."""
    text = question or ""
    if not MONEY_WORDS.search(text):
        return set()
    out = set()
    for raw, value, _ in faithfulness.parse_numbers(text):
        pos = text.find(raw)
        after, before = text[pos + len(raw) :], text[:pos]
        if (
            value >= 10
            and not NOT_A_BUDGET_AFTER.match(after)
            and not NOT_A_BUDGET_BEFORE.search(before)
        ):
            out.add(value)
    return out


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
            "analyst_messages": [],
            "analyst_results": [],
            "analyst_steps": 0,
            "draft_answer": None,
        }

    def understand(state: AgentState) -> dict[str, Any]:
        msgs = understand_messages(state["user_input"], _context(state))
        try:
            u = deps.understand(msgs)
        except Exception as err:  # réponse invalide du LLM, quota, réseau : on ne plante pas
            return {
                "understanding": {"intent": "echec"},
                "answer": QUOTA_MESSAGE
                if is_quota_error(err)
                else "Je n'ai pas réussi à analyser ta demande (service de langage indisponible "
                "ou réponse invalide). Peux-tu reformuler, ou réessayer dans un instant ?",
                "llm_calls": state.get("llm_calls", 0) + 1,
                "trace": state.get("trace", []) + [f"compréhension : échec ({type(err).__name__})"],
            }
        return {
            "understanding": u.model_dump(),
            "llm_calls": state.get("llm_calls", 0) + 1,
            "trace": state.get("trace", []) + [f"compréhension : {u.intent}"],
        }

    def router(state: AgentState) -> dict[str, Any]:
        understanding = dict(state["understanding"])
        if state.get("pending_intent"):
            understanding["intent"] = state["pending_intent"]
        understanding, notes = correct_intent(
            understanding, state.get("turn_question", ""), has_scenario=bool(state.get("scenario"))
        )
        intent = understanding["intent"]  # les étapes suivantes lisent l'intention corrigée
        trace = list(state["trace"]) + notes
        routes = {
            "echec": "end",
            "plan": "assemble",
            "modifier": "assemble",
            "conseil": "assemble",
            "expliquer": "assemble",
            "exporter": "assemble",
            "comparer": "compare",
            "donnees": "data",
            "application": "app_help",
            "causalite": "causal_help",
            "analyse": "analyst" if deps.analyze is not None else "not_yet",
            "hors_sujet": "refuse",
        }
        route = routes.get(intent, "not_yet")
        return {
            "understanding": understanding,
            "turn_intent": intent,
            "route": route,
            "trace": trace + [f"routeur -> {route}"],
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
        limit = unassigned_limit(state.get("turn_question", ""))
        if (
            limit
            and intent in ("plan", "modifier")
            and draft.get("max_homme") is None
            and draft.get("max_femme") is None
            and not draft.get("limite_resolue")
        ):
            choice = (
                limit_choice(state.get("user_input", "")) if state.get("pending_intent") else None
            )
            if choice in ("homme", "les_deux"):
                draft["max_homme"] = limit
            if choice in ("femme", "les_deux"):
                draft["max_femme"] = limit
            if choice:
                draft["limite_resolue"] = True
            else:
                issues.append(
                    f"Tu indiques une limite de {_fr(limit, 0)} envois : s'applique-t-elle à l'e-mail "
                    "homme, à l'e-mail femme, ou à chacun des deux ? (Une limite sur le total des "
                    "envois n'est pas prise en charge ; réponds « aucune » pour l'ignorer.)"
                )
        if not missing and not issues:
            issues += _scenario(draft).errors() + plausibility_issues(draft)
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
        plan = deps.planner.plan(sc)  # même mesure que l'écran (mêmes intervalles de confiance)
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
        # le guide de l'application cite quelques nombres dans ses textes (exemples de questions)
        extra = faithfulness.numbers_in_text(facts) if kind == "application" else []
        msgs = [
            ("system", prompts.PROMPTS[kind]),
            ("user", prompts.QUESTION.format(question=state.get("turn_question", ""))),
            ("user", "DOSSIER DE FAITS :\n" + json.dumps(facts, ensure_ascii=False)),
        ]
        calls, trace = state.get("llm_calls", 0), list(state["trace"])
        draft = state.get("draft_answer")  # réponse déjà rédigée par l'analyste : à contrôler
        for attempt in range(2):
            if attempt == 0 and draft:
                text = draft
            else:
                calls += 1
                try:
                    text = deps.write(msgs)
                except Exception as err:  # quota du jour atteint, réseau : texte standard calculé
                    trace.append(f"rédaction : échec ({type(err).__name__})")
                    break
            if not text.strip():
                trace.append("rédaction : réponse vide")
                continue
            report = faithfulness.check(text, facts, extra_allowed=extra)
            if report.ok:
                trace.append(f"rédaction : chiffres vérifiés (essai {attempt + 1})")
                return {
                    "answer": _with_source(MARKDOWN.sub("", text), kind, facts),
                    "llm_calls": calls,
                    "trace": trace,
                    "draft_answer": None,
                }
            fixes = []
            if report.unsupported:
                trace.append(f"rédaction : chiffres non vérifiés {report.unsupported}")
                fixes.append(prompts.REGENERATE.format(numbers=", ".join(report.unsupported)))
            if report.verdict:
                trace.append("rédaction : verdict contraire aux mesures")
                fixes.append(
                    prompts.REGENERATE_VERDICT.format(sentences=" / ".join(report.verdict))
                )
            msgs = msgs + [("assistant", text), ("user", "\n\n".join(fixes))]
        trace.append("rédaction : texte standard (repli sûr)")
        return {
            "answer": _with_source(fallback(kind, facts), kind, facts),
            "llm_calls": calls,
            "trace": trace,
            "draft_answer": None,
        }

    def analyst(state: AgentState) -> dict[str, Any]:
        """Agent à outils : le LLM décide lui-même quels outils appeler, ou rédige la réponse."""
        assert deps.analyze is not None
        messages = list(state.get("analyst_messages") or [])
        if not messages:
            scenario = state.get("scenario")
            messages = [
                {
                    "role": "system",
                    "content": prompts.ANALYST.replace("{max_steps}", str(MAX_ANALYST_STEPS)),
                },
                {
                    "role": "system",
                    "content": "Scénario affiché : "
                    + (json.dumps(scenario, ensure_ascii=False) if scenario else "aucun"),
                },
                {
                    "role": "user",
                    "content": prompts.QUESTION.format(question=state.get("turn_question", "")),
                },
            ]
        steps, calls = state.get("analyst_steps", 0) + 1, state.get("llm_calls", 0) + 1
        try:
            reply = deps.analyze(messages)
        except Exception as err:  # quota, réseau, appel d'outil mal formé : réponse sans l'analyste
            if not state.get("analyst_results"):  # aucun résultat à présenter
                return {
                    "llm_calls": calls,
                    "analyst_steps": steps,
                    "answer": QUOTA_MESSAGE
                    if is_quota_error(err)
                    else "L'analyse n'a pas pu aboutir (service de langage indisponible). "
                    "Réessaie dans un instant, ou pose une question plus simple.",
                    "route": "end",
                    "trace": state["trace"] + [f"analyste : échec ({type(err).__name__})"],
                }
            return {
                "llm_calls": calls,
                "analyst_steps": steps,
                "route": "write",
                "trace": state["trace"] + [f"analyste : échec ({type(err).__name__})"],
                **_analysis_facts(state),
            }
        tool_calls = reply["tool_calls"][:MAX_TOOL_CALLS]
        if tool_calls and steps < MAX_ANALYST_STEPS:
            messages.append(
                {"role": "assistant", "content": reply["content"], "tool_calls": tool_calls}
            )
            shown = ", ".join(tools.describe_call(c["name"], c["args"]) for c in tool_calls)
            return {
                "analyst_messages": messages,
                "analyst_steps": steps,
                "llm_calls": calls,
                "route": "run_tools",
                "trace": state["trace"] + [f"analyste : étape {steps} -> {shown}"],
            }
        # réponse finale (ou plus d'étapes disponibles) : elle passe par les contrôles habituels
        note = "réponse proposée" if not tool_calls else "limite d'étapes atteinte"
        return {
            "analyst_messages": messages,
            "analyst_steps": steps,
            "llm_calls": calls,
            "draft_answer": reply["content"] if not tool_calls else None,
            "route": "write",
            "trace": state["trace"] + [f"analyste : {note}"],
            **_analysis_facts(state),
        }

    def run_tools(state: AgentState) -> dict[str, Any]:
        """Exécute les outils demandés par l'analyste ; les résultats lui sont renvoyés."""
        messages = list(state["analyst_messages"])
        results = list(state.get("analyst_results") or [])
        done = []
        for call in messages[-1]["tool_calls"]:
            result = tools.run_tool(deps.planner, state.get("scenario"), call["name"], call["args"])
            results.append({"outil": call["name"], "arguments": call["args"], "resultat": result})
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": json.dumps(result, ensure_ascii=False, separators=(",", ":")),
                }
            )
            done.append(f"{call['name']} {'(erreur)' if 'erreur' in result else 'ok'}")
        return {
            "analyst_messages": messages,
            "analyst_results": results,
            "trace": state["trace"] + ["outils : " + ", ".join(done)],
        }

    def _analysis_facts(state: AgentState) -> dict[str, Any]:
        """Dossier de faits de l'analyse : scénario affiché et résultats de tous les outils."""
        return {
            "facts": {
                "scenario_affiche": state.get("scenario"),
                "resultats_des_outils": state.get("analyst_results") or [],
            },
            "facts_kind": "analyse",
        }

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
            elif req.get("element") == "indicateurs":
                facts = explainer.indicators_explanation(deps.planner, sc)
                kind = "explication_indicateurs"
            elif req.get("element") == "graphique_taux_de_visite":
                facts = explainer.policies_explanation(deps.planner, sc)
                kind = "explication_politiques"
            elif req.get("element") == "graphique_segments":
                facts = explainer.segments_explanation(deps.planner, sc)
                kind = "explication_segments"
            else:  # « explique-moi les résultats » : tout l'écran, de haut en bas
                facts = explainer.results_overview(deps.planner, sc)
                kind = "explication_resultats"
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

    def app_help(state: AgentState) -> dict[str, Any]:
        """Questions sur l'outil lui-même : le guide de l'application sert de dossier de faits."""
        return {
            "facts": guide.app_guide(deps.planner),
            "facts_kind": "application",
            "trace": state["trace"] + ["aide : guide de l'application"],
        }

    def causal_help(state: AgentState) -> dict[str, Any]:
        """Nature causale des résultats (et robustesse sans A/B test) : guide causal pré-calculé."""
        return {
            "facts": guide.causal_guide(deps.planner),
            "facts_kind": "causalite",
            "trace": state["trace"] + ["causalité : fondement de l'inférence"],
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
        if not is_yes(reply):
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
        "router": router,
        "assemble": assemble,
        "clarify": clarify,
        "compute": compute,
        "advise": advise,
        "compare": compare,
        "explain_decision": explain_decision,
        "data": data,
        "app_help": app_help,
        "causal_help": causal_help,
        "analyst": analyst,
        "run_tools": run_tools,
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
    g.add_edge("understand", "router")
    g.add_conditional_edges(
        "router",
        lambda s: s["route"],
        {
            "assemble": "assemble",
            "compare": "compare",
            "data": "data",
            "app_help": "app_help",
            "causal_help": "causal_help",
            "analyst": "analyst",
            "refuse": "refuse",
            "not_yet": "not_yet",
            "end": END,  # la compréhension a échoué : la réponse d'excuse est déjà prête
        },
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
    g.add_edge("app_help", "write")
    g.add_edge("causal_help", "write")
    g.add_conditional_edges(
        "analyst", lambda s: s["route"], {"run_tools": "run_tools", "write": "write", "end": END}
    )
    g.add_edge("run_tools", "analyst")
    g.add_conditional_edges(
        "clarify", lambda s: s["route"], {"understand": "understand", "end": END}
    )
    g.add_conditional_edges("compute", lambda s: s["route"], {"write": "write", "end": END})
    g.add_conditional_edges("compare", lambda s: s["route"], {"write": "write", "end": END})
    g.add_edge("advise", "write")
    for name in ("write", "refuse", "not_yet", "confirm_export", "too_long"):
        g.add_edge(name, END)
    return g.compile(checkpointer=checkpointer)
