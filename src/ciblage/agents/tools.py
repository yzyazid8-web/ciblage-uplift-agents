"""Outils de l'agent analyste : fonctions déterministes, en lecture seule, appelées par le LLM."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from ciblage.serve import advisor, explainer
from ciblage.serve.planner import Planner, Scenario

MAX_BUDGETS = 6
REQUIRED = {
    "budget": "le budget",
    "cost_homme": "le coût d'un e-mail homme",
    "cost_femme": "le coût d'un e-mail femme",
}
MAX_PLAUSIBLE_COST = 2.0


class ScenarioArgs(BaseModel):
    """Paramètres facultatifs : ceux qui ne sont pas donnés viennent du scénario affiché."""

    budget: float | None = Field(None, description="Budget total en euros")
    cout_homme: float | None = Field(None, description="Coût d'UN e-mail homme en euros")
    cout_femme: float | None = Field(None, description="Coût d'UN e-mail femme en euros")
    max_homme: int | None = Field(None, description="Nombre maximal d'e-mails homme")
    max_femme: int | None = Field(None, description="Nombre maximal d'e-mails femme")
    objectif: Literal["visites", "profit"] | None = Field(None, description="Objectif du plan")
    valeur_visite: float | None = Field(
        None, description="Euros rapportés par visite (objectif profit)"
    )


class PlansArgs(ScenarioArgs):
    budgets: list[float] = Field(
        description=f"Un ou plusieurs budgets à calculer (de 1 à {MAX_BUDGETS}), en euros"
    )


class BudgetChangeArgs(ScenarioArgs):
    variation_eur: float = Field(
        description="Variation du budget en euros (positive ou négative), ex. 100 ou -50"
    )


class CostArgs(ScenarioArgs):
    email: Literal["homme", "femme"] = Field(description="E-mail dont on cherche le coût limite")


class ClientArgs(ScenarioArgs):
    client_id: int = Field(description="Numéro du client dans le portefeuille")


class PortfolioArgs(BaseModel):
    segment: Literal["homme seulement", "femme seulement", "homme et femme"] | None = Field(
        None, description="Segment à décrire ; absent = tout le portefeuille"
    )


def _scenario(base: dict[str, Any] | None, args: ScenarioArgs) -> Scenario:
    """Scénario affiché, modifié par les arguments du LLM ; ValueError si incomplet ou invalide."""
    merged = dict(base or {})
    names = {
        "budget": "budget",
        "cout_homme": "cost_homme",
        "cout_femme": "cost_femme",
        "max_homme": "max_homme",
        "max_femme": "max_femme",
        "objectif": "objective",
        "valeur_visite": "value_per_visit",
    }
    for arg, field in names.items():
        value = getattr(args, arg)
        if value is not None:
            merged[field] = value
    missing = [label for key, label in REQUIRED.items() if merged.get(key) is None]
    if missing:
        raise ValueError(f"Il manque {', '.join(missing)} (aucun scénario affiché).")
    sc = Scenario(**{k: v for k, v in merged.items() if k in Scenario.__dataclass_fields__})
    problems = sc.errors() + [
        f"coût de {c:g} € par e-mail peu plausible (centimes ?)"
        for c in (sc.cost_homme, sc.cost_femme)
        if c > MAX_PLAUSIBLE_COST
    ]
    if problems:
        raise ValueError(" ".join(problems))
    return sc


def _plan_row(planner: Planner, sc: Scenario) -> dict[str, Any]:
    plan = planner.plan(sc)
    s = plan.summary
    g = plan.comparison.set_index("politique").loc["plan optimisé"]
    return {
        "budget": sc.budget,
        "clients_contactes": s["clients_contactes"],
        "email_homme": s["email_homme"],
        "email_femme": s["email_femme"],
        "depense_eur": round(s["depense"], 2),
        "visites_en_plus_predites": round(s["visites_en_plus_predites"]),
        "cout_par_visite_gagnee_eur": round(s["cout_par_visite_en_plus"], 2),
        # rendement MOYEN du plan (toutes les visites / toute la dépense), à ne pas confondre
        # avec la valeur du dernier euro (marginale)
        "visites_par_euro_depense": round(s["visites_en_plus_predites"] / s["depense"], 3)
        if s["depense"] > 0
        else 0.0,
        "visites_du_dernier_euro": round(s["prix_marginal_budget"], 3),
        "gain_mesure_vs_classique_points": round(float(g["gain vs classique (pts)"]), 2),
        "gain_ic95_bas": round(float(g["gain_bas"]), 2),
        "gain_ic95_haut": round(float(g["gain_haut"]), 2),
        "gain_demontre": bool(g["gain_bas"] > 0),
        "par_segment": [
            {
                "segment": r["segment"],
                "email_homme": r["e-mail homme"],
                "email_femme": r["e-mail femme"],
                "rien": r["rien"],
            }
            for r in plan.by_segment.to_dict(orient="records")
        ],
    }


def calculer_plans(planner: Planner, base: dict | None, args: PlansArgs) -> dict[str, Any]:
    budgets = args.budgets[:MAX_BUDGETS]
    if not budgets:
        raise ValueError("Donne au moins un budget.")
    rows = []
    for b in budgets:
        sc = _scenario(base, args.model_copy(update={"budget": b}))
        rows.append(_plan_row(planner, sc))
    return {
        "couts_utilises": {"email_homme": sc.cost_homme, "email_femme": sc.cost_femme},
        "objectif": sc.objective,
        "plans": rows,
        "niveau_de_confiance_pct": 95,
    }


def effet_variation_budget(planner: Planner, base: dict | None, args: BudgetChangeArgs) -> dict:
    return advisor.marginal_value(planner, _scenario(base, args), args.variation_eur)


def courbe_budget(planner: Planner, base: dict | None, args: ScenarioArgs) -> dict:
    return advisor.budget_curve(planner, _scenario(base, args))


def seuil_de_cout(planner: Planner, base: dict | None, args: CostArgs) -> dict:
    return advisor.cost_threshold(planner, _scenario(base, args), args.email)


def expliquer_segments(planner: Planner, base: dict | None, args: ScenarioArgs) -> dict:
    facts = explainer.segments_explanation(planner, _scenario(base, args))
    # version courte pour le LLM : les textes répétés par segment sont retirés
    for seg in facts["segments"]:
        seg["verification_sur_la_campagne"].pop("role", None)
    facts.pop("element_affiche", None)
    return facts


def expliquer_client(planner: Planner, base: dict | None, args: ClientArgs) -> dict:
    return explainer.client_explanation(planner, _scenario(base, args), args.client_id)


def profil_des_ecartes(planner: Planner, base: dict | None, args: ScenarioArgs) -> dict:
    return explainer.excluded_profile(planner, _scenario(base, args))


def comparer_politiques(planner: Planner, base: dict | None, args: ScenarioArgs) -> dict:
    return explainer.policies_explanation(planner, _scenario(base, args))


def portrait_portefeuille(planner: Planner, base: dict | None, args: PortfolioArgs) -> dict:
    return explainer.portfolio_overview(planner, args.segment)


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    args: type[BaseModel]
    run: Callable[[Planner, dict | None, Any], dict[str, Any]]

    def schema(self) -> dict[str, Any]:
        """Définition au format « function calling » (compatible OpenAI / Groq)."""
        params = self.args.model_json_schema()
        params.pop("title", None)
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": params},
        }


TOOLS = {
    t.name: t
    for t in (
        Tool(
            "calculer_plans",
            "Calcule le plan optimal pour un ou plusieurs budgets (mêmes coûts) : clients "
            "contactés, e-mails, dépense, visites en plus prédites, visites du dernier euro, gain "
            "MESURÉ face au ciblage classique avec IC 95 %, répartition par segment.",
            PlansArgs,
            calculer_plans,
        ),
        Tool(
            "effet_variation_budget",
            "Effet d'une hausse ou d'une baisse du budget sur les visites prédites, et budget de "
            "saturation (au-delà, un euro de plus ne rapporte plus rien).",
            BudgetChangeArgs,
            effet_variation_budget,
        ),
        Tool(
            "courbe_budget",
            "Visites prédites et visites du dernier euro pour une grille de budgets, de 0 à la "
            "saturation (rendements décroissants).",
            ScenarioArgs,
            courbe_budget,
        ),
        Tool(
            "seuil_de_cout",
            "Coût d'un e-mail au-delà duquel le plan ne l'envoie plus, avec ce budget.",
            CostArgs,
            seuil_de_cout,
        ),
        Tool(
            "expliquer_segments",
            "Pourquoi chaque segment (selon ce que le client a déjà acheté) reçoit tel e-mail : "
            "répartition, raisons comptées, uplift prédit, vérification mesurée sur la campagne.",
            ScenarioArgs,
            expliquer_segments,
        ),
        Tool(
            "expliquer_client",
            "Pourquoi un client donné reçoit (ou non) un e-mail : profil, uplift prédit, "
            "rendement, gain net, raison de la décision.",
            ClientArgs,
            expliquer_client,
        ),
        Tool(
            "profil_des_ecartes",
            "Qui sont les clients écartés par le plan, comparés aux contactés, et pourquoi.",
            ScenarioArgs,
            profil_des_ecartes,
        ),
        Tool(
            "comparer_politiques",
            "Taux de visite MESURÉ de quatre politiques au même budget (personne, ciblage "
            "classique, plan optimisé, tout le monde), avec IC 95 %.",
            ScenarioArgs,
            comparer_politiques,
        ),
        Tool(
            "portrait_portefeuille",
            "Portrait des clients (tout le portefeuille ou un segment) : effectifs, canaux, "
            "dépense, ancienneté, taux de visite sans e-mail, effet mesuré des e-mails.",
            PortfolioArgs,
            portrait_portefeuille,
        ),
    )
}


def schemas() -> list[dict[str, Any]]:
    return [t.schema() for t in TOOLS.values()]


def run_tool(
    planner: Planner, base: dict[str, Any] | None, name: str, arguments: dict[str, Any]
) -> dict[str, Any]:
    """Exécute un appel d'outil ; toute erreur devient un résultat lisible par le LLM."""
    tool = TOOLS.get(name)
    if tool is None:
        return {"erreur": f"Outil inconnu : {name}. Outils disponibles : {', '.join(TOOLS)}."}
    try:
        args = tool.args.model_validate(arguments or {})
        return tool.run(planner, base, args)
    except ValidationError as err:
        problems = "; ".join(f"{'.'.join(map(str, e['loc']))} : {e['msg']}" for e in err.errors())
        return {"erreur": f"Arguments invalides pour {name} ({problems})."}
    except ValueError as err:
        return {"erreur": str(err)}


def describe_call(name: str, arguments: dict[str, Any]) -> str:
    """Appel lisible pour les étapes affichées (« calculer_plans(budgets=[500, 700]) »)."""
    shown = ", ".join(
        f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in (arguments or {}).items()
    )
    return f"{name}({shown})"
