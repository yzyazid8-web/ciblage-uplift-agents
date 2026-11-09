"""Conseil : analyses « et si » calculées (jamais estimées par le LLM)."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Literal

import numpy as np

from ciblage.serve.planner import Planner, Scenario


def _visits(planner: Planner, sc: Scenario) -> tuple[float, dict[str, float]]:
    s = planner.plan(sc, compare=False).summary
    return s["visites_en_plus_predites"], s


def saturation_budget(planner: Planner, sc: Scenario) -> float:
    """Dépense du plan sans limite de budget : au-delà, un euro de plus ne rapporte plus rien."""
    _, s = _visits(planner, replace(sc, budget=1e12))
    return float(s["depense"])


def marginal_value(planner: Planner, sc: Scenario, delta: float) -> dict[str, Any]:
    """Effet d'un changement de budget de `delta` euros (positif ou négatif)."""
    new_budget = max(0.0, sc.budget + delta)
    v0, s0 = _visits(planner, sc)
    v1, s1 = _visits(planner, replace(sc, budget=new_budget))
    real_delta = new_budget - sc.budget
    return {
        "budget_actuel": sc.budget,
        "budget_envisage": new_budget,
        "variation_budget": real_delta,
        "visites_actuelles": round(v0),
        "visites_envisagees": round(v1),
        "visites_en_plus_ou_moins": round(v1 - v0, 1),
        "visites_par_euro_de_variation": round((v1 - v0) / real_delta, 3) if real_delta else 0.0,
        "prix_marginal_actuel_visites_par_euro": round(s0["prix_marginal_budget"], 3),
        "clients_contactes_actuels": s0["clients_contactes"],
        "clients_contactes_envisages": s1["clients_contactes"],
        "budget_de_saturation": round(saturation_budget(planner, sc), 2),
    }


def budget_curve(planner: Planner, sc: Scenario, points: int = 8) -> dict[str, Any]:
    """Visites prédites et prix marginal pour une grille de budgets, de 0 à la saturation."""
    sat = saturation_budget(planner, sc)
    # points arrondis à la dizaine, plus le budget actuel et le budget de saturation EXACTS
    steps = {round(float(b), -1) for b in np.linspace(0, sat, points)[:-1]}
    grid = sorted(steps | {round(sat, 2), round(sc.budget, 2)})
    rows = []
    for b in grid:
        v, s = _visits(planner, replace(sc, budget=b))
        rows.append(
            {
                "budget": b,
                "visites_en_plus": round(v),
                "visites_par_euro_supplementaire": round(s["prix_marginal_budget"], 3),
            }
        )
    return {"budget_actuel": sc.budget, "budget_de_saturation": round(sat, 2), "courbe": rows}


def cost_threshold(
    planner: Planner, sc: Scenario, email: Literal["homme", "femme"], iters: int = 22
) -> dict[str, Any]:
    """Coût de l'e-mail au-delà duquel l'optimiseur ne l'envoie plus (dichotomie).

    Le nombre d'envois d'un e-mail décroît quand son propre coût augmente (il devient moins
    rentable face à l'autre e-mail et face au budget) : on cherche le premier coût où il tombe à 0.
    """
    field = "cost_homme" if email == "homme" else "cost_femme"
    count_key = f"email_{email}"

    def count(cost: float) -> int:
        changed = replace(sc, cost_homme=cost) if email == "homme" else replace(sc, cost_femme=cost)
        return int(_visits(planner, changed)[1][count_key])

    current = count(getattr(sc, field))
    lo, hi = 0.0, max(1.0, 2 * getattr(sc, field))
    while count(hi) > 0 and hi < 1e4:  # on élargit jusqu'à trouver un coût dissuasif
        lo, hi = hi, hi * 2
    if count(lo) == 0:
        return {
            "email": email,
            "cout_actuel": getattr(sc, field),
            "envois_actuels": current,
            "seuil_cout_au_dela_duquel_plus_aucun_envoi": 0.0,
        }
    for _ in range(iters):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if count(mid) > 0 else (lo, mid)
    return {
        "email": email,
        "cout_actuel": getattr(sc, field),
        "envois_actuels": current,
        "seuil_cout_au_dela_duquel_plus_aucun_envoi": round(hi, 3),
        "envois_juste_sous_le_seuil": count(lo),
    }


def compare_scenarios(history: list[dict[str, Any]], last: int = 5) -> list[dict[str, Any]]:
    """Tableau des derniers scénarios calculés (du plus ancien au plus récent)."""
    rows = []
    for k, h in enumerate(history[-last:], start=max(1, len(history) - last + 1)):
        sc, s, m = h["scenario"], h["summary"], h.get("mesure", {})
        rows.append(
            {
                "scenario": k,
                "budget": sc["budget"],
                "cout_homme": sc["cost_homme"],
                "cout_femme": sc["cost_femme"],
                "clients_contactes": s["clients_contactes"],
                "email_homme": s["email_homme"],
                "email_femme": s["email_femme"],
                "depense": round(s["depense"], 2),
                "visites_en_plus_predites": round(s["visites_en_plus_predites"]),
                "cout_par_visite_gagnee": round(s["cout_par_visite_en_plus"], 2),
                "gain_mesure_vs_classique_points": m.get("gain"),
                "ic95_gain_bas": m.get("ic95_bas"),
                "ic95_gain_haut": m.get("ic95_haut"),
                # un gain n'est démontré (positif ou négatif) que si l'IC ne contient pas 0
                "difference_demontree": None
                if m.get("ic95_bas") is None
                else bool(m["ic95_bas"] > 0 or m["ic95_haut"] < 0),
            }
        )
    return rows
