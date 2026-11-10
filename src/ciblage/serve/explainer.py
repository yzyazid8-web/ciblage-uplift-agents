"""Explications chiffrées des résultats, et portrait du portefeuille (jamais estimés par le LLM)."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd

from ciblage.causal.ate import welch
from ciblage.serve.planner import LABELS, Plan, Planner, Scenario

SEGMENTS = ("homme seulement", "femme seulement", "homme et femme")
CONFIDENCE_PCT = 95

# raisons des décisions (règle de l'optimiseur : gain net = gain prédit - seuil × coût)
NO_GAIN = {
    "visites": "aucun e-mail ne lui fait gagner de visite (uplift prédit négatif ou nul)",
    "profit": "aucun e-mail ne rapporte plus qu'il ne coûte pour lui",
}
BELOW = "gain positif mais rendement sous le seuil de rentabilité"
LIMIT = "la limite d'envois de son meilleur e-mail empêche de le servir"
EDGE = "au niveau du seuil : le budget est épuisé avant lui"
BEST = "c'est l'e-mail au meilleur gain net pour lui"
OTHER_FULL = "son meilleur e-mail est réservé à d'autres clients (limite d'envois)"
EDGE_CHOICE = "départagé au niveau du seuil, en fin de budget"

UNITS = {
    "visites": "visites gagnées par euro dépensé",
    "profit": "euros de profit par euro dépensé",
}
RULE = (
    "Un client est contacté si le rendement de son meilleur e-mail (gain prédit par euro) dépasse "
    "le seuil de rentabilité. Entre les deux e-mails, il reçoit celui au meilleur gain net (gain "
    "prédit moins seuil × coût) : ce n'est pas forcément celui au meilleur rendement."
)
VERIFICATION = (
    "Mesure sur la campagne randomisée : elle vérifie après coup que les prédictions tiennent, "
    "mais n'intervient jamais dans la décision."
)
POLICIES = {
    "personne": "aucun e-mail : le taux de visite naturel du portefeuille",
    "ciblage classique": (
        "e-mail homme aux clients les plus susceptibles de visiter (modèle de réponse classique), "
        "dans le même budget : la référence à battre"
    ),
    "plan optimisé": (
        "le plan du copilote : chaque client reçoit l'e-mail qui change le plus son comportement "
        "par euro (uplift), dans le budget"
    ),
    "tout le monde (e-mail homme)": (
        "e-mail homme à tous les clients, sans tenir compte du budget : un repère, pas une option "
        "dans le budget"
    ),
}


def _plan(planner: Planner, sc: Scenario, compare: bool = False) -> tuple[Plan, float]:
    plan = planner.plan(sc, compare=compare)  # même nombre de rééchantillonnages que l'écran
    return plan, float(plan.summary["prix_marginal_budget"])


def _measured(planner: Planner, mask: np.ndarray, action: int) -> dict[str, float] | None:
    """Uplift MESURÉ (différence de moyennes dans la campagne randomisée) sur un groupe de clients."""
    t, y = planner.treatment[mask], planner.visit[mask]
    yt, yc = y[t == action], y[t == 0]
    if len(yt) < 30 or len(yc) < 30:
        return None
    w = welch(yt, yc)
    return {
        "points": round(100 * w["ate"], 2),
        "ic95_bas": round(100 * w["ci_low"], 2),
        "ic95_haut": round(100 * w["ci_high"], 2),
        "difference_demontree": bool(w["ci_low"] > 0 or w["ci_high"] < 0),
    }


def _costs(sc: Scenario) -> np.ndarray:
    return np.array([sc.cost_homme, sc.cost_femme], dtype=float)


def decision_reasons(planner: Planner, sc: Scenario, plan: Plan, lam: float) -> np.ndarray:
    """Raison de la décision de chaque client, recalculée avec la règle de l'optimiseur.

    Gain net d'un e-mail = gain prédit - seuil x coût (- prix de sa limite d'envois, s'il y en a
    une qui pèse). Un client reçoit l'e-mail au meilleur gain net s'il est positif ; sinon, rien.
    """
    score = planner.problem(sc).score  # gain prédit (visites, ou € de profit) par e-mail
    net0 = score - lam * _costs(sc)[None, :]  # sans les limites d'envois
    net = net0 - plan.action_prices[None, :]  # avec le prix des limites
    pref0 = net0.argmax(axis=1) + 1  # e-mail préféré si les envois étaient illimités
    limited = np.concatenate([[False], plan.action_prices > 1e-9])  # limite qui pèse, par e-mail
    a = plan.assignment
    out = np.empty(len(a), dtype=object)
    none, sent = a == 0, a > 0
    out[none & (score.max(axis=1) <= 0)] = NO_GAIN[sc.objective]
    out[none & (score.max(axis=1) > 0) & (net0.max(axis=1) <= 0)] = BELOW
    rest = none & (net0.max(axis=1) > 0)
    out[rest & (net.max(axis=1) <= 0)] = LIMIT
    out[rest & (net.max(axis=1) > 0)] = EDGE
    out[sent & (a == pref0)] = BEST
    out[sent & (a != pref0) & limited[pref0]] = OTHER_FULL
    out[sent & (a != pref0) & ~limited[pref0]] = EDGE_CHOICE
    return out


def _count(reasons: np.ndarray) -> dict[str, int]:
    return {str(k): int(v) for k, v in Counter(reasons.tolist()).most_common()}


def _rule(sc: Scenario, lam: float) -> dict[str, Any]:
    return {
        "objectif": sc.objective,
        "seuil_de_rentabilite": round(lam, 3),
        "unite_du_seuil_et_des_rendements": UNITS[sc.objective],
        "regle_de_decision": RULE,
    }


def _yield(planner: Planner, sc: Scenario, rows: np.ndarray, action: int) -> float | None:
    cost = _costs(sc)[action - 1]
    if cost <= 0 or not rows.any():
        return None
    return round(float(planner.problem(sc).score[rows, action - 1].mean()) / float(cost), 3)


def _segment_facts(
    planner: Planner, sc: Scenario, plan: Plan, reasons: np.ndarray, segment: str
) -> dict[str, Any]:
    m = (planner.segment == segment).to_numpy()
    a = plan.assignment
    facts: dict[str, Any] = {
        "segment": segment,
        "clients": int(m.sum()),
        "email_homme": int((a[m] == 1).sum()),
        "email_femme": int((a[m] == 2).sum()),
        "rien": int((a[m] == 0).sum()),
        "uplift_predit_moyen_du_segment_points": {
            "email_homme": round(100 * float(planner.uplift[m, 0].mean()), 2),
            "email_femme": round(100 * float(planner.uplift[m, 1].mean()), 2),
        },
    }
    for k, key in (
        (1, "clients_qui_recoivent_l_email_homme"),
        (2, "clients_qui_recoivent_l_email_femme"),
    ):
        rows = m & (a == k)
        if rows.any():  # groupe vide : pas de valeur manquante qui pourrait passer pour une raison
            facts[key] = {
                "clients": int(rows.sum()),
                "uplift_predit_moyen_email_homme_points": round(
                    100 * float(planner.uplift[rows, 0].mean()), 2
                ),
                "uplift_predit_moyen_email_femme_points": round(
                    100 * float(planner.uplift[rows, 1].mean()), 2
                ),
                "rendement_moyen_de_l_email_recu": _yield(planner, sc, rows, k),
                "raisons": _count(reasons[rows]),
            }
    rows = m & (a == 0)
    if rows.any():
        facts["clients_sans_email"] = {
            "clients": int(rows.sum()),
            "uplift_predit_moyen_email_homme_points": round(
                100 * float(planner.uplift[rows, 0].mean()), 2
            ),
            "uplift_predit_moyen_email_femme_points": round(
                100 * float(planner.uplift[rows, 1].mean()), 2
            ),
            "raisons": _count(reasons[rows]),
        }
    facts["verification_sur_la_campagne"] = {
        "role": VERIFICATION,
        "uplift_mesure_email_homme": _measured(planner, m, 1),
        "uplift_mesure_email_femme": _measured(planner, m, 2),
    }
    return facts


def segment_explanation(planner: Planner, sc: Scenario, segment: str) -> dict[str, Any]:
    """Pourquoi tel segment reçoit-il tel e-mail ? Répartition, raisons comptées, vérification."""
    if segment not in SEGMENTS:
        raise ValueError(f"Segment inconnu : {segment}")
    plan, lam = _plan(planner, sc)
    reasons = decision_reasons(planner, sc, plan, lam)
    return {
        **_rule(sc, lam),
        **_segment_facts(planner, sc, plan, reasons, segment),
        "niveau_de_confiance_pct": CONFIDENCE_PCT,
    }


def segments_explanation(planner: Planner, sc: Scenario) -> dict[str, Any]:
    """Graphique 2 (« Qui reçoit quoi ») et son tableau : les trois segments d'un coup."""
    plan, lam = _plan(planner, sc)
    reasons = decision_reasons(planner, sc, plan, lam)
    segments = [_segment_facts(planner, sc, plan, reasons, seg) for seg in SEGMENTS]
    return {
        "element_affiche": "graphique « Qui reçoit quoi, selon ce que le client a déjà acheté » et son tableau",
        **_rule(sc, lam),
        "segments": segments,
        "verification_resumee": _verification_summary(segments),
        "niveau_de_confiance_pct": CONFIDENCE_PCT,
    }


def _verification_summary(segments: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Dans quels segments l'effet mesuré de chaque e-mail est-il démontré ? (listes exhaustives)

    Sans cette synthèse, le LLM généralisait à tort (« confirmé seulement pour... »).
    """
    out: dict[str, list[str]] = {}
    for email in ("homme", "femme"):
        shown: list[str] = []
        not_shown: list[str] = []
        for seg in segments:
            m = seg["verification_sur_la_campagne"][f"uplift_mesure_email_{email}"]
            if m is not None:
                (shown if m["difference_demontree"] else not_shown).append(seg["segment"])
        out[f"effet_mesure_email_{email}_demontre_dans"] = shown
        out[f"effet_mesure_email_{email}_non_demontre_dans"] = not_shown
    return out


def client_explanation(planner: Planner, sc: Scenario, client_id: int) -> dict[str, Any]:
    """Pourquoi ce client reçoit-il (ou non) un e-mail ?"""
    if not 0 <= client_id < planner.n_clients:
        raise ValueError(f"Le numéro de client doit être entre 0 et {planner.n_clients - 1}")
    plan, lam = _plan(planner, sc)
    reason = decision_reasons(planner, sc, plan, lam)[client_id]
    row = planner.clients.iloc[client_id]
    u = planner.uplift[client_id]
    score = planner.problem(sc).score[client_id]
    costs = _costs(sc)
    rendements = {
        name: round(float(score[k]) / float(costs[k]), 3) if costs[k] > 0 else None
        for k, name in enumerate(("homme", "femme"))
    }
    unit = "points" if sc.objective == "visites" else "eur"
    scale = 100 if sc.objective == "visites" else 1
    net = {
        name: round(scale * float(score[k] - lam * costs[k]), 3)
        for k, name in enumerate(("homme", "femme"))
    }
    best_yield = max((v for v in rendements.values() if v is not None), default=0.0)
    return {
        "client": client_id,
        "segment": str(planner.segment.iloc[client_id]),
        "mois_depuis_dernier_achat": int(row["recency"]),
        "depense_annee_passee_eur": round(float(row["history"]), 2),
        "nouveau_client": bool(row["newbie"]),
        "canal": str(row["channel"]),
        "decision": LABELS[int(plan.assignment[client_id])],
        "raison": str(reason),
        **_rule(sc, lam),
        "uplift_predit_homme_points": round(100 * float(u[0]), 2),
        "uplift_predit_femme_points": round(100 * float(u[1]), 2),
        "rendement_homme": rendements["homme"],
        "rendement_femme": rendements["femme"],
        f"gain_net_homme_{unit}": net["homme"],
        f"gain_net_femme_{unit}": net["femme"],
        "meilleur_rendement_au_dessus_du_seuil": bool(best_yield > lam),
    }


def excluded_profile(planner: Planner, sc: Scenario) -> dict[str, Any]:
    """Qui sont les clients écartés, comparés aux clients contactés, et pourquoi ?"""
    plan, lam = _plan(planner, sc)
    reasons = decision_reasons(planner, sc, plan, lam)
    contacted = plan.assignment > 0
    out: dict[str, Any] = _rule(sc, lam)
    for name, mask in (("contactes", contacted), ("ecartes", ~contacted)):
        if not mask.any():
            out[name] = {"clients": 0}
            continue
        clients = planner.clients[mask]
        seg = planner.segment[mask].value_counts(normalize=True)
        out[name] = {
            "clients": int(mask.sum()),
            "uplift_predit_moyen_homme_points": round(
                100 * float(planner.uplift[mask, 0].mean()), 2
            ),
            "uplift_predit_moyen_femme_points": round(
                100 * float(planner.uplift[mask, 1].mean()), 2
            ),
            "part_uplift_negatif_ou_nul_pct": round(
                100 * float((planner.uplift[mask].max(axis=1) <= 0).mean()), 1
            ),
            "mois_depuis_dernier_achat_moyen": round(float(clients["recency"].mean()), 1),
            "depense_annee_passee_mediane_eur": round(float(clients["history"].median()), 2),
            "part_nouveaux_clients_pct": round(100 * float(clients["newbie"].mean()), 1),
            "repartition_segments_pct": {k: round(100 * float(v), 1) for k, v in seg.items()},
        }
    if (~contacted).any():
        out["ecartes"]["raisons"] = _count(reasons[~contacted])
    return out


def _indicator_facts(plan: Plan) -> list[dict[str, Any]]:
    s = plan.summary
    g = plan.comparison.set_index("politique").loc["plan optimisé"]
    visits = plan.scenario.objective == "visites"
    return [
        {
            "indicateur": "Clients contactés",
            "valeur": s["clients_contactes"],
            "email_homme": s["email_homme"],
            "email_femme": s["email_femme"],
            "definition": "nombre de clients qui reçoivent un e-mail dans le plan",
        },
        {
            "indicateur": "Dépense",
            "valeur_eur": round(s["depense"], 2),
            "budget_non_utilise_eur": round(s["budget_inutilise"], 2),
            "definition": "coût total des e-mails envoyés (coûts saisis comme hypothèses)",
        },
        {
            "indicateur": "Visites en plus (prédites)",
            "valeur": round(s["visites_en_plus_predites"]),
            "cout_par_visite_gagnee_eur": round(s["cout_par_visite_en_plus"], 2),
            "definition": (
                "somme des uplifts prédits des clients contactés : ce que le modèle attend du plan, "
                "pas une mesure"
            ),
        },
        {
            "indicateur": "Gain mesuré vs ciblage classique",
            "valeur_points": round(float(g["gain vs classique (pts)"]), 2),
            "ic95_bas": round(float(g["gain_bas"]), 2),
            "ic95_haut": round(float(g["gain_haut"]), 2),
            "gain_demontre": bool(g["gain_bas"] > 0),
            "perte_demontree": bool(g["gain_haut"] < 0),
            "definition": (
                "écart de taux de visite entre le plan et le ciblage classique au même budget, "
                "MESURÉ sur la campagne randomisée (pondération inverse) ; si l'intervalle contient "
                "0, aucune différence n'est démontrée"
            ),
        },
        {
            "indicateur": "Visites par euro en plus" if visits else "Profit par euro en plus",
            "valeur": round(s["prix_marginal_budget"], 3),
            "attention": (
                "c'est ce que rapporte le DERNIER euro (prix marginal), pas le rendement moyen du "
                "plan : le coût moyen est donné par le coût par visite gagnée"
            ),
            "ecart_prouve_a_l_optimum_pct": round(100 * plan.bound_gap, 3),
            "definition": (
                "ce que rapporterait un euro de budget supplémentaire (prix marginal du budget, "
                "relaxation lagrangienne) ; c'est aussi le seuil de rentabilité qu'un e-mail doit "
                "dépasser pour être envoyé. L'écart prouvé borne la distance à la meilleure "
                "allocation possible"
            ),
        },
    ]


def _policy_facts(plan: Plan) -> list[dict[str, Any]]:
    rows = []
    for r in plan.comparison.to_dict(orient="records"):
        row: dict[str, Any] = {
            "politique": r["politique"],
            "description": POLICIES.get(r["politique"], ""),
            "clients_contactes": int(r["contactés"]),
            "depense_eur": round(float(r["dépense (€)"]), 2),
            "taux_de_visite_mesure_pct": round(100 * float(r["taux de visite"]), 2),
            "ic95_bas_pct": round(100 * float(r["ic_bas"]), 2),
            "ic95_haut_pct": round(100 * float(r["ic_haut"]), 2),
        }
        if r["politique"] != "ciblage classique":
            row |= {
                "gain_vs_ciblage_classique_points": round(float(r["gain vs classique (pts)"]), 2),
                "gain_ic95_bas": round(float(r["gain_bas"]), 2),
                "gain_ic95_haut": round(float(r["gain_haut"]), 2),
                "difference_demontree": bool(r["gain_bas"] > 0 or r["gain_haut"] < 0),
            }
        rows.append(row)
    return rows


METHOD = (
    "Taux de visite MESURÉS sur la campagne randomisée : chaque politique est évaluée sur les "
    "clients qui ont reçu, par tirage au sort, l'e-mail qu'elle prévoit (pondération inverse, IPW). "
    "Les intervalles de confiance viennent d'un rééchantillonnage (bootstrap)."
)


def indicators_explanation(planner: Planner, sc: Scenario) -> dict[str, Any]:
    """Les cinq indicateurs du haut de l'écran, avec leur définition."""
    plan, _ = _plan(planner, sc, compare=True)
    return {
        "element_affiche": "les cinq indicateurs du haut de l'écran",
        "scenario": asdict(sc),
        "indicateurs": _indicator_facts(plan),
        "niveau_de_confiance_pct": CONFIDENCE_PCT,
    }


def policies_explanation(planner: Planner, sc: Scenario) -> dict[str, Any]:
    """Graphique 1 : taux de visite mesuré de chaque politique, au même budget."""
    plan, _ = _plan(planner, sc, compare=True)
    return {
        "element_affiche": "graphique « Taux de visite mesuré sur le portefeuille (IC 95 %) »",
        "budget_eur": sc.budget,
        "methode": METHOD,
        "politiques": _policy_facts(plan),
        "niveau_de_confiance_pct": CONFIDENCE_PCT,
    }


def results_overview(planner: Planner, sc: Scenario) -> dict[str, Any]:
    """Tous les résultats affichés, de haut en bas : indicateurs, graphique 1, graphique 2."""
    plan, lam = _plan(planner, sc, compare=True)
    return {
        "scenario": asdict(sc),
        "indicateurs": _indicator_facts(plan),
        "graphique_1_taux_de_visite_mesure": {
            "methode": METHOD,
            "politiques": [
                {k: v for k, v in p.items() if k != "description"} for p in _policy_facts(plan)
            ],
        },
        "graphique_2_qui_recoit_quoi": [
            {
                "segment": r["segment"],
                "clients": r["clients"],
                "email_homme": r["e-mail homme"],
                "email_femme": r["e-mail femme"],
                "rien": r["rien"],
            }
            for r in plan.by_segment.to_dict(orient="records")
        ],
        **_rule(sc, lam),
        "niveau_de_confiance_pct": CONFIDENCE_PCT,
    }


def portfolio_overview(planner: Planner, segment: str | None = None) -> dict[str, Any]:
    """Portrait du portefeuille (ou d'un segment) : requête prédéfinie, aucun code libre."""
    mask = (
        np.ones(planner.n_clients, dtype=bool)
        if segment is None
        else (planner.segment == segment).to_numpy()
    )
    if segment is not None and segment not in SEGMENTS:
        raise ValueError(f"Segment inconnu : {segment}")
    c = planner.clients[mask]
    t, y = planner.treatment[mask], planner.visit[mask]

    def shares(col: pd.Series) -> dict[str, float]:
        return {
            str(k): round(100 * float(v), 1) for k, v in col.value_counts(normalize=True).items()
        }

    out: dict[str, Any] = {
        "perimetre": segment or "tout le portefeuille",
        "clients_du_perimetre": int(mask.sum()),
        "clients_du_portefeuille_entier": planner.n_clients,
    }
    if segment is None:  # filtrée sur un segment, cette répartition (100 %) induisait en erreur
        out["repartition_segments_pct"] = shares(planner.segment)
        out["repartition_segments_clients"] = {
            str(k): int(v) for k, v in planner.segment.value_counts().items()
        }
    else:
        out["part_du_portefeuille_pct"] = round(100 * float(mask.mean()), 1)
    return {
        **out,
        "nouveaux_clients": int(c["newbie"].sum()),
        "part_nouveaux_clients_pct": round(100 * float(c["newbie"].mean()), 1),
        "canal_pct": shares(c["channel"].astype(str)),
        "zone_pct": shares(c["zip_code"].astype(str)),
        "depense_annee_passee_mediane_eur": round(float(c["history"].median()), 2),
        "depense_annee_passee_moyenne_eur": round(float(c["history"].mean()), 2),
        "mois_depuis_dernier_achat_moyen": round(float(c["recency"].mean()), 1),
        "taux_de_visite_sans_email_pct": round(100 * float(y[t == 0].mean()), 2),
        "uplift_mesure_email_homme": _measured(planner, mask, 1),
        "uplift_mesure_email_femme": _measured(planner, mask, 2),
    }
