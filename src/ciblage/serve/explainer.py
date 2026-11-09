"""Explications chiffrées des décisions, et portrait du portefeuille (jamais estimés par le LLM)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ciblage.causal.ate import welch
from ciblage.serve.planner import LABELS, Planner, Scenario

SEGMENTS = ("homme seulement", "femme seulement", "homme et femme")


def _plan(planner: Planner, sc: Scenario):
    plan = planner.plan(sc, compare=False)
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
    }


def segment_explanation(planner: Planner, sc: Scenario, segment: str) -> dict[str, Any]:
    """Pourquoi tel segment reçoit-il tel e-mail ? Prédictions, mesures et seuil de rentabilité."""
    if segment not in SEGMENTS:
        raise ValueError(f"Segment inconnu : {segment}")
    plan, lam = _plan(planner, sc)
    m = (planner.segment == segment).to_numpy()
    a = plan.assignment[m]
    u = planner.uplift[m]
    facts: dict[str, Any] = {
        "segment": segment,
        "clients": int(m.sum()),
        "email_homme": int((a == 1).sum()),
        "email_femme": int((a == 2).sum()),
        "rien": int((a == 0).sum()),
        "seuil_de_rentabilite_visites_par_euro": round(lam, 3),
    }
    for action, name, cost in ((1, "homme", sc.cost_homme), (2, "femme", sc.cost_femme)):
        mean_uplift = float(u[:, action - 1].mean())
        facts[f"uplift_predit_{name}_points"] = round(100 * mean_uplift, 2)
        facts[f"rendement_moyen_{name}_visites_par_euro"] = (
            round(mean_uplift / cost, 3) if cost > 0 else None
        )
        facts[f"uplift_mesure_{name}"] = _measured(planner, m, action)
        # la moyenne du segment peut être sous le seuil alors que les clients CHOISIS le dépassent
        chosen = a == action
        facts[f"rendement_moyen_des_clients_qui_recoivent_{name}_visites_par_euro"] = (
            round(float(u[chosen, action - 1].mean()) / cost, 3)
            if chosen.any() and cost > 0
            else None
        )
    return facts


def client_explanation(planner: Planner, sc: Scenario, client_id: int) -> dict[str, Any]:
    """Pourquoi ce client reçoit-il (ou non) un e-mail ?"""
    if not 0 <= client_id < planner.n_clients:
        raise ValueError(f"Le numéro de client doit être entre 0 et {planner.n_clients - 1}")
    plan, lam = _plan(planner, sc)
    row = planner.clients.iloc[client_id]
    u = planner.uplift[client_id]
    costs = {"homme": sc.cost_homme, "femme": sc.cost_femme}
    rendements = {
        name: round(float(u[k]) / costs[name], 3) if costs[name] > 0 else None
        for k, name in enumerate(("homme", "femme"))
    }
    decision = LABELS[int(plan.assignment[client_id])]
    best = max(rendements, key=lambda k: rendements[k] or -np.inf)
    above = (rendements[best] or 0) > lam
    if decision == "rien":
        raison = (
            "aucun e-mail ne lui fait gagner de visite (uplift prédit négatif ou nul)"
            if max(u) <= 0
            else "son meilleur rendement est sous le seuil de rentabilité : le budget sert mieux ailleurs"
        )
    else:
        raison = (
            "son rendement dépasse le seuil de rentabilité, et c'est le meilleur des deux e-mails"
        )
    return {
        "client": client_id,
        "segment": str(planner.segment.iloc[client_id]),
        "mois_depuis_dernier_achat": int(row["recency"]),
        "depense_annee_passee_eur": round(float(row["history"]), 2),
        "nouveau_client": bool(row["newbie"]),
        "canal": str(row["channel"]),
        "decision": decision,
        "uplift_predit_homme_points": round(100 * float(u[0]), 2),
        "uplift_predit_femme_points": round(100 * float(u[1]), 2),
        "rendement_homme_visites_par_euro": rendements["homme"],
        "rendement_femme_visites_par_euro": rendements["femme"],
        "seuil_de_rentabilite_visites_par_euro": round(lam, 3),
        "meilleur_rendement_au_dessus_du_seuil": bool(above),
        "raison": raison,
    }


def excluded_profile(planner: Planner, sc: Scenario) -> dict[str, Any]:
    """Qui sont les clients écartés, comparés aux clients contactés ?"""
    plan, lam = _plan(planner, sc)
    contacted = plan.assignment > 0
    out: dict[str, Any] = {"seuil_de_rentabilite_visites_par_euro": round(lam, 3)}
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
    return out


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
