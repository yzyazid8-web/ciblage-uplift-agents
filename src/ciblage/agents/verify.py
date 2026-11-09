"""Vérificateur déterministe d'un plan, INDÉPENDANT de l'optimiseur."""

from __future__ import annotations

import numpy as np

from ciblage.serve.planner import Plan


def verify_plan(plan: Plan, tol: float = 1e-6) -> list[str]:
    """Liste des problèmes détectés (vide = plan conforme au scénario)."""
    sc, a = plan.scenario, np.asarray(plan.assignment)
    issues = []
    if not np.isin(a, [0, 1, 2]).all():
        issues.append("Codes d'action inconnus dans le plan.")
        return issues
    n_h, n_f = int((a == 1).sum()), int((a == 2).sum())
    spend = n_h * sc.cost_homme + n_f * sc.cost_femme
    if spend > sc.budget + tol:
        issues.append(f"Dépense {spend:.2f} € supérieure au budget {sc.budget:.2f} €.")
    if sc.max_homme is not None and n_h > sc.max_homme:
        issues.append(f"{n_h} e-mails homme pour une limite de {sc.max_homme}.")
    if sc.max_femme is not None and n_f > sc.max_femme:
        issues.append(f"{n_f} e-mails femme pour une limite de {sc.max_femme}.")
    s = plan.summary
    if s["email_homme"] != n_h or s["email_femme"] != n_f:
        issues.append("Le résumé du plan ne correspond pas à l'affectation.")
    if abs(s["depense"] - spend) > 1e-3:
        issues.append("La dépense annoncée ne correspond pas à l'affectation.")
    return issues
