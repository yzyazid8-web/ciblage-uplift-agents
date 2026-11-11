"""Analyse de sensibilité au confounding caché : la E-value."""

from __future__ import annotations

from dataclasses import dataclass

_EPS = 1e-9


def risk_ratio(ate: float, baseline: float) -> float:
    """Risque relatif implicite d'une différence de risque : (p0 + ate) / p0.

    `baseline` = p0 = taux de résultat chez les témoins. Le risque traité p1 = p0 + ate est
    borné à ]0, 1[ pour rester un risque valide.
    """
    if baseline <= 0:
        raise ValueError("Le risque témoin (baseline) doit être strictement positif")
    p1 = min(max(baseline + ate, _EPS), 1 - _EPS)
    return p1 / baseline


def e_value_rr(rr: float) -> float:
    """E-value d'un risque relatif (formule de VanderWeele & Ding, symétrique autour de 1)."""
    if rr <= 0:
        return float("inf")
    if rr < 1:
        rr = 1.0 / rr
    return rr + (rr * (rr - 1.0)) ** 0.5


@dataclass
class EValue:
    risk_ratio: float  # RR implicite de l'estimation ponctuelle
    point: float  # E-value de l'estimation
    ci: float  # E-value de la borne d'IC la plus proche de 1 (1.0 si l'IC contient l'absence d'effet)


def e_value(ate: float, ci_low: float, ci_high: float, baseline: float) -> EValue:
    """E-value d'un effet exprimé en différence de risque, avec son intervalle de confiance."""
    rr = risk_ratio(ate, baseline)
    rr_low = risk_ratio(ci_low, baseline)
    rr_high = risk_ratio(ci_high, baseline)
    if rr_low > 1.0:  # effet positif significatif : borne basse au-dessus de 1
        ci = e_value_rr(rr_low)
    elif rr_high < 1.0:  # effet négatif significatif : borne haute en dessous de 1
        ci = e_value_rr(rr_high)
    else:  # l'IC traverse 1 (effet non démontré)
        ci = 1.0
    return EValue(risk_ratio=rr, point=e_value_rr(rr), ci=ci)
