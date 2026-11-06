"""Hétérogénéité de l'effet : pour quels profils une action marche-t-elle le mieux ?"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ciblage.causal.ate import holm, welch


def uplift_by_segment(
    df: pd.DataFrame,
    segment: str,
    outcome: str,
    treatment_col: str = "treatment",
    control: int = 0,
    min_per_group: int = 30,
) -> pd.DataFrame:
    """Effet de chaque action dans chaque modalité de `segment`, avec IC de Welch et p de Holm."""
    treatments = [g for g in np.unique(t) if g != control]
    rows = []
    for level, part in df.groupby(segment, observed=True, sort=True):
        pt = part[treatment_col].to_numpy()
        yc = part.loc[pt == control, outcome].to_numpy()
        for g in treatments:
            yt = part.loc[pt == g, outcome].to_numpy()
            if len(yt) < min_per_group or len(yc) < min_per_group:
                continue
            w = welch(yt, yc)
            rows.append(
                {
                    "segment": segment,
                    "niveau": level,
                    "treatment": int(g),
                    "n_traite": len(yt),
                    "n_temoin": len(yc),
                    "taux_temoin": float(yc.mean()),
                    "uplift": w["ate"],
                    "ci_low": w["ci_low"],
                    "ci_high": w["ci_high"],
                    "p_value": w["p_value"],
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_holm"] = holm(out["p_value"].to_numpy())
    return out


def bin_quantiles(x: pd.Series, q: int = 4, label: str = "Q") -> pd.Series:
    """Découpe une variable continue en `q` classes d'effectifs proches."""
    return pd.qcut(x, q=q, labels=[f"{label}{i + 1}" for i in range(q)], duplicates="drop")
