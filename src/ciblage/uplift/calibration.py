"""Calibration de l'uplift : l'effet prédit correspond-il à l'effet observé ?"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ciblage.causal.ate import welch


def calibration_table(y, predicted: np.ndarray, treatment, bins: int = 10) -> pd.DataFrame:
    """Uplift prédit vs observé par tranche (traitement binaire : 1 = traité, 0 = témoin)."""
    yy = np.asarray(y, dtype=np.float64)
    pred = np.asarray(predicted, dtype=np.float64)
    t = np.asarray(treatment)
    order = np.argsort(-pred, kind="mergesort")
    rows = []
    for k, idx in enumerate(np.array_split(order, bins), start=1):
        yt, yc = yy[idx][t[idx] == 1], yy[idx][t[idx] == 0]
        if len(yt) < 2 or len(yc) < 2:
            continue
        w = welch(yt, yc)
        rows.append(
            {
                "tranche": k,
                "n": len(idx),
                "predit": float(pred[idx].mean()),
                "observe": w["ate"],
                "ci_low": w["ci_low"],
                "ci_high": w["ci_high"],
            }
        )
    return pd.DataFrame(rows)


def calibration_error(table: pd.DataFrame) -> float:
    """Écart absolu moyen prédit / observé, pondéré par la taille des tranches."""
    w = table["n"].to_numpy(dtype=np.float64)
    gap = np.abs(table["predit"] - table["observe"]).to_numpy()
    return float((w * gap).sum() / w.sum())
