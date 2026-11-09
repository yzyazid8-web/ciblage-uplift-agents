"""Petits outils de mise en forme partagés par les rapports générés."""

from __future__ import annotations

import pandas as pd


def md_table(df: pd.DataFrame) -> str:
    """Tableau markdown minimal (évite une dépendance supplémentaire)."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def n_fmt(v: float) -> str:
    """Entier avec espace fine insécable comme séparateur de milliers."""
    return f"{round(v):,}".replace(",", " ")


def pts(v: float) -> str:
    """Proportion exprimée en points de pourcentage, signée."""
    return f"{100 * v:+.2f}"
