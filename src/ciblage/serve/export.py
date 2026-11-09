"""Export du fichier de campagne : la liste des clients à contacter et l'e-mail à leur envoyer.

C'est la seule action de l'agent qui produit un effet réel : elle n'est exécutée qu'après une
confirmation explicite de l'utilisateur (voir le graphe). L'aperçu (`preview`) n'écrit rien.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ciblage import config
from ciblage.serve.planner import LABELS, Planner, Scenario

EXPORTS_DIR = config.DATA_DIR / "exports"


def _campaign(planner: Planner, sc: Scenario) -> tuple[pd.DataFrame, dict[str, Any]]:
    plan = planner.plan(sc, compare=False)
    a = plan.assignment
    chosen = a > 0
    table = pd.DataFrame(
        {
            "client": chosen.nonzero()[0],
            "segment": planner.segment.to_numpy()[chosen],
            "email": [LABELS[int(v)] for v in a[chosen]],
            "uplift_predit_points": [
                round(100 * float(planner.uplift[i, int(v) - 1]), 2)
                for i, v in zip(chosen.nonzero()[0], a[chosen], strict=True)
            ],
        }
    )
    s = plan.summary
    summary = {
        "lignes": int(chosen.sum()),
        "email_homme": s["email_homme"],
        "email_femme": s["email_femme"],
        "depense_eur": round(s["depense"], 2),
        "visites_en_plus_predites": round(s["visites_en_plus_predites"]),
    }
    return table, summary


def preview(planner: Planner, sc: Scenario) -> dict[str, Any]:
    """Ce que contiendrait le fichier, sans rien écrire."""
    return _campaign(planner, sc)[1]


def export_campaign(
    planner: Planner, sc: Scenario, directory: Path = EXPORTS_DIR
) -> dict[str, Any]:
    """Écrit le fichier CSV (écriture atomique) et renvoie son chemin et son contenu résumé."""
    table, summary = _campaign(planner, sc)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"campagne_{datetime.now():%Y%m%d_%H%M%S}_{int(sc.budget)}eur.csv"
    tmp = path.with_suffix(".part")
    table.to_csv(tmp, index=False, encoding="utf-8")
    tmp.replace(path)
    return {"fichier": str(path), **summary}
