"""Tracing de l'agent avec MLflow (local, gratuit) : chaque étape du graphe, chaque appel au LLM
(messages, réponse, durée, tokens) et chaque outil sont enregistrés et consultables."""

from __future__ import annotations

import os

from ciblage import config

EXPERIMENT = "agent-copilote"


def enable_tracing(force: bool = False) -> bool:
    """Active le tracing si demandé et possible. Renvoie True s'il est actif."""
    if not force and os.environ.get("CIBLAGE_TRACING", "0") != "1":
        return False
    try:
        import mlflow
        import mlflow.langchain
    except ImportError:
        return False
    mlflow.set_tracking_uri(f"sqlite:///{config.ROOT / 'mlflow.db'}")
    mlflow.set_experiment(EXPERIMENT)
    mlflow.langchain.autolog()
    return True
