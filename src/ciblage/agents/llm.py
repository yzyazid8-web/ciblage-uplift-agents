"""Accès aux LLM (Groq, offre gratuite) : modèle principal, modèle de secours, nouvel essai."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import SecretStr

from ciblage import config
from ciblage.agents import prompts
from ciblage.agents.schema import Understanding

Messages = list[tuple[str, str]]  # [(role, contenu)], rôle = "system" | "user"
Understander = Callable[[Messages], Understanding]
Writer = Callable[[Messages], str]

DEFAULT_MODEL = "openai/gpt-oss-120b"
DEFAULT_FALLBACK = "openai/gpt-oss-20b"


class LLMUnavailable(RuntimeError):
    """Aucun LLM utilisable (clé absente ou service injoignable)."""


def _load_env_file() -> None:
    """Charge `.env` (CLÉ=valeur) sans dépendance ; les variables déjà définies sont prioritaires."""
    path = config.ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def _to_messages(messages: Messages) -> list[BaseMessage]:
    return [SystemMessage(c) if r == "system" else HumanMessage(c) for r, c in messages]


def _chat(model: str) -> Any:
    from langchain_groq import ChatGroq

    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        raise LLMUnavailable("GROQ_API_KEY absente : renseigne-la dans le fichier .env")
    return ChatGroq(model=model, api_key=SecretStr(key), temperature=0, max_retries=2, timeout=30)


def models() -> tuple[str, str]:
    _load_env_file()
    return (
        os.environ.get("LLM_MODEL", DEFAULT_MODEL),
        os.environ.get("LLM_FALLBACK_MODEL", DEFAULT_FALLBACK),
    )


def make_understander() -> Understander:
    """Extraction structurée (intention + paramètres), avec secours automatique."""
    main, fallback = models()
    chain = (
        _chat(main)
        .with_structured_output(Understanding)
        .with_retry(stop_after_attempt=2)
        .with_fallbacks([_chat(fallback).with_structured_output(Understanding)])
    )

    def understand(messages: Messages) -> Understanding:
        result = chain.invoke(_to_messages(messages))
        return result if isinstance(result, Understanding) else Understanding.model_validate(result)

    return understand


def make_writer() -> Writer:
    """Rédaction en texte libre, avec secours automatique."""
    main, fallback = models()
    chain = _chat(main).with_retry(stop_after_attempt=2).with_fallbacks([_chat(fallback)])

    def write(messages: Messages) -> str:
        return str(chain.invoke(_to_messages(messages)).content).strip()

    return write


def understand_messages(user_text: str, context: str) -> Messages:
    """Messages envoyés pour comprendre un message utilisateur, avec le contexte du scénario."""
    return [("system", prompts.UNDERSTAND), ("system", context), ("user", user_text)]
