"""Accès aux LLM (Groq, offre gratuite) : modèle principal, modèle de secours, nouvel essai."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import SecretStr

from ciblage import config
from ciblage.agents import prompts
from ciblage.agents.schema import Understanding

Messages = list[tuple[str, str]]  # [(role, contenu)], rôle = "system" | "user" | "assistant"
Understander = Callable[[Messages], Understanding]
Writer = Callable[[Messages], str]
# agent à outils : historique de l'analyse (dicts sérialisables) -> {"content", "tool_calls"}
ChatMessages = list[dict[str, Any]]
Analyst = Callable[[ChatMessages], dict[str, Any]]

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
    roles = {"system": SystemMessage, "assistant": AIMessage}
    return [roles.get(r, HumanMessage)(c) for r, c in messages]


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


class UsageCounter(BaseCallbackHandler):
    """Compte les appels et les tokens réellement consommés (suivi du quota, évaluation)."""

    def __init__(self) -> None:
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.reasoning_tokens = 0
        self.models: dict[str, int] = {}

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        out = response.llm_output or {}
        usage = out.get("token_usage") or {}
        self.calls += 1
        self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
        self.completion_tokens += int(usage.get("completion_tokens") or 0)
        details = usage.get("completion_tokens_details") or {}
        self.reasoning_tokens += int(details.get("reasoning_tokens") or 0)
        name = str(out.get("model_name") or "inconnu")
        self.models[name] = self.models.get(name, 0) + 1


def make_understander(fallback: bool = True, counter: UsageCounter | None = None) -> Understander:
    """Extraction structurée (intention + paramètres), avec secours automatique.

    `fallback=False` : modèle principal seul (évaluation : on mesure UN modèle, pas un mélange).
    """
    main, backup = models()
    chain: Any = _chat(main).with_structured_output(Understanding).with_retry(stop_after_attempt=2)
    if fallback:
        chain = chain.with_fallbacks([_chat(backup).with_structured_output(Understanding)])
    config = {"callbacks": [counter]} if counter else {}

    def understand(messages: Messages) -> Understanding:
        result = chain.invoke(_to_messages(messages), config=config)
        return result if isinstance(result, Understanding) else Understanding.model_validate(result)

    return understand


def make_writer(fallback: bool = True, counter: UsageCounter | None = None) -> Writer:
    """Rédaction en texte libre, avec secours automatique (désactivable)."""
    main, backup = models()
    chain: Any = _chat(main).with_retry(stop_after_attempt=2)
    if fallback:
        chain = chain.with_fallbacks([_chat(backup)])
    config = {"callbacks": [counter]} if counter else {}

    def write(messages: Messages) -> str:
        return str(chain.invoke(_to_messages(messages), config=config).content).strip()

    return write


def _to_chat_messages(messages: ChatMessages) -> list[BaseMessage]:
    out: list[BaseMessage] = []
    for m in messages:
        role, content = m["role"], m.get("content") or ""
        if role == "system":
            out.append(SystemMessage(content))
        elif role == "user":
            out.append(HumanMessage(content))
        elif role == "assistant":
            out.append(AIMessage(content, tool_calls=m.get("tool_calls") or []))
        else:  # résultat d'outil
            out.append(ToolMessage(content, tool_call_id=m["tool_call_id"]))
    return out


def make_analyst(
    tool_schemas: list[dict[str, Any]], fallback: bool = True, counter: UsageCounter | None = None
) -> Analyst:
    """LLM qui choisit lui-même ses outils (function calling), avec secours automatique."""
    main, backup = models()
    chain: Any = _chat(main).bind_tools(tool_schemas).with_retry(stop_after_attempt=2)
    if fallback:
        chain = chain.with_fallbacks([_chat(backup).bind_tools(tool_schemas)])
    config = {"callbacks": [counter]} if counter else {}

    def analyze(messages: ChatMessages) -> dict[str, Any]:
        reply = chain.invoke(_to_chat_messages(messages), config=config)
        calls = [
            {"id": c.get("id") or f"appel-{k}", "name": c["name"], "args": c.get("args") or {}}
            for k, c in enumerate(getattr(reply, "tool_calls", None) or [])
        ]
        return {"content": str(reply.content or "").strip(), "tool_calls": calls}

    return analyze


def understand_messages(user_text: str, context: str) -> Messages:
    """Messages envoyés pour comprendre un message utilisateur, avec le contexte du scénario."""
    return [("system", prompts.UNDERSTAND), ("system", context), ("user", user_text)]
