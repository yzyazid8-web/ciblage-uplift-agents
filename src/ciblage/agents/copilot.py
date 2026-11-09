"""Point d'entrée de l'agent : un tour de conversation = un message -> une réponse ou une question."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from ciblage import config
from ciblage.agents.graph import Deps, build_graph
from ciblage.agents.llm import make_understander, make_writer
from ciblage.agents.tracing import enable_tracing
from ciblage.serve.planner import Planner

SESSIONS_DB = config.DATA_DIR / "sessions.db"


@dataclass
class TurnResult:
    kind: Literal["reponse", "question"]
    text: str
    scenario: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    trace: list[str] = field(default_factory=list)
    llm_calls: int = 0


class Copilot:
    def __init__(self, deps: Deps, db_path: Path | None = None) -> None:
        if db_path is None:
            saver: Any = InMemorySaver()
        else:
            db_path.parent.mkdir(parents=True, exist_ok=True)
            saver = SqliteSaver(sqlite3.connect(str(db_path), check_same_thread=False))
        self.graph = build_graph(deps, checkpointer=saver)

    @classmethod
    def from_env(
        cls, planner: Planner | None = None, db_path: Path | None = SESSIONS_DB
    ) -> Copilot:
        enable_tracing()  # actif seulement si CIBLAGE_TRACING=1
        deps = Deps(
            planner=planner or Planner(), understand=make_understander(), write=make_writer()
        )
        return cls(deps, db_path)

    @staticmethod
    def _config(thread_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": thread_id}}

    def waiting_for_answer(self, thread_id: str) -> bool:
        """Le graphe attend-il la réponse à une question de clarification ?"""
        snap = self.graph.get_state(self._config(thread_id))
        return any(task.interrupts for task in snap.tasks)

    def state(self, thread_id: str) -> dict[str, Any]:
        return dict(self.graph.get_state(self._config(thread_id)).values)

    def send(self, thread_id: str, text: str) -> TurnResult:
        cfg = self._config(thread_id)
        payload: Any = (
            Command(resume=text) if self.waiting_for_answer(thread_id) else {"user_input": text}
        )
        out = self.graph.invoke(payload, cfg)
        common = {
            "scenario": out.get("scenario"),
            "result": out.get("result"),
            "trace": out.get("trace", []),
            "llm_calls": out.get("llm_calls", 0),
        }
        if out.get("__interrupt__"):
            return TurnResult(
                kind="question", text=out["__interrupt__"][0].value["question"], **common
            )
        return TurnResult(kind="reponse", text=out.get("answer", ""), **common)
