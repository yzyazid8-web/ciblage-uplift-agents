"""Point d'entrée de l'agent : un tour de conversation = un message -> une réponse ou une question."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from ciblage import config
from ciblage.agents import tools
from ciblage.agents.graph import Deps, build_graph
from ciblage.agents.llm import make_analyst, make_understander, make_writer
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
            planner=planner or Planner(),
            understand=make_understander(),
            write=make_writer(),
            analyze=make_analyst(tools.schemas()),
        )
        return cls(deps, db_path)

    @staticmethod
    def _config(thread_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": thread_id}}

    def waiting_for_answer(self, thread_id: str) -> bool:
        """Le graphe attend-il la réponse à une question de clarification ?"""
        snap = self.graph.get_state(self._config(thread_id))
        return any(task.interrupts for task in snap.tasks)

    def sync_scenario(self, thread_id: str, scenario: dict[str, Any]) -> bool:
        """Aligne le scénario de l'agent sur celui affiché ailleurs (formulaire du tableau de bord)."""
        if self.waiting_for_answer(thread_id) or self.state(thread_id).get("scenario") == scenario:
            return False
        # écrit « comme si » l'étape finale venait de s'exécuter : aucune étape n'est en attente
        self.graph.update_state(self._config(thread_id), {"scenario": scenario}, as_node="write")
        return True

    def state(self, thread_id: str) -> dict[str, Any]:
        return dict(self.graph.get_state(self._config(thread_id)).values)

    def _payload(self, thread_id: str, text: str) -> Any:
        """Nouveau tour, ou reprise du graphe s'il attend la réponse à une question."""
        return Command(resume=text) if self.waiting_for_answer(thread_id) else {"user_input": text}

    def stream(self, thread_id: str, text: str) -> Iterator[dict[str, Any]]:
        """Diffuse le tour en direct : un événement par nouvelle étape, puis le résultat final.

        Événements : {"type": "etape", "texte": ...}, puis un seul
        {"type": "reponse" | "question", "texte": ..., "scenario": ..., "trace": ..., "llm_calls": ...}.
        """
        cfg = self._config(thread_id)
        seen = 0
        question = None
        for chunk in self.graph.stream(self._payload(thread_id, text), cfg, stream_mode="updates"):
            for node, update in chunk.items():
                if node == "__interrupt__":
                    question = update[0].value["question"]
                    continue
                trace = (update or {}).get("trace")
                if trace and len(trace) > seen:
                    for step in trace[max(seen, 1) :]:  # « début » n'est pas affiché
                        yield {"type": "etape", "texte": step}
                    seen = len(trace)
        values = self.state(thread_id)
        yield {
            "type": "question" if question else "reponse",
            "texte": question or values.get("answer", ""),
            "scenario": values.get("scenario"),
            "trace": values.get("trace", []),
            "llm_calls": values.get("llm_calls", 0),
        }

    def send(self, thread_id: str, text: str) -> TurnResult:
        cfg = self._config(thread_id)
        out = self.graph.invoke(self._payload(thread_id, text), cfg)
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
