"""API HTTP du copilote (FastAPI).

Lancement : uvicorn ciblage.serve.api:app --port 8000   (documentation interactive : /docs)

Routes :
- GET  /sante                                   état du service
- POST /plan                                    plan pour un scénario (sans LLM)
- POST /conversation/{session}                  un tour de conversation avec l'agent
- POST /conversation/{session}/flux             le même tour, diffusé en direct (Server-Sent Events)
- GET  /conversation/{session}/etat             scénario courant et question en attente

Les modèles sont chargés une seule fois au démarrage. Sans clé Groq, les routes de conversation
répondent 503 (service indisponible) et le reste de l'API fonctionne.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ciblage.serve.planner import Planner, Scenario


class ScenarioIn(BaseModel):
    budget: float = Field(..., ge=0, description="Budget total en euros")
    cost_homme: float = Field(0.12, ge=0, description="Coût d'un e-mail homme (€)")
    cost_femme: float = Field(0.10, ge=0, description="Coût d'un e-mail femme (€)")
    max_homme: int | None = Field(None, ge=0)
    max_femme: int | None = Field(None, ge=0)
    objective: Literal["visites", "profit"] = "visites"
    value_per_visit: float = Field(1.0, gt=0)
    compare: bool = Field(True, description="Mesurer le plan face aux alternatives (plus lent)")


class MessageIn(BaseModel):
    message: str = Field(..., min_length=1, max_length=5_000)


def create_app(
    planner: Planner | None = None, copilot: Any = None, load_agent: bool = True
) -> FastAPI:
    """Fabrique de l'application. Les tests injectent un planificateur et un agent (faux LLM)."""
    services: dict[str, Any] = {"planner": planner, "copilot": copilot, "agent_error": None}

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if services["planner"] is None:
            services["planner"] = Planner()
        if services["copilot"] is None and load_agent:
            from ciblage.agents.copilot import Copilot
            from ciblage.agents.llm import LLMUnavailable

            try:
                services["copilot"] = Copilot.from_env(planner=services["planner"])
            except LLMUnavailable as err:
                services["agent_error"] = str(err)
        yield

    app = FastAPI(
        title="Copilote de ciblage",
        description="Qui contacter, avec quel e-mail, pour un budget donné (uplift + optimisation + agent).",
        version="0.1.0",
        lifespan=lifespan,
    )

    def agent() -> Any:
        if services["copilot"] is None:
            raise HTTPException(503, services["agent_error"] or "Agent indisponible")
        return services["copilot"]

    @app.get("/sante")
    def sante() -> dict[str, Any]:
        p = services["planner"]
        return {
            "statut": "ok",
            "clients_portefeuille": p.n_clients if p else None,
            "agent_disponible": services["copilot"] is not None,
        }

    @app.post("/plan")
    def plan(body: ScenarioIn) -> dict[str, Any]:
        sc = Scenario(**body.model_dump(exclude={"compare"}))
        errors = sc.errors()
        if errors:
            raise HTTPException(422, " ".join(errors))
        return services["planner"].plan(sc, compare=body.compare, n_boot=200).to_dict()

    @app.post("/conversation/{session}")
    def conversation(session: str, body: MessageIn) -> dict[str, Any]:
        result = agent().send(session, body.message)
        return asdict(result)

    @app.post("/conversation/{session}/flux")
    def conversation_flux(session: str, body: MessageIn) -> StreamingResponse:
        copilote = agent()

        def events() -> Iterator[str]:
            for event in copilote.stream(session, body.message):
                yield f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    @app.get("/conversation/{session}/etat")
    def etat(session: str) -> dict[str, Any]:
        copilote = agent()
        values = copilote.state(session)
        return {
            "scenario": values.get("scenario"),
            "question_en_attente": values.get("question")
            if copilote.waiting_for_answer(session)
            else None,
            "scenarios_calcules": len(values.get("history", [])),
        }

    return app


app = create_app()
