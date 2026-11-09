"""Tests de l'API avec l'agent sur FAUX LLM (aucune clé, aucun réseau)."""

import json

import pytest
from fastapi.testclient import TestClient
from test_agent_graph import faithful_writer, fake_understand  # faux LLM partagé

from ciblage.agents.copilot import Copilot
from ciblage.agents.graph import Deps
from ciblage.serve.api import create_app
from ciblage.serve.planner import Planner


@pytest.fixture(scope="module")
def client():
    planner = Planner()
    copilot = Copilot(Deps(planner=planner, understand=fake_understand, write=faithful_writer))
    with TestClient(create_app(planner=planner, copilot=copilot)) as c:
        yield c


def test_health(client):
    r = client.get("/sante").json()
    assert r["statut"] == "ok" and r["agent_disponible"] and r["clients_portefeuille"] == 19_200


def test_plan_endpoint(client):
    r = client.post("/plan", json={"budget": 300, "compare": False})
    assert r.status_code == 200
    assert r.json()["summary"]["depense"] <= 300


def test_plan_rejects_invalid_scenario(client):
    assert client.post("/plan", json={"budget": -5}).status_code == 422
    r = client.post("/plan", json={"budget": 100, "objective": "profit", "value_per_visit": 0})
    assert r.status_code == 422


def test_conversation_with_clarification(client):
    r = client.post("/conversation/api1", json={"message": "plan sans coût femme"}).json()
    assert r["kind"] == "question"
    state = client.get("/conversation/api1/etat").json()
    assert state["question_en_attente"] and state["scenario"] is None
    r = client.post("/conversation/api1", json={"message": "0,10 €"}).json()
    assert r["kind"] == "reponse" and r["scenario"]["budget"] == 600


def test_conversation_stream_is_server_sent_events(client):
    with client.stream("POST", "/conversation/api2/flux", json={"message": "plan complet"}) as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        events = [json.loads(line[6:]) for line in r.iter_lines() if line.startswith("data: ")]
    assert any(e["type"] == "etape" for e in events)
    assert events[-1]["type"] == "reponse"


def test_agent_unavailable_returns_503():
    with TestClient(create_app(planner=Planner(), load_agent=False)) as c:
        r = c.post("/conversation/x", json={"message": "bonjour"})
        assert r.status_code == 503
