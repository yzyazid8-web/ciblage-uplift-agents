"""Sorties structurées du LLM et état de la conversation."""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

from pydantic import BaseModel, Field

Intent = Literal[
    "plan",  # nouvelle demande de plan d'envoi
    "modifier",  # « et si » : changer un paramètre du scénario en cours
    "comparer",  # comparer des scénarios déjà calculés
    "conseil",  # « ça vaut le coup d'ajouter 100 € ? », seuils de coût
    "expliquer",  # pourquoi tel segment / tel client reçoit tel e-mail
    "donnees",  # question sur le portefeuille de clients
    "exporter",  # produire le fichier de campagne (action à valider)
    "hors_sujet",
]


class Understanding(BaseModel):
    """Ce que le LLM comprend d'un message. Tout champ non mentionné reste à null."""

    intent: Intent = Field(description="Intention principale du message")
    budget: float | None = Field(
        None, description="Budget total en euros, s'il est donné en valeur absolue"
    )
    budget_factor: float | None = Field(
        None,
        description="Changement RELATIF du budget en cours, ex. 'la moitié' -> 0.5, '+20 %' -> 1.2",
    )
    budget_delta: float | None = Field(
        None,
        description="Changement ABSOLU du budget en cours, ex. '100 € de plus' -> 100, '50 € de moins' -> -50",
    )
    cost_homme: float | None = Field(None, description="Coût d'UN e-mail homme en euros")
    cost_femme: float | None = Field(None, description="Coût d'UN e-mail femme en euros")
    max_homme: int | None = Field(
        None, description="Nombre maximal d'e-mails homme, seulement si la demande le précise"
    )
    max_femme: int | None = Field(
        None, description="Nombre maximal d'e-mails femme, seulement si la demande le précise"
    )
    remove_limits: bool = Field(
        False, description="L'utilisateur demande de supprimer les limites d'envoi"
    )
    objective: Literal["visites", "profit"] | None = Field(
        None,
        description="'visites' pour maximiser les visites, 'profit' pour maximiser le gain en euros",
    )
    value_per_visit: float | None = Field(
        None, description="Ce que rapporte une visite en euros (objectif profit)"
    )
    segment: Literal["homme seulement", "femme seulement", "homme et femme"] | None = Field(
        None, description="Segment de clients visé par une question"
    )
    client_id: int | None = Field(None, description="Numéro de client mentionné")
    ambiguities: list[str] = Field(
        default_factory=list,
        description="Points réellement ambigus qui empêchent de répondre sans deviner (vide sinon)",
    )


class AgentState(TypedDict, total=False):
    """État d'une conversation, sauvegardé entre les tours (checkpointer)."""

    user_input: str  # message du tour en cours
    understanding: dict[str, Any]  # dernière compréhension (Understanding.model_dump)
    scenario: dict[str, Any] | None  # scénario courant (Scenario sous forme de dict)
    result: dict[str, Any] | None  # dernier plan calculé (Plan.to_dict)
    history: Annotated[list[dict[str, Any]], operator.add]  # scénarios et résultats précédents
    question: str | None  # question de clarification posée à l'utilisateur
    draft: dict[str, Any] | None  # scénario en cours de construction (avant clarification)
    pending_intent: str | None  # intention mise en attente pendant une clarification
    clarify_count: int  # nombre de questions posées pendant ce tour (limite anti-boucle)
    turn_intent: str  # intention traitée pendant ce tour (après clarification éventuelle)
    previous_result: dict[str, Any] | None  # plan précédent (pour comparer un « et si »)
    answer: str  # réponse finale du tour
    route: str  # étape choisie par le superviseur
    trace: list[str]  # étapes parcourues PENDANT CE TOUR (remis à zéro à chaque message)
    llm_calls: int  # appels au LLM pendant ce tour (suivi du quota)
