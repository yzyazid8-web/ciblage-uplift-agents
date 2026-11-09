"""Réglages retenus en phase 2 pour Hillstrom (visite), réutilisés par les phases suivantes."""

from __future__ import annotations

from typing import Any

from ciblage.uplift.learners import SLearner

LEARNER = SLearner
PARAMS: dict[str, Any] = {"max_depth": 2, "n_estimators": 300, "min_child_weight": 5}
TEST_SIZE = 0.3  # même découpage entraînement / test qu'en phase 2 (graine config.SEED)
