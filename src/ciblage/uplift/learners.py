"""Méta-learners d'uplift : estimer, client par client, l'effet de chaque action."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier, XGBRegressor

from ciblage import config

Outcome = Literal["binary", "continuous"]

DEFAULT_PARAMS: dict[str, Any] = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "tree_method": "hist",
    "n_jobs": -1,
}


class BaseUplift:
    """Socle commun : gestion des actions, de la propension et des modèles de base."""

    name = "base"

    def __init__(
        self,
        outcome: Outcome = "binary",
        params: dict[str, Any] | None = None,
        propensity: dict[int, float] | None = None,
        control: int = 0,
        seed: int = config.SEED,
    ) -> None:
        self.outcome = outcome
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.propensity = propensity
        self.control = control
        self.seed = seed

    # modèles de base 
    def _response_model(self) -> XGBClassifier | XGBRegressor:
        """Modèle de E[Y | x] : classifieur (probabilité) si Y est binaire, régresseur sinon."""
        if self.outcome == "binary":
            return XGBClassifier(**self.params, random_state=self.seed, eval_metric="logloss")
        return XGBRegressor(**self.params, random_state=self.seed)

    def _effect_model(self) -> XGBRegressor:
        """Modèle d'un effet (quantité continue, éventuellement négative)."""
        return XGBRegressor(**self.params, random_state=self.seed)

    @staticmethod
    def _predict_response(model: XGBClassifier | XGBRegressor, X: np.ndarray) -> np.ndarray:
        if isinstance(model, XGBClassifier):
            return model.predict_proba(X)[:, 1]
        return model.predict(X)

    # préparation 
    def _prepare(self, X, treatment, y) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        Xa = np.asarray(X, dtype=np.float64)
        t = np.asarray(treatment)
        ya = np.asarray(y, dtype=np.float64)
        if not (len(Xa) == len(t) == len(ya)):
            raise ValueError("X, treatment et y doivent avoir la même longueur")
        if self.outcome == "binary" and not np.isin(ya, [0, 1]).all():
            raise ValueError("outcome='binary' mais y n'est pas binaire")
        groups = np.unique(t)
        if self.control not in groups:
            raise ValueError("Aucun client témoin dans les données")
        self.actions_ = [int(g) for g in groups if g != self.control]
        if not self.actions_:
            raise ValueError("Aucun client traité dans les données")
        if self.propensity is None:
            self.propensity_ = {int(g): float(np.mean(t == g)) for g in groups}
        else:
            self.propensity_ = dict(self.propensity)
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
        return Xa, t, ya

    def fit(self, X, treatment, y) -> BaseUplift:
        raise NotImplementedError

    def predict(self, X) -> np.ndarray:
        raise NotImplementedError


class SLearner(BaseUplift):
    name = "S-learner"

    def _with_treatment(self, X: np.ndarray, code: int) -> np.ndarray:
        onehot = np.zeros((len(X), len(self.actions_)))
        if code != self.control:
            onehot[:, self.actions_.index(code)] = 1.0
        return np.hstack([X, onehot])

    def fit(self, X, treatment, y) -> SLearner:
        Xa, t, ya = self._prepare(X, treatment, y)
        onehot = np.column_stack([(t == a).astype(np.float64) for a in self.actions_])
        self.model_ = self._response_model().fit(np.hstack([Xa, onehot]), ya)
        return self

    def predict(self, X) -> np.ndarray:
        Xa = np.asarray(X, dtype=np.float64)
        base = self._predict_response(self.model_, self._with_treatment(Xa, self.control))
        return np.column_stack(
            [
                self._predict_response(self.model_, self._with_treatment(Xa, a)) - base
                for a in self.actions_
            ]
        )


class TLearner(BaseUplift):
    name = "T-learner"

    def _fit_arms(self, Xa: np.ndarray, t: np.ndarray, ya: np.ndarray) -> None:
        self.models_ = {
            g: self._response_model().fit(Xa[t == g], ya[t == g])
            for g in [self.control, *self.actions_]
        }

    def _mu(self, Xa: np.ndarray) -> dict[int, np.ndarray]:
        return {g: self._predict_response(m, Xa) for g, m in self.models_.items()}

    def fit(self, X, treatment, y) -> TLearner:
        Xa, t, ya = self._prepare(X, treatment, y)
        self._fit_arms(Xa, t, ya)
        return self

    def predict(self, X) -> np.ndarray:
        mu = self._mu(np.asarray(X, dtype=np.float64))
        return np.column_stack([mu[a] - mu[self.control] for a in self.actions_])


class XLearner(TLearner):
    name = "X-learner"

    def fit(self, X, treatment, y) -> XLearner:
        Xa, t, ya = self._prepare(X, treatment, y)
        self._fit_arms(Xa, t, ya)  # étape 1 : un modèle de réponse par groupe
        mu = self._mu(Xa)
        c = self.control
        self.effect_models_: dict[int, tuple[XGBRegressor, XGBRegressor]] = {}
        for a in self.actions_:
            in_a, in_c = t == a, t == c
            # étape 2 : effets imputés
            #   traité  : ce qu'il a fait - ce qu'il aurait fait sans action (prédit)
            #   témoin  : ce qu'il aurait fait avec l'action (prédit) - ce qu'il a fait
            d_treated = ya[in_a] - mu[c][in_a]
            d_control = mu[a][in_c] - ya[in_c]
            self.effect_models_[a] = (
                self._effect_model().fit(Xa[in_a], d_treated),
                self._effect_model().fit(Xa[in_c], d_control),
            )
        return self

    def predict(self, X) -> np.ndarray:
        Xa = np.asarray(X, dtype=np.float64)
        cols = []
        for a in self.actions_:
            m_treated, m_control = self.effect_models_[a]
            # étape 3 : pondération par la propension relative de l'action face au témoin
            g = self.propensity_[a] / (self.propensity_[a] + self.propensity_[self.control])
            cols.append(g * m_control.predict(Xa) + (1 - g) * m_treated.predict(Xa))
        return np.column_stack(cols)


class DRLearner(BaseUplift):
    name = "DR-learner"

    def __init__(self, *args, n_folds: int = 5, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n_folds = n_folds

    def pseudo_outcomes(self, Xa: np.ndarray, t: np.ndarray, ya: np.ndarray) -> np.ndarray:
        """Pseudo-réponses doublement robustes, n x K, calculées en cross-fitting."""
        groups = [self.control, *self.actions_]
        mu = {g: np.zeros(len(ya)) for g in groups}
        strata = t.astype(np.int64) * 2 + (ya > np.median(ya)).astype(np.int64)
        folds = StratifiedKFold(self.n_folds, shuffle=True, random_state=self.seed)
        for train, test in folds.split(Xa, strata):
            for g in groups:
                rows = train[t[train] == g]
                model = self._response_model().fit(Xa[rows], ya[rows])
                mu[g][test] = self._predict_response(model, Xa[test])
        c, e = self.control, self.propensity_
        resid_c = (t == c) * (ya - mu[c]) / e[c]
        return np.column_stack(
            [mu[a] - mu[c] + (t == a) * (ya - mu[a]) / e[a] - resid_c for a in self.actions_]
        )

    def fit(self, X, treatment, y) -> DRLearner:
        Xa, t, ya = self._prepare(X, treatment, y)
        phi = self.pseudo_outcomes(Xa, t, ya)
        self.final_models_ = [
            self._effect_model().fit(Xa, phi[:, j]) for j in range(len(self.actions_))
        ]
        return self

    def predict(self, X) -> np.ndarray:
        Xa = np.asarray(X, dtype=np.float64)
        return np.column_stack([m.predict(Xa) for m in self.final_models_])


LEARNERS: dict[str, type[BaseUplift]] = {
    cls.name: cls for cls in (SLearner, TLearner, XLearner, DRLearner)
}
