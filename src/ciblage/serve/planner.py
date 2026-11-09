"""Planificateur : d'un scénario (budget, coûts, contraintes) à un plan d'envoi, avec ses preuves."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from ciblage import config
from ciblage.data import hillstrom
from ciblage.optim import lagrange
from ciblage.optim.problem import evaluate, from_uplift, violations
from ciblage.uplift import chosen, policy, tuning
from ciblage.uplift.select import fit_response

PROP = {0: 1 / 3, 1: 1 / 3, 2: 1 / 3}  # protocole de la campagne Hillstrom
ACTIONS = ["email_homme", "email_femme"]
LABELS = {0: "rien", 1: "e-mail homme", 2: "e-mail femme"}


@dataclass(frozen=True)
class Scenario:
    """Paramètres d'une demande. Les coûts sont fournis par l'utilisateur (absents des données)."""

    budget: float
    cost_homme: float = 0.12
    cost_femme: float = 0.10
    max_homme: int | None = None  # capacité maximale d'envois (None = pas de limite)
    max_femme: int | None = None
    objective: Literal["visites", "profit"] = "visites"
    value_per_visit: float = 1.0  # € rapporté par visite supplémentaire (objectif « profit »)

    def errors(self) -> list[str]:
        """Problèmes de saisie, en clair (liste vide = scénario valide)."""
        out = []
        if not np.isfinite(self.budget) or self.budget < 0:
            out.append("Le budget doit être un montant positif.")
        if self.cost_homme < 0 or self.cost_femme < 0:
            out.append("Les coûts des e-mails doivent être positifs ou nuls.")
        for name, cap in (("homme", self.max_homme), ("femme", self.max_femme)):
            if cap is not None and cap < 0:
                out.append(f"La capacité de l'e-mail {name} doit être positive.")
        if self.objective not in ("visites", "profit"):
            out.append("L'objectif doit être « visites » ou « profit ».")
        if self.objective == "profit" and self.value_per_visit <= 0:
            out.append("Pour l'objectif « profit », la valeur d'une visite doit être positive.")
        return out


@dataclass
class Plan:
    scenario: Scenario
    assignment: np.ndarray  # 0 = rien, 1 = e-mail homme, 2 = e-mail femme
    summary: dict[str, float]
    bound_gap: float  # écart maximal prouvé à l'optimum (relaxation lagrangienne)
    seconds: float
    by_segment: pd.DataFrame
    comparison: pd.DataFrame = field(default_factory=pd.DataFrame)

    def to_dict(self) -> dict:
        """Version sérialisable (pour l'API et les agents)."""
        return {
            "scenario": asdict(self.scenario),
            "summary": self.summary,
            "bound_gap": self.bound_gap,
            "seconds": self.seconds,
            "by_segment": self.by_segment.to_dict(orient="records"),
            "comparison": self.comparison.to_dict(orient="records"),
        }


def _segments(df: pd.DataFrame) -> pd.Series:
    return pd.Series(
        np.select(
            [(df["mens"] == 1) & (df["womens"] == 1), df["mens"] == 1],
            ["homme et femme", "homme seulement"],
            "femme seulement",
        ),
        index=df.index,
    )


class Planner:
    """Entraîne une fois les modèles, puis produit des plans."""

    def __init__(self, seed: int = config.SEED) -> None:
        h = hillstrom.load()
        X = hillstrom.encode_features(h).to_numpy()
        t, y = h["treatment"].to_numpy(), h["visit"].to_numpy()
        tr, te = tuning.holdout_split(t, y, test_size=chosen.TEST_SIZE, seed=seed)
        model = chosen.LEARNER(params=chosen.PARAMS, propensity=PROP, seed=seed).fit(
            X[tr], t[tr], y[tr]
        )
        self.seed = seed
        self.uplift = model.predict(X[te])  # (n_test, 2) : visites en plus par e-mail
        self.response = fit_response(X[tr], t[tr], y[tr], seed).predict_proba(X[te])[:, 1]
        self.treatment, self.visit = t[te], y[te]
        self.clients = h.iloc[te].reset_index(drop=True)
        self.segment = _segments(self.clients)

    @property
    def n_clients(self) -> int:
        return len(self.visit)

    def _problem(self, sc: Scenario):
        cost = [sc.cost_homme, sc.cost_femme]
        capacity = [np.inf if c is None else c for c in (sc.max_homme, sc.max_femme)]
        if sc.objective == "profit":
            return from_uplift(
                self.uplift,
                cost,
                "profit",
                sc.value_per_visit,
                sc.budget,
                capacity,
                action_names=ACTIONS,
            )
        return from_uplift(
            self.uplift, cost, "outcome", 1.0, sc.budget, capacity, action_names=ACTIONS
        )

    def plan(self, sc: Scenario, compare: bool = True, n_boot: int = 300) -> Plan:
        errs = sc.errors()
        if errs:
            raise ValueError(" ".join(errs))
        start = time.perf_counter()
        problem = self._problem(sc)
        res = lagrange.solve_lagrange(problem)
        a = res.assignment
        assert violations(problem, a) == [], "Plan infaisable : bogue de l'optimiseur"
        ev = evaluate(problem, a)
        chosen_rows = np.flatnonzero(a > 0)
        predicted_visits = float(self.uplift[chosen_rows, a[chosen_rows] - 1].sum())
        summary = {
            "clients_portefeuille": self.n_clients,
            "clients_contactes": ev["n_contacted"],
            "email_homme": ev["n_email_homme"],
            "email_femme": ev["n_email_femme"],
            "depense": ev["spend"],
            "budget_inutilise": max(0.0, sc.budget - ev["spend"])
            if np.isfinite(sc.budget)
            else 0.0,
            "visites_en_plus_predites": predicted_visits,
            "cout_par_visite_en_plus": ev["spend"] / predicted_visits
            if predicted_visits > 0
            else float("nan"),
        }
        if sc.objective == "profit":
            summary["profit_predit"] = sc.value_per_visit * predicted_visits - ev["spend"]
        plan = Plan(
            scenario=sc,
            assignment=a,
            summary=summary,
            bound_gap=res.gap,
            seconds=time.perf_counter() - start,
            by_segment=self._by_segment(a),
        )
        if compare:
            plan.comparison = self.compare(sc, a, n_boot=n_boot)
        return plan

    def _by_segment(self, a: np.ndarray) -> pd.DataFrame:
        rows = []
        for seg in ("homme seulement", "femme seulement", "homme et femme"):
            m = (self.segment == seg).to_numpy()
            rows.append(
                {
                    "segment": seg,
                    "clients": int(m.sum()),
                    "e-mail homme": int((a[m] == 1).sum()),
                    "e-mail femme": int((a[m] == 2).sum()),
                    "rien": int((a[m] == 0).sum()),
                    "uplift prédit homme (pts)": round(100 * float(self.uplift[m, 0].mean()), 2),
                    "uplift prédit femme (pts)": round(100 * float(self.uplift[m, 1].mean()), 2),
                }
            )
        return pd.DataFrame(rows)

    def compare(self, sc: Scenario, a: np.ndarray, n_boot: int = 300) -> pd.DataFrame:
        """Valeur MESURÉE (IPW sur le test randomisé) du plan face aux alternatives, même budget."""
        n_classic = (
            int(min(self.n_clients, sc.budget // sc.cost_homme))
            if sc.cost_homme > 0
            else self.n_clients
        )
        if sc.max_homme is not None:
            n_classic = min(n_classic, sc.max_homme)
        classic = np.zeros(self.n_clients, dtype=np.int64)
        classic[np.argsort(-self.response, kind="mergesort")[:n_classic]] = 1
        pols = {
            "personne": np.zeros(self.n_clients, dtype=np.int64),
            "ciblage classique": classic,
            "plan optimisé": a,
            "tout le monde (e-mail homme)": np.ones(self.n_clients, dtype=np.int64),
        }
        cmp = policy.compare_policies(
            self.visit,
            self.treatment,
            pols,
            PROP,
            baseline="ciblage classique",
            n_boot=n_boot,
            seed=self.seed,
        )
        costs = np.array([0.0, sc.cost_homme, sc.cost_femme])
        rows = []
        for name, v in cmp.items():
            rows.append(
                {
                    "politique": name,
                    "contactés": int((pols[name] > 0).sum()),
                    "dépense (€)": float(costs[pols[name]].sum()),
                    "taux de visite": v["value"],
                    "ic_bas": v["value_low"],
                    "ic_haut": v["value_high"],
                    "gain vs classique (pts)": 100 * v["gain"],
                    "gain_bas": 100 * v["gain_low"],
                    "gain_haut": 100 * v["gain_high"],
                }
            )
        return pd.DataFrame(rows)
