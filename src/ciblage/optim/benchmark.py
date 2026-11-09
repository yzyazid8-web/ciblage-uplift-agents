"""Rapport de la phase 3 : qualité des solveurs, puis valeur métier de l'allocation optimisée."""

from __future__ import annotations

import time
from datetime import date

import numpy as np
import pandas as pd

from ciblage import config
from ciblage.data import hillstrom
from ciblage.data import simulate as sim
from ciblage.optim import exact, greedy, lagrange, robust
from ciblage.optim.problem import Problem, evaluate, from_uplift, violations
from ciblage.reporting import md_table, n_fmt, pts
from ciblage.uplift import chosen, policy, tuning
from ciblage.uplift.select import fit_response

OUTPUT = config.REPORTS_DIR / "phase3_optim.md"
COSTS = np.array([0.12, 0.10])  # €/envoi : e-mail homme, e-mail femme (hypothèse)
PROP = {0: 1 / 3, 1: 1 / 3, 2: 1 / 3}
ACTIONS = ["email_homme", "email_femme"]


def prepare(seed: int = config.SEED):
    """Données, découpage de la phase 2, uplift prédit (modèle entraîné sur l'entraînement)."""
    h = hillstrom.load()
    X = hillstrom.encode_features(h).to_numpy()
    t, y = h["treatment"].to_numpy(), h["visit"].to_numpy()
    tr, te = tuning.holdout_split(t, y, test_size=chosen.TEST_SIZE, seed=seed)
    model = chosen.LEARNER(params=chosen.PARAMS, propensity=PROP, seed=seed).fit(
        X[tr], t[tr], y[tr]
    )
    uplift = model.predict(X)
    response = fit_response(X[tr], t[tr], y[tr], seed).predict_proba(X)[:, 1]
    return X, t, y, tr, te, uplift, response


# 1. solveurs
def _instance(uplift: np.ndarray, n: int, scenario: str, seed: int) -> Problem:
    """Instance de taille n tirée de l'uplift prédit (avec un léger bruit pour éviter les doublons)."""
    rng = np.random.default_rng(seed)
    u = uplift[rng.integers(0, len(uplift), n)] + rng.normal(0, 0.002, (n, 2))
    budget = 0.3 * n * COSTS[0]  # de quoi contacter 30 % des clients avec l'e-mail homme
    capacity = [np.inf, 0.15 * n] if scenario == "budget + capacité" else None
    return from_uplift(u, COSTS, budget=budget, capacity=capacity, action_names=ACTIONS)


def bench_solvers(uplift: np.ndarray, seed: int = config.SEED) -> str:
    rows = []
    for scenario in ("budget seul", "budget + capacité"):
        for n in (1_000, 5_000, 20_000, 64_000, 1_000_000):
            p = _instance(uplift, n, scenario, seed)
            results: dict[str, greedy.HeuristicResult | exact.ExactResult] = {
                "glouton": greedy.solve_greedy(p),
                "lagrangien (maison)": lagrange.solve_lagrange(p),
            }
            if n <= 64_000:
                results["relaxation linéaire (HiGHS)"] = exact.solve_lp(p)
            if n <= 5_000:
                results["MILP exact (HiGHS)"] = exact.solve_milp(p, time_limit=120)
            bounds = [r.bound for r in results.values() if np.isfinite(r.bound)]
            best_bound = min(bounds)
            for name, r in results.items():
                assert violations(p, r.assignment) == [], f"{name} infaisable"
                rows.append(
                    {
                        "scénario": scenario,
                        "clients": n_fmt(n),
                        "méthode": name,
                        "objectif (visites en plus)": f"{r.objective:,.1f}".replace(",", " "),
                        "écart à la meilleure borne": f"{100 * (best_bound - r.objective) / best_bound:.3f} %",
                        "temps (s)": f"{r.seconds:.2f}",
                    }
                )
    return (
        "## 1. Solveurs\n\n"
        "Instances tirées de l'uplift prédit sur Hillstrom ; budget = coût de 30 % des clients en "
        "e-mail homme ; scénario « capacité » : au plus 15 % de clients en e-mail femme. "
        "« Écart à la meilleure borne » = perte maximale possible par rapport à l'optimum "
        "(la borne est prouvée : relaxation linéaire, dual lagrangien ou MILP). Toutes les "
        "solutions ont été vérifiées faisables.\n\n" + md_table(pd.DataFrame(rows))
    )


# 2. valeur métier
def _top_policy(score: np.ndarray, action: int, n_contact: int) -> np.ndarray:
    pi = np.zeros(len(score), dtype=np.int64)
    pi[np.argsort(-score, kind="mergesort")[:n_contact]] = action
    return pi


def business_value(
    t: np.ndarray,
    y: np.ndarray,
    te: np.ndarray,
    uplift: np.ndarray,
    response: np.ndarray,
    seed: int,
) -> str:
    yt, tt, ut, rt = y[te], t[te], uplift[te], response[te]
    n = len(te)
    sections = []
    summary_rows = []
    for frac in (0.1, 0.2, 0.3, 0.5):
        budget = frac * n * COSTS[0]
        n_mens = int(budget // COSTS[0])
        problem = from_uplift(ut, COSTS, budget=budget, action_names=ACTIONS)
        opt = lagrange.solve_lagrange(problem).assignment
        pols = {
            "classique (e-mail homme)": _top_policy(rt, 1, n_mens),
            "uplift, e-mail homme seul": _top_policy(ut[:, 0], 1, n_mens),
            "optimisé (client + e-mail)": opt,
        }
        cmp = policy.compare_policies(
            yt, tt, pols, PROP, baseline="classique (e-mail homme)", seed=seed
        )
        for name, v in cmp.items():
            ev = evaluate(problem, pols[name])
            summary_rows.append(
                {
                    "budget": f"{budget:,.0f} €".replace(",", " "),
                    "politique": name,
                    "contactés": f"{n_fmt(ev['n_contacted'])} (homme {n_fmt(ev['n_email_homme'])}, femme {n_fmt(ev['n_email_femme'])})",
                    "dépense": f"{ev['spend']:,.0f} €".replace(",", " "),
                    "taux de visite": f"{100 * v['value']:.2f} %",
                    "gain vs classique (pts)": f"{pts(v['gain'])} [{pts(v['gain_low'])} ; {pts(v['gain_high'])}]",
                    "visites en plus / 1 000 clients": f"{1000 * v['gain']:+.1f}",
                }
            )
    sections.append(
        "## 2. Valeur métier sur le TEST de Hillstrom (IPW, IC 95 % bootstrap apparié)\n\n"
        f"{n_fmt(n)} clients jamais vus par les modèles. Pour chaque budget, les trois politiques "
        "dépensent au plus la même somme. « classique » = les clients les plus susceptibles de "
        "visiter reçoivent l'e-mail homme (le meilleur en moyenne) ; « optimisé » = relaxation "
        "lagrangienne sur l'uplift prédit, qui choisit QUI contacter ET avec QUEL e-mail, en tenant "
        "compte de son coût.\n\n" + md_table(pd.DataFrame(summary_rows))
    )
    return "\n\n".join(sections)


def simulator_value(seed: int = config.SEED, n: int = 64_000) -> str:
    """Même comparaison sur le simulateur : la valeur VRAIE de chaque politique est connue
    (réponses potentielles), donc aucun bruit d'estimation, et un « oracle » (uplift vrai) sert
    de plafond. Montre ce que la méthode apporte quand l'uplift varie réellement d'un client à l'autre."""
    df = sim.simulate(n, seed=seed)
    X, t, y = df[sim.FEATURES].to_numpy(), df["treatment"].to_numpy(), df["y"].to_numpy()
    tr, te = tuning.holdout_split(t, y, test_size=chosen.TEST_SIZE, seed=seed)
    u = chosen.LEARNER(params=chosen.PARAMS, seed=seed).fit(X[tr], t[tr], y[tr]).predict(X[te])
    r = fit_response(X[tr], t[tr], y[tr], seed).predict_proba(X[te])[:, 1]
    potential = df[["y0", "y1", "y2"]].to_numpy()[te]
    truth = df[["tau_1", "tau_2"]].to_numpy()[te]
    m = len(te)

    def value(pi: np.ndarray) -> float:
        return float(potential[np.arange(m), pi].mean())

    base = value(np.zeros(m, dtype=np.int64))
    rows = []
    for frac in (0.1, 0.3, 0.5):
        budget = frac * m * COSTS[0]
        pols = {
            "classique (e-mail homme)": _top_policy(r, 1, int(budget // COSTS[0])),
            "optimisé (uplift prédit)": lagrange.solve_lagrange(
                from_uplift(u, COSTS, budget=budget, action_names=ACTIONS)
            ).assignment,
            "oracle (uplift VRAI, inatteignable)": lagrange.solve_lagrange(
                from_uplift(truth, COSTS, budget=budget, action_names=ACTIONS)
            ).assignment,
        }
        oracle_gain = value(pols["oracle (uplift VRAI, inatteignable)"]) - base
        for name, pi in pols.items():
            gain = value(pi) - base
            rows.append(
                {
                    "budget": f"{100 * frac:.0f} %",
                    "politique": name,
                    "taux de visite (vrai)": f"{100 * value(pi):.2f} %",
                    "visites en plus vs personne (pts)": pts(gain),
                    "part du gain de l'oracle": f"{100 * gain / oracle_gain:.0f} %",
                }
            )
    return (
        f"## 4. Contrôle sur le simulateur ({n_fmt(n)} clients, valeurs VRAIES)\n\n"
        "Mêmes modèles, mêmes coûts, même protocole (test de 30 %), mais l'uplift varie fortement "
        "d'un client à l'autre (persuadables, « ne pas déranger », sensibilité propre à chaque "
        f"e-mail). Sans aucun envoi, le taux de visite vrai est de {100 * base:.2f} %.\n\n"
        + md_table(pd.DataFrame(rows))
    )


def robust_value(
    X: np.ndarray, t: np.ndarray, y: np.ndarray, tr: np.ndarray, te: np.ndarray, seed: int
) -> str:
    """À budget fixé (30 %), l'optimisation prudente (moyenne - z x écart-type) fait-elle mieux ?"""
    mean, std = robust.uplift_with_uncertainty(
        chosen.LEARNER, chosen.PARAMS, X[tr], t[tr], y[tr], X[te], propensity=PROP, seed=seed
    )
    yt, tt = y[te], t[te]
    budget = 0.3 * len(te) * COSTS[0]
    pols = {}
    for z in (0.0, 0.5, 1.0, 2.0):
        score = robust.prudent_score(mean, std, z)
        pols[f"z = {z:g}"] = lagrange.solve_lagrange(
            from_uplift(score, COSTS, budget=budget, action_names=ACTIONS)
        ).assignment
    cmp = policy.compare_policies(yt, tt, pols, PROP, baseline="z = 0", seed=seed)
    rows = []
    for name, v in cmp.items():
        ev = evaluate(from_uplift(mean, COSTS, budget=budget, action_names=ACTIONS), pols[name])
        rows.append(
            {
                "prudence": name,
                "contactés": f"{n_fmt(ev['n_contacted'])} (homme {n_fmt(ev['n_email_homme'])}, femme {n_fmt(ev['n_email_femme'])})",
                "taux de visite": f"{100 * v['value']:.2f} %",
                "gain vs sans prudence (pts)": f"{pts(v['gain'])} [{pts(v['gain_low'])} ; {pts(v['gain_high'])}]",
            }
        )
    return (
        "## 3. Optimisation prudente (budget 30 %, TEST)\n\n"
        "Uplift = moyenne de 5 modèles entraînés sur des sous-échantillons ; score optimisé = "
        "moyenne - z x écart-type entre modèles (z = 0 : aucune prudence).\n\n"
        f"Dispersion moyenne entre modèles : {100 * std.mean():.2f} pts (e-mail homme "
        f"{100 * std[:, 0].mean():.2f}, femme {100 * std[:, 1].mean():.2f}), pour un uplift "
        f"prédit moyen de {100 * mean[:, 0].mean():.2f} et {100 * mean[:, 1].mean():.2f} pts.\n\n"
        + md_table(pd.DataFrame(rows))
    )


def build() -> str:
    start = time.perf_counter()
    X, t, y, tr, te, uplift, response = prepare()
    parts = [
        "# Phase 3 : optimisation de l'allocation sous budget",
        f"Généré le {date.today().isoformat()} par `python -m ciblage.optim.benchmark`. "
        f"Uplift : {chosen.LEARNER.name} réglé en phase 2 ({chosen.PARAMS}), entraîné sur les "
        f"{n_fmt(len(tr))} clients d'entraînement. Coûts supposés : e-mail homme {COSTS[0]:.2f} €, "
        f"e-mail femme {COSTS[1]:.2f} €.",
        bench_solvers(uplift),
        business_value(t, y, te, uplift, response, config.SEED),
        robust_value(X, t, y, tr, te, config.SEED),
        simulator_value(config.SEED),
        f"*Durée totale : {(time.perf_counter() - start) / 60:.1f} min.*",
    ]
    return "\n\n".join(parts) + "\n"


def main() -> None:
    config.REPORTS_DIR.mkdir(exist_ok=True)
    OUTPUT.write_text(build(), encoding="utf-8")
    print(f"Rapport écrit : {OUTPUT}")


if __name__ == "__main__":
    main()
