"""Comparaison des learners d'uplift en validation croisée, et rapport de la phase 2.

Usage : python -m ciblage.uplift.select   ->   reports/phase2_uplift.md (+ suivi MLflow local)

Toutes les prédictions évaluées sont HORS ÉCHANTILLON (« out-of-fold ») : chaque client est
noté par un modèle qui ne l'a jamais vu. Sans cela, un modèle qui apprend par cœur paraîtrait
excellent.

Trois niveaux de preuve :
1. Simulateur : l'uplift vrai est connu -> erreur (PEHE) et corrélation directement mesurables.
2. Hillstrom : vraie campagne -> métriques de classement (Qini, AUUC, uplift@30 %) et surtout
   VALEUR des politiques de ciblage (IPW), comparée au ciblage classique.
3. Criteo : autre secteur, autre échelle -> les conclusions tiennent-elles ?
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import date
from functools import partial
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from ciblage import config
from ciblage.data import criteo, hillstrom
from ciblage.data import simulate as sim
from ciblage.uplift import calibration, metrics, policy
from ciblage.uplift.learners import DEFAULT_PARAMS, LEARNERS, BaseUplift

OUTPUT = config.REPORTS_DIR / "phase2_uplift.md"
BUDGET_FRAC = 0.30  # part des clients qu'on s'autorise à contacter


def _folds(t: np.ndarray, y: np.ndarray, n_folds: int, seed: int):
    strata = t.astype(np.int64) * 2 + (y > np.median(y)).astype(np.int64)
    return StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(np.zeros(len(t)), strata)


def cross_val_uplift(
    make: Callable[[], BaseUplift], X, t, y, n_folds: int = 5, seed: int = config.SEED
) -> np.ndarray:
    """Uplift prédit hors échantillon pour chaque client (matrice n x K)."""
    Xa, ta, ya = np.asarray(X, dtype=np.float64), np.asarray(t), np.asarray(y, dtype=np.float64)
    n_actions = len(np.unique(ta)) - 1
    oof = np.zeros((len(ya), n_actions))
    for train, test in _folds(ta, ya, n_folds, seed):
        model = make().fit(Xa[train], ta[train], ya[train])
        oof[test] = model.predict(Xa[test])
    return oof


def cross_val_response(X, t, y, n_folds: int = 5, seed: int = config.SEED) -> np.ndarray:
    """Score du ciblage classique : P(réponse | x) appris sur les clients ayant reçu un e-mail."""
    Xa, ta, ya = np.asarray(X, dtype=np.float64), np.asarray(t), np.asarray(y, dtype=np.float64)
    score = np.zeros(len(ya))
    for train, test in _folds(ta, ya, n_folds, seed):
        treated = train[ta[train] != 0]
        clf = XGBClassifier(**DEFAULT_PARAMS, random_state=seed, eval_metric="logloss")
        clf.fit(Xa[treated], ya[treated] > 0)
        score[test] = clf.predict_proba(Xa[test])[:, 1]
    return score


def ranking_metrics(y, t, uplift: np.ndarray) -> dict[int, dict[str, float]]:
    """Qini, AUUC et uplift@30 % de chaque action contre le témoin."""
    out: dict[int, dict[str, float]] = {}
    fns: dict[str, Callable[..., float]] = {
        "qini": metrics.qini,
        "auuc": metrics.auuc,
        "uplift_at_30": partial(metrics.uplift_at_k, k=BUDGET_FRAC),
    }
    for name, fn in fns.items():
        for a, v in metrics.per_treatment(fn, y, uplift, t).items():
            out.setdefault(a, {})[name] = v
    return out


def _log_mlflow(run_name: str, params: dict[str, Any], values: dict[str, float]) -> None:
    """Suivi MLflow local (facultatif : ignoré si MLflow n'est pas installé)."""
    try:
        import mlflow
    except ImportError:
        return
    mlflow.set_tracking_uri(f"sqlite:///{config.ROOT / 'mlflow.db'}")
    mlflow.set_experiment("phase2-uplift")
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_metrics({k: float(v) for k, v in values.items() if np.isfinite(v)})


# mise en forme 
def _md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def _pts(v: float) -> str:
    return f"{100 * v:+.2f}"


# simulateur
def bench_simulator(n: int = 40_000, seed: int = config.SEED) -> str:
    df = sim.simulate(n, seed=seed)
    X, t, y = df[sim.FEATURES], df["treatment"].to_numpy(), df["y"].to_numpy()
    truth = df[["tau_1", "tau_2"]].to_numpy()
    rows = []
    for name, cls in LEARNERS.items():
        start = time.perf_counter()
        oof = cross_val_uplift(cls, X, t, y, seed=seed)
        secs = time.perf_counter() - start
        pehe = np.sqrt(((oof - truth) ** 2).mean(axis=0))
        corr = [np.corrcoef(oof[:, j], truth[:, j])[0, 1] for j in range(2)]
        q = metrics.per_treatment(metrics.qini, y, oof, t)
        rows.append(
            {
                "learner": name,
                "PEHE homme (pts)": f"{100 * pehe[0]:.2f}",
                "PEHE femme (pts)": f"{100 * pehe[1]:.2f}",
                "corr. homme": f"{corr[0]:.2f}",
                "corr. femme": f"{corr[1]:.2f}",
                "Qini homme": f"{q[1]:.3f}",
                "Qini femme": f"{q[2]:.3f}",
                "temps (s)": f"{secs:.0f}",
            }
        )
        _log_mlflow(
            f"simulateur/{name}",
            {"dataset": "simulateur", "learner": name, "n": n},
            {"pehe_1": pehe[0], "pehe_2": pehe[1], "qini_1": q[1], "qini_2": q[2]},
        )
    constant = np.sqrt(((truth - truth.mean(axis=0)) ** 2).mean(axis=0))
    q_true = metrics.per_treatment(metrics.qini, y, truth, t)
    rows.append(
        {
            "learner": "*référence : effet moyen pour tous*",
            "PEHE homme (pts)": f"{100 * constant[0]:.2f}",
            "PEHE femme (pts)": f"{100 * constant[1]:.2f}",
            "corr. homme": "-",
            "corr. femme": "-",
            "Qini homme": "-",
            "Qini femme": "-",
            "temps (s)": "-",
        }
    )
    rows.append(
        {
            "learner": "*référence : uplift VRAI (inatteignable)*",
            "PEHE homme (pts)": "0.00",
            "PEHE femme (pts)": "0.00",
            "corr. homme": "1.00",
            "corr. femme": "1.00",
            "Qini homme": f"{q_true[1]:.3f}",
            "Qini femme": f"{q_true[2]:.3f}",
            "temps (s)": "-",
        }
    )
    return (
        f"## 1. Simulateur ({n:,} clients, uplift vrai connu)\n\n".replace(",", " ")
        + "PEHE = erreur quadratique moyenne entre uplift prédit et uplift vrai (plus bas = mieux). "
        "Prédictions hors échantillon (validation croisée à 5 plis).\n\n" + _md(pd.DataFrame(rows))
    )


# 2. Hillstrom 
def bench_hillstrom(seed: int = config.SEED) -> tuple[str, str]:
    h = hillstrom.load()
    X = hillstrom.encode_features(h)
    t = h["treatment"].to_numpy()
    prop = {0: 1 / 3, 1: 1 / 3, 2: 1 / 3}  # protocole de l'essai
    names = {1: "homme", 2: "femme"}
    sections = []

    # 2a. Visite : classement
    y = h["visit"].to_numpy()
    oofs: dict[str, np.ndarray] = {}
    rows = []
    for name, cls in LEARNERS.items():
        oof = cross_val_uplift(partial(cls, propensity=prop), X, t, y, seed=seed)
        oofs[name] = oof
        rk = ranking_metrics(y, t, oof)
        row: dict[str, str] = {"learner": name}
        for a in (1, 2):
            row[f"Qini {names[a]}"] = f"{rk[a]['qini']:.3f}"
            row[f"AUUC {names[a]}"] = f"{rk[a]['auuc']:.3f}"
            row[f"uplift@30% {names[a]} (pts)"] = _pts(rk[a]["uplift_at_30"])
        rows.append(row)
        _log_mlflow(
            f"hillstrom-visit/{name}",
            {"dataset": "hillstrom", "outcome": "visit", "learner": name},
            {f"{m}_{a}": v for a, d in rk.items() for m, v in d.items()},
        )
    response = cross_val_response(X, t, y, seed=seed)
    rk_resp = ranking_metrics(y, t, np.column_stack([response, response]))
    row = {"learner": "*ciblage classique (P(visite))*"}
    for a in (1, 2):
        row[f"Qini {names[a]}"] = f"{rk_resp[a]['qini']:.3f}"
        row[f"AUUC {names[a]}"] = f"{rk_resp[a]['auuc']:.3f}"
        row[f"uplift@30% {names[a]} (pts)"] = _pts(rk_resp[a]["uplift_at_30"])
    rows.append(row)
    overall = {a: h.loc[t == a, "visit"].mean() - h.loc[t == 0, "visit"].mean() for a in (1, 2)}
    sections.append(
        "## 2. Hillstrom : visite (vraie campagne, 64 000 clients)\n\n"
        "### Qualité du classement (hors échantillon)\n\n"
        f"Repère : envoyer l'e-mail à TOUT le monde donne +{100 * overall[1]:.2f} pts (homme) et "
        f"+{100 * overall[2]:.2f} pts (femme). Un bon modèle fait mieux sur ses 30 % préférés.\n\n"
        + _md(pd.DataFrame(rows))
    )

    best = max(
        oofs,
        key=lambda k: np.mean(list(metrics.per_treatment(metrics.qini, y, oofs[k], t).values())),
    )

    # 2b. Visite : valeur des politiques à budget égal
    pols = {
        "personne": np.zeros(len(h), dtype=np.int64),
        "hasard 30 % (e-mail homme)": policy.top_score_policy(
            np.random.default_rng(seed).random(len(h)), 1, BUDGET_FRAC
        ),
        "ciblage classique 30 % (e-mail homme)": policy.top_score_policy(response, 1, BUDGET_FRAC),
        f"uplift 30 % ({best}, meilleur e-mail)": policy.best_action_policy(
            oofs[best], [1, 2], BUDGET_FRAC
        ),
        "tout le monde (e-mail homme)": np.ones(len(h), dtype=np.int64),
    }
    cmp = policy.compare_policies(
        y, t, pols, prop, baseline="ciblage classique 30 % (e-mail homme)", seed=seed
    )
    prow = [
        {
            "politique": k,
            "clients contactés": f"{100 * v['share_treated']:.0f} %",
            "taux de visite": f"{100 * v['value']:.2f} % [{100 * v['value_low']:.2f} ; {100 * v['value_high']:.2f}]",
            "gain vs ciblage classique (pts)": f"{_pts(v['gain'])} [{_pts(v['gain_low'])} ; {_pts(v['gain_high'])}]",
        }
        for k, v in cmp.items()
    ]
    sections.append(
        f"### Valeur des politiques de ciblage (IPW, IC 95 % bootstrap apparié)\n\n"
        f"Budget : {100 * BUDGET_FRAC:.0f} % des clients contactés. Le ciblage par uplift choisit "
        "AUSSI le meilleur e-mail pour chaque client.\n\n" + _md(pd.DataFrame(prow))
    )

    # 2c. Calibration du meilleur learner
    cal_rows = []
    for j, a in enumerate((1, 2)):
        mask = (t == a) | (t == 0)
        tab = calibration.calibration_table(
            y[mask], oofs[best][mask, j], (t[mask] == a).astype(int), bins=5
        )
        for r in tab.itertuples():
            cal_rows.append(
                {
                    "e-mail": names[a],
                    "tranche (1 = plus fort uplift prédit)": r.tranche,
                    "prédit (pts)": _pts(r.predit),
                    "observé (pts)": f"{_pts(r.observe)} [{_pts(r.ci_low)} ; {_pts(r.ci_high)}]",
                }
            )
    sections.append(
        f"### Calibration du {best} : l'uplift prédit correspond-il à l'uplift observé ?\n\n"
        + _md(pd.DataFrame(cal_rows))
    )

    # 2d. Dépense : valeur en dollars
    spend = h["spend"].to_numpy()
    oof_spend = cross_val_uplift(
        lambda: LEARNERS[best](outcome="continuous", propensity=prop), X, t, spend, seed=seed
    )
    resp_spend = cross_val_response(X, t, spend, seed=seed)
    pols_s = {
        "personne": pols["personne"],
        "ciblage classique 30 % (e-mail homme)": policy.top_score_policy(
            resp_spend, 1, BUDGET_FRAC
        ),
        f"uplift 30 % ({best}, meilleur e-mail)": policy.best_action_policy(
            oof_spend, [1, 2], BUDGET_FRAC
        ),
        "tout le monde (e-mail homme)": pols["tout le monde (e-mail homme)"],
    }
    cmp_s = policy.compare_policies(
        spend, t, pols_s, prop, baseline="ciblage classique 30 % (e-mail homme)", seed=seed
    )
    srow = [
        {
            "politique": k,
            "clients contactés": f"{100 * v['share_treated']:.0f} %",
            "dépense moyenne par client ($)": f"{v['value']:.3f} [{v['value_low']:.3f} ; {v['value_high']:.3f}]",
            "gain vs ciblage classique ($)": f"{v['gain']:+.3f} [{v['gain_low']:+.3f} ; {v['gain_high']:+.3f}]",
        }
        for k, v in cmp_s.items()
    ]
    sections.append(
        "## 3. Hillstrom : dépense (achats rares, environ 1 %, donc estimations bruitées)\n\n"
        + _md(pd.DataFrame(srow))
    )
    return "\n\n".join(sections), best


# 3. Criteo 
def bench_criteo(n: int = 400_000, seed: int = config.SEED) -> str:
    c = criteo.load_sample().sample(n=n, random_state=seed)
    X, t, y = c[criteo.FEATURES], c["treatment"].to_numpy(), c["visit"].to_numpy()
    rows = []
    for name, cls in LEARNERS.items():
        start = time.perf_counter()
        oof = cross_val_uplift(cls, X, t, y, n_folds=3, seed=seed)
        secs = time.perf_counter() - start
        rk = ranking_metrics(y, t, oof)[1]
        rows.append(
            {
                "learner": name,
                "Qini": f"{rk['qini']:.3f}",
                "AUUC": f"{rk['auuc']:.3f}",
                "uplift@30% (pts)": _pts(rk["uplift_at_30"]),
                "temps (s)": f"{secs:.0f}",
            }
        )
        _log_mlflow(
            f"criteo-visit/{name}",
            {"dataset": "criteo", "outcome": "visit", "learner": name, "n": n},
            {k: v for k, v in rk.items()},
        )
    response = cross_val_response(X, t, y, n_folds=3, seed=seed)
    rk = ranking_metrics(y, t, response[:, None])[1]
    rows.append(
        {
            "learner": "*ciblage classique (P(visite))*",
            "Qini": f"{rk['qini']:.3f}",
            "AUUC": f"{rk['auuc']:.3f}",
            "uplift@30% (pts)": _pts(rk["uplift_at_30"]),
            "temps (s)": "-",
        }
    )
    overall = y[t == 1].mean() - y[t == 0].mean()
    return (
        f"## 4. Criteo : visite ({n:,} lignes tirées de l'échantillon, 85 % traités)\n\n".replace(
            ",", " "
        )
        + f"Repère : effet moyen de la publicité = {_pts(overall)} pts. Validation croisée à 3 plis.\n\n"
        + _md(pd.DataFrame(rows))
    )


def build() -> str:
    start = time.perf_counter()
    hill, best = bench_hillstrom()
    parts = [
        "# Phase 2 : moteur d'uplift",
        f"Généré le {date.today().isoformat()} par `python -m ciblage.uplift.select`. "
        f"Modèle de base : XGBoost {DEFAULT_PARAMS['n_estimators']} arbres, profondeur "
        f"{DEFAULT_PARAMS['max_depth']}. Toutes les prédictions sont hors échantillon.",
        bench_simulator(),
        hill,
        bench_criteo(),
        f"*Learner retenu sur Hillstrom (meilleur Qini moyen) : **{best}**. "
        f"Durée totale : {(time.perf_counter() - start) / 60:.0f} min.*",
    ]
    return "\n\n".join(parts) + "\n"


def main() -> None:
    config.REPORTS_DIR.mkdir(exist_ok=True)
    OUTPUT.write_text(build(), encoding="utf-8")
    print(f"Rapport écrit : {OUTPUT}")


if __name__ == "__main__":
    main()
