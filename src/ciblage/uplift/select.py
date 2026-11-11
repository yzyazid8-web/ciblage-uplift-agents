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
from ciblage.reporting import md_table
from ciblage.uplift import calibration, metrics, policy, tuning
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
def _n(v: int) -> str:
    return f"{v:,}".replace(",", "\u202f")  # espace fine insécable


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
        f"## 1. Simulateur ({_n(n)} clients, uplift vrai connu)\n\n"
        + "PEHE = erreur quadratique moyenne entre uplift prédit et uplift vrai (plus bas = mieux). "
        "Prédictions hors échantillon (validation croisée à 5 plis).\n\n"
        + md_table(pd.DataFrame(rows))
    )


# 2. Hillstrom
def fit_response(X, t, y, seed: int = config.SEED) -> XGBClassifier:
    """Modèle du ciblage classique, P(réponse | x), appris sur les clients ayant reçu un e-mail."""
    Xa, ta, ya = np.asarray(X, dtype=np.float64), np.asarray(t), np.asarray(y, dtype=np.float64)
    clf = XGBClassifier(**DEFAULT_PARAMS, random_state=seed, eval_metric="logloss")
    return clf.fit(Xa[ta != 0], ya[ta != 0] > 0)


def _policy_table(cmp: dict[str, dict[str, float]], unit: str, baseline: str) -> str:
    def fmt(v: float) -> str:
        return _pts(v) if unit == "pts" else f"{v:+.3f}"

    def level(v: float) -> str:
        return f"{100 * v:.2f} %" if unit == "pts" else f"{v:.3f} $"

    rows = [
        {
            "politique": k,
            "contactés": f"{100 * v['share_treated']:.0f} %",
            "résultat moyen par client": f"{level(v['value'])} [{level(v['value_low'])} ; {level(v['value_high'])}]",
            f"gain vs {baseline} ({unit})": f"{fmt(v['gain'])} [{fmt(v['gain_low'])} ; {fmt(v['gain_high'])}]",
        }
        for k, v in cmp.items()
    ]
    return md_table(pd.DataFrame(rows))


def bench_hillstrom(seed: int = config.SEED) -> tuple[str, str]:
    h = hillstrom.load()
    X = hillstrom.encode_features(h).to_numpy()
    t = h["treatment"].to_numpy()
    y = h["visit"].to_numpy()
    prop = {0: 1 / 3, 1: 1 / 3, 2: 1 / 3}  # protocole de l'essai
    names = {1: "homme", 2: "femme"}
    tr, te = tuning.holdout_split(t, y, test_size=0.3, seed=seed)
    sections = [
        "## 2. Hillstrom : visite (vraie campagne, 64 000 clients)\n\n"
        f"**Protocole** : {_n(len(te))} clients (30 %) mis de côté AVANT tout réglage ; réglage par "
        f"validation croisée à 5 plis sur les {_n(len(tr))} autres, critère fixé à l'avance = Qini "
        "moyen des deux e-mails ; le test ne sert qu'une fois, pour tous les chiffres ci-dessous."
    ]

    # 2a. Réglage (entraînement seulement)
    results = {
        name: tuning.tune(cls, X[tr], t[tr], y[tr], cross_val_uplift, propensity=prop, seed=seed)
        for name, cls in LEARNERS.items()
    }
    trow = []
    for name, r in results.items():
        has_effect = name in ("X-learner", "DR-learner")
        trow.append(
            {
                "learner": name,
                "configurations testées": len(r.table),
                "Qini CV défaut": f"{r.default_score:.4f}",
                "Qini CV réglé": f"{r.best_score:.4f}",
                "réponse retenue": r.best_params or "défaut",
                "effet retenu": r.best_effect_params or ("défaut" if has_effect else "-"),
                "durée (min)": f"{r.seconds / 60:.1f}",
            }
        )
        _log_mlflow(
            f"hillstrom-tuning/{name}",
            {
                "learner": name,
                **{f"rep_{k}": v for k, v in r.best_params.items()},
                **{f"eff_{k}": v for k, v in r.best_effect_params.items()},
            },
            {"qini_cv_default": r.default_score, "qini_cv_tuned": r.best_score},
        )
    best = max(results, key=lambda k: results[k].best_score)
    sections.append("### Réglage (entraînement uniquement)\n\n" + md_table(pd.DataFrame(trow)))

    # 2b. Modèles finaux : entraînement complet -> prédiction sur le test
    preds = {}
    for name, cls in LEARNERS.items():
        r = results[name]
        model = cls(
            params=r.best_params, effect_params=r.best_effect_params, propensity=prop, seed=seed
        )
        preds[name] = model.fit(X[tr], t[tr], y[tr]).predict(X[te])
    response = fit_response(X[tr], t[tr], y[tr], seed).predict_proba(X[te])[:, 1]
    yt, tt = y[te], t[te]
    rows = []
    candidates = [
        *preds.items(),
        ("*ciblage classique (P(visite))*", np.column_stack([response, response])),
    ]
    for name, u_ in candidates:
        rk = ranking_metrics(yt, tt, u_)
        row: dict[str, str] = {"learner": name}
        for a in (1, 2):
            row[f"Qini {names[a]}"] = f"{rk[a]['qini']:.3f}"
            row[f"uplift@30% {names[a]} (pts)"] = _pts(rk[a]["uplift_at_30"])
        rows.append(row)
    overall = {a: yt[tt == a].mean() - yt[tt == 0].mean() for a in (1, 2)}
    sections.append(
        "### Qualité du classement sur le TEST\n\n"
        f"Repère sur le test : e-mail à tout le monde = {_pts(overall[1])} pts (homme), "
        f"{_pts(overall[2])} pts (femme).\n\n" + md_table(pd.DataFrame(rows))
    )

    # 2c. Valeur des politiques sur le test, en séparant classement et choix de l'e-mail
    u = preds[best]
    rand = np.random.default_rng(seed).random(len(te))
    base = "classique homme"
    pols = {
        "personne": np.zeros(len(te), dtype=np.int64),
        "hasard 30 %, e-mail homme": policy.top_score_policy(rand, 1, BUDGET_FRAC),
        base: policy.top_score_policy(response, 1, BUDGET_FRAC),
        f"uplift ({best}) 30 %, e-mail homme": policy.top_score_policy(u[:, 0], 1, BUDGET_FRAC),
        "classique 30 %, e-mail femme": policy.top_score_policy(response, 2, BUDGET_FRAC),
        f"uplift ({best}) 30 %, e-mail femme": policy.top_score_policy(u[:, 1], 2, BUDGET_FRAC),
        f"uplift ({best}) 30 %, meilleur e-mail par client": policy.best_action_policy(
            u, [1, 2], BUDGET_FRAC
        ),
        "tout le monde, e-mail homme": np.ones(len(te), dtype=np.int64),
    }
    cmp = policy.compare_policies(yt, tt, pols, prop, baseline=base, seed=seed)
    sections.append(
        "### Valeur des politiques sur le TEST (IPW, IC 95 % bootstrap apparié)\n\n"
        f"Budget : {100 * BUDGET_FRAC:.0f} % des clients. « classique homme » = les 30 % les plus "
        "susceptibles de visiter reçoivent l'e-mail homme (pratique courante). Les lignes "
        "« e-mail homme » et « e-mail femme » isolent l'effet du CLASSEMENT ; la ligne « meilleur "
        "e-mail » ajoute le CHOIX de l'e-mail client par client.\n\n"
        + _policy_table(cmp, "pts", base)
    )

    # 2c bis. Comparaison à e-mail egal : chaque e-mail contre le ciblage classique du même e-mail
    fair_rows = []
    for a in (1, 2):
        ref = f"classique 30 %, e-mail {names[a]}"
        duel = {ref: policy.top_score_policy(response, a, BUDGET_FRAC)}
        for name, u_ in preds.items():
            duel[f"uplift ({name}) 30 %, e-mail {names[a]}"] = policy.top_score_policy(
                u_[:, a - 1], a, BUDGET_FRAC
            )
        res = policy.compare_policies(yt, tt, duel, prop, baseline=ref, seed=seed)
        for k, v in res.items():
            if k == ref:
                continue
            fair_rows.append(
                {
                    "e-mail": names[a],
                    "politique": k,
                    "taux de visite": f"{100 * v['value']:.2f} %",
                    "gain vs classique, même e-mail (pts)": f"{_pts(v['gain'])} [{_pts(v['gain_low'])} ; {_pts(v['gain_high'])}]",
                }
            )
    sections.append(
        "### A e-mail égal : l'uplift vise-t-il mieux que le ciblage classique ? (TEST)\n\n"
        "Chaque ligne compare un learner au ciblage classique qui envoie LE MÊME e-mail aux "
        "30 % les plus susceptibles de visiter.\n\n" + md_table(pd.DataFrame(fair_rows))
    )

    # 2d. Calibration du meilleur learner sur le test
    cal_rows = []
    for j, a in enumerate((1, 2)):
        mask = (tt == a) | (tt == 0)
        tab = calibration.calibration_table(
            yt[mask], u[mask, j], (tt[mask] == a).astype(int), bins=5
        )
        for c in tab.itertuples():
            cal_rows.append(
                {
                    "e-mail": names[a],
                    "tranche (1 = plus fort uplift prédit)": c.tranche,
                    "prédit (pts)": _pts(c.predit),
                    "observé (pts)": f"{_pts(c.observe)} [{_pts(c.ci_low)} ; {_pts(c.ci_high)}]",
                }
            )
    sections.append(f"### Calibration du {best} sur le TEST\n\n" + md_table(pd.DataFrame(cal_rows)))

    # 2e. Dépense (dollars) : même learner et mêmes réglages, cible continue
    spend = h["spend"].to_numpy()
    rb = results[best]
    model_s = LEARNERS[best](
        outcome="continuous",
        params=rb.best_params,
        effect_params=rb.best_effect_params,
        propensity=prop,
        seed=seed,
    )
    u_s = model_s.fit(X[tr], t[tr], spend[tr]).predict(X[te])
    resp_s = fit_response(X[tr], t[tr], spend[tr], seed).predict_proba(X[te])[:, 1]
    pols_s = {
        "personne": pols["personne"],
        base: policy.top_score_policy(resp_s, 1, BUDGET_FRAC),
        f"uplift ({best}) 30 %, meilleur e-mail par client": policy.best_action_policy(
            u_s, [1, 2], BUDGET_FRAC
        ),
        "tout le monde, e-mail homme": pols["tout le monde, e-mail homme"],
    }
    cmp_s = policy.compare_policies(spend[te], tt, pols_s, prop, baseline=base, seed=seed)
    sections.append(
        "## 3. Hillstrom : dépense sur le TEST (achats rares, environ 1 % : estimations bruitées)\n\n"
        "Réglages repris de la visite (pas de réglage spécifique : trop peu d'achats).\n\n"
        + _policy_table(cmp_s, "$", base)
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
        f"## 4. Criteo : visite ({_n(n)} lignes tirées de l'échantillon, 85 % traités ; réglages par défaut)\n\n"
        + f"Repère : effet moyen de la publicité = {_pts(overall)} pts. Validation croisée à 3 plis.\n\n"
        + md_table(pd.DataFrame(rows))
    )


def build() -> str:
    start = time.perf_counter()
    hill, best = bench_hillstrom()
    parts = [
        "# Phase 2 : moteur d'uplift",
        f"Généré le {date.today().isoformat()} par `python -m ciblage.uplift.select`. "
        f"Modèle de base : XGBoost. Réglages par défaut ({DEFAULT_PARAMS['n_estimators']} arbres, "
        f"profondeur {DEFAULT_PARAMS['max_depth']}) pour le simulateur et Criteo ; réglés par "
        "validation croisée sur Hillstrom. Toutes les prédictions sont hors échantillon.",
        bench_simulator(),
        hill,
        bench_criteo(),
        f"*Learner retenu sur Hillstrom (meilleur Qini en validation croisée sur l'entraînement) : **{best}**. "
        f"Durée totale : {(time.perf_counter() - start) / 60:.0f} min.*",
    ]
    return "\n\n".join(parts) + "\n"


def main() -> None:
    config.REPORTS_DIR.mkdir(exist_ok=True)
    OUTPUT.write_text(build(), encoding="utf-8")
    print(f"Rapport écrit : {OUTPUT}")


if __name__ == "__main__":
    main()
