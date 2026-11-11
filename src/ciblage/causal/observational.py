"""Phase 1 bis : estimation causale hors essai randomisé (données observationnelles)."""

from __future__ import annotations

from datetime import date

import pandas as pd

from ciblage import config
from ciblage.causal import confounding, sensitivity
from ciblage.causal.aipw import Estimate, aipw_ate, ipw_ate, naive_ate
from ciblage.causal.propensity import overlap, propensity_cv, smd_max
from ciblage.data import hillstrom
from ciblage.reporting import md_table

OUTPUT = config.REPORTS_DIR / "phase1b_observationnel.md"

ACTION = 1  # e-mail homme (effet moyen fort sur Hillstrom : biais bien visible)
COVARIATE = "history"  # montant déjà dépensé : corrélé à la visite -> bon confondant
STRENGTH = 1.5
OUTCOME = "visit"


def _pts(x: float) -> str:
    return f"{100 * x:+.2f}"


def _n(v: int) -> str:
    return f"{v:,}".replace(",", " ")  # espace fine comme séparateur de milliers (français)


def _ic(est: Estimate) -> str:
    return f"[{_pts(est.ci_low)} ; {_pts(est.ci_high)}]"


def _estimator_table(truth: Estimate, estimates: list[Estimate]) -> str:
    rows = []
    for est in estimates:
        rows.append(
            {
                "méthode": est.method,
                "effet estimé (pts)": _pts(est.ate),
                "IC 95 % (pts)": _ic(est),
                "écart à la vérité (pts)": _pts(est.ate - truth.ate),
                "significatif": "oui" if est.significant else "non",
            }
        )
    return md_table(pd.DataFrame(rows))


def build() -> str:
    h = hillstrom.load()
    name = hillstrom.TREATMENT_NAMES[ACTION]

    # Vérité terrain : l'effet mesuré sur les données randomisées complètes (non biaisé).
    full = h[h["treatment"].isin([0, ACTION])]
    y_full = full[OUTCOME].to_numpy()
    w_full = (full["treatment"] == ACTION).to_numpy().astype(int)
    truth = naive_ate(y_full, w_full)

    # Échantillon observationnel : randomisation cassée via `COVARIATE`.
    conf = confounding.make_confounded(h, action=ACTION, covariate=COVARIATE, strength=STRENGTH)
    y = conf[OUTCOME].to_numpy()
    w = (conf["treatment"] == ACTION).to_numpy().astype(int)
    X = hillstrom.encode_features(conf)

    e = propensity_cv(X, w)
    ov = overlap(e, w)
    smd_brute = smd_max(X, w, e=None)
    smd_pondere = smd_max(X, w, e=e)

    est_naive = naive_ate(y, w)
    est_ipw = ipw_ate(y, w, e)
    est_aipw = aipw_ate(X, w, y, e)

    baseline = float(y[w == 0].mean())  # taux de visite des témoins de l'échantillon biaisé
    ev = sensitivity.e_value(est_aipw.ate, est_aipw.ci_low, est_aipw.ci_high, baseline)

    biais_naif = est_naive.ate - truth.ate
    parts = [
        "# Phase 1 bis : estimation causale hors essai randomisé",
        f"Généré le {date.today().isoformat()} par `python -m ciblage.causal.observational`.",
        f"Action étudiée : **{name}** sur le résultat **{OUTCOME}**. Covariable de confusion "
        f"injectée : **{COVARIATE}** (intensité {STRENGTH}).",
        "## 1. Vérité terrain (données randomisées complètes)",
        f"Sur les {_n(len(full))} clients randomisés (témoin + {name}), l'effet mesuré est "
        f"**{_pts(truth.ate)} pts** {_ic(truth)}. C'est la cible à retrouver.",
        "## 2. Le biais de confusion",
        f"Après avoir cassé la randomisation, l'échantillon observationnel compte "
        f"{_n(len(conf))} clients ({_n(ov.n_treated)} traités, {_n(ov.n_control)} témoins). "
        f"La **différence de moyennes naïve** y donne **{_pts(est_naive.ate)} pts** "
        f"{_ic(est_naive)} : soit un biais de **{_pts(biais_naif)} pts** par rapport à la vérité. "
        "Le confondant gonfle l'effet apparent.",
        "## 3. Recouvrement et rééquilibrage par la propension",
        f"- Propension estimée dans [{ov.e_min:.3f} ; {ov.e_max:.3f}], "
        f"part de clients extrêmes (e<0,05 ou e>0,95) : {100 * ov.share_extreme:.1f} %.\n"
        f"- Effectif efficace conservé : {100 * ov.ess_ratio_treated:.0f} % des traités, "
        f"{100 * ov.ess_ratio_control:.0f} % des témoins "
        f"(recouvrement **{'correct' if ov.is_good else 'limité'}**).\n"
        f"- **|SMD| maximale** entre groupes : **{smd_brute:.3f}** sans pondération → "
        f"**{smd_pondere:.3f}** après pondération par 1/propension "
        f"(seuil d'équilibre 0,1). La propension rééquilibre bien les groupes.",
        "## 4. Correction du biais",
        "« écart à la vérité » = effet estimé − vérité terrain ; plus il est proche de 0, mieux "
        "c'est. L'AIPW est doublement robuste : juste si le modèle de réponse OU la propension l'est.",
        _estimator_table(truth, [est_naive, est_ipw, est_aipw]),
        "## 5. Sensibilité au confounding caché (E-value)",
        "Même corrigé, l'effet pourrait venir d'un confondant **non mesuré**. La E-value est le "
        "risque relatif minimal qu'un tel confondant devrait avoir avec le traitement ET le "
        "résultat pour annuler l'effet.",
        f"- Risque relatif implicite de l'AIPW : **{ev.risk_ratio:.2f}**.\n"
        f"- **E-value ponctuelle : {ev.point:.2f}** — un confondant caché devrait être associé "
        f"d'un facteur ≥ {ev.point:.2f} au traitement et au résultat pour expliquer l'effet.\n"
        f"- **E-value de l'IC : {ev.ci:.2f}** "
        + (
            "(l'IC exclut 0 : un confondant plus faible que cela ne suffirait pas à rendre le "
            "résultat non significatif)."
            if ev.ci > 1.0
            else "(l'IC contient 0 : l'effet n'est pas démontré, rien à expliquer)."
        ),
    ]
    return "\n\n".join(parts) + "\n"


def main() -> None:
    config.REPORTS_DIR.mkdir(exist_ok=True)
    OUTPUT.write_text(build(), encoding="utf-8")
    print(f"Rapport écrit : {OUTPUT}")


if __name__ == "__main__":
    main()
