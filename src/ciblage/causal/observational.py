"""Phase 1 bis : estimation causale hors essai randomisé (données observationnelles)."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from ciblage import config
from ciblage.causal import confounding
from ciblage.causal.aipw import Estimate, aipw_ate, ipw_ate, naive_ate
from ciblage.causal.propensity import overlap, propensity_cv, smd_max
from ciblage.data import hillstrom
from ciblage.reporting import md_table

OUTPUT = config.REPORTS_DIR / "phase1b_observationnel.md"

ACTION = 1  # e-mail homme (effet moyen fort sur Hillstrom : biais bien visible)
COVARIATE = "history"  # montant déjà dépensé : corrélé à la visite -> bon confondant
STRENGTH = 1.5
STRENGTHS = [0.0, 0.5, 1.0, 1.5, 2.0]  # balayage de l'intensité du confounding
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


def _sample(conf: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Extrait (X encodé, w binaire, y) d'un sous-échantillon {témoin, ACTION}."""
    y = conf[OUTCOME].to_numpy()
    w = (conf["treatment"] == ACTION).to_numpy().astype(int)
    X = hillstrom.encode_features(conf)
    return X, w, y


def aipw_without(conf: pd.DataFrame, dropped: str) -> Estimate:
    """AIPW en RETIRANT une covariable de l'ajustement (propension ET réponse) : confondant non mesuré."""
    X, w, y = _sample(conf)
    X = X.drop(columns=[dropped])
    return aipw_ate(X, w, y, propensity_cv(X, w))


def strength_sweep(h: pd.DataFrame, truth: float, strengths: list[float]) -> pd.DataFrame:
    """Pour plusieurs intensités de confusion : biais naïf vs AIPW (écart à la vérité)."""
    rows = []
    for s in strengths:
        conf = confounding.make_confounded(h, action=ACTION, covariate=COVARIATE, strength=s)
        X, w, y = _sample(conf)
        naive = naive_ate(y, w)
        aipw = aipw_ate(X, w, y, propensity_cv(X, w))
        rows.append(
            {
                "intensité": f"{s:.1f}",
                "clients": _n(len(conf)),
                "naïf (pts)": _pts(naive.ate),
                "écart naïf": _pts(naive.ate - truth),
                "AIPW (pts)": _pts(aipw.ate),
                "écart AIPW": _pts(aipw.ate - truth),
            }
        )
    return pd.DataFrame(rows)


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
    est_unmeasured = aipw_without(conf, COVARIATE)  # AIPW sans le confondant (= non mesuré)
    sweep = strength_sweep(h, truth.ate, STRENGTHS)

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
        "## 5. Et si le confondant n'était pas mesuré ?",
        f"Démonstration directe plutôt qu'un indice théorique : on refait l'AIPW en **retirant "
        f"`{COVARIATE}` de l'ajustement** (propension et réponse), comme si ce confondant n'avait "
        "pas été observé.",
        f"- AIPW **avec** `{COVARIATE}` (ajustement complet) : **{_pts(est_aipw.ate)} pts** "
        f"{_ic(est_aipw)}, écart à la vérité {_pts(est_aipw.ate - truth.ate)} pts.\n"
        f"- AIPW **sans** `{COVARIATE}` (confondant non mesuré) : **{_pts(est_unmeasured.ate)} pts** "
        f"{_ic(est_unmeasured)}, écart {_pts(est_unmeasured.ate - truth.ate)} pts.\n"
        "Retirer le confondant fait **revenir le biais** : aucune méthode ne corrige un confondant "
        "qu'elle ne voit pas. La correction ne vaut donc que si les bons déterminants sont mesurés.",
        "## 6. Robustesse selon l'intensité du confounding",
        "On fait varier l'intensité du biais injecté (0 = randomisation préservée). Le biais naïf "
        "croît avec l'intensité, tandis que l'AIPW reste proche de la vérité tant que le confondant "
        "est mesuré.",
        md_table(sweep),
    ]
    return "\n\n".join(parts) + "\n"


def main() -> None:
    config.REPORTS_DIR.mkdir(exist_ok=True)
    OUTPUT.write_text(build(), encoding="utf-8")
    print(f"Rapport écrit : {OUTPUT}")


if __name__ == "__main__":
    main()
