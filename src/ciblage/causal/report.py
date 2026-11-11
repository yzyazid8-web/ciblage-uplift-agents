"""Rapport de la phase 1 : équilibre des groupes et effets moyens, sur Hillstrom et Criteo.

Usage : python -m ciblage.causal.report   ->   reports/phase1_effets_moyens.md
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from ciblage import config
from ciblage.causal import ate, balance
from ciblage.data import criteo, hillstrom
from ciblage.reporting import md_table

OUTPUT = config.REPORTS_DIR / "phase1_effets_moyens.md"


def _n(v: int) -> str:
    return f"{v:,}".replace(",", " ")  # espace fine insécable (typographie française)


def _pct(x: float) -> str:
    return f"{100 * x:+.2f} pts" if abs(x) < 1 else f"{x:+.3f}"


def _effects_section(effects: pd.DataFrame, names: dict[int, str], money: set[str]) -> str:
    rows = []
    for e in effects.itertuples():
        fmt = (lambda v: f"{v:+.3f} $") if e.outcome in money else _pct
        rows.append(
            {
                "action": names[e.treatment],
                "résultat": e.outcome,
                "témoin": f"{e.mean_control:.4f}",
                "traité": f"{e.mean_treated:.4f}",
                "effet moyen": fmt(e.ate),
                "effet relatif": f"{100 * e.lift:+.0f} %",
                "IC 95 % (Welch)": f"[{fmt(e.ci_low)} ; {fmt(e.ci_high)}]",
                "IC 95 % (bootstrap)": f"[{fmt(e.boot_low)} ; {fmt(e.boot_high)}]",
                "p ajustée (Holm)": f"{e.p_holm:.1e}",
            }
        )
    return md_table(pd.DataFrame(rows))


def _balance_section(rep: balance.BalanceReport, names: dict[int, str], n_rows: int) -> str:
    smd = rep.smd.copy()
    smd.columns = [f"SMD {names[int(c.split('_')[1])]}" for c in smd.columns]
    table = smd.map(lambda v: f"{v:+.3f}")
    table["test"] = rep.tests["test"]
    table["p-valeur"] = rep.tests["p_value"].map(lambda v: f"{v:.2f}")
    table.insert(0, "variable", table.index)
    verdict = "✅ équilibré" if rep.is_balanced else "❌ déséquilibre détecté"
    if rep.sample_ratio is None:
        ratio = "- Effectifs : protocole d'affectation non publié, test non applicable\n"
    else:
        ratio = (
            f"- Effectifs (test du chi-2 vs protocole) : p = {rep.sample_ratio['p_value']:.2f}\n"
        )
    sub = (
        f" (sur un sous-échantillon aléatoire de {_n(balance.TEST_MAX_N)} lignes)"
        if n_rows > balance.TEST_MAX_N
        else ""
    )
    return (
        ratio + f"- Tests par variable{sub} : indicatifs ; le critère retenu est la SMD\n"
        f"- |SMD| maximale : {rep.max_abs_smd:.3f} (seuil {balance.SMD_THRESHOLD}) → **{verdict}**\n\n"
        + md_table(table)
    )


def build() -> str:
    parts = [
        "# Phase 1 : équilibre des groupes et effets moyens",
        f"Généré le {date.today().isoformat()} par `python -m ciblage.causal.report`.",
    ]

    # Hillstrom
    h = hillstrom.load()
    names = hillstrom.TREATMENT_NAMES
    X = hillstrom.encode_features(h)
    rep = balance.check_balance(X, h["treatment"].to_numpy(), {0: 1 / 3, 1: 1 / 3, 2: 1 / 3})
    eff = ate.estimate_effects(h, outcomes=hillstrom.OUTCOMES)
    counts = h["treatment"].value_counts().sort_index()
    parts += [
        "## Hillstrom (64 000 clients, essai randomisé 1/3 - 1/3 - 1/3)",
        "Effectifs : " + ", ".join(f"{names[k]} = {_n(v)}" for k, v in counts.items()),
        "### Équilibre des groupes avant la campagne",
        _balance_section(rep, names, len(h)),
        "### Effet moyen de chaque e-mail",
        _effects_section(eff, names, money={"spend"}),
    ]

    # Criteo (échantillon)
    c = criteo.load_sample()
    share = float(c["treatment"].mean())
    rep_c = balance.check_balance(c[criteo.FEATURES], c["treatment"].to_numpy(), expected=None)
    eff_c = ate.estimate_effects(c, outcomes=criteo.OUTCOMES, n_boot=500)
    names_c = {0: "témoin", 1: "publicité"}
    parts += [
        f"## Criteo (échantillon aléatoire de {_n(len(c))} lignes, {100 * share:.1f} % traités)",
        "### Équilibre des groupes",
        _balance_section(rep_c, names_c, len(c)),
        "### Effet moyen de la publicité",
        _effects_section(eff_c, names_c, money=set()),
    ]
    return "\n\n".join(parts) + "\n"


def main() -> None:
    config.REPORTS_DIR.mkdir(exist_ok=True)
    OUTPUT.write_text(build(), encoding="utf-8")
    print(f"Rapport écrit : {OUTPUT}")


if __name__ == "__main__":
    main()
