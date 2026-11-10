"""Rapport d'évaluation de l'agent (généré à partir du cache, aucun appel au LLM).

Chaque taux est donné avec le nombre de cas évalués et un intervalle de confiance de Wilson à
95 % : sur quelques dizaines de cas, « 90 % » peut vouloir dire « entre 75 et 97 % ».
"""

from __future__ import annotations

import math
from datetime import date
from pathlib import Path
from statistics import median
from typing import Any

import pandas as pd

from ciblage import config
from ciblage.eval.run import EVAL_DIR, _load, _version
from ciblage.reporting import md_table

OUTPUT = config.REPORTS_DIR / "phase5_eval_agent.md"


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Intervalle de confiance de Wilson pour une proportion (robuste aux petits effectifs)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _rate(values: list[bool]) -> dict[str, Any]:
    n, k = len(values), sum(values)
    lo, hi = wilson(k, n)
    return {
        "réussis / évalués": f"{k} / {n}",
        "taux": f"{100 * k / n:.0f} %" if n else "-",
        "IC 95 %": f"[{100 * lo:.0f} ; {100 * hi:.0f}] %" if n else "-",
    }


def _commit() -> str:
    """Identifiant du commit courant, lu dans .git (sans lancer git)."""
    git = config.ROOT / ".git"
    try:
        head = (git / "HEAD").read_text(encoding="utf-8").strip()
        if head.startswith("ref: "):
            ref = head[5:]
            path = git / ref
            if path.exists():
                return f"{ref.split('/')[-1]} @ {path.read_text(encoding='utf-8').strip()[:8]}"
            for line in (git / "packed-refs").read_text(encoding="utf-8").splitlines():
                if line.endswith(ref):
                    return f"{ref.split('/')[-1]} @ {line[:8]}"
        return head[:8]
    except OSError:
        return "inconnu"


def level1_metrics(ok: list[dict[str, Any]]) -> dict[str, list[bool]]:
    return {
        "Intention reconnue (LLM seul)": [r["intent_brut_ok"] for r in ok],
        "Intention reconnue (après règles déterministes)": [r["intent_corrige_ok"] for r in ok],
        "Paramètres extraits exactement": [r["parametres_ok"] for r in ok if r["a_des_parametres"]],
        "Aucun paramètre inventé (demandes à vérifier)": [
            not r["inventes"] for r in ok if r["categorie"] in ("plan", "plan_incomplet")
        ],
        "Demande ambiguë -> question posée": [
            r["question_posee"] for r in ok if r["ambigu_attendu"]
        ],
        "Demande claire -> pas de fausse ambiguïté": [
            not r["ambiguite_signalee"] for r in ok if r["categorie"] in ("plan", "plan_profit")
        ],
        "Hors sujet et injections refusés": [
            r["intent_corrige_ok"] for r in ok if r["categorie"] in ("hors_sujet", "injection")
        ],
    }


def _as_failures(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Passage v1 : ses « erreurs techniques » de validation de schéma étaient des échecs du
    modèle ; on les compte comme tels pour une comparaison équitable avec la v2."""
    out = []
    for r in rows:
        if "erreur" in r and "validation" in r["erreur"].lower():
            out.append(
                {
                    **r,
                    "categorie": r.get("categorie", "?"),
                    "intent_brut_ok": False,
                    "intent_corrige_ok": False,
                    "parametres_ok": False,
                    "a_des_parametres": True,
                    "inventes": [],
                    "ambiguite_signalee": False,
                    "question_posee": False,
                    "ambigu_attendu": False,
                }
            )
        elif "erreur" not in r:
            out.append(r)
    return out


def comparison_section(v1: list[dict[str, Any]], v2: list[dict[str, Any]]) -> str:
    """Avant / après les corrections issues de la première évaluation."""
    m1, m2 = level1_metrics(v1), level1_metrics(v2)
    rows = []
    for name in m1:
        a, b = m1[name], m2.get(name, [])
        rows.append(
            {
                "métrique": name,
                "v1 (avant corrections)": f"{sum(a)} / {len(a)}" if a else "-",
                "v2 (après corrections)": f"{sum(b)} / {len(b)}" if b else "-",
                "v2 : taux et IC 95 %": f"{_rate(b)['taux']} {_rate(b)['IC 95 %']}" if b else "-",
            }
        )
    return (
        "## Avant / après : ce que la première évaluation a permis de corriger\n\n"
        "Le premier passage (v1) a révélé : 10 % de réponses rejetées par la validation de schéma "
        "(un booléen mis à null, ce qui faisait planter le tour), des demandes « qui dois-je "
        "contacter ? » mal classées, et des coûts sans unité (« à 12 ») acceptés sans question. "
        "Corrections : schéma tolérant, échec du LLM géré sans plantage, contrôle de vraisemblance "
        "déterministe des coûts, consigne précisée. Dans la v1, les rejets de schéma sont comptés "
        "comme des échecs.\n\n" + md_table(pd.DataFrame(rows))
    )


def level1_section(rows: list[dict[str, Any]]) -> str:
    ok = [r for r in rows if "erreur" not in r]
    errors = len(rows) - len(ok)
    if not ok:
        return "## Niveau 1 : compréhension\n\nAucun cas évalué pour l'instant."
    model_failures = sum(1 for r in ok if r.get("echec_modele"))
    table = pd.DataFrame([{"métrique": k, **_rate(v)} for k, v in level1_metrics(ok).items()])
    by_cat = (
        pd.DataFrame(ok)
        .groupby("categorie")
        .agg(
            cas=("id", "count"),
            intention=("intent_corrige_ok", "mean"),
            parametres=("parametres_ok", "mean"),
        )
        .reset_index()
    )
    by_cat["intention"] = (100 * by_cat["intention"]).round().astype(int).astype(str) + " %"
    by_cat["parametres"] = (100 * by_cat["parametres"]).round().astype(int).astype(str) + " %"
    failures = [
        f"- `{r['id']}` ({r['categorie']}) « {r['message'][:90]} » → intention {r['comprehension']['intent']}"
        + (f", écarts {r['erreurs']}" if r["erreurs"] else "")
        + (f", inventés {r['inventes']}" if r["inventes"] else "")
        + (", ambiguïté NON signalée" if r["ambigu_attendu"] and not r["question_posee"] else "")
        for r in ok
        if not r["intent_corrige_ok"]
        or (r["a_des_parametres"] and not r["parametres_ok"])
        or r["inventes"]
        or (r["ambigu_attendu"] and not r["question_posee"])
    ]
    latency = [r["secondes"] for r in ok]
    return "\n\n".join(
        [
            f"## Niveau 1 : compréhension ({len(ok)} cas évalués"
            + (
                f", dont {model_failures} réponses invalides du modèle comptées en échec"
                if model_failures
                else ""
            )
            + (f", {errors} erreurs d'infrastructure exclues" if errors else "")
            + ")",
            md_table(table),
            "### Par catégorie (intention après règles, paramètres exacts)",
            md_table(by_cat),
            f"Latence de la compréhension : médiane {median(latency):.1f} s, maximum {max(latency):.1f} s.",
            "### Analyse des échecs",
            "\n".join(failures) or "Aucun échec.",
        ]
    )


def level2_section(rows: list[dict[str, Any]]) -> str:
    ok = [r for r in rows if "erreur" not in r]
    if not ok:
        return (
            "## Niveau 2 : conversations complètes\n\nAucune conversation évaluée pour l'instant."
        )
    checks: dict[str, list[bool]] = {}
    for r in ok:
        for k, v in r["verifications"].items():
            checks.setdefault(k, []).append(v)
    turns = [t for r in ok for t in r["tours"]]
    written = [t for t in turns if any(s.startswith("rédaction") for s in t["trace"])]
    first_try = [any("chiffres vérifiés (essai 1)" in s for s in t["trace"]) for t in written]
    fallback = [any("repli sûr" in s for s in t["trace"]) for t in written]
    labels = {
        "type_final": "Bon type de réponse finale (réponse / question)",
        "premiere_reponse": "Question posée quand il le faut (1er tour)",
        "scenario": "Scénario final correct",
        "type_de_reponse": "Bonne analyse effectuée (plan, conseil, explication...)",
        "export": "Export seulement après « oui »",
        "refus": "Hors sujet refusé",
        "pas_de_fuite": "Aucune fuite des consignes (injections)",
        "chiffre_attendu": "Chiffre attendu présent dans la réponse",
        "appels_llm": "Message trop long : aucun appel au LLM",
    }
    rows_t = [{"vérification": labels.get(k, k), **_rate(v)} for k, v in checks.items()]
    rows_t.append(
        {"vérification": "Conversation entièrement réussie", **_rate([r["succes"] for r in ok])}
    )
    rows_t.append({"vérification": "Chiffres vérifiés dès la 1re rédaction", **_rate(first_try)})
    rows_t.append(
        {"vérification": "Repli sur le texte standard (moins = mieux)", **_rate(fallback)}
    )
    real = [t for t in turns if t["appels_llm"] > 0]
    cost = (
        (
            f"Par tour (hors tours sans LLM) : {median([t['appels_llm'] for t in real]):.0f} appels au LLM "
            f"en médiane, {median([t['tokens'] for t in real]):,.0f} tokens en médiane, latence médiane "
            f"{median([t['secondes'] for t in real]):.1f} s (maximum {max(t['secondes'] for t in real):.1f} s)."
        ).replace(",", " ")
        if real
        else ""
    )
    failures = [
        f"- `{r['id']}` ({r['categorie']}) : échec de {[k for k, v in r['verifications'].items() if not v]} ; "
        f"dernière réponse : « {r['tours'][-1]['reponse'][:160]} »"
        for r in ok
        if not r["succes"]
    ]
    return "\n\n".join(
        [
            f"## Niveau 2 : conversations complètes ({len(ok)} conversations, {len(turns)} tours)",
            md_table(pd.DataFrame(rows_t)),
            cost,
            "### Analyse des échecs",
            "\n".join(failures) or "Aucun échec.",
        ]
    )


def write_report() -> Path:
    version = _version()
    l1 = [r for r in _load(EVAL_DIR / "niveau1.jsonl").values() if r.get("version") == version]
    l2 = [r for r in _load(EVAL_DIR / "niveau2.jsonl").values() if r.get("version") == version]
    v1_path = EVAL_DIR / "niveau1_v1.jsonl"
    v1 = _as_failures(list(_load(v1_path).values())) if v1_path.exists() else []
    models = sorted({r.get("modele", "?") for r in l1 + l2})
    parts = [
        "# Phase 5 : évaluation automatique de l'agent",
        f"Généré le {date.today().isoformat()} par `python -m ciblage.eval.run --rapport` ; "
        f"code : {_commit()} ; modèle évalué : {', '.join(models) or '-'} (modèle de secours désactivé).",
        "**Méthode** : les cas sont générés à partir de scénarios connus (la bonne réponse est connue "
        "d'avance, aucune annotation manuelle), puis formulés en français avec des gabarits variés. "
        "Niveau 1 = compréhension seule (1 appel au LLM par cas) ; niveau 2 = conversations complètes "
        "(système entier). Taux donnés avec le nombre de cas évalués et un IC de Wilson à 95 %.",
        level1_section(l1),
        comparison_section(v1, [r for r in l1 if "erreur" not in r]) if v1 and l1 else "",
        level2_section(l2),
        "## Limites",
        "- Gabarits rédigés à la main : ils ne couvrent pas toute la variété des formulations réelles.\n"
        "- Un seul passage par cas : la variabilité du LLM d'un appel à l'autre n'est pas mesurée.\n"
        "- Échantillons modestes (quota gratuit) : les intervalles de confiance restent larges.\n"
        "- La qualité rédactionnelle n'est pas notée ; seuls la fidélité des chiffres et le comportement "
        "le sont.",
    ]
    config.REPORTS_DIR.mkdir(exist_ok=True)
    OUTPUT.write_text("\n\n".join(parts) + "\n", encoding="utf-8")
    return OUTPUT
