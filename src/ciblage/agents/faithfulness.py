"""Contrôle de fidélité : le LLM n'a-t-il cité que des chiffres calculés, et un verdict juste ?."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

_NUMBER = re.compile(
    r"(?<![\d.,])[+\-−]?\d{1,3}(?:[ \u00a0\u202f]\d{3})+(?:[,.]\d+)?|(?<![\d.,])[+\-−]?\d+(?:[,.]\d+)?"
)
_SIGNS = "+-−"
# Petits entiers d'usage courant (« les 2 e-mails », « 3 segments ») : jamais considérés comme inventés
TRIVIAL = {0.0, 1.0, 2.0, 3.0, 95.0, 100.0}  # 95 : « IC 95 % »
# Champs déjà exprimés dans leur unité d'affichage : jamais multipliés par 100 (sinon une perte de
# 0,23 point pouvait être citée « 23 % », constat I3)
_SCALED = re.compile(
    r"(^|_)(pct|points|pts|eur|ic95|valeur)(_|$)|euro|budget|clients|visites|cout|depense|seuil"
    r"|rendement|ecart|prix"
)

# verdict : une affirmation de supériorité, visant le plan, comparée à une autre politique
CLAIM = re.compile(
    r"\b(prouv\w*|démontr\w*|significati\w*|bat|battent|surperform\w*|sur-perform\w*"
    r"|nettement (meilleur|supérieur)\w*|fait mieux|meilleur que)",
    re.IGNORECASE,
)
ABOUT_PLAN = re.compile(r"\b(plan|optimis\w*|copilote)\b", re.IGNORECASE)
COMPARED = re.compile(
    r"classique|\bbat\b|battent|surperform|sur-perform|mieux|meilleur|supérieur|\bgain",
    re.IGNORECASE,
)
NEGATION = re.compile(r"\b(pas|non|aucun\w*|ni|jamais|sans|ne)\b|\bn['’]", re.IGNORECASE)


@dataclass
class FaithfulnessReport:
    numbers: list[float] = field(default_factory=list)  # nombres trouvés dans le texte
    unsupported: list[str] = field(default_factory=list)  # nombres absents du dossier de faits
    verdict: list[str] = field(default_factory=list)  # phrases au verdict contraire aux faits

    @property
    def ok(self) -> bool:
        return not self.unsupported and not self.verdict


def parse_numbers(text: str) -> list[tuple[str, float, int]]:
    """(texte brut, valeur, nombre de décimales) pour chaque nombre du texte.

    Un nombre illisible reçoit la valeur NaN : il sera compté comme non vérifié, jamais ignoré.
    """
    out = []
    for m in _NUMBER.finditer(text):
        raw = m.group(0)
        clean = re.sub(r"[ \u00a0\u202f]", "", raw).replace("−", "-").replace(",", ".")
        try:
            value = float(clean)
        except ValueError:
            value = float("nan")
        decimals = len(clean.split(".")[1]) if "." in clean else 0
        out.append((raw.strip(), value, decimals))
    return out


def flatten_items(facts: object, key: str = "") -> list[tuple[str, float]]:
    """(nom du champ, valeur) pour chaque valeur numérique d'un dossier de faits."""
    items: list[tuple[str, float]] = []
    if isinstance(facts, bool):
        return items
    if isinstance(facts, int | float):
        if math.isfinite(facts):
            items.append((key, float(facts)))
    elif isinstance(facts, Mapping):
        for k, v in facts.items():
            items += flatten_items(v, str(k))
    elif isinstance(facts, Iterable) and not isinstance(facts, str | bytes):
        for v in facts:
            items += flatten_items(v, key)
    return items


def flatten(facts: object) -> list[float]:
    """Toutes les valeurs numériques d'un dossier de faits (dictionnaires et listes imbriqués)."""
    return [v for _, v in flatten_items(facts)]


def numbers_in_text(facts: object) -> list[float]:
    """Nombres écrits DANS les textes d'un dossier."""
    if isinstance(facts, str):
        return [value for _, value, _ in parse_numbers(facts) if math.isfinite(value)]
    if isinstance(facts, Mapping):
        return [x for v in facts.values() for x in numbers_in_text(v)]
    if isinstance(facts, Iterable) and not isinstance(facts, bytes):
        return [x for v in facts for x in numbers_in_text(v)]
    return []


def _candidates(items: Iterable[tuple[str, float]]) -> list[float]:
    """Valeurs citables : la valeur, et ×100 seulement pour une proportion (taux, part, uplift)."""
    out = []
    for key, v in items:
        out.append(v)
        if abs(v) <= 1 and not _SCALED.search(key.lower()):
            out.append(100 * v)
    return out


def _matches(x: float, decimals: int, signed: bool, allowed: Iterable[float]) -> bool:
    tol = 0.5 * 10 ** (-decimals) + 1e-9  # un arrondi correct à `decimals` décimales
    for v in allowed:
        # signe écrit : il doit être le bon ; signe omis (« une baisse de 0,24 ») : toléré
        target = v if signed else abs(v)
        if abs((x if signed else abs(x)) - target) <= max(tol, 0.005 * abs(v)):
            return True
    return False


def plan_gain_shown(facts: object) -> bool | None:
    """Le gain du plan face au ciblage classique est-il démontré d'après le dossier ?

    None si le dossier n'en dit rien ; False dès qu'une mesure du plan dit « non démontré ».
    """
    verdicts: list[bool] = []

    def walk(node: object) -> None:
        if isinstance(node, Mapping):
            if "gain_demontre" in node and isinstance(node["gain_demontre"], bool):
                verdicts.append(node["gain_demontre"])
            if node.get("politique") == "plan optimisé" and isinstance(
                node.get("difference_demontree"), bool
            ):
                low = node.get("gain_ic95_bas")
                verdicts.append(bool(node["difference_demontree"]) and (low or 0) > 0)
            for v in node.values():
                walk(v)
        elif isinstance(node, list | tuple):
            for v in node:
                walk(v)

    walk(facts)
    return None if not verdicts else all(verdicts)


def verdict_issues(text: str, facts: object) -> list[str]:
    """Phrases qui affirment un gain du plan alors que le dossier dit qu'il n'est pas démontré."""
    if plan_gain_shown(facts) is not False:
        return []
    issues = []
    for sentence in re.split(r"(?<=[.!?;])\s+|\n+", text):
        claim = CLAIM.search(sentence) and ABOUT_PLAN.search(sentence) and COMPARED.search(sentence)
        if claim and not NEGATION.search(sentence):
            issues.append(sentence.strip())
    return issues


def check(text: str, facts: object, extra_allowed: Iterable[float] = ()) -> FaithfulnessReport:
    """Vérifie que chaque nombre du texte provient du dossier de faits, et que le verdict est juste."""
    allowed = _candidates(flatten_items(facts)) + list(extra_allowed)
    report = FaithfulnessReport()
    for raw, value, decimals in parse_numbers(text):
        report.numbers.append(value)
        if not math.isfinite(value):
            report.unsupported.append(raw)
            continue
        if abs(value) in TRIVIAL:
            continue
        signed = raw[:1] in _SIGNS
        if not _matches(value, decimals, signed, allowed):
            report.unsupported.append(raw)
    report.verdict = verdict_issues(text, facts)
    return report
