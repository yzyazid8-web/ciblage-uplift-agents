"""Contrôle de fidélité numérique : le LLM n'a-t-il cité que des chiffres réellement calculés ?"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

# Nombres « français »
_NUMBER = re.compile(
    r"(?<![\w.,])[+-−]?\d{1,3}(?:[   ]\d{3})+(?:[,.]\d+)?|(?<![\w.,])[+-−]?\d+(?:[,.]\d+)?"
)
# Petits entiers d'usage courant (« les 2 e-mails », « 3 segments ») : jamais considérés comme inventés
TRIVIAL = {0.0, 1.0, 2.0, 3.0, 100.0}


@dataclass
class FaithfulnessReport:
    numbers: list[float] = field(default_factory=list)  # nombres trouvés dans le texte
    unsupported: list[str] = field(default_factory=list)  # nombres absents du dossier de faits

    @property
    def ok(self) -> bool:
        return not self.unsupported


def parse_numbers(text: str) -> list[tuple[str, float, int]]:
    """(texte brut, valeur, nombre de décimales) pour chaque nombre du texte."""
    out = []
    for m in _NUMBER.finditer(text):
        raw = m.group(0)
        clean = re.sub(r"[   ]", "", raw).replace("−", "-").replace(",", ".")
        try:
            value = float(clean)
        except ValueError:
            continue
        decimals = len(clean.split(".")[1]) if "." in clean else 0
        out.append((raw.strip(), value, decimals))
    return out


def flatten(facts: object) -> list[float]:
    """Toutes les valeurs numériques d'un dossier de faits (dictionnaires et listes imbriqués)."""
    values: list[float] = []
    if isinstance(facts, bool):
        return values
    if isinstance(facts, int | float):
        if math.isfinite(facts):
            values.append(float(facts))
    elif isinstance(facts, Mapping):
        for v in facts.values():
            values += flatten(v)
    elif isinstance(facts, Iterable) and not isinstance(facts, str | bytes):
        for v in facts:
            values += flatten(v)
    return values


def numbers_in_text(facts: object) -> list[float]:
    """Nombres écrits DANS les textes d'un dossier."""
    if isinstance(facts, str):
        return [value for _, value, _ in parse_numbers(facts)]
    if isinstance(facts, Mapping):
        return [x for v in facts.values() for x in numbers_in_text(v)]
    if isinstance(facts, Iterable) and not isinstance(facts, bytes):
        return [x for v in facts for x in numbers_in_text(v)]
    return []


def _matches(x: float, decimals: int, allowed: Iterable[float]) -> bool:
    tol = 0.5 * 10 ** (-decimals) + 1e-9  # un arrondi correct à `decimals` décimales
    for v in allowed:
        for candidate in (v, 100 * v, abs(v), 100 * abs(v)):  # proportion citée en %, signe omis
            if abs(abs(x) - abs(candidate)) <= max(tol, 0.005 * abs(candidate)):
                return True
    return False


def check(text: str, facts: object, extra_allowed: Iterable[float] = ()) -> FaithfulnessReport:
    """Vérifie que chaque nombre du texte provient du dossier de faits."""
    allowed = flatten(facts) + list(extra_allowed)
    report = FaithfulnessReport()
    for raw, value, decimals in parse_numbers(text):
        report.numbers.append(value)
        if abs(value) in TRIVIAL:
            continue
        if not _matches(value, decimals, allowed):
            report.unsupported.append(raw)
    return report
