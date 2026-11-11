"""Jeu Hillstrom (MineThatData, 2008) : vraie campagne e-mail randomisée sur 64 000 clients."""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from ciblage import config

URL = (
    "http://www.minethatdata.com/"
    "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
)
RAW_PATH = config.DATA_DIR / "raw" / "hillstrom.csv"

# Codage du traitement : 0 = témoin, puis une valeur par action possible.
TREATMENTS: dict[str, int] = {"No E-Mail": 0, "Mens E-Mail": 1, "Womens E-Mail": 2}
TREATMENT_NAMES: dict[int, str] = {0: "temoin", 1: "email_homme", 2: "email_femme"}

OUTCOMES = ["visit", "conversion", "spend"]

# Variables connues avant la campagne.
NUMERIC_FEATURES = ["recency", "history", "mens", "womens", "newbie"]
CATEGORIES: dict[str, list[str]] = {
    # Orthographe d'origine du fichier conservée.
    "zip_code": ["Urban", "Surburban", "Rural"],
    "channel": ["Phone", "Web", "Multichannel"],
}

_EXPECTED_COLUMNS = [
    "recency",
    "history_segment",
    "history",
    "mens",
    "womens",
    "zip_code",
    "newbie",
    "channel",
    "segment",
    *OUTCOMES,
]


def download(path: Path = RAW_PATH, force: bool = False) -> Path:
    """Télécharge le CSV d'origine s'il n'est pas déjà présent."""
    if path.exists() and not force:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    urllib.request.urlretrieve(URL, tmp)
    tmp.replace(path)  # écriture atomique : pas de fichier à moitié téléchargé
    return path


def validate(raw: pd.DataFrame) -> None:
    """Vérifie les propriétés attendues du fichier brut. Lève ValueError sinon."""
    missing = set(_EXPECTED_COLUMNS) - set(raw.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes : {sorted(missing)}")
    if raw[_EXPECTED_COLUMNS].isna().any().any():
        raise ValueError("Valeurs manquantes inattendues")

    unknown = set(raw["segment"]) - set(TREATMENTS)
    if unknown:
        raise ValueError(f"Groupes inconnus : {sorted(unknown)}")
    for col, cats in CATEGORIES.items():
        bad = set(raw[col]) - set(cats)
        if bad:
            raise ValueError(f"Modalités inconnues pour {col} : {sorted(bad)}")

    for col in ["mens", "womens", "newbie", "visit", "conversion"]:
        if not raw[col].isin([0, 1]).all():
            raise ValueError(f"{col} doit être binaire (0/1)")
    if (raw["spend"] < 0).any():
        raise ValueError("Dépense négative")

    # Cohérence du parcours client : acheter suppose d'avoir visité, et dépense > 0 <=> achat.
    if ((raw["conversion"] == 1) & (raw["visit"] == 0)).any():
        raise ValueError("Conversion sans visite")
    if ((raw["spend"] > 0) != (raw["conversion"] == 1)).any():
        raise ValueError("Incohérence entre dépense et conversion")


def load(path: Path = RAW_PATH) -> pd.DataFrame:
    """Charge et valide Hillstrom. Ajoute la colonne entière `treatment` (0, 1, 2).

    Retourne les variables explicatives brutes (catégories en texte), `treatment`
    et les trois résultats. Télécharge le fichier au besoin.
    """
    raw = pd.read_csv(download(path))
    validate(raw)
    df = raw[NUMERIC_FEATURES + list(CATEGORIES) + OUTCOMES].copy()
    df["treatment"] = raw["segment"].map(TREATMENTS).astype(np.int8)
    for col in ["mens", "womens", "newbie", "visit", "conversion"]:
        df[col] = df[col].astype(np.int8)
    for col, cats in CATEGORIES.items():
        df[col] = pd.Categorical(df[col], categories=cats)
    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Matrice numérique des variables explicatives (indicatrices à modalités FIXES)."""
    out = df[NUMERIC_FEATURES].astype(np.float64)
    for col, cats in CATEGORIES.items():
        values = df[col].astype(str)
        for cat in cats[1:]:
            out[f"{col}_{cat}"] = (values == cat).astype(np.float64)
    return out


def feature_names() -> list[str]:
    """Noms des colonnes produites par `encode_features`, dans l'ordre."""
    return NUMERIC_FEATURES + [f"{c}_{v}" for c, cats in CATEGORIES.items() for v in cats[1:]]
