"""Jeu Criteo Uplift : essai randomisé publicitaire."""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from ciblage import config

URL = (
    "https://huggingface.co/datasets/criteo/criteo-uplift/resolve/main/"
    "criteo-research-uplift-v2.1.csv.gz"
)
RAW_PATH = config.DATA_DIR / "raw" / "criteo-research-uplift-v2.1.csv.gz"
PROCESSED_DIR = config.DATA_DIR / "processed"

FEATURES = [f"f{i}" for i in range(12)]
OUTCOMES = ["visit", "conversion"]
_BINARY = ["treatment", "visit", "conversion", "exposure"]
_EXPECTED_COLUMNS = FEATURES + _BINARY

CHUNK_SIZE = 1_000_000


def download(path: Path = RAW_PATH, force: bool = False) -> Path:
    """Télécharge l'archive si elle n'est pas déjà présente."""
    if path.exists() and not force:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    urllib.request.urlretrieve(URL, tmp)
    tmp.replace(path)
    return path


def validate(chunk: pd.DataFrame) -> None:
    """Controles sur un morceau du fichier brut. Lève ValueError sinon."""
    missing = set(_EXPECTED_COLUMNS) - set(chunk.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes : {sorted(missing)}")
    if chunk[_EXPECTED_COLUMNS].isna().any().any():
        raise ValueError("Valeurs manquantes inattendues")
    for col in _BINARY:
        if not chunk[col].isin([0, 1]).all():
            raise ValueError(f"{col} doit être binaire (0/1)")
    if ((chunk["conversion"] == 1) & (chunk["visit"] == 0)).any():
        raise ValueError("Conversion sans visite")
    if ((chunk["treatment"] == 0) & (chunk["exposure"] == 1)).any():
        raise ValueError("Exposition à la publicité dans le groupe témoin")


def sample(
    frac: float = 0.15,
    seed: int = config.SEED,
    path: Path = RAW_PATH,
    chunk_size: int = CHUNK_SIZE,
) -> pd.DataFrame:
    """Échantillon aléatoire simple de taux `frac`, sur tout le fichier, lu par morceaux."""
    if not 0 < frac <= 1:
        raise ValueError("frac doit être dans ]0, 1]")
    rng = np.random.default_rng(seed)
    parts = []
    for chunk in pd.read_csv(download(path), chunksize=chunk_size, dtype=np.float32):
        validate(chunk)
        parts.append(chunk[rng.random(len(chunk)) < frac])
    df = pd.concat(parts, ignore_index=True)
    for col in _BINARY:
        df[col] = df[col].astype(np.int8)
    return df


def load_sample(frac: float = 0.15, seed: int = config.SEED) -> pd.DataFrame:
    """Échantillon mis en cache en Parquet."""
    cache = PROCESSED_DIR / f"criteo_sample_frac{frac:g}_seed{seed}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    df = sample(frac=frac, seed=seed)
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".part")
    df.to_parquet(tmp, index=False)
    tmp.replace(cache)
    return df
