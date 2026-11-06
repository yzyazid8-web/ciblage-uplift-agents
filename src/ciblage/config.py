"""Configuration centrale : chemins du projet et graine aléatoire."""

from pathlib import Path

# Racine du dépôt (src/ciblage/config.py -> remonte de 2 niveaux)
ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"  # données brutes, non versionnées
REPORTS_DIR = ROOT / "reports"  # rapports générés par script, versionnés

SEED = 42
