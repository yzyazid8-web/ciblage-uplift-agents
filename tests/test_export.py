import pandas as pd
import pytest

from ciblage.serve import export
from ciblage.serve.planner import Planner, Scenario


@pytest.fixture(scope="module")
def planner():
    return Planner()


def test_preview_writes_nothing(planner, tmp_path):
    p = export.preview(planner, Scenario(budget=300.0))
    assert p["lignes"] == p["email_homme"] + p["email_femme"]
    assert p["depense_eur"] <= 300.0
    assert not list(tmp_path.iterdir())


def test_export_writes_the_planned_campaign(planner, tmp_path):
    sc = Scenario(budget=300.0)
    res = export.export_campaign(planner, sc, directory=tmp_path)
    df = pd.read_csv(res["fichier"])
    assert len(df) == res["lignes"] == export.preview(planner, sc)["lignes"]
    assert set(df["email"]) <= {"e-mail homme", "e-mail femme"}
    assert df["client"].is_unique
    assert not list(tmp_path.glob("*.part"))  # écriture atomique : pas de fichier partiel
