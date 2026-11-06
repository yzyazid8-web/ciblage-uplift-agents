import pandas as pd
import pytest

from ciblage.data import hillstrom


def _raw(**overrides) -> pd.DataFrame:
    """Petit fichier brut valide (3 clients, un par groupe), modifiable pour les cas d'erreur."""
    data = {
        "recency": [1, 6, 12],
        "history_segment": ["1) $0 - $100", "3) $200 - $350", "2) $100 - $200"],
        "history": [42.0, 329.08, 142.44],
        "mens": [1, 1, 0],
        "womens": [0, 1, 1],
        "zip_code": ["Urban", "Rural", "Surburban"],
        "newbie": [1, 1, 0],
        "channel": ["Web", "Phone", "Multichannel"],
        "segment": ["No E-Mail", "Mens E-Mail", "Womens E-Mail"],
        "visit": [0, 1, 1],
        "conversion": [0, 1, 0],
        "spend": [0.0, 55.5, 0.0],
    }
    data.update(overrides)
    return pd.DataFrame(data)


@pytest.fixture
def csv_path(tmp_path):
    path = tmp_path / "hillstrom.csv"
    _raw().to_csv(path, index=False)
    return path


def test_validate_accepts_valid_file():
    hillstrom.validate(_raw())


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"segment": ["No E-Mail", "Mens E-Mail", "SMS"]}, "Groupes inconnus"),
        ({"zip_code": ["Urban", "Rural", "Paris"]}, "Modalités inconnues"),
        ({"visit": [0, 2, 1]}, "binaire"),
        ({"spend": [0.0, -1.0, 0.0]}, "négative"),
        ({"visit": [0, 0, 1]}, "Conversion sans visite"),
        ({"spend": [0.0, 55.5, 10.0]}, "dépense et conversion"),
    ],
)
def test_validate_rejects_inconsistent_data(overrides, message):
    with pytest.raises(ValueError, match=message):
        hillstrom.validate(_raw(**overrides))


def test_validate_rejects_missing_column():
    with pytest.raises(ValueError, match="Colonnes manquantes"):
        hillstrom.validate(_raw().drop(columns="spend"))


def test_load_maps_treatment_and_keeps_only_pre_campaign_features(csv_path):
    df = hillstrom.load(csv_path)
    assert df["treatment"].tolist() == [0, 1, 2]
    assert "history_segment" not in df.columns
    assert "segment" not in df.columns


def test_encode_features_is_stable_on_any_subset(csv_path):
    df = hillstrom.load(csv_path)
    full = hillstrom.encode_features(df)
    one = hillstrom.encode_features(df.iloc[[2]])  # un seul client : memes colonnes
    assert list(full.columns) == hillstrom.feature_names()
    assert list(one.columns) == hillstrom.feature_names()
    # Client 3 : Surburban, Multichannel ; références supprimées = Urban, Phone.
    row = one.iloc[0]
    assert row["zip_code_Surburban"] == 1 and row["zip_code_Rural"] == 0
    assert row["channel_Multichannel"] == 1 and row["channel_Web"] == 0
