import numpy as np
import pandas as pd
import pytest

from ciblage.data import criteo


def _raw(n: int, treated_first: bool = True, seed: int = 0) -> pd.DataFrame:
    """Faux fichier Criteo tiré par groupe (comme l'original) : traités d'abord, témoins ensuite."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(n, 12)), columns=criteo.FEATURES)
    n_treated = int(0.85 * n)
    df["treatment"] = [1] * n_treated + [0] * (n - n_treated) if treated_first else 1
    df["visit"] = (rng.random(n) < 0.05).astype(int)
    df["conversion"] = df["visit"] * (rng.random(n) < 0.1)
    df["exposure"] = df["treatment"] * (rng.random(n) < 0.3)
    return df


def test_sample_reads_whole_sorted_file(tmp_path):
    """Sur un fichier trié, l'échantillon doit contenir des témoins (lus en fin de fichier)."""
    path = tmp_path / "criteo.csv"
    _raw(20_000).to_csv(path, index=False)
    df = criteo.sample(frac=0.2, path=path, chunk_size=3_000)  # plusieurs morceaux
    assert 0.15 * 20_000 < len(df) < 0.25 * 20_000
    assert df["treatment"].mean() == pytest.approx(0.85, abs=0.03)
    assert set(df.columns) == set(
        criteo.FEATURES + ["treatment", "visit", "conversion", "exposure"]
    )


def test_sample_is_reproducible(tmp_path):
    path = tmp_path / "criteo.csv"
    _raw(5_000).to_csv(path, index=False)
    a = criteo.sample(frac=0.3, seed=1, path=path, chunk_size=1_000)
    b = criteo.sample(frac=0.3, seed=1, path=path, chunk_size=1_000)
    pd.testing.assert_frame_equal(a, b)


def test_sample_rejects_invalid_frac(tmp_path):
    with pytest.raises(ValueError, match="frac"):
        criteo.sample(frac=0, path=tmp_path / "absent.csv")


@pytest.mark.parametrize(
    "column, values, message",
    [
        ("visit", [0, 2, 0], "binaire"),
        ("conversion", [0, 1, 0], "Conversion sans visite"),
        ("exposure", [0, 0, 1], "témoin"),
    ],
)
def test_validate_rejects_inconsistent_chunk(column, values, message):
    chunk = _raw(3, treated_first=False)
    chunk["visit"] = 0
    chunk["conversion"] = 0
    chunk["exposure"] = 0
    chunk["treatment"] = [1, 1, 0]
    chunk[column] = values
    with pytest.raises(ValueError, match=message):
        criteo.validate(chunk)
