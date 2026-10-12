import pandas as pd
import pytest
from src.data import load_and_validate, chronological_split

def test_validation_and_chronological_split(tmp_path):
    df = pd.DataFrame({
        "Time": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "Amount": [10.0] * 10,
        "V1": [0.1] * 10,
        "Class": [0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    })
    p = tmp_path / "toy.csv"
    df.to_csv(p, index=False)
    loaded, report = load_and_validate(p)
    assert report.rows == 10
    assert report.fraud_count == 2
    splits = chronological_split(loaded, 0.6, 0.2)
    assert len(splits["train"]) == 6
    assert len(splits["validation"]) == 2
    assert len(splits["test"]) == 2
    assert splits["train"]["Time"].max() < splits["validation"]["Time"].min()
    assert splits["validation"]["Time"].max() < splits["test"]["Time"].min()

def test_rejects_non_chronological_data():
    df = pd.DataFrame({"Time": [2, 1, 3, 4, 5], "Amount": [1]*5, "Class": [0,1,0,0,0]})
    with pytest.raises(ValueError, match="not chronological"):
        chronological_split(df)

def test_missing_required_column(tmp_path):
    df = pd.DataFrame({"Time": [1,2], "Class": [0,1]})
    p = tmp_path / "bad.csv"
    df.to_csv(p, index=False)
    with pytest.raises(ValueError, match="Missing required"):
        load_and_validate(p)
