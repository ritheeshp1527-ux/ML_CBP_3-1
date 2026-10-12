from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

EXPECTED_TARGET = "Class"
EXPECTED_TIME = "Time"

@dataclass
class DataReport:
    rows: int
    columns: int
    fraud_count: int
    legitimate_count: int
    fraud_rate: float
    missing_cells: int
    infinite_cells: int
    duplicate_rows_keep_first: int
    rows_in_duplicate_groups: int
    time_is_nondecreasing: bool
    target_values: list

def load_and_validate(path: str | Path) -> tuple[pd.DataFrame, DataReport]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    required = {EXPECTED_TARGET, EXPECTED_TIME, "Amount"}
    missing_cols = sorted(required - set(df.columns))
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    if df.empty:
        raise ValueError("Dataset is empty.")
    if not set(pd.unique(df[EXPECTED_TARGET])).issubset({0, 1}):
        raise ValueError("Class must contain only binary labels 0 and 1.")
    numeric = df.select_dtypes(include=[np.number])
    infinite_cells = int(np.isinf(numeric.to_numpy()).sum())
    duplicate_mask = df.duplicated(keep="first")
    duplicate_group_mask = df.duplicated(keep=False)
    counts = df[EXPECTED_TARGET].value_counts()
    report = DataReport(
        rows=int(len(df)),
        columns=int(df.shape[1]),
        fraud_count=int(counts.get(1, 0)),
        legitimate_count=int(counts.get(0, 0)),
        fraud_rate=float(counts.get(1, 0) / len(df)),
        missing_cells=int(df.isna().sum().sum()),
        infinite_cells=infinite_cells,
        duplicate_rows_keep_first=int(duplicate_mask.sum()),
        rows_in_duplicate_groups=int(duplicate_group_mask.sum()),
        time_is_nondecreasing=bool(df[EXPECTED_TIME].is_monotonic_increasing),
        target_values=sorted(int(v) for v in df[EXPECTED_TARGET].dropna().unique()),
    )
    if report.missing_cells:
        raise ValueError(f"Dataset has {report.missing_cells} missing cells; inspect before training.")
    if report.infinite_cells:
        raise ValueError(f"Dataset has {report.infinite_cells} infinite numeric values; inspect before training.")
    return df, report

def chronological_split(df: pd.DataFrame, train_fraction=0.60, validation_fraction=0.20):
    """Split by row order after verifying Time is nondecreasing; does not shuffle or drop rows."""
    if not (0 < train_fraction < 1 and 0 < validation_fraction < 1 and
            train_fraction + validation_fraction < 1):
        raise ValueError("Fractions must be positive and train + validation must be < 1.")
    if not df["Time"].is_monotonic_increasing:
        raise ValueError("Rows are not chronological. Sort explicitly by Time and document tie handling.")
    n = len(df)
    i_train = int(n * train_fraction)
    i_val = int(n * (train_fraction + validation_fraction))
    splits = {"train": df.iloc[:i_train].copy(),
              "validation": df.iloc[i_train:i_val].copy(),
              "test": df.iloc[i_val:].copy()}
    if any(part.empty for part in splits.values()):
        raise ValueError("A split is empty.")
    return splits

def split_summary(splits: dict[str, pd.DataFrame]) -> dict:
    result = {}
    for name, part in splits.items():
        counts = part["Class"].value_counts()
        result[name] = {
            "rows": int(len(part)),
            "fraud": int(counts.get(1, 0)),
            "legitimate": int(counts.get(0, 0)),
            "fraud_rate": float(counts.get(1, 0) / len(part)),
            "time_min": float(part["Time"].min()),
            "time_max": float(part["Time"].max()),
        }
    return result
