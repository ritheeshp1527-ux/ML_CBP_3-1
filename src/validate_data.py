from __future__ import annotations
import argparse, json
from dataclasses import asdict
from pathlib import Path
from .data import load_and_validate, chronological_split, split_summary

def main():
    parser = argparse.ArgumentParser(description="Validate credit-card fraud CSV.")
    parser.add_argument("--data", default="data/raw/creditcard.csv")
    parser.add_argument("--out", default="artifacts/metrics")
    args = parser.parse_args()
    df, report = load_and_validate(args.data)
    splits = chronological_split(df)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    payload = {"data_quality": asdict(report), "chronological_split": split_summary(splits)}
    target = out / "data_quality.json"
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"\nSaved: {target}")

if __name__ == "__main__":
    main()
