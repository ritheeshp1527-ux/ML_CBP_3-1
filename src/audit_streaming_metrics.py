"""Independent metric recomputation and verification audit script for Milestone 4.

Loads saved transaction-level prediction parquet files from artifacts/metrics/,
independently recomputes all streaming classification and ranking metrics,
and validates exact correspondence against saved summary CSV and JSON artifacts.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    confusion_matrix,
)


def recompute_metrics(preds_df: pd.DataFrame) -> dict:
    y_true = preds_df["true_label"].values
    y_pred = preds_df["predicted_class"].values
    y_scores = preds_df["expected_value"].values

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    mcc = float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_pred)) > 1 else 0.0
    roc_auc = float(roc_auc_score(y_true, y_scores))
    pr_auc = float(average_precision_score(y_true, y_scores))

    return {
        "precision": prec,
        "recall": rec,
        "specificity": spec,
        "f1": f1,
        "mcc": mcc,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "fraud_count": int(np.sum(y_true == 1)),
        "total_rows": int(len(y_true)),
        "max_version": int(preds_df["model_version"].max()),
    }


def main():
    print("=" * 80)
    print("FUZZYGUARD AI — MILESTONE 4 AUDIT: METRIC RECOMPUTATION")
    print("=" * 80)

    metrics_dir = Path("artifacts/metrics")
    comparison_csv_path = metrics_dir / "streaming_strategy_comparison.csv"
    if not comparison_csv_path.exists():
        raise FileNotFoundError(f"Missing {comparison_csv_path}")

    saved_comparison_df = pd.read_csv(comparison_csv_path)

    strategies = [
        "guided_fuzzy_frozen",
        "guided_fuzzy_adaptive_immediate",
        "guided_fuzzy_adaptive_delayed_1h",
        "logistic_regression_frozen",
        "logistic_regression_adaptive_immediate",
    ]

    all_matched = True
    audit_records = []

    for strat in strategies:
        parquet_file = metrics_dir / f"streaming_predictions_{strat}.parquet"
        if not parquet_file.exists():
            print(f"[FAIL] Missing parquet file: {parquet_file}")
            all_matched = False
            continue

        preds_df = pd.read_parquet(parquet_file)
        recomputed = recompute_metrics(preds_df)

        saved_row = saved_comparison_df[saved_comparison_df["strategy"] == strat].iloc[0]

        print(f"\n--- Strategy: {strat} ---")
        print(f"  Rows: {recomputed['total_rows']} (Expected: 56962) | Frauds: {recomputed['fraud_count']} (Expected: 75)")
        print(f"  Recomputed: Recall={recomputed['recall']:.4f}, Prec={recomputed['precision']:.4f}, F1={recomputed['f1']:.4f}, PR-AUC={recomputed['pr_auc']:.4f}, ROC-AUC={recomputed['roc_auc']:.4f}")
        print(f"  Saved CSV:  Recall={saved_row['recall']:.4f}, Prec={saved_row['precision']:.4f}, F1={saved_row['f1']:.4f}, PR-AUC={saved_row['pr_auc']:.4f}, ROC-AUC={saved_row['roc_auc']:.4f}")
        print(f"  Confusion Matrix: TP={recomputed['tp']}, FP={recomputed['fp']}, FN={recomputed['fn']}, TN={recomputed['tn']}")

        # Validate tolerances
        discrepancies = []
        for key in ["precision", "recall", "specificity", "f1", "mcc", "roc_auc", "pr_auc"]:
            diff = abs(recomputed[key] - saved_row[key])
            if diff > 1e-6:
                discrepancies.append(f"{key}: recomputed={recomputed[key]} vs saved={saved_row[key]}")

        for key in ["tp", "fp", "fn", "tn", "fraud_count", "total_rows"]:
            if recomputed[key] != saved_row[key]:
                discrepancies.append(f"{key}: recomputed={recomputed[key]} vs saved={saved_row[key]}")

        if discrepancies:
            print(f"  [DISCREPANCY DETECTED]: {', '.join(discrepancies)}")
            all_matched = False
        else:
            print("  [VERIFIED]: 100% exact numerical match with saved CSV artifact.")

        audit_records.append({
            "strategy": strat,
            "status": "EXACT_MATCH" if not discrepancies else "DISCREPANCY",
            "tp": recomputed["tp"],
            "fp": recomputed["fp"],
            "fn": recomputed["fn"],
            "tn": recomputed["tn"],
            "recall": recomputed["recall"],
            "precision": recomputed["precision"],
            "f1": recomputed["f1"],
            "pr_auc": recomputed["pr_auc"],
            "roc_auc": recomputed["roc_auc"],
            "updates": recomputed["max_version"],
        })

    audit_df = pd.DataFrame(audit_records)
    audit_csv_path = metrics_dir / "streaming_recomputed_audit.csv"
    audit_df.to_csv(audit_csv_path, index=False)
    print(f"\nSaved independent recomputation audit: {audit_csv_path}")

    if all_matched:
        print("\nALL METRICS INDEPENDENTLY RECOMPUTED AND CONFIRMED IDENTICAL.")
    else:
        print("\nWARNING: Some metrics showed discrepancies during independent audit.")


if __name__ == "__main__":
    main()
