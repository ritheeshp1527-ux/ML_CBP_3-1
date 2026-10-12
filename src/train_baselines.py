from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, matthews_corrcoef, confusion_matrix)
from .data import load_and_validate, chronological_split, split_summary

def metrics(y_true, scores, threshold=0.5):
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall_sensitivity": float(recall_score(y_true, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "average_precision_pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) == 2 else None,
        "mcc": float(matthews_corrcoef(y_true, pred)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
        "false_negative_rate": float(fn / (fn + tp)) if fn + tp else 0.0,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/creditcard.csv")
    parser.add_argument("--out", default="artifacts")
    parser.add_argument("--rf-max-train-rows", type=int, default=100000,
                        help="Cap only the RF training rows to keep first runs manageable; use 0 for all.")
    parser.add_argument("--quick", action="store_true", help="Run only Logistic Regression for smoke testing.")
    args = parser.parse_args()
    out = Path(args.out); (out / "metrics").mkdir(parents=True, exist_ok=True)
    (out / "models").mkdir(parents=True, exist_ok=True)

    df, report = load_and_validate(args.data)
    splits = chronological_split(df)
    train, val, test = splits["train"], splits["validation"], splits["test"]
    print("Data report:", json.dumps(report.__dict__, indent=2))
    print("Split summary:", json.dumps(split_summary(splits), indent=2))

    feature_cols = [c for c in df.columns if c != "Class"]
    # All preprocessing is fitted on training data only.
    X_train, y_train = train[feature_cols], train["Class"].astype(int)
    X_val, y_val = val[feature_cols], val["Class"].astype(int)
    X_test, y_test = test[feature_cols], test["Class"].astype(int)

    models = {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight=None, solver="lbfgs")
        ),
        "logistic_regression_balanced": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs")
        ),
    }
    if not args.quick:
        models["random_forest"] = RandomForestClassifier(
            n_estimators=120, max_depth=None, min_samples_leaf=2,
            class_weight="balanced_subsample", n_jobs=-1, random_state=42
        )

    all_results = []
    from joblib import dump
    for name, model in models.items():
        fit_X, fit_y = X_train, y_train
        if name == "random_forest" and args.rf_max_train_rows > 0 and len(fit_X) > args.rf_max_train_rows:
            # Preserve chronology by using the most recent portion of training data.
            fit_X, fit_y = fit_X.iloc[-args.rf_max_train_rows:], fit_y.iloc[-args.rf_max_train_rows:]
            if fit_y.nunique() < 2:
                raise ValueError("RF capped training slice has only one class; increase --rf-max-train-rows.")
        start = time.perf_counter()
        model.fit(fit_X, fit_y)
        fit_seconds = time.perf_counter() - start
        start = time.perf_counter()
        val_scores = model.predict_proba(X_val)[:, 1]
        val_latency_ms_per_row = (time.perf_counter() - start) * 1000 / max(len(X_val), 1)
        # Threshold remains 0.5 for the baseline run; validation scores are saved for later tuning.
        start = time.perf_counter()
        test_scores = model.predict_proba(X_test)[:, 1]
        test_latency_ms_per_row = (time.perf_counter() - start) * 1000 / max(len(X_test), 1)
        result = {
            "model": name,
            "fit_seconds": float(fit_seconds),
            "validation_latency_ms_per_row": float(val_latency_ms_per_row),
            "test_latency_ms_per_row": float(test_latency_ms_per_row),
            "training_rows_used": int(len(fit_X)),
            "validation": metrics(y_val.to_numpy(), val_scores),
            "test": metrics(y_test.to_numpy(), test_scores),
            "threshold_tuning_note": "Threshold fixed at 0.5 for initial baseline; tune on validation only in a later step."
        }
        all_results.append(result)
        dump(model, out / "models" / f"{name}.joblib")
        pd.DataFrame({"y_true": y_val.to_numpy(), "score": val_scores}).to_csv(
            out / "metrics" / f"{name}_validation_scores.csv", index=False)
        pd.DataFrame({"y_true": y_test.to_numpy(), "score": test_scores}).to_csv(
            out / "metrics" / f"{name}_test_scores.csv", index=False)
        print(f"{name}: test recall={result['test']['recall_sensitivity']:.4f}, "
              f"PR-AUC={result['test']['average_precision_pr_auc']:.4f}, "
              f"MCC={result['test']['mcc']:.4f}")

    payload = {
        "dataset": report.__dict__,
        "split": split_summary(splits),
        "protocol": {
            "split": "chronological 60/20/20 by original row order",
            "threshold": 0.5,
            "test_used_for_tuning": False,
            "duplicates_dropped": False,
            "caveat": "Initial baseline run; inspect split fraud counts and threshold behavior before interpreting."
        },
        "results": all_results
    }
    (out / "metrics" / "baseline_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pd.DataFrame([{"model": r["model"], **r["test"]} for r in all_results]).to_csv(
        out / "metrics" / "baseline_comparison.csv", index=False)
    print(f"Saved results under {out.resolve()}")

if __name__ == "__main__":
    main()
