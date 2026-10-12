"""Comprehensive Model Evaluation & Comparison Script.

Evaluates and compares:
1. Conventional Logistic Regression
2. Class-weighted Logistic Regression
3. Random Forest (balanced subsample)
4. Published Table 5 Fuzzy Logistic Regression
5. Newly learned Fuzzy Logistic Regression (Monte Carlo reference)
6. Guided Fuzzy Logistic Regression (bounded L-BFGS-B optimization)

Saves results in:
- artifacts/metrics/model_comparison.csv
- artifacts/metrics/model_comparison.json
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, matthews_corrcoef, confusion_matrix
)
from .data import load_and_validate, chronological_split
from .fuzzy_model import FuzzyLogisticRegression
from .guided_fuzzy_model import GuidedFuzzyLogisticRegression


def evaluate_scores(y_true: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> dict:
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    prec = float(precision_score(y_true, pred, zero_division=0))
    rec = float(recall_score(y_true, pred, zero_division=0))
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    f1 = float(f1_score(y_true, pred, zero_division=0))
    mcc = float(matthews_corrcoef(y_true, pred))
    pr_auc = float(average_precision_score(y_true, scores))
    roc_auc = float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) == 2 else None

    return {
        "threshold": float(threshold),
        "precision": prec,
        "recall_sensitivity": rec,
        "specificity": spec,
        "f1": f1,
        "average_precision_pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "mcc": mcc,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0,
    }


def compute_bootstrap_uncertainty(
    y_true: np.ndarray, scores: np.ndarray, threshold: float, n_boot: int = 500, seed: int = 42
) -> dict:
    """Computes 95% bootstrap confidence intervals for test set metrics."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    f1_list, rec_list, prec_list, prauc_list = [], [], [], []

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y_b = y_true[idx]
        if np.sum(y_b) == 0:
            continue
        sc_b = scores[idx]
        pr_b = (sc_b >= threshold).astype(int)
        f1_list.append(f1_score(y_b, pr_b, zero_division=0))
        rec_list.append(recall_score(y_b, pr_b, zero_division=0))
        prec_list.append(precision_score(y_b, pr_b, zero_division=0))
        prauc_list.append(average_precision_score(y_b, sc_b))

    return {
        "f1_ci95": [float(np.percentile(f1_list, 2.5)), float(np.percentile(f1_list, 97.5))],
        "recall_ci95": [float(np.percentile(rec_list, 2.5)), float(np.percentile(rec_list, 97.5))],
        "precision_ci95": [float(np.percentile(prec_list, 2.5)), float(np.percentile(prec_list, 97.5))],
        "pr_auc_ci95": [float(np.percentile(prauc_list, 2.5)), float(np.percentile(prauc_list, 97.5))],
    }


def compare_all_models(data_path: str = "data/raw/creditcard.csv", out_dir: str = "artifacts") -> pd.DataFrame:
    out = Path(out_dir)
    models_dir = out / "models"
    metrics_dir = out / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    df, report = load_and_validate(data_path)
    splits = chronological_split(df)
    train_df, val_df, test_df = splits["train"], splits["validation"], splits["test"]

    # Predictors: 30 features for sklearn pipelines, 29 features for fuzzy models
    feature_cols_all = [c for c in df.columns if c != "Class"]
    feature_cols_fuzzy = [f"V{i}" for i in range(1, 29)] + ["Amount"]

    X_val_all = val_df[feature_cols_all]
    y_val = val_df["Class"].to_numpy().astype(int)

    X_test_all = test_df[feature_cols_all]
    y_test = test_df["Class"].to_numpy().astype(int)

    X_val_fuzzy = val_df[feature_cols_fuzzy].to_numpy()
    X_test_fuzzy = test_df[feature_cols_fuzzy].to_numpy()

    # Prevalences
    val_fraud_rate = float(np.mean(y_val))
    test_fraud_rate = float(np.mean(y_test))

    # Baseline fit times
    baseline_json = metrics_dir / "baseline_results.json"
    fit_times = {}
    if baseline_json.exists():
        b_data = json.loads(baseline_json.read_text(encoding="utf-8"))
        for r in b_data.get("results", []):
            fit_times[r["model"]] = r.get("fit_seconds", 0.0)

    # Fuzzy MC fit time
    fuzzy_train_json = metrics_dir / "fuzzy_training_results.json"
    if fuzzy_train_json.exists():
        ft_data = json.loads(fuzzy_train_json.read_text(encoding="utf-8"))
        fit_times["fuzzy_learned_mc"] = ft_data.get("configuration", {}).get("fit_time_seconds", 0.0)

    # Guided fuzzy results
    guided_json = metrics_dir / "guided_fuzzy_training_results.json"
    guided_obj_data = {}
    if guided_json.exists():
        g_data = json.loads(guided_json.read_text(encoding="utf-8"))
        guided_obj_data = g_data.get("objective_comparison", {})
        primary_obj = g_data.get("primary_objective", "weighted_bce")
        fit_times["guided_fuzzy"] = guided_obj_data.get(primary_obj, {}).get("fit_time_seconds", 0.0)

    # Load models
    models = {}
    if (models_dir / "logistic_regression.joblib").exists():
        models["logistic_regression"] = ("sklearn", load(models_dir / "logistic_regression.joblib"), None)
    if (models_dir / "logistic_regression_balanced.joblib").exists():
        models["logistic_regression_balanced"] = ("sklearn", load(models_dir / "logistic_regression_balanced.joblib"), None)
    if (models_dir / "random_forest.joblib").exists():
        models["random_forest"] = ("sklearn", load(models_dir / "random_forest.joblib"), None)

    # Fuzzy models
    models["fuzzy_table_5"] = ("fuzzy", FuzzyLogisticRegression.from_table_5(), None)

    if (models_dir / "fuzzy_logistic_regression.joblib").exists():
        models["fuzzy_learned_mc"] = ("fuzzy", load(models_dir / "fuzzy_logistic_regression.joblib"), None)

    if (models_dir / "guided_fuzzy_logistic_regression.joblib").exists():
        guided_bundle = load(models_dir / "guided_fuzzy_logistic_regression.joblib")
        models["guided_fuzzy"] = ("guided", guided_bundle["model"], guided_bundle["scaler"])

    rows = []
    detailed_results = {}

    for name, (m_type, model, scaler) in models.items():
        fit_time = fit_times.get(name, 0.0)

        if m_type == "sklearn":
            start = time.perf_counter()
            val_scores = model.predict_proba(X_val_all)[:, 1]
            val_lat = (time.perf_counter() - start) * 1000.0 / len(X_val_all)

            start = time.perf_counter()
            test_scores = model.predict_proba(X_test_all)[:, 1]
            test_lat = (time.perf_counter() - start) * 1000.0 / len(X_test_all)

        elif m_type == "fuzzy":
            start = time.perf_counter()
            val_scores = model.predict_proba(X_val_fuzzy)[:, 1]
            val_lat = (time.perf_counter() - start) * 1000.0 / len(X_val_fuzzy)

            start = time.perf_counter()
            test_scores = model.predict_proba(X_test_fuzzy)[:, 1]
            test_lat = (time.perf_counter() - start) * 1000.0 / len(X_test_fuzzy)

        elif m_type == "guided":
            X_val_s = scaler.transform(X_val_fuzzy)
            X_test_s = scaler.transform(X_test_fuzzy)

            start = time.perf_counter()
            val_scores = model.predict_proba(X_val_s)[:, 1]
            val_lat = (time.perf_counter() - start) * 1000.0 / len(X_val_fuzzy)

            start = time.perf_counter()
            test_scores = model.predict_proba(X_test_s)[:, 1]
            test_lat = (time.perf_counter() - start) * 1000.0 / len(X_test_fuzzy)

        # Threshold selection on validation partition ONLY
        best_f1 = -1.0
        best_t = 0.5
        for t in np.arange(0.05, 0.95, 0.05):
            pred_v = (val_scores >= t).astype(int)
            f1_v = f1_score(y_val, pred_v, zero_division=0)
            if f1_v > best_f1:
                best_f1 = f1_v
                best_t = float(t)

        val_m_default = evaluate_scores(y_val, val_scores, threshold=0.5)
        val_m_tuned = evaluate_scores(y_val, val_scores, threshold=best_t)

        # Apply validation-selected threshold to untouched test partition
        test_m_default = evaluate_scores(y_test, test_scores, threshold=0.5)
        test_m_tuned = evaluate_scores(y_test, test_scores, threshold=best_t)

        # Bootstrap confidence intervals on test partition
        test_ci = compute_bootstrap_uncertainty(y_test, test_scores, threshold=best_t)

        detailed_results[name] = {
            "fit_time_seconds": fit_time,
            "validation_latency_ms_per_row": val_lat,
            "test_latency_ms_per_row": test_lat,
            "optimal_validation_threshold": best_t,
            "validation_default": val_m_default,
            "validation_tuned": val_m_tuned,
            "test_default": test_m_default,
            "test_tuned": test_m_tuned,
            "test_ci95": test_ci,
        }

        # 1. Validation default row
        rows.append({
            "model": name,
            "split": "validation",
            "threshold_mode": "default_0.5",
            "threshold": 0.5,
            "precision": val_m_default["precision"],
            "recall": val_m_default["recall_sensitivity"],
            "specificity": val_m_default["specificity"],
            "f1": val_m_default["f1"],
            "pr_auc": val_m_default["average_precision_pr_auc"],
            "roc_auc": val_m_default["roc_auc"],
            "mcc": val_m_default["mcc"],
            "tn": val_m_default["tn"],
            "fp": val_m_default["fp"],
            "fn": val_m_default["fn"],
            "tp": val_m_default["tp"],
            "fit_seconds": fit_time,
            "latency_ms_per_row": val_lat,
            "fraud_prevalence": val_fraud_rate,
            "f1_ci95_low": None, "f1_ci95_high": None,
        })

        # 2. Validation tuned row
        rows.append({
            "model": name,
            "split": "validation",
            "threshold_mode": "val_tuned",
            "threshold": best_t,
            "precision": val_m_tuned["precision"],
            "recall": val_m_tuned["recall_sensitivity"],
            "specificity": val_m_tuned["specificity"],
            "f1": val_m_tuned["f1"],
            "pr_auc": val_m_tuned["average_precision_pr_auc"],
            "roc_auc": val_m_tuned["roc_auc"],
            "mcc": val_m_tuned["mcc"],
            "tn": val_m_tuned["tn"],
            "fp": val_m_tuned["fp"],
            "fn": val_m_tuned["fn"],
            "tp": val_m_tuned["tp"],
            "fit_seconds": fit_time,
            "latency_ms_per_row": val_lat,
            "fraud_prevalence": val_fraud_rate,
            "f1_ci95_low": None, "f1_ci95_high": None,
        })

        # 3. Test row using validation-selected threshold
        rows.append({
            "model": name,
            "split": "test",
            "threshold_mode": "val_selected_threshold",
            "threshold": best_t,
            "precision": test_m_tuned["precision"],
            "recall": test_m_tuned["recall_sensitivity"],
            "specificity": test_m_tuned["specificity"],
            "f1": test_m_tuned["f1"],
            "pr_auc": test_m_tuned["average_precision_pr_auc"],
            "roc_auc": test_m_tuned["roc_auc"],
            "mcc": test_m_tuned["mcc"],
            "tn": test_m_tuned["tn"],
            "fp": test_m_tuned["fp"],
            "fn": test_m_tuned["fn"],
            "tp": test_m_tuned["tp"],
            "fit_seconds": fit_time,
            "latency_ms_per_row": test_lat,
            "fraud_prevalence": test_fraud_rate,
            "f1_ci95_low": test_ci["f1_ci95"][0],
            "f1_ci95_high": test_ci["f1_ci95"][1],
        })

    comp_df = pd.DataFrame(rows)
    comp_df.to_csv(metrics_dir / "model_comparison.csv", index=False)
    (metrics_dir / "model_comparison.json").write_text(json.dumps(detailed_results, indent=2), encoding="utf-8")
    print(f"Saved model comparison to: {(metrics_dir / 'model_comparison.csv').resolve()}")
    return comp_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/creditcard.csv")
    parser.add_argument("--out", default="artifacts")
    args = parser.parse_args()
    compare_all_models(args.data, args.out)
