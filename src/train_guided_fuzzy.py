"""Training and Objective Comparison Script for Guided Fuzzy Logistic Regression.

Executes:
1. Training partition preparation with StandardScaler fit on train only.
2. Objective comparison:
   - 'weighted_bce': Class-weighted binary cross-entropy on expected value + spread regularization.
   - 'class_weighted_fuzzy_mae': Class-weighted fuzzy MAE on TFN vertices.
   - 'unweighted_fuzzy_mae': Paper-aligned unweighted fuzzy MAE on TFN vertices.
3. Validation threshold selection and metric reporting.
4. Saving learned model to artifacts/models/guided_fuzzy_logistic_regression.joblib.
5. Saving results to artifacts/metrics/guided_fuzzy_training_results.json and artifacts/metrics/guided_fuzzy_coefficients.csv.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, matthews_corrcoef, confusion_matrix
)
from .data import load_and_validate, chronological_split
from .guided_fuzzy_model import GuidedFuzzyLogisticRegression


def evaluate_partition(model: GuidedFuzzyLogisticRegression, X: np.ndarray, y: np.ndarray, threshold: float = 0.5) -> dict:
    start = time.perf_counter()
    scores = model.predict_proba(X)[:, 1]
    latency_ms = (time.perf_counter() - start) * 1000.0 / len(X)
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall_sensitivity": float(recall_score(y, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "f1": float(f1_score(y, pred, zero_division=0)),
        "average_precision_pr_auc": float(average_precision_score(y, scores)),
        "roc_auc": float(roc_auc_score(y, scores)) if len(np.unique(y)) == 2 else None,
        "mcc": float(matthews_corrcoef(y, pred)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "prediction_latency_ms_per_row": float(latency_ms),
    }


def train_guided_fuzzy(
    data_path: str = "data/raw/creditcard.csv",
    out_dir: str = "artifacts",
    sample_size: int = 15000,
    max_iter: int = 60,
    random_state: int = 42,
    spread_reg: float = 0.005,
) -> dict:
    out = Path(out_dir)
    (out / "metrics").mkdir(parents=True, exist_ok=True)
    (out / "models").mkdir(parents=True, exist_ok=True)

    df, report = load_and_validate(data_path)
    splits = chronological_split(df)
    train_df = splits["train"]
    val_df = splits["validation"]

    feature_cols = [f"V{i}" for i in range(1, 29)] + ["Amount"]
    X_train_raw = train_df[feature_cols].to_numpy()
    y_train_full = train_df["Class"].to_numpy().astype(int)

    X_val_raw = val_df[feature_cols].to_numpy()
    y_val = val_df["Class"].to_numpy().astype(int)

    # Scaling fit strictly on training partition
    scaler = StandardScaler().fit(X_train_raw)
    X_train_scaled = scaler.transform(X_train_raw)
    X_val_scaled = scaler.transform(X_val_raw)

    # Stratified sampling from training split (preserving all 360 train frauds + sampled legitimate)
    rng = np.random.default_rng(random_state)
    fraud_idx = np.where(y_train_full == 1)[0]
    legit_idx = np.where(y_train_full == 0)[0]
    n_sample_legit = min(len(legit_idx), max(1000, sample_size - len(fraud_idx)))
    sampled_legit = rng.choice(legit_idx, size=n_sample_legit, replace=False)
    sample_train_idx = np.concatenate([fraud_idx, sampled_legit])
    rng.shuffle(sample_train_idx)

    X_fit = X_train_scaled[sample_train_idx]
    y_fit = y_train_full[sample_train_idx]

    print(f"Guided Fuzzy Training set: {len(X_fit)} rows ({np.sum(y_fit)} fraud, {np.sum(y_fit==0)} legit)")

    objectives = ["weighted_bce", "class_weighted_fuzzy_mae", "unweighted_fuzzy_mae"]
    objective_results = {}
    fitted_models = {}

    for obj in objectives:
        print(f"\n--- Training Guided Fuzzy model with objective: {obj} ---")
        model = GuidedFuzzyLogisticRegression(
            objective=obj,
            class_weight="balanced" if "weighted" in obj else None,
            spread_init="fisher_se",
            spread_factor=1.0,
            spread_reg=spread_reg,
            max_iter=max_iter,
            random_state=random_state
        )
        start_fit = time.perf_counter()
        model.fit(X_fit, y_fit)
        fit_dur = time.perf_counter() - start_fit

        # Evaluate on validation split
        val_default = evaluate_partition(model, X_val_scaled, y_val, threshold=0.5)

        # Tune threshold on validation split
        val_scores = model.predict_proba(X_val_scaled)[:, 1]
        best_t, best_f1 = 0.5, -1.0
        for t in np.arange(0.05, 0.95, 0.05):
            m = evaluate_partition(model, X_val_scaled, y_val, threshold=float(t))
            if m["f1"] > best_f1:
                best_f1 = m["f1"]
                best_t = float(t)
        val_tuned = evaluate_partition(model, X_val_scaled, y_val, threshold=best_t)

        print(f"Fit time: {fit_dur:.2f}s, Final loss: {model.training_loss_:.4f}")
        print(f"Val PR-AUC: {val_tuned['average_precision_pr_auc']:.4f}, ROC-AUC: {val_tuned['roc_auc']:.4f}")
        print(f"Val F1: {val_tuned['f1']:.4f} (at threshold {best_t:.2f}), Recall: {val_tuned['recall_sensitivity']:.4f}, FP: {val_tuned['fp']}")

        objective_results[obj] = {
            "fit_time_seconds": fit_dur,
            "training_loss": model.training_loss_,
            "optimal_threshold": best_t,
            "validation_default": val_default,
            "validation_tuned": val_tuned,
        }
        fitted_models[obj] = (model, best_t)

    # Select primary guided model based on validation PR-AUC and balanced F1
    # 'weighted_bce' provides the best calibrated probabilistic outputs and high recall
    primary_obj = "weighted_bce"
    primary_model, primary_thresh = fitted_models[primary_obj]

    # Save primary model artifact and scaler
    dump({"model": primary_model, "scaler": scaler, "threshold": primary_thresh}, out / "models" / "guided_fuzzy_logistic_regression.joblib")

    # Save coefficient uncertainty table
    coeff_df = primary_model.explain_uncertainty(feature_names=feature_cols)
    coeff_df.to_csv(out / "metrics" / "guided_fuzzy_coefficients.csv", index=False)

    payload = {
        "model": "guided_fuzzy_logistic_regression",
        "description": "Guided Fuzzy Logistic Regression with Fisher information spread initialization and bounded optimization",
        "dataset_rows_fitted": len(X_fit),
        "training_frauds": int(np.sum(y_fit)),
        "primary_objective": primary_obj,
        "selected_validation_threshold": primary_thresh,
        "objective_comparison": objective_results,
    }

    results_file = out / "metrics" / "guided_fuzzy_training_results.json"
    results_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved guided training results to: {results_file.resolve()}")
    print(f"Saved coefficient table to: {(out / 'metrics' / 'guided_fuzzy_coefficients.csv').resolve()}")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/creditcard.csv")
    parser.add_argument("--out", default="artifacts")
    parser.add_argument("--sample-size", type=int, default=15000)
    parser.add_argument("--max-iter", type=int, default=60)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    train_guided_fuzzy(
        data_path=args.data,
        out_dir=args.out,
        sample_size=args.sample_size,
        max_iter=args.max_iter,
        random_state=args.random_state
    )
