"""Fuzzy Logistic Regression Coefficient Learning Script.

Reproduces the Monte Carlo optimization framework from Charizanos et al. (2024)
and https://github.com/GZanos/Fuzzy-Logistic-Regression.

Features:
- Fits fuzzy TFN coefficients strictly on the chronological training partition.
- Minimizes the author's fuzzy MAE / MSE / RMSE loss function against fuzzified labels.
- Preserves exact triangular fuzzy number (TFN) ordering and arithmetic.
- Configurable iterations, sample sizes, random seeds, and parameter bounds.
- Saves learned coefficients and training metadata to artifacts/metrics/fuzzy_training_results.json.
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
from .fuzzy_model import (
    FuzzyLogisticRegression, fuzzy_linear_combination,
    fuzzy_logistic_predict_tfn, fuzzify_binary_response,
    fuzzy_loss, dubois_prade_expected_value
)


def evaluate_partition(model: FuzzyLogisticRegression, X: np.ndarray, y: np.ndarray, threshold: float = 0.5) -> dict:
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


def train_fuzzy_mc(
    data_path: str = "data/raw/creditcard.csv",
    out_dir: str = "artifacts",
    n_iterations: int = 2000,
    sample_size: int = 5000,
    random_state: int = 42,
    b0_lower: float = -1.0,
    b0_upper: float = 1.0,
    bi_lower: float = -1.0,
    bi_upper: float = 1.0,
    loss_metric: str = "mae",
    use_scaling: bool = False,
    quick_smoke_test: bool = False
) -> dict:
    out = Path(out_dir)
    (out / "metrics").mkdir(parents=True, exist_ok=True)
    (out / "models").mkdir(parents=True, exist_ok=True)

    df, report = load_and_validate(data_path)
    splits = chronological_split(df)
    train_df = splits["train"]
    val_df = splits["validation"]

    feature_cols = [f"V{i}" for i in range(1, 29)] + ["Amount"]
    X_train_full = train_df[feature_cols].to_numpy()
    y_train_full = train_df["Class"].to_numpy().astype(int)
    X_val = val_df[feature_cols].to_numpy()
    y_val = val_df["Class"].to_numpy().astype(int)

    if use_scaling:
        scaler = StandardScaler().fit(X_train_full)
        X_train_full = scaler.transform(X_train_full)
        X_val = scaler.transform(X_val)

    if quick_smoke_test:
        n_iterations = 200
        sample_size = 2000

    # Sample training partition matching author's sample size (n=5000 with stratified fraud rate)
    rng = np.random.default_rng(random_state)
    if sample_size > 0 and sample_size < len(X_train_full):
        # Extract stratified sample from train partition to ensure fraud representation
        fraud_idx = np.where(y_train_full == 1)[0]
        legit_idx = np.where(y_train_full == 0)[0]
        # Match paper's 0.2% fraud proportion in sample
        n_fraud = min(len(fraud_idx), max(10, int(np.ceil(0.002 * sample_size))))
        n_legit = sample_size - n_fraud
        sampled_fraud = rng.choice(fraud_idx, size=n_fraud, replace=False)
        sampled_legit = rng.choice(legit_idx, size=n_legit, replace=False)
        train_idx = np.concatenate([sampled_fraud, sampled_legit])
        rng.shuffle(train_idx)
        X_fit = X_train_full[train_idx]
        y_fit = y_train_full[train_idx]
    else:
        X_fit = X_train_full
        y_fit = y_train_full

    n_features = X_fit.shape[1]
    print(f"Starting Fuzzy MC training: {n_iterations} iterations, {len(X_fit)} samples ({np.sum(y_fit)} fraud)...")

    # Author fuzzification parameters: m=2.5, U ~ Uniform(0, 1.5)
    y_fit_tfn = fuzzify_binary_response(y_fit, m=2.5, u_max=1.5, seed=random_state)

    best_beta = None
    best_loss = float("inf")
    loss_history = []
    start_fit = time.perf_counter()

    for it in range(n_iterations):
        # Sample b0 TFN [a <= b <= c] uniformly in [b0_lower, b0_upper]
        b0_cand = np.sort(rng.uniform(b0_lower, b0_upper, size=3))
        # Sample bi TFNs [a <= b <= c] uniformly in [bi_lower, bi_upper]
        bi_cand = np.sort(rng.uniform(bi_lower, bi_upper, size=(n_features, 3)), axis=1)
        candidate_beta = np.vstack([b0_cand, bi_cand])

        # Evaluate fuzzy predictions and loss
        p_tfn = fuzzy_logistic_predict_tfn(X_fit, candidate_beta)
        loss = fuzzy_loss(y_fit_tfn, p_tfn, metric=loss_metric)

        if loss < best_loss:
            best_loss = loss
            best_beta = candidate_beta
            loss_history.append({"iteration": it, "loss": float(loss)})

    fit_seconds = time.perf_counter() - start_fit
    print(f"Fit complete in {fit_seconds:.2f}s! Best {loss_metric.upper()} loss: {best_loss:.4f}")

    # Build model with learned coefficients
    model = FuzzyLogisticRegression(
        beta=best_beta,
        n_iterations=n_iterations,
        random_state=random_state,
        b0_lower=b0_lower,
        b0_upper=b0_upper,
        bi_lower=bi_lower,
        bi_upper=bi_upper,
        loss_metric=loss_metric
    )
    model.best_loss_ = best_loss

    # Evaluate on validation partition
    val_eval_default = evaluate_partition(model, X_val, y_val, threshold=0.5)

    # Threshold tuning on validation partition only
    val_scores = model.predict_proba(X_val)[:, 1]
    best_thresh = 0.5
    best_f1 = -1.0
    for t in np.arange(0.1, 0.9, 0.05):
        m = evaluate_partition(model, X_val, y_val, threshold=float(t))
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_thresh = float(t)
    val_eval_optimal = evaluate_partition(model, X_val, y_val, threshold=best_thresh)

    # Dump model artifact
    dump(model, out / "models" / "fuzzy_logistic_regression.joblib")

    # Serialize results
    results_payload = {
        "model": "fuzzy_logistic_regression_mc",
        "description": "Fuzzy Logistic Regression fitted via author Monte Carlo random search on training partition",
        "configuration": {
            "n_iterations": n_iterations,
            "sample_size": len(X_fit),
            "training_fraud_count": int(np.sum(y_fit)),
            "random_state": random_state,
            "b0_bounds": [b0_lower, b0_upper],
            "bi_bounds": [bi_lower, bi_upper],
            "loss_metric": loss_metric,
            "use_scaling": use_scaling,
            "fit_time_seconds": fit_seconds,
        },
        "best_training_loss": float(best_loss),
        "loss_progression": loss_history,
        "learned_coefficients": [
            {
                "index": i,
                "name": "Intercept" if i == 0 else f"V{i}" if i < 29 else "Amount",
                "tfn_a": float(best_beta[i, 0]),
                "tfn_b": float(best_beta[i, 1]),
                "tfn_c": float(best_beta[i, 2]),
            }
            for i in range(len(best_beta))
        ],
        "validation_evaluation_default_threshold": val_eval_default,
        "optimal_validation_threshold": best_thresh,
        "validation_evaluation_optimal_threshold": val_eval_optimal,
    }

    results_file = out / "metrics" / "fuzzy_training_results.json"
    results_file.write_text(json.dumps(results_payload, indent=2), encoding="utf-8")
    print(f"Saved training results to: {results_file.resolve()}")
    return results_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/creditcard.csv")
    parser.add_argument("--out", default="artifacts")
    parser.add_argument("--n-iterations", type=int, default=2000)
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--use-scaling", action="store_true")
    parser.add_argument("--quick", action="store_true", help="Quick smoke test with 200 iterations")
    args = parser.parse_args()

    train_fuzzy_mc(
        data_path=args.data,
        out_dir=args.out,
        n_iterations=args.n_iterations,
        sample_size=args.sample_size,
        random_state=args.random_state,
        use_scaling=args.use_scaling,
        quick_smoke_test=args.quick
    )
