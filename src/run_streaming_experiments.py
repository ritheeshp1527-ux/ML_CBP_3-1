"""Executes Milestone 4 streaming experiments and concept drift evaluation.

Runs prequential streaming simulations on the untouched chronological test partition:
1. Frozen Guided Fuzzy Model
2. Adaptive Guided Fuzzy Model (Immediate Labels, delay = 0s)
3. Adaptive Guided Fuzzy Model (Delayed Labels, delay = 3600s / 1 hr)
4. Frozen Logistic Regression Baseline
5. Adaptive Logistic Regression Baseline (Immediate Labels, delay = 0s)

Evaluates:
- Full prequential test metrics (Precision, Recall, F1, PR-AUC, ROC-AUC, MCC, Latency)
- Concept drift over 4 chronological time windows
- Uncertainty-directed triage policy across human review budgets [0.5%, 1.0%, 2.0%, 5.0%]

Outputs saved to artifacts/metrics/.
"""
from __future__ import annotations
import json
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd

from src.data import load_and_validate, chronological_split
from src.guided_fuzzy_model import GuidedFuzzyLogisticRegression
from src.streaming_adaptation import (
    StreamingSimulator,
    OnlineLogisticRegressionBaseline,
    compute_stream_metrics,
    evaluate_streaming_windows,
    evaluate_triage_policy,
)


def main():
    print("=" * 80)
    print("FUZZYGUARD AI — MILESTONE 4: PREQUENTIAL STREAMING EVALUATION")
    print("=" * 80)

    data_path = Path("data/raw/creditcard.csv")
    df, report = load_and_validate(data_path)
    feature_cols = [c for c in df.columns if c not in ("Time", "Class")]

    splits = chronological_split(df)
    train_df = splits["train"]
    val_df = splits["validation"]
    test_df = splits["test"]

    print(f"Loaded dataset: {len(df)} transactions.")
    print(f"  Train: {len(train_df)} rows, {train_df['Class'].sum()} frauds")
    print(f"  Val:   {len(val_df)} rows, {val_df['Class'].sum()} frauds")
    print(f"  Test:  {len(test_df)} rows, {test_df['Class'].sum()} frauds (untouched eval partition)")

    # Load trained guided fuzzy model artifact
    model_artifact = joblib.load("artifacts/models/guided_fuzzy_logistic_regression.joblib")
    guided_fuzzy_model: GuidedFuzzyLogisticRegression = model_artifact["model"]
    scaler = model_artifact["scaler"]
    fuzzy_val_threshold = float(model_artifact["threshold"])
    print(f"Loaded Guided Fuzzy Model (Validation threshold: {fuzzy_val_threshold:.2f})")

    # Build balanced historical replay reservoir from training partition
    # (100 positive fraud cases, 100 negative legitimate cases)
    train_frauds = train_df[train_df["Class"] == 1]
    train_legit = train_df[train_df["Class"] == 0]
    replay_fraud = train_frauds.sample(n=min(100, len(train_frauds)), random_state=42)
    replay_legit = train_legit.sample(n=100, random_state=42)
    replay_df = pd.concat([replay_fraud, replay_legit]).sample(frac=1.0, random_state=42)

    replay_X = scaler.transform(replay_df[feature_cols].values)
    replay_y = replay_df["Class"].values.astype(int)

    # Prepare Baseline Online Logistic Regression
    baseline_lr = OnlineLogisticRegressionBaseline(random_state=42)
    train_X_scaled = scaler.transform(train_df[feature_cols].values)
    train_y = train_df["Class"].values.astype(int)
    baseline_lr.fit(train_X_scaled, train_y)
    baseline_val_threshold = 0.50

    # Define streaming strategies
    strategies = [
        {
            "name": "guided_fuzzy_frozen",
            "model": guided_fuzzy_model,
            "threshold": fuzzy_val_threshold,
            "adaptive": False,
            "label_delay_seconds": 0.0,
            "description": "Frozen Guided Fuzzy Model (no online updates)",
        },
        {
            "name": "guided_fuzzy_adaptive_immediate",
            "model": guided_fuzzy_model,
            "threshold": fuzzy_val_threshold,
            "adaptive": True,
            "label_delay_seconds": 0.0,
            "description": "Adaptive Guided Fuzzy (immediate labels, delay = 0s)",
        },
        {
            "name": "guided_fuzzy_adaptive_delayed_1h",
            "model": guided_fuzzy_model,
            "threshold": fuzzy_val_threshold,
            "adaptive": True,
            "label_delay_seconds": 3600.0,
            "description": "Adaptive Guided Fuzzy (delayed labels, delay = 1 hour / 3600s)",
        },
        {
            "name": "logistic_regression_frozen",
            "model": baseline_lr,
            "threshold": baseline_val_threshold,
            "adaptive": False,
            "label_delay_seconds": 0.0,
            "description": "Frozen Online Logistic Regression Baseline",
        },
        {
            "name": "logistic_regression_adaptive_immediate",
            "model": baseline_lr,
            "threshold": baseline_val_threshold,
            "adaptive": True,
            "label_delay_seconds": 0.0,
            "description": "Adaptive Online Logistic Regression (immediate labels, delay = 0s)",
        },
    ]

    metrics_dir = Path("artifacts/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)

    strategy_metrics_list = []
    all_window_records = []
    predictions_cache = {}

    print("\nExecuting Prequential Streaming Simulations on Test Partition...")
    for strat in strategies:
        strat_name = strat["name"]
        print(f"\n--- Strategy: {strat_name} ({strat['description']}) ---")

        sim = StreamingSimulator(
            model=strat["model"],
            scaler=scaler,
            threshold=strat["threshold"],
            label_delay_seconds=strat["label_delay_seconds"],
            batch_size=2500,
            replay_reservoir_X=replay_X,
            replay_reservoir_y=replay_y,
            adaptive=strat["adaptive"],
            proximal_weight=1.0,
            max_iter=10,
            random_state=42,
        )

        t_sim_start = time.perf_counter()
        preds_df, audit_df = sim.run_stream(test_df, feature_cols)
        sim_elapsed = time.perf_counter() - t_sim_start

        predictions_cache[strat_name] = preds_df

        # Save transaction-level predictions to disk for independent verification
        strat_pred_path = metrics_dir / f"streaming_predictions_{strat_name}.parquet"
        preds_df.to_parquet(strat_pred_path, index=False)

        # Overall stream metrics
        y_true = preds_df["true_label"].values
        y_pred = preds_df["predicted_class"].values
        y_scores = preds_df["expected_value"].values

        m = compute_stream_metrics(y_true, y_pred, y_scores)
        m["strategy"] = strat_name
        m["description"] = strat["description"]
        m["threshold"] = strat["threshold"]
        m["adaptive"] = strat["adaptive"]
        m["label_delay_seconds"] = strat["label_delay_seconds"]
        m["model_updates_count"] = int(sim.model_version_)
        m["total_simulation_seconds"] = float(sim_elapsed)
        m["adaptation_time_seconds"] = float(sim.total_adaptation_seconds_)
        m["avg_inference_latency_ms"] = float((sim.total_inference_seconds_ / len(test_df)) * 1000.0)

        strategy_metrics_list.append(m)
        print(f"  Recall: {m['recall']*100:.2f}% | Precision: {m['precision']*100:.2f}% | F1: {m['f1']:.4f}")
        print(f"  PR-AUC: {m['pr_auc']:.4f} | ROC-AUC: {m['roc_auc']:.4f} | MCC: {m['mcc']:.4f}")
        print(f"  FP: {m['fp']} | FN: {m['fn']} | Model Updates: {m['model_updates_count']}")
        print(f"  Adaptation Time: {m['adaptation_time_seconds']:.2f}s | Latency: {m['avg_inference_latency_ms']:.4f} ms/tx")

        # Evaluate sequential windows (drift analysis)
        windows_df = evaluate_streaming_windows(preds_df, n_windows=4)
        windows_df["strategy"] = strat_name
        all_window_records.append(windows_df)

    # Assemble summary tables
    strategy_comparison_df = pd.DataFrame(strategy_metrics_list)
    streaming_windows_df = pd.concat(all_window_records, ignore_index=True)

    strategy_csv_path = metrics_dir / "streaming_strategy_comparison.csv"
    strategy_comparison_df.to_csv(strategy_csv_path, index=False)
    print(f"\nSaved strategy comparison: {strategy_csv_path}")

    windows_csv_path = metrics_dir / "streaming_windows_drift.csv"
    streaming_windows_df.to_csv(windows_csv_path, index=False)
    print(f"Saved drift windows: {windows_csv_path}")

    # Phase 5: Validate Fuzzy Uncertainty & Triage Policy
    # First, run validation stream to calibrate cutoffs without test leakage
    print("\nRunning Validation Prequential Stream for Triage Calibration...")
    sim_val = StreamingSimulator(
        model=guided_fuzzy_model,
        scaler=scaler,
        threshold=fuzzy_val_threshold,
        adaptive=False,
    )
    val_preds_df, _ = sim_val.run_stream(val_df, feature_cols)
    val_preds_path = metrics_dir / "streaming_predictions_guided_fuzzy_val.parquet"
    val_preds_df.to_parquet(val_preds_path, index=False)
    print(f"Saved validation predictions: {val_preds_path}")

    print("\nEvaluating Uncertainty Triage Policies (Validation-Calibrated vs Fixed-Budget)...")
    guided_fuzzy_preds = predictions_cache["guided_fuzzy_frozen"]
    triage_results_df = evaluate_triage_policy(
        guided_fuzzy_preds,
        val_predictions_df=val_preds_df,
        budget_percentages=[0.5, 1.0, 2.0, 5.0],
        threshold=fuzzy_val_threshold,
    )

    triage_csv_path = metrics_dir / "streaming_triage_comparison.csv"
    triage_results_df.to_csv(triage_csv_path, index=False)
    print(f"Saved triage comparison: {triage_csv_path}")

    print("\nTriage Results Summary:")
    print(triage_results_df[["strategy", "protocol", "target_budget_pct", "actual_reviewed_count", "reviewed_frauds", "fraud_recall_in_review", "reviewed_cleared_fp", "oracle_auto_false_negatives", "oracle_auto_false_positives"]].to_string(index=False))


    summary_json = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "test_transactions_count": len(test_df),
        "test_fraud_count": int(test_df["Class"].sum()),
        "test_fraud_prevalence": float(test_df["Class"].sum() / len(test_df)),
        "strategies_evaluated": [s["name"] for s in strategies],
        "primary_findings": {
            "frozen_vs_adaptive_recall": {
                "guided_fuzzy_frozen": float(strategy_comparison_df.loc[strategy_comparison_df["strategy"] == "guided_fuzzy_frozen", "recall"].values[0]),
                "guided_fuzzy_adaptive_immediate": float(strategy_comparison_df.loc[strategy_comparison_df["strategy"] == "guided_fuzzy_adaptive_immediate", "recall"].values[0]),
                "guided_fuzzy_adaptive_delayed_1h": float(strategy_comparison_df.loc[strategy_comparison_df["strategy"] == "guided_fuzzy_adaptive_delayed_1h", "recall"].values[0]),
            },
            "frozen_vs_adaptive_f1": {
                "guided_fuzzy_frozen": float(strategy_comparison_df.loc[strategy_comparison_df["strategy"] == "guided_fuzzy_frozen", "f1"].values[0]),
                "guided_fuzzy_adaptive_immediate": float(strategy_comparison_df.loc[strategy_comparison_df["strategy"] == "guided_fuzzy_adaptive_immediate", "f1"].values[0]),
                "guided_fuzzy_adaptive_delayed_1h": float(strategy_comparison_df.loc[strategy_comparison_df["strategy"] == "guided_fuzzy_adaptive_delayed_1h", "f1"].values[0]),
            },
        }
    }

    summary_json_path = metrics_dir / "streaming_experiments_summary.json"
    with open(summary_json_path, "w") as f:
        json.dump(summary_json, f, indent=2)
    print(f"Saved experiment summary: {summary_json_path}")
    print("\nMilestone 4 Streaming Experiment Run Successfully Completed.")


if __name__ == "__main__":
    main()
