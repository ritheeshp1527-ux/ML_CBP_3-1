"""FastAPI Backend Server for FuzzyGuard AI Research Dashboard.

Provides read-only REST endpoints serving:
- Dataset summary and experiment metadata
- Offline & prequential streaming model comparisons
- Chronological concept drift across sequential time windows
- Human review triage evaluation across review budgets
- Searchable, paginated transaction predictions from stream artifacts
- Interactive single-transaction fuzzy explanation and feature attribution
- Custom transaction scoring using the trained GuidedFuzzyLogisticRegression model
- Research methodology notes and scientific limitations
"""
from __future__ import annotations
import json
import math
from pathlib import Path
from typing import Any, Literal
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.data import load_and_validate, chronological_split
from src.guided_fuzzy_model import GuidedFuzzyLogisticRegression

app = FastAPI(
    title="FuzzyGuard AI — Research API",
    description="Online Fuzzy Credit Card Fraud Detection Research Dashboard API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent.parent
METRICS_DIR = BASE_DIR / "artifacts" / "metrics"
MODELS_DIR = BASE_DIR / "artifacts" / "models"
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "creditcard.csv"

# Global lazy cache
_MODEL_CACHE: dict[str, Any] = {}
_DF_CACHE: dict[str, Any] = {}


def get_guided_fuzzy_model_artifacts() -> tuple[GuidedFuzzyLogisticRegression, Any, float]:
    """Loads and caches the primary guided fuzzy model, scaler, and validation threshold."""
    if "guided_fuzzy" not in _MODEL_CACHE:
        model_path = MODELS_DIR / "guided_fuzzy_logistic_regression.joblib"
        if not model_path.exists():
            raise HTTPException(status_code=500, detail="Guided fuzzy model artifact not found.")
        artifact = joblib.load(model_path)
        _MODEL_CACHE["guided_fuzzy"] = (artifact["model"], artifact["scaler"], float(artifact["threshold"]))
    return _MODEL_CACHE["guided_fuzzy"]


def get_raw_dataframe() -> pd.DataFrame:
    """Loads and caches the raw credit card dataframe."""
    if "raw_df" not in _DF_CACHE:
        if not RAW_DATA_PATH.exists():
            raise HTTPException(status_code=500, detail="Raw credit card dataset not found.")
        df, _ = load_and_validate(RAW_DATA_PATH)
        _DF_CACHE["raw_df"] = df
    return _DF_CACHE["raw_df"]


def get_predictions_df(strategy: str = "guided_fuzzy_frozen") -> pd.DataFrame:
    """Loads and caches streaming predictions parquet file."""
    cache_key = f"preds_{strategy}"
    if cache_key not in _DF_CACHE:
        parquet_path = METRICS_DIR / f"streaming_predictions_{strategy}.parquet"
        if not parquet_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Predictions artifact for strategy '{strategy}' not found at {parquet_path.name}."
            )
        df = pd.read_parquet(parquet_path)
        _DF_CACHE[cache_key] = df
    return _DF_CACHE[cache_key]


# ==============================================================================
# 1. Overview & Dataset Summary
# ==============================================================================
@app.get("/api/health")
def health_check():
    """System health check and artifact verification."""
    model_exists = (MODELS_DIR / "guided_fuzzy_logistic_regression.joblib").exists()
    comp_exists = (METRICS_DIR / "streaming_strategy_comparison.csv").exists()
    return {
        "status": "healthy",
        "artifacts_loaded": {
            "guided_fuzzy_model": model_exists,
            "streaming_metrics": comp_exists,
            "raw_dataset": RAW_DATA_PATH.exists(),
        }
    }


@app.get("/api/summary")
def get_summary():
    """Returns dataset summary, split breakdown, and primary experiment results."""
    try:
        raw_df = get_raw_dataframe()
        splits = chronological_split(raw_df)
        train_df, val_df, test_df = splits["train"], splits["validation"], splits["test"]

        # Load experiment summary json if available
        exp_summary_path = METRICS_DIR / "streaming_experiments_summary.json"
        exp_summary = {}
        if exp_summary_path.exists():
            with open(exp_summary_path) as f:
                exp_summary = json.load(f)

        return {
            "project_name": "FuzzyGuard AI",
            "subtitle": "Online Fuzzy Credit Card Fraud Detection",
            "paper_reference": "Charizanos, Demirhan, & Icen (2024), Expert Systems with Applications",
            "simulation_notice": "Historical Simulation using Chronological Data — Not Live Banking Activity",
            "dataset": {
                "total_rows": int(len(raw_df)),
                "total_columns": int(raw_df.shape[1]),
                "total_frauds": int(raw_df["Class"].sum()),
                "total_legitimate": int(len(raw_df) - raw_df["Class"].sum()),
                "overall_fraud_rate": float(raw_df["Class"].sum() / len(raw_df)),
                "time_span_seconds": float(raw_df["Time"].max() - raw_df["Time"].min()),
                "time_span_hours": float((raw_df["Time"].max() - raw_df["Time"].min()) / 3600.0),
            },
            "partitions": {
                "train": {
                    "rows": int(len(train_df)),
                    "frauds": int(train_df["Class"].sum()),
                    "fraud_rate": float(train_df["Class"].sum() / len(train_df)),
                    "time_start_sec": float(train_df["Time"].min()),
                    "time_end_sec": float(train_df["Time"].max()),
                },
                "validation": {
                    "rows": int(len(val_df)),
                    "frauds": int(val_df["Class"].sum()),
                    "fraud_rate": float(val_df["Class"].sum() / len(val_df)),
                    "time_start_sec": float(val_df["Time"].min()),
                    "time_end_sec": float(val_df["Time"].max()),
                },
                "test": {
                    "rows": int(len(test_df)),
                    "frauds": int(test_df["Class"].sum()),
                    "fraud_rate": float(test_df["Class"].sum() / len(test_df)),
                    "time_start_sec": float(test_df["Time"].min()),
                    "time_end_sec": float(test_df["Time"].max()),
                    "evaluation_period_hours": float((test_df["Time"].max() - test_df["Time"].min()) / 3600.0),
                },
            },
            "primary_experiment": exp_summary,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating summary: {str(e)}")


# ==============================================================================
# 2. Model Comparisons
# ==============================================================================
@app.get("/api/models/comparison")
def get_model_comparison(evaluation_type: str = Query("all", pattern="^(all|static_held_out|prequential_streaming)$")):
    """
    Returns consolidated model comparisons, clearly separating static held-out
    evaluation from prequential streaming adaptation.
    """
    records = []

    # 1. Offline models from model_comparison.csv
    offline_csv = METRICS_DIR / "model_comparison.csv"
    if offline_csv.exists() and evaluation_type in ("all", "static_held_out"):
        df_off = pd.read_csv(offline_csv)
        # Select test split rows
        test_rows = df_off[df_off["split"] == "test"]
        for _, r in test_rows.iterrows():
            records.append({
                "model_key": str(r["model"]),
                "display_name": str(r["model"]).replace("_", " ").title(),
                "evaluation_type": "static_held_out",
                "evaluation_split": "test",
                "threshold": float(r["threshold"]),
                "threshold_mode": str(r["threshold_mode"]),
                "precision": float(r["precision"]),
                "recall": float(r["recall"]),
                "specificity": float(r["specificity"]),
                "f1": float(r["f1"]),
                "pr_auc": float(r["pr_auc"]),
                "roc_auc": float(r["roc_auc"]),
                "mcc": float(r["mcc"]),
                "tn": int(r["tn"]),
                "fp": int(r["fp"]),
                "fn": int(r["fn"]),
                "tp": int(r["tp"]),
                "fit_seconds": float(r["fit_seconds"]) if pd.notna(r.get("fit_seconds")) else 0.0,
                "latency_ms": float(r["latency_ms_per_row"]) if pd.notna(r.get("latency_ms_per_row")) else 0.0,
                "f1_ci95_low": float(r["f1_ci95_low"]) if pd.notna(r.get("f1_ci95_low")) else None,
                "f1_ci95_high": float(r["f1_ci95_high"]) if pd.notna(r.get("f1_ci95_high")) else None,
            })

    # 2. Streaming strategies from streaming_strategy_comparison.csv
    streaming_csv = METRICS_DIR / "streaming_strategy_comparison.csv"
    if streaming_csv.exists() and evaluation_type in ("all", "prequential_streaming"):
        df_stream = pd.read_csv(streaming_csv)
        for _, r in df_stream.iterrows():
            is_adaptive = bool(r["adaptive"])
            eval_type = "prequential_streaming" if is_adaptive else "static_held_out"
            records.append({
                "model_key": str(r["strategy"]),
                "display_name": str(r["description"]),
                "evaluation_type": eval_type,
                "evaluation_split": "test_stream",
                "threshold": float(r["threshold"]),
                "threshold_mode": "val_selected_threshold",
                "precision": float(r["precision"]),
                "recall": float(r["recall"]),
                "specificity": float(r["specificity"]),
                "f1": float(r["f1"]),
                "pr_auc": float(r["pr_auc"]),
                "roc_auc": float(r["roc_auc"]),
                "mcc": float(r["mcc"]),
                "tn": int(r["tn"]),
                "fp": int(r["fp"]),
                "fn": int(r["fn"]),
                "tp": int(r["tp"]),
                "fit_seconds": float(r["adaptation_time_seconds"]) if pd.notna(r.get("adaptation_time_seconds")) else 0.0,
                "latency_ms": float(r["avg_inference_latency_ms"]) if pd.notna(r.get("avg_inference_latency_ms")) else 0.0,
                "model_updates_count": int(r["model_updates_count"]) if pd.notna(r.get("model_updates_count")) else 0,
                "label_delay_seconds": float(r["label_delay_seconds"]) if pd.notna(r.get("label_delay_seconds")) else 0.0,
            })

    return {"count": len(records), "data": records}


# ==============================================================================
# 3. Streaming Windows & Concept Drift
# ==============================================================================
@app.get("/api/streaming/windows")
def get_streaming_windows(strategy: str = Query("all")):
    """Returns chronological window drift metrics across the test timeline."""
    windows_csv = METRICS_DIR / "streaming_windows_drift.csv"
    if not windows_csv.exists():
        raise HTTPException(status_code=404, detail="Streaming windows drift artifact not found.")

    df = pd.read_csv(windows_csv)
    if strategy != "all":
        df = df[df["strategy"] == strategy]

    records = []
    for _, r in df.iterrows():
        records.append({
            "strategy": str(r["strategy"]),
            "window_id": int(r["window_id"]),
            "start_timestamp": float(r["start_timestamp"]),
            "end_timestamp": float(r["end_timestamp"]),
            "duration_hours": float(r["duration_hours"]),
            "fraud_count": int(r["fraud_count"]),
            "total_rows": int(r["total_rows"]),
            "fraud_prevalence": float(r["fraud_prevalence"]),
            "recall": float(r["recall"]),
            "precision": float(r["precision"]),
            "specificity": float(r["specificity"]),
            "f1": float(r["f1"]),
            "pr_auc": float(r["pr_auc"]) if pd.notna(r["pr_auc"]) else None,
            "roc_auc": float(r["roc_auc"]) if pd.notna(r["roc_auc"]) else None,
            "mcc": float(r["mcc"]),
            "fp": int(r["fp"]),
            "fn": int(r["fn"]),
            "tp": int(r["tp"]),
            "tn": int(r["tn"]),
            "model_version_min": int(r["model_version_min"]),
            "model_version_max": int(r["model_version_max"]),
        })

    return {"count": len(records), "data": records}


# ==============================================================================
# 4. Human-Review Triage Policy
# ==============================================================================
@app.get("/api/streaming/triage")
def get_triage_results(protocol: str = Query("all", pattern="^(all|validation_calibrated_cutoff|stream_rank_cutoff)$")):
    """Returns human-review triage trade-offs across review budgets."""
    triage_csv = METRICS_DIR / "streaming_triage_comparison.csv"
    if not triage_csv.exists():
        raise HTTPException(status_code=404, detail="Triage comparison artifact not found.")

    df = pd.read_csv(triage_csv)
    if protocol != "all":
        df = df[df["protocol"] == protocol]

    records = []
    for _, r in df.iterrows():
        records.append({
            "strategy": str(r["strategy"]),
            "protocol": str(r["protocol"]),
            "target_budget_pct": float(r["target_budget_pct"]),
            "actual_reviewed_count": int(r["actual_reviewed_count"]),
            "actual_reviewed_pct": float(r["actual_reviewed_pct"]),
            "base_classifier_tp": int(r["base_classifier_tp"]),
            "base_classifier_fp": int(r["base_classifier_fp"]),
            "base_classifier_fn": int(r["base_classifier_fn"]),
            "reviewed_frauds": int(r["reviewed_frauds"]),
            "fraud_recall_in_review": float(r["fraud_recall_in_review"]),
            "reviewed_prior_tp": int(r["reviewed_prior_tp"]),
            "reviewed_salvaged_fn": int(r["reviewed_salvaged_fn"]),
            "reviewed_cleared_fp": int(r["reviewed_cleared_fp"]),
            "reviewed_inspected_tn": int(r["reviewed_inspected_tn"]),
            "oracle_auto_false_negatives": int(r["oracle_auto_false_negatives"]),
            "oracle_auto_false_positives": int(r["oracle_auto_false_positives"]),
            "oracle_auto_error_rate_pct": float(r["oracle_auto_error_rate_pct"]),
            "oracle_review_assumed": bool(r["oracle_review_assumed"]),
        })

    return {
        "count": len(records),
        "oracle_disclosure": "Results assume an oracle human investigator who correctly resolves 100% of reviewed transactions. This is a decision-support simulation, not empirical proof of live human investigator accuracy.",
        "data": records,
    }


# ==============================================================================
# 5. Transaction Explorer & Search
# ==============================================================================
@app.get("/api/transactions")
def get_transactions(
    strategy: str = Query("guided_fuzzy_frozen"),
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=100),
    view_mode: str = Query("offline_research_evaluation", pattern="^(offline_research_evaluation|operational_simulation)$"),
    filter_pred: str = Query("all", pattern="^(all|fraud|legitimate)$"),
    filter_actual: str = Query("all", pattern="^(all|fraud|legitimate)$"),
    min_prob: float = Query(0.0, ge=0.0, le=1.0),
    max_prob: float = Query(1.0, ge=0.0, le=1.0),
    min_width: float = Query(0.0, ge=0.0),
    search_id: str | None = None,
):
    """
    Searchable, filtered, paginated transactions from prequential stream artifacts.
    Enforces operational masking of true labels when view_mode is 'operational_simulation'.
    """
    df = get_predictions_df(strategy)

    # Filter by probability range
    mask = (df["expected_value"] >= min_prob) & (df["expected_value"] <= max_prob)

    # Filter by uncertainty width
    if min_width > 0.0:
        mask &= (df["uncertainty_width"] >= min_width)

    # Filter by predicted class
    if filter_pred == "fraud":
        mask &= (df["predicted_class"] == 1)
    elif filter_pred == "legitimate":
        mask &= (df["predicted_class"] == 0)

    # Filter by actual class (only allowed in offline_research_evaluation mode)
    if view_mode == "offline_research_evaluation" and filter_actual != "all":
        target_val = 1 if filter_actual == "fraud" else 0
        mask &= (df["true_label"] == target_val)

    filtered_df = df[mask]

    # Search by index id
    if search_id is not None and search_id.strip():
        try:
            target_id = int(search_id.strip())
            filtered_df = filtered_df[filtered_df["index"] == target_id]
        except ValueError:
            filtered_df = filtered_df.iloc[0:0]

    total_matched = len(filtered_df)
    total_pages = max(1, math.ceil(total_matched / page_size))
    start_idx = (page - 1) * page_size
    end_idx = min(start_idx + page_size, total_matched)

    paged_rows = filtered_df.iloc[start_idx:end_idx]

    # Triage routing threshold from 2% budget validation cutoff (margin <= 0.24398)
    margin_triage_cutoff = 0.24398
    fuzzy_width_triage_cutoff = 0.05764

    items = []
    for _, r in paged_rows.iterrows():
        ev = float(r["expected_value"])
        margin_dist = abs(ev - 0.90)
        routed_margin = bool(margin_dist <= margin_triage_cutoff)
        routed_width = bool(r["uncertainty_width"] >= fuzzy_width_triage_cutoff)

        items.append({
            "index": int(r["index"]),
            "timestamp": float(r["timestamp"]),
            "expected_value": ev,
            "p_left": float(r["p_left"]),
            "p_center": float(r["p_center"]),
            "p_right": float(r["p_right"]),
            "uncertainty_width": float(r["uncertainty_width"]),
            "margin_distance": float(margin_dist),
            "predicted_class": int(r["predicted_class"]),
            "true_label": int(r["true_label"]) if view_mode == "offline_research_evaluation" else None,
            "routed_to_review_margin": routed_margin,
            "routed_to_review_width": routed_width,
            "model_version": int(r["model_version"]),
        })

    return {
        "page": page,
        "page_size": page_size,
        "total_matched": total_matched,
        "total_pages": total_pages,
        "strategy": strategy,
        "view_mode": view_mode,
        "data": items,
    }


# ==============================================================================
# 6. Single Transaction Explanation & Feature Attribution
# ==============================================================================
@app.get("/api/transactions/{transaction_index}")
def get_transaction_explanation(transaction_index: int):
    """
    Returns granular fuzzy arithmetic explanation, vertex bounds, and feature attribution
    for an individual transaction using the trained GuidedFuzzyLogisticRegression model.
    """
    raw_df = get_raw_dataframe()
    if transaction_index not in raw_df.index:
        raise HTTPException(status_code=404, detail=f"Transaction ID {transaction_index} not found in dataset.")

    row = raw_df.loc[transaction_index]
    feature_cols = [c for c in raw_df.columns if c not in ("Time", "Class")]

    model, scaler, threshold = get_guided_fuzzy_model_artifacts()

    # Scale transaction features
    X_raw = row[feature_cols].values.reshape(1, -1)
    X_scaled = scaler.transform(X_raw)

    # Fuzzy prediction
    P_tfn = model.predict_fuzzy(X_scaled)[0]
    pl, pc, pr = float(P_tfn[0]), float(P_tfn[1]), float(P_tfn[2])
    ev = float((pl + 2.0 * pc + pr) / 4.0)
    width = float(pr - pl)
    margin_dist = abs(ev - threshold)
    pred_class = int(ev >= threshold)

    # Feature contribution breakdown
    beta = model.beta_  # shape: (30, 3)
    a0, b0, c0 = beta[0]
    a_feats, b_feats, c_feats = beta[1:, 0], beta[1:, 1], beta[2:, 2]
    x_vec = X_scaled[0]

    contributions = []
    for idx, col in enumerate(feature_cols):
        xj = float(x_vec[idx])
        raw_val = float(row[col])
        bj = float(beta[idx + 1, 1])
        aj = float(beta[idx + 1, 0])
        cj = float(beta[idx + 1, 2])

        center_contrib = bj * xj
        if xj >= 0:
            left_contrib = aj * xj
            right_contrib = cj * xj
        else:
            left_contrib = cj * xj
            right_contrib = aj * xj

        contributions.append({
            "feature": col,
            "raw_value": raw_val,
            "scaled_value": xj,
            "tfn_a": aj,
            "tfn_b": bj,
            "tfn_c": cj,
            "center_contribution": center_contrib,
            "left_contribution": left_contrib,
            "right_contribution": right_contrib,
            "spread_contribution": abs(right_contrib - left_contrib),
            "is_risk_increasing": bool(center_contrib > 0),
        })

    # Sort by absolute center contribution magnitude
    contributions.sort(key=lambda item: abs(item["center_contribution"]), reverse=True)

    return {
        "transaction_index": transaction_index,
        "timestamp": float(row["Time"]),
        "amount": float(row["Amount"]),
        "true_label": int(row["Class"]),
        "threshold": threshold,
        "predicted_class": pred_class,
        "classification_label": "FRAUD" if pred_class == 1 else "LEGITIMATE",
        "expected_value": ev,
        "fuzzy_bounds": {
            "p_left": pl,
            "p_center": pc,
            "p_right": pr,
            "uncertainty_width": width,
            "margin_to_threshold": margin_dist,
        },
        "semantics": {
            "expected_value_desc": "Crisp probability defuzzified via the Dubois & Prade expected value: E(P) = (p_left + 2*p_center + p_right)/4.",
            "fuzzy_bounds_desc": "Lower and upper probability bounds resulting from coefficient spread propagation: [p_left, p_right]. Not frequentist confidence intervals.",
            "fuzzy_width_desc": "Width (p_right - p_left) reflecting model parameter ambiguity for this transaction's feature vector.",
        },
        "top_features": contributions[:10],
        "all_features": contributions,
    }


# ==============================================================================
# 7. Interactive Prediction on Custom Features
# ==============================================================================
class CustomTransactionRequest(BaseModel):
    Time: float = Field(default=150000.0, description="Timestamp in seconds")
    Amount: float = Field(default=149.99, description="Transaction amount in currency")
    V1: float = 0.0
    V2: float = 0.0
    V3: float = 0.0
    V4: float = 0.0
    V5: float = 0.0
    V6: float = 0.0
    V7: float = 0.0
    V8: float = 0.0
    V9: float = 0.0
    V10: float = 0.0
    V11: float = 0.0
    V12: float = 0.0
    V13: float = 0.0
    V14: float = 0.0
    V15: float = 0.0
    V16: float = 0.0
    V17: float = 0.0
    V18: float = 0.0
    V19: float = 0.0
    V20: float = 0.0
    V21: float = 0.0
    V22: float = 0.0
    V23: float = 0.0
    V24: float = 0.0
    V25: float = 0.0
    V26: float = 0.0
    V27: float = 0.0
    V28: float = 0.0
    custom_threshold: float | None = None
    is_synthetic: bool = True


@app.post("/api/predict")
def predict_custom_transaction(req: CustomTransactionRequest):
    """
    Scores a custom or synthetic transaction using the trained GuidedFuzzyLogisticRegression model.
    """
    model, scaler, default_threshold = get_guided_fuzzy_model_artifacts()
    threshold = req.custom_threshold if req.custom_threshold is not None else default_threshold

    feature_cols = [f"V{i}" for i in range(1, 29)] + ["Amount"]
    features_dict = req.model_dump()
    X_raw = np.array([[features_dict[col] for col in feature_cols]])

    X_scaled = scaler.transform(X_raw)
    P_tfn = model.predict_fuzzy(X_scaled)[0]

    pl, pc, pr = float(P_tfn[0]), float(P_tfn[1]), float(P_tfn[2])
    ev = float((pl + 2.0 * pc + pr) / 4.0)
    width = float(pr - pl)
    margin_dist = abs(ev - threshold)
    pred_class = int(ev >= threshold)

    return {
        "is_synthetic": req.is_synthetic,
        "threshold": threshold,
        "predicted_class": pred_class,
        "verdict": "FRAUD ALERT" if pred_class == 1 else "APPROVED",
        "expected_value": ev,
        "p_left": pl,
        "p_center": pc,
        "p_right": pr,
        "uncertainty_width": width,
        "margin_distance": margin_dist,
        "routed_to_review": bool(margin_dist <= 0.24398),
    }


# ==============================================================================
# 8. Learned Model Coefficients
# ==============================================================================
@app.get("/api/coefficients")
def get_coefficients():
    """Returns learned TFN coefficient parameters, spreads, and vertex asymmetry."""
    coef_csv = METRICS_DIR / "guided_fuzzy_coefficients.csv"
    if not coef_csv.exists():
        raise HTTPException(status_code=404, detail="Coefficients artifact not found.")

    df = pd.read_csv(coef_csv)
    records = df.to_dict(orient="records")
    return {"count": len(records), "data": records}


# ==============================================================================
# 9. Research Methodology & Limitations
# ==============================================================================
@app.get("/api/methodology")
def get_methodology():
    """Returns research methodology notes, audit conclusions, and limitations."""
    return {
        "source_paper": {
            "title": "An Online Fuzzy Fraud Detection Framework for Credit Card Transactions",
            "authors": "Charizanos, G., Demirhan, H., & Icen, D. (2024)",
            "journal": "Expert Systems with Applications, 252, 124127",
            "discrepancy_summary": "Author Table 5 coefficients inverted signs on decisive fraud features due to pairwise stratified SRS blending. Pure Monte Carlo random search yielded ROC-AUC 0.69 and PR-AUC 0.03.",
        },
        "guided_learning_formulation": {
            "parameterization": "Triangular Fuzzy Numbers A_j = (a_j, b_j, c_j) with a_j = b_j - s_l, c_j = b_j + s_r (s >= 0.005) guaranteeing strict ordering.",
            "initialization": "Centers from discriminative training Logistic Regression; semi-spreads from inverse Fisher information matrix asymptotic standard errors.",
            "objective": "Regularized class-weighted binary cross-entropy on Dubois-Prade expected value E(P).",
        },
        "prequential_streaming_evaluation": {
            "protocol": "Test-Then-Train prequential evaluation where prediction strictly precedes label arrival queue and update ingestion.",
            "delay_policy": "Simulates 1-hour feedback latency (delta_t = 3600s) where dispute labels mature in FIFO queue.",
            "adaptive_results": "Online adaptation reduced false alarms by 23.6% (267 -> 204) under 1-hr delay while maintaining identical 85.33% fraud recall.",
        },
        "triage_validation_findings": {
            "oracle_review_disclosure": "Reported false-positive elimination assumes an oracle human investigator with 100% accuracy. Reviewed frauds (64/65) were prior model alerts; human review salvaged only 1-4 missed frauds.",
            "fuzzy_width_vs_margin": "Fuzzy width peaks at P=0.50 and does not serve as an effective standalone triage metric under extreme class imbalance with threshold 0.90 compared to threshold margin distance.",
        },
        "scientific_limitations": [
            "Dataset spans only 48 hours (~172,800s); macro concept drift cannot be observed in this timeframe.",
            "Simulated 1-hour label delay models automated charge dispute signals, not full 30-90 day chargeback lifecycles.",
            "Fuzzy probability bounds [p_left, p_right] are possibility intervals, not statistically calibrated frequentist confidence intervals.",
            "Streaming updates require a historical replay reservoir to prevent catastrophic forgetting under 0.13% class imbalance.",
        ]
    }
