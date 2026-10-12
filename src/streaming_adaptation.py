"""Streaming Simulator and Prequential Evaluation Module for FuzzyGuard AI.

Implements realistic online-adaptation simulation under prequential (Test-Then-Train)
evaluation and configurable label delays.

Key principles:
1. Prequential ordering: Prediction strictly precedes label availability.
2. Queue-based delay: Transaction labels mature at t_i + delay_seconds.
3. Class-imbalance stability: Incorporates a historical replay reservoir and proximal
   regularization to prevent catastrophic forgetting.
4. Complete auditability: Preserves audit logs of predictions, label maturity, and updates.
5. Explainable uncertainty: Logs fuzzy bounds [p_left, p_center, p_right] and width.

Based on:
Charizanos, G., Demirhan, H., & Icen, D. (2024).
An online fuzzy fraud detection framework for credit card transactions.
Expert Systems with Applications, 252, 124127.
"""
from __future__ import annotations
import copy
import time
from dataclasses import dataclass, field
from typing import Literal
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
from sklearn.linear_model import SGDClassifier

from .guided_fuzzy_model import GuidedFuzzyLogisticRegression


@dataclass
class PendingItem:
    available_time: float
    index: int
    x: np.ndarray
    y: int


class OnlineLogisticRegressionBaseline:
    """
    Standard online linear model baseline using SGD log-loss and partial_fit.
    Maintains identical interface for fair comparison against the guided fuzzy model.
    """
    def __init__(self, random_state: int = 42, l2_reg: float = 1e-4):
        self.random_state = random_state
        self.l2_reg = l2_reg
        self.clf_ = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=l2_reg,
            random_state=random_state,
            max_iter=1000,
        )
        self.classes_ = np.array([0, 1])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "OnlineLogisticRegressionBaseline":
        self.clf_.partial_fit(X, y, classes=self.classes_)
        return self

    def update_batch(
        self,
        X_batch: np.ndarray,
        y_batch: np.ndarray,
        replay_X: np.ndarray | None = None,
        replay_y: np.ndarray | None = None,
        proximal_weight: float = 1.0,
        max_iter: int = 10,
    ) -> float:
        if replay_X is not None and replay_y is not None and len(replay_X) > 0:
            X_comb = np.vstack([X_batch, replay_X])
            y_comb = np.concatenate([y_batch, replay_y])
        else:
            X_comb = X_batch
            y_comb = y_batch

        # Class weighting via sample weights
        n_samples = len(y_comb)
        n_pos = np.sum(y_comb == 1)
        n_neg = np.sum(y_comb == 0)
        w_pos = n_samples / (2.0 * max(n_pos, 1))
        w_neg = n_samples / (2.0 * max(n_neg, 1))
        sample_weight = np.where(y_comb == 1, w_pos, w_neg)

        for _ in range(max(1, max_iter // 2)):
            self.clf_.partial_fit(X_comb, y_comb, sample_weight=sample_weight)
        return 0.0

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.clf_.predict_proba(X)


class StreamingSimulator:
    """
    Simulates sequential prequential transaction streaming with delayed label feedback.
    """
    def __init__(
        self,
        model: GuidedFuzzyLogisticRegression | OnlineLogisticRegressionBaseline,
        scaler,
        threshold: float = 0.5,
        label_delay_seconds: float = 0.0,
        batch_size: int = 2500,
        replay_reservoir_X: np.ndarray | None = None,
        replay_reservoir_y: np.ndarray | None = None,
        adaptive: bool = True,
        proximal_weight: float = 1.0,
        max_iter: int = 10,
        random_state: int = 42,
    ):
        self.model = copy.deepcopy(model)
        self.scaler = scaler
        self.threshold = float(threshold)
        self.label_delay_seconds = float(label_delay_seconds)
        self.batch_size = int(batch_size)
        self.replay_reservoir_X = replay_reservoir_X
        self.replay_reservoir_y = replay_reservoir_y
        self.adaptive = bool(adaptive)
        self.proximal_weight = float(proximal_weight)
        self.max_iter = int(max_iter)
        self.random_state = random_state

        self.model_version_ = 0
        self.pending_queue_: list[PendingItem] = []
        self.matured_X_: list[np.ndarray] = []
        self.matured_y_: list[int] = []
        self.audit_log_: list[dict] = []
        self.total_adaptation_seconds_: float = 0.0
        self.total_inference_seconds_: float = 0.0

    def run_stream(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        time_col: str = "Time",
        target_col: str = "Class",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Executes prequential simulation over the provided DataFrame.
        """
        if not df[time_col].is_monotonic_increasing:
            raise ValueError(f"Stream DataFrame must be sorted chronologically by {time_col}.")

        timestamps = df[time_col].values
        X_raw = df[feature_cols].values
        y_true = df[target_col].values
        n_rows = len(df)

        # Scale features
        X_scaled = self.scaler.transform(X_raw)

        prediction_records = []
        self.pending_queue_.clear()
        self.matured_X_.clear()
        self.matured_y_.clear()
        self.audit_log_.clear()
        self.model_version_ = 0
        self.total_adaptation_seconds_ = 0.0
        self.total_inference_seconds_ = 0.0

        is_fuzzy = isinstance(self.model, GuidedFuzzyLogisticRegression)

        for i in range(n_rows):
            t_curr = float(timestamps[i])
            x_i = X_scaled[i:i+1]
            y_i = int(y_true[i])

            # 1. Prequential Prediction Step (using model active at t_curr)
            t_inf_start = time.perf_counter()
            if is_fuzzy:
                P_tfn = self.model.predict_fuzzy(x_i)[0]
                ev = float((P_tfn[0] + 2.0 * P_tfn[1] + P_tfn[2]) / 4.0)
                pl, pc, pr = float(P_tfn[0]), float(P_tfn[1]), float(P_tfn[2])
                width = float(pr - pl)
            else:
                prob = float(self.model.predict_proba(x_i)[0, 1])
                pl, pc, pr = prob, prob, prob
                ev = prob
                width = 0.0
            self.total_inference_seconds_ += (time.perf_counter() - t_inf_start)

            pred_class = int(ev >= self.threshold)

            prediction_records.append({
                "index": int(df.index[i]),
                "timestamp": t_curr,
                "p_left": pl,
                "p_center": pc,
                "p_right": pr,
                "expected_value": ev,
                "uncertainty_width": width,
                "predicted_class": pred_class,
                "true_label": y_i,
                "model_version": self.model_version_,
            })

            # 2. Enqueue transaction label with future availability timestamp
            avail_time = t_curr + self.label_delay_seconds
            self.pending_queue_.append(PendingItem(
                available_time=avail_time,
                index=int(df.index[i]),
                x=X_scaled[i],
                y=y_i
            ))

            # 3. Check for matured labels (available_time <= t_curr)
            # Efficient drain because pending_queue is monotonically sorted by available_time
            matured_count = 0
            while self.pending_queue_ and self.pending_queue_[0].available_time <= t_curr:
                item = self.pending_queue_.pop(0)
                self.matured_X_.append(item.x)
                self.matured_y_.append(item.y)
                matured_count += 1

            if matured_count > 0:
                self.audit_log_.append({
                    "event_type": "label_matured",
                    "timestamp": t_curr,
                    "matured_count": matured_count,
                    "total_matured_buffer": len(self.matured_X_),
                    "model_version": self.model_version_,
                })

            # 4. Trigger model update if adaptive and buffer reaches batch_size
            if self.adaptive and len(self.matured_X_) >= self.batch_size:
                X_batch = np.array(self.matured_X_)
                y_batch = np.array(self.matured_y_)
                n_batch = len(y_batch)
                n_frauds = int(np.sum(y_batch == 1))

                t_adapt_start = time.perf_counter()
                loss = self.model.update_batch(
                    X_batch=X_batch,
                    y_batch=y_batch,
                    replay_X=self.replay_reservoir_X,
                    replay_y=self.replay_reservoir_y,
                    proximal_weight=self.proximal_weight,
                    max_iter=self.max_iter,
                )
                adapt_duration = time.perf_counter() - t_adapt_start
                self.total_adaptation_seconds_ += adapt_duration

                old_version = self.model_version_
                self.model_version_ += 1

                self.audit_log_.append({
                    "event_type": "model_update",
                    "timestamp": t_curr,
                    "old_version": old_version,
                    "new_version": self.model_version_,
                    "batch_size": n_batch,
                    "frauds_in_batch": n_frauds,
                    "loss": float(loss),
                    "update_duration_seconds": float(adapt_duration),
                })

                # Clear matured buffer after consumption
                self.matured_X_.clear()
                self.matured_y_.clear()

        predictions_df = pd.DataFrame(prediction_records)
        audit_df = pd.DataFrame(self.audit_log_)
        return predictions_df, audit_df


def compute_stream_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_scores: np.ndarray) -> dict:
    """Computes comprehensive binary classification and ranking metrics."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    mcc = float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_pred)) > 1 else 0.0

    n_pos = np.sum(y_true == 1)
    if n_pos > 0 and n_pos < len(y_true):
        roc_auc = float(roc_auc_score(y_true, y_scores))
        pr_auc = float(average_precision_score(y_true, y_scores))
    else:
        roc_auc = float("nan")
        pr_auc = float("nan")

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
        "fraud_count": int(n_pos),
        "total_rows": int(len(y_true)),
        "fraud_prevalence": float(n_pos / len(y_true)),
    }


def evaluate_streaming_windows(
    predictions_df: pd.DataFrame,
    n_windows: int = 4
) -> pd.DataFrame:
    """
    Evaluates performance across n sequential chronological time windows.
    """
    n_rows = len(predictions_df)
    window_size = int(np.ceil(n_rows / n_windows))

    records = []
    for w in range(n_windows):
        start_idx = w * window_size
        end_idx = min((w + 1) * window_size, n_rows)
        if start_idx >= n_rows:
            break

        sub_df = predictions_df.iloc[start_idx:end_idx]
        y_true = sub_df["true_label"].values
        y_pred = sub_df["predicted_class"].values
        y_scores = sub_df["expected_value"].values

        m = compute_stream_metrics(y_true, y_pred, y_scores)
        m["window_id"] = w + 1
        m["start_timestamp"] = float(sub_df["timestamp"].min())
        m["end_timestamp"] = float(sub_df["timestamp"].max())
        m["duration_hours"] = float((m["end_timestamp"] - m["start_timestamp"]) / 3600.0)
        m["model_version_min"] = int(sub_df["model_version"].min())
        m["model_version_max"] = int(sub_df["model_version"].max())
        records.append(m)

    return pd.DataFrame(records)


def evaluate_triage_policy(
    predictions_df: pd.DataFrame,
    val_predictions_df: pd.DataFrame | None = None,
    budget_percentages: list[float] = [0.5, 1.0, 2.0, 5.0],
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Evaluates human review triage policies across multiple review budgets.
    
    Supports:
    1. Validation-calibrated cutoff protocol: Cutoffs are learned strictly on
       validation predictions and applied out-of-sample to test predictions.
    2. Stream-rank cutoff protocol (oracle fixed-budget simulation): Review queue
       is populated by sorting the evaluation stream directly.
       
    Reports detailed breakdown of:
    - Base classifier errors before review (TP, FP, FN, TN)
    - Review composition (Reviewed TPs, salvage FNs, dismissed FPs, inspected TNs)
    - Residual automated errors after hypothetical oracle human review
    """
    y_true = predictions_df["true_label"].values
    y_pred = predictions_df["predicted_class"].values
    ev = predictions_df["expected_value"].values
    w_fuzzy = predictions_df["uncertainty_width"].values
    margin = np.abs(ev - threshold)  # small margin = high ambiguity

    total_frauds = int(np.sum(y_true == 1))
    n_total = len(predictions_df)

    # Base classifier performance before review
    base_tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    base_fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    base_fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    base_tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    protocols_to_run = ["stream_rank_cutoff"]
    if val_predictions_df is not None:
        val_ev = val_predictions_df["expected_value"].values
        val_w = val_predictions_df["uncertainty_width"].values
        val_margin = np.abs(val_ev - threshold)
        protocols_to_run.append("validation_calibrated_cutoff")

    records = []

    for protocol in protocols_to_run:
        for budget in budget_percentages:
            q = budget / 100.0

            if protocol == "stream_rank_cutoff":
                k = int(np.round(n_total * q))
                if k <= 0:
                    continue
                # Top k by rank
                top_fuzzy_idx = np.argsort(w_fuzzy)[-k:]
                rev_fuzzy_mask = np.zeros(n_total, dtype=bool)
                rev_fuzzy_mask[top_fuzzy_idx] = True

                top_margin_idx = np.argsort(margin)[:k]
                rev_margin_mask = np.zeros(n_total, dtype=bool)
                rev_margin_mask[top_margin_idx] = True
            else:
                # Validation-calibrated cutoffs
                cutoff_w = float(np.quantile(val_w, 1.0 - q))
                cutoff_m = float(np.quantile(val_margin, q))

                rev_fuzzy_mask = (w_fuzzy >= cutoff_w)
                rev_margin_mask = (margin <= cutoff_m)

            for strat_name, mask in [("fuzzy_width_triage", rev_fuzzy_mask), ("margin_uncertainty_triage", rev_margin_mask)]:
                n_rev = int(np.sum(mask))
                frauds_rev = int(np.sum(y_true[mask] == 1))

                # Breakdown of reviewed items
                rev_tp = int(np.sum((y_true[mask] == 1) & (y_pred[mask] == 1)))
                rev_fn = int(np.sum((y_true[mask] == 1) & (y_pred[mask] == 0)))  # frauds salvaged by review
                rev_fp = int(np.sum((y_true[mask] == 0) & (y_pred[mask] == 1)))  # false alarms cleared by review
                rev_tn = int(np.sum((y_true[mask] == 0) & (y_pred[mask] == 0)))

                # Post-review automated decisions (transactions outside review)
                auto_y_true = y_true[~mask]
                auto_y_pred = y_pred[~mask]
                auto_n = len(auto_y_true)
                auto_fn = int(np.sum((auto_y_true == 1) & (auto_y_pred == 0)))
                auto_fp = int(np.sum((auto_y_true == 0) & (auto_y_pred == 1)))
                auto_err = auto_fn + auto_fp

                records.append({
                    "strategy": strat_name,
                    "protocol": protocol,
                    "target_budget_pct": budget,
                    "actual_reviewed_count": n_rev,
                    "actual_reviewed_pct": float(n_rev / n_total * 100.0),
                    "base_classifier_tp": base_tp,
                    "base_classifier_fp": base_fp,
                    "base_classifier_fn": base_fn,
                    "reviewed_frauds": frauds_rev,
                    "fraud_recall_in_review": float(frauds_rev / max(total_frauds, 1)),
                    "reviewed_prior_tp": rev_tp,
                    "reviewed_salvaged_fn": rev_fn,
                    "reviewed_cleared_fp": rev_fp,
                    "reviewed_inspected_tn": rev_tn,
                    "oracle_auto_false_negatives": auto_fn,
                    "oracle_auto_false_positives": auto_fp,
                    "oracle_auto_error_rate_pct": float(auto_err / max(auto_n, 1) * 100.0),
                    "oracle_review_assumed": True,
                })

    return pd.DataFrame(records)

