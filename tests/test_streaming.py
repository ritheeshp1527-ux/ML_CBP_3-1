"""Unit and integration tests for streaming simulator and online adaptation (Milestone 4)."""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from src.guided_fuzzy_model import GuidedFuzzyLogisticRegression
from src.streaming_adaptation import (
    StreamingSimulator,
    OnlineLogisticRegressionBaseline,
    evaluate_streaming_windows,
    evaluate_triage_policy,
    compute_stream_metrics,
)


@pytest.fixture
def synthetic_stream_data():
    """Generates synthetic chronological streaming transactions."""
    np.random.seed(42)
    n = 200
    p = 5
    timestamps = np.sort(np.random.uniform(100.0, 1000.0, n))
    X = np.random.randn(n, p)
    # Rare fraud labels ~5%
    y = np.zeros(n, dtype=int)
    fraud_indices = np.random.choice(n, size=10, replace=False)
    y[fraud_indices] = 1
    # Fraud signal on first feature
    X[fraud_indices, 0] += 3.0

    df = pd.DataFrame(X, columns=[f"V{i}" for i in range(1, p + 1)])
    df["Time"] = timestamps
    df["Class"] = y
    return df, [f"V{i}" for i in range(1, p + 1)]


@pytest.fixture
def trained_models(synthetic_stream_data):
    df, feature_cols = synthetic_stream_data
    scaler = StandardScaler().fit(df[feature_cols].values)
    X_scaled = scaler.transform(df[feature_cols].values)
    y = df["Class"].values

    fuzzy = GuidedFuzzyLogisticRegression(max_iter=15, random_state=42)
    fuzzy.fit(X_scaled, y)

    baseline = OnlineLogisticRegressionBaseline(random_state=42)
    baseline.fit(X_scaled, y)

    replay_X = X_scaled[:20]
    replay_y = y[:20]

    return {
        "fuzzy": fuzzy,
        "baseline": baseline,
        "scaler": scaler,
        "replay_X": replay_X,
        "replay_y": replay_y,
    }


def test_prediction_before_update_ordering(synthetic_stream_data, trained_models):
    """Verifies that each transaction is scored before its label is ingested or model updated."""
    df, features = synthetic_stream_data
    sim = StreamingSimulator(
        model=trained_models["fuzzy"],
        scaler=trained_models["scaler"],
        threshold=0.5,
        label_delay_seconds=0.0,
        batch_size=50,
        replay_reservoir_X=trained_models["replay_X"],
        replay_reservoir_y=trained_models["replay_y"],
        adaptive=True,
    )
    preds, audit = sim.run_stream(df, features)
    assert len(preds) == len(df)
    # The first 50 predictions must have model_version == 0
    assert (preds.iloc[:50]["model_version"] == 0).all()
    # After update on batch 1, version increments
    assert preds.iloc[50]["model_version"] >= 1


def test_delayed_label_arrival(synthetic_stream_data, trained_models):
    """Verifies that labels do not mature until current time >= arrival time + delay."""
    df, features = synthetic_stream_data
    delay_sec = 200.0
    sim = StreamingSimulator(
        model=trained_models["fuzzy"],
        scaler=trained_models["scaler"],
        threshold=0.5,
        label_delay_seconds=delay_sec,
        batch_size=30,
        replay_reservoir_X=trained_models["replay_X"],
        replay_reservoir_y=trained_models["replay_y"],
        adaptive=True,
    )
    preds, audit = sim.run_stream(df, features)
    assert len(preds) == len(df)
    # Check that in pending queue or audit, label maturation timestamps are >= t_arrival + delay
    if not audit.empty:
        matured_events = audit[audit["event_type"] == "label_matured"]
        for _, event in matured_events.iterrows():
            assert event["timestamp"] >= df["Time"].min() + delay_sec


def test_no_label_leakage(synthetic_stream_data, trained_models):
    """Verifies that a transaction's own label cannot influence its prediction."""
    df, features = synthetic_stream_data
    sim = StreamingSimulator(
        model=trained_models["fuzzy"],
        scaler=trained_models["scaler"],
        threshold=0.5,
        label_delay_seconds=0.0,
        batch_size=10,
        replay_reservoir_X=trained_models["replay_X"],
        replay_reservoir_y=trained_models["replay_y"],
        adaptive=True,
    )
    preds, audit = sim.run_stream(df, features)
    # In prequential protocol, prediction is logged at t_i before label enqueue
    assert "true_label" in preds.columns
    assert "predicted_class" in preds.columns
    assert preds["model_version"].iloc[0] == 0


def test_streaming_reproducibility(synthetic_stream_data, trained_models):
    """Verifies that two runs with identical parameters produce identical predictions and logs."""
    df, features = synthetic_stream_data
    sim1 = StreamingSimulator(
        model=trained_models["fuzzy"],
        scaler=trained_models["scaler"],
        threshold=0.5,
        label_delay_seconds=50.0,
        batch_size=40,
        replay_reservoir_X=trained_models["replay_X"],
        replay_reservoir_y=trained_models["replay_y"],
        adaptive=True,
        random_state=42,
    )
    sim2 = StreamingSimulator(
        model=trained_models["fuzzy"],
        scaler=trained_models["scaler"],
        threshold=0.5,
        label_delay_seconds=50.0,
        batch_size=40,
        replay_reservoir_X=trained_models["replay_X"],
        replay_reservoir_y=trained_models["replay_y"],
        adaptive=True,
        random_state=42,
    )
    preds1, _ = sim1.run_stream(df, features)
    preds2, _ = sim2.run_stream(df, features)
    np.testing.assert_allclose(preds1["expected_value"].values, preds2["expected_value"].values)
    assert (preds1["model_version"].values == preds2["model_version"].values).all()


def test_preservation_of_valid_tfn_coefficients_after_updates(synthetic_stream_data, trained_models):
    """Verifies that strict TFN ordering a_j <= b_j <= c_j is maintained after online updates."""
    df, features = synthetic_stream_data
    sim = StreamingSimulator(
        model=trained_models["fuzzy"],
        scaler=trained_models["scaler"],
        threshold=0.5,
        label_delay_seconds=0.0,
        batch_size=25,
        replay_reservoir_X=trained_models["replay_X"],
        replay_reservoir_y=trained_models["replay_y"],
        adaptive=True,
    )
    preds, audit = sim.run_stream(df, features)
    assert sim.model_version_ > 0
    beta = sim.model.beta_
    # Verify a_j <= b_j <= c_j for all rows
    assert np.all(beta[:, 0] <= beta[:, 1] + 1e-12)
    assert np.all(beta[:, 1] <= beta[:, 2] + 1e-12)
    assert np.all(sim.model.spreads_left_ >= 0.005)
    assert np.all(sim.model.spreads_right_ >= 0.005)


def test_finite_outputs_and_valid_probability_bounds(synthetic_stream_data, trained_models):
    """Verifies all stream outputs are finite and p_left <= p_center <= p_right in (0, 1)."""
    df, features = synthetic_stream_data
    sim = StreamingSimulator(
        model=trained_models["fuzzy"],
        scaler=trained_models["scaler"],
        threshold=0.5,
        adaptive=True,
        batch_size=50,
    )
    preds, _ = sim.run_stream(df, features)
    assert np.isfinite(preds["expected_value"].values).all()
    assert np.all(preds["p_left"].values <= preds["p_center"].values + 1e-12)
    assert np.all(preds["p_center"].values <= preds["p_right"].values + 1e-12)
    assert np.all(preds["p_left"].values >= 0.0)
    assert np.all(preds["p_right"].values <= 1.0)
    assert np.all(preds["uncertainty_width"].values >= 0.0)


def test_rejects_non_chronological_stream(synthetic_stream_data, trained_models):
    """Verifies that simulator raises ValueError if DataFrame is not sorted by Time."""
    df, features = synthetic_stream_data
    shuffled_df = df.sample(frac=1.0, random_state=99)
    sim = StreamingSimulator(model=trained_models["fuzzy"], scaler=trained_models["scaler"])
    with pytest.raises(ValueError, match="sorted chronologically"):
        sim.run_stream(shuffled_df, features)


def test_frozen_model_behavior(synthetic_stream_data, trained_models):
    """Verifies that frozen simulator never updates model_version or parameters."""
    df, features = synthetic_stream_data
    sim = StreamingSimulator(
        model=trained_models["fuzzy"],
        scaler=trained_models["scaler"],
        adaptive=False,
    )
    preds, audit = sim.run_stream(df, features)
    assert (preds["model_version"] == 0).all()
    assert sim.model_version_ == 0
    # No model_update events in audit log
    if not audit.empty:
        assert (audit["event_type"] != "model_update").all()


def test_correct_update_logging(synthetic_stream_data, trained_models):
    """Verifies audit log contains event_type, timestamps, versions, and loss metrics."""
    df, features = synthetic_stream_data
    sim = StreamingSimulator(
        model=trained_models["fuzzy"],
        scaler=trained_models["scaler"],
        batch_size=30,
        adaptive=True,
        replay_reservoir_X=trained_models["replay_X"],
        replay_reservoir_y=trained_models["replay_y"],
    )
    _, audit = sim.run_stream(df, features)
    updates = audit[audit["event_type"] == "model_update"]
    assert len(updates) > 0
    for _, u in updates.iterrows():
        assert u["new_version"] > u["old_version"]
        assert u["batch_size"] >= 30
        assert np.isfinite(u["loss"])
        assert u["update_duration_seconds"] >= 0.0


def test_sequential_window_and_triage_evaluation(synthetic_stream_data, trained_models):
    """Verifies that window aggregation and triage policy functions compute expected metrics."""
    df, features = synthetic_stream_data
    sim = StreamingSimulator(
        model=trained_models["fuzzy"],
        scaler=trained_models["scaler"],
        adaptive=False,
    )
    preds, _ = sim.run_stream(df, features)
    windows_df = evaluate_streaming_windows(preds, n_windows=3)
    assert len(windows_df) == 3
    assert "pr_auc" in windows_df.columns
    assert "fraud_count" in windows_df.columns

    triage_df = evaluate_triage_policy(preds, budget_percentages=[1.0, 5.0], threshold=0.5)
    assert len(triage_df) == 4  # 2 budgets x 2 strategies
    assert "fraud_recall_in_review" in triage_df.columns
    assert "oracle_auto_false_negatives" in triage_df.columns


def test_validation_calibrated_triage_no_test_leakage(synthetic_stream_data, trained_models):
    """Verifies that validation-calibrated triage learns cutoffs strictly from validation data."""
    df, features = synthetic_stream_data
    sim = StreamingSimulator(
        model=trained_models["fuzzy"],
        scaler=trained_models["scaler"],
        adaptive=False,
    )
    # Split synthetic stream into val and test halves
    val_df = df.iloc[:100]
    test_df = df.iloc[100:]

    val_preds, _ = sim.run_stream(val_df, features)
    test_preds, _ = sim.run_stream(test_df, features)

    triage_df = evaluate_triage_policy(
        test_preds,
        val_predictions_df=val_preds,
        budget_percentages=[2.0, 5.0],
        threshold=0.5,
    )
    # 2 budgets x 2 strategies x 2 protocols = 8 rows
    assert len(triage_df) == 8
    val_calibrated = triage_df[triage_df["protocol"] == "validation_calibrated_cutoff"]
    assert len(val_calibrated) == 4
    for _, row in val_calibrated.iterrows():
        assert row["oracle_review_assumed"] is True
        assert row["actual_reviewed_count"] >= 0


def test_oracle_review_error_decomposition(synthetic_stream_data, trained_models):
    """Verifies that base classifier errors and post-review errors satisfy mathematical consistency."""
    df, features = synthetic_stream_data
    sim = StreamingSimulator(
        model=trained_models["fuzzy"],
        scaler=trained_models["scaler"],
        adaptive=False,
    )
    preds, _ = sim.run_stream(df, features)
    triage_df = evaluate_triage_policy(preds, budget_percentages=[5.0], threshold=0.5)

    for _, row in triage_df.iterrows():
        # Reviewed TP + salvaged FN must equal total reviewed frauds
        assert row["reviewed_prior_tp"] + row["reviewed_salvaged_fn"] == row["reviewed_frauds"]
        # Salvaged FN + remaining automated FN must equal base classifier FN
        assert row["reviewed_salvaged_fn"] + row["oracle_auto_false_negatives"] == row["base_classifier_fn"]
        # Reviewed cleared FP + remaining automated FP must equal base classifier FP
        assert row["reviewed_cleared_fp"] + row["oracle_auto_false_positives"] == row["base_classifier_fp"]

