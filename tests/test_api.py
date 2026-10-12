"""Unit and integration tests for FastAPI research dashboard endpoints (Milestone 5)."""
from __future__ import annotations
import pytest
from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)


def test_api_health():
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["artifacts_loaded"]["guided_fuzzy_model"] is True
    assert data["artifacts_loaded"]["streaming_metrics"] is True


def test_api_summary():
    res = client.get("/api/summary")
    assert res.status_code == 200
    data = res.json()
    assert data["project_name"] == "FuzzyGuard AI"
    assert data["dataset"]["total_rows"] == 284807
    assert data["dataset"]["total_frauds"] == 492
    assert "partitions" in data
    assert data["partitions"]["test"]["rows"] == 56962


def test_api_model_comparison():
    res = client.get("/api/models/comparison")
    assert res.status_code == 200
    data = res.json()
    assert data["count"] >= 10
    # Must distinguish static held out from prequential streaming
    eval_types = {item["evaluation_type"] for item in data["data"]}
    assert "static_held_out" in eval_types
    assert "prequential_streaming" in eval_types

    # Test filtering by evaluation_type
    res_static = client.get("/api/models/comparison?evaluation_type=static_held_out")
    assert res_static.status_code == 200
    for item in res_static.json()["data"]:
        assert item["evaluation_type"] == "static_held_out"


def test_api_streaming_windows():
    res = client.get("/api/streaming/windows")
    assert res.status_code == 200
    data = res.json()
    assert data["count"] >= 16
    first_window = data["data"][0]
    assert "window_id" in first_window
    assert "recall" in first_window
    assert "fraud_count" in first_window


def test_api_streaming_triage():
    res = client.get("/api/streaming/triage")
    assert res.status_code == 200
    data = res.json()
    assert data["count"] >= 8
    assert "oracle_disclosure" in data
    first_item = data["data"][0]
    assert "reviewed_frauds" in first_item
    assert "reviewed_prior_tp" in first_item
    assert "oracle_auto_false_positives" in first_item


def test_api_transactions_pagination_and_operational_masking():
    # 1. Research mode exposes true_label
    res_research = client.get("/api/transactions?page=1&page_size=10&view_mode=offline_research_evaluation")
    assert res_research.status_code == 200
    data_res = res_research.json()
    assert len(data_res["data"]) == 10
    assert data_res["total_matched"] == 56962
    for item in data_res["data"]:
        assert item["true_label"] in (0, 1)
        assert 0.0 <= item["expected_value"] <= 1.0

    # 2. Operational mode strictly masks true_label
    res_op = client.get("/api/transactions?page=1&page_size=10&view_mode=operational_simulation")
    assert res_op.status_code == 200
    data_op = res_op.json()
    for item in data_op["data"]:
        assert item["true_label"] is None


def test_api_transaction_explanation():
    res = client.get("/api/transactions/0")
    assert res.status_code == 200
    data = res.json()
    assert data["transaction_index"] == 0
    assert "fuzzy_bounds" in data
    assert data["fuzzy_bounds"]["p_left"] <= data["fuzzy_bounds"]["p_center"] <= data["fuzzy_bounds"]["p_right"]
    assert "top_features" in data
    assert len(data["top_features"]) == 10


def test_api_predict_custom():
    payload = {
        "Time": 150000.0,
        "Amount": 199.99,
        "V1": -2.0,
        "V4": 3.5,
        "V14": -4.0,
        "is_synthetic": True
    }
    res = client.post("/api/predict", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["is_synthetic"] is True
    assert 0.0 <= data["expected_value"] <= 1.0
    assert data["p_left"] <= data["p_center"] <= data["p_right"]
    assert data["verdict"] in ("FRAUD ALERT", "APPROVED")


def test_api_coefficients_and_methodology():
    res_coef = client.get("/api/coefficients")
    assert res_coef.status_code == 200
    assert res_coef.json()["count"] == 30

    res_meth = client.get("/api/methodology")
    assert res_meth.status_code == 200
    data_meth = res_meth.json()
    assert "source_paper" in data_meth
    assert "scientific_limitations" in data_meth
    assert len(data_meth["scientific_limitations"]) >= 4
