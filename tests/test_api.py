import io
import pytest
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["engine_version"] == "3.0.0"


def test_tasks_endpoint():
    response = client.get("/api/v1/tasks")
    assert response.status_code == 200
    tasks = response.json()["tasks"]
    assert len(tasks) == 4
    task_ids = [t["task_id"] for t in tasks]
    assert "supervised_classification" in task_ids
    assert "time_series_forecasting" in task_ids


def test_demo_endpoint():
    response = client.get("/api/v1/demo/clean?task=supervised_classification")
    assert response.status_code == 200
    data = response.json()
    assert "trust_score" in data
    assert data["trust_score"] >= 80.0
    assert data["confidence_tier"] in ["High Fitness / Low Risk", "Moderate Fitness / Conditional"]


def test_evaluate_csv_upload():
    # Create simple in-memory CSV
    df = pd.DataFrame({
        "feature1": np.random.randn(50),
        "feature2": np.random.randn(50),
        "target": np.random.choice([0, 1], 50)
    })
    csv_bytes = df.to_csv(index=False).encode('utf-8')

    response = client.post(
        "/api/v1/evaluate",
        files={"file": ("test_data.csv", csv_bytes, "text/csv")},
        data={"task": "supervised_classification", "target_column": "target"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "trust_score" in data
    assert data["ahp_consistency_ratio"] < 0.10
    assert "missingness" in data["dimensions"] or "completeness" in data["dimensions"]


def test_auto_detect_endpoint():
    df = pd.DataFrame({
        "feature1": np.random.randn(50),
        "feature2": np.random.randn(50),
        "target": np.random.choice([0, 1], 50)
    })
    csv_bytes = df.to_csv(index=False).encode('utf-8')

    response = client.post(
        "/api/v1/auto-detect",
        files={"file": ("test_detect.csv", csv_bytes, "text/csv")},
        data={"target_column": "target"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "inferred_task" in data
    assert data["inferred_task"] == "supervised_classification"
    assert data["confidence"] > 0.5


def test_remediate_and_validate_endpoint():
    # Create dataset with missing values, extreme outlier, and target
    np.random.seed(42)
    n = 60
    df = pd.DataFrame({
        "feature_a": np.random.randn(n),
        "feature_b": np.random.randn(n),
        "target": np.random.randn(n) * 10 + 50
    })
    # Inject defects: missing values and extreme outlier
    df.loc[0:4, "feature_a"] = np.nan
    df.loc[10, "feature_b"] = 1e6

    csv_bytes = df.to_csv(index=False).encode('utf-8')

    response = client.post(
        "/api/v1/remediate-and-validate",
        files={"file": ("test_remed.csv", csv_bytes, "text/csv")},
        data={"task": "supervised_regression", "target_column": "target"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "report_before" in data
    assert "report_after" in data
    assert "validation" in data
    val = data["validation"]
    assert val["fitness_after"] >= val["fitness_before"]
    assert val["metric_after"] is not None


def test_sensitivity_matrix_endpoint():
    response = client.get("/api/v1/sensitivity-matrix")
    assert response.status_code == 200
    data = response.json()
    assert "matrix" in data
    assert "supervised_regression" in data["matrix"]
    assert "supervised_classification" in data["matrix"]
    assert len(data["dimensions"]) == 8


def test_reproducibility_receipt_endpoint():
    # Evaluate first to get report
    df = pd.DataFrame({
        "feature1": np.random.randn(50),
        "target": np.random.choice([0, 1], 50)
    })
    csv_bytes = df.to_csv(index=False).encode('utf-8')
    resp = client.post(
        "/api/v1/evaluate",
        files={"file": ("test_receipt.csv", csv_bytes, "text/csv")},
        data={"task": "supervised_classification", "target_column": "target"}
    )
    assert resp.status_code == 200
    report_dict = resp.json()

    receipt_resp = client.post("/api/v1/reproducibility-receipt", json=report_dict)
    assert receipt_resp.status_code == 200
    receipt = receipt_resp.json()
    assert "dataset_hash" in receipt
    assert "evaluation_id" in receipt
    assert "timestamp" in receipt
    assert receipt["task"] == "supervised_classification"


def test_scenario_presets_endpoints():
    # Test validate-downstream with scenario
    resp_val = client.post("/api/v1/validate-downstream", data={"scenario": "clean"})
    assert resp_val.status_code == 200
    assert "metric_name" in resp_val.json()

    # Test remediate-and-validate with scenario
    resp_rem = client.post("/api/v1/remediate-and-validate", data={"scenario": "extreme_outliers"})
    assert resp_rem.status_code == 200
    assert "validation" in resp_rem.json()
    assert resp_rem.json()["validation"]["metric_name"] == "RMSE"

    # Test PDF report generation with scenario
    resp_pdf = client.post("/api/v1/report/pdf", data={"scenario": "temporal_leakage"})
    assert resp_pdf.status_code == 200
    assert resp_pdf.headers["content-type"] == "application/pdf"
    assert len(resp_pdf.content) > 1000


