import pytest
import pandas as pd
import numpy as np
from datatrust.task_infer import TaskInferenceEngine
from datatrust.models import TaskType


def test_infer_time_series():
    n = 50
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="D"),
        "sales": np.random.normal(100, 10, n),
        "feature_x": np.random.randn(n)
    })
    res = TaskInferenceEngine.infer_task(df)
    assert res.inferred_task == TaskType.TIME_SERIES_FORECASTING
    assert res.confidence >= 0.85
    assert "timestamp" in res.candidate_timestamps


def test_infer_classification():
    n = 60
    df = pd.DataFrame({
        "feature1": np.random.randn(n),
        "feature2": np.random.randn(n),
        "churn": np.random.choice([0, 1], n)
    })
    res = TaskInferenceEngine.infer_task(df)
    assert res.inferred_task == TaskType.SUPERVISED_CLASSIFICATION
    assert res.confidence >= 0.85
    assert "churn" in res.candidate_targets


def test_infer_regression():
    n = 60
    df = pd.DataFrame({
        "area_sqft": np.random.uniform(500, 3000, n),
        "bedrooms": np.random.randint(1, 5, n),
        "price": np.random.uniform(100000, 900000, n)
    })
    res = TaskInferenceEngine.infer_task(df)
    assert res.inferred_task == TaskType.SUPERVISED_REGRESSION
    assert res.confidence >= 0.85
    assert "price" in res.candidate_targets


def test_infer_descriptive_bi():
    n = 50
    df = pd.DataFrame({
        "store_id": ["S1", "S2"] * 25,
        "region": ["North", "South"] * 25,
        "traffic": np.random.poisson(200, n)
    })
    res = TaskInferenceEngine.infer_task(df)
    assert res.inferred_task == TaskType.CLUSTERING
    # Condition 1: Clustering confidence bounded at <= 0.60 with ambiguity warning
    assert res.confidence <= 0.60
    assert res.ambiguity_warning is not None


def test_task_inference_distribution_properties():
    n = 60
    df = pd.DataFrame({
        "feature1": np.random.randn(n),
        "target": np.random.choice([0, 1], n)
    })
    res1 = TaskInferenceEngine.infer_task(df)
    res2 = TaskInferenceEngine.infer_task(df)

    # 1. Populated and covers all 4 tasks
    assert len(res1.p_task_given_data) == 4
    for t in TaskType:
        assert t.value in res1.p_task_given_data

    # 2. Non-negative probabilities
    assert all(p >= 0.0 for p in res1.p_task_given_data.values())

    # 3. Sums approximately to 1.0
    prob_sum = sum(res1.p_task_given_data.values())
    assert 0.99 <= prob_sum <= 1.01

    # 4. Deterministic under identical input
    assert res1.p_task_given_data == res2.p_task_given_data
    assert res1.inferred_task == res2.inferred_task
    assert res1.confidence == res2.confidence


def test_task_inference_manual_override():
    df = pd.DataFrame({
        "feature1": [1.0, 2.0, 3.0],
        "feature2": [4.0, 5.0, 6.0]
    })
    # Inferred as clustering, but user manually overrides to supervised_regression
    res = TaskInferenceEngine.infer_task(df, user_override_task=TaskType.SUPERVISED_REGRESSION)
    assert res.inferred_task == TaskType.CLUSTERING
    assert res.final_selected_task == TaskType.SUPERVISED_REGRESSION
    assert res.user_overrode is True


def test_infer_tabular_regression_with_datetime_feature():
    """
    Validates that a transactional tabular dataset with a datetime column and continuous
    target (e.g. sales transactions with multiple events per date across regions) is
    inferred as SUPERVISED_REGRESSION, NOT time-series forecasting.
    """
    n = 100
    dates = pd.date_range("2023-01-01", periods=10, freq="D")
    df = pd.DataFrame({
        "order_date": np.random.choice(dates, n),
        "region": np.random.choice(["East", "West", "North", "South"], n),
        "units_sold": np.random.randint(1, 50, n),
        "price": np.random.uniform(20.0, 500.0, n)
    })
    res = TaskInferenceEngine.infer_task(df)
    assert res.inferred_task == TaskType.SUPERVISED_REGRESSION
    assert res.confidence >= 0.75
    assert res.p_task_given_data[TaskType.SUPERVISED_REGRESSION.value] > res.p_task_given_data[TaskType.TIME_SERIES_FORECASTING.value]
    assert any("transactional" in ev.lower() or "collided" in ev.lower() for ev in res.positive_evidence)


def test_infer_classification_with_datetime_feature():
    """
    Validates that a transactional event dataset with a datetime column and discrete
    target (e.g. churn or fraud events) is inferred as SUPERVISED_CLASSIFICATION.
    """
    n = 100
    dates = pd.date_range("2023-01-01", periods=15, freq="D")
    df = pd.DataFrame({
        "event_time": np.random.choice(dates, n),
        "user_id": [f"user_{i%20}" for i in range(n)],
        "amount": np.random.uniform(5.0, 100.0, n),
        "is_fraud": np.random.choice([0, 1], n, p=[0.9, 0.1])
    })
    res = TaskInferenceEngine.infer_task(df)
    assert res.inferred_task == TaskType.SUPERVISED_CLASSIFICATION
    assert res.confidence >= 0.75
    assert res.p_task_given_data[TaskType.SUPERVISED_CLASSIFICATION.value] > res.p_task_given_data[TaskType.TIME_SERIES_FORECASTING.value]


def test_infer_genuine_time_series_high_confidence():
    """
    Validates that a genuine single-stream time series with 100% unique, monotonic timestamps
    is inferred as TIME_SERIES_FORECASTING with high confidence (>= 0.85).
    """
    n = 60
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=n, freq="h"),
        "sensor_reading": np.sin(np.linspace(0, 10, n)) + np.random.normal(0, 0.1, n)
    })
    res = TaskInferenceEngine.infer_task(df)
    assert res.inferred_task == TaskType.TIME_SERIES_FORECASTING
    assert res.confidence >= 0.85
    assert any("single-stream" in ev.lower() or "monotonic" in ev.lower() for ev in res.positive_evidence)


def test_datetime_presence_does_not_force_timeseries():
    """
    Direct verification of prompt constraint:
    A datetime column alone must NOT force time-series forecasting when dataset grain is transactional.
    """
    n = 80
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    df = pd.DataFrame({
        "transaction_date": np.random.choice(dates, n),
        "customer_id": np.random.randint(100, 200, n),
        "revenue": np.random.uniform(10.0, 1000.0, n)
    })
    res = TaskInferenceEngine.infer_task(df)
    assert res.inferred_task != TaskType.TIME_SERIES_FORECASTING
    assert res.p_task_given_data[TaskType.TIME_SERIES_FORECASTING.value] < 0.35


