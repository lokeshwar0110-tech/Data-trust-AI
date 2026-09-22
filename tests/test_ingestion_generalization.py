import inspect
import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from backend.main import app
from datatrust.models import TaskType
from datatrust.profiler import DataProfiler
from datatrust.task_infer import TaskInferenceEngine
from datatrust.engine import DataTrustEngine

client = TestClient(app)


# ==============================================================================
# SCENARIO A: Regression with Unfamiliar Continuous Target
# ==============================================================================
def test_scenario_a_regression_unfamiliar_target_structural_numeric_outcome():
    """
    Scenario A: Regression with unfamiliar continuous target column.
    Columns: SquareFeet, Bedrooms, Age, structural_numeric_outcome.
    Must discover structural_numeric_outcome purely structurally without keyword matching.
    """
    rng = np.random.RandomState(42)
    n = 150
    df = pd.DataFrame({
        "SquareFeet": rng.uniform(500, 3500, n),
        "Bedrooms": rng.randint(1, 6, n),
        "Age": rng.randint(1, 50, n),
        "structural_numeric_outcome": rng.uniform(100000, 750000, n)
    })

    # 1. Structural target discovery
    candidate_details = DataProfiler.find_candidate_target_details(df)
    target_names = [c["column"] for c in candidate_details]
    assert "structural_numeric_outcome" in target_names
    top_candidate = candidate_details[0]
    assert top_candidate["column"] == "structural_numeric_outcome"
    assert top_candidate["target_type"] == "regression"

    # 2. Autonomous Task Inference
    infer = TaskInferenceEngine.infer_task(df)
    assert infer.inferred_task == TaskType.SUPERVISED_REGRESSION
    assert infer.detected_target == "structural_numeric_outcome"
    assert infer.confidence > 0.75
    assert sum(infer.confidence_vector.values()) == pytest.approx(1.0, abs=1e-3)

    # 3. Downstream Propagation & Model/Metric Alignment
    report = DataTrustEngine.evaluate(df=df)
    assert report.task_type == TaskType.SUPERVISED_REGRESSION
    assert report.target_column == "structural_numeric_outcome"
    assert report.model_family == "Random Forest Regressor"
    assert report.metric == "RMSE"
    assert report.observed_metric_value is not None


# ==============================================================================
# SCENARIO B: Regression with Semantic Target Keyword
# ==============================================================================
def test_scenario_b_regression_semantic_target_price():
    """
    Scenario B: Regression with semantic target keyword 'Price'.
    Columns: SquareFeet, Bedrooms, Age, Price.
    """
    rng = np.random.RandomState(42)
    n = 100
    df = pd.DataFrame({
        "SquareFeet": rng.uniform(500, 3500, n),
        "Bedrooms": rng.randint(1, 6, n),
        "Age": rng.randint(1, 50, n),
        "Price": rng.uniform(100000, 750000, n)
    })

    infer = TaskInferenceEngine.infer_task(df)
    assert infer.inferred_task == TaskType.SUPERVISED_REGRESSION
    assert infer.detected_target == "Price"
    assert "Price" in infer.candidate_targets

    report = DataTrustEngine.evaluate(df=df)
    assert report.task_type == TaskType.SUPERVISED_REGRESSION
    assert report.target_column == "Price"
    assert report.model_family == "Random Forest Regressor"
    assert report.metric == "RMSE"


# ==============================================================================
# SCENARIO C: Concrete-Style Regression (No "strength" Keyword Rule)
# ==============================================================================
def test_scenario_c_regression_concrete_style_strength_no_keyword_rule():
    """
    Scenario C: External regression dataset (Concrete Compressive Strength archetype):
    - 1,030 rows, 9 columns (8 numeric predictors, 1 continuous outcome named 'strength').
    - Verifies that 'strength' is discovered structurally, NOT through a hardcoded keyword rule.
    """
    # Verify research integrity: ensure NO hardcoded keyword rule for 'strength' exists
    source_profiler = inspect.getsource(DataProfiler.find_candidate_target_details)
    assert "strength" not in source_profiler.lower(), (
        "Research Integrity Violation: 'strength' must NOT be hardcoded in profiler rules!"
    )

    rng = np.random.RandomState(42)
    n = 1030
    df_concrete = pd.DataFrame({
        "Cement": rng.uniform(100.0, 500.0, n),
        "BlastFurnaceSlag": rng.uniform(0.0, 300.0, n),
        "FlyAsh": rng.uniform(0.0, 200.0, n),
        "Water": rng.uniform(120.0, 250.0, n),
        "Superplasticizer": rng.uniform(0.0, 30.0, n),
        "CoarseAggregate": rng.uniform(800.0, 1200.0, n),
        "FineAggregate": rng.uniform(600.0, 900.0, n),
        "Age": rng.choice([1, 3, 7, 14, 28, 56, 90, 180, 365], n),
        "strength": rng.uniform(5.0, 85.0, n)
    })

    # 1. Structural Candidate Profiling
    details = DataProfiler.find_candidate_target_details(df_concrete)
    assert len(details) > 0
    assert details[0]["column"] == "strength"
    assert details[0]["target_type"] == "regression"
    assert details[0]["score"] >= 60.0

    # 2. Autonomous Task Inference
    infer = TaskInferenceEngine.infer_task(df_concrete)
    assert infer.inferred_task == TaskType.SUPERVISED_REGRESSION
    assert infer.detected_target == "strength"
    assert infer.confidence >= 0.85
    assert sum(infer.confidence_vector.values()) == pytest.approx(1.0, abs=1e-3)
    assert len(infer.target_evidence) > 0

    # 3. Downstream Validation Alignment
    report = DataTrustEngine.evaluate(df=df_concrete)
    assert report.task_type == TaskType.SUPERVISED_REGRESSION
    assert report.target_column == "strength"
    assert report.model_family == "Random Forest Regressor"
    assert report.metric == "RMSE"
    assert report.observed_metric_value is not None


# ==============================================================================
# SCENARIO D: Classification (Features + Categorical Outcome)
# ==============================================================================
def test_scenario_d_classification_features_categorical_outcome():
    """
    Scenario D: Classification dataset with numeric measurements and categorical target.
    Iris archetype: 4 numeric feature measurements and discrete target 'Species'.
    """
    df_iris = pd.DataFrame({
        "Id": range(1, 151),
        "SepalLengthCm": [5.1, 4.9, 4.7] * 50,
        "SepalWidthCm": [3.5, 3.0, 3.2] * 50,
        "PetalLengthCm": [1.4, 1.4, 1.3] * 50,
        "PetalWidthCm": [0.2, 0.2, 0.2] * 50,
        "Species": ["Iris-setosa"] * 50 + ["Iris-versicolor"] * 50 + ["Iris-virginica"] * 50
    })

    # 1. Profiler verification: feature floats are NOT timestamps or regression targets
    prof, _ = DataProfiler.profile(df_iris)
    assert "SepalWidthCm" not in prof.candidate_timestamps
    assert "PetalWidthCm" not in prof.candidate_timestamps
    assert len(prof.candidate_timestamps) == 0
    assert "Species" in prof.candidate_targets

    # 2. Autonomous Task Inference verification
    infer = TaskInferenceEngine.infer_task(df_iris)
    assert infer.inferred_task == TaskType.SUPERVISED_CLASSIFICATION
    assert infer.confidence > 0.80
    assert infer.detected_target == "Species"

    # 3. End-to-end evaluation verification
    report = DataTrustEngine.evaluate(df=df_iris)
    assert report.task_type == TaskType.SUPERVISED_CLASSIFICATION
    assert report.target_column == "Species"
    assert report.time_column is None
    assert report.model_family == "Random Forest Classifier"
    assert report.metric == "F1"
    assert report.observed_metric_value is not None
    assert report.observed_metric_value > 0.85


# ==============================================================================
# SCENARIO E: Unlabeled Clustering (Coordinates and Feature Identifiers)
# ==============================================================================
def test_scenario_e_unlabeled_clustering_coordinates_and_features():
    """
    Scenario E: Unlabeled clustering dataset.
    Columns: coord_x, coord_y, coord_z, feature_1, feature_2.
    Must suppress coordinate and feature identifiers, produce 0 candidate targets,
    and bound clustering confidence to <= 0.60.
    """
    rng = np.random.RandomState(42)
    n = 100
    df_cluster = pd.DataFrame({
        "coord_x": rng.randn(n),
        "coord_y": rng.randn(n),
        "coord_z": rng.randn(n),
        "feature_1": rng.randn(n),
        "feature_2": rng.randn(n)
    })

    # 1. Profiler: all columns suppressed as coordinates/features
    details = DataProfiler.find_candidate_target_details(df_cluster)
    assert len(details) == 0

    # 2. Inference: Falls back to CLUSTERING with <= 0.60 confidence
    infer = TaskInferenceEngine.infer_task(df_cluster)
    assert infer.inferred_task == TaskType.CLUSTERING
    assert infer.confidence <= 0.60
    assert infer.detected_target is None
    assert len(infer.candidate_targets) == 0
    assert infer.ambiguity_warning is not None
    assert sum(infer.confidence_vector.values()) == pytest.approx(1.0, abs=1e-3)

    # 3. Downstream Evaluation: K-Means Clustering and Silhouette Score
    report = DataTrustEngine.evaluate(df=df_cluster)
    assert report.task_type == TaskType.CLUSTERING
    assert report.model_family == "K-Means Clustering"
    assert report.metric == "Silhouette Score"


# ==============================================================================
# SCENARIO F: Transactional Table with Date and Numeric Outcome
# ==============================================================================
def test_scenario_f_transactional_dataset_with_date_feature():
    """
    Scenario F: Transactional dataset with date feature and continuous outcome.
    Columns: transaction_date, store_id, item_quantity, revenue.
    Date has high collision rate -> tabular feature, not single-stream forecasting.
    Inferred as SUPERVISED_REGRESSION targeting revenue.
    """
    dates = pd.date_range("2023-01-01", periods=15, freq="D")
    rng = np.random.RandomState(42)
    n = 200
    df_txn = pd.DataFrame({
        "transaction_date": rng.choice(dates, n),
        "store_id": [f"Store_{i%5}" for i in range(n)],
        "item_quantity": rng.randint(1, 10, n),
        "revenue": rng.uniform(10, 500, n)
    })

    infer = TaskInferenceEngine.infer_task(df_txn)
    assert infer.inferred_task != TaskType.TIME_SERIES_FORECASTING
    assert infer.inferred_task == TaskType.SUPERVISED_REGRESSION
    assert infer.detected_target == "revenue"

    report = DataTrustEngine.evaluate(df=df_txn)
    assert report.veto_applied is False
    assert report.fitness_status == "NORMAL"
    assert report.raw_fitness > 80.0
    assert report.task_type == TaskType.SUPERVISED_REGRESSION
    assert report.target_column == "revenue"


# ==============================================================================
# SCENARIO G: Multiple Continuous Numeric Outcomes Ranked
# ==============================================================================
def test_scenario_g_multiple_continuous_numeric_outcomes_ranked():
    """
    Scenario G: Multiple continuous numeric outcome candidates in schema.
    Columns: sensor_reading_a, sensor_reading_b, batch_yield, product_purity.
    Discovers multiple ranked candidates with evidence, selects primary,
    preserves alternatives, and flags ambiguity when appropriate.
    """
    rng = np.random.RandomState(42)
    n = 120
    df_multi = pd.DataFrame({
        "sensor_reading_a": rng.uniform(10, 50, n),
        "sensor_reading_b": rng.uniform(100, 200, n),
        "batch_yield": rng.uniform(70.0, 99.0, n),
        "product_purity": rng.uniform(85.0, 99.9, n)
    })

    details = DataProfiler.find_candidate_target_details(df_multi)
    target_names = [d["column"] for d in details]
    # sensor readings are suppressed by sensor_ prefix
    assert "sensor_reading_a" not in target_names
    assert "sensor_reading_b" not in target_names
    # Both continuous outcomes discovered
    assert "batch_yield" in target_names
    assert "product_purity" in target_names
    assert len(details) >= 2

    # Verify each detail exposes required fields
    for d in details:
        assert "column" in d
        assert "target_type" in d
        assert "score" in d
        assert "confidence" in d
        assert "evidence" in d

    infer = TaskInferenceEngine.infer_task(df_multi)
    assert infer.inferred_task == TaskType.SUPERVISED_REGRESSION
    assert infer.detected_target in ["batch_yield", "product_purity"]
    assert len(infer.candidate_targets) >= 2
    assert sum(infer.confidence_vector.values()) == pytest.approx(1.0, abs=1e-3)


# ==============================================================================
# PRESERVED HELPER & COMPATIBILITY TESTS
# ==============================================================================
def test_iris_like_classification_dataset():
    """Compatibility alias for Scenario D."""
    test_scenario_d_classification_features_categorical_outcome()


def test_numeric_tabular_regression_dataset():
    """Compatibility alias for Scenario B."""
    test_scenario_b_regression_semantic_target_price()


def test_genuine_time_series_dataset():
    """Genuine chronological time-series dataset with single-stream temporal progression."""
    dates = pd.date_range("2023-01-01", periods=120, freq="D")
    df_ts = pd.DataFrame({
        "timestamp": dates,
        "energy_demand": 50.0 + np.sin(np.linspace(0, 20, 120)) * 15.0 + np.random.randn(120) * 2.0
    })

    infer = TaskInferenceEngine.infer_task(df_ts)
    assert infer.inferred_task == TaskType.TIME_SERIES_FORECASTING
    assert "timestamp" in infer.candidate_timestamps

    report = DataTrustEngine.evaluate(df=df_ts)
    assert report.task_type == TaskType.TIME_SERIES_FORECASTING
    assert report.time_column == "timestamp"
    assert report.model_family == "Autoregressive / Lagged Regression"
    assert report.metric == "RMSE"


def test_transactional_dataset_with_date_feature():
    """Compatibility alias for Scenario F."""
    test_scenario_f_transactional_dataset_with_date_feature()


def test_unlabeled_clustering_dataset():
    """Compatibility alias for Scenario E."""
    test_scenario_e_unlabeled_clustering_coordinates_and_features()


def test_manual_target_selection_api():
    """Manual target selection flows seamlessly from API to engine and validation."""
    df = pd.DataFrame({
        "feat_a": [1.0, 2.0, 3.0, 4.0] * 15,
        "feat_b": [5.0, 6.0, 7.0, 8.0] * 15,
        "my_custom_label": ["alpha", "beta", "alpha", "gamma"] * 15
    })
    csv_bytes = df.to_csv(index=False).encode('utf-8')

    response = client.post(
        "/api/v1/evaluate",
        files={"file": ("custom_dataset.csv", csv_bytes, "text/csv")},
        data={"task": "supervised_classification", "target_column": "my_custom_label"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["task_type"] == "supervised_classification"
    assert data["target_column"] == "my_custom_label"
    assert data["model_family"] == "Random Forest Classifier"
    assert data["metric"] == "F1"
    assert data["observed_metric_value"] is not None


def test_manual_task_selection_auto_resolved_target():
    """Manual task selection without explicit target auto-resolves candidate target."""
    df = pd.DataFrame({
        "feat_1": np.random.randn(60),
        "feat_2": np.random.randn(60),
        "Species": ["setosa"] * 20 + ["versicolor"] * 20 + ["virginica"] * 20
    })

    report = DataTrustEngine.evaluate(df=df, task=TaskType.SUPERVISED_CLASSIFICATION)
    assert report.task_type == TaskType.SUPERVISED_CLASSIFICATION
    assert report.target_column == "Species"
    assert report.model_family == "Random Forest Classifier"
    assert report.observed_metric_value is not None


def test_numeric_column_not_datetime_protection():
    """Numeric measurements, floats, and substrings are protected from false timestamp detection."""
    # 1. Continuous floats
    assert not DataProfiler.is_likely_datetime_series(pd.Series([1.0, 2.3, 5.8]), "SepalWidthCm")
    assert not DataProfiler.is_likely_datetime_series(pd.Series([0.2, 0.4, 0.6]), "PetalWidthCm")

    # 2. Integer measurements without calendar year context
    assert not DataProfiler.is_likely_datetime_series(pd.Series([10, 20, 30, 40]), "width_mm")
    assert not DataProfiler.is_likely_datetime_series(pd.Series([650, 720, 810]), "credit_score")

    # 3. Numeric string measurements
    assert not DataProfiler.is_likely_datetime_series(pd.Series(["1.5", "2.8", "5.4"]), "measurement")

    # 4. Column names containing 'dt' substring in normal words
    for col in ["width", "bandwidth", "auditor", "condition", "credit"]:
        assert not DataProfiler.is_likely_datetime_series(pd.Series([1.5, 2.0, 2.5]), col)

    # 5. Genuine calendar dates must pass
    assert DataProfiler.is_likely_datetime_series(pd.Series(["2023-01-01", "2023-01-02", "2023-01-03"]), "order_date")
    assert DataProfiler.is_likely_datetime_series(pd.Series(["01/15/2022", "02/15/2022", "03/15/2022"]), "date")


def test_missing_target_invalid_state_handling():
    """Supervised task with missing target does not fabricate zero metrics or false alignment."""
    df = pd.DataFrame({
        "feature_1": [1.0, 2.0, 3.0] * 10,
        "feature_2": [4.0, 5.0, 6.0] * 10
    })

    report = DataTrustEngine.evaluate(df=df, task=TaskType.SUPERVISED_CLASSIFICATION, target_column="nonexistent")
    assert report.target_column == "nonexistent"
    assert report.observed_metric_value is None
    assert report.metric_residual is None
    assert report.calibration is None
    assert "Target Missing" in report.model_family
