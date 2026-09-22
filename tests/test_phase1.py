import pytest
import numpy as np
import pandas as pd
from datatrust.models import TaskType, QualityDimension, DimensionResult, DefectPolicyResult
from datatrust.profiler import DataProfiler
from datatrust.task_infer import TaskInferenceEngine
from datatrust.quality_engine import QualityEngine
from datatrust.sensitivity import TaskQualitySensitivityMatrix
from datatrust.weighting import AHPWeightingEngine
from datatrust.veto_engine import DefectPolicyEngine
from datatrust.engine import DataTrustEngine
from datatrust.config import (
    AHP_RANDOM_INDEX,
    CANONICAL_DIMENSIONS,
    TASK_QUALITY_SENSITIVITY_MATRIX,
    TASK_AHP_PAIRWISE_MATRICES,
    PROVENANCE_PIPELINE
)


def _make_dim_result(dim: QualityDimension, score: float, raw_metrics: dict = None, is_veto: bool = False, veto_msg: str = None) -> DimensionResult:
    return DimensionResult(
        dimension=dim,
        score=score,
        weight=0.125,
        weighted_score=score * 0.125,
        summary="Test summary",
        raw_metrics=raw_metrics or {},
        is_veto_triggered=is_veto,
        veto_reason=veto_msg
    )


def test_profiler_metadata_and_sha256():
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=30, freq="D"),
        "feature_x": np.random.randn(30),
        "target": np.random.choice([0, 1], 30)
    })
    meta, _ = DataProfiler.profile(df)
    assert meta.num_rows == 30
    assert meta.num_columns == 3
    assert len(meta.dataset_sha256) == 64
    # Determinism: same df gives same sha256
    meta2, _ = DataProfiler.profile(df)
    assert meta.dataset_sha256 == meta2.dataset_sha256
    assert "timestamp" in meta.candidate_timestamps
    assert "target" in meta.candidate_targets


def test_condition1_task_inference_clustering_bounds():
    # Unlabelled dataset -> clustering hypothesis must be bounded to <= 0.60 with ambiguity warning
    df = pd.DataFrame({
        "f1": np.random.randn(40),
        "f2": np.random.randn(40),
        "f3": np.random.randn(40)
    })
    res = TaskInferenceEngine.infer_task(df)
    assert res.inferred_task == TaskType.CLUSTERING
    assert res.confidence <= 0.60
    assert res.ambiguity_warning is not None
    assert "clustering" in res.ambiguity_warning.lower() and "inferred" in res.ambiguity_warning.lower()


def test_task_inference_user_override():
    df = pd.DataFrame({
        "f1": np.random.randn(40),
        "f2": np.random.randn(40)
    })
    res = TaskInferenceEngine.infer_task(df, user_override_task=TaskType.SUPERVISED_REGRESSION)
    assert res.inferred_task == TaskType.CLUSTERING
    assert res.final_selected_task == TaskType.SUPERVISED_REGRESSION
    assert res.user_overrode is True


def test_condition2_consistency_structural_contradictions():
    # Feature collinearity / structural consistency
    df = pd.DataFrame({
        "feat_a": [1.0, 2.0, 3.0, 4.0, 5.0] * 4,
        "feat_b": [1.0, 2.0, 3.0, 4.0, 5.0] * 4,
        "feat_c": np.random.randn(20)
    })
    engine = QualityEngine(df)
    res = engine.evaluate_consistency()
    assert res.dimension == QualityDimension.CONSISTENCY
    assert "collinear" in res.formula.lower() or "collinear_pairs" in res.raw_metrics
    assert res.provenance == "HEURISTIC"


def test_condition3_outlier_isolation_forest_and_kurtosis():
    np.random.seed(42)
    vals = np.random.normal(10, 2, 100).tolist()
    vals[0] = 99999.0
    vals[1] = -88888.0
    df = pd.DataFrame({"feat1": vals, "feat2": np.random.normal(5, 1, 100)})
    engine = QualityEngine(df)
    res = engine.evaluate_outliers()
    assert res.dimension == QualityDimension.OUTLIER_ANOMALY
    assert "isolation_forest" in res.formula.lower() or "isolation forest" in res.assumptions.lower()
    assert "kurtosis" in res.formula.lower() or "kurtosis" in res.assumptions.lower()
    assert "anomaly_ratio" in res.raw_metrics
    assert res.raw_metrics["kurtosis_role"] == "descriptive_supporting_metric_only"


def test_condition4_coverage_representativeness_unavailable():
    df = pd.DataFrame({
        "feature_1": [1.0, 2.0, 3.0, 4.0, 5.0] * 4,
        "feature_2": [10.0, 20.0, 30.0, 40.0, 50.0] * 4
    })
    engine = QualityEngine(df)
    res = engine.evaluate_coverage_representativeness()
    assert res.dimension == QualityDimension.COVERAGE_REPRESENTATIVENESS
    assert res.raw_metrics["population_representativeness"] == "UNAVAILABLE"
    assert "observable feature support" in res.assumptions.lower()


def test_eight_canonical_dimensions_evaluation():
    n = 50
    df = pd.DataFrame({
        "time": pd.date_range("2024-01-01", periods=n, freq="D"),
        "feat_a": np.random.randn(n),
        "feat_b": np.random.choice(["X", "Y"], n),
        "target": np.random.choice([0, 1], n)
    })
    engine = QualityEngine(df, target_column="target", time_column="time")
    results = engine.evaluate_all(TaskType.SUPERVISED_CLASSIFICATION)
    assert len(results) == 8
    for dim in CANONICAL_DIMENSIONS:
        assert dim in results
        assert results[dim].provenance != ""
        assert results[dim].formula != ""


def test_task_quality_sensitivity_matrix():
    matrix = TaskQualitySensitivityMatrix.get_matrix_dataframe()
    assert len(matrix) == 4
    for task_name, row in matrix.items():
        assert len(row) == 8
        for dim in CANONICAL_DIMENSIONS:
            assert dim.value in row


def test_ahp_matrices_ri_141_and_cr_below_010():
    assert AHP_RANDOM_INDEX == 1.41
    for task in TaskType:
        weights, cr = AHPWeightingEngine.get_task_weights(task)
        assert cr < 0.10, f"CR for {task} is {cr:.4f} >= 0.10"
        assert pytest.approx(sum(weights.values()), abs=1e-4) == 1.0
        assert len(weights) == 8


def test_three_level_defect_policy_green():
    # Clean scores -> GREEN
    dim_results = {dim: _make_dim_result(dim, 95.0) for dim in CANONICAL_DIMENSIONS}
    res = DefectPolicyEngine.evaluate_policy(TaskType.SUPERVISED_CLASSIFICATION, dim_results, raw_fitness=95.0)
    assert res.tier == "GREEN"
    assert res.fitness_status == "NORMAL"
    assert res.effective_fitness == 95.0


def test_three_level_defect_policy_yellow():
    # Moderate defect -> YELLOW cap applied
    dim_results = {dim: _make_dim_result(dim, 85.0) for dim in CANONICAL_DIMENSIONS}
    dim_results[QualityDimension.COMPLETENESS] = _make_dim_result(
        QualityDimension.COMPLETENESS, 55.0, {"missing_rate": 0.35}
    )
    res = DefectPolicyEngine.evaluate_policy(TaskType.SUPERVISED_CLASSIFICATION, dim_results, raw_fitness=80.0)
    assert res.tier == "YELLOW"
    assert res.fitness_status == "WARNING"
    assert res.effective_fitness <= 70.0


def test_three_level_defect_policy_red():
    # Critical defect -> RED hard veto sets fitness to 0.0 and status to UNSAFE
    dim_results = {dim: _make_dim_result(dim, 85.0) for dim in CANONICAL_DIMENSIONS}
    dim_results[QualityDimension.TIMELINESS] = _make_dim_result(
        QualityDimension.TIMELINESS, 20.0, {"is_monotonic_increasing": False}
    )
    res = DefectPolicyEngine.evaluate_policy(TaskType.TIME_SERIES_FORECASTING, dim_results, raw_fitness=80.0)
    assert res.tier == "RED"
    assert res.fitness_status == "UNSAFE"
    assert res.effective_fitness == 0.0


def test_end_to_end_phase1_pipeline():
    n = 60
    df = pd.DataFrame({
        "feature_1": np.random.randn(n),
        "feature_2": np.random.randn(n),
        "target": np.random.choice([0, 1], n)
    })
    report = DataTrustEngine.evaluate(df=df, task=TaskType.SUPERVISED_CLASSIFICATION, target_column="target")
    assert report.task_type == TaskType.SUPERVISED_CLASSIFICATION
    assert report.raw_fitness > 0.0
    assert report.fitness_status in ["NORMAL", "WARNING", "UNSAFE"]
    assert report.defect_policy is not None
    assert report.profiling is not None
    assert len(report.profiling.dataset_sha256) == 64
    assert len(report.dimensions) >= 8
    assert report.ahp_consistency_ratio < 0.10
    assert report.pipeline_provenance == PROVENANCE_PIPELINE
