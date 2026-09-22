import pytest
import pandas as pd
import numpy as np
from datatrust.models import (
    TaskType,
    QualityDimension,
    DimensionResult,
    ImbalancePolicy
)
from datatrust.veto_engine import IntelligentVetoEngine
from datatrust.quality_engine import QualityEngine


def _make_dummy_dim_result(dim: QualityDimension, score: float, raw_metrics=None, is_veto=False, veto_reason=None):
    return DimensionResult(
        dimension=dim,
        score=score,
        weight=0.125,
        weighted_score=score * 0.125,
        raw_metrics=raw_metrics or {},
        summary=f"Score: {score}",
        is_veto_triggered=is_veto,
        veto_reason=veto_reason
    )


def test_red_veto_catastrophic_missingness():
    """Validates that missing rate > 70% triggers RED Hard Veto (F_final=0.0, UNSAFE)."""
    dim_results = {
        QualityDimension.COMPLETENESS: _make_dummy_dim_result(
            QualityDimension.COMPLETENESS,
            score=20.0,
            raw_metrics={"missing_rate": 0.75}
        )
    }
    res = IntelligentVetoEngine.evaluate_defect_policy(
        task=TaskType.SUPERVISED_REGRESSION,
        dimension_results=dim_results,
        raw_fitness=85.0,
        num_rows=100
    )
    assert res.tier == "RED"
    assert res.fitness_status == "UNSAFE"
    assert res.is_unsafe is True
    assert res.final_fitness == 0.0
    assert res.cap_applied == 0.0
    assert any(d.defect_name == "CATASTROPHIC_MISSINGNESS" for d in res.defects_detected)


def test_red_veto_temporal_sequence_disorder():
    """Validates that non-monotonic or duplicate timestamps in time series trigger RED Hard Veto."""
    dim_results = {
        QualityDimension.TIMELINESS: _make_dummy_dim_result(
            QualityDimension.TIMELINESS,
            score=30.0,
            raw_metrics={"is_monotonic_increasing": False, "duplicate_count": 5}
        )
    }
    res = IntelligentVetoEngine.evaluate_defect_policy(
        task=TaskType.TIME_SERIES_FORECASTING,
        dimension_results=dim_results,
        raw_fitness=90.0,
        num_rows=100
    )
    assert res.tier == "RED"
    assert res.fitness_status == "UNSAFE"
    assert res.is_unsafe is True
    assert res.final_fitness == 0.0
    assert any(d.defect_name == "TEMPORAL_SEQUENCE_DISORDER" for d in res.defects_detected)


def test_red_veto_target_leakage():
    """Validates that post-outcome target leakage (|r| > 0.98) triggers RED Hard Veto."""
    dim_results = {
        QualityDimension.CONSISTENCY: _make_dummy_dim_result(
            QualityDimension.CONSISTENCY,
            score=25.0,
            raw_metrics={"target_leakage_suspects": [{"feature": "leak_feat", "correlation": 0.99}]}
        )
    }
    res = IntelligentVetoEngine.evaluate_defect_policy(
        task=TaskType.SUPERVISED_CLASSIFICATION,
        dimension_results=dim_results,
        raw_fitness=95.0,
        num_rows=200
    )
    assert res.tier == "RED"
    assert res.fitness_status == "UNSAFE"
    assert res.is_unsafe is True
    assert res.final_fitness == 0.0
    assert any(d.defect_name == "TARGET_LEAKAGE" for d in res.defects_detected)


def test_red_veto_severe_class_collapse():
    """Validates that minority class < 2.0% in classification triggers RED Hard Veto."""
    dim_results = {
        QualityDimension.DISTRIBUTION_BALANCE: _make_dummy_dim_result(
            QualityDimension.DISTRIBUTION_BALANCE,
            score=15.0,
            raw_metrics={"minority_percentage": 1.2, "majority_percentage": 98.8}
        )
    }
    res = IntelligentVetoEngine.evaluate_defect_policy(
        task=TaskType.SUPERVISED_CLASSIFICATION,
        dimension_results=dim_results,
        raw_fitness=88.0,
        num_rows=500
    )
    assert res.tier == "RED"
    assert res.fitness_status == "UNSAFE"
    assert res.is_unsafe is True
    assert res.final_fitness == 0.0
    assert any(d.defect_name == "SEVERE_CLASS_COLLAPSE" for d in res.defects_detected)


def test_yellow_cap_moderate_missingness():
    """Validates that missing rate > 30% triggers YELLOW warning cap (F <= 65.0, WARNING)."""
    dim_results = {
        QualityDimension.COMPLETENESS: _make_dummy_dim_result(
            QualityDimension.COMPLETENESS,
            score=50.0,
            raw_metrics={"missing_rate": 0.35}
        )
    }
    res = IntelligentVetoEngine.evaluate_defect_policy(
        task=TaskType.SUPERVISED_REGRESSION,
        dimension_results=dim_results,
        raw_fitness=85.0,
        num_rows=100
    )
    assert res.tier == "YELLOW"
    assert res.fitness_status == "WARNING"
    assert res.is_unsafe is False
    assert res.final_fitness <= 65.0
    assert res.cap_applied == 65.0


def test_yellow_cap_collinearity():
    """Validates that >6 collinear pairs trigger YELLOW warning cap (F <= 70.0, WARNING)."""
    dim_results = {
        QualityDimension.CONSISTENCY: _make_dummy_dim_result(
            QualityDimension.CONSISTENCY,
            score=60.0,
            raw_metrics={"collinear_pairs": [("a", "b")] * 7}
        )
    }
    res = IntelligentVetoEngine.evaluate_defect_policy(
        task=TaskType.SUPERVISED_REGRESSION,
        dimension_results=dim_results,
        raw_fitness=90.0,
        num_rows=100
    )
    assert res.tier == "YELLOW"
    assert res.fitness_status == "WARNING"
    assert res.is_unsafe is False
    assert res.final_fitness <= 70.0
    assert res.cap_applied == 70.0


def test_green_normal_operating_state():
    """Validates that a dataset without defects achieves GREEN tier with uncapped fitness."""
    dim_results = {
        QualityDimension.COMPLETENESS: _make_dummy_dim_result(QualityDimension.COMPLETENESS, 95.0, {"missing_rate": 0.01}),
        QualityDimension.VALIDITY: _make_dummy_dim_result(QualityDimension.VALIDITY, 92.0),
        QualityDimension.CONSISTENCY: _make_dummy_dim_result(QualityDimension.CONSISTENCY, 88.0, {"target_leakage_suspects": [], "collinear_pairs": []}),
        QualityDimension.DISTRIBUTION_BALANCE: _make_dummy_dim_result(QualityDimension.DISTRIBUTION_BALANCE, 85.0, {"minority_percentage": 35.0})
    }
    res = IntelligentVetoEngine.evaluate_defect_policy(
        task=TaskType.SUPERVISED_CLASSIFICATION,
        dimension_results=dim_results,
        raw_fitness=90.0,
        num_rows=200
    )
    assert res.tier == "GREEN"
    assert res.fitness_status == "NORMAL"
    assert res.is_unsafe is False
    assert res.final_fitness == 90.0
    assert res.cap_applied is None
    assert len(res.defects_detected) == 0


def test_duplicate_timestamp_groups_vs_rows_metrics():
    """
    Validates the disambiguation between:
    - duplicate_timestamp_groups: distinct timestamps that appear >= 2 times
    - duplicate_timestamp_rows: total surplus observations (N - N_unique)
    """
    dates = (
        ["2024-01-01"] * 3
        + ["2024-01-02"] * 2
        + ["2024-01-03", "2024-01-04", "2024-01-05"]
    )
    df = pd.DataFrame({
        "timestamp": dates,
        "value": np.random.randn(len(dates))
    })
    qe = QualityEngine(df, time_column="timestamp")
    res = qe.evaluate_timeliness(TaskType.TIME_SERIES_FORECASTING)

    raw_m = res.raw_metrics
    assert raw_m["duplicate_timestamp_groups"] == 2
    assert raw_m["duplicate_timestamp_rows"] == 3
    assert raw_m["duplicate_count"] == 3
    assert raw_m["timestamp_unique"] == 5
