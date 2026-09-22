import pytest
import numpy as np
import pandas as pd
from datatrust.models import TaskType, QualityDimension
from datatrust.uncertainty import FitnessUncertaintyEstimator
from datatrust.reproducibility import ReproducibilityEngine
from datatrust.calibration import PerformanceCalibrationEngine
from datatrust.engine import DataTrustEngine


def test_fitness_uncertainty_estimator():
    dim_scores = {
        QualityDimension.COMPLETENESS: 90.0,
        QualityDimension.OUTLIER_RESILIENCE: 80.0,
        QualityDimension.DRIFT_STABILITY: 85.0,
        QualityDimension.LABEL_INTEGRITY: 75.0,
        QualityDimension.TEMPORAL_VALIDITY: 95.0,
        QualityDimension.CARDINALITY_SCHEMA: 90.0,
        QualityDimension.CORRELATION_INTEGRITY: 85.0
    }
    weights = {dim: 1.0 / 7.0 for dim in dim_scores}
    u = FitnessUncertaintyEstimator.estimate_uncertainty(
        fitness_score=85.0,
        dimension_scores=dim_scores,
        weights=weights,
        num_rows=200
    )
    assert u.ci_lower < 85.0 < u.ci_upper
    assert u.margin_of_error > 0.0
    assert u.confidence_level == 0.95


def test_reproducibility_engine():
    df = pd.DataFrame({
        "col_a": [1, 2, 3, 4, 5],
        "col_b": ["A", "B", "C", "D", "E"]
    })
    h1 = ReproducibilityEngine.compute_dataset_hash(df)
    h2 = ReproducibilityEngine.compute_dataset_hash(df)
    assert h1 == h2
    assert len(h1) == 64


def test_native_rmse_calibration():
    task = TaskType.SUPERVISED_REGRESSION
    initial_weights = {
        QualityDimension.COMPLETENESS: 0.144,
        QualityDimension.OUTLIER_RESILIENCE: 0.250,
        QualityDimension.DRIFT_STABILITY: 0.144,
        QualityDimension.LABEL_INTEGRITY: 0.081,
        QualityDimension.TEMPORAL_VALIDITY: 0.050,
        QualityDimension.CARDINALITY_SCHEMA: 0.081,
        QualityDimension.CORRELATION_INTEGRITY: 0.250
    }
    dim_scores = {
        QualityDimension.COMPLETENESS: 95.0,
        QualityDimension.OUTLIER_RESILIENCE: 40.0, # Heavy outlier defect
        QualityDimension.DRIFT_STABILITY: 90.0,
        QualityDimension.LABEL_INTEGRITY: 90.0,
        QualityDimension.TEMPORAL_VALIDITY: 95.0,
        QualityDimension.CARDINALITY_SCHEMA: 95.0,
        QualityDimension.CORRELATION_INTEGRITY: 90.0
    }
    # Initial fitness is 82.5, predicted RMSE is 2.30, but observed is 3.10
    res = PerformanceCalibrationEngine.calibrate(
        task=task,
        initial_weights=initial_weights,
        dimension_scores=dim_scores,
        initial_trust_score=82.5,
        observed_performance=5.80,
        metric_name="RMSE",
        target_std=10.0
    )
    assert res.metric_name == "RMSE"
    assert res.residual > 0
    # Outlier resilience weight must increase
    assert res.weight_adjustments["outlier_resilience"]["calibrated"] > res.weight_adjustments["outlier_resilience"]["initial"]
    # Calibrated trust score must drop to reflect higher defect penalty
    assert res.calibrated_trust_score < res.initial_trust_score
    # Error reduction guaranteed: |residual_after| <= |residual|
    assert abs(res.residual_after) <= abs(res.residual)
