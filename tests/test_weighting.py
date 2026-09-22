import pytest
import numpy as np
from datatrust.models import TaskType, QualityDimension, CANONICAL_QUALITY_DIMENSIONS
from datatrust.weighting import AHPWeightingEngine, TASK_PAIRWISE_PREFERENCES


def test_ahp_eigenvector_weights_sum_to_one():
    """Verify that for all task types, AHP computes positive weights across all 8 canonical dimensions summing to 1.0."""
    for task in TaskType:
        weights, cr = AHPWeightingEngine.get_task_weights(task)
        # Must have exactly the 8 canonical dimensions
        for dim in CANONICAL_QUALITY_DIMENSIONS:
            assert dim in weights, f"Canonical dimension {dim} missing in {task} weights"
            assert weights[dim] > 0.0, f"Weight for {dim} in {task} must be strictly positive"

        total_w = sum(weights[dim] for dim in CANONICAL_QUALITY_DIMENSIONS)
        assert pytest.approx(total_w, abs=1e-4) == 1.0, f"Canonical weights for {task} do not sum to 1.0"


def test_ahp_consistency_ratio():
    """Verify that all task-specific pairwise comparison matrices satisfy Saaty's axiom (CR < 0.10)."""
    for task in TaskType:
        resolved = task if task != TaskType.DESCRIPTIVE_BI else TaskType.CLUSTERING
        matrix = TASK_PAIRWISE_PREFERENCES[resolved]
        weights, lambda_max, cr = AHPWeightingEngine.solve_weights_and_consistency(matrix)
        # Saaty's axiom: CR must be < 0.10 for mathematically defensible consistency
        assert cr < 0.10, f"Task {task} pairwise matrix has inconsistent CR = {cr:.4f} (must be < 0.10)"


def test_task_specific_divergence():
    """Verify task conditioning causes significant, domain-appropriate weight divergence across canonical dimensions."""
    ts_weights, _ = AHPWeightingEngine.get_task_weights(TaskType.TIME_SERIES_FORECASTING)
    clf_weights, _ = AHPWeightingEngine.get_task_weights(TaskType.SUPERVISED_CLASSIFICATION)
    reg_weights, _ = AHPWeightingEngine.get_task_weights(TaskType.SUPERVISED_REGRESSION)
    clust_weights, _ = AHPWeightingEngine.get_task_weights(TaskType.CLUSTERING)

    # 1. In Time Series, Timeliness (chronological ordering) must dominate and exceed 0.30
    assert ts_weights[QualityDimension.TIMELINESS] > ts_weights[QualityDimension.DISTRIBUTION_BALANCE]
    assert ts_weights[QualityDimension.TIMELINESS] > 0.30

    # 2. In Supervised Classification, Distribution Balance and Validity dominate over Timeliness
    assert clf_weights[QualityDimension.DISTRIBUTION_BALANCE] > clf_weights[QualityDimension.TIMELINESS]
    assert clf_weights[QualityDimension.VALIDITY] > clf_weights[QualityDimension.TIMELINESS]

    # 3. In Regression, Outlier Anomaly resilience is weighted significantly higher than in Clustering
    assert reg_weights[QualityDimension.OUTLIER_ANOMALY] > clust_weights[QualityDimension.OUTLIER_ANOMALY]


def test_non_compensatory_veto_rule():
    """Verify non-compensatory veto policies enforce score ceilings for fatal defects under task conditioning."""
    # 1. Temporal defect under Time Series
    dim_scores_ts = {
        QualityDimension.COMPLETENESS: 100.0,
        QualityDimension.VALIDITY: 100.0,
        QualityDimension.CONSISTENCY: 100.0,
        QualityDimension.UNIQUENESS: 100.0,
        QualityDimension.TIMELINESS: 20.0,  # CRITICAL FAILURE
        QualityDimension.OUTLIER_ANOMALY: 100.0,
        QualityDimension.DISTRIBUTION_BALANCE: 100.0,
        QualityDimension.COVERAGE_REPRESENTATIVENESS: 100.0
    }

    vetoed_ts, msg_ts, cap_ts = AHPWeightingEngine.evaluate_veto_penalties(
        TaskType.TIME_SERIES_FORECASTING,
        dim_scores_ts,
        {"temporal_veto_reason": "Severe chronological disruption"}
    )
    assert vetoed_ts is True
    assert cap_ts == 25.0

    # 2. Same temporal defect under Clustering should NOT trigger temporal veto
    vetoed_clust, _, _ = AHPWeightingEngine.evaluate_veto_penalties(
        TaskType.CLUSTERING,
        dim_scores_ts,
        {}
    )
    assert vetoed_clust is False

    # 3. Universal completeness veto (< 30.0 score)
    dim_scores_missing = {dim: 100.0 for dim in CANONICAL_QUALITY_DIMENSIONS}
    dim_scores_missing[QualityDimension.COMPLETENESS] = 20.0
    vetoed_univ, msg_univ, cap_univ = AHPWeightingEngine.evaluate_veto_penalties(
        TaskType.CLUSTERING,
        dim_scores_missing,
        {}
    )
    assert vetoed_univ is True
    assert cap_univ == 20.0

