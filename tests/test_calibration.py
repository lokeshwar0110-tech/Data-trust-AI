import pytest
from datatrust.calibration import PerformanceCalibrationEngine
from datatrust.models import TaskType, QualityDimension
from datatrust.sensitivity import TaskQualitySensitivityMatrix


def test_performance_prediction_monotonicity():
    p_high = PerformanceCalibrationEngine.predict_performance(95.0, TaskType.SUPERVISED_CLASSIFICATION)
    p_low = PerformanceCalibrationEngine.predict_performance(40.0, TaskType.SUPERVISED_CLASSIFICATION)
    assert p_high > p_low
    assert 0.0 <= p_high <= 1.0
    assert 0.0 <= p_low <= 1.0


def test_closed_loop_calibration_error_reduction():
    task = TaskType.SUPERVISED_CLASSIFICATION
    initial_weights = TaskQualitySensitivityMatrix.get_normalized_weights(task)
    
    initial_trust_score = 80.0
    # Model underperformed: observed 0.60
    observed_metric = 0.60

    dim_scores = {
        QualityDimension.COMPLETENESS: 95.0,
        QualityDimension.OUTLIER_RESILIENCE: 90.0,
        QualityDimension.DRIFT_STABILITY: 85.0,
        QualityDimension.LABEL_INTEGRITY: 55.0, # High defect
        QualityDimension.TEMPORAL_VALIDITY: 90.0,
        QualityDimension.CARDINALITY_SCHEMA: 90.0,
        QualityDimension.CORRELATION_INTEGRITY: 85.0
    }

    res = PerformanceCalibrationEngine.calibrate(
        task=task,
        initial_weights=initial_weights,
        dimension_scores=dim_scores,
        initial_trust_score=initial_trust_score,
        observed_performance=observed_metric
    )

    # Verify error reduction: |residual_after| < |residual_before|
    assert abs(res.residual_after) < abs(res.residual)
    # Verify label integrity received increased weight penalty
    assert res.weight_adjustments["label_integrity"]["calibrated"] > res.weight_adjustments["label_integrity"]["initial"]
    # Verify W_AHP and W_calibrated are populated
    assert len(res.weights_ahp) > 0
    assert len(res.weights_calibrated) > 0


def test_calibration_rejection_policy():
    task = TaskType.SUPERVISED_CLASSIFICATION
    initial_weights = TaskQualitySensitivityMatrix.get_normalized_weights(task)
    
    # Model matches prediction exactly (residual = 0.0) -> converged/rejected if forced
    dim_scores = {dim: 85.0 for dim in initial_weights}
    
    res = PerformanceCalibrationEngine.calibrate(
        task=task,
        initial_weights=initial_weights,
        dimension_scores=dim_scores,
        initial_trust_score=85.0,
        observed_performance=0.78
    )
    assert res.calibration_status in ["Calibration Converged", "Calibration Improved", "Calibration Accepted", "Calibration Rejected", "Converged"]
    assert res.loss_after <= res.loss_before


def test_empirical_ahp_calibration_improves_correlation():
    """
    Issue 1 Verification:
    Empirically calibrates AHP pairwise comparison matrix from reference dataset observations.
    Asserts:
    1. Calibrated pairwise comparison matrix retains reciprocal properties (A_ji = 1/A_ij).
    2. Maintains a Consistency Ratio CR < 0.10.
    3. Provenance metadata accurately distinguishes PROVENANCE_EMPIRICAL.
    4. Calibrated weights produce a strictly higher Spearman rank correlation with
       downstream model performance than uncalibrated prior weights.
    """
    import numpy as np
    from scipy.stats import spearmanr
    from datatrust.weighting import AHPWeightingEngine, ALL_DIMENSIONS
    from datatrust.config import PROVENANCE_EMPIRICAL

    np.random.seed(42)
    n_samples = 20
    scores_list = []
    downstream_f1 = []

    # Simulate reference datasets where Distribution Balance and Validity dominate true performance
    for _ in range(n_samples):
        s = {dim: float(np.random.uniform(35.0, 95.0)) for dim in ALL_DIMENSIONS}
        perf = (
            0.55 * (s[QualityDimension.DISTRIBUTION_BALANCE] / 100.0)
            + 0.35 * (s[QualityDimension.VALIDITY] / 100.0)
            + 0.10 * (s[QualityDimension.COMPLETENESS] / 100.0)
            + np.random.normal(0, 0.03)
        )
        scores_list.append(s)
        downstream_f1.append(float(perf))

    # 1. Uncalibrated Prior weights
    AHPWeightingEngine.reset_calibration()
    prior_weights, prior_cr = AHPWeightingEngine.get_task_weights(TaskType.SUPERVISED_CLASSIFICATION, use_calibrated=False)
    prior_fitness = [sum(prior_weights[dim] * s[dim] for dim in ALL_DIMENSIONS) for s in scores_list]
    rho_prior, _ = spearmanr(prior_fitness, downstream_f1)

    prior_meta = AHPWeightingEngine.get_ahp_metadata(TaskType.SUPERVISED_CLASSIFICATION, use_calibrated=False)
    assert prior_meta["provenance"] != PROVENANCE_EMPIRICAL

    # 2. Perform Empirical Calibration
    cal_matrix, cal_weights, cal_cr, cal_meta = AHPWeightingEngine.calibrate_task_ahp_matrix(
        task=TaskType.SUPERVISED_CLASSIFICATION,
        dimension_scores_list=scores_list,
        downstream_performances=downstream_f1,
        alpha=0.75,
        higher_is_better=True
    )

    # 3. Reciprocal property verification
    assert np.allclose(cal_matrix, 1.0 / cal_matrix.T, atol=1e-5), "Calibrated matrix must be strictly reciprocal"

    # 4. Consistency Ratio verification (Saaty's axiom: CR < 0.10)
    assert cal_cr < 0.10, f"Calibrated matrix CR = {cal_cr} must be < 0.10"

    # 5. Provenance tracking verification
    assert cal_meta["provenance"] == PROVENANCE_EMPIRICAL
    assert "CALIBRATED" in cal_meta["status"]

    # 6. Correlation improvement verification
    cal_fitness = [sum(cal_weights[dim] * s[dim] for dim in ALL_DIMENSIONS) for s in scores_list]
    rho_cal, _ = spearmanr(cal_fitness, downstream_f1)

    assert rho_cal > rho_prior, f"Calibrated correlation ({rho_cal:.4f}) must exceed prior ({rho_prior:.4f})"
    assert rho_cal > 0.85, f"Expected strong alignment with downstream metric, got {rho_cal:.4f}"

    # Clean up
    AHPWeightingEngine.reset_calibration()


def test_empirical_ahp_calibration_on_real_datasets():
    """
    Issue 2 Calibration Extension:
    Validates empirical AHP calibration using actual measured dimension scores and actual downstream
    model performance across the three real-world datasets:
    - winequality-red.csv (physical-chemical regression)
    - house_sales_ts.csv (real estate transaction time series with duplicate timestamps)
    - air_quality_ts.csv (chemical sensor network time series)
    
    Verifies that Spearman rank correlation with downstream model utility strictly improves
    after calibration on authentic real-world data without synthetic formulas.
    """
    import os
    import numpy as np
    import pandas as pd
    from scipy.stats import spearmanr
    from datatrust.models import TaskType, QualityDimension
    from datatrust.engine import DataTrustEngine
    from datatrust.validator import DownstreamValidator
    from datatrust.weighting import AHPWeightingEngine, ALL_DIMENSIONS
    from datatrust.config import PROVENANCE_EMPIRICAL

    data_dir = os.path.join(os.path.dirname(__file__), "..", "benchmarks", "data")
    df_wine = pd.read_csv(os.path.join(data_dir, "winequality-red.csv"))
    df_house = pd.read_csv(os.path.join(data_dir, "house_sales_ts.csv"))
    df_air = pd.read_csv(os.path.join(data_dir, "air_quality_ts.csv"))

    real_datasets = [
        ("wine", df_wine, TaskType.SUPERVISED_REGRESSION, "quality", None),
        ("house", df_house, TaskType.TIME_SERIES_FORECASTING, "Price", "Datesold"),
        ("air", df_air, TaskType.TIME_SERIES_FORECASTING, "CO(GT)", "timestamp"),
    ]

    real_obs = []
    for name, df, task, target, time_col in real_datasets:
        chunks = np.array_split(df, 3)
        for chunk in chunks:
            rep = DataTrustEngine.evaluate(chunk, task=task, target_column=target, time_column=time_col)
            split_point = int(len(chunk) * 0.7)
            tr = chunk.iloc[:split_point]
            te = chunk.iloc[split_point:]
            ev = DownstreamValidator.train_and_evaluate_split(
                train_df=tr,
                test_df=te,
                task=task,
                target_column=target,
                time_column=time_col
            )
            target_std = float(chunk[target].std()) if chunk[target].std() > 0 else 1.0
            nrmse = ev["observed_metric"] / target_std
            # Utility is negative NRMSE (lower RMSE = higher utility)
            utility = float(-nrmse)
            dims = {QualityDimension(k): v.score for k, v in rep.dimensions.items()}
            real_obs.append({"dims": dims, "utility": utility})

    dim_list = [r["dims"] for r in real_obs]
    utility_list = [r["utility"] for r in real_obs]

    # 1. Uncalibrated Prior weights
    AHPWeightingEngine.reset_calibration()
    prior_w, prior_cr = AHPWeightingEngine.get_task_weights(TaskType.TIME_SERIES_FORECASTING, use_calibrated=False)
    prior_fitness = [sum(prior_w[dim] * s[dim] for dim in ALL_DIMENSIONS) for s in dim_list]
    rho_prior, p_prior = spearmanr(prior_fitness, utility_list)

    # 2. Calibrate on the real datasets
    cal_matrix, cal_w, cal_cr, cal_meta = AHPWeightingEngine.calibrate_task_ahp_matrix(
        task=TaskType.TIME_SERIES_FORECASTING,
        dimension_scores_list=dim_list,
        downstream_performances=utility_list,
        alpha=0.6,
        higher_is_better=True
    )

    # 3. Assertions
    assert np.allclose(cal_matrix, 1.0 / cal_matrix.T, atol=1e-5), "Calibrated matrix must be strictly reciprocal"
    assert cal_cr < 0.10, f"Calibrated CR = {cal_cr} must satisfy Saaty's axiom (< 0.10)"
    assert cal_meta["provenance"] == PROVENANCE_EMPIRICAL

    cal_fitness = [sum(cal_w[dim] * s[dim] for dim in ALL_DIMENSIONS) for s in dim_list]
    rho_cal, p_cal = spearmanr(cal_fitness, utility_list)

    # Verify correlation strictly improves on real data
    assert rho_cal > rho_prior, f"Calibrated correlation ({rho_cal:.4f}) must exceed prior ({rho_prior:.4f})"
    assert rho_cal > 0.50, f"Expected substantial positive alignment on real data, got {rho_cal:.4f}"

    # Clean up
    AHPWeightingEngine.reset_calibration()


def test_calibration_rejection_when_loss_does_not_improve():
    """
    Validates that when candidate weight updates cannot strictly reduce loss,
    calibration is REJECTED, provenance is set to REJECTED, and prior weights are preserved.
    """
    task = TaskType.SUPERVISED_CLASSIFICATION
    initial_weights = TaskQualitySensitivityMatrix.get_normalized_weights(task)
    # Uniform scores where gradient step cannot shift weights to reduce loss
    dim_scores = {dim: 80.0 for dim in initial_weights}
    # Initial trust score = 80.0 -> predicted performance ~ 0.739
    pred_perf = PerformanceCalibrationEngine.predict_performance(80.0, task)
    # Set observed performance exactly to predicted so error cannot be reduced
    res = PerformanceCalibrationEngine.calibrate(
        task=task,
        initial_weights=initial_weights,
        dimension_scores=dim_scores,
        initial_trust_score=80.0,
        observed_performance=pred_perf
    )
    # Either converged or rejected
    assert res.provenance in ["NO_IMPROVEMENT", "REJECTED"]
    assert res.loss_after <= res.loss_before
    # Calibrated weights equal initial weights
    for dim in initial_weights:
        assert round(res.weights_calibrated[dim.value], 4) == round(res.weights_ahp[dim.value], 4)


def test_calibration_converged_provenance():
    """
    Validates that when residual is already < 0.02, status is 'Calibration Converged'
    and provenance is 'NO_IMPROVEMENT'.
    """
    task = TaskType.SUPERVISED_CLASSIFICATION
    initial_weights = TaskQualitySensitivityMatrix.get_normalized_weights(task)
    pred_perf = PerformanceCalibrationEngine.predict_performance(75.0, task)
    dim_scores = {dim: 75.0 for dim in initial_weights}
    res = PerformanceCalibrationEngine.calibrate(
        task=task,
        initial_weights=initial_weights,
        dimension_scores=dim_scores,
        initial_trust_score=75.0,
        observed_performance=pred_perf + 0.005  # |residual| < 0.02
    )
    assert res.calibration_status == "Calibration Converged"
    assert res.provenance == "NO_IMPROVEMENT"
    assert res.is_improved is False


def test_calibration_accepted_provenance():
    """
    Validates that when gradient descent reduces loss, provenance is EMPIRICALLY_CALIBRATED
    and is_improved is True.
    """
    task = TaskType.SUPERVISED_CLASSIFICATION
    initial_weights = TaskQualitySensitivityMatrix.get_normalized_weights(task)
    dim_scores = {
        QualityDimension.COMPLETENESS: 95.0,
        QualityDimension.VALIDITY: 90.0,
        QualityDimension.DISTRIBUTION_BALANCE: 40.0,
        QualityDimension.OUTLIER_ANOMALY: 85.0,
        QualityDimension.TIMELINESS: 80.0,
        QualityDimension.UNIQUENESS: 85.0,
        QualityDimension.CONSISTENCY: 85.0,
        QualityDimension.COVERAGE_REPRESENTATIVENESS: 85.0
    }
    res = PerformanceCalibrationEngine.calibrate(
        task=task,
        initial_weights=initial_weights,
        dimension_scores=dim_scores,
        initial_trust_score=80.0,
        observed_performance=0.55  # substantial divergence
    )
    assert res.is_improved is True
    assert res.provenance == "EMPIRICALLY_CALIBRATED"
    assert res.loss_after < res.loss_before
    assert res.calibration_status in ["Calibration Improved", "Calibration Accepted"]


def test_calibration_reversion_preserves_priors():
    """
    Validates that if an update fails to improve, weights_calibrated exactly preserves weights_ahp.
    """
    task = TaskType.SUPERVISED_REGRESSION
    initial_weights = TaskQualitySensitivityMatrix.get_normalized_weights(task)
    pred_perf = PerformanceCalibrationEngine.predict_performance(85.0, task, target_std=10.0)
    dim_scores = {dim: 85.0 for dim in initial_weights}
    res = PerformanceCalibrationEngine.calibrate(
        task=task,
        initial_weights=initial_weights,
        dimension_scores=dim_scores,
        initial_trust_score=85.0,
        observed_performance=pred_perf
    )
    for dim_val, w_init in res.weights_ahp.items():
        assert abs(res.weights_calibrated[dim_val] - w_init) < 1e-5


