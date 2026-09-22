import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from datatrust.models import TaskType, QualityDimension
from datatrust.engine import DataTrustEngine
from datatrust.calibration import PerformanceCalibrationEngine

# Test Stress Scenarios
print("Testing Stress Scenarios...")

# 1. Stress Class Collapse (< 1%)
rng = np.random.RandomState(42)
df_collapse = pd.DataFrame({
    "x1": rng.randn(300),
    "x2": rng.randn(300),
    "target": [1 if i < 2 else 0 for i in range(300)]
})
rep_col = DataTrustEngine.evaluate(df=df_collapse, task=TaskType.SUPERVISED_CLASSIFICATION, target_column="target")
print("1. Class Collapse Tier:", rep_col.defect_policy.tier, "Final:", rep_col.final_fitness, "Status:", rep_col.fitness_status)
assert rep_col.defect_policy.tier == "RED"
assert rep_col.final_fitness == 0.0
assert rep_col.fitness_status == "UNSAFE"

# 2. Stress Temporal Disorder
df_disorder = pd.DataFrame({
    "date": ["2023-01-05", "2023-01-01", "2023-01-03", "2023-01-02", "2023-01-01"], # disordered + duplicate
    "temp": [15.2, 14.1, 13.8, 16.0, 14.5]
})
rep_time = DataTrustEngine.evaluate(df=df_disorder, task=TaskType.TIME_SERIES_FORECASTING, target_column="temp", time_column="date")
print("2. Temporal Disorder Tier:", rep_time.defect_policy.tier, "Final:", rep_time.final_fitness, "Status:", rep_time.fitness_status)
assert rep_time.defect_policy.tier == "RED"
assert rep_time.final_fitness == 0.0
assert rep_time.fitness_status == "UNSAFE"

# 3. Stress Flat Dimension Scores (Calibration Gradient Vanishing)
flat_scores = {dim: 75.0 for dim in QualityDimension}
weights_init = {dim: 1.0/8.0 for dim in QualityDimension}
cal_res = PerformanceCalibrationEngine.calibrate(
    task=TaskType.SUPERVISED_CLASSIFICATION,
    initial_weights=weights_init,
    dimension_scores=flat_scores,
    initial_trust_score=75.0,
    observed_performance=0.50
)
print("3. Flat Scores Calibration Status:", cal_res.calibration_status, "Provenance:", cal_res.provenance)
assert cal_res.calibration_status == "Calibration Rejected"
assert cal_res.provenance == "REJECTED"

# 4. Stress Insufficient Data (N < 10)
df_tiny = pd.DataFrame({
    "x1": [1.0, 2.0, 3.0, 4.0, 5.0],
    "target": [2.0, 4.0, 6.0, 8.0, 10.0]
})
rep_tiny = DataTrustEngine.evaluate(df=df_tiny, task=TaskType.SUPERVISED_REGRESSION, target_column="target")
print("4. Insufficient Data Obs Metric:", rep_tiny.observed_metric_value)
assert rep_tiny.observed_metric_value is None

print("ALL STRESS SCENARIO CHECKS PASSED!")
