import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from datatrust.models import TaskType, QualityDimension
from datatrust.engine import DataTrustEngine
from datatrust.validator import DownstreamValidator
from datatrust.weighting import AHPWeightingEngine, ALL_DIMENSIONS

data_dir = "benchmarks/data"
df_wine = pd.read_csv(os.path.join(data_dir, "winequality-red.csv"))
df_house = pd.read_csv(os.path.join(data_dir, "house_sales_ts.csv"))
df_air = pd.read_csv(os.path.join(data_dir, "air_quality_ts.csv"))

print("=== EVALUATION ON REAL DATASETS ===")

# Test 1: Real Datasets Evaluation
# Evaluate all 3 real datasets directly:
datasets = [
    ("wine", df_wine, TaskType.SUPERVISED_REGRESSION, "quality", None),
    ("house", df_house, TaskType.TIME_SERIES_FORECASTING, "Price", "Datesold"),
    ("air", df_air, TaskType.TIME_SERIES_FORECASTING, "CO(GT)", "timestamp"),
]

real_obs = []
for name, df, task, target, time_col in datasets:
    # 3 slices per dataset to evaluate real variations (9 real sample cohorts)
    chunks = np.array_split(df, 3)
    for c_idx, chunk in enumerate(chunks):
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
        # Utility is inverted NRMSE (higher is better)
        utility = float(-nrmse)
        dims = {QualityDimension(k): v.score for k, v in rep.dimensions.items()}
        real_obs.append({
            "dataset": name,
            "chunk": c_idx,
            "task": task,
            "dims": dims,
            "rmse": ev["observed_metric"],
            "utility": utility
        })

print(f"Collected {len(real_obs)} real cohorts across 3 real datasets.")

dim_list = [r["dims"] for r in real_obs]
utility_list = [r["utility"] for r in real_obs]

# 1. Uncalibrated Prior weights (for Regression / TS)
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

cal_fitness = [sum(cal_w[dim] * s[dim] for dim in ALL_DIMENSIONS) for s in dim_list]
rho_cal, p_cal = spearmanr(cal_fitness, utility_list)

print(f"Prior Spearman Rank Correlation:      {rho_prior:.4f} (p = {p_prior:.4f})")
print(f"Calibrated Spearman Rank Correlation: {rho_cal:.4f} (p = {p_cal:.4f})")
print(f"Correlation Delta (Cal - Prior):      {rho_cal - rho_prior:+.4f}")
print(f"Consistency Ratio (CR):               {cal_cr:.4f} (Saaty axiom < 0.10: {cal_cr < 0.10})")
print(f"Reciprocal Matrix Verified:           {np.allclose(cal_matrix, 1.0 / cal_matrix.T, atol=1e-5)}")
