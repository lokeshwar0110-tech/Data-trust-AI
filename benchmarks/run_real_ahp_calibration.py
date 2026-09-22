"""
Real-World Dataset AHP Calibration Validation Script
Evaluates empirical AHP calibration using actual measured dimension scores and actual downstream
model performance across the three real-world benchmark datasets:
1. UCI Red Wine Quality (winequality-red.csv) - Tabular Physical-Chemical Regression
2. House Sales Transactions (house_sales_ts.csv) - Real Estate Transaction Time Series
3. UCI Air Quality Chemical Sensors (air_quality_ts.csv) - Environmental Sensor Network Time Series

Evaluates:
- Part A: Direct Whole-Dataset Calibration (3 authentic datasets)
- Part B: Real Multi-Cohort Calibration (real partitions across the 3 authentic datasets)
Reports exact Spearman rank correlation BEFORE and AFTER calibration on real data only.
"""

import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from datatrust.models import TaskType, QualityDimension
from datatrust.engine import DataTrustEngine
from datatrust.validator import DownstreamValidator
from datatrust.weighting import AHPWeightingEngine, ALL_DIMENSIONS


def run_real_ahp_calibration():
    print("=" * 105)
    print("EMPIRICAL AHP CALIBRATION VALIDATION ON REAL-WORLD DATASETS")
    print("=" * 105)

    data_dir = os.path.join(os.path.dirname(__file__), "data")
    wine_path = os.path.join(data_dir, "winequality-red.csv")
    house_path = os.path.join(data_dir, "house_sales_ts.csv")
    air_path = os.path.join(data_dir, "air_quality_ts.csv")

    df_wine = pd.read_csv(wine_path)
    df_house = pd.read_csv(house_path)
    df_air = pd.read_csv(air_path)

    # ---------------------------------------------------------
    # PART A: DIRECT WHOLE-DATASET REAL CALIBRATION (3 DATASETS)
    # ---------------------------------------------------------
    print("\n--- PART A: DIRECT EVALUATION ON 3 WHOLE REAL DATASETS ---")
    
    # 1. Wine
    rep_w = DataTrustEngine.evaluate(df_wine, task=TaskType.SUPERVISED_REGRESSION, target_column="quality")
    ev_w = DownstreamValidator.train_and_evaluate_split(
        df_wine.iloc[:int(len(df_wine) * 0.7)],
        df_wine.iloc[int(len(df_wine) * 0.7):],
        task=TaskType.SUPERVISED_REGRESSION,
        target_column="quality"
    )
    rmse_w = ev_w["observed_metric"]
    nrmse_w = rmse_w / float(df_wine["quality"].std())

    # 2. House Sales
    rep_h = DataTrustEngine.evaluate(df_house, task=TaskType.TIME_SERIES_FORECASTING, target_column="Price", time_column="Datesold")
    ev_h = DownstreamValidator.train_and_evaluate_split(
        df_house.iloc[:int(len(df_house) * 0.7)],
        df_house.iloc[int(len(df_house) * 0.7):],
        task=TaskType.TIME_SERIES_FORECASTING,
        target_column="Price",
        time_column="Datesold"
    )
    rmse_h = ev_h["observed_metric"]
    nrmse_h = rmse_h / float(df_house["Price"].std())

    # 3. Air Quality
    rep_a = DataTrustEngine.evaluate(df_air, task=TaskType.TIME_SERIES_FORECASTING, target_column="CO(GT)", time_column="timestamp")
    ev_a = DownstreamValidator.train_and_evaluate_split(
        df_air.iloc[:int(len(df_air) * 0.7)],
        df_air.iloc[int(len(df_air) * 0.7):],
        task=TaskType.TIME_SERIES_FORECASTING,
        target_column="CO(GT)",
        time_column="timestamp"
    )
    rmse_a = ev_a["observed_metric"]
    nrmse_a = rmse_a / float(df_air["CO(GT)"].std())

    real_3_table = [
        {"Dataset": "UCI Red Wine Quality", "Task": "Regression", "Rows": len(df_wine), "Raw_Fitness": rep_w.raw_fitness, "RMSE": rmse_w, "NRMSE": round(nrmse_w, 3)},
        {"Dataset": "House Sales Transactions", "Task": "Time-Series", "Rows": len(df_house), "Raw_Fitness": rep_h.raw_fitness, "RMSE": rmse_h, "NRMSE": round(nrmse_h, 3)},
        {"Dataset": "UCI Air Quality Sensors", "Task": "Time-Series", "Rows": len(df_air), "Raw_Fitness": rep_a.raw_fitness, "RMSE": rmse_a, "NRMSE": round(nrmse_a, 3)},
    ]
    print(pd.DataFrame(real_3_table).to_string(index=False))

    dims_3 = [
        {QualityDimension(k): v.score for k, v in rep_w.dimensions.items()},
        {QualityDimension(k): v.score for k, v in rep_h.dimensions.items()},
        {QualityDimension(k): v.score for k, v in rep_a.dimensions.items()}
    ]
    # Utility = negative NRMSE (higher utility = better prediction)
    perf_3 = [-nrmse_w, -nrmse_h, -nrmse_a]

    AHPWeightingEngine.reset_calibration()
    prior_w_3, _ = AHPWeightingEngine.get_task_weights(TaskType.TIME_SERIES_FORECASTING, use_calibrated=False)
    fit_prior_3 = [sum(prior_w_3[d] * s[d] for d in ALL_DIMENSIONS) for s in dims_3]
    rho_prior_3, p_prior_3 = spearmanr(fit_prior_3, perf_3)

    cal_matrix_3, cal_w_3, cr_3, meta_3 = AHPWeightingEngine.calibrate_task_ahp_matrix(
        task=TaskType.TIME_SERIES_FORECASTING,
        dimension_scores_list=dims_3,
        downstream_performances=perf_3,
        alpha=0.6,
        higher_is_better=True
    )
    fit_cal_3 = [sum(cal_w_3[d] * s[d] for d in ALL_DIMENSIONS) for s in dims_3]
    rho_cal_3, p_cal_3 = spearmanr(fit_cal_3, perf_3)

    print(f"\n3 Whole Datasets Results:")
    print(f"  Prior Spearman rho:      {rho_prior_3:.4f}")
    print(f"  Calibrated Spearman rho: {rho_cal_3:.4f}")
    print(f"  CR:                      {cr_3:.4f} (Saaty < 0.10: {cr_3 < 0.10})")

    # ---------------------------------------------------------
    # PART B: MULTI-COHORT REAL CALIBRATION (9 REAL COHORTS)
    # ---------------------------------------------------------
    print("\n--- PART B: MULTI-COHORT EVALUATION ACROSS 3 REAL DATASETS (9 COHORTS) ---")
    datasets = [
        ("UCI Red Wine", df_wine, TaskType.SUPERVISED_REGRESSION, "quality", None),
        ("House Sales", df_house, TaskType.TIME_SERIES_FORECASTING, "Price", "Datesold"),
        ("UCI Air Quality", df_air, TaskType.TIME_SERIES_FORECASTING, "CO(GT)", "timestamp"),
    ]

    cohort_results = []
    for name, df, task, target, time_col in datasets:
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
            utility = float(-nrmse)
            dims = {QualityDimension(k): v.score for k, v in rep.dimensions.items()}
            cohort_results.append({
                "dataset": name,
                "cohort": f"{name}_part_{c_idx+1}",
                "dims": dims,
                "raw_fitness": round(rep.raw_fitness, 1),
                "rmse": round(ev["observed_metric"], 2),
                "nrmse": round(nrmse, 3),
                "utility": utility
            })

    print(f"Successfully evaluated {len(cohort_results)} real cohorts.")
    dim_list_9 = [r["dims"] for r in cohort_results]
    utility_list_9 = [r["utility"] for r in cohort_results]

    AHPWeightingEngine.reset_calibration()
    prior_w_9, _ = AHPWeightingEngine.get_task_weights(TaskType.TIME_SERIES_FORECASTING, use_calibrated=False)
    fit_prior_9 = [sum(prior_w_9[d] * s[d] for d in ALL_DIMENSIONS) for s in dim_list_9]
    rho_prior_9, p_prior_9 = spearmanr(fit_prior_9, utility_list_9)

    cal_matrix_9, cal_w_9, cr_9, meta_9 = AHPWeightingEngine.calibrate_task_ahp_matrix(
        task=TaskType.TIME_SERIES_FORECASTING,
        dimension_scores_list=dim_list_9,
        downstream_performances=utility_list_9,
        alpha=0.6,
        higher_is_better=True
    )
    fit_cal_9 = [sum(cal_w_9[d] * s[d] for d in ALL_DIMENSIONS) for s in dim_list_9]
    rho_cal_9, p_cal_9 = spearmanr(fit_cal_9, utility_list_9)

    print(f"\n9 Real Cohorts Results:")
    print(f"  Prior Spearman rho:      {rho_prior_9:.4f} (p = {p_prior_9:.4f})")
    print(f"  Calibrated Spearman rho: {rho_cal_9:.4f} (p = {p_cal_9:.4f})")
    print(f"  Correlation Delta:       {rho_cal_9 - rho_prior_9:+.4f}")
    print(f"  CR:                      {cr_9:.4f} (Saaty < 0.10: {cr_9 < 0.10})")
    print(f"  Reciprocity Verified:    {np.allclose(cal_matrix_9, 1.0 / cal_matrix_9.T, atol=1e-5)}")
    print(f"  Provenance:              {meta_9['provenance']}")

    AHPWeightingEngine.reset_calibration()
    return {
        "rho_prior_3": rho_prior_3,
        "rho_cal_3": rho_cal_3,
        "rho_prior_9": rho_prior_9,
        "rho_cal_9": rho_cal_9,
        "p_cal_9": p_cal_9,
        "cr_9": cr_9
    }


if __name__ == "__main__":
    run_real_ahp_calibration()
