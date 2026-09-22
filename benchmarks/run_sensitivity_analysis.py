"""
Sensitivity Analysis Script for Partition Weights and Remediation Simulation Discounting
Evaluates the robustness of the partition-aware post-remediation fitness and MRU predicted gain
across a wide parameter envelope: w_train in [0.50, 0.60, 0.70, 0.80, 0.85].

STRUCTURAL DERIVATION (ZERO HARDCODED NUMBERS):
- F_train is computed dynamically by evaluating the remediated training partition (rem_train).
- F_test and F_test_raw are computed dynamically by evaluating the immutable holdout test partition (test_df).
- Predicted gain is computed dynamically via RemediationOptimizer.optimize_remediations().
- RMSE before and after are computed dynamically via DownstreamValidator.train_and_evaluate_split().
"""

import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datatrust.models import TaskType, RemediationDecision
from datatrust.engine import DataTrustEngine
from datatrust.remediator import DataRemediator
from datatrust.validator import DownstreamValidator
from datatrust.optimizer import RemediationOptimizer


def run_sensitivity_analysis():
    print("=" * 115)
    print("SENSITIVITY ANALYSIS: PARTITION-AWARE FITNESS & MRU GAIN PREDICTION ROBUSTNESS (DYNAMIC EVALUATION)")
    print("=" * 115)

    # Generate benchmark dataset with temporal disorder (Dataset C)
    rng = np.random.RandomState(42)
    n = 200
    base_dates = pd.date_range("2024-01-01", periods=n, freq="D").tolist()
    y = np.zeros(n)
    y[0] = 50.0
    for i in range(1, n):
        y[i] = 0.70 * y[i - 1] + 6.0 * np.sin(2 * np.pi * (i % 7) / 7.0) + rng.normal(0, 1.0)
    seasonality = np.array([float(i % 7) for i in range(n)])
    sensor_reading = rng.normal(100, 5, n)
    for i in [10, 25, 40, 55, 70, 85, 100, 115, 130, 145, 160, 175]:
        base_dates[i] = base_dates[i - 1]
    shuffle_idx = rng.permutation(n)
    df_c = pd.DataFrame({
        "timestamp": [base_dates[idx] for idx in shuffle_idx],
        "day_of_week": seasonality[shuffle_idx],
        "sensor_reading": sensor_reading[shuffle_idx],
        "target": y[shuffle_idx]
    })

    # Initial evaluation of raw dataset
    rep_initial = DataTrustEngine.evaluate(
        df=df_c,
        task=TaskType.TIME_SERIES_FORECASTING,
        target_column="target",
        time_column="timestamp"
    )
    f_initial = rep_initial.final_fitness if rep_initial.final_fitness is not None else 0.0

    split_ratios = [0.50, 0.60, 0.70, 0.80, 0.85]
    results = []

    for w_train in split_ratios:
        w_test = 1.0 - w_train
        n_train = int(n * w_train)
        raw_train = df_c.iloc[:n_train].copy()
        raw_test = df_c.iloc[n_train:].copy()

        # 1. Remediate training partition while keeping holdout test partition strictly immutable
        rem_train, rem_test, execution_log, fitted_params = DataRemediator.fit_and_transform_splits(
            train_df=raw_train,
            test_df=raw_test,
            task=TaskType.TIME_SERIES_FORECASTING,
            action_ids=["sort_timestamps"],
            target_column="target",
            time_column="timestamp"
        )

        # 2. Evaluate downstream model before and after remediation
        eval_before = DownstreamValidator.train_and_evaluate_split(
            train_df=raw_train,
            test_df=raw_test,
            task=TaskType.TIME_SERIES_FORECASTING,
            target_column="target",
            time_column="timestamp"
        )
        rmse_before = round(eval_before["observed_metric"], 2)

        eval_after = DownstreamValidator.train_and_evaluate_split(
            train_df=rem_train,
            test_df=rem_test,
            task=TaskType.TIME_SERIES_FORECASTING,
            target_column="target",
            time_column="timestamp"
        )
        rmse_after = round(eval_after["observed_metric"], 2)
        model_improved = (rmse_after < rmse_before)

        # 3. Dynamic evaluation of cleaned training partition
        report_train = DataTrustEngine.evaluate(
            df=rem_train,
            task=TaskType.TIME_SERIES_FORECASTING,
            target_column="target",
            time_column="timestamp",
            observed_performance=rmse_after
        )
        f_train_eff = (
            report_train.defect_policy.final_fitness
            if report_train.defect_policy and report_train.defect_policy.final_fitness is not None
            else report_train.raw_fitness
        )
        f_train_raw = report_train.raw_fitness

        # 4. Dynamic evaluation of immutable holdout test partition
        report_test = DataTrustEngine.evaluate(
            df=rem_test,
            task=TaskType.TIME_SERIES_FORECASTING,
            target_column="target",
            time_column="timestamp"
        )
        f_test_eff = (
            report_test.defect_policy.final_fitness
            if report_test.defect_policy and report_test.defect_policy.final_fitness is not None
            else report_test.raw_fitness
        )
        f_test_raw = report_test.raw_fitness

        # 5. Dynamic optimizer predicted gain
        opt_actions = RemediationOptimizer.optimize_remediations(
            task=TaskType.TIME_SERIES_FORECASTING,
            dimensions=rep_initial.dimensions,
            current_trust_score=f_initial,
            is_vetoed=(rep_initial.defect_policy.tier == "RED" if rep_initial.defect_policy else False),
            train_ratio=w_train,
            test_defect_fraction=w_test
        )
        sort_action = next((a for a in opt_actions if a.action_type == "sort_timestamps"), None)
        pred_gain = round(sort_action.fitness_improvement if sort_action else 0.0, 1)

        # 6. Structurally derived post-remediation fitness and observed gain
        f_post = round(w_train * f_train_eff + w_test * f_test_eff, 1)
        obs_gain = round(f_post - f_initial, 1)
        delta_pred_obs = round(abs(pred_gain - obs_gain), 2)

        fitness_improved = (obs_gain > 0)
        no_contradiction = (model_improved == fitness_improved)
        within_tolerance = (delta_pred_obs <= 10.0)

        # Decision classification
        has_train_red = (report_train.defect_policy and report_train.defect_policy.tier == "RED")
        has_test_red = (report_test.defect_policy and report_test.defect_policy.tier == "RED")
        if has_train_red:
            decision = RemediationDecision.BLOCKED.value
        elif has_test_red:
            decision = RemediationDecision.REVIEW_REQUIRED.value if model_improved else RemediationDecision.BLOCKED.value
        elif model_improved:
            decision = RemediationDecision.ACCEPTED.value
        else:
            decision = RemediationDecision.NO_IMPROVEMENT.value

        results.append({
            "w_train": f"{w_train:.2f}",
            "w_test": f"{w_test:.2f}",
            "F_train": round(f_train_eff, 1),
            "F_train_raw": round(f_train_raw, 1),
            "F_test": round(f_test_eff, 1),
            "F_test_raw": round(f_test_raw, 1),
            "F_post": f_post,
            "Pred_Gain": pred_gain,
            "Obs_Gain": obs_gain,
            "Delta_|P-O|": delta_pred_obs,
            "Within_10pt": within_tolerance,
            "RMSE_Pre": rmse_before,
            "RMSE_Post": rmse_after,
            "RMSE_Imp": model_improved,
            "Fit_Imp": fitness_improved,
            "No_Contr": no_contradiction,
            "Decision": decision
        })

    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False))
    print("\nSensitivity Analysis Verification Conclusions:")
    print(f"- All runs within +-10 pt tolerance: {all(r['Within_10pt'] for r in results)}")
    print(f"- Zero directional contradictions across range: {all(r['No_Contr'] for r in results)}")

    # Save to results/research/sensitivity_results.json
    out_dir = os.path.join(os.path.dirname(__file__), "..", "results", "research")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "sensitivity_results.json")
    import json
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "experiment": "partition_weight_sensitivity",
            "sweep_parameter": "w_train",
            "sweep_range": [0.50, 0.60, 0.70, 0.80, 0.85],
            "structural_discounting": "1.0 - test_defect_fraction",
            "results": results,
            "within_tolerance": all(r['Within_10pt'] for r in results),
            "no_directional_contradictions": all(r['No_Contr'] for r in results)
        }, f, indent=2)
    print(f"\nSaved sensitivity results to: {out_path}")
    return res_df


if __name__ == "__main__":
    run_sensitivity_analysis()
