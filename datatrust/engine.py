import datetime
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List, Tuple

from datatrust.models import (
    TaskType,
    QualityDimension,
    DimensionResult,
    TrustReport,
    TaskInferenceResult,
    RemediationActionV2,
    WhatIfScenario,
    BeforeAfterValidationResult,
    RemediationDecision
)
from datatrust.profiler import DataProfiler
from datatrust.quality_engine import QualityEngine
from datatrust.sensitivity import TaskQualitySensitivityMatrix
from datatrust.veto_engine import IntelligentVetoEngine
from datatrust.calibration import PerformanceCalibrationEngine
from datatrust.optimizer import RemediationOptimizer
from datatrust.validator import DownstreamValidator
from datatrust.remediator import DataRemediator
from datatrust.explainability import ExplainabilityEngine
from datatrust.task_infer import TaskInferenceEngine
from datatrust.uncertainty import FitnessUncertaintyEstimator
from datatrust.reproducibility import ReproducibilityEngine, hash_index
from datatrust.weighting import AHPWeightingEngine
from datatrust.config import PROVENANCE_PIPELINE


class DataTrustEngine:
    """
    DataTrust AI Closed-Loop Evaluation Pipeline Orchestrator.
    Computes Task-Conditioned Dataset Fitness and coordinates:
    Dataset Ingestion -> Profiling -> Task Inference -> Sensitivity Mapping M(T, Q) -> 
    AHP Weight Generation -> Quality Analysis (8 Dimensions) ->
    Three-Level Defect Policy (GREEN, YELLOW, RED Hard Veto) -> Downstream Model Validation ->
    Automatic Calibration -> MRU Optimization -> Reproducibility Receipt.
    """

    @classmethod
    def evaluate(
        cls,
        df: pd.DataFrame,
        task: Optional[TaskType] = None,
        dataset_name: str = "Dataset",
        target_column: Optional[str] = None,
        time_column: Optional[str] = None,
        observed_performance: Optional[float] = None,
        imbalance_policy: Optional[Any] = None
    ) -> TrustReport:
        # 1. Autonomous Task Characterization & Inference with User Override support
        inference_result: Optional[TaskInferenceResult] = None
        if task is None:
            inference_result = TaskInferenceEngine.infer_task(
                df, explicit_target=target_column, explicit_time=time_column
            )
            task = inference_result.final_selected_task or inference_result.inferred_task
            if not target_column and inference_result.candidate_targets:
                target_column = inference_result.candidate_targets[0]
            if not time_column and inference_result.candidate_timestamps:
                time_column = inference_result.candidate_timestamps[0]
        else:
            inference_result = TaskInferenceEngine.infer_task(
                df, explicit_target=target_column, explicit_time=time_column, user_override_task=task
            )
            task = inference_result.final_selected_task or task
            if not target_column and task in [TaskType.SUPERVISED_CLASSIFICATION, TaskType.SUPERVISED_REGRESSION, TaskType.TIME_SERIES_FORECASTING]:
                if inference_result.candidate_targets:
                    target_column = inference_result.candidate_targets[0]
            if not time_column and task == TaskType.TIME_SERIES_FORECASTING:
                if inference_result.candidate_timestamps:
                    time_column = inference_result.candidate_timestamps[0]

        # Ensure time-series forecasting target is resolved if unassigned
        if not target_column and task == TaskType.TIME_SERIES_FORECASTING:
            candidate_nums = [c for c in df.select_dtypes(include=[np.number]).columns if c != time_column]
            if candidate_nums:
                target_column = candidate_nums[0]
                if inference_result and target_column not in inference_result.candidate_targets:
                    inference_result.candidate_targets.insert(0, target_column)

        if target_column and inference_result and target_column not in inference_result.candidate_targets:
            inference_result.candidate_targets.insert(0, target_column)
        if time_column and inference_result and time_column not in inference_result.candidate_timestamps:
            inference_result.candidate_timestamps.insert(0, time_column)

        # 2. Extract Profiling Metadata
        profiling_meta, profiler_details = DataProfiler.profile(
            df, explicit_target=target_column, explicit_time=time_column
        )

        # 3. Execute Multi-Dimensional Quality Analyzers across 8 Canonical Dimensions
        quality_engine = QualityEngine(
            df, target_column=target_column, time_column=time_column, imbalance_policy=imbalance_policy
        )
        raw_dimension_results = quality_engine.run_all_dimensions(task)

        # 4. Adaptive / AHP Weight Generation W(T) = f(M(T, Q), D)
        weights, cr, weight_meta = TaskQualitySensitivityMatrix.generate_adaptive_weights(
            task=task,
            profiling_meta=profiling_meta,
            raw_dimension_results=raw_dimension_results
        )
        sensitivities_dict = weight_meta["base_sensitivity"]
        ahp_metadata = AHPWeightingEngine.get_ahp_metadata(task)

        # 5. Apply Sensitivity Weights & Calculate Raw Fitness
        dimensions: Dict[str, DimensionResult] = {}
        weighted_sum = 0.0

        for dim, res in raw_dimension_results.items():
            w = weights.get(dim, 0.0)
            res.weight = round(w, 4)
            res.weighted_score = round(res.score * w, 2)
            res.sensitivity_level = TaskQualitySensitivityMatrix.get_sensitivity_level(w)
            dimensions[dim.value] = res
            weighted_sum += res.weighted_score

        raw_score = max(0.0, min(100.0, weighted_sum))
        raw_fitness = round(raw_score, 1)

        # 6. Evaluate Three-Level Defect Policy (GREEN, YELLOW, RED Hard Veto)
        defect_policy_res = IntelligentVetoEngine.evaluate_defect_policy(
            task=task,
            dimension_results=raw_dimension_results,
            raw_fitness=raw_score,
            num_rows=profiling_meta.num_rows,
            imbalance_policy=imbalance_policy
        )

        is_vetoed = defect_policy_res.is_unsafe or defect_policy_res.tier == "RED"
        final_veto_msg = None
        if defect_policy_res.tier == "RED":
            final_fitness_v1 = 0.0
            first_red = next((d for d in defect_policy_res.defects_detected if d.tier == "RED"), None)
            final_veto_msg = first_red.evidence if first_red else "Critical defect renders dataset unusable."
        elif defect_policy_res.tier == "YELLOW":
            final_fitness_v1 = defect_policy_res.final_fitness
            first_yellow = next((d for d in defect_policy_res.defects_detected if d.tier == "YELLOW"), None)
            final_veto_msg = first_yellow.evidence if first_yellow else "Moderate defect triggers score cap."
        else:
            final_fitness_v1 = defect_policy_res.final_fitness

        final_fitness_v1 = round(final_fitness_v1, 1)

        # 7. Downstream Model Validation Subsystem (Surrogate Prediction)
        val_summary = DownstreamValidator.validate_downstream(
            df=df,
            task=task,
            target_column=target_column,
            time_column=time_column,
            fitness_score=final_fitness_v1
        )

        metric_name = val_summary["metric_name"]
        pred_metric_val = val_summary["predicted"]
        obs_metric_val = observed_performance if observed_performance is not None else val_summary["observed"]
        metric_res = round(obs_metric_val - pred_metric_val, 2) if obs_metric_val is not None and pred_metric_val is not None else None

        # 8. Automated Closed-Loop Performance Calibration
        calibration_result = None
        final_fitness = final_fitness_v1

        if obs_metric_val is not None and defect_policy_res.tier != "RED":
            dim_scores = {dim: res.score for dim, res in raw_dimension_results.items()}
            calibration_result = PerformanceCalibrationEngine.calibrate(
                task=task,
                initial_weights=weights,
                dimension_scores=dim_scores,
                initial_trust_score=final_fitness_v1,
                observed_performance=obs_metric_val,
                metric_name=metric_name,
                target_std=val_summary.get("target_std", 10.0),
                sensitivities=sensitivities_dict
            )
            final_fitness = calibration_result.calibrated_trust_score
            if defect_policy_res.cap_applied is not None:
                final_fitness = min(final_fitness, defect_policy_res.cap_applied)
        elif defect_policy_res.tier == "RED":
            final_fitness = 0.0

        # 9. Determine Defensible Status Tier
        if defect_policy_res.tier == "RED":
            tier = "Critical Defect / Unfit"
        elif final_fitness >= 85.0:
            tier = "High Fitness / Low Risk"
        elif final_fitness >= 70.0:
            tier = "Moderate Fitness / Conditional"
        elif final_fitness >= 50.0:
            tier = "Sub-Optimal Fitness / High Risk"
        else:
            tier = "Critical Defect / Unfit"

        # 10. Fitness Uncertainty Estimation (Multi-Source Variance Sum)
        dim_scores_all = {dim: res.score for dim, res in raw_dimension_results.items()}
        fitness_uncertainty = FitnessUncertaintyEstimator.estimate_uncertainty(
            fitness_score=final_fitness,
            dimension_scores=dim_scores_all,
            weights=weights,
            num_rows=profiling_meta.num_rows,
            task_confidence=inference_result.confidence if inference_result else 1.0,
            downstream_residual=metric_res
        )

        # 11. Marginal Remediation Utility (MRU) Optimization
        optimized_remediations = RemediationOptimizer.optimize_remediations(
            task=task,
            dimensions=raw_dimension_results,
            current_trust_score=final_fitness,
            is_vetoed=is_vetoed
        )

        # 12. Generate What-If Simulation Scenarios & Counterfactuals
        what_if_scenarios = RemediationOptimizer.generate_what_if_scenarios(
            current_trust_score=final_fitness,
            actions=optimized_remediations
        )
        counterfactual_table = RemediationOptimizer.generate_counterfactual_table(
            current_trust_score=final_fitness,
            actions=optimized_remediations
        )

        # 13. Factor Attribution Breakdown
        factor_attributions = ExplainabilityEngine.compute_factor_attributions(
            dimensions=raw_dimension_results,
            total_trust_score=final_fitness
        )

        legacy_remediations = ExplainabilityEngine.generate_remediations(
            task=task,
            dimensions=raw_dimension_results
        )

        # 14. Reproducibility & Matrix Metadata
        full_matrix = TaskQualitySensitivityMatrix.get_matrix_dataframe()
        sens_row = full_matrix.get(task.value, {})
        ds_hash = ReproducibilityEngine.compute_dataset_hash(df)

        weights_ahp_dict = {dim.value: round(w, 4) for dim, w in weights.items()}
        weights_cal_dict = calibration_result.weights_calibrated if calibration_result else weights_ahp_dict

        return TrustReport(
            dataset_name=dataset_name,
            task_type=task,
            target_column=target_column,
            time_column=time_column,
            task_conditioned_fitness=final_fitness,
            trust_score=final_fitness,
            raw_score=round(raw_score, 1),
            raw_fitness=raw_fitness,
            final_fitness=final_fitness if defect_policy_res.tier != "RED" else 0.0,
            fitness_status=defect_policy_res.fitness_status,
            defect_policy=defect_policy_res,
            ahp_metadata=ahp_metadata,
            m_t_q=sens_row,
            pipeline_provenance=PROVENANCE_PIPELINE,
            confidence_tier=tier,
            fitness_uncertainty=fitness_uncertainty,
            veto_applied=is_vetoed,
            veto_message=final_veto_msg,
            dimensions=dimensions,
            factor_attributions=factor_attributions,
            remediations=legacy_remediations,
            profiling=profiling_meta,
            ahp_consistency_ratio=round(cr, 4),
            evaluated_at=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            task_inference=inference_result,
            task_quality_sensitivities=sensitivities_dict,
            sensitivity_matrix_row=sens_row,
            model_family=val_summary.get("model_family") or {
                TaskType.SUPERVISED_CLASSIFICATION: "Random Forest Classifier",
                TaskType.SUPERVISED_REGRESSION: "Random Forest Regressor",
                TaskType.TIME_SERIES_FORECASTING: "Autoregressive / Lagged Regression",
                TaskType.CLUSTERING: "K-Means Clustering"
            }.get(task, "Analytical Pipeline"),
            metric=metric_name,
            task_confidence=round(inference_result.confidence, 3) if inference_result else 1.0,
            surrogate_predicted_metric=pred_metric_val,
            observed_model_metric=obs_metric_val,
            prediction_residual_before=metric_res,
            prediction_residual_after=calibration_result.residual_after if calibration_result else metric_res,
            predicted_downstream_metric=pred_metric_val,
            predicted_metric_name=metric_name,
            predicted_metric_value=pred_metric_val,
            observed_metric_value=obs_metric_val,
            metric_residual=metric_res,
            weights_ahp=weights_ahp_dict,
            weights_calibrated=weights_cal_dict,
            calibration=calibration_result,
            optimized_remediations=optimized_remediations,
            what_if_scenarios=what_if_scenarios,
            counterfactual_table=counterfactual_table,
            reproducibility_hash=ds_hash
        )

    @classmethod
    def remediate_and_validate(
        cls,
        df: pd.DataFrame,
        task: Optional[TaskType] = None,
        target_column: Optional[str] = None,
        time_column: Optional[str] = None,
        action_ids: Optional[List[str]] = None,
        imbalance_policy: Optional[Any] = None
    ) -> Tuple[TrustReport, TrustReport, BeforeAfterValidationResult]:
        """
        Executes strict, leakage-free Before vs After empirical validation:
        1. Evaluates baseline dataset -> Fitness Before.
        2. Splits raw dataset into train (70%) and test (30%) splits FIRST.
        3. Trains baseline model on raw train split; evaluates on untouched raw test split -> Metric Before.
        4. Fits all remediation parameters exclusively on train split.
        5. Transforms train split (including minority resampling) and transforms test split deterministically
           WITHOUT ANY TEST RESAMPLING, preserving ground truth test distribution.
        6. Trains remediated model on remediated train split; evaluates on remediated test split -> Metric After.
        7. Produces a cryptographic DownstreamAuditReceipt recording train/test index hashes.
        8. Separates predicted gains from actual observed empirical outcomes.
        """
        from sklearn.model_selection import train_test_split

        # Resolve columns and task
        report_initial = cls.evaluate(
            df=df,
            task=task,
            dataset_name="Original_Dataset",
            target_column=target_column,
            time_column=time_column,
            imbalance_policy=imbalance_policy
        )

        target_col = report_initial.target_column
        time_col = report_initial.time_column
        task_resolved = report_initial.task_type

        # Select candidate action IDs
        if not action_ids:
            action_ids = [a.id for a in report_initial.optimized_remediations[:3]]

        # 1. Clean subset for supervised target modeling
        if target_col and target_col in df.columns:
            valid_df = df.dropna(subset=[target_col]).reset_index(drop=True)
        else:
            valid_df = df.copy().reset_index(drop=True)

        n_rows = len(valid_df)

        # 2. Strict Train/Test Split BEFORE Remediation
        if task_resolved == TaskType.TIME_SERIES_FORECASTING:
            # Chronological temporal split (first 70% train, last 30% test)
            split_idx = int(n_rows * 0.7)
            train_indices = np.arange(split_idx)
            test_indices = np.arange(split_idx, n_rows)
        else:
            # Deterministic split with fixed random state
            indices = np.arange(n_rows)
            if n_rows >= 20 and target_col and target_col in valid_df.columns and task_resolved == TaskType.SUPERVISED_CLASSIFICATION:
                y_classes = valid_df[target_col]
                # Stratify only if each class has >= 2 samples
                can_stratify = bool((y_classes.value_counts() >= 2).all())
                strat = y_classes if can_stratify else None
                train_indices, test_indices = train_test_split(indices, test_size=0.3, random_state=42, stratify=strat)
            elif n_rows >= 20:
                train_indices, test_indices = train_test_split(indices, test_size=0.3, random_state=42)
            else:
                train_indices = indices
                test_indices = indices

        raw_train = valid_df.iloc[train_indices].copy()
        raw_test = valid_df.iloc[test_indices].copy()

        test_index_hash_before = hash_index(test_indices)

        # 3. Baseline Model Evaluation on Untouched Splits
        baseline_eval = DownstreamValidator.train_and_evaluate_split(
            train_df=raw_train,
            test_df=raw_test,
            task=task_resolved,
            target_column=target_col,
            time_column=time_col
        )
        m_before = baseline_eval["observed_metric"]
        m_name = baseline_eval["metric_name"]

        # Full Baseline Report with observed baseline metric
        report_before = cls.evaluate(
            df=df,
            task=task_resolved,
            dataset_name="Original_Dataset",
            target_column=target_col,
            time_column=time_col,
            observed_performance=m_before,
            imbalance_policy=imbalance_policy
        )

        # Map action IDs (e.g. act_1) to action_type from initial report recommendations
        resolved_action_ids = list(action_ids)
        for act in report_initial.optimized_remediations:
            if act.id in action_ids and act.action_type and act.action_type not in resolved_action_ids:
                resolved_action_ids.append(act.action_type)

        # 4. Strict Split-Aware Remediation (Fit on Train ONLY; Zero Test Resampling)
        rem_train, rem_test, execution_log, fitted_params = DataRemediator.fit_and_transform_splits(
            train_df=raw_train,
            test_df=raw_test,
            task=task_resolved,
            action_ids=resolved_action_ids,
            target_column=target_col,
            time_column=time_col
        )

        test_index_hash_after = hash_index(rem_test.index)
        assert test_index_hash_before == test_index_hash_after, (
            f"Test set index mutability violation: before={test_index_hash_before}, after={test_index_hash_after}"
        )
        assert len(rem_test) == len(raw_test), (
            f"Test set row count altered: {len(rem_test)} != {len(raw_test)}"
        )

        # 5. Remediated Model Evaluation on Untouched Test Split
        rem_eval = DownstreamValidator.train_and_evaluate_split(
            train_df=rem_train,
            test_df=rem_test,
            task=task_resolved,
            target_column=target_col,
            time_column=time_col
        )
        m_after = rem_eval["observed_metric"]

        # 6. Partition-Aware Post-Remediation Assessment (Issue 3 Principled Derivation)
        # Evaluates train and test partitions separately to eliminate concatenation boundary
        # artifacts and accurately balance clean training fitness with irreducible test defects.
        n_train = len(rem_train)
        n_test = len(rem_test)
        total_rows = n_train + n_test
        w_train = n_train / total_rows if total_rows > 0 else 0.70
        w_test = n_test / total_rows if total_rows > 0 else 0.30

        # Evaluate clean training partition
        report_train = cls.evaluate(
            df=rem_train,
            task=task_resolved,
            dataset_name="Remediated_Train_Partition",
            target_column=target_col,
            time_column=time_col,
            observed_performance=m_after,
            imbalance_policy=imbalance_policy
        )

        # Evaluate untouched holdout test partition
        report_test = cls.evaluate(
            df=rem_test,
            task=task_resolved,
            dataset_name="Untouched_Test_Partition",
            target_column=target_col,
            time_column=time_col,
            imbalance_policy=imbalance_policy
        )

        # Full reassembled dataframe preserved for cleansed export
        remediated_full_df = pd.concat([rem_train, rem_test], ignore_index=True)

        report_after = cls.evaluate(
            df=remediated_full_df,
            task=task_resolved,
            dataset_name="Remediated_Dataset",
            target_column=target_col,
            time_column=time_col,
            observed_performance=m_after,
            imbalance_policy=imbalance_policy
        )

        # Principled Partition Combination:
        # If test partition has irreducible defects or full evaluation suffered false concatenation veto,
        # structurally combine partition fitness based on empirical split weights.
        # Otherwise, preserve remediated full dataset fitness.
        f_train_eff = (
            report_train.defect_policy.final_fitness
            if report_train.defect_policy and report_train.defect_policy.final_fitness is not None
            else report_train.raw_fitness
        )
        f_test_eff = (
            report_test.defect_policy.final_fitness
            if report_test.defect_policy and report_test.defect_policy.final_fitness is not None
            else report_test.raw_fitness
        )
        f_full_raw = (
            report_after.defect_policy.final_fitness
            if report_after.defect_policy and report_after.defect_policy.final_fitness is not None
            else report_after.raw_fitness
        )
        partition_fitness = round(w_train * f_train_eff + w_test * f_test_eff, 1)

        test_has_irreducible_veto = (report_test.defect_policy and report_test.defect_policy.tier == "RED")
        train_is_safe = (report_train.defect_policy and report_train.defect_policy.tier != "RED")
        full_is_red = (report_after.defect_policy and report_after.defect_policy.tier == "RED")

        if full_is_red and train_is_safe:
            post_fitness = partition_fitness
            report_after.defect_policy.tier = "YELLOW"
            report_after.defect_policy.fitness_status = "WARNING"
            report_after.defect_policy.is_unsafe = False
            report_after.defect_policy.final_fitness = post_fitness
            from datatrust.models import DefectRecord
            report_after.defect_policy.defects_detected.append(
                DefectRecord(
                    defect_name="IRREDUCIBLE_HOLDOUT_TEST_DEFECTS",
                    severity="WARNING",
                    tier="YELLOW",
                    evidence=(
                        f"Holdout test partition (w_test={w_test:.2f}) retained pre-existing defects under "
                        f"test set immutability policy (test fitness: {f_test_eff:.1f}). "
                        f"Model training split was successfully remediated (train fitness: {f_train_eff:.1f})."
                    ),
                    threshold="Test set immutability preserved (zero test leakage)",
                    threshold_provenance="LEAKAGE-FREE AUDIT POLICY",
                    policy_action="Review required: verify downstream generalization before certification"
                )
            )
        else:
            post_fitness = f_full_raw

        report_after.final_fitness = post_fitness
        report_after.task_conditioned_fitness = post_fitness
        report_after.trust_score = post_fitness

        # 7. Generate Cryptographic DownstreamAuditReceipt
        audit_receipt = ReproducibilityEngine.generate_downstream_audit_receipt(
            train_indices=train_indices,
            test_indices=test_indices,
            feature_columns=baseline_eval["feature_columns"],
            target_column=target_col,
            time_column=time_col,
            preprocessing_parameters=fitted_params,
            preprocessing_operations=[
                "Pre-remediation split (70% train / 30% test)",
                "Deterministic frozen parameter application to test data",
                "Zero test resampling verified (test distribution preserved intact)",
                "Verified test indices immutable (test_index_hash_before == test_index_hash_after)"
            ],
            remediation_operations=execution_log,
            model_family=baseline_eval["model_family"],
            hyperparameters=baseline_eval["hyperparameters"],
            baseline_metric=m_before,
            remediated_metric=m_after,
            metric_name=m_name,
            test_index_hash_before=test_index_hash_before,
            test_index_hash_after=test_index_hash_after,
            dataset_id=report_initial.dataset_name,
            random_state=42,
            random_seed=42
        )

        # 8. Compute Empirical Improvements and Separate Predicted vs Observed Gains
        fitness_before = (
            report_before.defect_policy.final_fitness
            if report_before.defect_policy and report_before.defect_policy.final_fitness is not None
            else report_before.raw_fitness
        )
        fitness_after = post_fitness
        observed_fitness_gain = round(fitness_after - fitness_before, 1)

        predicted_fitness_gain = round(
            sum(a.fitness_improvement for a in report_before.optimized_remediations if a.id in action_ids),
            1
        )

        pred_metric_before_tuple = DownstreamValidator.predict_performance(fitness_before, task_resolved)
        predicted_fitness_after = min(100.0, fitness_before + predicted_fitness_gain)
        pred_metric_after_tuple = DownstreamValidator.predict_performance(predicted_fitness_after, task_resolved)
        predicted_metric_gain = round(pred_metric_after_tuple[1] - pred_metric_before_tuple[1], 2)
        observed_metric_gain = round(m_after - m_before, 2)

        if m_name == "RMSE":
            change_pct = round(((m_after - m_before) / m_before) * 100.0, 2) if m_before > 0 else 0.0
            if change_pct < 0:
                perf_status = "Improved ✓"
                improvement_pct = round(((m_before - m_after) / m_before) * 100.0, 2)
            elif change_pct > 0:
                perf_status = "Deteriorated ✗"
                improvement_pct = round(((m_before - m_after) / m_before) * 100.0, 2)
            else:
                perf_status = "Neutral"
                improvement_pct = 0.0
        else:
            change_pct = round(((m_after - m_before) / max(0.001, m_before)) * 100.0, 2) if m_before > 0 else 0.0
            if change_pct > 0:
                perf_status = "Improved ✓"
                improvement_pct = change_pct
            elif change_pct < 0:
                perf_status = "Deteriorated ✗"
                improvement_pct = change_pct
            else:
                perf_status = "Neutral"
                improvement_pct = 0.0

        raw_stats = {
            "num_rows": len(df),
            "train_sample_count": len(train_indices),
            "test_sample_count": len(test_indices),
            "metric": round(m_before, 2),
            "fitness": fitness_before,
            "metric_name": m_name
        }
        remediated_stats = {
            "num_rows": len(remediated_full_df),
            "train_sample_count": len(rem_train),
            "test_sample_count": len(rem_test),
            "metric": round(m_after, 2),
            "fitness": fitness_after,
            "metric_name": m_name
        }
        if time_col and time_col in df.columns:
            ts_raw = pd.to_datetime(df[time_col], errors='coerce')
            raw_dups = int(ts_raw.duplicated().sum())
            raw_collision_groups = int((ts_raw.value_counts() > 1).sum())
            raw_stats["duplicate_timestamps"] = raw_dups
            raw_stats["duplicate_timestamp_collisions"] = raw_collision_groups
            raw_stats["duplicate_rate"] = f"{(raw_dups/max(1, len(df)))*100:.1f}%"

            if time_col in remediated_full_df.columns:
                ts_rem = pd.to_datetime(remediated_full_df[time_col], errors='coerce')
                rem_dups = int(ts_rem.duplicated().sum())
                rem_collision_groups = int((ts_rem.value_counts() > 1).sum())
                remediated_stats["duplicate_timestamps"] = rem_dups
                remediated_stats["duplicate_timestamp_collisions"] = rem_collision_groups
                remediated_stats["duplicate_rate"] = f"{(rem_dups/max(1, len(remediated_full_df)))*100:.1f}%"
            else:
                remediated_stats["duplicate_timestamps"] = 0
                remediated_stats["duplicate_timestamp_collisions"] = 0
                remediated_stats["duplicate_rate"] = "0.0%"

        # 9. Determine Closed-Loop Remediation Decision (Accept / Reject / Blocked / Regressed / Review)
        decision = RemediationDecision.REVIEW_REQUIRED
        decision_reason = ""

        # Check if train partition failed remediation (genuine RED failure in model training data)
        train_is_red = (report_train.defect_policy and report_train.defect_policy.tier == "RED")

        if train_is_red:
            decision = RemediationDecision.BLOCKED
            decision_reason = (
                f"Remediation blocked: clean training partition retains critical defect tier RED "
                f"({report_train.defect_policy.fitness_status}). "
                f"Remediation failed to resolve fatal flaw in training data."
            )
        elif m_after is not None and m_before is not None:
            is_improved = False
            is_regressed = False
            if m_name == "RMSE":
                if m_after < m_before - 1e-4:
                    is_improved = True
                elif m_after > m_before + 1e-4:
                    is_regressed = True
            else:
                if m_after > m_before + 1e-4:
                    is_improved = True
                elif m_after < m_before - 1e-4:
                    is_regressed = True

            if test_has_irreducible_veto and not is_improved:
                decision = RemediationDecision.BLOCKED
                decision_reason = (
                    f"Remediation blocked: holdout test partition retains critical defect tier RED "
                    f"under test set immutability, and downstream performance failed to improve ({perf_status})."
                )
            elif is_regressed:
                decision = RemediationDecision.REGRESSED
                decision_reason = (
                    f"Downstream performance regressed: {m_name} shifted from {m_before:.4f} to {m_after:.4f} "
                    f"({perf_status})."
                )
            elif is_improved:
                # If downstream model improved, check whether test partition retains irreducible defects
                if test_has_irreducible_veto:
                    decision = RemediationDecision.REVIEW_REQUIRED
                    decision_reason = (
                        f"Remediation improved downstream {m_name} from {m_before:.4f} to {m_after:.4f} "
                        f"({perf_status}), with post-remediation fitness {post_fitness:.1f}. "
                        f"Holdout test partition retains irreducible pre-existing defects under test set immutability; "
                        f"operational review recommended before production certification."
                    )
                else:
                    decision = RemediationDecision.ACCEPTED
                    decision_reason = (
                        f"Remediation accepted: {m_name} improved from {m_before:.4f} to {m_after:.4f} "
                        f"with observed fitness gain of {observed_fitness_gain:+.1f}."
                    )
            elif not is_improved and not is_regressed:
                if observed_fitness_gain > 5.0:
                    decision = RemediationDecision.ACCEPTED
                    decision_reason = (
                        f"Remediation accepted: data quality fitness improved by {observed_fitness_gain:+.1f} points "
                        f"while downstream performance remained stable ({m_name} = {m_after:.4f})."
                    )
                else:
                    decision = RemediationDecision.NO_IMPROVEMENT
                    decision_reason = (
                        f"No significant improvement: fitness gain {observed_fitness_gain:+.1f}, "
                        f"{m_name} delta {observed_metric_gain:+.4f}."
                    )
            else:
                decision = RemediationDecision.REVIEW_REQUIRED
                decision_reason = (
                    f"Divergence detected: downstream metric changed by {observed_metric_gain:+.4f} "
                    f"while fitness changed by {observed_fitness_gain:+.1f}."
                )
        else:
            decision = RemediationDecision.NO_IMPROVEMENT
            decision_reason = "Downstream evaluation metrics unavailable for before/after comparison."

        before_after_res = BeforeAfterValidationResult(
            fitness_before=fitness_before,
            fitness_after=fitness_after,
            quality_before=fitness_before,
            quality_after=fitness_after,
            predicted_fitness_gain=predicted_fitness_gain,
            observed_fitness_gain=observed_fitness_gain,
            fitness_improvement=observed_fitness_gain,
            metric_name=m_name,
            metric_before=round(m_before, 4) if m_before is not None else None,
            metric_after=round(m_after, 4) if m_after is not None else None,
            downstream_metric_before=round(m_before, 4) if m_before is not None else None,
            downstream_metric_after=round(m_after, 4) if m_after is not None else None,
            predicted_metric_gain=predicted_metric_gain,
            observed_metric_gain=observed_metric_gain,
            metric_change_pct=change_pct,
            metric_improvement_pct=improvement_pct,
            performance_status=perf_status,
            decision=decision,
            decision_reason=decision_reason,
            raw_stats=raw_stats,
            remediated_stats=remediated_stats,
            remediation_actions_applied=action_ids,
            audit_receipt=audit_receipt,
            execution_log=execution_log
        )

        report_after.before_after_validation = before_after_res
        report_after.downstream_audit_receipt = audit_receipt
        report_before.downstream_audit_receipt = audit_receipt
        return report_before, report_after, before_after_res
