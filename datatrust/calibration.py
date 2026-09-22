import numpy as np
from typing import Dict, Tuple, Optional
from datatrust.models import (
    TaskType,
    QualityDimension,
    CalibrationResult
)
from datatrust.validator import DownstreamValidator


class PerformanceCalibrationEngine:
    """
    Subsystem for Closed-Loop Downstream Performance Feedback & Calibration.
    Implements the:
    Assessment -> Prediction -> Observation -> Adaptation -> Reassessment cycle.

    Operates directly in the native downstream metric domain:
    - RMSE for Regression & Forecasting
    - F1-Score for Classification
    - Completeness for Descriptive BI

    Mathematical gradient adaptation:
    L(w) = |y_hat(F(w)) - y_obs|
    g_i = sign(y_hat - y_obs) * (dy_hat / dF) * (s_i - mean_s) * M(T, Q_i)
    w_i^(t+1) = Normalize(Clip(w_i^(t) - eta * g_i, 0.02, 0.60))
    """

    @classmethod
    def predict_performance(
        cls,
        trust_score: float,
        task: TaskType,
        target_std: float = 10.0
    ) -> float:
        """Maps dataset fitness to expected metric in the native task domain."""
        _, val = DownstreamValidator.predict_performance(trust_score, task, target_std)
        return val

    @classmethod
    def calibrate(
        cls,
        task: TaskType,
        initial_weights: Dict[QualityDimension, float],
        dimension_scores: Dict[QualityDimension, float],
        initial_trust_score: float,
        observed_performance: float,
        metric_name: Optional[str] = None,
        target_std: float = 10.0,
        learning_rate: float = 1.0,
        sensitivities: Optional[Dict[QualityDimension, float]] = None
    ) -> CalibrationResult:
        """
        Calculates surrogate residual epsilon = observed - surrogate_predicted,
        adapts prior task weights W_AHP via closed-loop gradient descent on loss L(w) = |epsilon|,
        and projects candidate weights onto the probability simplex:
           w_i^(t+1) = Normalize(Clip(w_i^(t) - eta * g_i, w_min, w_max))

        Strict Acceptance Policy:
        An update is accepted ONLY if min_eta L_after < L_before. If no candidate learning
        rate strictly reduces the loss, the update is rejected and initial weights W_AHP are preserved.
        """
        inferred_metric, p_predicted = DownstreamValidator.predict_performance(initial_trust_score, task, target_std)
        if not metric_name:
            metric_name = inferred_metric

        residual_before = round(observed_performance - p_predicted, 2)
        loss_before = round(abs(residual_before), 4)

        # Dimension scores and defect severities d_i in [0, 1]
        scores = {dim: float(dimension_scores.get(dim, initial_trust_score)) for dim in initial_weights}
        mean_score = float(np.mean(list(scores.values()))) if scores else initial_trust_score

        # If already converged (|residual| < 0.02)
        weights_ahp_dict = {dim.value: round(w, 4) for dim, w in initial_weights.items()}
        aliases = {
            "label_integrity": "distribution_balance",
            "outlier_resilience": "outlier_anomaly",
            "missingness": "completeness",
            "drift_stability": "timeliness",
            "temporal_validity": "timeliness",
            "bias_imbalance": "distribution_balance",
            "correlation_integrity": "validity",
            "cardinality_schema": "validity",
            "leakage": "validity"
        }
        for alias_key, canon_key in aliases.items():
            if canon_key in weights_ahp_dict and alias_key not in weights_ahp_dict:
                weights_ahp_dict[alias_key] = weights_ahp_dict[canon_key]

        if loss_before < 0.02:
            adj = {dim.value: {"initial": round(w, 4), "calibrated": round(w, 4), "delta": 0.0} for dim, w in initial_weights.items()}
            for alias_key, canon_key in aliases.items():
                if canon_key in adj and alias_key not in adj:
                    adj[alias_key] = adj[canon_key]
            return CalibrationResult(
                metric_name=metric_name,
                surrogate_predicted_performance=p_predicted,
                predicted_performance=p_predicted,
                observed_performance=observed_performance,
                residual=residual_before,
                residual_before=residual_before,
                residual_after=residual_before,
                loss_before=loss_before,
                loss_after=loss_before,
                residual_reduction_pct=0.0,
                is_improved=False,
                calibration_status="Calibration Converged",
                weights_ahp=weights_ahp_dict,
                weights_before=weights_ahp_dict,
                weights_calibrated=weights_ahp_dict,
                weights_after=weights_ahp_dict,
                weight_adjustments=adj,
                initial_trust_score=initial_trust_score,
                calibrated_trust_score=initial_trust_score,
                calibrated_predicted_performance=p_predicted,
                provenance="NO_IMPROVEMENT",
                explanation=f"Downstream {metric_name} is already converged with surrogate prediction (|epsilon| < 0.02)."
            )

        # 1. Compute surrogate gradient: g_i = dL/dw_i via Chain Rule
        # L(w) = |y_hat(F(w)) - y_obs|
        # dL/dw_i = sign(y_hat - y_obs) * (d y_hat / d F) * (d F / d w_i)
        sign_err = 1.0 if (p_predicted - observed_performance) > 0 else -1.0

        if metric_name == "RMSE":
            # y_hat = target_std * (0.20 + (100 - F) * 0.015)
            # dy_hat / dF = -0.015 * target_std
            dy_df = -0.015 * max(0.5, target_std)
        elif metric_name == "F1":
            # y_hat = (F / 100)^1.15 * 0.94
            # dy_hat / dF > 0
            curr_norm = max(0.01, initial_trust_score / 100.0)
            dy_df = 0.94 * 1.15 * (curr_norm ** 0.15) / 100.0
        else:
            dy_df = 1.0

        # Mean-centered score contribution scaled by task sensitivity M(T, Q)
        gradients: Dict[QualityDimension, float] = {}
        for dim, w in initial_weights.items():
            s_i = scores.get(dim, initial_trust_score)
            sens_weight = sensitivities.get(dim, 1.0) if sensitivities else 1.0
            # Gradient: sign(y_hat - y_obs) * (dy_hat/dF) * (s_i - mean_score) * M(T, Q)
            grad = sign_err * dy_df * (s_i - mean_score) * sens_weight
            gradients[dim] = float(grad)

        # 2. Line-Search over Candidate Learning Rates
        candidate_etas = [0.001, 0.002, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2, 0.4, 0.8, 1.5, 3.0]
        best_calibrated_weights = initial_weights
        best_fitness = initial_trust_score
        best_pred_perf = p_predicted
        best_res_after = residual_before
        best_loss_after = loss_before

        for eta in candidate_etas:
            # Candidate weight update with clipping to [0.02, 0.60]
            raw_new = {}
            for dim, w in initial_weights.items():
                # Gradient descent step: w_new = w - eta * g_i
                w_new = w - eta * gradients[dim]
                raw_new[dim] = max(0.02, min(0.60, w_new))

            # Simplex L1 Normalization: sum(w_i) = 1.0
            tot = sum(raw_new.values())
            cand_weights = {dim: raw_new[dim] / tot for dim in raw_new}

            # Recalculate fitness on simplex
            cand_fitness = sum(cand_weights[dim] * scores.get(dim, initial_trust_score) for dim in cand_weights)
            cand_fitness = float(np.clip(cand_fitness, 0.0, 100.0))
            cand_pred = cls.predict_performance(cand_fitness, task, target_std)
            cand_res = round(observed_performance - cand_pred, 2)
            cand_loss = abs(cand_res)

            if cand_loss < best_loss_after:
                best_loss_after = cand_loss
                best_res_after = cand_res
                best_calibrated_weights = cand_weights
                best_fitness = cand_fitness
                best_pred_perf = cand_pred

        # 3. Strict Acceptance & Rejection Policy
        # An update is accepted ONLY if best_loss_after < loss_before
        is_improved = (best_loss_after < (loss_before - 1e-4))
        if not is_improved:
            cal_status = "Calibration Rejected"
            cal_provenance = "REJECTED"
            best_calibrated_weights = initial_weights
            best_fitness = initial_trust_score
            best_pred_perf = p_predicted
            residual_after = residual_before
            best_loss_after = loss_before
            reduction_pct = 0.0
        else:
            residual_after = round(best_res_after, 2)
            best_fitness = round(best_fitness, 1)
            best_pred_perf = round(best_pred_perf, 2)
            reduction_pct = round(((loss_before - best_loss_after) / max(1e-4, loss_before)) * 100.0, 1)
            reduction_pct = max(0.0, reduction_pct)
            cal_status = "Calibration Improved" if reduction_pct >= 5.0 else "Calibration Accepted"
            cal_provenance = "EMPIRICALLY_CALIBRATED"

        weights_calibrated_dict = {dim.value: round(best_calibrated_weights[dim], 4) for dim in initial_weights}

        weight_adjustments = {}
        for dim in initial_weights:
            w0 = round(initial_weights[dim], 4)
            w1 = round(best_calibrated_weights[dim], 4)
            weight_adjustments[dim.value] = {
                "initial": w0,
                "calibrated": w1,
                "delta": round(w1 - w0, 4)
            }

        for alias_key, canon_key in aliases.items():
            if canon_key in weight_adjustments and alias_key not in weight_adjustments:
                weight_adjustments[alias_key] = weight_adjustments[canon_key]
            if canon_key in weights_calibrated_dict and alias_key not in weights_calibrated_dict:
                weights_calibrated_dict[alias_key] = weights_calibrated_dict[canon_key]

        explanation = (
            f"Observed {metric_name} ({observed_performance:.2f}) diverged from surrogate predicted ({p_predicted:.2f}) "
            f"with initial loss L(w) = |epsilon| = {loss_before:.2f}. "
            f"Optimization status: {cal_status}; "
            f"post-calibration loss reduced from {loss_before:.2f} to {best_loss_after:.2f} "
            f"({reduction_pct:.1f}% reduction). Note: Calibration adjusts surrogate score alignment "
            f"without guaranteeing arbitrary downstream model generalization."
        )

        return CalibrationResult(
            metric_name=metric_name,
            surrogate_predicted_performance=p_predicted,
            predicted_performance=p_predicted,
            observed_performance=observed_performance,
            residual=residual_before,
            residual_before=residual_before,
            residual_after=residual_after,
            loss_before=loss_before,
            loss_after=round(best_loss_after, 4),
            residual_reduction_pct=reduction_pct,
            is_improved=is_improved,
            calibration_status=cal_status,
            weights_ahp=weights_ahp_dict,
            weights_before=weights_ahp_dict,
            weights_calibrated=weights_calibrated_dict,
            weights_after=weights_calibrated_dict,
            weight_adjustments=weight_adjustments,
            initial_trust_score=initial_trust_score,
            calibrated_trust_score=best_fitness,
            calibrated_predicted_performance=best_pred_perf,
            provenance=cal_provenance,
            explanation=explanation
        )
