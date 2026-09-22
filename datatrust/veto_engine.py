from typing import Dict, Optional, Tuple, Any, List
from datatrust.models import TaskType, QualityDimension, DimensionResult, DefectRecord, DefectPolicyResult
from datatrust.config import (
    DEFECT_POLICY_CONFIG,
    PROVENANCE_DEFECT_POLICY,
    DEFECT_TIER_GREEN,
    DEFECT_TIER_YELLOW,
    DEFECT_TIER_RED
)


class IntelligentVetoEngine:
    """
    Subsystem for the Three-Level Defect Policy:
    1. GREEN: Normal operating state. No fatal defects. Final fitness = Raw fitness.
    2. YELLOW: Conditional warning / score ceiling (capped at 60.0-70.0).
    3. RED: Hard non-compensatory veto. Fatal defect renders dataset unusable.
       Status = UNSAFE, is_unsafe = True, final_fitness = 0.0 (or None).
    Provenance: HEURISTIC / POLICY PRIOR.
    """

    @classmethod
    def evaluate_defect_policy(
        cls,
        task: TaskType,
        dimension_results: Dict[QualityDimension, DimensionResult],
        raw_fitness: float,
        num_rows: int,
        imbalance_policy: Optional[Any] = None
    ) -> DefectPolicyResult:
        resolved_task = task if task != TaskType.DESCRIPTIVE_BI else TaskType.CLUSTERING
        defects: List[DefectRecord] = []

        cmp_res = dimension_results.get(QualityDimension.COMPLETENESS)
        val_res = dimension_results.get(QualityDimension.VALIDITY)
        cons_res = dimension_results.get(QualityDimension.CONSISTENCY)
        uniq_res = dimension_results.get(QualityDimension.UNIQUENESS)
        time_res = dimension_results.get(QualityDimension.TIMELINESS)
        out_res = dimension_results.get(QualityDimension.OUTLIER_ANOMALY)
        dist_res = dimension_results.get(QualityDimension.DISTRIBUTION_BALANCE)

        # 1. COMPLETENESS DEFECTS
        if cmp_res:
            missing_rate = cmp_res.raw_metrics.get("missing_rate", 0.0)
            if missing_rate > 0.70 or cmp_res.score < 25.0:
                defects.append(DefectRecord(
                    defect_name="CATASTROPHIC_MISSINGNESS",
                    severity="CRITICAL",
                    tier=DEFECT_TIER_RED,
                    evidence=f"Dataset missing cell rate is {missing_rate*100:.1f}% (completeness score {cmp_res.score:.1f})",
                    threshold="missing_rate <= 0.70",
                    threshold_provenance=PROVENANCE_DEFECT_POLICY,
                    policy_action="Hard Veto: dataset unusable. status=UNSAFE, final_fitness=0.0."
                ))
            elif missing_rate > 0.30 or cmp_res.score < 60.0:
                defects.append(DefectRecord(
                    defect_name="HIGH_MISSINGNESS",
                    severity="WARNING",
                    tier=DEFECT_TIER_YELLOW,
                    evidence=f"Elevated missing cell rate: {missing_rate*100:.1f}%",
                    threshold="missing_rate <= 0.30",
                    threshold_provenance=PROVENANCE_DEFECT_POLICY,
                    policy_action="Cap fitness at 65.0."
                ))

        # 2. VALIDITY DEFECTS
        if val_res and getattr(val_res, "is_veto_triggered", False):
            defects.append(DefectRecord(
                defect_name="INVALID_SCHEMA_OR_TARGET",
                severity="CRITICAL",
                tier=DEFECT_TIER_RED,
                evidence=getattr(val_res, "veto_reason", None) or "Target or schema invalid for task",
                threshold="valid target and column schema conforming to task specification",
                threshold_provenance=PROVENANCE_DEFECT_POLICY,
                policy_action="Hard Veto: invalid target/schema. status=UNSAFE, final_fitness=0.0."
            ))

        # 3. CONSISTENCY / TARGET LEAKAGE DEFECTS
        if cons_res:
            leakage_suspects = cons_res.raw_metrics.get("target_leakage_suspects", [])
            if leakage_suspects:
                feat_names = [s["feature"] for s in leakage_suspects]
                defects.append(DefectRecord(
                    defect_name="TARGET_LEAKAGE",
                    severity="CRITICAL",
                    tier=DEFECT_TIER_RED,
                    evidence=f"Post-outcome target leakage detected in features: {feat_names} (correlation > 0.98)",
                    threshold="pearson_correlation(feature, target) < 0.98",
                    threshold_provenance=PROVENANCE_DEFECT_POLICY,
                    policy_action="Hard Veto: target leakage detected. status=UNSAFE, final_fitness=0.0."
                ))
            elif len(cons_res.raw_metrics.get("collinear_pairs", [])) > 6:
                defects.append(DefectRecord(
                    defect_name="SEVERE_MULTICOLLINEARITY",
                    severity="WARNING",
                    tier=DEFECT_TIER_YELLOW,
                    evidence=f"{len(cons_res.raw_metrics.get('collinear_pairs', []))} collinear feature pairs detected",
                    threshold="collinear_pairs <= 6",
                    threshold_provenance=PROVENANCE_DEFECT_POLICY,
                    policy_action="Cap fitness at 70.0."
                ))

        # 4. TIMELINESS / TEMPORAL ORDER DEFECTS
        if resolved_task == TaskType.TIME_SERIES_FORECASTING and time_res:
            raw_time = time_res.raw_metrics
            is_monotonic = raw_time.get("is_monotonic_increasing", True)
            dup_timestamps = raw_time.get("duplicate_count", raw_time.get("duplicate_timestamps", 0))
            if not is_monotonic or dup_timestamps > 0 or time_res.is_veto_triggered:
                defects.append(DefectRecord(
                    defect_name="TEMPORAL_SEQUENCE_DISORDER",
                    severity="CRITICAL",
                    tier=DEFECT_TIER_RED,
                    evidence=(
                        f"Non-chronological order (monotonic={is_monotonic}) or "
                        f"{dup_timestamps} duplicate timestamp collisions detected"
                    ),
                    threshold="is_monotonic_increasing=True and duplicate_timestamps=0",
                    threshold_provenance=PROVENANCE_DEFECT_POLICY,
                    policy_action="Hard Veto: time-series sequence broken. status=UNSAFE, final_fitness=0.0."
                ))
        elif time_res:
            drifted_cols = time_res.raw_metrics.get("drifted_columns", [])
            if len(drifted_cols) >= 3:
                defects.append(DefectRecord(
                    defect_name="PARTITION_DISTRIBUTION_DRIFT",
                    severity="WARNING",
                    tier=DEFECT_TIER_YELLOW,
                    evidence=f"{len(drifted_cols)} numeric features show significant partition shift (>0.50 pooled std)",
                    threshold="drifted_columns < 3",
                    threshold_provenance=PROVENANCE_DEFECT_POLICY,
                    policy_action="Cap fitness at 70.0."
                ))

        # 5. DISTRIBUTION BALANCE / CLASS COLLAPSE DEFECTS
        if resolved_task == TaskType.SUPERVISED_CLASSIFICATION and dist_res:
            min_thresh = 2.0
            veto_active = True
            if imbalance_policy is not None:
                min_thresh = getattr(imbalance_policy, "minority_threshold_pct", 2.0)
                veto_active = getattr(imbalance_policy, "veto_enabled", True)

            min_pct = dist_res.raw_metrics.get("minority_percentage", 50.0)
            maj_pct = dist_res.raw_metrics.get("majority_percentage", 50.0)
            if veto_active and (min_pct < min_thresh or maj_pct >= 98.0 or dist_res.is_veto_triggered):
                defects.append(DefectRecord(
                    defect_name="SEVERE_CLASS_COLLAPSE",
                    severity="CRITICAL",
                    tier=DEFECT_TIER_RED,
                    evidence=f"Minority class represents {min_pct:.2f}% (<{min_thresh:.1f}% threshold, majority={maj_pct:.1f}%)",
                    threshold=f"minority_percentage >= {min_thresh:.1f}%",
                    threshold_provenance=PROVENANCE_DEFECT_POLICY,
                    policy_action="Hard Veto: class collapse. status=UNSAFE, final_fitness=0.0."
                ))
            elif min_pct < 10.0:
                defects.append(DefectRecord(
                    defect_name="MODERATE_CLASS_IMBALANCE",
                    severity="WARNING",
                    tier=DEFECT_TIER_YELLOW,
                    evidence=f"Minority class represents {min_pct:.2f}% (<10.0% standard guidance)",
                    threshold="minority_percentage >= 10.0%",
                    threshold_provenance=PROVENANCE_DEFECT_POLICY,
                    policy_action="Cap fitness at 65.0."
                ))

        # 6. UNIQUENESS DEFECTS
        if uniq_res:
            dup_rate = uniq_res.raw_metrics.get("duplicate_row_rate", 0.0)
            if dup_rate > 0.40:
                defects.append(DefectRecord(
                    defect_name="HIGH_ROW_DUPLICATION",
                    severity="WARNING",
                    tier=DEFECT_TIER_YELLOW,
                    evidence=f"Exact duplicate rows represent {dup_rate*100:.1f}% of observations",
                    threshold="duplicate_row_rate <= 0.40",
                    threshold_provenance=PROVENANCE_DEFECT_POLICY,
                    policy_action="Cap fitness at 60.0."
                ))

        # 7. OUTLIER / ANOMALY DEFECTS FOR REGRESSION
        if resolved_task == TaskType.SUPERVISED_REGRESSION and out_res:
            extreme_kurt = out_res.raw_metrics.get("extreme_kurtosis_cols", [])
            anomaly_ratio = out_res.raw_metrics.get("anomaly_ratio", 0.0)
            if len(extreme_kurt) >= 2 and anomaly_ratio > 0.08:
                defects.append(DefectRecord(
                    defect_name="EXTREME_OUTLIER_BURDEN",
                    severity="WARNING",
                    tier=DEFECT_TIER_YELLOW,
                    evidence=f"{anomaly_ratio*100:.1f}% anomalies and {len(extreme_kurt)} heavy-tailed columns",
                    threshold="anomaly_ratio <= 0.08",
                    threshold_provenance=PROVENANCE_DEFECT_POLICY,
                    policy_action="Cap fitness at 65.0."
                ))

        # Tier Resolution
        has_red = any(d.tier == DEFECT_TIER_RED for d in defects)
        has_yellow = any(d.tier == DEFECT_TIER_YELLOW for d in defects)

        if has_red:
            return DefectPolicyResult(
                tier=DEFECT_TIER_RED,
                fitness_status="UNSAFE",
                raw_fitness=round(raw_fitness, 2),
                final_fitness=0.0,
                is_unsafe=True,
                cap_applied=0.0,
                defects_detected=defects,
                provenance=PROVENANCE_DEFECT_POLICY
            )
        elif has_yellow:
            # Determine lowest cap
            caps = []
            for d in defects:
                if "60.0" in d.policy_action:
                    caps.append(60.0)
                elif "65.0" in d.policy_action:
                    caps.append(65.0)
                elif "70.0" in d.policy_action:
                    caps.append(70.0)
            cap_val = min(caps) if caps else 65.0
            final_fit = min(raw_fitness, cap_val)
            return DefectPolicyResult(
                tier=DEFECT_TIER_YELLOW,
                fitness_status="WARNING",
                raw_fitness=round(raw_fitness, 2),
                final_fitness=round(final_fit, 2),
                is_unsafe=False,
                cap_applied=cap_val,
                defects_detected=defects,
                provenance=PROVENANCE_DEFECT_POLICY
            )
        else:
            return DefectPolicyResult(
                tier=DEFECT_TIER_GREEN,
                fitness_status="NORMAL",
                raw_fitness=round(raw_fitness, 2),
                final_fitness=round(raw_fitness, 2),
                is_unsafe=False,
                cap_applied=None,
                defects_detected=[],
                provenance=PROVENANCE_DEFECT_POLICY
            )

    @classmethod
    def evaluate_veto(
        cls,
        task: TaskType,
        dimension_results: Dict[QualityDimension, DimensionResult],
        num_rows: int,
        confidence: float = 1.0,
        imbalance_policy: Optional[Any] = None
    ) -> Tuple[bool, Optional[str], Optional[float]]:
        """
        Backward-compatible interface for evaluate_veto.
        """
        policy_res = cls.evaluate_defect_policy(
            task=task,
            dimension_results=dimension_results,
            raw_fitness=100.0,
            num_rows=num_rows,
            imbalance_policy=imbalance_policy
        )
        if policy_res.tier == DEFECT_TIER_RED:
            red_defect = next((d for d in policy_res.defects_detected if d.tier == DEFECT_TIER_RED), None)
            reason = red_defect.evidence if red_defect else "Hard Veto Triggered"
            return True, f"Hard Veto (RED): {reason}", 0.0
        elif policy_res.tier == DEFECT_TIER_YELLOW:
            yellow_defect = next((d for d in policy_res.defects_detected if d.tier == DEFECT_TIER_YELLOW), None)
            reason = yellow_defect.evidence if yellow_defect else "Policy Warning Triggered"
            return True, f"Policy Warning (YELLOW): {reason}", policy_res.cap_applied
        return False, None, None

    @classmethod
    def evaluate_policy(
        cls,
        task: TaskType,
        dimension_results: Dict[QualityDimension, DimensionResult],
        raw_fitness: float,
        num_rows: int = 100,
        imbalance_policy: Optional[Any] = None
    ) -> DefectPolicyResult:
        return cls.evaluate_defect_policy(
            task=task,
            dimension_results=dimension_results,
            raw_fitness=raw_fitness,
            num_rows=num_rows,
            imbalance_policy=imbalance_policy
        )


DefectPolicyEngine = IntelligentVetoEngine
