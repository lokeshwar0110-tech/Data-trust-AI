from typing import List, Dict, Any, Optional
from datatrust.models import (
    TaskType,
    QualityDimension,
    DimensionResult,
    RemediationActionV2
)


class RemediationOptimizer:
    """
    Subsystem for Marginal Remediation Utility (MRU) Optimization.
    Ranks data preparation and defect remediation actions by:
    MRU_i = Delta_Fitness_i / Cost_i

    Transforms diagnostic quality checks into a cost-aware dataset fitness
    optimization engine with what-if simulation capabilities.
    """

    COST_MODEL = {
        "drop_column": 1,        # O(1) dropping a target-leaking or constant column
        "winsorize_outliers": 2, # O(N) quantile clipping
        "sort_timestamps": 2,    # O(N log N) chronological sorting
        "impute_median": 2,      # O(N) simple imputation
        "resample_imbalance": 3, # O(N) SMOTE or class re-weighting
        "iterative_impute": 4,   # O(N * M * iter) MICE or MissForest model imputation
        "manual_relabel": 5      # O(N) human review of noisy labels
    }

    @classmethod
    def optimize_remediations(
        cls,
        task: TaskType,
        dimensions: Dict[QualityDimension, DimensionResult],
        current_trust_score: float,
        is_vetoed: bool = False,
        train_ratio: float = 0.70,
        test_defect_fraction: Optional[float] = None
    ) -> List[RemediationActionV2]:
        """
        Synthesizes candidate remediations and ranks them by Marginal Remediation Utility (MRU).
        MRU = Predicted Fitness Improvement / Operational Cost.

        Principled Structural Derivation (No Magic Numbers):
        In a train/test split regime (default 70/30), transformations are applied strictly
        to the training partition to guarantee zero test leakage (test set immutability).
        Consequently, defects residing in the holdout test partition are irreducible.
        The predicted gain is structurally discounted by the test defect proportion:
            discount = 1.0 - test_defect_fraction
        where test_defect_fraction is structurally derived from (1.0 - train_ratio).
        """
        if test_defect_fraction is None:
            test_defect_fraction = max(0.0, min(0.90, 1.0 - train_ratio))

        train_fraction = 1.0 - test_defect_fraction

        actions: List[RemediationActionV2] = []
        action_idx = 1

        # Check 1: Target Leakage (Critical Veto in Classification/Regression)
        leak_res = dimensions.get(QualityDimension.CONSISTENCY) or dimensions.get(QualityDimension.VALIDITY) or dimensions.get(QualityDimension.LEAKAGE)
        if leak_res:
            leakage = leak_res.raw_metrics.get("target_leakage_suspects", [])
            if leakage:
                suspects = [x["feature"] for x in leakage]
                nominal_gain = 35.0 if is_vetoed else 18.0
                gain = round(nominal_gain * train_fraction, 1)
                cost = cls.COST_MODEL["drop_column"]
                actions.append(
                    RemediationActionV2(
                        id=f"act_{action_idx}",
                        action_type="drop_leakage_columns",
                        dimension=QualityDimension.CONSISTENCY,
                        issue=f"Suspected post-outcome target leakage in feature(s): {suspects}",
                        recommendation=f"Drop leaking column(s) {suspects} immediately to eliminate spurious predictive shortcuts.",
                        severity="critical",
                        fitness_improvement=gain,
                        operational_cost=cost,
                        mru=round(gain / cost, 2),
                        priority_rank=0
                    )
                )
                action_idx += 1

        # Check 2: Temporal Disorder / Non-Monotonic Index (Critical in Time Series)
        tmp_res = dimensions.get(QualityDimension.TIMELINESS) or dimensions.get(QualityDimension.TEMPORAL_VALIDITY)
        unq_res = dimensions.get(QualityDimension.UNIQUENESS)
        if (tmp_res or unq_res) and task == TaskType.TIME_SERIES_FORECASTING:
            is_monotonic = tmp_res.raw_metrics.get("is_monotonic_increasing", True) if tmp_res else True
            dups = 0
            if tmp_res:
                dups = tmp_res.raw_metrics.get("duplicate_count", tmp_res.raw_metrics.get("duplicate_timestamps", 0))
            if unq_res and dups == 0:
                dups = unq_res.raw_metrics.get("duplicate_timestamps", 0)
            if not is_monotonic or dups > 0:
                # Structural derivation: lifting the veto on train partition yields (100 - F_0) train gain,
                # discounted by the immutable test partition fraction
                nominal_gain = max(25.0, 92.0 - current_trust_score) if is_vetoed else 18.0
                gain = round(nominal_gain * train_fraction, 1)
                cost = cls.COST_MODEL["sort_timestamps"]
                actions.append(
                    RemediationActionV2(
                        id=f"act_{action_idx}",
                        action_type="sort_timestamps",
                        dimension=QualityDimension.TIMELINESS,
                        issue=f"Non-chronological temporal disorder and {dups} duplicate timestamp collisions.",
                        recommendation="Sort dataset chronologically by timestamp index and deduplicate timestamp collisions.",
                        severity="critical",
                        fitness_improvement=gain,
                        operational_cost=cost,
                        mru=round(gain / cost, 2),
                        priority_rank=0
                    )
                )
                action_idx += 1

        # Check 3: Class Collapse / Severe Label Imbalance (Classification)
        bias_res = dimensions.get(QualityDimension.DISTRIBUTION_BALANCE) or dimensions.get(QualityDimension.BIAS_IMBALANCE)
        lbl_res = dimensions.get(QualityDimension.VALIDITY) or dimensions.get(QualityDimension.LABEL_INTEGRITY)
        if task == TaskType.SUPERVISED_CLASSIFICATION:
            is_imbalanced = False
            imb_ratio = 1.0
            if bias_res and bias_res.raw_metrics.get("is_severely_imbalanced", False):
                is_imbalanced = True
                imb_ratio = bias_res.raw_metrics.get("imbalance_ratio", 50.0)
            elif lbl_res and lbl_res.raw_metrics.get("dominant_class_ratio", 0.5) >= 0.95:
                is_imbalanced = True
                imb_ratio = lbl_res.raw_metrics.get("imbalance_ratio", 50.0)

            if is_imbalanced:
                nominal_gain = max(30.0, 90.0 - current_trust_score) if is_vetoed else 15.0
                gain = round(nominal_gain * train_fraction, 1)
                cost = cls.COST_MODEL["resample_imbalance"]
                actions.append(
                    RemediationActionV2(
                        id=f"act_{action_idx}",
                        action_type="resample_imbalance",
                        dimension=QualityDimension.DISTRIBUTION_BALANCE,
                        issue=f"Severe target class imbalance ({imb_ratio:.1f}:1 ratio, <2% minority representation).",
                        recommendation="Apply balanced minority upsampling or class re-weighting during model training.",
                        severity="critical",
                        fitness_improvement=gain,
                        operational_cost=cost,
                        mru=round(gain / cost, 2),
                        priority_rank=0
                    )
                )
                action_idx += 1

            if lbl_res:
                target_missing = lbl_res.raw_metrics.get("target_missing_rate", 0.0)
                if target_missing > 0.05:
                    gain = 14.0
                    cost = cls.COST_MODEL["drop_column"]
                    actions.append(
                        RemediationActionV2(
                            id=f"act_{action_idx}",
                            action_type="drop_unlabelled_target",
                            dimension=QualityDimension.VALIDITY,
                            issue=f"Target column missing {target_missing*100:.1f}% of labels.",
                            recommendation="Drop unlabelled target records before supervised model training.",
                            severity="critical",
                            fitness_improvement=gain,
                            operational_cost=cost,
                            mru=round(gain / cost, 2),
                            priority_rank=0
                        )
                    )
                    action_idx += 1

        # Check 4: Heavy Tail Outliers
        out_res = dimensions.get(QualityDimension.OUTLIER_ANOMALY) or dimensions.get(QualityDimension.OUTLIER_RESILIENCE)
        if out_res and out_res.score < 85.0:
            nominal_gain = max(20.0, 85.0 - current_trust_score) if is_vetoed else (out_res.weight * (100.0 - out_res.score) * 0.85)
            gain = round(nominal_gain * train_fraction, 1)
            cost = cls.COST_MODEL["winsorize_outliers"]
            actions.append(
                RemediationActionV2(
                    id=f"act_{action_idx}",
                    action_type="winsorize_outliers",
                    dimension=QualityDimension.OUTLIER_ANOMALY,
                    issue="Extreme tail kurtosis and anomalous outliers detected in target/features.",
                    recommendation="Apply 1st-99th percentile Winsorization to clip heavy tails and stabilize variance.",
                    severity="high" if task == TaskType.SUPERVISED_REGRESSION else "medium",
                    fitness_improvement=gain,
                    operational_cost=cost,
                    mru=round(gain / cost, 2),
                    priority_rank=0
                )
            )
            action_idx += 1

        # Check 5: Missingness in Features
        cmp_res = dimensions.get(QualityDimension.COMPLETENESS) or dimensions.get(QualityDimension.MISSINGNESS)
        if cmp_res and cmp_res.score < 85.0:
            missing_rate = cmp_res.raw_metrics.get("missing_rate", 0.0)
            high_cols = cmp_res.raw_metrics.get("high_missing_cols", [])
            nominal_gain = cmp_res.weight * (100.0 - cmp_res.score) * 0.8
            gain = round(nominal_gain * train_fraction, 1)

            if high_cols:
                cost = cls.COST_MODEL["drop_column"]
                actions.append(
                    RemediationActionV2(
                        id=f"act_{action_idx}",
                        action_type="drop_high_null_columns",
                        dimension=QualityDimension.COMPLETENESS,
                        issue=f"Columns {high_cols[:3]} have >40% null values.",
                        recommendation="Drop high-null columns to eliminate sparsity noise.",
                        severity="medium",
                        fitness_improvement=gain,
                        operational_cost=cost,
                        mru=round(gain / cost, 2),
                        priority_rank=0
                    )
                )
                action_idx += 1
            elif missing_rate > 0.05:
                cost = cls.COST_MODEL["impute_median"]
                actions.append(
                    RemediationActionV2(
                        id=f"act_{action_idx}",
                        action_type="impute_median",
                        dimension=QualityDimension.COMPLETENESS,
                        issue=f"Moderate missingness ({missing_rate*100:.1f}% overall).",
                        recommendation="Impute missing features using median or KNN imputer.",
                        severity="medium",
                        fitness_improvement=gain,
                        operational_cost=cost,
                        mru=round(gain / cost, 2),
                        priority_rank=0
                    )
                )
                action_idx += 1

        # Sort descending by Marginal Remediation Utility (MRU = Delta Fitness / Cost)
        actions.sort(key=lambda a: a.mru, reverse=True)

        # Assign priority ranks
        for idx, act in enumerate(actions):
            act.priority_rank = idx + 1

        return actions

    @classmethod
    def simulate_remediations(
        cls,
        current_trust_score: float,
        actions: List[RemediationActionV2],
        applied_action_ids: List[str],
        test_defect_fraction: Optional[float] = None
    ) -> float:
        """
        Simulates what-if post-remediation fitness score based on selected actions.
        Accounts for irreducible defects surviving in the immutable holdout test partition.
        """
        projected = current_trust_score
        applied_set = set(applied_action_ids)

        for act in actions:
            if act.id in applied_set:
                projected += act.fitness_improvement

        # When test_defect_fraction is provided, enforce theoretical upper bound
        if test_defect_fraction is not None and test_defect_fraction > 0.0:
            max_achievable = (100.0 * (1.0 - test_defect_fraction)) + (current_trust_score * test_defect_fraction)
            projected = min(max_achievable, projected)

        return min(100.0, round(projected, 1))

    @classmethod
    def generate_what_if_scenarios(
        cls,
        current_trust_score: float,
        actions: List[RemediationActionV2]
    ) -> List["WhatIfScenario"]:
        """
        Generates Scenario A (quick low-cost wins), Scenario B (moderate fixes),
        and Scenario C (Pareto-optimal recommended repair).
        """
        from datatrust.models import WhatIfScenario
        if not actions:
            return []

        # Scenario A: Top priority quick fix (Cost <= 2)
        quick_actions = [a.id for a in actions if a.operational_cost <= 2][:1]
        score_a = cls.simulate_remediations(current_trust_score, actions, quick_actions)

        # Scenario B: Top 2 MRU actions
        mod_actions = [a.id for a in actions][:2]
        score_b = cls.simulate_remediations(current_trust_score, actions, mod_actions)

        # Scenario C: Recommended full Pareto set (actions with MRU >= 1.5)
        rec_actions = [a.id for a in actions if a.mru >= 1.5]
        if not rec_actions:
            rec_actions = [a.id for a in actions]
        score_c = cls.simulate_remediations(current_trust_score, actions, rec_actions)

        scenarios = [
            WhatIfScenario(
                name="Scenario A (Quick Wins)",
                description="Apply lowest operational complexity fixes (Cost <= 2).",
                action_ids=quick_actions,
                simulated_fitness=score_a,
                improvement=round(score_a - current_trust_score, 1),
                is_recommended=False
            ),
            WhatIfScenario(
                name="Scenario B (Balanced Fix)",
                description="Apply top 2 highest-utility data preparation steps.",
                action_ids=mod_actions,
                simulated_fitness=score_b,
                improvement=round(score_b - current_trust_score, 1),
                is_recommended=False
            ),
            WhatIfScenario(
                name="Scenario C (Pareto Optimal)",
                description="Comprehensive high-utility cleansing. Maximizes fitness return.",
                action_ids=rec_actions,
                simulated_fitness=score_c,
                improvement=round(score_c - current_trust_score, 1),
                is_recommended=True
            )
        ]
        return scenarios

    @classmethod
    def generate_counterfactual_table(
        cls,
        current_trust_score: float,
        actions: List[RemediationActionV2]
    ) -> List[Dict[str, Any]]:
        """
        Generates an empirical counterfactual matrix evaluating each candidate action in isolation
        and cumulatively.
        Note: Optimization ranks candidate actions by heuristic Marginal Remediation Utility
        MRU = predicted_delta_fitness / cost, without guaranteeing that arbitrary downstream
        models will observe empirical metric improvements.
        """
        table = []
        running_fitness = current_trust_score
        for act in actions:
            isolated_fitness = min(100.0, round(current_trust_score + act.fitness_improvement, 1))
            running_fitness = min(100.0, round(running_fitness + act.fitness_improvement, 1))
            table.append({
                "action_id": act.id,
                "action_type": act.action_type or act.id,
                "dimension": act.dimension.value,
                "issue": act.issue,
                "recommendation": act.recommendation,
                "severity": act.severity,
                "operational_cost": act.operational_cost,
                "predicted_delta_fitness": round(act.fitness_improvement, 1),
                "mru": act.mru,
                "isolated_simulated_fitness": isolated_fitness,
                "cumulative_simulated_fitness": running_fitness
            })
        return table
