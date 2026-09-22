from typing import Dict, List, Tuple
from datatrust.models import (
    QualityDimension,
    DimensionResult,
    FactorAttribution,
    RemediationRecommendation,
    TaskType
)


class ExplainabilityEngine:
    """
    Computes factor attribution breakdowns and generates prioritized,
    actionable remediation guidance based on task sensitivity.
    """

    @staticmethod
    def compute_factor_attributions(
        dimensions: Dict[QualityDimension, DimensionResult],
        total_trust_score: float
    ) -> List[FactorAttribution]:
        total_loss = max(0.001, 100.0 - total_trust_score)
        attributions = []

        for dim, res in dimensions.items():
            dim_loss = res.weight * (100.0 - res.score)
            share_of_loss = (dim_loss / total_loss) * 100.0 if total_loss > 0 else 0.0

            explanation = (
                f"{dim.value.replace('_', ' ').title()} deducted {dim_loss:.1f} points "
                f"(scored {res.score:.1f}/100 with AHP weight {res.weight*100:.1f}%)."
            )

            attributions.append(
                FactorAttribution(
                    dimension=dim,
                    penalty_points=round(dim_loss, 2),
                    percentage_of_loss=round(share_of_loss, 1),
                    explanation=explanation
                )
            )

        # Sort descending by penalty points
        attributions.sort(key=lambda x: x.penalty_points, reverse=True)
        return attributions

    @staticmethod
    def generate_remediations(
        task: TaskType,
        dimensions: Dict[QualityDimension, DimensionResult]
    ) -> List[RemediationRecommendation]:
        remediations: List[RemediationRecommendation] = []

        for dim, res in dimensions.items():
            if res.score >= 88.0:
                continue  # Dimension is in good health

            potential_gain = round(res.weight * (100.0 - res.score) * 0.8, 1)

            if dim == QualityDimension.COMPLETENESS:
                missing_rate = res.raw_metrics.get("missing_rate", 0.0)
                high_cols = res.raw_metrics.get("high_missing_cols", [])
                severely_empty_rows = res.raw_metrics.get("severely_empty_rows", 0)

                if high_cols:
                    remediations.append(
                        RemediationRecommendation(
                            dimension=dim,
                            severity="high" if res.weight > 0.15 else "medium",
                            issue=f"Columns {high_cols[:3]} have >40% missing values.",
                            recommendation="Consider dropping columns with >40% nulls or imputing using MICE / MissForest rather than naive mean imputation.",
                            estimated_score_impact=potential_gain
                        )
                    )
                if severely_empty_rows > 0:
                    remediations.append(
                        RemediationRecommendation(
                            dimension=dim,
                            severity="medium",
                            issue=f"{severely_empty_rows} rows have >50% missing fields.",
                            recommendation="Drop severely empty records to prevent noise amplification in downstream models.",
                            estimated_score_impact=round(potential_gain * 0.4, 1)
                        )
                    )

            elif dim == QualityDimension.OUTLIER_RESILIENCE:
                kurt_cols = res.raw_metrics.get("extreme_kurtosis_cols", [])
                anomaly_ratio = res.raw_metrics.get("anomaly_ratio", 0.0)

                sev = "critical" if task == TaskType.SUPERVISED_REGRESSION else "medium"
                remediations.append(
                    RemediationRecommendation(
                        dimension=dim,
                        severity=sev,
                        issue=f"Detected {anomaly_ratio*100:.1f}% multivariate outliers and heavy-tail features: {kurt_cols[:3]}.",
                        recommendation="Apply Yeo-Johnson / Box-Cox power transform or Winsorization to cap extreme tails (vital for MSE/RMSE minimization).",
                        estimated_score_impact=potential_gain
                    )
                )

            elif dim == QualityDimension.DRIFT_STABILITY:
                drifted = res.raw_metrics.get("drifted_columns", [])
                remediations.append(
                    RemediationRecommendation(
                        dimension=dim,
                        severity="high" if task in [TaskType.SUPERVISED_CLASSIFICATION, TaskType.TIME_SERIES_FORECASTING] else "medium",
                        issue=f"Significant distribution shift detected across partitions in features: {drifted[:3]}.",
                        recommendation="Apply covariate shift adaptation or sample re-weighting; inspect if data was collected across non-stationary regimes.",
                        estimated_score_impact=potential_gain
                    )
                )

            elif dim == QualityDimension.LABEL_INTEGRITY:
                target_missing = res.raw_metrics.get("target_missing_rate", 0.0)
                imbalance = res.raw_metrics.get("imbalance_ratio", 1.0)

                if target_missing > 0:
                    remediations.append(
                        RemediationRecommendation(
                            dimension=dim,
                            severity="critical",
                            issue=f"Target column contains {target_missing*100:.1f}% missing values.",
                            recommendation="Supervised learning cannot train on missing targets without semi-supervised methods. Drop unlabelled target rows.",
                            estimated_score_impact=potential_gain
                        )
                    )
                if imbalance > 5.0:
                    remediations.append(
                        RemediationRecommendation(
                            dimension=dim,
                            severity="high",
                            issue=f"Target class imbalance ratio is {imbalance:.1f}:1.",
                            recommendation="Employ SMOTE / ADASYN oversampling or calibrate focal loss / class weights during training.",
                            estimated_score_impact=round(potential_gain * 0.7, 1)
                        )
                    )

            elif dim == QualityDimension.TEMPORAL_VALIDITY:
                monotonic = res.raw_metrics.get("is_monotonic_increasing", True)
                dups = res.raw_metrics.get("duplicate_timestamps", 0)

                if not monotonic:
                    remediations.append(
                        RemediationRecommendation(
                            dimension=dim,
                            severity="critical",
                            issue="Timestamps are out of chronological order (structural temporal invalidity).",
                            recommendation="Sort dataset chronologically by timestamp index before windowing or lag feature engineering.",
                            estimated_score_impact=potential_gain
                        )
                    )
                if dups > 0:
                    remediations.append(
                        RemediationRecommendation(
                            dimension=dim,
                            severity="high",
                            issue=f"Found {dups} duplicate timestamps on the temporal index.",
                            recommendation="Aggregate duplicate timestamp entries by mean/median or verify multi-entity grouping key.",
                            estimated_score_impact=round(potential_gain * 0.5, 1)
                        )
                    )

            elif dim == QualityDimension.CARDINALITY_SCHEMA:
                const_cols = res.raw_metrics.get("constant_columns", [])
                high_card = res.raw_metrics.get("high_cardinality_text_columns", [])

                if const_cols:
                    remediations.append(
                        RemediationRecommendation(
                            dimension=dim,
                            severity="low",
                            issue=f"Columns {const_cols} have zero variance (constant value).",
                            recommendation="Drop constant columns to reduce feature dimensionality and eliminate zero-variance warnings.",
                            estimated_score_impact=round(potential_gain * 0.5, 1)
                        )
                    )
                if high_card:
                    remediations.append(
                        RemediationRecommendation(
                            dimension=dim,
                            severity="medium",
                            issue=f"High cardinality identifier columns detected: {high_card[:2]}.",
                            recommendation="Drop unique identifier keys or apply target encoding / entity embeddings instead of one-hot encoding.",
                            estimated_score_impact=round(potential_gain * 0.5, 1)
                        )
                    )

            elif dim == QualityDimension.CORRELATION_INTEGRITY:
                leakage = res.raw_metrics.get("target_leakage_suspects", [])
                collinear = res.raw_metrics.get("collinear_pairs", [])

                if leakage:
                    remediations.append(
                        RemediationRecommendation(
                            dimension=dim,
                            severity="critical",
                            issue=f"Suspected target leakage in features: {[x['feature'] for x in leakage]}.",
                            recommendation="Remove features with >0.98 target correlation that represent post-outcome artifacts.",
                            estimated_score_impact=potential_gain
                        )
                    )
                if collinear:
                    remediations.append(
                        RemediationRecommendation(
                            dimension=dim,
                            severity="medium",
                            issue=f"Found {len(collinear)} highly collinear feature pairs (r > 0.92).",
                            recommendation="Apply Variance Inflation Factor (VIF) filtering or PCA to remove collinear redundancies.",
                            estimated_score_impact=round(potential_gain * 0.5, 1)
                        )
                    )

        # Sort remediations: critical first, then high estimated impact
        severity_rank = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        remediations.sort(key=lambda x: (severity_rank.get(x.severity, 4), -x.estimated_score_impact))
        return remediations
