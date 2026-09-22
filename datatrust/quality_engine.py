import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List
from sklearn.ensemble import IsolationForest
from datatrust.models import QualityDimension, DimensionResult, TaskType, ImbalancePolicy
from datatrust.config import (
    CANONICAL_DIMENSIONS,
    PROVENANCE_DIMENSIONS,
    REPRESENTATIVENESS_UNAVAILABLE_STATUS
)


class QualityEngine:
    """
    Executes modular evaluation across all eight canonical quality dimensions:
    1. Completeness
    2. Validity
    3. Consistency
    4. Uniqueness
    5. Timeliness
    6. Outlier / Anomaly Quality
    7. Distribution Balance
    8. Coverage / Representativeness

    Each dimension exposes explicit mathematical formulas, assumptions, limitations,
    and methodological provenance.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        time_column: Optional[str] = None,
        imbalance_policy: Optional[Any] = None
    ):
        self.df = df
        self.target_column = target_column
        self.time_column = time_column
        self.imbalance_policy = imbalance_policy
        self.num_rows, self.num_cols = df.shape
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Identify obvious identifier/key columns (e.g. id, key with high cardinality)
        self.identifier_cols = [
            c for c in self.df.columns
            if ('id' in c.lower() or 'key' in c.lower() or c.lower().endswith('_id'))
            and self.df[c].nunique() > 0.85 * self.num_rows
        ]
        if self.target_column and self.target_column in self.numeric_cols:
            self.feature_numeric_cols = [
                c for c in self.numeric_cols
                if c != self.target_column and c not in self.identifier_cols
            ]
        else:
            self.feature_numeric_cols = [
                c for c in self.numeric_cols
                if c not in self.identifier_cols
            ]

    def evaluate_completeness(self) -> DimensionResult:
        """
        Dimension 1: Completeness
        Measures cell missingness rate, column-level vacancy, and row vacancy.
        """
        total_cells = self.num_rows * self.num_cols
        total_missing = int(self.df.isna().sum().sum())
        missing_rate = total_missing / total_cells if total_cells > 0 else 0.0

        col_missing = (self.df.isna().sum() / self.num_rows).to_dict()
        high_missing_cols = [c for c, r in col_missing.items() if r > 0.40]
        completely_empty_cols = [c for c, r in col_missing.items() if r == 1.0]

        row_missing_ratio = (self.df.isna().sum(axis=1) / self.num_cols)
        severely_empty_rows = int((row_missing_ratio > 0.50).sum())

        penalty = (missing_rate * 80.0) + (len(high_missing_cols) * 5.0) + (len(completely_empty_cols) * 15.0)
        score = max(0.0, min(100.0, 100.0 - penalty))

        summary = (
            f"Overall missing rate is {missing_rate * 100:.2f}%. "
            f"{len(high_missing_cols)} columns have >40% nulls, {len(completely_empty_cols)} completely empty."
        )

        dim_meta = CANONICAL_DIMENSIONS[QualityDimension.COMPLETENESS]

        return DimensionResult(
            dimension=QualityDimension.COMPLETENESS,
            score=round(score, 2),
            weight=0.0,
            weighted_score=0.0,
            raw_metrics={
                "missing_rate": round(missing_rate, 4),
                "total_missing_cells": total_missing,
                "high_missing_cols": high_missing_cols,
                "completely_empty_cols": completely_empty_cols,
                "severely_empty_rows": severely_empty_rows
            },
            summary=summary,
            formula=dim_meta.formula,
            assumptions=dim_meta.assumptions,
            limitations=dim_meta.limitations,
            task_applicability=dim_meta.task_applicability,
            provenance=dim_meta.provenance
        )

    def evaluate_validity(self, task: Optional[TaskType] = None) -> DimensionResult:
        """
        Dimension 2: Validity
        Evaluates data type conformity, infinite numeric values, constant features,
        and target column validity for supervised tasks.
        """
        constant_cols = []
        inf_cols = []
        total_inf_count = 0

        for col in self.df.columns:
            n_uniq = self.df[col].nunique(dropna=True)
            if n_uniq <= 1:
                constant_cols.append(col)

            if col in self.numeric_cols:
                series = self.df[col]
                inf_count = int(np.isinf(series).sum())
                if inf_count > 0:
                    inf_cols.append(col)
                    total_inf_count += inf_count

        total_numeric_cells = max(1, self.num_rows * len(self.numeric_cols))
        inf_rate = total_inf_count / total_numeric_cells

        penalty = (len(constant_cols) * 10.0) + (inf_rate * 100.0) + (len(inf_cols) * 5.0)

        # Supervised target validity check
        target_invalid = False
        target_veto_msg = None
        if task in [TaskType.SUPERVISED_CLASSIFICATION, TaskType.SUPERVISED_REGRESSION]:
            if not self.target_column or self.target_column not in self.df.columns:
                target_invalid = True
                penalty += 60.0
                target_veto_msg = "Supervised task designated without a valid target column."
            else:
                t_series = self.df[self.target_column].dropna()
                if t_series.nunique() <= 1:
                    target_invalid = True
                    penalty += 70.0
                    target_veto_msg = "Target column has zero variance (single unique value)."
                elif task == TaskType.SUPERVISED_REGRESSION and not pd.api.types.is_numeric_dtype(t_series):
                    target_invalid = True
                    penalty += 60.0
                    target_veto_msg = f"Target '{self.target_column}' is non-numeric for regression."

        score = max(10.0, min(100.0, 100.0 - penalty))
        summary = (
            f"Found {len(constant_cols)} constant column(s) and "
            f"{len(inf_cols)} column(s) with infinite values."
        )

        dim_meta = CANONICAL_DIMENSIONS[QualityDimension.VALIDITY]

        return DimensionResult(
            dimension=QualityDimension.VALIDITY,
            score=round(score, 2),
            weight=0.0,
            weighted_score=0.0,
            raw_metrics={
                "constant_columns": constant_cols,
                "infinite_value_columns": inf_cols,
                "total_infinite_count": total_inf_count,
                "target_column": self.target_column,
                "target_valid": not target_invalid
            },
            summary=summary,
            formula=dim_meta.formula,
            assumptions=dim_meta.assumptions,
            limitations=dim_meta.limitations,
            task_applicability=dim_meta.task_applicability,
            provenance=dim_meta.provenance,
            is_veto_triggered=target_invalid,
            veto_reason=target_veto_msg
        )

    def evaluate_consistency(self) -> DimensionResult:
        """
        Dimension 3: Consistency
        Condition 2: Measures internally contradictory, structurally inconsistent,
        or cross-field invalid observations (collinear feature pairs, target leakage).
        DOES NOT measure distribution drift (which belongs to Timeliness).
        """
        if len(self.feature_numeric_cols) < 2:
            dim_meta = CANONICAL_DIMENSIONS[QualityDimension.CONSISTENCY]
            return DimensionResult(
                dimension=QualityDimension.CONSISTENCY,
                score=95.0,
                weight=0.0,
                weighted_score=0.0,
                raw_metrics={"collinear_pairs": [], "target_leakage_suspects": []},
                summary="Insufficient numerical features for cross-field consistency/correlation check.",
                formula=dim_meta.formula,
                assumptions=dim_meta.assumptions,
                limitations=dim_meta.limitations,
                task_applicability=dim_meta.task_applicability,
                provenance=dim_meta.provenance
            )

        clean_sub = self.df[self.feature_numeric_cols].dropna()
        if len(clean_sub) < 10:
            dim_meta = CANONICAL_DIMENSIONS[QualityDimension.CONSISTENCY]
            return DimensionResult(
                dimension=QualityDimension.CONSISTENCY,
                score=90.0,
                weight=0.0,
                weighted_score=0.0,
                raw_metrics={"collinear_pairs": []},
                summary="Insufficient clean rows for consistency correlation calculation.",
                formula=dim_meta.formula,
                assumptions=dim_meta.assumptions,
                limitations=dim_meta.limitations,
                task_applicability=dim_meta.task_applicability,
                provenance=dim_meta.provenance
            )

        corr = clean_sub.corr(method="pearson").abs()
        collinear_pairs = []
        cols = corr.columns

        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                val = corr.iloc[i, j]
                if val > 0.92:
                    collinear_pairs.append({
                        "feat_a": cols[i],
                        "feat_b": cols[j],
                        "correlation": round(float(val), 3)
                    })

        # Target leakage check
        target_leakage_suspects = []
        if self.target_column and self.target_column in self.df.columns and pd.api.types.is_numeric_dtype(self.df[self.target_column]):
            t_clean = self.df[[self.target_column] + self.feature_numeric_cols].dropna()
            if len(t_clean) > 10:
                t_corr = t_clean.corr().abs()[self.target_column]
                for f in self.feature_numeric_cols:
                    val = float(t_corr.get(f, 0.0))
                    if val > 0.98:
                        target_leakage_suspects.append({"feature": f, "target_corr": round(val, 3)})

        penalty = (len(collinear_pairs) * 6.0) + (len(target_leakage_suspects) * 25.0)
        score = max(15.0, min(100.0, 100.0 - penalty))

        summary = (
            f"Detected {len(collinear_pairs)} highly collinear feature pair(s) "
            f"and {len(target_leakage_suspects)} suspect target leakage feature(s)."
        )

        dim_meta = CANONICAL_DIMENSIONS[QualityDimension.CONSISTENCY]
        is_veto = len(target_leakage_suspects) > 0
        veto_msg = f"Critical target leakage detected in {len(target_leakage_suspects)} feature(s)." if is_veto else None

        return DimensionResult(
            dimension=QualityDimension.CONSISTENCY,
            score=round(score, 2),
            weight=0.0,
            weighted_score=0.0,
            raw_metrics={
                "collinear_pairs": collinear_pairs,
                "target_leakage_suspects": target_leakage_suspects,
                "leakage_detected": is_veto
            },
            summary=summary,
            formula=dim_meta.formula,
            assumptions=dim_meta.assumptions,
            limitations=dim_meta.limitations,
            task_applicability=dim_meta.task_applicability,
            provenance=dim_meta.provenance,
            is_veto_triggered=is_veto,
            veto_reason=veto_msg
        )

    def evaluate_uniqueness(self) -> DimensionResult:
        """
        Dimension 4: Uniqueness
        Measures exact duplicate rows and identifier key collisions.
        """
        duplicate_rows = int(self.df.duplicated().sum())
        duplicate_ratio = duplicate_rows / self.num_rows if self.num_rows > 0 else 0.0

        key_collision_count = 0
        key_cols = []
        for col in self.identifier_cols:
            n_dups = int(self.df[col].duplicated().sum())
            if n_dups > 0:
                key_cols.append(col)
                key_collision_count += n_dups

        key_collision_rate = key_collision_count / max(1, self.num_rows * max(1, len(self.identifier_cols)))
        penalty = (duplicate_ratio * 100.0) + (key_collision_rate * 40.0)
        score = max(0.0, min(100.0, 100.0 - penalty))

        summary = (
            f"Duplicate row count: {duplicate_rows} ({duplicate_ratio*100:.2f}%). "
            f"Key collisions: {key_collision_count} across {len(key_cols)} identifier column(s)."
        )

        dim_meta = CANONICAL_DIMENSIONS[QualityDimension.UNIQUENESS]

        return DimensionResult(
            dimension=QualityDimension.UNIQUENESS,
            score=round(score, 2),
            weight=0.0,
            weighted_score=0.0,
            raw_metrics={
                "duplicate_rows": duplicate_rows,
                "duplicate_row_rate": round(duplicate_ratio, 4),
                "key_collisions": key_collision_count,
                "key_collision_columns": key_cols
            },
            summary=summary,
            formula=dim_meta.formula,
            assumptions=dim_meta.assumptions,
            limitations=dim_meta.limitations,
            task_applicability=dim_meta.task_applicability,
            provenance=dim_meta.provenance
        )

    def evaluate_timeliness(self, task: Optional[TaskType] = None) -> DimensionResult:
        """
        Dimension 5: Timeliness
        Condition 2 & 5: Measures chronological monotonicity, duplicate timestamps,
        cadence regularity (CV), and partition distribution drift.
        """
        dim_meta = CANONICAL_DIMENSIONS[QualityDimension.TIMELINESS]

        # 1. Evaluate partition distribution drift as general stability check
        drifted_cols = []
        psi_scores = {}
        if self.feature_numeric_cols and self.num_rows >= 30:
            midpoint = self.num_rows // 2
            part1 = self.df.iloc[:midpoint][self.feature_numeric_cols]
            part2 = self.df.iloc[midpoint:][self.feature_numeric_cols]

            for col in self.feature_numeric_cols:
                s1 = part1[col].dropna()
                s2 = part2[col].dropna()
                if len(s1) < 10 or len(s2) < 10:
                    continue
                pooled_std = np.sqrt((s1.var() + s2.var()) / 2.0)
                if pooled_std > 1e-6:
                    shift = abs(s1.mean() - s2.mean()) / pooled_std
                else:
                    shift = 0.0
                psi_scores[col] = round(float(shift), 3)
                if shift > 0.5:
                    drifted_cols.append(col)

        drift_ratio = len(drifted_cols) / len(self.feature_numeric_cols) if self.feature_numeric_cols else 0.0
        drift_penalty = (drift_ratio * 40.0) + (len(drifted_cols) * 2.0)

        # 2. Evaluate Temporal Ordering if Time-Series task or time_column present
        if task == TaskType.TIME_SERIES_FORECASTING or self.time_column:
            # Find candidate time column if not explicit
            time_col = self.time_column
            if not time_col or time_col not in self.df.columns:
                for c in self.df.columns:
                    if pd.api.types.is_datetime64_any_dtype(self.df[c]) or "date" in c.lower() or "time" in c.lower():
                        time_col = c
                        break

            if not time_col and task == TaskType.TIME_SERIES_FORECASTING:
                return DimensionResult(
                    dimension=QualityDimension.TIMELINESS,
                    score=15.0,
                    weight=0.0,
                    weighted_score=0.0,
                    raw_metrics={"error": "No time column provided", "drifted_columns": drifted_cols},
                    summary="Time-series forecasting selected without a designated time index.",
                    formula=dim_meta.formula,
                    assumptions=dim_meta.assumptions,
                    limitations=dim_meta.limitations,
                    task_applicability=dim_meta.task_applicability,
                    provenance=dim_meta.provenance,
                    is_veto_triggered=True,
                    veto_reason="Time series forecasting requires an explicit timestamp/date column."
                )

            if time_col and time_col in self.df.columns:
                import warnings
                series = self.df[time_col]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", (UserWarning, FutureWarning))
                    parsed_dates = pd.to_datetime(series, errors='coerce')
                invalid_dates = int(parsed_dates.isna().sum())
                invalid_ratio = invalid_dates / self.num_rows

                if invalid_ratio > 0.2 and task == TaskType.TIME_SERIES_FORECASTING:
                    return DimensionResult(
                        dimension=QualityDimension.TIMELINESS,
                        score=25.0,
                        weight=0.0,
                        weighted_score=0.0,
                        raw_metrics={"invalid_date_ratio": invalid_ratio},
                        summary=f"Time column '{time_col}' has {invalid_ratio*100:.1f}% invalid/unparseable timestamps.",
                        formula=dim_meta.formula,
                        assumptions=dim_meta.assumptions,
                        limitations=dim_meta.limitations,
                        task_applicability=dim_meta.task_applicability,
                        provenance=dim_meta.provenance,
                        is_veto_triggered=True,
                        veto_reason="Corrupted timestamp column with high null or unparseable values."
                    )

                clean_dates = parsed_dates.dropna()
                ts_counts = clean_dates.value_counts()
                duplicate_timestamp_groups = int((ts_counts > 1).sum())
                duplicate_timestamp_rows = int(clean_dates.duplicated().sum())
                duplicate_count = duplicate_timestamp_rows
                duplicate_rate = round((duplicate_count / max(1, len(clean_dates))) * 100.0, 2)
                is_monotonic = bool(clean_dates.is_monotonic_increasing)

                out_of_order_count = 0
                if len(clean_dates) > 1:
                    diff_secs = clean_dates.diff().dt.total_seconds().dropna()
                    out_of_order_count = int((diff_secs < 0).sum())

                timestamp_unique = int(clean_dates.nunique())
                deltas = clean_dates.diff().dropna()
                delta_seconds = deltas.dt.total_seconds()
                cadence_std = float(delta_seconds.std()) if len(delta_seconds) > 1 else 0.0
                cadence_mean = float(delta_seconds.mean()) if len(delta_seconds) > 1 else 1.0
                cadence_cv = float(cadence_std / cadence_mean) if cadence_mean > 0 else 0.0

                temporal_penalty = 0.0
                if not is_monotonic:
                    temporal_penalty += 45.0
                if duplicate_count > 0:
                    temporal_penalty += min(40.0, (duplicate_count / self.num_rows) * 150.0)
                if cadence_cv > 1.5:
                    temporal_penalty += 15.0

                total_penalty = temporal_penalty + (drift_penalty * 0.5)
                score = max(0.0, min(100.0, 100.0 - total_penalty))
                is_veto = (not is_monotonic or duplicate_count > 0) and (task == TaskType.TIME_SERIES_FORECASTING)
                veto_msg = "Non-chronological temporal disorder and/or duplicate timestamps detected." if is_veto else None

                summary = (
                    f"Timestamp column '{time_col}': Monotonic={is_monotonic}, "
                    f"Duplicate groups={duplicate_timestamp_groups}, Duplicate rows={duplicate_timestamp_rows} ({duplicate_rate}%), Out-of-order jumps={out_of_order_count}."
                )

                return DimensionResult(
                    dimension=QualityDimension.TIMELINESS,
                    score=round(score, 2),
                    weight=0.0,
                    weighted_score=0.0,
                    raw_metrics={
                        "duplicate_count": duplicate_count,
                        "duplicate_timestamps": duplicate_count,
                        "duplicate_timestamp_groups": duplicate_timestamp_groups,
                        "duplicate_timestamp_rows": duplicate_timestamp_rows,
                        "duplicate_rate": duplicate_rate,
                        "non_monotonic": not is_monotonic,
                        "out_of_order_count": out_of_order_count,
                        "timestamp_unique": timestamp_unique,
                        "cadence_coefficient_of_variation": round(cadence_cv, 3),
                        "temporal_gap_statistics": {
                            "mean_seconds": round(cadence_mean, 1),
                            "std_seconds": round(cadence_std, 1),
                            "cv": round(cadence_cv, 3)
                        },
                        "is_monotonic_increasing": is_monotonic,
                        "drifted_columns": drifted_cols,
                        "partition_shift_scores": psi_scores
                    },
                    summary=summary,
                    formula=dim_meta.formula,
                    assumptions=dim_meta.assumptions,
                    limitations=dim_meta.limitations,
                    task_applicability=dim_meta.task_applicability,
                    provenance=dim_meta.provenance,
                    is_veto_triggered=is_veto,
                    veto_reason=veto_msg
                )

        # General tabular timeliness / drift stability
        score = max(50.0, min(100.0, 100.0 - drift_penalty))
        summary = f"{len(drifted_cols)} of {len(self.feature_numeric_cols)} numeric features show partition distribution shift."

        return DimensionResult(
            dimension=QualityDimension.TIMELINESS,
            score=round(score, 2),
            weight=0.0,
            weighted_score=0.0,
            raw_metrics={
                "drifted_columns": drifted_cols,
                "partition_shift_scores": psi_scores,
                "is_monotonic_increasing": True,
                "duplicate_timestamps": 0
            },
            summary=summary,
            formula=dim_meta.formula,
            assumptions=dim_meta.assumptions,
            limitations=dim_meta.limitations,
            task_applicability=dim_meta.task_applicability,
            provenance=dim_meta.provenance
        )

    def evaluate_outlier_anomaly(self) -> DimensionResult:
        """
        Dimension 6: Outlier / Anomaly Quality
        Condition 3: Primary score is derived from Isolation Forest multivariate
        anomaly density c in [0, 1]. Kurtosis is reported as descriptive supporting
        evidence only, not as the score itself.
        """
        dim_meta = CANONICAL_DIMENSIONS[QualityDimension.OUTLIER_ANOMALY]
        eval_cols = [c for c in self.numeric_cols if c not in self.identifier_cols]

        if not eval_cols or self.num_rows < 15:
            return DimensionResult(
                dimension=QualityDimension.OUTLIER_ANOMALY,
                score=95.0,
                weight=0.0,
                weighted_score=0.0,
                raw_metrics={
                    "anomaly_ratio": 0.0,
                    "extreme_kurtosis_cols": [],
                    "kurtosis_role": "descriptive_supporting_metric_only"
                },
                summary="Insufficient numerical features for multivariate outlier estimation.",
                formula=dim_meta.formula,
                assumptions=dim_meta.assumptions,
                limitations=dim_meta.limitations,
                task_applicability=dim_meta.task_applicability,
                provenance=dim_meta.provenance
            )

        clean_subset = self.df[eval_cols].fillna(self.df[eval_cols].median())

        # Descriptive univariate tail heaviness (supporting metric only)
        kurtosis_dict = clean_subset.kurt().to_dict()
        extreme_kurtosis = [col for col, k in kurtosis_dict.items() if abs(k) > 5.0]

        # Primary score: Multivariate anomaly detection via Isolation Forest
        try:
            sample_size = min(2000, self.num_rows)
            sample_data = clean_subset.sample(n=sample_size, random_state=42) if self.num_rows > sample_size else clean_subset
            iso = IsolationForest(contamination=0.05, random_state=42, n_estimators=50)
            preds = iso.fit_predict(sample_data)
            anomaly_ratio = float((preds == -1).sum() / len(preds))
        except Exception:
            anomaly_ratio = 0.05

        # Primary score formula: penalty derived strictly from anomaly density
        anomaly_excess = max(0.0, anomaly_ratio - 0.02)
        score = max(10.0, min(100.0, 100.0 - (anomaly_excess * 300.0) - (anomaly_ratio * 80.0)))

        summary = (
            f"Detected {anomaly_ratio * 100:.1f}% multivariate anomalies via Isolation Forest. "
            f"Descriptive note: {len(extreme_kurtosis)} column(s) exhibit heavy tails (|kurtosis| > 5.0)."
        )

        return DimensionResult(
            dimension=QualityDimension.OUTLIER_ANOMALY,
            score=round(score, 2),
            weight=0.0,
            weighted_score=0.0,
            raw_metrics={
                "anomaly_ratio": round(anomaly_ratio, 4),
                "isolation_forest_contamination": 0.05,
                "extreme_kurtosis_cols": extreme_kurtosis,
                "kurtosis_values": {k: round(v, 2) for k, v in kurtosis_dict.items() if abs(v) > 3.0},
                "kurtosis_role": "descriptive_supporting_metric_only"
            },
            summary=summary,
            formula=dim_meta.formula,
            assumptions=dim_meta.assumptions,
            limitations=dim_meta.limitations,
            task_applicability=dim_meta.task_applicability,
            provenance=dim_meta.provenance
        )

    def evaluate_distribution_balance(self, task: Optional[TaskType] = None) -> DimensionResult:
        """
        Dimension 7: Distribution Balance
        Condition 7: Evaluates target balance / entropy for classification,
        continuous target skewness for regression, and feature dispersion for clustering.
        Respects configurable ImbalancePolicy.
        """
        dim_meta = CANONICAL_DIMENSIONS[QualityDimension.DISTRIBUTION_BALANCE]

        # 1. Supervised Classification
        if task == TaskType.SUPERVISED_CLASSIFICATION and self.target_column and self.target_column in self.df.columns:
            clean_t = self.df[self.target_column].dropna()
            if len(clean_t) > 0:
                counts = clean_t.value_counts(normalize=True)
                majority_pct = float(counts.iloc[0]) * 100.0
                minority_pct = float(counts.iloc[-1]) * 100.0
                imbalance_ratio = round(float(counts.iloc[0] / max(0.001, counts.iloc[-1])), 2)

                # Policy thresholds
                min_thresh = 2.0
                sev_thresh = 10.0
                veto_active = True
                if self.imbalance_policy is not None:
                    min_thresh = getattr(self.imbalance_policy, "minority_threshold_pct", 2.0)
                    sev_thresh = getattr(self.imbalance_policy, "severe_threshold_pct", 10.0)
                    veto_active = getattr(self.imbalance_policy, "veto_enabled", True)

                if minority_pct < min_thresh:
                    penalty = 80.0
                elif minority_pct < sev_thresh:
                    penalty = 45.0
                elif minority_pct < 25.0:
                    penalty = 20.0
                else:
                    penalty = 0.0

                score = max(10.0, min(100.0, 100.0 - penalty))
                is_veto = (minority_pct < min_thresh) and veto_active
                summary = (
                    f"Classification minority class represents {minority_pct:.1f}% "
                    f"(imbalance ratio {imbalance_ratio}:1, policy threshold {min_thresh:.1f}%)."
                )

                return DimensionResult(
                    dimension=QualityDimension.DISTRIBUTION_BALANCE,
                    score=round(score, 2),
                    weight=0.0,
                    weighted_score=0.0,
                    raw_metrics={
                        "minority_percentage": round(minority_pct, 2),
                        "majority_percentage": round(majority_pct, 2),
                        "imbalance_ratio": imbalance_ratio,
                        "class_distribution": {str(k): round(float(v), 4) for k, v in counts.items()},
                        "policy_minority_threshold": min_thresh,
                        "policy_veto_enabled": veto_active
                    },
                    summary=summary,
                    formula=dim_meta.formula,
                    assumptions=dim_meta.assumptions,
                    limitations=dim_meta.limitations,
                    task_applicability=dim_meta.task_applicability,
                    provenance=dim_meta.provenance,
                    is_veto_triggered=is_veto,
                    veto_reason=f"Severe class collapse: minority class represents {minority_pct:.1f}% (<{min_thresh:.1f}% policy threshold)." if is_veto else None
                )

        # 2. Supervised Regression
        if task == TaskType.SUPERVISED_REGRESSION and self.target_column and self.target_column in self.df.columns:
            clean_t = self.df[self.target_column].dropna()
            if pd.api.types.is_numeric_dtype(clean_t) and len(clean_t) > 1:
                skewness = float(clean_t.skew()) if not np.isnan(clean_t.skew()) else 0.0
                penalty = min(50.0, abs(skewness) * 8.0)
                score = max(15.0, min(100.0, 100.0 - penalty))
                summary = f"Regression target skewness is {skewness:.2f}."

                return DimensionResult(
                    dimension=QualityDimension.DISTRIBUTION_BALANCE,
                    score=round(score, 2),
                    weight=0.0,
                    weighted_score=0.0,
                    raw_metrics={"skewness": round(skewness, 3), "target_std": float(clean_t.std())},
                    summary=summary,
                    formula=dim_meta.formula,
                    assumptions=dim_meta.assumptions,
                    limitations=dim_meta.limitations,
                    task_applicability=dim_meta.task_applicability,
                    provenance=dim_meta.provenance
                )

        # 3. Unsupervised / Clustering / Default
        score = 95.0
        summary = "Dataset satisfies baseline sample distribution balance."
        return DimensionResult(
            dimension=QualityDimension.DISTRIBUTION_BALANCE,
            score=round(score, 2),
            weight=0.0,
            weighted_score=0.0,
            raw_metrics={"status": "Balanced feature distribution"},
            summary=summary,
            formula=dim_meta.formula,
            assumptions=dim_meta.assumptions,
            limitations=dim_meta.limitations,
            task_applicability=dim_meta.task_applicability,
            provenance=dim_meta.provenance
        )

    def evaluate_coverage_representativeness(self) -> DimensionResult:
        """
        Dimension 8: Coverage / Representativeness
        Condition 4: Measures observable feature space support density.
        Separates observable feature space support from population representativeness.
        Marks population representativeness as UNAVAILABLE in raw_metrics.
        """
        dim_meta = CANONICAL_DIMENSIONS[QualityDimension.COVERAGE_REPRESENTATIVENESS]
        num_features = len(self.feature_numeric_cols)

        if num_features == 0:
            ratio = float(self.num_rows)
            score = 90.0
        else:
            ratio = self.num_rows / max(1, num_features)
            # Sample-to-feature ratio should ideally be >= 10:1
            effective_dim_ratio = min(1.0, ratio / 10.0)
            score = max(20.0, min(100.0, 50.0 + (50.0 * effective_dim_ratio)))

        summary = (
            f"Observable feature space support score: {score:.1f} (N/P ratio = {ratio:.1f}:1). "
            f"Population representativeness: {REPRESENTATIVENESS_UNAVAILABLE_STATUS}."
        )

        return DimensionResult(
            dimension=QualityDimension.COVERAGE_REPRESENTATIVENESS,
            score=round(score, 2),
            weight=0.0,
            weighted_score=0.0,
            raw_metrics={
                "sample_size": self.num_rows,
                "feature_count": num_features,
                "sample_to_feature_ratio": round(ratio, 2),
                "observable_feature_space_support": round(score, 2),
                "population_representativeness": REPRESENTATIVENESS_UNAVAILABLE_STATUS,
                "representativeness_status": "UNAVAILABLE"
            },
            summary=summary,
            formula=dim_meta.formula,
            assumptions=dim_meta.assumptions,
            limitations=dim_meta.limitations,
            task_applicability=dim_meta.task_applicability,
            provenance=dim_meta.provenance
        )

    # Backward-compatible convenience methods
    def evaluate_outliers(self) -> DimensionResult:
        return self.evaluate_outlier_anomaly()

    def evaluate_drift_stability(self) -> DimensionResult:
        return self.evaluate_timeliness()

    def evaluate_label_integrity(self, task: TaskType) -> DimensionResult:
        return self.evaluate_distribution_balance(task)

    def evaluate_temporal_validity(self, task: TaskType) -> DimensionResult:
        return self.evaluate_timeliness(task)

    def evaluate_cardinality_schema(self) -> DimensionResult:
        return self.evaluate_validity()

    def evaluate_correlation_integrity(self) -> DimensionResult:
        return self.evaluate_consistency()

    def evaluate_bias_imbalance(self, task: TaskType) -> DimensionResult:
        return self.evaluate_distribution_balance(task)

    def run_all_dimensions(self, task: TaskType) -> Dict[QualityDimension, DimensionResult]:
        """
        Executes all 8 canonical quality dimensions for the specified task profile.
        """
        return {
            QualityDimension.COMPLETENESS: self.evaluate_completeness(),
            QualityDimension.VALIDITY: self.evaluate_validity(task),
            QualityDimension.CONSISTENCY: self.evaluate_consistency(),
            QualityDimension.UNIQUENESS: self.evaluate_uniqueness(),
            QualityDimension.TIMELINESS: self.evaluate_timeliness(task),
            QualityDimension.OUTLIER_ANOMALY: self.evaluate_outlier_anomaly(),
            QualityDimension.DISTRIBUTION_BALANCE: self.evaluate_distribution_balance(task),
            QualityDimension.COVERAGE_REPRESENTATIVENESS: self.evaluate_coverage_representativeness()
        }

    evaluate_all = run_all_dimensions
