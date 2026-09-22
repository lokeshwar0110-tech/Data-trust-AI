import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from datatrust.models import TaskType, QualityDimension, RemediationActionV2
from datatrust.reproducibility import hash_index


@dataclass
class ActionMetadata:
    action_type: str
    name: str
    description: str
    preconditions: str
    transformation: str
    expected_quality_impact: str
    operational_cost: int
    affected_dimensions: List[QualityDimension]
    train_test_safety_policy: str


ACTION_REGISTRY: Dict[str, ActionMetadata] = {
    "sort_timestamps": ActionMetadata(
        action_type="sort_timestamps",
        name="Temporal Sorting & Deduplication",
        description="Chronological sorting and deduplication of timestamp sequence",
        preconditions="time_column exists and contains datetime-parseable entries",
        transformation="pd.to_datetime -> sort_values -> drop_duplicates on train; parse on test",
        expected_quality_impact="Eliminates non-monotonicity and timestamp collisions (+18 to +50 fitness)",
        operational_cost=2,
        affected_dimensions=[QualityDimension.TIMELINESS],
        train_test_safety_policy="TRAIN: sorted and deduplicated. TEST: immutable row membership and indices strictly preserved."
    ),
    "drop_leakage_columns": ActionMetadata(
        action_type="drop_leakage_columns",
        name="Target Leakage Feature Removal",
        description="Drop feature columns exhibiting near-deterministic (|r| > 0.98) correlation to target",
        preconditions="target_column exists and continuous/numeric with |r| > 0.98 features",
        transformation="drop(columns=leaking_cols) on train and test splits",
        expected_quality_impact="Eliminates spurious predictive shortcuts (+15 to +35 fitness)",
        operational_cost=1,
        affected_dimensions=[QualityDimension.CONSISTENCY, QualityDimension.VALIDITY],
        train_test_safety_policy="FIT on TRAIN: identified strictly on train correlation matrix. Applied deterministically to test."
    ),
    "winsorize_outliers": ActionMetadata(
        action_type="winsorize_outliers",
        name="Heavy-Tail Outlier Winsorization",
        description="Clip extreme values beyond 1st-99th percentiles for heavy-tailed features (|kurtosis| > 5)",
        preconditions="Continuous numeric features with extreme kurtosis > 5.0",
        transformation="clip(lower=q01, upper=q99) on train and test splits",
        expected_quality_impact="Stabilizes quadratic loss (MSE/RMSE) without dropping observations (+5 to +20 fitness)",
        operational_cost=2,
        affected_dimensions=[QualityDimension.OUTLIER_ANOMALY],
        train_test_safety_policy="FIT on TRAIN: 1st and 99th percentiles learned strictly on train. Applied deterministically to test."
    ),
    "resample_imbalance": ActionMetadata(
        action_type="resample_imbalance",
        name="Minority Class Resampling",
        description="Upsample minority target class in classification training set",
        preconditions="Task is SUPERVISED_CLASSIFICATION and minority class representation < 15%",
        transformation="sample(replace=True) on training split minority rows",
        expected_quality_impact="Restores decision boundary balance and minority recall (+15 to +35 fitness)",
        operational_cost=3,
        affected_dimensions=[QualityDimension.DISTRIBUTION_BALANCE],
        train_test_safety_policy="FIT on TRAIN ONLY: Test set is NEVER resampled or synthesized. Test distribution preserved."
    ),
    "impute_median": ActionMetadata(
        action_type="impute_median",
        name="Median/Mode Missing Value Imputation",
        description="Impute missing numeric values using train median and categoricals using train mode",
        preconditions="Dataset contains NaN or null cells",
        transformation="fillna(train_medians) on train and test splits",
        expected_quality_impact="Restores complete feature vectors (+5 to +20 fitness)",
        operational_cost=2,
        affected_dimensions=[QualityDimension.COMPLETENESS],
        train_test_safety_policy="FIT on TRAIN: Medians/modes learned strictly on train. Applied deterministically to test."
    ),
    "drop_redundant_identifiers": ActionMetadata(
        action_type="drop_redundant_identifiers",
        name="Redundant Identifier & Zero-Variance Column Drop",
        description="Drop non-target columns with single unique value or >95% unique object keys",
        preconditions="Zero-variance or high-cardinality unique string columns",
        transformation="drop(columns=redundant_cols) on train and test splits",
        expected_quality_impact="Reduces dimensionality and eliminates singular covariance matrices (+5 to +10 fitness)",
        operational_cost=1,
        affected_dimensions=[QualityDimension.VALIDITY, QualityDimension.UNIQUENESS],
        train_test_safety_policy="FIT on TRAIN: Columns identified on train. Applied deterministically to test."
    ),
    "drop_unlabelled_target": ActionMetadata(
        action_type="drop_unlabelled_target",
        name="Drop Unlabelled Target Records",
        description="Drop training observations where the supervised target column contains missing or null values",
        preconditions="target_column exists and contains missing/null entries",
        transformation="dropna(subset=[target_column]) on training split",
        expected_quality_impact="Removes unusable unlabelled training instances without corrupting labels (+10 to +20 fitness)",
        operational_cost=1,
        affected_dimensions=[QualityDimension.VALIDITY, QualityDimension.COMPLETENESS],
        train_test_safety_policy="TRAIN: dropped unlabelled rows. TEST: strictly preserved (immutable row membership and indices)."
    ),
    "drop_high_null_columns": ActionMetadata(
        action_type="drop_high_null_columns",
        name="High-Missing Feature Removal",
        description="Drop feature columns with excessive missingness rate exceeding 40%",
        preconditions="Non-target, non-timestamp columns with >40% missing values",
        transformation="drop(columns=high_null_cols) on train and test splits",
        expected_quality_impact="Eliminates high-sparsity noise features (+5 to +15 fitness)",
        operational_cost=1,
        affected_dimensions=[QualityDimension.COMPLETENESS, QualityDimension.VALIDITY],
        train_test_safety_policy="FIT on TRAIN: columns with >40% missingness identified on train. Applied deterministically to test."
    )
}

ACTION_ALIASES: Dict[str, List[str]] = {
    "sort_timestamps": [
        "sort_timestamps", "temporal_sort_and_deduplicate", "act_sort", "act_temporal",
        "act_time", "sort_time", "deduplicate_timestamps"
    ],
    "drop_leakage_columns": [
        "drop_leakage_columns", "drop_target_leakage", "act_leakage", "act_leak",
        "drop_leakage", "drop_leaking_columns"
    ],
    "winsorize_outliers": [
        "winsorize_outliers", "clip_heavy_tail_outliers", "act_winsorize", "act_outlier",
        "act_outliers", "winsorize", "clip_outliers"
    ],
    "resample_imbalance": [
        "resample_imbalance", "resample_classification_imbalance", "act_resample",
        "act_imbalance", "resample_minority", "balance_classes"
    ],
    "impute_median": [
        "impute_median", "impute_missing_median", "impute_missing_values", "act_impute",
        "act_missing", "impute_nulls"
    ],
    "drop_redundant_identifiers": [
        "drop_redundant_identifiers", "drop_constant_or_id_columns", "act_drop_id",
        "act_identifiers", "drop_identifiers"
    ],
    "drop_unlabelled_target": [
        "drop_unlabelled_target", "drop_missing_target", "drop_unlabeled_target",
        "act_unlabelled", "act_unlabeled"
    ],
    "drop_high_null_columns": [
        "drop_high_null_columns", "drop_sparse_columns", "drop_missing_columns",
        "act_null_cols", "act_high_null"
    ]
}


class DataRemediator:
    """
    Subsystem for Executing Action-Authoritative Dataset Remediation Interventions.
    Applies recommended data transformations according to an authoritative Action Registry.
    
    CRITICAL GOVERNING PRINCIPLE:
    If the optimizer selects action_ids = [A, B], then ONLY A and B may be executed.
    Unrelated transformations MUST NOT be silently executed.
    Every execution tracks requested, executed, and skipped actions with reasons.
    """

    @classmethod
    def is_action_requested(cls, action_type: str, action_ids: Optional[List[str]]) -> bool:
        """
        Determines whether a canonical action_type is explicitly requested in action_ids.
        Supports canonical names, action aliases, and generic IDs.
        If action_ids is None, all applicable actions are allowed (unconstrained baseline).
        If action_ids is a list (even empty), ONLY explicitly matching actions are executed.
        """
        if action_ids is None:
            return True
        if "all" in action_ids:
            return True

        valid_aliases = ACTION_ALIASES.get(action_type, [action_type])
        for req in action_ids:
            if req in valid_aliases:
                return True
            # Handle prefix matching, e.g. "act_sort" matching "sort_timestamps"
            for alias in valid_aliases:
                if req == alias or req.startswith(f"{alias}_") or alias.startswith(f"{req}_"):
                    return True
        return False

    @classmethod
    def apply_remediations(
        cls,
        df: pd.DataFrame,
        task: TaskType,
        action_ids: Optional[List[str]] = None,
        target_column: Optional[str] = None,
        time_column: Optional[str] = None
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Applies requested remediation actions in an action-authoritative manner.
        """
        remediated_df = df.copy()
        execution_log: List[str] = []
        executed_actions: List[str] = []
        skipped_actions: Dict[str, str] = {}

        # 1. Chronological Sorting & Deduplication (Time Series)
        if cls.is_action_requested("sort_timestamps", action_ids):
            if time_column and time_column in remediated_df.columns:
                try:
                    raw_parsed = pd.to_datetime(remediated_df[time_column], errors='coerce')
                    raw_dups = int(raw_parsed.duplicated().sum())
                    raw_dup_rate = round((raw_dups / max(1, len(raw_parsed))) * 100.0, 1)
                    raw_monotonic = bool(raw_parsed.is_monotonic_increasing)

                    remediated_df[time_column] = raw_parsed
                    remediated_df = remediated_df.dropna(subset=[time_column])
                    remediated_df = remediated_df.sort_values(time_column).reset_index(drop=True)
                    initial_count = len(remediated_df)
                    remediated_df = remediated_df.drop_duplicates(subset=[time_column], keep='first').reset_index(drop=True)
                    dups_removed = initial_count - len(remediated_df)

                    rem_parsed = remediated_df[time_column]
                    rem_dups = int(rem_parsed.duplicated().sum())
                    rem_dup_rate = round((rem_dups / max(1, len(rem_parsed))) * 100.0, 1)

                    execution_log.append(
                        f"RAW DATASET: Duplicate timestamps: {raw_dups}, Duplicate rate: {raw_dup_rate}%, Monotonic: {raw_monotonic}"
                    )
                    execution_log.append(
                        f"EXECUTED 'sort_timestamps': Chronologically sorted by '{time_column}', removed {dups_removed} duplicate records."
                    )
                    execution_log.append(
                        f"REMEDIATED DATASET: {rem_dups} duplicate timestamps, {rem_dup_rate}% duplicate rate, Monotonic: True"
                    )
                    executed_actions.append("sort_timestamps")
                except Exception as e:
                    skipped_actions["sort_timestamps"] = f"Error during execution: {str(e)}"
            else:
                skipped_actions["sort_timestamps"] = "Precondition failed: time_column unassigned or absent"
        elif action_ids is not None:
            skipped_actions["sort_timestamps"] = "Not in requested action_ids"

        # 2. Target Leakage Column Removal
        if cls.is_action_requested("drop_leakage_columns", action_ids):
            if target_column and target_column in remediated_df.columns:
                numeric_cols = [c for c in remediated_df.select_dtypes(include=[np.number]).columns if c != target_column]
                if len(numeric_cols) > 0 and pd.api.types.is_numeric_dtype(remediated_df[target_column]):
                    corr = remediated_df[numeric_cols + [target_column]].corr().abs()[target_column]
                    leaking_cols = [col for col in numeric_cols if corr.get(col, 0) > 0.98]
                    if leaking_cols:
                        remediated_df = remediated_df.drop(columns=leaking_cols)
                        execution_log.append(f"EXECUTED 'drop_leakage_columns': Dropped target-leaking column(s): {leaking_cols} (|r| > 0.98).")
                        executed_actions.append("drop_leakage_columns")
                    else:
                        skipped_actions["drop_leakage_columns"] = "No features exceeded leakage correlation threshold (|r| > 0.98)"
                else:
                    skipped_actions["drop_leakage_columns"] = "Precondition failed: insufficient numeric features or non-numeric target"
            else:
                skipped_actions["drop_leakage_columns"] = "Precondition failed: target_column unassigned or absent"
        elif action_ids is not None:
            skipped_actions["drop_leakage_columns"] = "Not in requested action_ids"

        # 3. Heavy-Tail Outlier Winsorization
        if cls.is_action_requested("winsorize_outliers", action_ids):
            target_is_continuous = (
                task in [TaskType.SUPERVISED_REGRESSION, TaskType.TIME_SERIES_FORECASTING]
                and target_column in remediated_df.columns
                and pd.api.types.is_numeric_dtype(remediated_df[target_column])
            )
            num_cols = list(remediated_df.select_dtypes(include=[np.number]).columns)
            if not target_is_continuous and target_column in num_cols:
                num_cols.remove(target_column)

            outlier_cols_treated = []
            for col in num_cols:
                series = remediated_df[col].dropna()
                if len(series) > 10:
                    kurt = series.kurt()
                    if abs(kurt) > 5.0:
                        q01 = series.quantile(0.01)
                        q99 = series.quantile(0.99)
                        remediated_df[col] = remediated_df[col].clip(lower=q01, upper=q99)
                        outlier_cols_treated.append(col)
            if outlier_cols_treated:
                execution_log.append(f"EXECUTED 'winsorize_outliers': Applied 1st-99th percentile Winsorization to: {outlier_cols_treated}.")
                executed_actions.append("winsorize_outliers")
            else:
                skipped_actions["winsorize_outliers"] = "No continuous features exhibited extreme kurtosis (|kurt| > 5.0)"
        elif action_ids is not None:
            skipped_actions["winsorize_outliers"] = "Not in requested action_ids"

        # 4. Supervised Classification Imbalance Resampling
        if cls.is_action_requested("resample_imbalance", action_ids):
            if task == TaskType.SUPERVISED_CLASSIFICATION and target_column and target_column in remediated_df.columns:
                val_counts = remediated_df[target_column].value_counts()
                if len(val_counts) > 1:
                    min_class = val_counts.idxmin()
                    maj_class = val_counts.idxmax()
                    min_count = val_counts[min_class]
                    maj_count = val_counts[maj_class]
                    min_pct = (min_count / len(remediated_df)) * 100.0

                    if min_pct < 15.0:
                        target_min_count = max(min_count * 2, int(maj_count * 0.40))
                        min_subset = remediated_df[remediated_df[target_column] == min_class]
                        upsampled = min_subset.sample(target_min_count, replace=True, random_state=42)
                        remediated_df = pd.concat([
                            remediated_df[remediated_df[target_column] != min_class],
                            upsampled
                        ], ignore_index=True)
                        execution_log.append(
                            f"EXECUTED 'resample_imbalance': Balanced minority class '{min_class}' via resampling ({min_count} -> {target_min_count} records)."
                        )
                        executed_actions.append("resample_imbalance")
                    else:
                        skipped_actions["resample_imbalance"] = f"Class balance acceptable (minority class is {min_pct:.1f}% >= 15%)"
                else:
                    skipped_actions["resample_imbalance"] = "Target has <= 1 unique class"
            else:
                skipped_actions["resample_imbalance"] = "Task is not classification or target unassigned"
        elif action_ids is not None:
            skipped_actions["resample_imbalance"] = "Not in requested action_ids"

        # 5. Missing Value Imputation
        if cls.is_action_requested("impute_median", action_ids):
            missing_count_before = int(remediated_df.isna().sum().sum())
            if missing_count_before > 0:
                for col in remediated_df.columns:
                    if remediated_df[col].isna().sum() > 0:
                        if pd.api.types.is_numeric_dtype(remediated_df[col]):
                            remediated_df[col] = remediated_df[col].fillna(remediated_df[col].median())
                        else:
                            mode_val = remediated_df[col].mode()
                            fill_val = mode_val.iloc[0] if len(mode_val) > 0 else "Unknown"
                            remediated_df[col] = remediated_df[col].fillna(fill_val)
                execution_log.append(f"EXECUTED 'impute_median': Imputed {missing_count_before} missing cells using train medians/modes.")
                executed_actions.append("impute_median")
            else:
                skipped_actions["impute_median"] = "Dataset has zero missing cells"
        elif action_ids is not None:
            skipped_actions["impute_median"] = "Not in requested action_ids"

        # 6. Drop Uninformative Unique Identifiers or Zero-Variance Columns
        if cls.is_action_requested("drop_redundant_identifiers", action_ids):
            drop_id_cols = []
            for col in remediated_df.columns:
                if col != target_column and col != time_column:
                    uniq = remediated_df[col].nunique(dropna=True)
                    if uniq <= 1:
                        drop_id_cols.append(col)
                    elif remediated_df[col].dtype == object and uniq > (0.95 * len(remediated_df)):
                        drop_id_cols.append(col)
            if drop_id_cols:
                remediated_df = remediated_df.drop(columns=drop_id_cols)
                execution_log.append(f"EXECUTED 'drop_redundant_identifiers': Dropped redundant column(s): {drop_id_cols}.")
                executed_actions.append("drop_redundant_identifiers")
            else:
                skipped_actions["drop_redundant_identifiers"] = "No zero-variance or unique identifier columns detected"
        elif action_ids is not None:
            skipped_actions["drop_redundant_identifiers"] = "Not in requested action_ids"

        # 7. Drop Unlabelled Target Records (Supervised)
        if cls.is_action_requested("drop_unlabelled_target", action_ids):
            if target_column and target_column in remediated_df.columns:
                target_nulls = int(remediated_df[target_column].isna().sum())
                if target_nulls > 0:
                    remediated_df = remediated_df.dropna(subset=[target_column]).reset_index(drop=True)
                    execution_log.append(
                        f"EXECUTED 'drop_unlabelled_target': Dropped {target_nulls} records with unlabelled target '{target_column}'."
                    )
                    executed_actions.append("drop_unlabelled_target")
                else:
                    skipped_actions["drop_unlabelled_target"] = "Target column contains zero null values"
            else:
                skipped_actions["drop_unlabelled_target"] = "Precondition failed: target_column unassigned or absent"
        elif action_ids is not None:
            skipped_actions["drop_unlabelled_target"] = "Not in requested action_ids"

        # 8. Drop High-Null Columns (>40% missing)
        if cls.is_action_requested("drop_high_null_columns", action_ids):
            candidate_cols = [c for c in remediated_df.columns if c != target_column and c != time_column]
            high_null_cols = [c for c in candidate_cols if (remediated_df[c].isna().sum() / max(1, len(remediated_df))) > 0.40]
            if high_null_cols:
                remediated_df = remediated_df.drop(columns=high_null_cols)
                execution_log.append(
                    f"EXECUTED 'drop_high_null_columns': Dropped high-missing column(s): {high_null_cols} (>40% null)."
                )
                executed_actions.append("drop_high_null_columns")
            else:
                skipped_actions["drop_high_null_columns"] = "No non-target columns exceeded 40% missingness threshold"
        elif action_ids is not None:
            skipped_actions["drop_high_null_columns"] = "Not in requested action_ids"

        return remediated_df, execution_log

    @classmethod
    def fit_and_transform_splits(
        cls,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        task: TaskType,
        action_ids: Optional[List[str]] = None,
        target_column: Optional[str] = None,
        time_column: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], Dict[str, Any]]:
        """
        Executes strict out-of-sample, action-authoritative remediation transformations:
        1. Action-Authoritative: executes ONLY requested actions in action_ids.
        2. Fits all parameters EXCLUSIVELY on train_df.
        3. Applies fitted transformations to train_df and resamples ONLY on train_df.
        4. Applies ONLY deterministic fitted transformations to test_df.
        5. Preserves test_df row membership, ordering, and indices (immutability guaranteed).
        """
        rem_train = train_df.copy()
        rem_test = test_df.copy()

        # Canonical index hashing before remediation
        test_index_hash_before = hash_index(test_df.index)
        execution_log: List[str] = []
        fitted_params: Dict[str, Any] = {}
        executed_actions: List[str] = []
        skipped_actions: Dict[str, str] = {}

        # 1. Chronological Sorting & Deduplication (Time Series)
        if cls.is_action_requested("sort_timestamps", action_ids):
            if time_column and time_column in rem_train.columns:
                try:
                    rem_train[time_column] = pd.to_datetime(rem_train[time_column], errors='coerce')
                    rem_train = rem_train.dropna(subset=[time_column]).sort_values(time_column).reset_index(drop=True)
                    dups_before = len(rem_train)
                    rem_train = rem_train.drop_duplicates(subset=[time_column], keep='first').reset_index(drop=True)
                    dups_removed = dups_before - len(rem_train)
                    execution_log.append(f"TRAIN: Sorted chronologically by '{time_column}' and removed {dups_removed} duplicate timestamps.")

                    if time_column in rem_test.columns:
                        rem_test[time_column] = pd.to_datetime(rem_test[time_column], errors='coerce')
                        execution_log.append(f"TEST: Verified timestamps for '{time_column}'. Test membership and indices strictly preserved (0 rows dropped or deduplicated).")

                    executed_actions.append("sort_timestamps")
                except Exception as e:
                    skipped_actions["sort_timestamps"] = f"Error during execution: {str(e)}"
            else:
                skipped_actions["sort_timestamps"] = "Precondition failed: time_column absent"
        elif action_ids is not None:
            skipped_actions["sort_timestamps"] = "Not in requested action_ids"

        # 2. Target Leakage Column Removal (Fitted solely on train)
        if cls.is_action_requested("drop_leakage_columns", action_ids):
            if target_column and target_column in rem_train.columns:
                num_cols_tr = [c for c in rem_train.select_dtypes(include=[np.number]).columns if c != target_column]
                if len(num_cols_tr) > 0 and pd.api.types.is_numeric_dtype(rem_train[target_column]):
                    corr = rem_train[num_cols_tr + [target_column]].corr().abs()[target_column]
                    leaking_cols = [col for col in num_cols_tr if corr.get(col, 0) > 0.98]
                    fitted_params["dropped_leakage_cols"] = leaking_cols
                    if leaking_cols:
                        rem_train = rem_train.drop(columns=leaking_cols)
                        rem_test = rem_test.drop(columns=[c for c in leaking_cols if c in rem_test.columns])
                        execution_log.append(f"FIT on TRAIN: Dropped target-leaking column(s) {leaking_cols} (|r| > 0.98) from train and test splits.")
                        executed_actions.append("drop_leakage_columns")
                    else:
                        skipped_actions["drop_leakage_columns"] = "No features exceeded leakage threshold (|r| > 0.98)"
                else:
                    skipped_actions["drop_leakage_columns"] = "Precondition failed: insufficient numeric features or non-numeric target"
            else:
                skipped_actions["drop_leakage_columns"] = "Precondition failed: target_column absent"
        elif action_ids is not None:
            skipped_actions["drop_leakage_columns"] = "Not in requested action_ids"

        # 3. Heavy-Tail Outlier Winsorization (Fitted solely on train)
        if cls.is_action_requested("winsorize_outliers", action_ids):
            target_is_continuous = (
                task in [TaskType.SUPERVISED_REGRESSION, TaskType.TIME_SERIES_FORECASTING]
                and target_column in rem_train.columns
                and pd.api.types.is_numeric_dtype(rem_train[target_column])
            )
            num_cols = list(rem_train.select_dtypes(include=[np.number]).columns)
            if not target_is_continuous and target_column in num_cols:
                num_cols.remove(target_column)

            winsor_bounds = {}
            for col in num_cols:
                series = rem_train[col].dropna()
                if len(series) > 10:
                    kurt = series.kurt()
                    if abs(kurt) > 5.0:
                        q01 = float(series.quantile(0.01))
                        q99 = float(series.quantile(0.99))
                        winsor_bounds[col] = (q01, q99)
                        rem_train[col] = rem_train[col].clip(lower=q01, upper=q99)
                        if col in rem_test.columns:
                            rem_test[col] = rem_test[col].clip(lower=q01, upper=q99)

            fitted_params["winsor_bounds"] = winsor_bounds
            if winsor_bounds:
                execution_log.append(f"FIT on TRAIN: Learned 1st-99th percentile Winsorization bounds on {list(winsor_bounds.keys())}; applied deterministically to test.")
                executed_actions.append("winsorize_outliers")
            else:
                skipped_actions["winsorize_outliers"] = "No continuous features exhibited extreme kurtosis (|kurt| > 5.0)"
        elif action_ids is not None:
            skipped_actions["winsorize_outliers"] = "Not in requested action_ids"

        # 4. Supervised Classification Imbalance Resampling (TRAIN ONLY - NEVER TEST)
        if cls.is_action_requested("resample_imbalance", action_ids):
            if task == TaskType.SUPERVISED_CLASSIFICATION and target_column and target_column in rem_train.columns:
                val_counts = rem_train[target_column].value_counts()
                if len(val_counts) > 1:
                    min_class = val_counts.idxmin()
                    maj_class = val_counts.idxmax()
                    min_count = int(val_counts[min_class])
                    maj_count = int(val_counts[maj_class])
                    min_pct = (min_count / len(rem_train)) * 100.0

                    if min_pct < 15.0:
                        target_min_count = max(min_count * 2, int(maj_count * 0.40))
                        min_subset = rem_train[rem_train[target_column] == min_class]
                        upsampled = min_subset.sample(target_min_count, replace=True, random_state=42)
                        rem_train = pd.concat([
                            rem_train[rem_train[target_column] != min_class],
                            upsampled
                        ], ignore_index=True)
                        execution_log.append(
                            f"FIT on TRAIN ONLY: Balanced minority class '{min_class}' via resampling ({min_count} -> {target_min_count} records). "
                            f"TEST SET PRESERVED: Zero test resampling applied (test minority count: {int((rem_test[target_column] == min_class).sum()) if target_column in rem_test.columns else 0})."
                        )
                        executed_actions.append("resample_imbalance")
                    else:
                        skipped_actions["resample_imbalance"] = f"Class balance acceptable (minority is {min_pct:.1f}% >= 15%)"
                else:
                    skipped_actions["resample_imbalance"] = "Target has <= 1 class"
            else:
                skipped_actions["resample_imbalance"] = "Task is not classification or target unassigned"
        elif action_ids is not None:
            skipped_actions["resample_imbalance"] = "Not in requested action_ids"

        # 5. Missing Value Imputation (Fitted solely on train)
        if cls.is_action_requested("impute_median", action_ids):
            imputation_values = {}
            for col in rem_train.columns:
                if rem_train[col].isna().sum() > 0:
                    if pd.api.types.is_numeric_dtype(rem_train[col]):
                        fill_val = float(rem_train[col].median())
                    else:
                        mode_s = rem_train[col].mode()
                        fill_val = mode_s.iloc[0] if len(mode_s) > 0 else "Unknown"
                    imputation_values[col] = fill_val
                    rem_train[col] = rem_train[col].fillna(fill_val)
                    if col in rem_test.columns:
                        rem_test[col] = rem_test[col].fillna(fill_val)

            fitted_params["imputation_values"] = imputation_values
            if imputation_values:
                execution_log.append(f"FIT on TRAIN: Imputed missing cells using train medians/modes on columns {list(imputation_values.keys())}.")
                executed_actions.append("impute_median")
            else:
                skipped_actions["impute_median"] = "Dataset has zero missing cells"
        elif action_ids is not None:
            skipped_actions["impute_median"] = "Not in requested action_ids"

        # 6. Drop Redundant Unique Identifiers or Zero-Variance Columns (Fitted on train)
        if cls.is_action_requested("drop_redundant_identifiers", action_ids):
            drop_id_cols = []
            for col in rem_train.columns:
                if col != target_column and col != time_column:
                    uniq = rem_train[col].nunique(dropna=True)
                    if uniq <= 1:
                        drop_id_cols.append(col)
                    elif rem_train[col].dtype == object and uniq > (0.95 * len(rem_train)):
                        drop_id_cols.append(col)

            fitted_params["dropped_identifier_cols"] = drop_id_cols
            if drop_id_cols:
                rem_train = rem_train.drop(columns=drop_id_cols)
                rem_test = rem_test.drop(columns=[c for c in drop_id_cols if c in rem_test.columns])
                execution_log.append(f"FIT on TRAIN: Dropped redundant/zero-variance column(s) {drop_id_cols} from train and test splits.")
                executed_actions.append("drop_redundant_identifiers")
            else:
                skipped_actions["drop_redundant_identifiers"] = "No zero-variance or identifier columns detected"
        elif action_ids is not None:
            skipped_actions["drop_redundant_identifiers"] = "Not in requested action_ids"

        # 7. Drop Unlabelled Target Records (TRAIN ONLY - NEVER TEST to preserve test immutability)
        if cls.is_action_requested("drop_unlabelled_target", action_ids):
            if target_column and target_column in rem_train.columns:
                target_nulls = int(rem_train[target_column].isna().sum())
                if target_nulls > 0:
                    rem_train = rem_train.dropna(subset=[target_column]).reset_index(drop=True)
                    test_nulls = int(rem_test[target_column].isna().sum()) if target_column in rem_test.columns else 0
                    execution_log.append(
                        f"FIT on TRAIN ONLY: Dropped {target_nulls} unlabelled target records from train. "
                        f"TEST SET PRESERVED: Test membership and indices strictly preserved ({test_nulls} nulls retained)."
                    )
                    executed_actions.append("drop_unlabelled_target")
                else:
                    skipped_actions["drop_unlabelled_target"] = "Target column has zero null values in train"
            else:
                skipped_actions["drop_unlabelled_target"] = "Precondition failed: target_column absent"
        elif action_ids is not None:
            skipped_actions["drop_unlabelled_target"] = "Not in requested action_ids"

        # 8. Drop High-Null Columns (Fitted on train, applied deterministically to test)
        if cls.is_action_requested("drop_high_null_columns", action_ids):
            candidate_cols = [c for c in rem_train.columns if c != target_column and c != time_column]
            high_null_cols = [c for c in candidate_cols if (rem_train[c].isna().sum() / max(1, len(rem_train))) > 0.40]
            fitted_params["dropped_high_null_cols"] = high_null_cols
            if high_null_cols:
                rem_train = rem_train.drop(columns=high_null_cols)
                rem_test = rem_test.drop(columns=[c for c in high_null_cols if c in rem_test.columns])
                execution_log.append(f"FIT on TRAIN: Dropped high-missing column(s) {high_null_cols} (>40% null) from train and test splits.")
                executed_actions.append("drop_high_null_columns")
            else:
                skipped_actions["drop_high_null_columns"] = "No non-target columns exceeded 40% missingness threshold in train"
        elif action_ids is not None:
            skipped_actions["drop_high_null_columns"] = "Not in requested action_ids"

        # Canonical index verification after remediation
        test_index_hash_after = hash_index(rem_test.index)
        assert test_index_hash_before == test_index_hash_after, (
            f"Test set index mutability violation: before={test_index_hash_before}, after={test_index_hash_after}"
        )
        assert len(rem_test) == len(test_df), (
            f"Test set sample count violation: before={len(test_df)}, after={len(rem_test)}"
        )

        fitted_params["requested_actions"] = action_ids if action_ids is not None else list(ACTION_REGISTRY.keys())
        fitted_params["executed_actions"] = executed_actions
        fitted_params["skipped_actions"] = skipped_actions
        fitted_params["test_index_hash_before"] = test_index_hash_before
        fitted_params["test_index_hash_after"] = test_index_hash_after
        fitted_params["test_indices_immutable"] = True

        return rem_train, rem_test, execution_log, fitted_params
