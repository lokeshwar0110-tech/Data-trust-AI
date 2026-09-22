import hashlib
import re
import warnings
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from datatrust.models import ProfilingMetadata
from datatrust.config import PROVENANCE_PROFILING


class DataProfiler:
    """
    Extracts comprehensive statistical metadata and data profiles from DataFrames.
    Computes deterministic dataset SHA-256 and identifies candidate targets and timestamps.
    """

    TARGET_NAME_CANDIDATES = [
        "target", "label", "class", "churn", "fraud", "y", "outcome",
        "response", "status", "price", "sales", "revenue", "demand", "count"
    ]

    TIME_NAME_CANDIDATES = [
        "date", "timestamp", "time", "datetime", "dt", "day", "period", "year"
    ]

    @staticmethod
    def compute_sha256(df: pd.DataFrame, raw_bytes: Optional[bytes] = None) -> str:
        """
        Computes deterministic SHA-256 hash of dataset.
        Uses raw bytes if provided, else hashes pandas object deterministically.
        """
        if raw_bytes is not None and len(raw_bytes) > 0:
            return hashlib.sha256(raw_bytes).hexdigest()
        try:
            hashed_values = pd.util.hash_pandas_object(df, index=True).values.tobytes()
            return hashlib.sha256(hashed_values).hexdigest()
        except Exception:
            return "unavailable"

    @classmethod
    def is_likely_datetime_series(cls, series: pd.Series, col_name: str) -> bool:
        """
        Determines whether a series represents genuine temporal/date data.
        Numeric continuous measurements (floats) are strictly disallowed.
        """
        if pd.api.types.is_datetime64_any_dtype(series) or isinstance(series.dtype, pd.PeriodDtype):
            return True
        if pd.api.types.is_numeric_dtype(series):
            # Numeric continuous float measurements are NEVER timestamps
            if pd.api.types.is_float_dtype(series):
                return False
            # Integers: check if calendar year in [1800, 2100] with explicit year/time naming
            time_tokens = re.compile(r'(?i)(^|[_\s-])(year|yr|date|timestamp|datetime|time)([_\s-]|$)')
            if time_tokens.search(col_name):
                s_clean = series.dropna()
                if len(s_clean) > 0 and s_clean.between(1800, 2100).all():
                    return True
            return False

        # Non-numeric string/object columns
        s_clean = series.dropna()
        if len(s_clean) == 0:
            return False
        sample = s_clean.head(30)

        # If sample values are purely numeric strings (e.g. '1.5', '3.2'), do not treat as datetime
        try:
            pd.to_numeric(sample, errors='raise')
            return False
        except Exception:
            pass

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            parsed = pd.to_datetime(sample, format='mixed', errors='coerce')

        valid_count = int(parsed.notna().sum())
        if valid_count / len(sample) >= 0.80:
            years = parsed.dt.year.dropna()
            if len(years) > 0 and (years.between(1900, 2100).mean() >= 0.80):
                return True
        return False

    @classmethod
    def find_candidate_timestamps(cls, df: pd.DataFrame, explicit_time: Optional[str] = None) -> List[str]:
        """
        Discovers candidate timestamp/date columns based on verified semantic evidence.
        """
        if explicit_time and explicit_time in df.columns:
            return [explicit_time]
        time_tokens = re.compile(r'(?i)(^|[_\s-])(date|timestamp|datetime|time|day|period|year|month)([_\s-]|$)')
        candidates: List[str] = []
        for col in df.columns:
            s = df[col]
            if cls.is_likely_datetime_series(s, col):
                if col not in candidates:
                    candidates.append(col)
            elif time_tokens.search(col) and not pd.api.types.is_float_dtype(s):
                s_clean = s.dropna()
                if len(s_clean) > 0:
                    if pd.api.types.is_integer_dtype(s_clean) and s_clean.between(1800, 2100).all():
                        if col not in candidates:
                            candidates.append(col)
                    elif s.dtype == object:
                        with warnings.catch_warnings():
                            warnings.simplefilter('ignore')
                            parsed = pd.to_datetime(s_clean.head(20), errors='coerce')
                        if parsed.notna().mean() >= 0.50:
                            if col not in candidates:
                                candidates.append(col)
        return candidates

    @classmethod
    def find_candidate_target_details(cls, df: pd.DataFrame, explicit_target: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Discovers and ranks candidate target columns from data distribution (categorical, continuous)
        and structural/semantic evidence, ranking highest confidence first.
        Returns rich metadata: column, target_type ('regression' | 'classification'), score, confidence, evidence.
        """
        if explicit_target and explicit_target in df.columns:
            s = df[explicit_target].dropna()
            is_num = pd.api.types.is_numeric_dtype(s)
            t_type = "regression" if (is_num and s.nunique() > 15) else "classification"
            return [{
                "column": explicit_target,
                "target_type": t_type,
                "score": 100.0,
                "confidence": 1.0,
                "evidence": [f"Authoritative explicit target designated: '{explicit_target}'"]
            }]

        id_regex = re.compile(
            r'(?i)(^|[_\s-])(id|uuid|guid|index|key|row|record|num|code|hash)([_\s-]|$)|\b(row_?number|account_?id|transaction_?id|record_?id|customer_?id|user_?id|session_?id|order_?id|item_?id)\b'
        )
        cat_target_regex = re.compile(
            r'(?i)(^|[_\s-])(target|label|class|species|churn|fraud|outcome|response|status|'
            r'diagnosis|sentiment|survived|rating|result|decision|category|type|group)([_\s-]|$)'
        )
        reg_target_regex = re.compile(
            r'(?i)(^|[_\s-])(price|sales|revenue|demand|salary|cost|loss|profit|yield|'
            r'amount|score|count|value|target|outcome|response|measurement)([_\s-]|$)|^y$'
        )
        feature_suppression_regex = re.compile(
            r'(?i)(^|[_\s-])('
            r'coord|coordinate|pos|position|lat|latitude|lon|long|longitude|geo|altitude|elevation|'
            r'feat|feature|var|variable|attr|attribute|dim|dimension|sensor|channel|input|covariate|pred|predictor|'
            r'age|customer_age|year_born|dob|gender|sex|zip|zipcode|postal|phone|ssn'
            r')([_\s\d-]|$)|'
            r'^(x|z|f|v|c|col|column|feat|feature|var|dim|sensor)[_\s-]?\d+$|^[xz]$'
        )

        n_rows = len(df)
        cols = list(df.columns)
        numeric_cols_count = len(df.select_dtypes(include=[np.number]).columns)

        candidate_details = []

        for idx, col in enumerate(cols):
            s = df[col].dropna()
            if len(s) == 0:
                continue
            n_unique = s.nunique()
            score = 0.0
            evidence = []
            target_type = "classification"

            # 1. Timestamp / Datetime Protection
            if cls.is_likely_datetime_series(s, col):
                continue

            # 2. Identifier / Sequential Index Filter
            is_id = False
            if id_regex.search(col):
                is_id = True
            elif not pd.api.types.is_float_dtype(s):
                if pd.api.types.is_integer_dtype(s) and len(s) > 10:
                    if (s.diff().dropna() == 1).all():
                        is_id = True
                    elif (s.is_monotonic_increasing or s.is_monotonic_decreasing) and (n_unique / len(s)) > 0.90:
                        is_id = True
                if n_unique == n_rows and n_rows > 20:
                    is_id = True

            if is_id:
                continue

            # 3. Known Feature / Spatial / Demographic Suppression Check
            is_suppressed_feature = bool(feature_suppression_regex.search(col))

            # 4. Categorical Classification Target Evidence
            is_cat_dtype = (s.dtype == object or s.dtype.name == 'category')
            is_int_classes = pd.api.types.is_integer_dtype(s) and (
                (n_unique == 2 and set(s.unique()).issubset({0, 1})) or
                (2 <= n_unique <= 10 and cat_target_regex.search(col))
            )

            if (is_cat_dtype or is_int_classes) and not is_suppressed_feature:
                if 2 <= n_unique <= 30 and (n_unique / max(1, len(s)) < 0.50):
                    target_type = "classification"
                    if cat_target_regex.search(col):
                        score += 60.0
                        evidence.append(f"Semantic categorical target keyword in '{col}'")
                        if 2 <= n_unique <= 10:
                            score += 15.0
                            evidence.append(f"Discrete multi-class distribution ({n_unique} classes)")
                        if idx == len(cols) - 1:
                            score += 20.0
                            evidence.append("Terminal column in schema")
                        if is_int_classes and set(s.unique()).issubset({0, 1}):
                            score += 20.0
                            evidence.append("Binary [0, 1] indicator distribution")
                    elif idx == len(cols) - 1 and 2 <= n_unique <= 10:
                        score += 35.0
                        evidence.append(f"Terminal column with {n_unique} distinct classes")
                        if is_int_classes and set(s.unique()).issubset({0, 1}):
                            score += 20.0
                            evidence.append("Binary [0, 1] indicator distribution")
                        elif is_cat_dtype:
                            score += 20.0
                            evidence.append("Non-numeric discrete categorical distribution")
                        if numeric_cols_count >= 2:
                            score += 15.0
                            evidence.append(f"Supported by {numeric_cols_count} candidate predictor features")

            # 5. Continuous Numeric Regression Target Evidence
            elif pd.api.types.is_numeric_dtype(s) and not is_suppressed_feature:
                std_val = float(s.std()) if len(s) > 1 else 0.0
                val_range = float(s.max() - s.min()) if len(s) > 0 else 0.0

                # Reject constant / near-zero variance
                if std_val > 1e-7 and n_unique >= 5:
                    # Check for arithmetic index progression (e.g. constant step)
                    is_arithmetic_index = False
                    if pd.api.types.is_integer_dtype(s) and len(s) > 10:
                        diffs = s.diff().dropna()
                        if len(diffs) > 0 and (diffs == diffs.iloc[0]).all():
                            is_arithmetic_index = True

                    if not is_arithmetic_index:
                        target_type = "regression"
                        has_reg_keyword = bool(reg_target_regex.search(col))
                        has_timestamp = any(cls.is_likely_datetime_series(df[c], c) for c in cols if c != col)

                        # If no semantic keyword, require at least one other numeric feature or temporal sequence
                        if has_reg_keyword or numeric_cols_count >= 2 or has_timestamp:
                            # a) Continuous cardinality evidence
                            if n_unique >= 20 or (n_unique >= 12 and (n_unique / max(1, len(s))) >= 0.05):
                                score += 25.0
                                evidence.append(f"Continuous numeric cardinality ({n_unique} unique values)")
                            elif n_unique >= 10:
                                score += 15.0
                                evidence.append(f"Moderate numeric cardinality ({n_unique} unique values)")

                            if has_timestamp:
                                score += 15.0
                                evidence.append("Continuous numeric sequence aligned with temporal index")

                            # b) Float continuous measurement vs broad integer
                            if pd.api.types.is_float_dtype(s):
                                score += 15.0
                                evidence.append(f"Floating-point measurement with variance (std={std_val:.2f})")
                            elif pd.api.types.is_integer_dtype(s) and n_unique >= 20:
                                score += 10.0
                                evidence.append(f"Broad continuous integer distribution (std={std_val:.2f})")

                            # c) Dynamic range
                            if val_range > 0 and std_val > 1e-4:
                                score += 10.0
                                evidence.append(f"Meaningful dynamic range [{s.min():.2f}, {s.max():.2f}]")

                            # d) Semantic keyword bonus
                            if has_reg_keyword:
                                score += 40.0
                                evidence.append(f"Semantic regression target keyword in '{col}'")

                            # e) Schema position evidence (supporting, not dominant)
                            if idx == len(cols) - 1:
                                score += 20.0
                                evidence.append("Positioned as terminal column in schema")
                            elif idx == len(cols) - 2 and len(cols) >= 4:
                                score += 5.0
                                evidence.append("Positioned near terminal boundary")

                            # f) Predictor context
                            if len(cols) >= 3 and numeric_cols_count >= 3:
                                score += 10.0
                                evidence.append(f"Supported by {len(cols)-1} candidate predictor features")

            if score >= 45.0:
                confidence = round(min(0.99, score / 100.0), 3)
                candidate_details.append({
                    "column": col,
                    "target_type": target_type,
                    "score": round(score, 1),
                    "confidence": confidence,
                    "evidence": evidence
                })

        # Rank candidates by score descending
        candidate_details.sort(key=lambda x: x["score"], reverse=True)
        return candidate_details

    @classmethod
    def find_candidate_targets(cls, df: pd.DataFrame, explicit_target: Optional[str] = None) -> List[str]:
        details = cls.find_candidate_target_details(df, explicit_target)
        return [c["column"] for c in details]

    @classmethod
    def profile(
        cls,
        df: pd.DataFrame,
        raw_bytes: Optional[bytes] = None,
        explicit_target: Optional[str] = None,
        explicit_time: Optional[str] = None
    ) -> Tuple[ProfilingMetadata, Dict[str, Any]]:
        num_rows, num_columns = df.shape
        if num_rows == 0 or num_columns == 0:
            raise ValueError("Dataset is empty.")

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        datetime_cols = [c for c in df.columns if cls.is_likely_datetime_series(df[c], c)]
        categorical_cols = [
            c for c in df.columns if c not in numeric_cols and c not in datetime_cols
        ]

        # Candidate timestamps and targets using robust evidence discovery
        candidate_timestamps = cls.find_candidate_timestamps(df, explicit_time)
        candidate_target_details = cls.find_candidate_target_details(df, explicit_target)
        candidate_targets = [c["column"] for c in candidate_target_details]

        total_cells = num_rows * num_columns
        total_missing = int(df.isna().sum().sum())
        missing_percentage = (total_missing / total_cells) * 100.0 if total_cells > 0 else 0.0
        duplicate_rows = int(df.duplicated().sum())
        mem_mb = float(df.memory_usage(deep=True).sum() / (1024 * 1024))
        dataset_sha256 = cls.compute_sha256(df, raw_bytes=raw_bytes)

        metadata = ProfilingMetadata(
            num_rows=num_rows,
            num_columns=num_columns,
            column_types={c: str(df[c].dtype) for c in df.columns},
            numeric_columns=numeric_cols,
            categorical_columns=categorical_cols,
            datetime_columns=datetime_cols,
            missing_cell_percentage=round(missing_percentage, 2),
            duplicate_row_count=duplicate_rows,
            memory_usage_mb=round(mem_mb, 2),
            dataset_sha256=dataset_sha256,
            provenance=PROVENANCE_PROFILING,
            candidate_targets=candidate_targets,
            candidate_timestamps=candidate_timestamps,
            candidate_target_details=candidate_target_details
        )

        column_details: Dict[str, Any] = {}
        for col in df.columns:
            series = df[col]
            missing_cnt = int(series.isna().sum())
            unique_cnt = int(series.nunique(dropna=True))
            col_info = {
                "dtype": str(series.dtype),
                "missing_count": missing_cnt,
                "missing_rate": round(missing_cnt / num_rows, 4),
                "unique_count": unique_cnt,
                "cardinality_ratio": round(unique_cnt / num_rows, 4) if num_rows > 0 else 0.0,
            }

            if col in numeric_cols:
                clean_num = series.dropna()
                if len(clean_num) > 1:
                    col_info.update({
                        "mean": float(clean_num.mean()),
                        "std": float(clean_num.std()) if not np.isnan(clean_num.std()) else 0.0,
                        "min": float(clean_num.min()),
                        "max": float(clean_num.max()),
                        "median": float(clean_num.median()),
                        "skewness": float(clean_num.skew()) if not np.isnan(clean_num.skew()) else 0.0,
                        "kurtosis": float(clean_num.kurt()) if not np.isnan(clean_num.kurt()) else 0.0,
                    })

            column_details[col] = col_info

        # Correlation analysis for numeric features
        correlation_matrix = {}
        if len(numeric_cols) > 1:
            try:
                corr = df[numeric_cols].corr(method="pearson").fillna(0.0)
                correlation_matrix = corr.round(3).to_dict()
            except Exception:
                pass

        details = {
            "columns": column_details,
            "correlations": correlation_matrix
        }

        return metadata, details


DatasetProfiler = DataProfiler
