import pandas as pd
import numpy as np
from typing import Optional, List, Tuple, Any
from datatrust.models import TaskType, TaskInferenceResult


from datatrust.profiler import DataProfiler


class TaskInferenceEngine:
    """
    Subsystem for autonomous task characterization and inference.
    Analyzes schema, datatypes, target distributions, and datetime continuity
    via evidence-based schema scoring to produce a normalized empirical
    task-probability distribution P(T|D) across four task profiles.
    P(T|D) denotes an empirical task-probability distribution generated from
    evidence-based schema scoring. It is not a Bayesian posterior unless a formal
    Bayesian inference model with explicit priors and likelihoods is implemented.
    """

    TARGET_NAME_CANDIDATES = DataProfiler.TARGET_NAME_CANDIDATES
    TIME_NAME_CANDIDATES = DataProfiler.TIME_NAME_CANDIDATES

    @classmethod
    def infer_task(
        cls,
        df: pd.DataFrame,
        explicit_target: Optional[str] = None,
        explicit_time: Optional[str] = None,
        user_override_task: Optional[Any] = None
    ) -> TaskInferenceResult:
        num_rows, num_cols = df.shape
        if num_rows == 0 or num_cols == 0:
            raise ValueError("Dataset is empty.")

        # 1. Identify Candidate Timestamp Columns via Robust Profiler Evidence
        candidate_timestamps = DataProfiler.find_candidate_timestamps(df, explicit_time)

        # 2. Identify Candidate Target Columns via Robust Distribution & Semantic Evidence
        candidate_target_details = DataProfiler.find_candidate_target_details(df, explicit_target)
        candidate_targets = [c["column"] for c in candidate_target_details]

        # 3. Comprehensive Evidence & Multi-Task Score Computation
        selected_time = None
        if candidate_timestamps:
            if len(candidate_timestamps) == 1:
                selected_time = candidate_timestamps[0]
            else:
                import warnings
                best_t = candidate_timestamps[0]
                max_u = -1.0
                for ct in candidate_timestamps:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", (UserWarning, FutureWarning))
                        pts = pd.to_datetime(df[ct], errors='coerce').dropna()
                    if len(pts) > 0:
                        u = pts.nunique() / len(pts)
                        if u > max_u:
                            max_u = u
                            best_t = ct
                selected_time = best_t
        selected_target = candidate_targets[0] if candidate_targets else None
        selected_target_info = candidate_target_details[0] if candidate_target_details else None
        target_evidence = selected_target_info.get("evidence", []) if selected_target_info else []

        scores = {
            TaskType.SUPERVISED_REGRESSION: 0.02,
            TaskType.SUPERVISED_CLASSIFICATION: 0.02,
            TaskType.TIME_SERIES_FORECASTING: 0.02,
            TaskType.CLUSTERING: 0.04
        }
        positive_evidence: List[str] = []
        negative_evidence: List[str] = []
        ambiguity_warning: Optional[str] = None

        # Analyze Timestamp
        has_valid_time_series = False
        is_genuine_time_series = False
        is_transactional_with_date = False
        uniqueness_ratio = 1.0
        dup_groups = 0
        dup_rows = 0

        if selected_time:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", (UserWarning, FutureWarning))
                time_series = pd.to_datetime(df[selected_time], errors='coerce').dropna()

            if len(time_series) > 0.3 * len(df):
                has_valid_time_series = True
                n_valid = len(time_series)
                n_unique_ts = int(time_series.nunique())
                uniqueness_ratio = n_unique_ts / max(1, n_valid)
                is_sorted = bool(time_series.is_monotonic_increasing)
                ts_counts = time_series.value_counts()
                dup_groups = int((ts_counts > 1).sum())
                dup_rows = int(time_series.duplicated().sum())
                dup_ratio = dup_rows / max(1, n_valid)

                positive_evidence.append(f"Datetime timestamp column detected: '{selected_time}'")

                # Differentiate genuine time series vs transactional table with dates
                if uniqueness_ratio >= 0.85 and dup_ratio <= 0.10:
                    is_genuine_time_series = True
                    positive_evidence.append(
                        f"High timestamp uniqueness ({uniqueness_ratio*100:.1f}%) and chronological progression "
                        f"indicate single-stream temporal sequence in '{selected_time}'."
                    )
                elif uniqueness_ratio < 0.70 or dup_ratio > 0.15:
                    is_transactional_with_date = True
                    positive_evidence.append(
                        f"Transactional structure in '{selected_time}': {dup_groups} collided timestamp groups, "
                        f"{dup_rows} duplicate rows (uniqueness ratio {uniqueness_ratio*100:.1f}%). "
                        f"Timestamp serves as a temporal event feature in tabular schema, not a single-stream time series index."
                    )
                    negative_evidence.append(
                        f"Low timestamp uniqueness ({uniqueness_ratio*100:.1f}%) precludes single-stream autoregressive time-series forecasting."
                    )
                else:
                    # Intermediate boundary: mild duplicates
                    ambiguity_warning = (
                        f"Timestamp '{selected_time}' has intermediate uniqueness ({uniqueness_ratio*100:.1f}%, {dup_rows} duplicates); "
                        f"may represent panel time series or tabular records with dates."
                    )
            else:
                negative_evidence.append(f"Column '{selected_time}' had excessive unparseable dates (<30% valid)")
        else:
            negative_evidence.append("No datetime or timestamp column detected")

        # Analyze Target and Synthesize Multi-Task Evidence
        if selected_target:
            t_series = df[selected_target].dropna()
            n_unique = t_series.nunique()
            is_numeric = pd.api.types.is_numeric_dtype(t_series)
            t_type = selected_target_info.get("target_type") if selected_target_info else None

            # Add discovered target evidence
            if target_evidence:
                positive_evidence.append(f"Target '{selected_target}' discovered via structural profiling: {'; '.join(target_evidence[:2])}")
            if len(candidate_targets) > 1:
                alt_candidates = candidate_targets[1:4]
                positive_evidence.append(f"Alternative candidate targets: {', '.join(alt_candidates)}")
                if len(candidate_target_details) >= 2:
                    score_diff = candidate_target_details[0]["score"] - candidate_target_details[1]["score"]
                    if score_diff < 15.0 and not ambiguity_warning:
                        ambiguity_warning = (
                            f"Multiple candidate targets detected with comparable confidence ({', '.join(candidate_targets[:2])}). "
                            f"'{selected_target}' selected as primary target; manual confirmation supported."
                        )

            if (is_numeric and n_unique > 15) or t_type == "regression":
                # Continuous numeric target
                if is_transactional_with_date or not has_valid_time_series:
                    scores[TaskType.SUPERVISED_REGRESSION] += 1.25
                    if is_transactional_with_date:
                        scores[TaskType.TIME_SERIES_FORECASTING] += 0.12
                    positive_evidence.append(f"Continuous numeric target '{selected_target}' with {n_unique} unique values")
                    positive_evidence.append(f"Target variance ({t_series.std():.2f}) indicates continuous regression objective")
                    negative_evidence.append("Continuous target cardinality > 15 rules out discrete binary/multiclass classification")
                elif is_genuine_time_series:
                    scores[TaskType.TIME_SERIES_FORECASTING] += 1.55
                    scores[TaskType.SUPERVISED_REGRESSION] += 0.12
                    positive_evidence.append(f"Continuous target '{selected_target}' aligned with chronological sequence '{selected_time}'")
                    positive_evidence.append("Sequential temporal structure supports autoregressive/covariate forecasting")
                else:
                    # Intermediate timestamp case
                    scores[TaskType.SUPERVISED_REGRESSION] += 0.85
                    scores[TaskType.TIME_SERIES_FORECASTING] += 0.40
            elif not is_numeric or (is_numeric and n_unique <= 15) or t_type == "classification":
                # Discrete categorical target
                if is_transactional_with_date or not has_valid_time_series:
                    scores[TaskType.SUPERVISED_CLASSIFICATION] += 1.25
                    if is_transactional_with_date:
                        scores[TaskType.TIME_SERIES_FORECASTING] += 0.08
                    positive_evidence.append(f"Low-cardinality discrete target '{selected_target}' ({n_unique} classes)")
                    if not is_numeric:
                        positive_evidence.append("Target values are nominal categorical labels")
                    negative_evidence.append("Target is discrete, precluding standard continuous regression")
                elif is_genuine_time_series:
                    scores[TaskType.TIME_SERIES_FORECASTING] += 0.70
                    scores[TaskType.SUPERVISED_CLASSIFICATION] += 0.55
                    ambiguity_warning = "Dataset exhibits both chronological sequence and discrete target (regime forecasting vs classification)."
                else:
                    scores[TaskType.SUPERVISED_CLASSIFICATION] += 0.95
                    scores[TaskType.TIME_SERIES_FORECASTING] += 0.25
        else:
            # No Target Specified or Discovered
            if is_genuine_time_series:
                scores[TaskType.TIME_SERIES_FORECASTING] += 1.55
                scores[TaskType.CLUSTERING] += 0.08
                positive_evidence.append(f"Unlabelled chronological sequence indexed by '{selected_time}' (uniqueness {uniqueness_ratio*100:.1f}%)")
                positive_evidence.append("Autoregressive forecasting supported on numeric temporal sequence")
            elif is_transactional_with_date:
                scores[TaskType.CLUSTERING] = 0.55
                scores[TaskType.TIME_SERIES_FORECASTING] = 0.20
                scores[TaskType.SUPERVISED_REGRESSION] = 0.15
                scores[TaskType.SUPERVISED_CLASSIFICATION] = 0.10
                ambiguity_warning = "Unlabelled transactional dataset with non-unique dates. Unsupervised clustering hypothesis adopted."
            else:
                # Standard unlabelled tabular data
                scores[TaskType.CLUSTERING] = 0.55
                scores[TaskType.SUPERVISED_CLASSIFICATION] = 0.15
                scores[TaskType.SUPERVISED_REGRESSION] = 0.15
                scores[TaskType.TIME_SERIES_FORECASTING] = 0.15
                positive_evidence.append("Unlabelled tabular structure detected without explicit target or timestamp index")
                positive_evidence.append("Structure is compatible with unsupervised analysis / clustering hypothesis")
                negative_evidence.append("No explicit prediction target column designated or discovered")
                ambiguity_warning = (
                    "Clustering is an unsupervised exploratory hypothesis and cannot be definitively inferred "
                    "from schema structure alone; manual confirmation recommended."
                )

        # Normalize scores to form empirical task probability distribution P(T|D)
        total_score = sum(scores.values())
        confidence_vector = {
            task.value: round(score / total_score, 3)
            for task, score in scores.items()
        }
        # Ensure exact sum to 1.0
        total_rounded = sum(confidence_vector.values())
        diff = round(1.0 - total_rounded, 4)
        if abs(diff) > 1e-5:
            max_key = max(confidence_vector, key=confidence_vector.get)
            confidence_vector[max_key] = round(confidence_vector[max_key] + diff, 3)

        # Select task with maximum empirical task probability
        inferred_task = max(scores, key=scores.get)
        confidence = confidence_vector[inferred_task.value]

        if inferred_task == TaskType.SUPERVISED_REGRESSION:
            reasoning = (
                f"Inferred Supervised Regression ({confidence*100:.1f}% confidence). "
                f"Continuous numeric target '{selected_target}' detected."
            )
        elif inferred_task == TaskType.SUPERVISED_CLASSIFICATION:
            reasoning = (
                f"Inferred Supervised Classification ({confidence*100:.1f}% confidence). "
                f"Discrete categorical target '{selected_target}' detected."
            )
        elif inferred_task == TaskType.TIME_SERIES_FORECASTING:
            reasoning = (
                f"Inferred Time-Series Forecasting ({confidence*100:.1f}% confidence). "
                f"Chronological timestamp column '{selected_time}' detected."
            )
        else:
            reasoning = (
                f"Inferred Clustering hypothesis ({confidence*100:.1f}% heuristic score). "
                f"Unlabelled tabular structure detected without target label or timestamp. "
                f"{ambiguity_warning}"
            )

        # Handle user confirmation / override
        user_overrode = False
        final_selected_task = inferred_task
        if user_override_task is not None:
            resolved_override = user_override_task
            if isinstance(resolved_override, str):
                for t in TaskType:
                    if t.value == resolved_override.lower() or t.name.lower() == resolved_override.lower():
                        resolved_override = t
                        break
            if resolved_override != inferred_task:
                user_overrode = True
            final_selected_task = resolved_override

        return TaskInferenceResult(
            inferred_task=inferred_task,
            confidence=confidence,
            confidence_type="heuristic_structural_score",
            confidence_vector=confidence_vector,
            p_task_given_data=confidence_vector,
            positive_evidence=positive_evidence,
            negative_evidence=negative_evidence,
            reasoning=reasoning,
            candidate_targets=candidate_targets,
            candidate_timestamps=candidate_timestamps,
            detected_target=selected_target,
            target_evidence=target_evidence,
            candidate_target_details=candidate_target_details,
            ambiguity_warning=ambiguity_warning,
            user_overrode=user_overrode,
            final_selected_task=final_selected_task,
            provenance="HEURISTIC"
        )

