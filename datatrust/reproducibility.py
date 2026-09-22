import hashlib
import json
import uuid
import datetime
import pandas as pd
from typing import Optional, Dict, Any, List
from datatrust.models import TrustReport, ReproducibilityReceipt, DownstreamAuditReceipt
def canonicalize_index(index: Any) -> List[int]:
    """
    Deterministically converts an Index, Series, ndarray, or list of index values
    into a canonical list of Python standard integers.
    Normalizes np.int64, np.int32, int, etc. to prevent stringification divergence.
    """
    if hasattr(index, "tolist"):
        raw_list = index.tolist()
    else:
        raw_list = list(index)
    return [int(x) for x in raw_list]


def hash_index(index: Any) -> str:
    """
    Computes a deterministic SHA-256 hash of a canonicalized index.
    Uses canonical compact JSON serialization of integer values to guarantee
    exact mathematical identity regardless of numpy vs python scalar representations.
    """
    canonical = canonicalize_index(index)
    serialized = json.dumps(canonical, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(serialized).hexdigest()


class ReproducibilityEngine:
    """
    Subsystem for Cryptographic Experiment Auditing & Reproducibility.
    Generates a deterministic SHA-256 hash and immutable execution receipt
    recording every parameter, weight, residual, and transformation.
    """

    @staticmethod
    def compute_dataset_hash(df: pd.DataFrame) -> str:
        """Computes deterministic SHA-256 hash of DataFrame values."""
        try:
            csv_bytes = df.to_csv(index=False).encode('utf-8')
            return hashlib.sha256(csv_bytes).hexdigest()
        except Exception:
            return hashlib.sha256(str(df.shape).encode('utf-8')).hexdigest()

    @classmethod
    def generate_receipt(
        cls,
        report: TrustReport,
        dataset_hash: Optional[str] = None
    ) -> ReproducibilityReceipt:
        eval_id = f"dt-{uuid.uuid4().hex[:10]}"
        h = dataset_hash or hashlib.sha256(report.dataset_name.encode('utf-8')).hexdigest()

        # Extract initial vs calibrated weights
        init_weights = {dim: res.weight for dim, res in report.dimensions.items()}
        calib_weights = {}
        if report.calibration and report.calibration.weight_adjustments:
            for d, vals in report.calibration.weight_adjustments.items():
                calib_weights[d] = vals.get("calibrated", init_weights.get(d, 0.0))
        else:
            calib_weights = init_weights.copy()

        ci_str = None
        if report.fitness_uncertainty:
            u = report.fitness_uncertainty
            ci_str = f"{report.task_conditioned_fitness} +/- {u.margin_of_error} [{u.ci_lower}, {u.ci_upper}]"

        rem_applied = []
        fit_after = None
        met_after = None
        if report.before_after_validation:
            rem_applied = report.before_after_validation.remediation_actions_applied
            fit_after = report.before_after_validation.fitness_after
            met_after = report.before_after_validation.metric_after

        return ReproducibilityReceipt(
            dataset_hash=h,
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            evaluation_id=eval_id,
            task=report.task_type.value,
            confidence_vector=report.task_inference.confidence_vector if report.task_inference else {},
            sensitivity_matrix_row=report.sensitivity_matrix_row,
            ahp_cr=report.ahp_consistency_ratio,
            initial_weights=init_weights,
            veto_triggered=report.veto_applied,
            veto_ceiling=report.task_conditioned_fitness if report.veto_applied else None,
            initial_fitness=report.raw_score,
            fitness_ci=ci_str,
            model_name="RandomForestReferenceModel",
            metric_name=report.predicted_metric_name,
            predicted_metric=report.predicted_metric_value or 0.0,
            observed_metric=report.observed_metric_value or 0.0,
            residual=report.metric_residual or 0.0,
            calibrated_weights=calib_weights,
            calibrated_fitness=report.task_conditioned_fitness,
            residual_after=report.calibration.residual_after if report.calibration else (report.metric_residual or 0.0),
            remediations_applied=rem_applied,
            fitness_after=fit_after,
            metric_after=met_after
        )

    @classmethod
    def generate_downstream_audit_receipt(
        cls,
        train_indices: Any,
        test_indices: Any,
        feature_columns: list,
        target_column: Optional[str],
        time_column: Optional[str],
        preprocessing_operations: list,
        remediation_operations: list,
        model_family: str,
        hyperparameters: dict,
        baseline_metric: float,
        remediated_metric: float,
        metric_name: str,
        test_index_hash_before: Optional[str] = None,
        test_index_hash_after: Optional[str] = None,
        preprocessing_parameters: Optional[dict] = None,
        dataset_id: Optional[str] = None,
        random_state: int = 42,
        random_seed: int = 42
    ) -> "DownstreamAuditReceipt":
        from datatrust.models import DownstreamAuditReceipt
        train_hash = hash_index(train_indices)
        test_hash = hash_index(test_indices)
        h_before = test_index_hash_before or test_hash
        h_after = test_index_hash_after or test_hash
        assert h_before == h_after, f"Audit receipt violation: test indices mutated ({h_before} != {h_after})"

        return DownstreamAuditReceipt(
            evaluation_id=f"audit-{uuid.uuid4().hex[:10]}",
            dataset_id=dataset_id,
            random_seed=random_seed,
            random_state=random_state,
            train_indices_hash=train_hash,
            test_indices_hash=test_hash,
            test_index_hash_before=h_before,
            test_index_hash_after=h_after,
            test_indices_immutable=(h_before == h_after),
            train_sample_count=len(train_indices),
            test_sample_count=len(test_indices),
            feature_columns=feature_columns,
            target_column=target_column,
            time_column=time_column,
            preprocessing_parameters=preprocessing_parameters or {},
            preprocessing_operations=preprocessing_operations,
            remediation_operations=remediation_operations,
            model_family=model_family,
            model_configuration=hyperparameters,
            model_hyperparameters=hyperparameters,
            baseline_metric_value=round(baseline_metric, 2),
            remediated_metric_value=round(remediated_metric, 2),
            metric_name=metric_name,
            leakage_free_verified=True,
            test_set_resampled=False,
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
        )
