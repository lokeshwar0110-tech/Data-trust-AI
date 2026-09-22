import numpy as np
from scipy.stats import spearmanr
from typing import Dict, Tuple, Optional, Any, List
from datatrust.models import TaskType, QualityDimension
from datatrust.config import (
    AHP_PAIRWISE_MATRICES,
    AHP_RANDOM_INDEX,
    PROVENANCE_AHP_WEIGHTS,
    PROVENANCE_STATUS_PRE_EXPERIMENTAL,
    PROVENANCE_EMPIRICAL
)

# Saaty's Random Consistency Index (RI) table for matrix sizes 1 to 10
RI_TABLE = {
    1: 0.00,
    2: 0.00,
    3: 0.58,
    4: 0.90,
    5: 1.12,
    6: 1.24,
    7: 1.32,
    8: 1.41,
    9: 1.45,
    10: 1.49
}

ALL_DIMENSIONS = [
    QualityDimension.COMPLETENESS,
    QualityDimension.VALIDITY,
    QualityDimension.CONSISTENCY,
    QualityDimension.UNIQUENESS,
    QualityDimension.TIMELINESS,
    QualityDimension.OUTLIER_ANOMALY,
    QualityDimension.DISTRIBUTION_BALANCE,
    QualityDimension.COVERAGE_REPRESENTATIVENESS
]

# Pairwise preference matrices for each downstream task (8x8 reciprocal)
TASK_PAIRWISE_PREFERENCES: Dict[TaskType, np.ndarray] = AHP_PAIRWISE_MATRICES

# Empirical calibrated matrices storage
CALIBRATED_PAIRWISE_PREFERENCES: Dict[TaskType, np.ndarray] = {}
CALIBRATED_METADATA_REGISTRY: Dict[TaskType, Dict[str, Any]] = {}


class AHPWeightingEngine:
    """
    Implements Analytic Hierarchy Process (AHP) to compute mathematically consistent,
    dynamic task-specific quality dimension weights across eight canonical dimensions.
    Verified with Consistency Ratio CR < 0.10 and Saaty Random Index RI = 1.41.
    Supports both:
    - LITERATURE / EXPERT PRIOR (PRE-EXPERIMENTAL UNCALIBRATED)
    - EMPIRICALLY CALIBRATED (DOWNSTREAM REGRESSION)
    """

    _calibrated_matrices: Dict[TaskType, np.ndarray] = CALIBRATED_PAIRWISE_PREFERENCES
    _calibrated_metadata: Dict[TaskType, Dict[str, Any]] = CALIBRATED_METADATA_REGISTRY

    @staticmethod
    def solve_weights_and_consistency(matrix: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        Calculates normalized weight vector, principal eigenvalue lambda_max,
        and Consistency Ratio (CR) via the principal eigenvector method.
        """
        n = matrix.shape[0]
        eigvals, eigvecs = np.linalg.eig(matrix)
        max_idx = int(np.argmax(np.real(eigvals)))
        lambda_max = float(np.real(eigvals[max_idx]))
        principal_vec = np.real(eigvecs[:, max_idx])

        # Normalize weights to sum to 1.0
        weights = principal_vec / np.sum(principal_vec)
        weights = np.abs(weights)
        weights = weights / np.sum(weights)

        # Consistency index & ratio (Saaty RI = 1.41 for n=8)
        ci = (lambda_max - n) / (n - 1) if n > 1 else 0.0
        ri = RI_TABLE.get(n, AHP_RANDOM_INDEX)
        cr = float(ci / ri) if ri > 0 else 0.0

        return weights, float(lambda_max), cr

    @classmethod
    def get_task_weights(
        cls,
        task: TaskType,
        use_calibrated: bool = False
    ) -> Tuple[Dict[QualityDimension, float], float]:
        """
        Returns a dictionary mapping QualityDimension to normalized weights and the CR.
        If use_calibrated=True and task was calibrated, returns the empirically calibrated weights.
        Otherwise, returns the fixed expert prior weights.
        """
        resolved = task if task != TaskType.DESCRIPTIVE_BI else TaskType.CLUSTERING
        if use_calibrated and resolved in cls._calibrated_matrices:
            matrix = cls._calibrated_matrices[resolved]
        else:
            matrix = TASK_PAIRWISE_PREFERENCES[resolved]
        weights, _, cr = cls.solve_weights_and_consistency(matrix)
        weight_dict = {dim: float(weights[i]) for i, dim in enumerate(ALL_DIMENSIONS)}
        return weight_dict, cr

    @classmethod
    def get_ahp_metadata(cls, task: TaskType, use_calibrated: bool = False) -> Dict[str, Any]:
        """
        Returns comprehensive AHP metadata including matrix, lambda_max, CI, CR, and RI.
        Accurately distinguishes PROVENANCE_AHP_WEIGHTS from PROVENANCE_EMPIRICAL.
        """
        resolved = task if task != TaskType.DESCRIPTIVE_BI else TaskType.CLUSTERING
        is_cal = use_calibrated and (resolved in cls._calibrated_matrices)
        matrix = cls._calibrated_matrices[resolved] if is_cal else TASK_PAIRWISE_PREFERENCES[resolved]
        weights, lambda_max, cr = cls.solve_weights_and_consistency(matrix)
        n = matrix.shape[0]
        ci = (lambda_max - n) / (n - 1) if n > 1 else 0.0

        provenance = PROVENANCE_EMPIRICAL if is_cal else PROVENANCE_AHP_WEIGHTS
        status = "EMPIRICALLY CALIBRATED (DOWNSTREAM REGRESSION)" if is_cal else PROVENANCE_STATUS_PRE_EXPERIMENTAL

        return {
            "task": task.value,
            "matrix_dimension": f"{n}x{n}",
            "lambda_max": round(lambda_max, 4),
            "consistency_index": round(ci, 4),
            "random_index": AHP_RANDOM_INDEX,
            "consistency_ratio": round(cr, 4),
            "is_consistent": cr < 0.10,
            "provenance": provenance,
            "status": status,
            "weights": {dim.value: round(float(weights[i]), 4) for i, dim in enumerate(ALL_DIMENSIONS)}
        }

    @classmethod
    def calibrate_task_ahp_matrix(
        cls,
        task: TaskType,
        dimension_scores_list: Any,
        downstream_performances: Any,
        alpha: float = 0.5,
        min_weight: float = 0.02,
        max_weight: float = 0.60,
        higher_is_better: Optional[bool] = None
    ) -> Tuple[np.ndarray, Dict[QualityDimension, float], float, Dict[str, Any]]:
        """
        Empirically calibrates the AHP pairwise comparison matrix from reference dataset observations.
        
        Mathematical Procedure:
        1. Regress dimension scores against downstream task performance (Spearman rank correlation).
        2. Derive empirical weight vector w_emp in [min_weight, max_weight].
        3. Bayesian / convex blend with expert prior: w_cal = (1 - alpha) * w_prior + alpha * w_emp.
        4. Generate strictly reciprocal comparison matrix: A_ij = w_cal_i / w_cal_j.
        5. Mathematical theorem: A_ji = 1 / A_ij (strictly reciprocal) and A_ik * A_kj = A_ij (consistent),
           guaranteeing principal eigenvalue lambda_max == 8, CI == 0.0, and CR < 0.10.
        6. Registers calibrated matrix and updates provenance to PROVENANCE_EMPIRICAL.
        """
        resolved = task if task != TaskType.DESCRIPTIVE_BI else TaskType.CLUSTERING
        prior_matrix = TASK_PAIRWISE_PREFERENCES[resolved]
        prior_weights, _, _ = cls.solve_weights_and_consistency(prior_matrix)

        if higher_is_better is None:
            higher_is_better = (task == TaskType.SUPERVISED_CLASSIFICATION)

        y = np.array(downstream_performances, dtype=float)
        if not higher_is_better:
            y = -y

        # Step 1: Compute empirical sensitivity for each dimension via rank correlation
        empirical_sensitivities = []
        for i, dim in enumerate(ALL_DIMENSIONS):
            x = np.array([scores.get(dim, 50.0) for scores in dimension_scores_list], dtype=float)
            if np.std(x) < 1e-6 or np.std(y) < 1e-6:
                r = 0.0
            else:
                r, _ = spearmanr(x, y)
                if np.isnan(r):
                    r = 0.0
            empirical_sensitivities.append(max(0.01, float(r)))

        total_sens = sum(empirical_sensitivities)
        if total_sens > 0:
            w_emp = np.array(empirical_sensitivities) / total_sens
        else:
            w_emp = prior_weights.copy()

        # Step 2: Convex blend with expert prior
        w_cal = (1.0 - alpha) * prior_weights + alpha * w_emp

        # Clip to [min_weight, max_weight] and re-normalize
        w_cal = np.clip(w_cal, min_weight, max_weight)
        w_cal = w_cal / np.sum(w_cal)

        # Step 3: Construct strictly reciprocal, mathematically consistent 8x8 pairwise matrix
        n = len(ALL_DIMENSIONS)
        cal_matrix = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(n):
                cal_matrix[i, j] = float(w_cal[i] / w_cal[j])

        # Verify reciprocity and consistency
        solved_weights, lambda_max, cr = cls.solve_weights_and_consistency(cal_matrix)

        # Register calibrated matrix and metadata
        cls._calibrated_matrices[resolved] = cal_matrix
        weight_dict = {dim: float(solved_weights[i]) for i, dim in enumerate(ALL_DIMENSIONS)}

        metadata = {
            "task": task.value,
            "matrix_dimension": f"{n}x{n}",
            "lambda_max": round(lambda_max, 4),
            "consistency_index": round((lambda_max - n) / (n - 1), 4),
            "random_index": AHP_RANDOM_INDEX,
            "consistency_ratio": round(cr, 4),
            "is_consistent": cr < 0.10,
            "provenance": PROVENANCE_EMPIRICAL,
            "status": "EMPIRICALLY CALIBRATED (DOWNSTREAM REGRESSION)",
            "alpha_blend": alpha,
            "weights": {dim.value: round(float(solved_weights[i]), 4) for i, dim in enumerate(ALL_DIMENSIONS)}
        }
        cls._calibrated_metadata[resolved] = metadata

        return cal_matrix, weight_dict, cr, metadata

    @classmethod
    def reset_calibration(cls, task: Optional[TaskType] = None):
        """Resets calibrated matrices back to uncalibrated prior state."""
        if task is not None:
            resolved = task if task != TaskType.DESCRIPTIVE_BI else TaskType.CLUSTERING
            cls._calibrated_matrices.pop(resolved, None)
            cls._calibrated_metadata.pop(resolved, None)
        else:
            cls._calibrated_matrices.clear()
            cls._calibrated_metadata.clear()

    @staticmethod
    def evaluate_veto_penalties(
        task: TaskType,
        dimension_scores: Dict[QualityDimension, float],
        raw_metrics: Dict[str, Any]
    ) -> Tuple[bool, Optional[str], Optional[float]]:
        """
        Non-compensatory veto evaluation: if a fatal defect is present for a specific task,
        an absolute score ceiling is applied regardless of how high other dimensions score.
        """
        completeness = dimension_scores.get(QualityDimension.COMPLETENESS, 100.0)
        temporal = dimension_scores.get(QualityDimension.TIMELINESS, 100.0)
        dist_bal = dimension_scores.get(QualityDimension.DISTRIBUTION_BALANCE, 100.0)

        # Universal veto: Extreme dataset missingness (> 70% missing overall)
        if completeness < 30.0:
            return True, "Dataset has catastrophic missingness (>70% null density). Downstream analysis will fail.", 20.0

        # Time Series Veto: Severe chronological disorder, timestamp collision, or broken cadence
        if task == TaskType.TIME_SERIES_FORECASTING:
            if temporal < 40.0:
                reason = raw_metrics.get("temporal_veto_reason", "Critical chronological disorder or timestamp collision detected.")
                return True, f"Time-Series Veto Triggered: {reason}", 25.0

        # Classification Veto: Single class target or severe target corruption
        if task == TaskType.SUPERVISED_CLASSIFICATION:
            if dist_bal < 25.0:
                reason = raw_metrics.get("label_veto_reason", "Target column has near-zero variance or extreme corruption (>95% single class).")
                return True, f"Classification Veto Triggered: {reason}", 15.0

        return False, None, None
