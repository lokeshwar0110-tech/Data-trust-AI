"""
DataTrust AI Phase 1 Central Research Configuration
===================================================
Centralizes all methodological parameters, canonical dimensions, task profiles,
task-quality sensitivity tensor M(T, Q), fixed 8x8 AHP pairwise judgment matrices,
and three-tier defect policies with explicit provenance metadata.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, List
from datatrust.models import TaskType, QualityDimension

CONFIG_VERSION = "1.0.0-phase1"

# Formal Research Provenance Classifications
PROVENANCE_HEURISTIC = "HEURISTIC"
PROVENANCE_EXPERT_PRIOR = "LITERATURE / EXPERT PRIOR"
PROVENANCE_POLICY_PRIOR = "HEURISTIC / POLICY PRIOR"
PROVENANCE_EMPIRICAL = "EMPIRICALLY CALIBRATED"

# Module-level provenance constants
PROVENANCE_PROFILING = PROVENANCE_HEURISTIC
PROVENANCE_TASK_INFERENCE = PROVENANCE_HEURISTIC
PROVENANCE_DIMENSIONS = PROVENANCE_HEURISTIC
PROVENANCE_SENSITIVITY_MATRIX = PROVENANCE_EXPERT_PRIOR
PROVENANCE_STATUS_PRE_EXPERIMENTAL = "PRE-EXPERIMENTAL UNCALIBRATED"
PROVENANCE_AHP_WEIGHTS = PROVENANCE_EXPERT_PRIOR
PROVENANCE_DEFECT_POLICY = PROVENANCE_POLICY_PRIOR
PROVENANCE_PIPELINE = "PHASE 1 CORE FITNESS PIPELINE (PRE-EXPERIMENTAL)"

DEFECT_TIER_GREEN = "GREEN"
DEFECT_TIER_YELLOW = "YELLOW"
DEFECT_TIER_RED = "RED"

AHP_RANDOM_INDEX = 1.41
REPRESENTATIVENESS_UNAVAILABLE_STATUS = "UNAVAILABLE"

# Random Index for Saaty AHP Consistency (N=1..10)
RI_TABLE = {
    1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12,
    6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49
}

@dataclass
class DimensionMeta:
    name: str
    definition: str
    formula: str
    assumptions: str
    limitations: str
    task_applicability: str
    provenance: str = PROVENANCE_HEURISTIC

# -------------------------------------------------------------------------
# Supported Downstream Analytical / ML Tasks (Exactly Four)
# -------------------------------------------------------------------------
TASKS = {
    "supervised_classification": {
        "display_name": "Classification",
        "description": "Supervised discrete class target prediction.",
        "requires_target": True,
        "is_temporal": False,
        "provenance": PROVENANCE_EXPERT_PRIOR
    },
    "supervised_regression": {
        "display_name": "Regression",
        "description": "Supervised continuous numeric target prediction.",
        "requires_target": True,
        "is_temporal": False,
        "provenance": PROVENANCE_EXPERT_PRIOR
    },
    "clustering": {
        "display_name": "Clustering",
        "description": "Unsupervised geometric partitioning of unlabelled feature spaces.",
        "requires_target": False,
        "is_temporal": False,
        "provenance": PROVENANCE_EXPERT_PRIOR
    },
    "time_series_forecasting": {
        "display_name": "Time-Series Forecasting",
        "description": "Autoregressive and sequence-based temporal trajectory prediction.",
        "requires_target": True,
        "is_temporal": True,
        "provenance": PROVENANCE_EXPERT_PRIOR
    }
}

# -------------------------------------------------------------------------
# Eight Canonical Quality Dimensions (Ordered)
# -------------------------------------------------------------------------
CANONICAL_DIMENSION_NAMES: List[str] = [
    "completeness",
    "validity",
    "consistency",
    "uniqueness",
    "timeliness",
    "outlier_anomaly",
    "distribution_balance",
    "coverage_representativeness"
]

CANONICAL_DIMENSIONS: Dict[QualityDimension, DimensionMeta] = {
    QualityDimension.COMPLETENESS: DimensionMeta(
        name="Completeness",
        definition="Proportion of non-null, fully observed cell data across attributes.",
        formula="S_cmp = max(0, min(100, 100 - (80 * missing_rate + 5 * high_missing_cols + 15 * empty_cols)))",
        assumptions="Evaluates observed null footprint; does not assume missingness mechanism (MCAR/MAR/MNAR).",
        limitations="Cannot observe unrecorded non-responses that are coded as default values without a data dictionary.",
        task_applicability="Universal prerequisite for analytical pipelines; critical for distance and aggregation metrics.",
        provenance=PROVENANCE_HEURISTIC
    ),
    QualityDimension.VALIDITY: DimensionMeta(
        name="Validity",
        definition="Degree to which recorded values adhere to structural data types, syntax, non-zero variance, and admissible ranges.",
        formula="S_val = max(0, min(100, 100 - (10 * constant_cols + 100 * inf_rate + target_penalty)))",
        assumptions="Values outside inferred natural ranges, constant columns, or infinite values constitute data invalidity.",
        limitations="Domain-specific business rules require explicit schema configuration; defaults to heuristic type checking.",
        task_applicability="Essential across all tasks to prevent numerical instability and singular covariance matrices.",
        provenance=PROVENANCE_HEURISTIC
    ),
    QualityDimension.CONSISTENCY: DimensionMeta(
        name="Consistency",
        definition="Absence of internally contradictory, logically impossible, or cross-field invalid records (collinearity, target leakage).",
        formula="S_cns = max(15, min(100, 100 - (6 * collinear_pairs + 25 * target_leakage_suspects)))",
        assumptions="Records with conflicting cross-column relations or target leakage corrupt empirical learning. NOT defined as distribution drift.",
        limitations="Non-linear constraints require domain business rules; linear Pearson correlations capture linear collinearity.",
        task_applicability="High for supervised tasks to prevent spurious weights and post-outcome leakage.",
        provenance=PROVENANCE_HEURISTIC
    ),
    QualityDimension.UNIQUENESS: DimensionMeta(
        name="Uniqueness",
        definition="Absence of redundant duplicate rows or primary entity key collisions.",
        formula="S_unq = max(0, min(100, 100 * (1 - duplicate_row_rate) - 20 * candidate_id_collision_rate))",
        assumptions="Identical rows represent observation replication error unless discrete multi-attribute frequency is expected.",
        limitations="In datasets with small feature dimensionality, identical tuples may represent natural repeated states.",
        task_applicability="Universal; duplicate records artificially deflate bootstrap standard errors and skew cluster centroids.",
        provenance=PROVENANCE_HEURISTIC
    ),
    QualityDimension.TIMELINESS: DimensionMeta(
        name="Timeliness / Freshness",
        definition="Temporal cadence regularity, chronological sequence order, and temporal partition drift stability.",
        formula="S_tim = max(0, min(100, 100 - (45 * (1 - is_monotonic) + 30 * duplicate_ts_rate + 15 * cadence_cv + drift_penalty)))",
        assumptions="Autoregressive and time-series workflows require chronological sequence continuity and cadence stability.",
        limitations="When no temporal column is present, partition drift is evaluated as a proxy for structural stability.",
        task_applicability="CRITICAL for Time-Series Forecasting; moderate for general tabular models subject to covariate shift.",
        provenance=PROVENANCE_HEURISTIC
    ),
    QualityDimension.OUTLIER_ANOMALY: DimensionMeta(
        name="Outlier / Anomaly Quality",
        definition="Degree of freedom from multivariate anomalies. Isolation Forest anomaly density is primary score; kurtosis is descriptive supporting evidence only.",
        formula="S_out = max(10, min(100, 100 - 120 * isolation_forest_anomaly_ratio)); Kurtosis is descriptive supporting evidence only.",
        assumptions="Multivariate anomaly density from Isolation Forest captures joint feature corruption; kurtosis describes univariate tail heaviness.",
        limitations="Unsupervised anomaly detection assumes low contamination prior (5%); kurtosis is reported as descriptive context, not the score.",
        task_applicability="CRITICAL for Supervised Regression (squared-error loss explodes under heavy-tailed anomalies); moderate for classification.",
        provenance=PROVENANCE_HEURISTIC
    ),
    QualityDimension.DISTRIBUTION_BALANCE: DimensionMeta(
        name="Class Balance / Distribution Balance",
        definition="Equitable representation of classification target labels or balanced coverage of continuous target density.",
        formula="Classification: S_bal = max(10, min(100, 100 - penalty(minority_share))); Regression: S_bal = max(15, min(100, 100 - min(50, 8 * |skew|))).",
        assumptions="Extreme class collapse (minority < 2%) prevents learning minority decision boundaries.",
        limitations="In clustering, class balance does not apply and unlabelled multimodal distribution uniformity is assessed.",
        task_applicability="CRITICAL for Supervised Classification; moderate for regression target skewness.",
        provenance=PROVENANCE_HEURISTIC
    ),
    QualityDimension.COVERAGE_REPRESENTATIVENESS: DimensionMeta(
        name="Coverage / Representativeness",
        definition="Observable feature space support and domain coverage; population representativeness is explicitly marked UNAVAILABLE.",
        formula="S_cov = max(20, min(100, 50 + 50 * min(1.0, N / (10 * max(1, P))))); Representativeness status: 'UNAVAILABLE'.",
        assumptions="Observable feature support measures empirical sample-to-feature density (N/P >= 10:1); does not claim population-level representativeness.",
        limitations="Population representativeness cannot be mathematically established from sample data alone without an external census benchmark.",
        task_applicability="Universal; guards against curse of dimensionality and empty subspace extrapolation.",
        provenance=PROVENANCE_HEURISTIC
    ),
}

DIMENSION_METADATA: Dict[str, Dict[str, Any]] = {
    dim.value: {
        "name": meta.name,
        "definition": meta.definition,
        "formula": meta.formula,
        "assumptions": meta.assumptions,
        "limitations": meta.limitations,
        "task_applicability": meta.task_applicability,
        "provenance": meta.provenance
    }
    for dim, meta in CANONICAL_DIMENSIONS.items()
}

# -------------------------------------------------------------------------
# 4 x 8 Task-Quality Sensitivity Matrix M(T, Q)
# Status: LITERATURE / EXPERT PRIOR (Pre-Experimental, Uncalibrated)
# -------------------------------------------------------------------------
SENSITIVITY_MATRIX: Dict[str, Dict[str, Dict[str, Any]]] = {
    "supervised_classification": {
        "distribution_balance": {"value": 0.22, "tier": "HIGH", "rationale": "Severe class collapse destroys minority recall and decision margins.", "provenance": PROVENANCE_EXPERT_PRIOR},
        "validity":             {"value": 0.18, "tier": "HIGH", "rationale": "Corrupted or invalid categorical labels distort boundary estimation.", "provenance": PROVENANCE_EXPERT_PRIOR},
        "completeness":         {"value": 0.14, "tier": "MEDIUM", "rationale": "Missing attributes degrade classifier information density.", "provenance": PROVENANCE_EXPERT_PRIOR},
        "outlier_anomaly":      {"value": 0.12, "tier": "MEDIUM", "rationale": "Multivariate anomalies distort hyperplanes and tree splits.", "provenance": PROVENANCE_EXPERT_PRIOR},
        "consistency":          {"value": 0.12, "tier": "MEDIUM", "rationale": "Contradictory feature rows generate conflicting label gradients.", "provenance": PROVENANCE_EXPERT_PRIOR},
        "coverage_representativeness": {"value": 0.10, "tier": "MEDIUM", "rationale": "Unobserved feature spaces create classification generalization blind spots.", "provenance": PROVENANCE_EXPERT_PRIOR},
        "uniqueness":           {"value": 0.08, "tier": "LOW", "rationale": "Duplicate records inflate minority/majority weights spuriously.", "provenance": PROVENANCE_EXPERT_PRIOR},
        "timeliness":           {"value": 0.04, "tier": "LOW", "rationale": "Temporal sequence order is non-critical for non-sequential classification.", "provenance": PROVENANCE_EXPERT_PRIOR},
    },
    "supervised_regression": {
        "outlier_anomaly":      {"value": 0.24, "tier": "HIGH", "rationale": "Squared-error loss (MSE/RMSE) explodes quadratically under anomalies and heavy tails.", "provenance": PROVENANCE_EXPERT_PRIOR},
        "completeness":         {"value": 0.18, "tier": "HIGH", "rationale": "Matrix inversion and continuous regressors require dense numeric observations.", "provenance": PROVENANCE_EXPERT_PRIOR},
        "validity":             {"value": 0.16, "tier": "HIGH", "rationale": "Numeric range violations directly corrupt continuous prediction surfaces.", "provenance": PROVENANCE_EXPERT_PRIOR},
        "consistency":          {"value": 0.14, "tier": "MEDIUM", "rationale": "Inconsistent cross-column relations introduce collinearity and noise.", "provenance": PROVENANCE_EXPERT_PRIOR},
        "distribution_balance": {"value": 0.10, "tier": "MEDIUM", "rationale": "Continuous target density coverage prevents skewed regression planes.", "provenance": PROVENANCE_EXPERT_PRIOR},
        "coverage_representativeness": {"value": 0.08, "tier": "LOW", "rationale": "Feature range coverage allows stable continuous interpolation.", "provenance": PROVENANCE_EXPERT_PRIOR},
        "uniqueness":           {"value": 0.06, "tier": "LOW", "rationale": "Duplication mildly biases continuous point regression fits.", "provenance": PROVENANCE_EXPERT_PRIOR},
        "timeliness":           {"value": 0.04, "tier": "LOW", "rationale": "Static regression is invariant to chronological record ordering.", "provenance": PROVENANCE_EXPERT_PRIOR},
    },
    "clustering": {
        "outlier_anomaly":      {"value": 0.22, "tier": "HIGH", "rationale": "Distance metrics (Euclidean/Manhattan) and centroids are heavily distorted by outliers.", "provenance": PROVENANCE_EXPERT_PRIOR},
        "coverage_representativeness": {"value": 0.20, "tier": "HIGH", "rationale": "Unsupervised geometric partitioning requires representative density support.", "provenance": PROVENANCE_EXPERT_PRIOR},
        "completeness":         {"value": 0.16, "tier": "HIGH", "rationale": "Distance computations fail or lose fidelity when coordinates are missing.", "provenance": PROVENANCE_EXPERT_PRIOR},
        "validity":             {"value": 0.14, "tier": "MEDIUM", "rationale": "Invalid types or corrupted continuous ranges disrupt metric space geometry.", "provenance": PROVENANCE_EXPERT_PRIOR},
        "uniqueness":           {"value": 0.12, "tier": "MEDIUM", "rationale": "Identical duplicate coordinates artificially inflate cluster density centers.", "provenance": PROVENANCE_EXPERT_PRIOR},
        "consistency":          {"value": 0.08, "tier": "LOW", "rationale": "Structural contradiction in feature relationships.", "provenance": PROVENANCE_EXPERT_PRIOR},
        "distribution_balance": {"value": 0.05, "tier": "LOW", "rationale": "Unsupervised clustering discovers natural multimodal partitions; does not require balance.", "provenance": PROVENANCE_EXPERT_PRIOR},
        "timeliness":           {"value": 0.03, "tier": "LOW", "rationale": "Static clustering is non-sequential.", "provenance": PROVENANCE_EXPERT_PRIOR},
    },
    "time_series_forecasting": {
        "timeliness":           {"value": 0.30, "tier": "HIGH", "rationale": "Chronological monotonicity, cadence regularity, and lack of collisions are critical for lags.", "provenance": PROVENANCE_EXPERT_PRIOR},
        "completeness":         {"value": 0.18, "tier": "HIGH", "rationale": "Temporal gaps and null blocks break autoregressive sequential feature structures.", "provenance": PROVENANCE_EXPERT_PRIOR},
        "outlier_anomaly":      {"value": 0.14, "tier": "MEDIUM", "rationale": "Anomalous spikes vs true seasonal shocks disrupt stationarity and forecast bounds.", "provenance": PROVENANCE_EXPERT_PRIOR},
        "consistency":          {"value": 0.12, "tier": "MEDIUM", "rationale": "Temporal partition stability and structural consistency across time regimes.", "provenance": PROVENANCE_EXPERT_PRIOR},
        "validity":             {"value": 0.10, "tier": "MEDIUM", "rationale": "Valid timestamp formats and admissible target ranges ensure pipeline execution.", "provenance": PROVENANCE_EXPERT_PRIOR},
        "distribution_balance": {"value": 0.06, "tier": "LOW", "rationale": "Regime representation across temporal segments.", "provenance": PROVENANCE_EXPERT_PRIOR},
        "coverage_representativeness": {"value": 0.06, "tier": "LOW", "rationale": "Coverage of the forecasting horizon and historical cyclical seasons.", "provenance": PROVENANCE_EXPERT_PRIOR},
        "uniqueness":           {"value": 0.04, "tier": "LOW", "rationale": "Duplicate timestamps are evaluated under timeliness.", "provenance": PROVENANCE_EXPERT_PRIOR},
    }
}

# -------------------------------------------------------------------------
# Four Independent 8 x 8 AHP Pairwise Comparison Matrices
# Dimension Order:
# 0: completeness, 1: validity, 2: consistency, 3: uniqueness,
# 4: timeliness, 5: outlier_anomaly, 6: distribution_balance, 7: coverage_representativeness
# -------------------------------------------------------------------------
AHP_MATRICES: Dict[str, Dict[str, Any]] = {
    "supervised_classification": {
        "matrix": np.array([
            # CMP   VAL   CNS   UNQ   TIM   OUT   BAL   COV
            [1.0,  0.5,  1.0,  2.0,  3.0,  1.0,  0.5,  1.0],  # completeness (0.134)
            [2.0,  1.0,  2.0,  3.0,  4.0,  2.0,  0.5,  2.0],  # validity (0.161)
            [1.0,  0.5,  1.0,  2.0,  3.0,  1.0,  0.5,  1.0],  # consistency (0.119)
            [0.5,  0.33, 0.5,  1.0,  2.0,  0.5,  0.33, 1.0],  # uniqueness (0.083)
            [0.33, 0.25, 0.33, 0.5,  1.0,  0.33, 0.2,  0.5],  # timeliness (0.038)
            [1.0,  0.5,  1.0,  2.0,  3.0,  1.0,  0.5,  1.0],  # outlier_anomaly (0.129)
            [2.0,  2.0,  2.0,  3.0,  5.0,  2.0,  1.0,  2.0],  # distribution_balance (0.228)
            [1.0,  0.5,  1.0,  1.0,  2.0,  1.0,  0.5,  1.0],  # coverage_representativeness (0.109)
        ]),
        "lambda_max": 8.1132,
        "ci": 0.0162,
        "cr": 0.0115,
        "ri": 1.41,
        "provenance": PROVENANCE_EXPERT_PRIOR,
        "rationale": "Prioritizes distribution balance (class collapse) and validity; timeliness is lowest for tabular classification."
    },
    "supervised_regression": {
        "matrix": np.array([
            # CMP   VAL   CNS   UNQ   TIM   OUT   BAL   COV
            [1.0,  1.0,  1.0,  3.0,  4.0,  0.5,  2.0,  2.0],  # completeness (0.159)
            [1.0,  1.0,  1.0,  3.0,  4.0,  0.5,  2.0,  2.0],  # validity (0.159)
            [1.0,  1.0,  1.0,  2.0,  3.0,  0.5,  1.0,  2.0],  # consistency (0.140)
            [0.33, 0.33, 0.5,  1.0,  2.0,  0.25, 0.5,  0.5],  # uniqueness (0.053)
            [0.25, 0.25, 0.33, 0.5,  1.0,  0.2,  0.33, 0.5],  # timeliness (0.041)
            [2.0,  2.0,  2.0,  4.0,  5.0,  1.0,  3.0,  3.0],  # outlier_anomaly (0.259)
            [0.5,  0.5,  1.0,  2.0,  3.0,  0.33, 1.0,  1.0],  # distribution_balance (0.104)
            [0.5,  0.5,  0.5,  2.0,  2.0,  0.33, 1.0,  1.0],  # coverage_representativeness (0.085)
        ]),
        "lambda_max": 8.0982,
        "ci": 0.0140,
        "cr": 0.0100,
        "ri": 1.41,
        "provenance": PROVENANCE_EXPERT_PRIOR,
        "rationale": "Prioritizes outlier resilience due to quadratic loss sensitivity; completeness and validity follow."
    },
    "clustering": {
        "matrix": np.array([
            # CMP   VAL   CNS   UNQ   TIM   OUT   BAL   COV
            [1.0,  1.0,  2.0,  1.0,  5.0,  0.5,  3.0,  0.5],  # completeness (0.145)
            [1.0,  1.0,  2.0,  1.0,  5.0,  0.5,  3.0,  0.5],  # validity (0.132)
            [0.5,  0.5,  1.0,  0.5,  3.0,  0.33, 2.0,  0.33], # consistency (0.076)
            [1.0,  1.0,  2.0,  1.0,  4.0,  0.5,  2.0,  0.5],  # uniqueness (0.122)
            [0.2,  0.2,  0.33, 0.25, 1.0,  0.14, 0.5,  0.14], # timeliness (0.028)
            [2.0,  2.0,  3.0,  2.0,  7.0,  1.0,  4.0,  1.0],  # outlier_anomaly (0.232)
            [0.33, 0.33, 0.5,  0.5,  2.0,  0.25, 1.0,  0.25], # distribution_balance (0.051)
            [2.0,  2.0,  3.0,  2.0,  7.0,  1.0,  4.0,  1.0],  # coverage_representativeness (0.214)
        ]),
        "lambda_max": 8.0798,
        "ci": 0.0114,
        "cr": 0.0081,
        "ri": 1.41,
        "provenance": PROVENANCE_EXPERT_PRIOR,
        "rationale": "Prioritizes outlier anomaly quality and coverage density; distance spaces break under extreme anomalies."
    },
    "time_series_forecasting": {
        "matrix": np.array([
            # CMP   VAL   CNS   UNQ   TIM   OUT   BAL   COV
            [1.0,  2.0,  1.0,  4.0,  0.5,  1.0,  3.0,  3.0],  # completeness (0.161)
            [0.5,  1.0,  1.0,  3.0,  0.33, 1.0,  2.0,  2.0],  # validity (0.099)
            [1.0,  1.0,  1.0,  3.0,  0.33, 1.0,  2.0,  2.0],  # consistency (0.123)
            [0.25, 0.33, 0.33, 1.0,  0.14, 0.33, 0.5,  0.5],  # uniqueness (0.038)
            [2.0,  3.0,  3.0,  7.0,  1.0,  2.0,  5.0,  5.0],  # timeliness (0.310)
            [1.0,  1.0,  1.0,  3.0,  0.5,  1.0,  2.0,  2.0],  # outlier_anomaly (0.146)
            [0.33, 0.5,  0.5,  2.0,  0.2,  0.5,  1.0,  1.0],  # distribution_balance (0.062)
            [0.33, 0.5,  0.5,  2.0,  0.2,  0.5,  1.0,  1.0],  # coverage_representativeness (0.062)
        ]),
        "lambda_max": 8.0868,
        "ci": 0.0124,
        "cr": 0.0088,
        "ri": 1.41,
        "provenance": PROVENANCE_EXPERT_PRIOR,
        "rationale": "Timeliness is dominant due to chronological autoregression and lag dependence; completeness follows."
    }
}

TASK_AHP_PAIRWISE_MATRICES = AHP_MATRICES

# -------------------------------------------------------------------------
# Typed M(T, Q) Sensitivity Matrix and 8x8 AHP Pairwise Matrices
# -------------------------------------------------------------------------
TASK_QUALITY_SENSITIVITY_MATRIX: Dict[TaskType, Dict[QualityDimension, float]] = {
    TaskType.SUPERVISED_CLASSIFICATION: {
        QualityDimension.DISTRIBUTION_BALANCE: 0.22,
        QualityDimension.VALIDITY: 0.18,
        QualityDimension.COMPLETENESS: 0.14,
        QualityDimension.OUTLIER_ANOMALY: 0.12,
        QualityDimension.CONSISTENCY: 0.12,
        QualityDimension.COVERAGE_REPRESENTATIVENESS: 0.10,
        QualityDimension.UNIQUENESS: 0.08,
        QualityDimension.TIMELINESS: 0.04,
    },
    TaskType.SUPERVISED_REGRESSION: {
        QualityDimension.OUTLIER_ANOMALY: 0.24,
        QualityDimension.COMPLETENESS: 0.18,
        QualityDimension.VALIDITY: 0.16,
        QualityDimension.CONSISTENCY: 0.14,
        QualityDimension.DISTRIBUTION_BALANCE: 0.10,
        QualityDimension.COVERAGE_REPRESENTATIVENESS: 0.08,
        QualityDimension.UNIQUENESS: 0.06,
        QualityDimension.TIMELINESS: 0.04,
    },
    TaskType.CLUSTERING: {
        QualityDimension.OUTLIER_ANOMALY: 0.22,
        QualityDimension.COVERAGE_REPRESENTATIVENESS: 0.20,
        QualityDimension.COMPLETENESS: 0.16,
        QualityDimension.VALIDITY: 0.14,
        QualityDimension.UNIQUENESS: 0.12,
        QualityDimension.CONSISTENCY: 0.08,
        QualityDimension.DISTRIBUTION_BALANCE: 0.05,
        QualityDimension.TIMELINESS: 0.03,
    },
    TaskType.TIME_SERIES_FORECASTING: {
        QualityDimension.TIMELINESS: 0.30,
        QualityDimension.COMPLETENESS: 0.18,
        QualityDimension.OUTLIER_ANOMALY: 0.14,
        QualityDimension.CONSISTENCY: 0.12,
        QualityDimension.VALIDITY: 0.10,
        QualityDimension.DISTRIBUTION_BALANCE: 0.06,
        QualityDimension.COVERAGE_REPRESENTATIVENESS: 0.06,
        QualityDimension.UNIQUENESS: 0.04,
    },
}

AHP_PAIRWISE_MATRICES: Dict[TaskType, np.ndarray] = {
    TaskType.SUPERVISED_CLASSIFICATION: AHP_MATRICES["supervised_classification"]["matrix"],
    TaskType.SUPERVISED_REGRESSION: AHP_MATRICES["supervised_regression"]["matrix"],
    TaskType.CLUSTERING: AHP_MATRICES["clustering"]["matrix"],
    TaskType.TIME_SERIES_FORECASTING: AHP_MATRICES["time_series_forecasting"]["matrix"]
}

# -------------------------------------------------------------------------
# Three-Tier Defect Policy Configuration
# -------------------------------------------------------------------------
DEFECT_POLICY_CONFIG: Dict[str, Any] = {
    "provenance": PROVENANCE_POLICY_PRIOR,
    "tiers": {
        "GREEN": {
            "name": "NORMAL",
            "description": "Nominal data quality; fitness equals raw weighted fitness.",
            "enforce_cap": False,
            "provenance": PROVENANCE_POLICY_PRIOR
        },
        "YELLOW": {
            "name": "CAP / WARNING",
            "description": "Serious data defect limiting confidence; caps maximum fitness.",
            "enforce_cap": True,
            "rules": {
                "high_missingness": {"threshold_cell_rate": 0.15, "fitness_cap": 65.0},
                "high_duplicates": {"threshold_row_rate": 0.10, "fitness_cap": 70.0},
                "severe_imbalance": {"threshold_minority_share": 0.10, "fitness_cap": 60.0},
                "high_anomalies": {"threshold_anomaly_rate": 0.10, "fitness_cap": 65.0}
            },
            "provenance": PROVENANCE_POLICY_PRIOR
        },
        "RED": {
            "name": "HARD VETO / UNSAFE",
            "description": "Fatal structural defect rendering the dataset unsafe for the selected downstream task.",
            "enforce_cap": True,
            "rules": {
                "target_leakage": {
                    "threshold_correlation": 0.98,
                    "fitness_status": "UNSAFE",
                    "final_fitness": 0.0,
                    "provenance": PROVENANCE_POLICY_PRIOR
                },
                "temporal_sequence_disorder": {
                    "requires_task": "time_series_forecasting",
                    "fitness_status": "UNSAFE",
                    "final_fitness": 0.0,
                    "provenance": PROVENANCE_POLICY_PRIOR
                },
                "missing_required_target": {
                    "supervised_tasks": ["supervised_classification", "supervised_regression", "time_series_forecasting"],
                    "fitness_status": "UNSAFE",
                    "final_fitness": 0.0,
                    "provenance": PROVENANCE_POLICY_PRIOR
                },
                "catastrophic_missingness": {
                    "threshold_cell_rate": 0.70,
                    "fitness_status": "UNSAFE",
                    "final_fitness": 0.0,
                    "provenance": PROVENANCE_POLICY_PRIOR
                }
            },
            "provenance": PROVENANCE_POLICY_PRIOR
        }
    }
}
