from enum import Enum
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class TaskType(str, Enum):
    SUPERVISED_CLASSIFICATION = "supervised_classification"
    SUPERVISED_REGRESSION = "supervised_regression"
    CLUSTERING = "clustering"
    TIME_SERIES_FORECASTING = "time_series_forecasting"

    # Backward-compatibility aliases
    CLASSIFICATION = "supervised_classification"
    REGRESSION = "supervised_regression"
    DESCRIPTIVE_BI = "clustering"


class QualityDimension(str, Enum):
    # Eight Canonical Quality Dimensions (v3)
    COMPLETENESS = "completeness"                      # CMP: Completeness
    VALIDITY = "validity"                              # VAL: Validity
    CONSISTENCY = "consistency"                        # CNS: Consistency
    UNIQUENESS = "uniqueness"                          # UNQ: Uniqueness
    TIMELINESS = "timeliness"                          # TIM: Timeliness
    OUTLIER_ANOMALY = "outlier_anomaly"                # OUT: Outlier/Anomaly
    DISTRIBUTION_BALANCE = "distribution_balance"      # BAL: Distribution Balance
    COVERAGE_REPRESENTATIVENESS = "coverage_representativeness" # COV: Coverage/Support

    # Backward-compatibility aliases (Explicit aliases ONLY, NOT independent dimensions)
    MISSINGNESS = "completeness"
    OUTLIER_RESILIENCE = "outlier_anomaly"
    DRIFT_STABILITY = "timeliness"
    LABEL_INTEGRITY = "distribution_balance"
    LEAKAGE = "validity"
    CORRELATION_INTEGRITY = "consistency"
    BIAS_IMBALANCE = "distribution_balance"
    CARDINALITY_SCHEMA = "validity"
    TEMPORAL_VALIDITY = "timeliness"


CANONICAL_QUALITY_DIMENSIONS = [
    QualityDimension.COMPLETENESS,
    QualityDimension.VALIDITY,
    QualityDimension.CONSISTENCY,
    QualityDimension.UNIQUENESS,
    QualityDimension.TIMELINESS,
    QualityDimension.OUTLIER_ANOMALY,
    QualityDimension.DISTRIBUTION_BALANCE,
    QualityDimension.COVERAGE_REPRESENTATIVENESS
]

DIMENSION_CODES = {
    QualityDimension.COMPLETENESS: "CMP",
    QualityDimension.VALIDITY: "VAL",
    QualityDimension.CONSISTENCY: "CNS",
    QualityDimension.UNIQUENESS: "UNQ",
    QualityDimension.TIMELINESS: "TIM",
    QualityDimension.OUTLIER_ANOMALY: "OUT",
    QualityDimension.DISTRIBUTION_BALANCE: "BAL",
    QualityDimension.COVERAGE_REPRESENTATIVENESS: "COV"
}


class RemediationDecision(str, Enum):
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    NO_IMPROVEMENT = "NO_IMPROVEMENT"
    REGRESSED = "REGRESSED"
    BLOCKED = "BLOCKED"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"


class DimensionResult(BaseModel):
    dimension: QualityDimension
    score: float = Field(..., ge=0.0, le=100.0, description="Normalized score 0-100")
    weight: float = Field(..., ge=0.0, le=1.0, description="Normalized task weight")
    weighted_score: float = Field(..., ge=0.0, le=100.0, description="Score * weight")
    sensitivity_level: str = Field(default="MEDIUM", description="HIGH, MEDIUM, or LOW")
    raw_metrics: Dict[str, Any] = Field(default_factory=dict)
    summary: str
    formula: str = Field(default="", description="Mathematical formulation/transformation")
    assumptions: str = Field(default="", description="Methodological assumptions")
    limitations: str = Field(default="", description="Known limitations of metric")
    task_applicability: str = Field(default="", description="Task relevance context")
    provenance: str = Field(default="HEURISTIC", description="HEURISTIC, LITERATURE / EXPERT PRIOR, etc.")
    is_veto_triggered: bool = False
    veto_reason: Optional[str] = None


class RemediationRecommendation(BaseModel):
    dimension: QualityDimension
    severity: str = Field(..., description="'critical', 'high', 'medium', or 'low'")
    issue: str
    recommendation: str
    estimated_score_impact: float = Field(..., description="Potential point gain if resolved")


class FactorAttribution(BaseModel):
    dimension: QualityDimension
    penalty_points: float = Field(..., description="Points lost due to this dimension")
    percentage_of_loss: float = Field(..., description="Share of total lost points")
    explanation: str


class ProfilingMetadata(BaseModel):
    num_rows: int
    num_columns: int
    column_types: Dict[str, str]
    numeric_columns: List[str]
    categorical_columns: List[str]
    datetime_columns: List[str]
    missing_cell_percentage: float
    duplicate_row_count: int
    memory_usage_mb: float
    dataset_sha256: str = Field(default="unavailable", description="SHA-256 hash of dataset bytes where available")
    provenance: str = Field(default="unavailable", description="Dataset origin / provenance metadata")
    candidate_targets: List[str] = Field(default_factory=list, description="Candidate target columns")
    candidate_timestamps: List[str] = Field(default_factory=list, description="Candidate timestamp columns")
    candidate_target_details: List[Dict[str, Any]] = Field(default_factory=list, description="Detailed candidate target evidence and rankings")


class ImbalancePolicy(BaseModel):
    minority_threshold_pct: float = Field(default=2.0, ge=0.1, le=50.0, description="Minimum minority representation (%) before triggering bias penalty/veto")
    severe_threshold_pct: float = Field(default=10.0, ge=0.5, le=50.0, description="Threshold for severe imbalance penalty (%)")
    veto_enabled: bool = Field(default=True, description="Enforce non-compensatory veto ceiling if minority < minority_threshold_pct")
    task: TaskType = TaskType.SUPERVISED_CLASSIFICATION
    metric: str = "f1_weighted"


class DefectRecord(BaseModel):
    defect_name: str
    severity: str = Field(..., description="'CRITICAL', 'WARNING', or 'NORMAL'")
    tier: str = Field(..., description="'GREEN', 'YELLOW', or 'RED'")
    evidence: str
    threshold: str
    threshold_provenance: str = Field(default="HEURISTIC / POLICY PRIOR", description="Provenance of threshold")
    policy_action: str


class DefectPolicyResult(BaseModel):
    tier: str = Field(default="GREEN", description="'GREEN', 'YELLOW', or 'RED'")
    fitness_status: str = Field(default="NORMAL", description="'NORMAL', 'WARNING', or 'UNSAFE'")
    raw_fitness: float = Field(..., ge=0.0, le=100.0, description="Raw un-capped weighted fitness")
    final_fitness: Optional[float] = Field(default=None, description="Final fitness after defect policy (None if UNSAFE)")
    is_unsafe: bool = Field(default=False, description="True if RED hard veto is triggered")
    cap_applied: Optional[float] = Field(default=None, description="Numerical ceiling applied if YELLOW or RED cap")
    defects_detected: List[DefectRecord] = Field(default_factory=list, description="Detected defects and policy triggers")
    provenance: str = Field(default="HEURISTIC / POLICY PRIOR", description="Provenance of defect policy")

    @property
    def effective_fitness(self) -> float:
        return self.final_fitness if self.final_fitness is not None else 0.0


class DownstreamAuditReceipt(BaseModel):
    evaluation_id: str
    dataset_id: Optional[str] = None
    random_seed: int = 42
    random_state: int = 42
    train_indices_hash: str
    test_indices_hash: str
    test_index_hash_before: str = ""
    test_index_hash_after: str = ""
    test_indices_immutable: bool = True
    train_sample_count: int
    test_sample_count: int
    feature_columns: List[str]
    target_column: Optional[str] = None
    time_column: Optional[str] = None
    preprocessing_parameters: Dict[str, Any] = Field(default_factory=dict)
    preprocessing_operations: List[str] = Field(default_factory=list)
    remediation_operations: List[str] = Field(default_factory=list)
    model_family: str
    model_configuration: Dict[str, Any] = Field(default_factory=dict)
    model_hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    baseline_metric_value: float
    remediated_metric_value: float
    metric_name: str
    leakage_free_verified: bool = True
    test_set_resampled: bool = False
    timestamp: str


# --- DataTrust AI Research Schemas ---

class FitnessUncertainty(BaseModel):
    ci_lower: float = Field(..., description="Lower bound of estimated fitness interval")
    ci_upper: float = Field(..., description="Upper bound of estimated fitness interval")
    interval_lower: float = Field(default=0.0, description="Lower bound of estimated fitness interval")
    interval_upper: float = Field(default=100.0, description="Upper bound of estimated fitness interval")
    margin_of_error: float = Field(..., description="Margin of error delta (k * composite_se)")
    composite_variance: float = Field(default=1.0, description="Composite variance: sum of component variances")
    composite_se: float = Field(default=1.0, description="Composite standard error: sqrt(composite_variance)")
    std_dev: float = Field(..., description="Bootstrap standard deviation of fitness realizations")
    variance_components: Dict[str, float] = Field(default_factory=dict, description="Variance contributions")
    confidence_level: float = Field(default=0.95, description="Nominal coverage factor")
    interval_type: str = Field(default="Estimated Fitness Interval (Empirical)", description="Descriptive label for uncertainty interval")
    methodological_note: str = Field(
        default="Estimated Fitness Interval (Empirical) represents an empirical multi-source dispersion estimate and is not a formal parametric confidence interval.",
        description="Clarification on empirical uncertainty interpretation."
    )


class TaskInferenceResult(BaseModel):
    inferred_task: TaskType
    confidence: float = Field(..., ge=0.0, le=1.0, description="Inference confidence score (0-1)")
    confidence_type: str = Field(default="heuristic_structural_score", description="Explicit classification of confidence")
    confidence_vector: Dict[str, float] = Field(
        default_factory=dict,
        description="Probability distribution across all tasks P(T|D) (sums to 1.0)"
    )
    p_task_given_data: Dict[str, float] = Field(
        default_factory=dict,
        description="Normalized empirical task-probability distribution P(T|D)"
    )
    positive_evidence: List[str] = Field(
        default_factory=list,
        description="Empirical signals supporting this task classification"
    )
    negative_evidence: List[str] = Field(
        default_factory=list,
        description="Empirical signals refuting alternative task classifications"
    )
    ambiguity_warning: Optional[str] = Field(
        default=None,
        description="Warning if task determination is structurally ambiguous"
    )
    user_overrode: bool = Field(
        default=False,
        description="True if user manually confirmed a different task than inferred"
    )
    final_selected_task: Optional[TaskType] = Field(
        default=None,
        description="Governing task for downstream evaluation"
    )
    reasoning: str
    candidate_targets: List[str]
    candidate_timestamps: List[str]
    detected_target: Optional[str] = Field(default=None, description="Primary detected target column")
    target_evidence: List[str] = Field(default_factory=list, description="Evidence supporting target detection")
    candidate_target_details: List[Dict[str, Any]] = Field(default_factory=list, description="Detailed candidate target evidence and rankings")


class CalibrationResult(BaseModel):
    metric_name: str = Field(default="Metric", description="Task-specific metric (e.g. RMSE or F1)")
    surrogate_predicted_performance: float = Field(..., description="Initial surrogate prediction in native task domain")
    predicted_performance: float = Field(..., description="Alias for surrogate_predicted_performance")
    observed_performance: float = Field(..., description="Observed model metric in native domain")
    residual: float = Field(..., description="Residual epsilon = observed - surrogate_predicted")
    residual_before: float = Field(default=0.0, description="Residual before calibration")
    residual_after: float = Field(default=0.0, description="Residual after calibration")
    loss_before: float = Field(default=0.0, description="Loss L(w) = |residual_before|")
    loss_after: float = Field(default=0.0, description="Loss L(w_calibrated) = |residual_after|")
    residual_reduction_pct: float = Field(default=0.0, description="Percentage reduction in absolute residual")
    is_improved: bool = Field(default=False, description="True only if loss_after < loss_before")
    calibration_status: str = Field(default="No improvement", description="'Improved', 'No improvement', 'Converged', or 'Rejected'")
    weights_ahp: Dict[str, float] = Field(default_factory=dict, description="Prior task weights W_AHP")
    weights_before: Dict[str, float] = Field(default_factory=dict, description="Alias for weights_ahp")
    weights_calibrated: Dict[str, float] = Field(default_factory=dict, description="Calibrated weights W_calibrated")
    weights_after: Dict[str, float] = Field(default_factory=dict, description="Alias for weights_calibrated")
    weight_adjustments: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        description="Map of dimension to {'initial': w_ahp, 'calibrated': w_cal, 'delta': dw}"
    )
    initial_trust_score: float = 0.0
    calibrated_trust_score: float = 0.0
    calibrated_predicted_performance: float = 0.0
    provenance: str = Field(default="HEURISTIC_PRIOR", description="PRIOR, EMPIRICALLY_CALIBRATED, REJECTED, or NO_IMPROVEMENT")
    explanation: str = ""


class RemediationActionV2(BaseModel):
    id: str
    action_type: str = Field(default="", description="Canonical action type key in ActionRegistry")
    dimension: QualityDimension
    issue: str
    recommendation: str
    severity: str
    fitness_improvement: float = Field(..., description="Predicted gain in Fitness")
    operational_cost: int = Field(..., ge=1, le=5, description="Remediation operational cost (1 to 5 scale)")
    mru: float = Field(..., description="Marginal Remediation Utility: Delta Fitness / Cost")
    priority_rank: int


class WhatIfScenario(BaseModel):
    name: str
    description: str
    action_ids: List[str]
    simulated_fitness: float
    improvement: float
    is_recommended: bool = False


class BeforeAfterValidationResult(BaseModel):
    fitness_before: float
    fitness_after: float
    quality_before: float = Field(default=0.0, description="Quality score before remediation")
    quality_after: float = Field(default=0.0, description="Quality score after remediation")
    predicted_fitness_gain: float = Field(default=0.0, description="Predicted fitness gain")
    observed_fitness_gain: float = Field(default=0.0, description="Observed fitness gain")
    fitness_improvement: float = Field(default=0.0, description="Alias for observed_fitness_gain")
    metric_name: str
    metric_before: float
    metric_after: float
    downstream_metric_before: float = Field(default=0.0, description="Downstream metric before remediation")
    downstream_metric_after: float = Field(default=0.0, description="Downstream metric after remediation")
    residual_before: Optional[float] = Field(default=None, description="Model residual before remediation")
    residual_after: Optional[float] = Field(default=None, description="Model residual after remediation")
    predicted_metric_gain: Optional[float] = Field(default=None, description="Predicted metric change")
    observed_metric_gain: float = Field(default=0.0, description="Observed metric change")
    metric_change_pct: float = Field(default=0.0, description="Percentage change in downstream metric")
    metric_improvement_pct: float = Field(default=0.0, description="Alias for metric_change_pct")
    performance_status: str = Field(default="Neutral", description="'Improved ✓', 'Deteriorated ✗', or 'Neutral'")
    decision: RemediationDecision = Field(default=RemediationDecision.NO_IMPROVEMENT, description="Closed-loop remediation decision outcome")
    decision_reason: str = Field(default="", description="Scientific explanation of remediation decision")
    raw_stats: Dict[str, Any] = Field(default_factory=dict)
    remediated_stats: Dict[str, Any] = Field(default_factory=dict)
    remediation_actions_applied: List[str]
    audit_receipt: Optional[DownstreamAuditReceipt] = None
    execution_log: List[str]


class ReproducibilityReceipt(BaseModel):
    dataset_hash: str
    timestamp: str
    evaluation_id: str
    task: str
    confidence_vector: Dict[str, float]
    sensitivity_matrix_row: Dict[str, float]
    ahp_cr: float
    initial_weights: Dict[str, float]
    veto_triggered: bool
    veto_ceiling: Optional[float] = None
    initial_fitness: float
    fitness_ci: Optional[str] = None
    model_name: str
    metric_name: str
    predicted_metric: float
    observed_metric: float
    residual: float
    calibrated_weights: Dict[str, float]
    calibrated_fitness: float
    residual_after: float
    remediations_applied: List[str] = Field(default_factory=list)
    fitness_after: Optional[float] = None
    metric_after: Optional[float] = None


class TrustReport(BaseModel):
    dataset_name: str
    task_type: TaskType
    target_column: Optional[str] = None
    time_column: Optional[str] = None

    # Fitness Scores
    raw_fitness: float = Field(default=0.0, ge=0.0, le=100.0, description="Un-capped weighted linear fitness score")
    final_fitness: Optional[float] = Field(default=None, description="Final fitness after defect policy (None if UNSAFE)")
    task_conditioned_fitness: float = Field(default=0.0, ge=0.0, le=100.0, description="Central technical metric (capped/adjusted)")
    trust_score: float = Field(default=0.0, ge=0.0, le=100.0, description="Backward compatibility alias")
    raw_score: float = Field(default=0.0, ge=0.0, le=100.0, description="Backward compatibility alias")
    fitness_status: str = Field(default="NORMAL", description="'NORMAL', 'WARNING', or 'UNSAFE'")
    confidence_tier: str = Field(default="Normal", description="Fitness tier")

    # Three-Tier Defect Policy Result
    defect_policy: Optional[DefectPolicyResult] = None
    veto_applied: bool = False
    veto_message: Optional[str] = None

    # Multi-Dimensional Results (Eight Canonical Dimensions)
    dimensions: Dict[str, DimensionResult]
    factor_attributions: List[FactorAttribution] = Field(default_factory=list)
    remediations: List[RemediationRecommendation] = Field(default_factory=list)
    profiling: ProfilingMetadata
    evaluated_at: str

    # Autonomous Task Characterization
    task_inference: Optional[TaskInferenceResult] = None
    task_quality_sensitivities: Dict[str, float] = Field(default_factory=dict)
    sensitivity_matrix_row: Dict[str, float] = Field(default_factory=dict)
    m_t_q: Dict[str, Any] = Field(default_factory=dict, description="Task x Quality sensitivity matrix M(T, Q)")

    # Fixed AHP Matrix & Consistency Details
    ahp_pairwise_matrix: Optional[List[List[float]]] = None
    ahp_consistency_ratio: float = 0.0
    ahp_cr: Optional[float] = None
    ahp_ci: Optional[float] = None
    ahp_lambda_max: Optional[float] = None
    ahp_ri: float = 1.41
    ahp_consistency_status: str = "Consistent (CR < 0.10)"
    pipeline_provenance: str = Field(default="", description="Pipeline provenance")

    # Weights
    weights_ahp: Dict[str, float] = Field(default_factory=dict)
    weights_calibrated: Dict[str, float] = Field(default_factory=dict)

    # Uncertainty (Phase 2 preserved)
    fitness_uncertainty: Optional[FitnessUncertainty] = None

    # Downstream Validation & Calibration (Phase 2/3 preserved)
    model_family: Optional[str] = None
    metric: Optional[str] = None
    task_confidence: Optional[float] = None
    surrogate_predicted_metric: Optional[float] = None
    observed_model_metric: Optional[float] = None
    prediction_residual_before: Optional[float] = None
    prediction_residual_after: Optional[float] = None
    predicted_downstream_metric: Optional[float] = None
    predicted_metric_name: str = "RMSE"
    predicted_metric_value: Optional[float] = None
    observed_metric_value: Optional[float] = None
    metric_residual: Optional[float] = None

    # Optimization & Validation (Phase 2/3 preserved)
    calibration: Optional[CalibrationResult] = None
    optimized_remediations: List[RemediationActionV2] = Field(default_factory=list)
    what_if_scenarios: List[WhatIfScenario] = Field(default_factory=list)
    counterfactual_table: List[Dict[str, Any]] = Field(default_factory=list)
    before_after_validation: Optional[BeforeAfterValidationResult] = None
    downstream_audit_receipt: Optional[DownstreamAuditReceipt] = None

    # Reproducibility
    reproducibility_hash: Optional[str] = None
