# DataTrust AI v3 — End-to-End Pipeline Specification

This document traces the complete execution flow of DataTrust AI v3, from raw dataset ingestion to final fitness evaluation, remediation, downstream validation, performance calibration, and Marginal Remediation Utility (MRU) optimization.

Every stage records the actual implementation file, class/function, input/output schemas, governing mathematical formulas or decision rules, provenance metadata, and inter-stage dependencies.

---

## High-Level Execution Diagram

```
[1] Dataset Ingestion
       │
       ▼
[2] Profiling (Deterministic SHA-256)
       │
       ▼
[3] Task Inference P(T|D)
       │
       ▼
[4] User Confirmation / Manual Override
       │
       ▼
[5] Eight Canonical Quality Dimensions S_i
       │
       ▼
[6] Task-Quality Sensitivity Mapping M(T, Q)
       │
       ▼
[7] Fixed AHP Weighting W_AHP (8x8, CR < 0.10, RI = 1.41)
       │
       ▼
[8] Raw Fitness F_raw = sum(w_i * S_i)
       │
       ▼
[9] Three-Level Defect Policy (GREEN / YELLOW / RED)
       │
       ▼
[10] Final Fitness F_final & Status (NORMAL / WARNING / UNSAFE)
       │
       ▼
[11] Split-Aware Remediation (Fit on Train, Zero Test Resampling)
       │
       ▼
[12] Downstream Empirical Model Validation (RMSE / F1)
       │
       ▼
[13] Closed-Loop Performance Calibration L(w) = |y_hat - y_obs|
       │
       ▼
[14] Marginal Remediation Utility (MRU) Optimization & What-If Simulation
```

---

## Detailed Stage-by-Stage Specification

### Stage 1: Dataset Ingestion
- **Python File:** `datatrust/engine.py` / `backend/routes/evaluate.py`
- **Class / Function:** `DataTrustEngine.evaluate()`, `backend.routes.evaluate.parse_uploaded_file()`
- **Input:**
  - File upload (`.csv`, `.parquet`, `.xlsx`, `.xls`) or pandas DataFrame `df: pd.DataFrame`
  - Optional parameters: `task: Optional[TaskType]`, `target_column: Optional[str]`, `time_column: Optional[str]`, `observed_performance: Optional[float]`, `imbalance_policy: Optional[ImbalancePolicy]`
- **Output:** In-memory pandas DataFrame with inferred or designated target/timestamp column pointers.
- **Rules & Logic:**
  - Validates file non-emptiness; parses bytes into tabular DataFrame via `io.BytesIO`.
  - Normalizes column names and validates non-zero row count ($N \ge 1$).
- **Provenance / Configuration:** `PROVENANCE_PIPELINE = "PHASE 1 CORE FITNESS PIPELINE (PRE-EXPERIMENTAL)"`
- **Dependencies:** Base entry point for all subsequent pipeline stages.

---

### Stage 2: Profiling & Cryptographic Hashing
- **Python File:** `datatrust/profiler.py`
- **Class / Function:** `DataProfiler.profile(df: pd.DataFrame, raw_bytes: Optional[bytes] = None)`
- **Input:** `df: pd.DataFrame`, optional `raw_bytes: bytes`
- **Output:** `ProfilingMetadata`, `column_details: Dict[str, Any]`
- **Equations / Rules:**
  - **Deterministic SHA-256 Hash:**
    $$\text{SHA-256}(D) = \text{hashlib.sha256}(\text{pd.util.hash\_pandas\_object}(D).\text{values}.\text{tobytes}()).\text{hexdigest}()$$
  - **Missingness Rate:**
    $$\text{missing\_cell\_pct} = \frac{\sum_{i=1}^N \sum_{j=1}^P \mathbb{I}(x_{ij} \text{ is null})}{N \times P} \times 100$$
  - **Target Candidate Discovery:** Matching column names against regex/set `['target', 'label', 'class', 'churn', 'fraud', 'price', 'sales', 'revenue', ...]`
  - **Timestamp Candidate Discovery:** Matching column names against `['date', 'timestamp', 'time', 'datetime', 'dt', 'year']` and verifying datetime parsability $> 50\%$ rows.
- **Provenance / Configuration:** `PROVENANCE_PROFILING = "DETERMINISTIC / EMPIRICAL DATASET"`
- **Dependencies:** Ingestion (Stage 1).

---

### Stage 3: Autonomous Task Inference
- **Python File:** `datatrust/task_infer.py`
- **Class / Function:** `TaskInferenceEngine.infer_task(df, explicit_target, explicit_time, user_override_task)`
- **Input:** `df: pd.DataFrame`, optional explicit column names, optional user override.
- **Output:** `TaskInferenceResult`
- **Equations / Rules (Condition 1):**
  - Evaluates heuristic evidence vector across 4 task profiles:
    - **Time-Series Forecasting:** Explicit or discovered timestamp column with valid monotonic continuity ($+1.25$ score).
    - **Supervised Regression:** Continuous numeric target with cardinality $> 15$ ($+0.95$ score).
    - **Supervised Classification:** Discrete target with cardinality $\le 15$ or nominal categorical labels ($+0.95$ score).
    - **Clustering Hypothesis:** Unlabelled tabular data without designated target or timestamp index.
  - **Condition 1 Bounded Clustering Confidence:**
    $$P(\text{Clustering} \mid D) \le 0.60 \quad (\text{heuristic score} = 0.55)$$
    Generates mandatory ambiguity warning: `"Clustering is an unsupervised exploratory hypothesis and cannot be definitively inferred from schema structure alone; manual confirmation recommended."`
  - Normalization onto probability simplex:
    $$P(T_j \mid D) = \frac{\text{score}(T_j)}{\sum_{k=1}^4 \text{score}(T_k)}, \quad \sum_{j=1}^4 P(T_j \mid D) = 1.0$$
- **Provenance / Configuration:** `PROVENANCE_TASK_INFERENCE = "HEURISTIC / EMPIRICAL SCHEMA PRIOR"`
- **Dependencies:** Profiling (Stage 2).

---

### Stage 4: User Confirmation / Manual Override
- **Python File:** `datatrust/task_infer.py` / `datatrust/engine.py`
- **Class / Function:** `TaskInferenceEngine.infer_task(..., user_override_task=TaskType)`, `DataTrustEngine.evaluate(..., task=TaskType)`
- **Input:** `user_override_task: Optional[TaskType]`
- **Output:** `final_selected_task: TaskType`, `user_overrode: bool`
- **Rules:**
  - If user provides an explicit task choice, `final_selected_task` is set to user choice.
  - Inferred hypothesis is permanently preserved in `inferred_task` for scientific auditability.
  - Flag `user_overrode = (user_override_task != inferred_task)` recorded in `TaskInferenceResult`.
- **Provenance:** `HEURISTIC / USER OVERRIDE`
- **Dependencies:** Task Inference (Stage 3).

---

### Stage 5: Multi-Dimensional Quality Evaluation (8 Canonical Dimensions)
- **Python File:** `datatrust/quality_engine.py`
- **Class / Function:** `QualityEngine.run_all_dimensions(task: TaskType)` (alias `evaluate_all()`)
- **Input:** `df: pd.DataFrame`, `task: TaskType`, target and time column names.
- **Output:** `Dict[QualityDimension, DimensionResult]` across all 8 canonical dimensions.
- **Governing Equations & Research Rules:**

| Dimension | Formula / Metric | Assumptions & Condition Rules | Provenance |
| :--- | :--- | :--- | :--- |
| **1. Completeness** | $S_{\text{cmp}} = \max(0, \min(100, 100 - (80 \cdot \text{missing\_rate} + 5 \cdot C_{\text{high}} + 15 \cdot C_{\text{empty}})))$ | Evaluates observed missing cell footprint; agnostic to missingness mechanism (MCAR/MAR/MNAR). | `HEURISTIC` |
| **2. Validity** | $S_{\text{val}} = \max(0, \min(100, 100 - (10 \cdot C_{\text{const}} + 100 \cdot \text{inf\_rate} + \text{target\_penalty})))$ | Constant columns ($C_{\text{const}}$) and infinite values violate structural admissibility. Veto flag raised if target has zero variance. | `HEURISTIC` |
| **3. Consistency** | $S_{\text{cns}} = \max(15, \min(100, 100 - (6 \cdot N_{\text{collinear}} + 25 \cdot N_{\text{leakage}})))$ | **Condition 2:** Measures internal contradictions and collinearity ($|r| > 0.92$). Distribution drift is **strictly excluded** (evaluated in Timeliness). | `HEURISTIC` |
| **4. Uniqueness** | $S_{\text{unq}} = \max(0, \min(100, 100 \cdot (1 - \text{dup\_rate}) - 20 \cdot \text{id\_collision\_rate}))$ | Redundant rows artificially deflate standard errors and skew cluster centroids. | `HEURISTIC` |
| **5. Timeliness / Freshness** | $S_{\text{tim}} = \max(0, \min(100, 100 - (45 \cdot \mathbb{I}(\neg\text{monotonic}) + 30 \cdot \text{dup\_ts\_rate} + 15 \cdot \text{CV}_{\text{cadence}} + \text{drift})))$ | Sequence disorder, duplicate timestamps, and cadence coefficient of variation ($\text{CV} > 1.5$) penalize temporal continuity. Drift evaluated here. | `HEURISTIC` |
| **6. Outlier / Anomaly Quality** | $S_{\text{out}} = \max(10, \min(100, 100 - 300 \cdot \max(0, c - 0.02) - 80 \cdot c))$ | **Condition 3:** Primary score derived strictly from Isolation Forest multivariate anomaly density $c \in [0, 1]$. Kurtosis is reported as **descriptive supporting metric only**. | `HEURISTIC` |
| **7. Distribution Balance** | Classification: $S_{\text{bal}} = \max(10, \min(100, 100 - \text{penalty}(\text{minority\_share})))$<br>Regression: $S_{\text{bal}} = \max(15, \min(100, 100 - \min(50, 8 \cdot |\text{skew}|)))$ | **Condition 7:** Evaluates class entropy/imbalance ratio and target skewness. Supports configurable `ImbalancePolicy`. | `HEURISTIC` |
| **8. Coverage / Representativeness** | $S_{\text{cov}} = \max(20, \min(100, 50 + 50 \cdot \min(1.0, \frac{N}{10 \cdot \max(1, P)})))$ | **Condition 4:** Measures observable feature space support ($N/P \ge 10:1$). Population representativeness is explicitly marked **`"UNAVAILABLE"`**. | `HEURISTIC` |

- **Dependencies:** Task Inference / Override (Stages 3 & 4), Profiling (Stage 2).

---

### Stage 6: Task × Quality Sensitivity Mapping $M(T, Q)$
- **Python File:** `datatrust/sensitivity.py`
- **Class / Function:** `TaskQualitySensitivityMatrix.get_sensitivity_vector()`, `TaskQualitySensitivityMatrix.get_full_sensitivity_matrix()`
- **Input:** `task: TaskType`
- **Output:** 8-dimensional sensitivity vector $M(T, :) \in \mathbb{R}^8$
- **Rules (Condition 5):**
  - $4 \times 8$ sensitivity tensor loaded from `TASK_QUALITY_SENSITIVITY_MATRIX` in `datatrust/config.py`.
  - The $4 \times 8$ matrix $M(T, Q)$ is **not** an AHP pairwise comparison matrix; it represents task vulnerability priors:
    - **Classification:** Dominant in Distribution Balance ($0.22$) and Validity ($0.18$).
    - **Regression:** Dominant in Outlier / Anomaly ($0.24$) and Completeness ($0.18$).
    - **Time Series:** Dominant in Timeliness ($0.30$) and Completeness ($0.18$).
    - **Clustering:** Dominant in Outlier / Anomaly ($0.22$) and Coverage ($0.20$).
- **Provenance / Configuration:** `PROVENANCE_SENSITIVITY_MATRIX = "LITERATURE / EXPERT PRIOR (PRE-EXPERIMENTAL UNCALIBRATED)"`
- **Dependencies:** Task Selection (Stage 4).

---

### Stage 7: Fixed AHP Weighting Engine
- **Python File:** `datatrust/weighting.py`
- **Class / Function:** `AHPWeightingEngine.get_task_weights()`, `AHPWeightingEngine.solve_weights_and_consistency()`
- **Input:** $8 \times 8$ positive reciprocal pairwise comparison matrix $A_T \in \mathbb{R}^{8 \times 8}$ from `TASK_PAIRWISE_PREFERENCES`
- **Output:** Normalized AHP weight vector $W_{\text{AHP}} \in \Delta^7$ ($\sum_{i=1}^8 w_i = 1.0$), Consistency Ratio $\text{CR}$, $\lambda_{\max}$, $\text{CI}$.
- **Equations (Condition 6):**
  - **Principal Eigenvector Solution:**
    $$A_T w = \lambda_{\max} w, \quad w_i = \frac{v_i}{\sum_{j=1}^8 v_j}$$
  - **Consistency Index (CI):**
    $$\text{CI} = \frac{\lambda_{\max} - n}{n - 1} \quad (n = 8)$$
  - **Saaty Random Index (RI):** $\text{RI} = 1.41$ for $n=8$ (from Saaty 1980 table in `datatrust/config.py`).
  - **Consistency Ratio (CR):**
    $$\text{CR} = \frac{\text{CI}}{\text{RI}} < 0.10 \quad (\text{Verified: all 4 tasks satisfy } \text{CR} \le 0.021)$$
- **Provenance / Configuration:** `PROVENANCE_AHP_WEIGHTS = "LITERATURE / EXPERT PRIOR (PRE-EXPERIMENTAL UNCALIBRATED)"`
- **Dependencies:** Task Selection (Stage 4).

---

### Stage 8: Raw Fitness Computation
- **Python File:** `datatrust/engine.py`
- **Class / Function:** `DataTrustEngine.evaluate()` (Step 5)
- **Input:** Dimension scores $S_i \in [0, 100]$, AHP weights $w_i \in [0, 1]$
- **Output:** Raw un-capped linear fitness $F_{\text{raw}} \in [0, 100]$
- **Equation:**
  $$F_{\text{raw}} = \sum_{i=1}^8 w_i \cdot S_i, \quad \sum_{i=1}^8 w_i = 1.0$$
- **Provenance:** `EMPIRICAL / WEIGHTED AGGREGATION`
- **Dependencies:** Quality Evaluation (Stage 5), AHP Weights (Stage 7).

---

### Stage 9: Three-Level Defect Policy Subsystem
- **Python File:** `datatrust/veto_engine.py`
- **Class / Function:** `IntelligentVetoEngine.evaluate_defect_policy()`
- **Input:** `task: TaskType`, `dimension_results: Dict[QualityDimension, DimensionResult]`, `raw_fitness: float`, `num_rows: int`
- **Output:** `DefectPolicyResult` (`tier: str`, `fitness_status: str`, `final_fitness: Optional[float]`, `cap_applied: Optional[float]`, `defects_detected: List[DefectRecord]`)
- **Rules & Tiers (Condition 8):**
  - **GREEN (Normal):** No critical or warning defects detected.
    $$\text{tier} = \text{"GREEN"}, \quad \text{fitness\_status} = \text{"NORMAL"}, \quad F_{\text{effective}} = F_{\text{raw}}$$
  - **YELLOW (Warning / Cap):** Non-fatal defects detected (e.g. moderate missingness $30\text{--}70\%$, high collinearity $>6$ pairs, minority share $2\text{--}5\%$, target skewness $>2.5$).
    $$\text{tier} = \text{"YELLOW"}, \quad \text{fitness\_status} = \text{"WARNING"}, \quad F_{\text{effective}} = \min(F_{\text{raw}}, \text{cap}) \quad (\text{cap} \in [60.0, 70.0])$$
  - **RED (Hard Non-Compensatory Veto):** Fatal defects rendering dataset unusable for task:
    - *Classification:* Severe class collapse (minority share $< 2.0\%$).
    - *Time Series:* Non-monotonic sequence disorder ($\text{monotonic} = \text{False}$) or duplicate timestamps ($N_{\text{dup}} > 0$).
    - *All Tasks:* Catastrophic missingness ($>70\%$ null cells), target zero variance, target leakage ($|r| > 0.98$).
    $$\text{tier} = \text{"RED"}, \quad \text{fitness\_status} = \text{"UNSAFE"}, \quad F_{\text{final}} = 0.0, \quad \text{is\_unsafe} = \text{True}$$
- **Provenance / Configuration:** `PROVENANCE_DEFECT_POLICY = "HEURISTIC / POLICY PRIOR"`
- **Dependencies:** Quality Dimensions (Stage 5), Raw Fitness (Stage 8).

---

### Stage 10: Final Fitness Determination & Reporting
- **Python File:** `datatrust/engine.py`
- **Class / Function:** `DataTrustEngine.evaluate()` (Steps 6, 9, 14)
- **Input:** Raw fitness $F_{\text{raw}}$, `DefectPolicyResult`, dimension scores.
- **Output:** `TrustReport` containing `final_fitness`, `task_conditioned_fitness`, `fitness_status`, `confidence_tier`, `pipeline_provenance`, and cryptographic audit hashes.
- **Confidence Tiers:**
  - If RED: `"Critical Defect / Unfit"`
  - If $F \ge 85.0$: `"High Fitness / Low Risk"`
  - If $70.0 \le F < 85.0$: `"Moderate Fitness / Conditional"`
  - If $50.0 \le F < 70.0$: `"Sub-Optimal Fitness / High Risk"`
  - If $F < 50.0$: `"Critical Defect / Unfit"`
- **Dependencies:** Defect Policy (Stage 9).

---

### Stage 11: Split-Aware Remediation Engine
- **Python File:** `datatrust/remediator.py`
- **Class / Function:** `DataRemediator.fit_and_transform_splits()`
- **Input:** `train_df: pd.DataFrame`, `test_df: pd.DataFrame`, `task: TaskType`, `action_ids: List[str]`
- **Output:** `rem_train: pd.DataFrame`, `rem_test: pd.DataFrame`, `execution_log: List[str]`, `fitted_params: Dict[str, Any]`
- **Equations & Rules:**
  - **Strict Train/Test Isolation:**
    1. Pre-remediation 70% train / 30% test split.
    2. Fits all parameters (median imputation statistics, 1st/99th percentile Winsorization bounds, dropped leakage columns, class resampling) **solely on `train_df`**.
    3. Applies deterministic fitted bounds to `test_df`.
    4. **Zero test resampling / deduplication / row deletion:**
       $$\text{SHA-256}(\text{test\_indices}_{\text{before}}) == \text{SHA-256}(\text{test\_indices}_{\text{after}})$$
- **Dependencies:** Final Fitness (Stage 10).

---

### Stage 12: Downstream Empirical Model Validation
- **Python File:** `datatrust/validator.py`
- **Class / Function:** `DownstreamValidator.train_and_evaluate_split()`, `DownstreamValidator.predict_performance()`
- **Input:** `train_df`, `test_df`, `task: TaskType`, `target_column`, `time_column`
- **Output:** Observed downstream metric $y_{\text{obs}}$ (RMSE for regression/time-series, Macro-F1 for classification), model configuration, hyperparameters.
- **Surrogate Prediction Function $\hat{y}(F)$:**
  - *Regression:* $\hat{y}_{\text{RMSE}}(F) = \sigma_y \cdot (0.20 + (100 - F) \cdot 0.015)$
  - *Classification:* $\hat{y}_{\text{F1}}(F) = (F / 100)^{1.15} \cdot 0.94$
- **Dependencies:** Remediation (Stage 11).

---

### Stage 13: Closed-Loop Performance Calibration
- **Python File:** `datatrust/calibration.py`
- **Class / Function:** `PerformanceCalibrationEngine.calibrate()`
- **Input:** `initial_weights: W_AHP`, `initial_trust_score: F`, `observed_performance: y_obs`, `dimension_scores: S_i`
- **Output:** `CalibrationResult` (`weights_calibrated: W_calibrated`, `calibrated_trust_score`, `residual_before`, `residual_after`, `calibration_status`)
- **Loss Function & Optimization:**
  - **Objective:**
    $$L(w) = |\hat{y}(F(w)) - y_{\text{obs}}|$$
  - **Surrogate Gradient:**
    $$g_i = \frac{\partial L}{\partial w_i} = \text{sign}(\hat{y} - y_{\text{obs}}) \cdot \frac{\partial \hat{y}}{\partial F} \cdot (S_i - \bar{S}) \cdot M(T, Q_i)$$
  - **Line Search over $\eta \in [0.001, 3.0]$ with Simplex Projection:**
    $$w_i^{(t+1)} = \text{Normalize}\left(\text{Clip}(w_i^{(t)} - \eta g_i, 0.02, 0.60)\right)$$
  - **Strict Acceptance Policy:** Update accepted **only if** $\min_\eta L_{\text{after}} < L_{\text{before}} - 10^{-4}$. If rejected, $W_{\text{AHP}}$ is strictly preserved.
- **Dependencies:** Downstream Validation (Stage 12).

---

### Stage 14: Marginal Remediation Utility (MRU) Optimization
- **Python File:** `datatrust/optimizer.py`
- **Class / Function:** `RemediationOptimizer.optimize_remediations()`, `RemediationOptimizer.generate_what_if_scenarios()`
- **Input:** `task: TaskType`, `dimensions: Dict[QualityDimension, DimensionResult]`, `current_trust_score: float`, `is_vetoed: bool`
- **Output:** `List[RemediationActionV2]`, `List[WhatIfScenario]`, `counterfactual_table`
- **Equations & Cost Model:**
  - **MRU Definition:**
    $$\text{MRU}_i = \frac{\Delta F_i}{\text{Cost}_i}$$
  - **Discrete Cost Tiers:**
    - $\text{Cost}=1$: Column drop ($O(1)$)
    - $\text{Cost}=2$: Winsorization, Median imputation, Timestamp sorting ($O(N)$ / $O(N \log N)$)
    - $\text{Cost}=3$: SMOTE / Class re-weighting ($O(N)$)
    - $\text{Cost}=4$: Iterative MICE imputation ($O(N \cdot P \cdot \text{iter})$)
    - $\text{Cost}=5$: Manual relabeling / human-in-the-loop review
- **Dependencies:** Final Fitness (Stage 10).

---

## The Remediation Inconsistency (Documented Reality)

In accordance with scientific integrity standards, the following divergence between synthetic simulation and out-of-sample empirical reality is formally recorded:

```
[Dataset C Baseline: Scrambled Timestamps + Duplicates]
         │
         ├──> Intelligent Veto Engine ───> Final Fitness = 0.0 (Status: UNSAFE, RED)
         │
         ├──> MRU What-If Simulator ────> Predicts Simulated Fitness = 99.0
         │                                (Assumes complete removal of temporal veto)
         │
         └──> Actual fit_and_transform_splits()
                   │
                   ├── TRAIN: Sorted & Deduplicated (9 duplicates removed)
                   │
                   └── TEST: IMMUTABILITY MANDATE PRESERVES DUPLICATES & INDICES
                             (0 test rows dropped, original order intact)
                                   │
                                   ▼
                   remediated_full_df = pd.concat([rem_train, rem_test])
                   STILL CONTAINS TEST DUPLICATES & CONCAT DISCONTINUITIES
                                   │
                                   ▼
                   evaluate_temporal_validity() triggers RED Hard Veto AGAIN
                                   │
                                   ▼
                   ACTUAL REPORTED AFTER-FITNESS = 0.0 (Not 99.0!)
```

### Empirical Downstream RMSE Deterioration:
When autoregressive models (e.g. lag-1 Random Forest) are trained on cleanly sorted chronological training data but evaluated on an untouched, out-of-order test set, lag features create mismatch between train and test representations. Depending on the test partition structure, test RMSE can deteriorate or fail to achieve predicted gains.
