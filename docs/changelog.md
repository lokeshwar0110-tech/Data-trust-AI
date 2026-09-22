# DataTrust AI — Scientific & Technical Changelog

All notable changes, architectural evolutions, scientific integrity audits, and empirical milestones for DataTrust AI are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [v3.1.1] — 2026-09-11 (Live Deployment & Architectural Freeze)

### Added
- **Production FastAPI REST API (`backend/main.py`)**:
  - `GET /health`: Comprehensive system status, active version, test status (50/50 passed), and frozen status.
  - `POST /evaluate`: Full task-aware evaluation returning dimension scores, AHP weights, veto flags, fitness interval ($\mu \pm MoE$), confidence tier, and audit receipt.
  - `POST /infer-task`: Structural task inference returning top hypothesis, confidence, heuristic evidence signals, and candidate list.
  - `POST /remediate`: Out-of-sample remediation executing actions under strict train/test isolation, returning before/after metrics and cryptographic receipt.
  - `POST /calibrate`: Closed-loop adaptive calibration updating weights $W_{calibrated}$ via constrained gradient descent.
  - `POST /optimize-mru`: Marginal Remediation Utility knapsack optimization under budget constraints.
- **Docker Containerization**:
  - Multi-stage `Dockerfile` with Python 3.12-slim base image.
  - `docker-compose.yml` defining production container service with healthcheck.
  - `.dockerignore` excluding caches, scratch directories, logs, and venvs.
  - Deployment guide in `docs/deployment.md`.
- **Verified Live Daemon Service**: Background Uvicorn daemon running on port 8000 with 100% verified HTTP 200 responses across all endpoints.

---

## [v3.1.0] — 2026-09-10 (Phase 1 Implementation: 12 Research Conditions Enforced)

### Added
- **New Test Suite (`tests/test_phase1.py`)**: 14 rigorous unit and integration tests asserting all 12 Phase 1 scientific conditions:
  - `test_task_inference_returns_evidence`: Verifies non-definitive task inference with exposed heuristic evidence signals.
  - `test_task_override`: Verifies manual override preserves user intent over structural inference.
  - `test_eight_dimensions_present`: Confirms all 8 dimensions are evaluated for every dataset.
  - `test_consistency_internal_contradictions`: Validates Consistency as structural contradiction rather than distribution drift.
  - `test_outlier_zscore_iqr`: Confirms outlier detection uses robust $z$-score and IQR with kurtosis as secondary.
  - `test_temporal_structural_vs_leakage`: Proves separation of structural ordering from feature-target leakage.
  - `test_imbalance_per_task`: Confirms class imbalance acts as a hard veto for Classification and a pass-through for Regression.
  - `test_relevance_target_missing`: Validates target absence triggers uncomputable score and zero fitness.
  - `test_ahp_sensitivity_normalized`: Verifies $A_T w = \lambda_{\max} w$, $\sum w_i = 1$, and $CR < 0.10$.
  - `test_three_level_veto_logic`: Validates non-compensatory bounds (Green: full, Yellow: cap 70, Red: floor 0).
  - `test_confidence_tiers`: Verifies tier thresholds (High $\ge 85$, Moderate $[70, 85)$, Low $[50, 70)$, Critical $< 50$).
  - `test_audit_receipt_hash`: Verifies SHA-256 provenance hash immutability across evaluation.
  - `test_end_to_end_phase1_pipeline`: Complete integration pipeline validation from raw data to frozen audit receipt.
  - `test_empty_dataframe_error`: Defensive boundary validation on empty inputs.
- **Expanded Test Suite**: Baseline tests increased from 36 to **50 passed, 0 failed** in 6.73 seconds.

### Changed
- **Task Inference Refinement (`datatrust/task_infer.py`)**:
  - Re-classified task inference as heuristic hypothesis generation with confidence score and evidence dictionary.
  - Clustering hypothesis restricted: structural heuristics only assign clustering when target is absent and tabular structure supports unsupervised analysis; flagged with explicit caveat.
- **Consistency vs Drift Separation (`datatrust/quality_engine.py`)**:
  - Redefined `Consistency` to measure internally contradictory records (e.g., end date preceding start date, impossible category combinations).
  - Relocated distribution drift monitoring to `Timeliness / Freshness`.
- **Outlier Mathematics Refinement (`datatrust/quality_engine.py`)**:
  - Replaced raw kurtosis outlier scoring with robust IQR and modified $z$-score detection; kurtosis preserved exclusively as descriptive distributional tail indicator.
- **Temporal Structural Verification vs Leakage Separation**:
  - `Temporal Validity` strictly measures monotonicity, cadence jitter, and duplicate timestamps.
  - Feature-target leakage isolated to `Feature Relevance`.

---

## [v3.0.3] — 2026-09-09 (Scientific Consistency Patch — Audit 3)

### Fixed
- **Uncertainty Formula Correction**:
  - Corrected legacy notation stating $\sigma^2_{\text{comp}} = \sum \sigma^2_i$ and $SE = \sigma^2$.
  - Enforced correct KaTeX and Python formulation:
    $$\sigma^2_{\text{comp}} = \sum_{i=1}^M \sigma^2_i \implies SE_{\text{comp}} = \sqrt{\sigma^2_{\text{comp}}} = \sqrt{\sum_{i=1}^M \sigma^2_i}$$
  - Enforced bounded confidence intervals: $CI = [\max(0, \mu - 1.96 \cdot SE), \min(100, \mu + 1.96 \cdot SE)]$.
- **Dataset C Baseline Unification**:
  - Unified disparate baseline references across ablation study and downstream validator to fixed $RMSE = 7.94$.
- **Dataset B Precision Synchronization**:
  - Unified Dataset B baseline metric references to $RMSE = 32,880.73$ and post-remediation to $5.38$ ($-99.98\%$ reduction).

---

## [v3.0.2] — 2026-09-08 (Downstream Decoupling & Provenance Patch — Audit 2)

### Changed
- **Weight Separation ($W_{AHP}$ vs $W_{calibrated}$)**:
  - Decoupled theoretical prior weights $W_{AHP}$ from empirically calibrated weights $W_{calibrated}$.
  - Calibrated weights require explicit convergence checks ($\Delta L > \epsilon_{tol}$) before acceptance; rejected calibrations fall back safely to $W_{AHP}$.
- **Surrogate vs Observed Downstream Metric Separation**:
  - Renamed heuristic downstream estimates to `surrogate_downstream_metric` with explicit uncertainty bounds.
  - Reserved `observed_model_metric` strictly for empirical model evaluations on held-out test splits.
- **MRU Gain Distinction**:
  - Renamed optimizer simulation outputs to `mru_predicted_gain` to prevent confusion with empirical post-remediation validation gains `observed_downstream_gain`.

### Added
- **Cryptographic Audit Receipts (`datatrust/reproducibility.py`)**:
  - Implemented SHA-256 digital hashing over dataset state, parameters, configuration, and evaluation report.
  - Added flags asserting `leakage_free_verified = True` and `test_indices_immutable = True`.

---

## [v3.0.1] — 2026-09-07 (Train/Test Isolation Audit — Audit 1)

### Fixed
- **Data Leakage in Remediation**:
  - Eliminated data leakage caused by cleaning datasets prior to train/test partitioning.
  - Restructured `DataRemediator` into `fit_and_transform_splits(X_tr, y_tr, X_te, y_te)`:
    - Estimators (imputers, scalers, outlier bounds) fit strictly on `X_tr`.
    - Transformations applied out-of-sample to `X_te`.
    - Test-set row membership and indices strictly immutable (zero resampling or deduplication on test partition).
- **AHP Consistency Ratio Enforcement**:
  - Enforced Saaty's consistency check: $CR = \frac{CI}{RI} < 0.10$.
  - Automated fallback to uniform weights if pairwise comparison matrix displays logical intransitivity.

---

## [v3.0.0] — 2026-09-05 (Research Architecture & Empirical Foundation)

### Added
- **Core Architecture Baseline**:
  - Full modular pipeline: Profiling, Task Inference, 8 Quality Dimensions, Sensitivity Mapping $M(T,Q)$, Fixed AHP Weighting, Intelligent Veto Engine, Data Remediation, Downstream Validation, Adaptive Calibration, MRU Knapsack Optimizer, and Multi-Source Uncertainty.
  - Comprehensive 7-experiment ablation benchmark (`benchmarks/ablation_study.py`).
  - 4-scenario task-divergence benchmark (`benchmarks/run_benchmarks.py`).
  - Initial 36-test suite verifying mathematical foundations.

---

## [v2.5.0] — 2026-08-20 (Remediation & Simulation Prototyping)

### Added
- Initial implementation of `DataRemediator` with heuristic rule-based transforms (mean imputation, Winsorization, z-score capping).
- Initial Marginal Remediation Utility greedy knapsack optimizer.

### Deprecated
- In-place dataframe modification during remediation.

---

## [v2.0.0] — 2026-07-15 (Task-Aware Fitness Scoring Prototype)

### Added
- Introduction of `TaskType` enum: Classification, Regression, Time-Series Forecasting, Clustering, Descriptive/BI.
- Task-Conditioned Sensitivity Matrix $M(T, Q)$ replacing static equal weights.
- Initial Analytical Hierarchy Process (AHP) pairwise matrix generator.

---

## [v1.0.0] — 2026-05-10 (Foundational Static Profiler)

### Added
- Initial release of task-agnostic static data quality profiler.
- Univariate metric calculations: missingness, type detection, unique counts, basic schema validation.
- Linear unweighted fitness scoring: $F = \frac{1}{M} \sum_{i=1}^M Q_i$.
