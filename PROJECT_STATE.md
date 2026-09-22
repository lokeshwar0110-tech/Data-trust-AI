# DataTrust AI v3 — Current Project State & Research Snapshot

**Project**: DataTrust AI (MSc Computational Statistics & Data Analytics)  
**Current Phase**: Final Methodology-Hardened & Research-Validated Engine (Release-Grade)  
**Snapshot Date**: 2026-09-14  
**Repository Path**: `C:\Users\Lenovo\.gemini\antigravity\scratch\datatrust-ai`  
**License**: MIT / Research Academic License  
**Active Production Endpoint**: `http://localhost:8000` (FastAPI daemon)  

---

## 1. Project Overview & Research Mission

DataTrust AI is a principled, task-conditioned computational framework that replaces static, context-blind data quality profiling with dynamic fitness evaluation tailored to specific downstream analytical tasks (Supervised Classification, Supervised Regression, Time-Series Forecasting, Clustering). 

The platform implements:
1. **General Autonomous Task Inference**: Multi-signal evidence evaluation distinguishing transactional tables with date features from genuine temporal forecasting sequences via timestamp uniqueness ratio ($N_{\text{unique}}/N$), cadence regularity, and entity grouping. Exposes empirical task probabilities $P(T \mid D)$, winning task, confidence, and authoritative manual override.
2. **Canonical 8D Quality Engine**: Completeness (CMP), Validity (VAL), Consistency (CNS), Uniqueness (UNQ), Timeliness (TIM), Outlier/Anomaly (OUT), Distribution Balance (BAL), and Coverage/Support (COV) bounded strictly in $[0.0, 100.0]$ with explicit provenance metadata.
3. **Disambiguated Duplicate-Timestamp Metrics**: Distinguishes `duplicate_timestamp_groups` (distinct collided dates), `duplicate_timestamp_rows` (total surplus observations), and `rows_removed_during_remediation`.
4. **Task-Conditioned Sensitivity Prior $M(T, Q)$ & AHP Weighting**: Formal sensitivity prior mapped into 8×8 reciprocal Analytic Hierarchy Process (AHP) matrices ($CR < 0.10$, $RI = 1.41$).
5. **Harmonized Non-Compensatory Defect Policy**:
   - **RED Hard Veto**: Fatal defects (temporal sequence disorder in TS, extreme class collapse in classification, >70% null density, target leakage) enforce $F_{\text{final}} = 0.0$, `status = "UNSAFE"`, `is_unsafe = True`.
   - **YELLOW Warning Cap**: Non-fatal severe defects (outliers, collinearity, moderate imbalance) enforce $F_{\text{final}} \le C_{\text{cap}} \in [60.0, 70.0]$, `status = "WARNING"`, `is_unsafe = False`.
   - **GREEN Normal**: Unconstrained fitness $F_{\text{final}} = F_{\text{raw}}$, `status = "NORMAL"`.
6. **Strict Train/Test Holdout Immutability**: Transformations fit exclusively on training data; holdout test set is strictly immutable (cryptographically verified via SHA-256 index hashing).
7. **Counterfactual MRU Optimizer**: Evaluates $\text{MRU} = \Delta \text{Fitness} / \text{Cost}$ strictly as simulated counterfactual predicted fitness gains, structurally discounted by irreducible holdout test defects.
8. **Adaptive Calibration & Hardened Rejection**: Constrained gradient descent updating weights only upon verifiable error reduction ($L_{\text{after}} < L_{\text{before}}$); preserves priors and tags provenance (`EMPIRICALLY_CALIBRATED`, `REJECTED`, `NO_IMPROVEMENT`).
9. **Empirical Validation**: Validated across 9 cohorts derived from 3 real-world datasets, improving Spearman correlation from $\rho = 0.2000$ to $\rho = 0.7167$ ($p = 0.0298 < 0.05$, $CR = 0.0000$).
10. **Independent Generalization Framework**: Reusable multi-domain evaluation framework (`benchmarks/run_independent_validation.py`) tested across 6 diverse real-world datasets.

---

## 2. Completed Architecture Components

| Subsystem / Module | Implementation File | Status | Provenance / Methodology |
| :--- | :--- | :---: | :--- |
| **Canonical Models** | `datatrust/models.py` | Complete ✓ | 8 Canonical Dimensions, Codes, RemediationDecision, BeforeAfterValidationResult, CalibrationResult |
| **Data Profiler** | `datatrust/profiler.py` | Complete ✓ | Heuristic & descriptive column profiling |
| **Task Inference Engine** | `datatrust/task_infer.py` | Complete ✓ | General evidence-based inference distinguishing transactional dates vs time series; $P(T \mid D)$ |
| **8 Canonical Dimensions** | `datatrust/quality_engine.py` | Complete ✓ | CMP, VAL, CNS, UNQ, TIM, OUT, BAL, COV; exposes `duplicate_timestamp_groups` and `rows` |
| **Sensitivity Matrix $M(T,Q)$** | `datatrust/sensitivity.py` | Complete ✓ | 4×8 Literature/Expert Prior matrix $[0.2, 1.0]$ |
| **AHP Weighting Engine** | `datatrust/weighting.py` | Complete ✓ | 8×8 reciprocal pairwise comparison solver ($CR < 0.10$, $RI = 1.41$); empirical calibration |
| **Intelligent Veto Engine** | `datatrust/veto_engine.py` | Complete ✓ | Three-level non-compensatory circuit breaker (RED Hard Veto $F=0.0$, YELLOW Cap, GREEN Normal) |
| **Data Remediator** | `datatrust/remediator.py` | Complete ✓ | Action-Authoritative, 70/30 split, train-only fit, test set immutability |
| **Downstream Validator** | `datatrust/validator.py` | Complete ✓ | Empirical model evaluation on held-out test splits |
| **Performance Calibration** | `datatrust/calibration.py` | Complete ✓ | Constrained gradient descent with strict rejection rule and provenance tracking |
| **Remediation Optimizer** | `datatrust/optimizer.py` | Complete ✓ | Marginal Remediation Utility (MRU) counterfactual simulation engine |
| **Fitness Uncertainty** | `datatrust/uncertainty.py` | Complete ✓ | Multi-source standard error: $\sigma^2_{\text{composite}} = \sum \sigma^2_i$, $SE = \sqrt{\sigma^2_{\text{composite}}}$ |
| **Reproducibility Engine** | `datatrust/reproducibility.py` | Complete ✓ | SHA-256 audit receipts with canonical integer index serialization |
| **DataTrust Engine Facade** | `datatrust/engine.py` | Complete ✓ | Unified end-to-end evaluation & remediation pipeline orchestrator |
| **REST API Server** | `backend/main.py` | Complete ✓ | FastAPI production server exposing evaluated routes |
| **Interactive UI** | `backend/static/index.html` | Complete ✓ | Synchronized semantics: Counterfactual MRU, Raw vs Capped vs Final Fitness, $P(T \mid D)$ |

---

## 3. Test Suite Verification

- **Command**: `python -m pytest tests/ -v -W default`
- **Total Tests**: **102 passed, 0 failed**
- **Execution Time**: ~20.48 seconds
- **Project-Internal Warnings**: **0** (only 6 third-party warnings from Starlette/Scikit-learn/Numpy)
- **Test Modules**:
  - `tests/test_time_series_validation.py` (9 tests): Verifies temporal split strict chronological ordering, zero test target leakage, valid prediction generation, RMSE computation, constant target edge case, non-constant series behavior, small time series handling, irregular cadence handling, and Melbourne daily minimum temperatures benchmark.
  - `tests/test_ingestion_generalization.py` (16 tests): Verifies robust ingestion across tabular datasets, datetime rejection for continuous float measurements (e.g. `SepalWidthCm`), candidate target discovery (`Species`), genuine time series, transactional date features, unlabeled clustering, manual target/task selection, and missing target state handling.
  - `tests/test_defect_policy.py` (8 tests): Verifies RED hard veto ($F=0.0$, UNSAFE), YELLOW warning caps ($F \le C_{\text{cap}}$, WARNING), GREEN normal operation, and duplicate timestamp groups vs surplus rows.
  - `tests/test_calibration.py` (8 tests): Verifies performance prediction monotonicity, residual reduction, rejection policy, empirical AHP calibration on real datasets, provenance tags, and prior weight preservation.
  - `tests/test_task_infer.py` (10 tests): Verifies genuine time series, transactional tabular regression with dates, classification with dates, clustering bounds, distribution properties, and manual overrides.
  - `tests/test_v3_stabilization.py` (5 tests): Verifies canonical index hashing, action-authoritative execution, action registry metadata, remediation decisions, and 8 canonical dimensions.
  - `tests/test_phase1.py` (12 tests): Verifies all Phase 1 research conditions and end-to-end scoring.
  - `tests/test_validator.py` (6 tests): Verifies downstream model evaluation, what-if simulation, and test immutability.
  - `tests/test_weighting.py` (4 tests): Verifies AHP reciprocal properties, consistency ratio $CR < 0.10$, and veto rules.
  - `tests/test_quality.py` (4 tests): Verifies individual dimension evaluations and configurable imbalance policies.
  - `tests/test_real_datasets.py` (3 tests): Verifies real-world datasets (Wine, House Sales, Air Quality).
  - `tests/test_research_layers.py` (3 tests): Verifies uncertainty formulas, audit receipts, and native RMSE calibration.
  - `tests/test_api.py` (9 tests): Verifies REST endpoints, auto-detect, CSV upload, remediation, and preset scenarios.
  - `tests/test_optimizer.py` (3 tests): Verifies MRU ranking, what-if simulation, and irreducible test defect accounting.
  - `tests/test_engine.py` (1 test): Verifies end-to-end pipeline execution and PDF generation.

---

## 4. Empirical Benchmark Status

All benchmark suites execute dynamically without hardcoded stand-ins:
1. `benchmarks/run_benchmarks.py`: 10-seed multi-seed evaluation. DataTrust Spearman $\rho = 0.793 \pm 0.244$ vs Static $\rho = -0.272 \pm 0.544$ (paired t-test $t = 7.734, p = 2.8975 \times 10^{-5} < 0.05$).
2. `benchmarks/run_experiments_v2.py`: Experiments A, B, C, D all empirically validated.
3. `benchmarks/ablation_study.py`: Comprehensive 7-experiment architectural ablation study across 4 datasets. Demonstrates RED Hard Veto ($F=0.0$, `status = "UNSAFE"`) on Dataset C (temporal scramble/collisions) and Dataset D (severe class collapse < 2%). Crucially verifies Dataset D's independent structural risk assessment ($F=0.0$ despite high downstream Macro $F_1 = 0.98$).
4. `benchmarks/run_sensitivity_analysis.py`: Dynamic partition sweep ($w_{\text{train}} \in [0.50, 0.85]$) verifying post-remediation fitness within $\pm 10$ pt tolerance and zero directional contradictions; outputs `results/research/sensitivity_results.json`.
5. `benchmarks/run_real_ahp_calibration.py`: Empirical calibration on 9 cohorts from 3 real datasets; improves Spearman correlation from $\rho = 0.2000$ to $\rho = 0.7167$ ($p = 0.0298 < 0.05$), $CR = 0.0000$.
6. `benchmarks/run_independent_validation.py`: Multi-domain validation framework across 6 real datasets (UCI Wine, Air Quality, California Housing, Diabetes, Iris, Retail Store Transactions); outputs `results/research/independent_validation_results.json`.
7. `benchmarks/test_product_sales_region.py`: Three-mode comparative evaluation demonstrating autonomous transactional inference avoiding false veto, while manual time-series forcing triggers RED Hard Veto ($F=0.0$, UNSAFE).
8. `benchmarks/check_research_integrity.py`: Automated research integrity check verifying 100% of the 11 core scientific invariants.

---

## 5. Deployment & Runtime Verification
- **Host**: `0.0.0.0:8000`
- **Daemon Status**: Live (Uvicorn FastAPI daemon running)
- **UI Verification**: Dashboard displays Raw vs Capped vs Final Fitness, Counterfactual MRU Predicted Gains, and Closed-Loop Remediation Decision badges.
- **Reproducibility**: Canonical JSON integer index serialization guarantees exact cryptographic hashing across all platforms.

