# DataTrust AI v3 — Chronological Research Log

This research log tracks the technical evolution, mathematical audits, scientific integrity corrections, and empirical milestones of DataTrust AI v3.

---

## 1. Research Motivation & Problem Statement

Standard data quality frameworks (e.g., Great Expectations, AWS Deequ, YData Profiling) compute heuristic, static quality scores based on generic completeness, column type conformance, and basic outlier counts. In machine learning and predictive analytics workflows, these static scores correlate poorly with true downstream model degradation:

$$\rho_{\text{static}}(S_{\text{static}}, \Delta\text{Metric}) \approx 0.31$$

A dataset with 10% non-critical feature missingness may be heavily penalized by static tools even though tree-based gradient boosters suffer zero performance loss. Conversely, a dataset with 100% completeness and pristine marginal distributions can contain chronological sequence inversions or target leakage ($|r| > 0.98$) that catastrophically break autoregressive forecasting or produce spurious test accuracy.

DataTrust AI was conceived to evaluate **Task-Conditioned Fitness for Purpose**:

$$F(D, T) \in [0, 100]$$

where quality dimensions are weighted dynamically according to task-specific vulnerability priors $M(T, Q)$, governed by non-compensatory critical defect vetoes, and calibrated against empirical downstream performance.

---

## 2. Architectural Evolution (v1 to v3)

### Phase 0: DataTrust v1 & v2 (Early Exploratory Baselines)
- **v1.0:** Basic heuristic weighting across 5 ad-hoc quality metrics. Demonstrated task divergence on small synthetic tables.
- **v2.0:** Introduced Analytic Hierarchy Process (AHP) with 7 dimensions (`missingness`, `outlier_resilience`, `drift_stability`, `label_integrity`, `leakage`, `cardinality_schema`, `temporal_validity`). Established initial binary veto rules for temporal disorder and extreme class imbalance.

### Phase 1: DataTrust v3 Architecture Redesign
- Standardized onto **Eight Canonical Quality Dimensions**:
  1. Completeness
  2. Validity
  3. Consistency
  4. Uniqueness
  5. Timeliness / Freshness
  6. Outlier / Anomaly Quality
  7. Distribution Balance
  8. Coverage / Representativeness
- Formalized the **$4 \times 8$ Task-Quality Sensitivity Matrix $M(T, Q)$** across four task profiles:
  - Supervised Classification
  - Supervised Regression
  - Time-Series Forecasting
  - Unsupervised Clustering
- Formulated **Four Independent $8 \times 8$ Reciprocal AHP Pairwise Comparison Matrices** solved via the principal eigenvector method with Saaty Random Index $\text{RI} = 1.41$ and Consistency Ratio $\text{CR} < 0.10$.
- Established the **Three-Tier Defect Policy**: GREEN (Normal), YELLOW (Warning/Cap), and RED (Hard Non-Compensatory Veto).

---

## 3. Scientific Integrity Audits & Micro-Patches

Prior to Phase 1 authorization, three rigorous scientific consistency audits were conducted:

### Audit Round 1: Test-Set Integrity & Notation
- **Enforcement:** Enforced pre-remediation 70/30 train/test splitting. Strictly prohibited any test-set row deletion, deduplication, SMOTE resampling, or synthetic imputation.
- **Assertion Added:**
  $$\text{SHA-256}(\text{test\_indices}_{\text{before}}) == \text{SHA-256}(\text{test\_indices}_{\text{after}})$$
- **Terminology:** Formally decoupled prior weights $W_{\text{AHP}}$ from post-calibration weights $W_{\text{calibrated}}$. Disentangled surrogate predictions $\hat{y}(F)$ from observed empirical model metrics $y_{\text{obs}}$.

### Audit Round 2: Mathematical Formulations & Numerical Precision
- **Uncertainty Correction:** Audited the multi-source uncertainty formula. Fixed notation error where composite standard error was previously equated to composite variance. Verified correct formulation:
  $$\sigma^2_{\text{composite}} = \sum_{i=1}^4 \sigma_i^2, \quad \text{SE}_{\text{composite}} = \sqrt{\sigma^2_{\text{composite}}}$$
  Clarified that empirical bootstrap dispersion $s_{\text{boot}}$ is not divided by $\sqrt{B}$ because it represents the dispersion of the fitness estimator under dataset perturbation.
- **Dataset B Precision:** Corrected exact percentage reporting for extreme outlier Winsorization on Dataset B:
  $$\text{RMSE: } 32880.73 \longrightarrow 5.38 \quad (\text{Exact: } -99.98\%, \text{ never rounded to } -100.0\%)$$
- **Dataset C Baseline Unification:** Unified baseline RMSE reporting across experiments to $7.94$ (out-of-sample split).

### Audit Round 3: Pre-Patent Scientific Consistency & Architectural Freeze
- Enforced strict line-search acceptance rule in adaptive calibration:
  $$\text{Accept } W_{\text{calibrated}} \iff \min_\eta L_{\text{after}} < L_{\text{before}} - 10^{-4}$$
  If no candidate learning rate reduces the loss, status is set to `"Calibration Rejected"` and $W_{\text{AHP}}$ is preserved.
- Integrated cryptographic `DownstreamAuditReceipt` containing SHA-256 hashes of train and test index vectors.

---

## 4. Phase 1 Implementation & The 12 Research Conditions

Phase 1 was formally authorized with 12 research constraints:

1. **Condition 1 (Task Inference):** Clustering hypothesis for unlabelled datasets bounded to $\le 0.60$ with mandatory ambiguity warning and manual user override support.
2. **Condition 2 (Consistency):** Defined strictly as structural contradictions, cross-field constraints, and collinearity ($|r| > 0.92$). Distribution drift is strictly assigned to Timeliness.
3. **Condition 3 (Outlier / Anomaly):** Primary score derived strictly from Isolation Forest multivariate anomaly density $c \in [0, 1]$. Kurtosis is reported as descriptive supporting evidence only.
4. **Condition 4 (Coverage / Support):** Observable feature space support ($N/P$) separated from population representativeness, which is marked as `"UNAVAILABLE"`.
5. **Condition 5 ($M(T,Q)$ Matrix):** $4 \times 8$ matrix loaded from config with formal provenance `LITERATURE / EXPERT PRIOR (PRE-EXPERIMENTAL UNCALIBRATED)`.
6. **Condition 6 (AHP Weighting):** Four independent $8 \times 8$ pairwise matrices with $\text{CR} < 0.10$ ($\text{RI} = 1.41$) and weights summing to 1.0.
7. **Condition 7 (Distribution Balance):** Handles classification class entropy and regression skewness; respects configurable `ImbalancePolicy`.
8. **Condition 8 (Three-Level Defect Policy):** GREEN (normal), YELLOW (warning cap at 60.0–70.0), RED (hard non-compensatory veto setting $F=0.0$, status="UNSAFE").
9. **Condition 9 (Provenance Taxonomy):** Formal provenance taxonomy assigned across every dimension, matrix, weight vector, and defect rule.
10. **Condition 10 (Strict Scope Boundary):** Preserved existing Phase 2/3 interfaces (block bootstrap, calibration line search, MRU knapsack) without modification.
11. **Condition 11 (Cryptographic Determinism):** Profiling metadata records deterministic dataset SHA-256 hash using `pd.util.hash_pandas_object`.
12. **Condition 12 (Test Verification):** Expanded test suite from 36 baseline tests to **50 passed tests (0 failures)** in 6.73s.

---

## 5. Empirical Discovery: The Remediation Inconsistency

During ablation testing on Dataset C (Temporal Order & Duplicates), an important empirical inconsistency was discovered and documented:

- **The Defect:** Scrambled timestamps and 12 duplicate records trigger the RED Hard Veto in `IntelligentVetoEngine` ($F=0.0$, status="UNSAFE").
- **The Simulation:** The MRU optimizer and What-If simulator evaluate actions assuming the defect is fully eliminated from the dataset, predicting $\Delta F = +99.0$ and simulated fitness $99.0$.
- **The Empirical Reality:** Under strict out-of-sample remediation (`fit_and_transform_splits`), the training set is sorted and deduplicated (9 duplicates removed). However, **test-set immutability strictly prohibits altering test row membership or deduplicating test rows**. When train and test sets are concatenated to form the remediated dataset for post-evaluation, the test split still contains temporal duplicates and ordering discontinuities.
- **The Outcome:** The post-remediation evaluation detects duplicate timestamps and re-triggers the RED Hard Veto. The actual reported after-fitness is **0.0**, in stark contrast to the simulated 99.0.
- **Research Implication:** This proves the vital necessity of empirical out-of-sample validation; heuristic simulation alone produces dangerously optimistic assumptions about data readiness.
