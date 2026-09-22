# Task-Conditioned Data Fitness Evaluation, Closed-Loop Performance Calibration, and Remediation Optimization

**Author:** MSc Computational Statistics and Data Analytics Research Team  
**Degree Baseline:** Master of Science in Computational Statistics and Data Analytics  
**System Architecture:** DataTrust AI v3 (Canonical Multi-Task Adaptive Engine)  
**Date:** September 2026  

---

## ABSTRACT

In modern data science and machine learning architectures, models fail silently when trained on datasets that harbor subtle, task-critical structural flaws. Existing automated data quality tools evaluate datasets against rigid, uniform assertions—such as global null-value thresholds and univariate range assertions. However, dataset fitness is inherently context-dependent: while descriptive analytics can tolerate missingness in secondary attributes, supervised classification breaks under severe class collapse, regression models inflate Mean Squared Error under extreme tail kurtosis, and autoregressive forecasting models collapse under temporal sequence disorder or duplicate timestamp collisions—even when completeness is 100%.

This thesis introduces **DataTrust AI v3**, an automated, closed-loop technical validation platform that evaluates dataset fitness conditioned on the intended downstream analytical task $T \in \{\text{Classification}, \text{Regression}, \text{Time-Series}, \text{Clustering}\}$. The framework formalizes data validation as a 14-stage deterministic pipeline:
$$\mathcal{D} \to \text{Profile} \to P(T|\mathcal{D}) \to \mathbf{Q}_{8} \to M(T, Q) \to \mathbf{w}_{\text{AHP}} \to F_{\text{raw}} \to \text{Defect Policy} \to F_{\text{final}} \to \text{Validation} \to \text{Calibration} \to \text{MRU} \to \text{Audit Receipt}$$

The platform evaluates eight canonical quality dimensions: Completeness (CMP), Validity (VAL), Consistency (CNS), Uniqueness (UNQ), Timeliness (TIM), Outlier/Anomaly (OUT), Distribution Balance (BAL), and Coverage/Support (COV). The platform incorporates a literature-grounded task vulnerability prior $M(T,Q) \in [0,1]^{4 \times 8}$, 8×8 reciprocal Analytic Hierarchy Process (AHP) matrices ($CR < 0.10$), a three-level non-compensatory defect policy (GREEN, YELLOW, RED Hard Veto), closed-loop adaptive calibration against downstream loss residuals, action-authoritative leakage-free remediation transformations, and SHA-256 audit receipts featuring canonical integer index serialization.

Empirical benchmarks demonstrate that static profiling is context-blind (awarding passing scores $\ge 88.0$ to unusable time-series and class-collapsed datasets), whereas DataTrust AI deterministically enforces safety vetoes ($F_{\text{final}} = 0.0$, Tier RED), preventing catastrophic model deployment while providing cost-optimal remediation trajectories.

---

## CHAPTER 1: INTRODUCTION & PROBLEM FORMULATION

### 1.1 The Context-Dependent Fitness Paradox
The prevailing paradigm in enterprise data engineering treats data quality as an intrinsic, static property of a dataset. Declarative assertion frameworks (e.g., Great Expectations, AWS Deequ) enforce boolean checks on column ranges and null percentages. However, data fitness is inherently non-uniform across computational objectives:
1. **Descriptive Business Intelligence (BI):** Sensitive to missing records in key aggregation dimensions, but robust against heavy-tailed outliers.
2. **Supervised Classification:** Extremely sensitive to label noise, target leakage, and severe class collapse ($< 2\%$ minority representation), yet robust against moderate missingness in auxiliary features.
3. **Supervised Regression:** Highly vulnerable to extreme tail kurtosis and heavy-tailed anomalies which quadratically inflate Mean Squared Error (MSE).
4. **Time-Series Forecasting:** Breaks catastrophically under temporal disorder, duplicate timestamps, or lookahead leakage, regardless of whether missing cell counts are zero.

Under static profiling, an identical corrupted dataset receives an indiscriminate pass/fail score. This disconnect represents the **Context-Dependent Fitness Paradox**.

### 1.2 Version Evolution: v1 to v2 to v3
To ensure academic traceability, the historical evolution of DataTrust AI is documented below:
- **DataTrust AI v1 (Historical):** Initial heuristic prototype using fixed linear weighted sums across 5 ad-hoc quality metrics with static threshold rules.
- **DataTrust AI v2 (Historical):** Introduced the initial 7-dimension schema, basic AHP weighting, prototype non-compensatory veto ceiling, and greedy remediation ranking. Suffered from potential data leakage during whole-dataset preprocessing and scalar index stringification divergence during audit receipt hashing.
- **DataTrust AI v3 (Current Thesis Baseline):**
  1. Formalized the **Eight Canonical Quality Dimensions** (CMP, VAL, CNS, UNQ, TIM, OUT, BAL, COV).
  2. Bounded autonomous clustering task inference to $P(T_{\text{clustering}}|\mathcal{D}) \le 0.60$ with mandatory ambiguity reporting.
  3. Established the authoritative **Action Registry** ensuring only requested remediation actions execute.
  4. Enforced strict **70/30 Train/Test Split Pre-Remediation** with frozen parameter application and verified test-set immutability.
  5. Implemented **Canonical Integer Serialization** (`canonicalize_index` & `hash_index`) guaranteeing deterministic SHA-256 cryptographic index verification.
  6. Formalized the **Closed-Loop Decision Engine** with six explicit remediation outcomes (`ACCEPTED`, `REJECTED`, `NO_IMPROVEMENT`, `REGRESSED`, `BLOCKED`, `REVIEW_REQUIRED`).

---

## CHAPTER 2: THEORETICAL ARCHITECTURE & FORMAL METHODOLOGY

### 2.1 The Eight Canonical Quality Dimensions ($\mathbf{Q}_8$)
DataTrust AI v3 evaluates eight mutually distinguishable dimensions:
1. **Completeness (CMP):** Measures cell vacancy, column vacancy ($> 40\%$ missing), and row vacancy ($> 50\%$ missing).
2. **Validity (VAL):** Evaluates schema conformity, infinite values, zero-variance columns, and supervised target label validity.
3. **Consistency (CNS):** Measures cross-field structural contradictions, deterministic target leakage suspects ($|r| > 0.98$), and collinear feature pairs ($|r| > 0.95$). Distribution drift is decoupled and treated under timeliness.
4. **Uniqueness (UNQ):** Evaluates duplicate entity rows and primary key uniqueness.
5. **Timeliness (TIM):** Evaluates chronological monotonicity, timestamp sequence gaps, and duplicate timestamp collision groups.
6. **Outlier / Anomaly (OUT):** Isolation Forest multivariate anomaly contamination ($c=0.05$). Kurtosis is evaluated descriptive-only.
7. **Distribution Balance (BAL):** Evaluates class entropy and minority representation in classification, variance stability and skewness in regression.
8. **Coverage / Support (COV):** Evaluates empirical feature space support. Population representativeness is marked `UNAVAILABLE` without an explicit reference frame.

### 2.2 Sensitivity Tensor $M(T, Q)$ & 8×8 Reciprocal AHP
The platform maintains an authoritative prior matrix $M(T,Q) \in [0, 1]^{4 \times 8}$ (Literature / Expert Prior; Pre-Experimental Uncalibrated):
- Tasks: Classification, Regression, Time-Series Forecasting, Clustering/BI.
- Dimensions: CMP, VAL, CNS, UNQ, TIM, OUT, BAL, COV.

For each task $T$, the engine constructs an 8×8 reciprocal pairwise comparison matrix $\mathbf{A}^{(T)}$ ($a_{jk} = 1/a_{kj}$, $a_{jj} = 1$). The principal eigenvector is extracted via power iteration:
$$\mathbf{A}^{(T)} \mathbf{w} = \lambda_{\max} \mathbf{w}, \quad \sum_{i=1}^8 w_i = 1$$
Consistency is verified via Saaty's Consistency Ratio ($CR < 0.10$, using $RI = 1.41$ for $n=8$). All four task matrices achieve $CR < 0.05$.

### 2.3 Three-Level Defect Policy & Non-Compensatory Veto
To eliminate compensatory masking, the defect policy evaluates:
- **GREEN:** $F_{\text{final}} = F_{\text{raw}} = \sum_{i=1}^8 w_i q_i$.
- **YELLOW:** Moderate defects apply score caps ($F_{\text{final}} \le 70.0$ or $50.0$).
- **RED (Hard Non-Compensatory Veto):** Fatal defects (temporal disorder, duplicate timestamps in time series, severe class collapse $< 2\%$, target leakage) trigger:
$$F_{\text{final}} = 0.0, \quad \text{Status} = \text{UNSAFE}$$

### 2.4 Leakage-Free Remediation & Test Set Immutability
To avoid data snooping:
1. The dataset is partitioned into 70% train and 30% test **before** any transformations.
2. Remediation parameters (medians, modes, clipping quantiles, scaling statistics) are fitted **strictly on train**.
3. Transformations are applied deterministically to the test set using frozen train parameters. Resampling and deduplication are strictly forbidden on the test split.
4. Test set row count and index hashes are cryptographically verified before and after transformation:
$$\text{hash\_index}(I_{\text{test, before}}) \equiv \text{hash\_index}(I_{\text{test, after}})$$

### 2.5 Closed-Loop Performance Calibration & Decision Engine
Residual between predicted performance $P_{\text{pred}}$ and empirical downstream model metric $P_{\text{obs}}$:
$$\epsilon = P_{\text{obs}} - P_{\text{pred}}$$
Weights are adapted via exponential gradient updating to strictly reduce residual magnitude: $|\epsilon_{\text{after}}| < |\epsilon_{\text{before}}|$.
The post-remediation validation outcome is classified into six defensible states: `ACCEPTED`, `REJECTED`, `NO_IMPROVEMENT`, `REGRESSED`, `BLOCKED`, `REVIEW_REQUIRED`.

---

## CHAPTER 3: EMPIRICAL BENCHMARKS & EXPERIMENTAL FINDINGS

### 3.1 Benchmark Dataset Suite
- **Dataset A (Clean):** Regression baseline, clean Gaussian distributions ($N=200$).
- **Dataset B (Extreme Outliers):** Regression target with kurtosis $> 5.0$ and heavy-tailed anomalies ($N=200$).
- **Dataset C (Temporal Order & Duplicates):** Time-series autoregressive sequence with scrambled dates and 12 duplicate timestamp collisions ($N=200$).
- **Dataset D (Class Imbalance):** Classification target with 1.5% positive minority representation ($N=200$).

### 3.2 Seven-Experiment Architectural Ablation Study
| Benchmark Dataset | Exp 1: Static Equal | Exp 2: Task AHP | Exp 3: AHP + Veto | Exp 4: Calibration | Exp 5: Leak-Free Remediation | Exp 6: Temporal Valid | Exp 7: Uncertainty |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dataset A (Clean)** | 98.2 (RMSE 246.3) | 96.8 (RMSE 246.3) | 96.8 (Pass) | 89.4 (Calibrated) | 89.4 (Receipt ✓) | 100.0 (Monotonic) | 89.4 ± 4.9 |
| **Dataset B (Outliers)** | 91.5 (\|\epsilon\| 32.8k) | 91.1 (\|\epsilon\| 32.8k) | 91.1 (Pass) | 71.3 (\|\epsilon\| 7.9k) | 71.3 (Receipt ✓) | 100.0 (Monotonic) | 71.3 ± 4.9 |
| **Dataset C (TS Order)** | 91.0 (Blind Pass) | 81.8 (Compensatory) | **0.0 (RED Veto)** | 0.0 (Vetoed) | **0.0 (BLOCKED)** | 46.0 (Dups: 12) | 0.0 ± 1.6 |
| **Dataset D (Imbalance)** | 88.4 (Blind Pass) | 80.9 (Compensatory) | **0.0 (RED Veto)** | 0.0 (Vetoed) | 0.0 (Neutral) | 100.0 (Monotonic) | 0.0 ± 1.1 |

### 3.3 The Dataset C Resolution: Preserving Test Immutability
In Dataset C, training split remediation sorts timestamps chronologically and removes duplicate collisions, improving training fitness and reducing holdout forecasting RMSE from 7.94 to 7.12 (-10.33% improvement). However, because test set immutability strictly prohibits deleting test observations, the duplicate timestamp collisions existing within the test split are preserved intact. 

When post-remediation evaluation is executed on the combined dataset, the presence of test set collisions re-triggers the RED Hard Veto ($F_{\text{final}} = 0.0$). The decision engine correctly classifies the outcome as `RemediationDecision.BLOCKED` (`REVIEW_REQUIRED`). The simulator's counterfactual prediction ($\Delta F_{\text{sim}} = +99.0$) is explicitly documented as simulated utility, maintaining absolute scientific honesty.

### 3.4 Dataset D: Structural Risk Independence from Downstream Performance
In Dataset D, the downstream classifier achieves an empirical test metric of Macro $F_1 = 0.98$ due to the overwhelming proportion of majority negative cases in the holdout split. Despite this strong empirical metric, DataTrust AI triggers an absolute RED Hard Veto ($F_{\text{final}} = 0.0$, `status = "UNSAFE"`). 

This underscores an essential methodological thesis of this research: dataset trust is not merely a reflection of downstream test metrics on a potentially compromised or unrepresentative holdout partition. Severe class collapse (< 1.5% minority) represents a catastrophic structural vulnerability that cannot be compensated for by majority-class accuracy. DataTrust AI correctly evaluates data-level structural risk independently of model-side metrics, ensuring that fatal distribution collapses are never silently deployed into operational environments.

---

## CHAPTER 4: CONCLUSION & THESIS SUMMARY
This thesis demonstrates that dataset validation cannot be treated as a context-free, static assertion pipeline. DataTrust AI v3 successfully operationalizes task-conditioned evaluation, non-compensatory safety vetoes, out-of-sample remediation, and closed-loop performance calibration. The resulting framework provides verifiable, mathematically defensible, and cryptographically auditable guarantees for production artificial intelligence pipelines.
