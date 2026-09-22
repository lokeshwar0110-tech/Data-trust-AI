# 1. Title: DataTrust AI v3: A Task-Conditioned Structural Dataset-Fitness Framework with Reciprocal AHP Weighting, Non-Compensatory Defect Policies, and Nested Empirical Calibration

**Academic Research Manuscript**  
**Programme:** MSc in Computational Statistics & Data Analytics  
**System Baseline:** DataTrust AI v3 (Research-Hardened & Experimentally Frozen)  
**Author:** MSc Research Candidate  
**Date:** September 2026  
**Keywords:** Data Quality, Machine Learning Reliability, Task-Conditioned Fitness, Analytic Hierarchy Process, Non-Compensatory Defect Policy, Train-Only Remediation, Closed-Loop Calibration, Nested Outer-Test Validation, Cryptographic Auditability  

---

## 2. Abstract

Existing data-quality systems primarily provide assertion-, schema-, profiling-, and anomaly-validation mechanisms (e.g., Great Expectations, AWS Deequ, TensorFlow Data Validation); DataTrust AI adds an explicit task-conditioned fitness and non-compensatory decision layer. Empirical dataset fitness is inherently task-conditioned: a tree-based classifier can tolerate auxiliary feature missingness with negligible performance loss, whereas supervised regression is acutely vulnerable to heavy-tailed outlier contamination, classification models suffer under severe class collapse, and autoregressive time-series models break entirely under temporal sequence disorder even when missingness is zero percent.

In this paper, we present **DataTrust AI v3**, an automated, task-conditioned decision framework that formalizes dataset fitness evaluation, non-compensatory risk control, and empirical calibration. DataTrust AI defines eight operationally distinct quality dimensions ($\mathbf{Q}_8$: Completeness, Validity, Consistency, Uniqueness, Timeliness, Outlier/Anomaly, Distribution Balance, and Coverage/Support). It maps tasks to dimension vulnerabilities via an authoritative literature-informed expert prior matrix $M(T,Q) \in [0,1]^{4 \times 8}$, generates reciprocal $8 \times 8$ Analytic Hierarchy Process (AHP) weighting matrices satisfying Saaty's consistency ratio ($CR < 0.10$), and enforces a three-tier non-compensatory defect policy (GREEN, YELLOW warning cap, RED hard veto) to eliminate compensatory masking. When data remediation is requested, the framework executes actions strictly on training splits while verifying holdout index integrity.

We evaluate DataTrust AI across a rigorous experimental program. In an omnibus controlled experiment ($N=8$ independent datasets, 5 deterministic seeds, 200 total runs), Full DataTrust produced an omnibus mean difference of $-20.34$ fitness points compared to static equal-weight baselines ($t = -5.067, df = 7, p = 0.0015$, Cohen's $d = -1.79$). Component isolation revealed that the non-compensatory defect policy is the dominant structural driver ($\Delta = -21.48$ points, $t = -4.637, p = 0.0024$), while AHP task weighting alone provided a directional reallocation ($\Delta = +1.14$ points, $t = 1.270, p = 0.2448$). In a nested 60/20/20 train/calibration/outer-test validation protocol evaluated on scale-free dimensionless relative error reduction ($R$), calibration was associated with reduced outer-test prediction error across all three uncapped GREEN datasets ($N=3$ exploratory subgroup, mean relative error reduction $44.65\%$, range $18.75\%$ to $86.02\%$), while safely masking calibration on defective YELLOW datasets ($N=5$) where the defect ceiling took precedence. All evaluations maintained 100% cryptographic determinism with zero reported mismatches across sequential runs, establishing an auditable foundation for trustworthy data-centric machine learning.

---

## 3. Introduction

Modern artificial intelligence and machine learning (ML) systems operate under the foundational assumption that ingested data accurately reflects the underlying phenomenon to be modeled. When production pipelines ingest corrupted, misaligned, or structurally degraded datasets, models experience silent performance degradation, severe test loss inflation, or uncalibrated predictions (Sculley et al., 2015). Existing data-quality systems primarily provide assertion-, schema-, profiling-, and anomaly-validation mechanisms, as exemplified by Great Expectations (Superconductive, 2020), AWS Deequ (Schelter et al., 2018, 2019), and TensorFlow Data Validation (Baylor et al., 2017; Breck et al., 2019).

While these platforms excel at verifying deterministic constraints and detecting distribution drift, they predominantly evaluate tabular data against uniform rules applied identically regardless of whether the downstream workload is descriptive business reporting, gradient-boosted decision tree regression, deep neural classification, or autoregressive time-series forecasting. This uniform validation paradigm overlooks a fundamental operational reality: **data quality is not an absolute, context-free property of a matrix; it is a task-conditioned relationship between data topology and downstream model assumptions**.

DataTrust AI v3 addresses this gap by adding an explicit task-conditioned fitness and non-compensatory decision layer. Rather than treating validation as an open-loop collection of assertions, DataTrust AI establishes a formal decision-theoretic control architecture. By integrating multi-criteria decision analysis (MCDA), non-compensatory choice theory, empirical error-gradient calibration, and leakage-free data hygiene, DataTrust AI provides an automated, reproducible, and verifiable framework for dataset trust certification.

---

## 4. Problem Statement

Let $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^n$ denote an empirical dataset intended for a machine learning task $T \in \mathcal{T}$. In conventional data validation frameworks, fitness is evaluated via a scalar objective function $S(\mathcal{D}) = \frac{1}{m} \sum_{j=1}^m \mathbb{I}(C_j(\mathcal{D}))$, where each $C_j$ is a static constraint.

This conventional formulation exhibits three fundamental failure modes:

1. **Context-Blind Scoring:** The utility of a dataset is treated as invariant across downstream tasks:
   $$\nabla_T S(\mathcal{D}) = \mathbf{0} \quad \forall T \in \mathcal{T}$$
   In reality, downstream model sensitivity to specific data dimensions varies by orders of magnitude depending on $T$.
2. **Compensatory Masking:** Under linear or unweighted additive aggregation, high scores in non-critical dimensions mathematically compensate for fatal defects in task-critical dimensions:
   $$S(\mathcal{D}) = \sum_{k=1}^K w_k q_k \ge \tau \quad \text{even when } \exists k^* \text{ s.t. } q_{k^*} = 0$$
   For example, an autoregressive time-series dataset with 100% completeness and perfect schema validity can score $> 90.0$ under additive scoring even if chronological timestamps are completely scrambled, causing downstream forecasting models to fail.
3. **Surrogate Misalignment:** Static scores operate without empirical feedback from downstream tasks. If a data quality score fails to correlate with or predict downstream performance degradation, it provides no actionable operational guarantee.

---

## 5. Research Gap

Existing data-quality systems primarily provide assertion-, schema-, profiling-, and anomaly-validation mechanisms; DataTrust AI adds an explicit task-conditioned fitness and non-compensatory decision layer. Prior academic and industrial contributions to data validation exhibit four primary research gaps:

| Limitation | Existing Validation Frameworks (e.g., Deequ, Great Expectations, TFDV) | DataTrust AI v3 Approach |
| :--- | :--- | :--- |
| **Task Conditionality** | Assertion thresholds applied identically across all analytical workloads (Schelter et al., 2018; Breck et al., 2019; Superconductive, 2020). | Task-conditioned sensitivity mapping ($M(T,Q)$) and reciprocal AHP weighting. |
| **Defect Aggregation** | Purely compensatory additive aggregation or binary pipeline halts without tiering (Schelter et al., 2019). | Non-compensatory defect policy enforcing warning ceilings (YELLOW) and hard circuit breakers (RED). |
| **Feedback Loop** | Open-loop execution; diagnostic profiling terminates without validation feedback. | Closed-loop calibration adapting quality weights against downstream empirical error gradients. |
| **Remediation Hygiene** | Preprocessing often fitted across entire datasets, introducing data leakage. | Train-only parameter estimation with holdout index integrity verification. |

---

## 6. Contributions

The primary contributions of DataTrust AI v3 are:

1. **Canonical 8D Quality Model ($\mathbf{Q}_8$):** A comprehensive, mathematically bounded quality model defining eight operationally distinct dimensions on $[0.0, 100.0]$.
2. **Autonomous Probabilistic Task Inference with Bounded Task Probabilities:** A structural classifier producing a normalized $P(T|\mathcal{D})$ task-probability vector across four canonical task types, with clustering confidence strictly bounded ($P \le 0.60$) to reflect inherent unsupervised ambiguity.
3. **Task Vulnerability Mapping & Reciprocal AHP Weighting:** An authoritative literature-informed expert prior matrix $M(T,Q) \in [0,1]^{4 \times 8}$ paired with four $8 \times 8$ pairwise reciprocal comparison matrices satisfying Saaty's consistency constraint ($CR < 0.10$).
4. **Non-Compensatory Defect Policy Engine:** An authoritative decision engine implementing GREEN, YELLOW ($C_{\text{cap}} = 70.0$), and RED ($F = 0.0$, `status = UNSAFE`) tiers that override compensatory scoring.
5. **Leakage-Free, Train-Only Remediation:** A transformation pipeline that fits statistical parameters exclusively on training splits and guarantees holdout index integrity via deterministic cryptographic hashing.
6. **Nested Calibration & Outer-Test Validation Protocol:** A 60/20/20 train/calibration/outer-test protocol that isolates calibration-outcome adaptation from independent predictive generalization in scale-free metric spaces.
7. **Cryptographic Reproducibility:** End-to-end deterministic execution generating SHA-256 audit receipts with canonical integer serialization.

---

## 7. Methodology & System Architecture

DataTrust AI v3 structures data fitness assessment into an end-to-end deterministic pipeline:

$$\mathcal{D} \xrightarrow{\text{Profile}} \mathbf{P} \xrightarrow{\text{Task Inference}} P(T|\mathcal{D}) \xrightarrow{\text{Quality Engine}} \mathbf{Q}_8 \xrightarrow{M(T,Q) \times \mathbf{A}^{(T)}} \mathbf{w}_{\text{AHP}} \xrightarrow{\text{Aggregation}} F_{\text{raw}} \xrightarrow{\text{Defect Policy}} F_{\text{final}}$$

$$\xrightarrow{\text{Train Split}} \mathcal{M}_{\text{ref}} \xrightarrow{\text{Calibration Split}} \epsilon_{\text{cal}} \xrightarrow{\text{Line Search}} \mathbf{w}_{\text{cal}} \xrightarrow{\text{Outer Test Split}} \epsilon_{\text{test}} \xrightarrow{\text{Audit}} \text{Receipt}_{\text{SHA-256}}$$

```
[Raw Ingestion D] ──> [Structural Profiler] ──> [Autonomous Task Inference P(T|D)]
                                                              │
┌─────────────────────────────────────────────────────────────┘
▼
[8 Quality Dimensions (Q8)] ──> [M(T,Q) Prior] ──> [AHP Weighting (CR < 0.10)]
                                                              │
┌─────────────────────────────────────────────────────────────┘
▼
[Raw Fitness F_raw = Σ w_i * q_i] ──> [Defect Policy (GREEN / YELLOW / RED)]
                                                              │
┌─────────────────────────────────────────────────────────────┘
▼
[60% Train: Fit Model M] ──> [20% Calibration: Observe Metric & Calibrate w_cal]
                                                              │
┌─────────────────────────────────────────────────────────────┘
▼
[20% Outer Test: Evaluate Implied vs Observed Metric] ──> [SHA-256 Audit Receipt]
```

---

## 8. Canonical 8D Quality Model

DataTrust AI standardizes on eight operationally distinct quality dimensions grounded in data quality literature (Wang & Strong, 1996):

1. **Completeness ($Q_{\text{CMP}}$):** Evaluates matrix density across cells, columns, and rows:
   $$Q_{\text{CMP}} = 100 \times \left(1 - \frac{N_{\text{missing}}}{N_{\text{total}}}\right) \times \prod_{j=1}^p \left(1 - \mathbb{I}(\text{miss}_j > 0.40)\right)$$
2. **Validity ($Q_{\text{VAL}}$):** Measures conformance to inferred data types, detection of infinite values, constant/zero-variance features, and target variable admissibility.
3. **Consistency ($Q_{\text{CNS}}$):** Quantifies structural cross-field contradictions, severe collinearity ($|r| > 0.95$), and deterministic target leakage ($|r| > 0.98$). Feature drift is explicitly excluded to preserve operational distinction from timeliness.
4. **Uniqueness ($Q_{\text{UNQ}}$):** Assesses entity duplication rates and primary key integrity across row observations.
5. **Timeliness ($Q_{\text{TIM}}$):** Evaluates temporal sequence monotonicity, chronological ordering, sampling interval regularity, and timestamp collision frequency in time-indexed datasets.
6. **Outlier / Anomaly ($Q_{\text{OUT}}$):** Evaluates multivariate contamination via Isolation Forest anomaly proportions ($c = 0.05$). Heavy-tailed kurtosis ($|\kappa| > 5.0$) is evaluated strictly as a descriptive indicator.
7. **Distribution Balance ($Q_{\text{BAL}}$):** Computes normalized Shannon entropy across target classes for classification:
   $$H_{\text{norm}}(Y) = -\sum_{c=1}^C \frac{p_c \log p_c}{\log C}$$
   For regression, it evaluates skewness and variance stability.
8. **Coverage / Support ($Q_{\text{COV}}$):** Quantifies feature space support and empirical domain density. Population representativeness is marked `UNAVAILABLE` unless an authoritative external sampling frame is provided.

All dimensions are mathematically bounded: $Q_k \in [0.0, 100.0]$.

---

## 9. $M(T,Q)$ Sensitivity Mapping

Downstream tasks exhibit fundamentally different failure surfaces when exposed to data contamination. DataTrust AI defines an authoritative, pre-experimental literature-informed expert prior matrix $M(T,Q) \in [0,1]^{4 \times 8}$. While the vulnerability sensitivities across tasks are qualitatively informed by empirical data quality literature (Wang & Strong, 1996; Schelter et al., 2018), the exact numerical matrix coefficients represent expert-specified prior weights rather than values uniquely derived from closed-form literature formulas:

| Task Type ($T$) | CMP | VAL | CNS | UNQ | TIM | OUT | BAL | COV |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Supervised Classification** | 0.85 | 0.90 | 0.70 | 0.75 | 0.30 | 0.65 | 0.95 | 0.80 |
| **Supervised Regression** | 0.85 | 0.90 | 0.80 | 0.75 | 0.30 | 0.95 | 0.60 | 0.75 |
| **Time-Series Forecasting** | 0.90 | 0.95 | 0.60 | 0.80 | 1.00 | 0.80 | 0.50 | 0.70 |
| **Clustering / Unsupervised BI** | 0.75 | 0.80 | 0.60 | 0.85 | 0.20 | 0.70 | 0.50 | 0.90 |

*Interpretation:* In time-series forecasting, timeliness is maximally critical ($M = 1.00$), whereas in classification, class distribution balance dominates ($M = 0.95$).

---

## 10. AHP Pairwise Reciprocity & Consistency Weighting

To transform sensitivity priors into mathematically consistent operational weights, DataTrust AI constructs four $8 \times 8$ positive reciprocal pairwise comparison matrices $\mathbf{A}^{(T)} = [a_{ij}^{(T)}]$ satisfying Saaty's reciprocity axiom (Saaty, 1977, 1980):

$$a_{ij}^{(T)} = \frac{1}{a_{ji}^{(T)}}, \quad a_{ii}^{(T)} = 1, \quad \forall i, j \in \{1, \dots, 8\}$$

The relative priority vector $\mathbf{w}_{\text{AHP}}$ corresponds to the principal eigenvector:

$$\mathbf{A}^{(T)} \mathbf{w}_{\text{AHP}} = \lambda_{\max} \mathbf{w}_{\text{AHP}}, \quad \sum_{i=1}^8 w_i = 1$$

To evaluate pairwise consistency, the Consistency Ratio ($CR$) is evaluated against Saaty's Random Index ($RI = 1.41$ for $n=8$):

$$CI = \frac{\lambda_{\max} - n}{n - 1}, \quad CR = \frac{CI}{RI}$$

Across all four task profiles, DataTrust AI matrices strictly enforce AHP pairwise reciprocity and consistency verification ($CR < 0.10$: $CR_{\text{clf}} = 0.041, CR_{\text{reg}} = 0.038, CR_{\text{ts}} = 0.045, CR_{\text{clu}} = 0.032$), guaranteeing mathematical reciprocity ($a_{jk} = 1/a_{kj}$) and acceptable consistency ($CR < 0.10$). Formal pairwise transitivity is not claimed without an explicit transitivity test.

---

## 11. Non-Compensatory Defect Policy

Linear additive scoring models exhibit *compensatory masking*, where a score of 100 in seven dimensions offsets a fatal 0 in an eighth dimension. Inspired by non-compensatory choice and multi-attribute decision theory (Fishburn, 1974; Tversky, 1972), DataTrust AI implements a non-compensatory defect policy with operational threshold tiers (GREEN, YELLOW, RED). The exact numerical thresholds ($C_{\text{cap}} = 70.0$, $F_{\text{veto}} = 0.0$) represent pragmatic operational risk ceilings designed to enforce safety precedence over additive aggregation, rather than mathematically derived values from axiomatic utility functions:

1. **GREEN (Unconstrained):** No severe defects detected. Final fitness equals raw weighted fitness:
   $$F_{\text{final}} = F_{\text{raw}} = \sum_{k=1}^8 w_k Q_k$$
2. **YELLOW (Warning Ceiling):** Non-fatal but severe quality anomalies (e.g., moderate feature missingness or high outlier concentration). Fitness is strictly capped:
   $$F_{\text{final}} = \min(F_{\text{raw}}, C_{\text{cap}}), \quad C_{\text{cap}} = 70.0$$
3. **RED (Hard Non-Compensatory Veto):** Fatal, irrecoverable defects detected (e.g., deterministic target leakage, complete chronological disorder in time-series, or severe class collapse $< 2\%$). The veto operator immediately overrides raw fitness:
   $$F_{\text{final}} = 0.0, \quad \text{Status} = \text{UNSAFE}$$

---

## 12. Train-Only Remediation & Holdout Immutability

Data preprocessing often introduces severe data leakage by fitting transformation parameters (imputation medians, scaling ranges, outlier fences) across the combined dataset prior to splitting. DataTrust AI enforces strict separation:

1. **Partitioning Precedence:** The raw dataset is partitioned into training and evaluation splits before any remediation transformation is computed.
2. **Train-Only Fitting:** Statistical parameters $\boldsymbol{\theta}_{\text{rem}}$ are estimated exclusively on training observations:
   $$\hat{\boldsymbol{\theta}} = g(\mathcal{D}_{\text{train}})$$
3. **Holdout Index Integrity:** Transformations are applied to holdout test splits strictly using frozen parameters $\hat{\boldsymbol{\theta}}$. Test observations are never dropped, synthesized, or re-indexed. Holdout index integrity is verified via deterministic SHA-256 index hashing before and after remediation.

---

## 13. Closed-Loop Performance Calibration

DataTrust AI incorporates closed-loop calibration to adapt dimension weights against empirical downstream validation metrics. 

Given an empirical reference model $\mathcal{M}_{\text{ref}}$ fit on training data, performance is observed on a calibration partition to produce an observed validation metric $y_{\text{cal}}$ (e.g., RMSE or Macro F1). The implied predicted performance is derived via:

$$\hat{y}_{\text{pred}} = \text{predict\_performance}(F_{\text{raw}}, T, \sigma_y)$$

Calibration optimizes weight vector $\mathbf{w}$ to minimize prediction loss $\mathcal{L}(\mathbf{w}) = |y_{\text{cal}} - \hat{y}_{\text{pred}}(\mathbf{w})|$ subject to simplex constraints:

$$\min_{\mathbf{w}} \mathcal{L}(\mathbf{w}) \quad \text{s.t.} \quad \sum_{i=1}^8 w_i = 1, \quad w_i \ge 0$$

An adapted weight vector $\mathbf{w}_{\text{cal}}$ is accepted if and only if it strictly reduces prediction loss: $\mathcal{L}(\mathbf{w}_{\text{cal}}) < \mathcal{L}(\mathbf{w}_{\text{AHP}})$. If loss increases or boundaries are violated, calibration is rejected and prior AHP weights are preserved.

---

## 14. Marginal Remediation Utility (MRU) Optimization

Remediation actions incur operational and computational costs $c_a \in \{1, 2, 3, 4, 5\}$. DataTrust AI formalizes intervention selection via Marginal Remediation Utility (MRU):

$$\text{MRU}_a = \frac{\Delta F_{\text{sim}, a}}{c_a}$$

where $\Delta F_{\text{sim}, a}$ represents the counterfactual predicted fitness gain from eliminating defect $a$. Actions are ranked greedily by MRU under an operational budget constraint $B$:

$$\max_{\mathcal{A}' \subseteq \mathcal{A}} \sum_{a \in \mathcal{A}'} \Delta F_{\text{sim}, a} \quad \text{s.t.} \quad \sum_{a \in \mathcal{A}'} c_a \le B$$

The system explicitly labels simulated gains as `PREDICTED / SIMULATED` to distinguish them from observed empirical downstream metrics.

---

## 15. Epistemic / Empirical Uncertainty Formulation

Data quality scores are subject to measurement variability stemming from finite sample sizes, heuristic boundary noise, and surrogate residual gaps. DataTrust AI calculates an **empirical uncertainty interval** (or estimated fitness interval) based on composite variance aggregation:

For datasets with $N < 15$, a bounded small-sample margin ($3.5$) is applied. For $N \ge 15$, four empirical uncertainty components are aggregated:

1. **Sampling Variance ($\sigma_{\text{sampling}}^2$):** Evaluated via empirical bootstrap score dispersion ($B=10$) under sample perturbation ($s_{\text{boot}}^2$ without dividing by $\sqrt{B}$, directly reflecting fitness dispersion under sample perturbation).
2. **Measurement Variance ($\sigma_{\text{measurement}}^2$):** Weighted sum of heuristic boundary noise across dimension estimators:
   $$\sigma_{\text{measurement}}^2 = \sum_{i=1}^8 w_i^2 \sigma_{Q_i}^2$$
3. **Task Inference Uncertainty ($\sigma_{\text{task}}^2$):** Derived from classification entropy based on $(1.0 - \text{task\_confidence})$.
4. **Surrogate Residual Variance ($\sigma_{\text{residual}}^2$):** Accounts for the empirical gap between surrogate prediction and observed downstream model performance.

The composite variance is evaluated as:

$$\sigma_{\text{composite}}^2 = \sigma_{\text{sampling}}^2 + \sigma_{\text{measurement}}^2 + \sigma_{\text{task}}^2 + \sigma_{\text{residual}}^2$$

$$\text{SE}_{\text{composite}} = \sqrt{\sigma_{\text{composite}}^2}$$

**Explicit Independence Assumption:** This composite summation assumes zero covariance (orthogonality) between sampling, measurement, task inference, and downstream residual noise components.

With nominal coverage multiplier $k = 1.96$, the margin of error is bounded as $\text{margin} = \text{clip}(1.96 \cdot \text{SE}_{\text{composite}}, \; 0.5, \; 6.0)$, producing the estimated fitness interval:

$$\mathcal{I}_F = [\max(0.0, F_{\text{final}} - \text{margin}), \; \min(100.0, F_{\text{final}} + \text{margin})]$$

In accordance with rigorous statistical terminology, this interval is designated an **estimated fitness interval** rather than a classical Neyman-Pearson confidence interval.

---

## 16. Cryptographic Auditability & Reproducibility

To provide deterministic integrity checking and auditability, DataTrust AI serializes execution parameters into a deterministic JSON object and computes a SHA-256 cryptographic receipt:

$$\text{Receipt}_{\text{SHA-256}} = \text{SHA256}(\text{CanonicalJSON}(\mathbf{Q}_8, \mathbf{w}_{\text{AHP}}, F_{\text{raw}}, F_{\text{final}}, \text{PolicyTier}, \mathcal{H}_{\text{test\_idx}}))$$

To eliminate cross-platform hashing divergence caused by NumPy scalar wrapper representations (e.g., `np.int64(42)` stringifying differently from Python integer `42`), the platform enforces a strict canonical integer serialization function:

$$\text{canonicalize\_index}(I) = [\text{int}(x) \text{ for } x \in I]$$

guaranteeing identical cryptographic receipts across deterministic repeated executions.

---

## 17. Experimental Design

### 17.1 Five Controlled Experimental Conditions
To isolate full-system behavior and individual architectural components, we define five experimental conditions:

1. **`STATIC_BASELINE`:** Static equal-weight scoring ($\mathbf{w} = [1/8, \dots, 1/8]^\top$) without task conditioning, defect policies, or calibration.
2. **`FULL_DATATRUST`:** The complete DataTrust AI v3 framework including $M(T,Q)$, AHP weighting, non-compensatory defect policy, train-only remediation, and empirical calibration.
3. **`NO_VETO`:** Full DataTrust with the non-compensatory defect policy disabled ($F_{\text{final}} = F_{\text{raw}}$), isolating linear task-conditioned AHP weighting.
4. **`NO_CALIBRATION`:** Full DataTrust with closed-loop calibration disabled, preserving unadjusted prior AHP weights ($\mathbf{w}_{\text{AHP}}$) through the defect policy.
5. **`NO_REMEDIATION`:** Full DataTrust evaluated on un-remediated data, isolating automated data transformation contributions.

### 17.2 Experimental Unit Discipline
In accordance with computational statistics standards:
- The **independent experimental unit is the DATASET** ($N=8$ datasets, $df=7$).
- Repeated deterministic random seeds ($s \in \{42, 43, 44, 45, 46\}$) represent **repeated measurements**, not independent datasets. Seed-level variance is aggregated within datasets to avoid degrees-of-freedom inflation.

---

## 18. Dataset Provenance & Benchmark Matrix

The experimental evaluation spans an audited benchmark matrix categorized into distinct provenance tiers:

| Tier | Dataset Name | ID | Task Type | Dimensions ($N \times p$) | Target / Temporal Index | Audited Provenance & Source |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Tier B** | UCI Red Wine Quality | `B1_wine_quality` | Regression | $1,599 \times 12$ | `quality` | UCI Machine Learning Repository (Cortez et al., 2009) |
| **Tier B** | UCI Air Quality Sensors | `B2_air_quality` | Time-Series | $1,500 \times 15$ | `CO(GT)` / `timestamp` | UCI ML Repository (De Vito et al., 2008) |
| **Tier B** | Melbourne Daily Temperatures | `B3_melb_temperatures` | Time-Series | $3,650 \times 2$ | `Temp` / `Date` | Australian Bureau of Meteorology (Diagnostic Benchmark) |
| **Tier B** | Iris Species Morphometrics | `B4_iris_species` | Classification | $150 \times 5$ | `variety` | Fisher / Anderson Morphometrics (Pedregosa et al., 2011) |
| **Tier C** | California Housing | `C1_cal_housing` | Regression | $1,500 \times 9$ | `MedHouseVal` | 1990 U.S. Census Bureau (Pace & Barry, 1997) |
| **Tier C** | Diabetes Progression | `C2_diabetes` | Regression | $442 \times 11$ | `disease_progression` | Efron et al. (2004) Diagnostic Benchmark |
| **Tier C** | Breast Cancer Diagnostic | `C3_breast_cancer` | Classification | $569 \times 31$ | `biopsy_result` | Wisconsin Diagnostic Database (Street et al., 1993) |
| **Tier C** | Retail Store Sales | `C4_retail_sales` | Regression | $350 \times 6$ | `sales_amount` / `order_date` | Enterprise Tabular Holdout with Transactional Dates |

*Provenance Distinction:* Tier B datasets represent diagnostic and calibration reference benchmarks utilized during algorithmic tuning. Tier C datasets represent held-out generalization benchmarks evaluated to test schema robustness across unseen domains. Tier A comprises synthetic stress scenarios evaluated for boundary safety.

---

## 19. Main Controlled Research Experiment Results

The main controlled experiment evaluated 5 conditions across 8 benchmark datasets and 5 deterministic seeds ($8 \times 5 \times 5 = 200$ total evaluations).

### 19.1 Primary Omnibus Results Table ($N=8$ Datasets, $df=7$)

| Condition Contrast | Isolated Component | Mean $\Delta$ (pts) | Median $\Delta$ | SD ($\Delta$) | Paired $t$ | df | $p$-value | Wilcoxon $p$ | Cohen's $d$ | Formal Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `FULL_DATATRUST` vs `STATIC_BASELINE` | Full-System Differentiation ($H_1$) | **-20.34** | -21.44 | 11.35 | -5.067 | 7 | **0.0015** | **0.0078** | **-1.79** | **Supported (Omnibus)** |
| `NO_VETO` vs `STATIC_BASELINE` | AHP Weighting Isolation ($H_3$) | **+1.14** | +0.47 | 2.54 | +1.270 | 7 | 0.2448 | 0.3125 | +0.45 | **Partially Supported (Directional)** |
| `FULL_DATATRUST` vs `NO_VETO` | Defect Policy Circuit Breaker ($H_2$) | **-21.48** | -24.47 | 13.10 | -4.637 | 7 | **0.0024** | **0.0078** | **-1.64** | **Supported** |
| `FULL_DATATRUST` vs `NO_CALIBRATION` | Calibration Adaptation ($H_4$) | **-7.28** | 0.00 | 16.25 | -1.267 | 7 | 0.2456 | 0.2500 | -0.45 | **Partially Supported (Safety-Constrained)** |
| `FULL_DATATRUST` vs `NO_REMEDIATION` | Automated Remediation Impact ($H_5$) | **0.00** | 0.00 | 0.00 | N/A | 7 | N/A | N/A | 0.00 | **Not Estimable on Clean Benchmarks** |

### 19.2 Interpretation of Omnibus Contrasts
- **Full-System Omnibus Effect ($H_1$):** `FULL_DATATRUST` differs significantly from `STATIC_BASELINE` ($p = 0.0015, d = -1.79$). This represents an omnibus system-level divergence; it does not claim that $M(T,Q)$ sensitivity mapping alone caused the $-20.34$ point shift.

---

## 20. Nested Calibration / Outer Test Validation

### 20.1 Protocol Architecture
In the original 200-run experiment, a single validation split was used for both calibration and evaluation, accurately labeled as *"Outcome-adaptive calibration evaluated on the calibration outcome"*.

To evaluate true out-of-sample predictive generalization, we implemented a **Nested Calibration Protocol** partitioning data into three sequestered splits:
- **60% Train Split:** Fit downstream reference model.
- **20% Calibration Split:** Downstream model is evaluated to obtain $y_{\text{cal}}$; empirical calibration line search adapts $\mathbf{w}_{\text{AHP}} \to \mathbf{w}_{\text{cal}}$. Weights are then **frozen**.
- **20% Outer Test Split:** Completely untouched during model training and calibration. Used exclusively to assess generalization error between fitness-implied predicted performance ($\hat{y}_{\text{pred}}$) and observed outer performance ($y_{\text{outer}}$).
- **Chronological Guarantee:** For all time-series, partitions strictly follow temporal order:
  $$\max(t_{\text{train}}) < \min(t_{\text{cal}}) \quad \text{AND} \quad \max(t_{\text{cal}}) < \min(t_{\text{test}})$$

### 20.2 Dimensionless Relative Error Reduction Formulation
Because benchmark datasets span incompatible physical scales, pooling raw absolute errors ($\text{mean}(|\epsilon|)$) is methodologically invalid. We define the **Dimensionless Relative Error Reduction** ($R$) for each dataset:

$$R = \frac{|\epsilon_{\text{before}}| - |\epsilon_{\text{after}}|}{|\epsilon_{\text{before}}| + \epsilon_{\text{tol}}}$$

where $\epsilon_{\text{tol}} = 10^{-9}$ prevents division by zero. If baseline error is zero, $R$ is defined strictly as $0.0000$.

### 20.3 Outer-Test Results Table ($N=8$ Datasets)

| Dataset | Task | Policy Masked | Abs Error Before ($|\epsilon_{\text{before}}|$) | Abs Error After ($|\epsilon_{\text{after}}|$) | Relative Error Reduction ($R$) | Rel Error Reduction (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Breast Cancer Diagnostic** | Supervised Classification | **True** | 0.3232 | 0.3232 | 0.0000 | 0.00% |
| **California Housing** | Supervised Regression | **True** | 0.1151 | 0.1151 | 0.0000 | 0.00% |
| **Diabetes Progression** | Supervised Regression | **False** | 36.9323 | 30.0083 | 0.1875 | 18.75% |
| **Iris Species Morphometrics** | Supervised Classification | **True** | 0.3507 | 0.3507 | 0.0000 | 0.00% |
| **Melbourne Daily Temperatures** | Time-Series Forecasting | **False** | 1.3361 | 0.9461 | 0.2919 | 29.19% |
| **Retail Store Sales** | Supervised Regression | **False** | 203.6315 | 28.4606 | 0.8602 | 86.02% |
| **UCI Air Quality Sensors** | Time-Series Forecasting | **True** | 14.1129 | 14.1129 | 0.0000 | 0.00% |
| **UCI Red Wine Quality** | Supervised Regression | **True** | 0.0736 | 0.0736 | 0.0000 | 0.00% |

### 20.4 Aggregate Statistics & Subgroup Analysis
1. **All Benchmark Datasets ($N=8$ Datasets, $df=7$):**
   - Mean Relative Error Reduction: **0.1674 (16.74%)**
   - Median Relative Error Reduction: **0.0000 (0.00%)**
   - Standard Deviation: **0.3014**
   - Range: **[0.0000, 0.8602] (0.00% to 86.02%)**
   - *Descriptive Role:* This aggregate serves as a **descriptive cross-dataset summary**, not universal inferential evidence. Datasets across distinct tasks (classification, regression, time-series) are not treated as statistically homogeneous merely because $R$ is dimensionless.
2. **Uncapped GREEN Benchmarks ($N=3$ Exploratory Subgroup):**
   - Mean Relative Error Reduction: **0.4465 (44.65%)**
   - Median Relative Error Reduction: **0.2919 (29.19%)**
   - Standard Deviation: **0.3620**
   - Range: **[0.1875, 0.8602] (18.75% to 86.02%)**
   - Inferential Paired Contrast: $t = -1.063, df = 2, p = 0.3990$, Cohen's $d = -0.61$.
   - **Exploratory Interpretation:** In the three uncapped benchmark datasets evaluated, calibration learned from the calibration partition was associated with reduced outer-test prediction error. However, with $N=3$ datasets ($df=2$), this subgroup is **exploratory and statistically underpowered** ($p = 0.3990$). It must not be described as statistically established evidence or general superiority.
3. **Policy-Masked YELLOW Benchmarks ($N=5$ Datasets):**
   - In all 5 datasets, `policy_masked = True` and $R = 0.0000$.
   - **Safety Precedence:** Reported fitness before and after calibration was clamped at $70.00$ by the non-compensatory warning ceiling ($C_{\text{cap}} = 70.0$). This is not a calibration failure; it proves that DataTrust AI's safety policy takes precedence over empirical tuning, preventing downstream optimization from bypassing data defect alerts.

---

## 21. Component Isolation & Ablation Study

Contrasting individual ablation conditions isolates the specific mathematical contributions of each architectural component:

1. **AHP Weighting Isolation (`NO_VETO` vs `STATIC_BASELINE`):** Isolates unconstrained task-conditioned weighting without defect policies. The mean difference of $+1.14$ points ($p = 0.2448, d = +0.45$) indicates a measurable directional reallocation, but the independent effect was not statistically established on this benchmark set.
2. **Defect Policy Isolation (`FULL_DATATRUST` vs `NO_VETO`):** Isolates the contribution of the non-compensatory circuit breaker. The large, statistically significant shift ($\Delta = -21.48$ points, $p = 0.0024, d = -1.64$) establishes that the defect policy is the primary structural driver of differentiation in the full system.
3. **Calibration Adaptation (`FULL_DATATRUST` vs `NO_CALIBRATION`):** Isolates empirical weight adjustment. The shift ($\Delta = -7.28$ points, $p = 0.2456, d = -0.45$) reflects empirical tuning on unmasked surfaces while remaining constrained by policy caps on defective datasets.
4. **Remediation Impact (`FULL_DATATRUST` vs `NO_REMEDIATION`):** Across standard curated benchmarks, initial dimension scores exceeded defect thresholds ($\ge 70.0$), resulting in zero mandatory automated row-dropping actions ($\Delta = 0.00$ points). Remediation contribution is therefore not estimable from clean benchmark datasets and must be evaluated via synthetic fault injection.

---

## 22. Safety Stress Tests & Adversarial Verification

To verify that DataTrust AI's safety circuit breakers trigger deterministically under extreme data corruptions, six synthetic adversarial scenarios were evaluated, matching the frozen empirical benchmark artifact (`stress_test_results.csv`):

| Scenario ID | Tested Stress / Edge Case | Injected Corruption Description | Expected Action / Tier | Observed Tier | Observed Final Fitness | Observed Operational Status | Safety Verified |
| :--- | :--- | :--- | :--- | :---: | :---: | :--- | :---: |
| **S1** | Severe Class Collapse | Minority class represents 0.67% ($< 2.0\%$ veto threshold) | RED Hard Veto ($F=0.0$, `status=UNSAFE`) | **RED** | **0.0** | `UNSAFE` (Hard Veto Fired) | **True** |
| **S2** | Temporal Sequence Disorder | Non-monotonic chronological timestamps with collisions | RED Hard Veto (`TEMPORAL_SEQUENCE_DISORDER`) | **RED** | **0.0** | `UNSAFE` (Sequencing Veto) | **True** |
| **S3** | Moderate Class Imbalance | Minority class represents 6.0% ($< 10.0\%$ warning threshold) | YELLOW Warning Cap ($F \le 65.0$, `status=WARNING`) | **YELLOW** | **65.0** | `WARNING` (Capped at Ceiling) | **True** |
| **S4** | Flat Dimension Scores | Identical dimension scores across all 8 dimensions ($\nabla F = \mathbf{0}$) | Calibration Rejected (`provenance=REJECTED`) | **N/A** | **75.0** | `Calibration Rejected` | **True** |
| **S5** | Simplex Boundary Clipping | Gradient updates bounded within $[0.02, 0.60]$ on probability simplex | Sum of canonical calibrated weights strictly equals 1.0000 | **N/A** | **75.0** | `Sum = 1.0000` (Simplex Preserved) | **True** |
| **S6** | Insufficient Validation Data | Dataset with $N=5$ ($< 10$ required observations for validation) | Validation Omitted (`observed_metric=None`, no fake 0.0) | **GREEN** | **95.8** | `NORMAL` (Validation Safely Omitted) | **True** |

All 6 scenarios (100%) passed deterministically, verifying that non-compensatory safety rules enforce absolute circuit breakers regardless of completeness.

---

## 23. Statistical Analysis Methodology

To ensure defensible inferential conclusions:
1. **Experimental Unit:** All inferential tests treat the **dataset** as the independent unit of analysis ($N=8, df=7$). Repeated deterministic random seeds ($s \in \{42, 43, 44, 45, 46\}$) represent repeated measurements.
2. **Paired Statistical Tests:** Two-tailed paired Student's $t$-tests evaluate mean parametric shifts across paired condition runs, complemented by non-parametric Wilcoxon signed-rank tests to account for potential non-normality.
3. **Effect Size Quantification:** Standardized effect sizes are computed via Cohen's $d$:
   $$d = \frac{\bar{\Delta}}{s_{\Delta}}$$
4. **Dimensionless Outer-Test Error:** Cross-dataset pooling is conducted strictly on scale-free relative error reduction ($R$), preventing physical metric scale mixing. The aggregate across all $N=8$ datasets is reported descriptively without assuming cross-task population homogeneity.

---

## 24. Results Summary

Integrating all empirical campaigns:
1. **Generalization Task Inference:** In an independent 11-dataset generalization campaign, autonomous task inference achieved 90.9% agreement (10/11), correctly flagging one unlabeled continuous tabular matrix as structurally ambiguous and requiring user authority.
2. **Time-Series Validation Bug Fix:** Eliminating the legacy fake-zero bug in univariate series and enforcing lag-1 autoregressive features with strict chronological sorting yielded an authentic empirical forecast error ($\text{Observed RMSE} = 2.46$ on Melbourne Temperatures).
3. **Main Controlled Experiment ($N=8$):** Full DataTrust demonstrated statistically significant structural differentiation from static equal weighting ($p = 0.0015, d = -1.79$), driven predominantly by the non-compensatory defect policy ($p = 0.0024, d = -1.64$).
4. **Nested Outer-Test Validation ($N=8$):** Calibration learned on a 20% calibration split was associated with reduced outer-test prediction error across all three uncapped GREEN datasets ($N=3$ exploratory subgroup, mean $R = 44.65\%$), while safely enforcing policy ceilings on five defective YELLOW datasets ($R = 0.00\%$).
5. **Clustering Semantics:** Clustering downstream validation is strictly defined as internal K-Means silhouette validity, not an out-of-sample generalization metric.
6. **Reproducibility:** Across two complete sequential executions ($2 \times 40 = 80$ nested evaluations and $2 \times 200 = 400$ main evaluations), 0 numerical mismatches were observed, with 100% of automated research invariants satisfied.

---

## 25. Discussion

### 25.1 Decoupling Structural Dataset Fitness from Downstream Model Accuracy
A critical insight established by DataTrust AI is that **dataset fitness cannot be inferred solely from downstream model performance**. In Scenario S1 (Severe Class Collapse), a majority-class classifier can achieve 99% accuracy on a test set containing 99% majority instances. Traditional accuracy-driven profilers would declare the dataset exceptional. In contrast, DataTrust AI detects the structural collapse, enforces a RED Hard Veto ($F = 0.0$), and blocks operational deployment. High downstream accuracy on a structurally corrupted dataset does not indicate safety; it highlights the acute danger of relying on model-side metrics alone to validate data trust.

### 25.2 The Mechanics of Policy Masking
The nested calibration protocol illuminated the vital interaction between empirical optimization and safety boundaries. In five of eight datasets, calibration successfully found weight shifts that reduced surrogate loss, yet the final reported fitness remained clamped at $70.00$. This policy masking is not a technical failure; it reflects deliberate decision-theoretic design. If a dataset exhibits known quality anomalies, empirical surrogate optimization must not be permitted to optimize away safety warnings.

---

## 26. Limitations

We explicitly acknowledge five operational and architectural limitations, establishing the negative scope of this research:

1. **Prior Sensitivity Dependence:** While $M(T,Q)$ is grounded in qualitative data quality literature (Wang & Strong, 1996; Schelter et al., 2018), it represents an expert-curated prior rather than a mathematically unique optimum.
2. **Surrogate Model Scope:** Downstream validation evaluates standard surrogate model families (Random Forests, Ridge Autoregression, K-Means); highly specialized architectures (e.g., Transformers, Graph Neural Networks) may exhibit different vulnerability profiles. DataTrust AI does not claim to improve the intrinsic predictive capacity of downstream ML models, but rather calibrates the fidelity of fitness-to-metric predictions.
3. **Remediation Benchmark Inestimability:** Curated academic benchmarks contain few structural defects, precluding empirical estimation of automated remediation utility on standard benchmark sets without synthetic corruption. We do not claim that standard benchmarks demonstrate remediation efficacy.
4. **Epistemic Ambiguity of Unsupervised Tables:** Truly unlabeled continuous tables cannot be autonomously mapped to clustering with high confidence ($P \le 0.60$) without human authority. We do not claim universal autonomous task inference.
5. **Sample Size Constraints in Calibration Subgroup:** The uncapped calibration subgroup comprises $N=3$ datasets ($df=2$), rendering it exploratory and statistically underpowered ($p = 0.3990$). We do not claim statistically established calibration superiority across the benchmark suite.
6. **No Universal Superiority:** We do not claim universal superiority over existing validation frameworks; rather, DataTrust AI provides a formal decision-theoretic and task-conditioned layer that complements static assertions.

---

## 27. Threats to Validity

- **Internal Validity:** Potential data leakage was eliminated by enforcing strict chronological sorting for time-series:
  $$\max(t_{\text{train}}) < \min(t_{\text{cal}}) \quad \text{AND} \quad \max(t_{\text{cal}}) < \min(t_{\text{test}})$$
  and verifying holdout index integrity using SHA-256 index hashes.
- **External Validity:** While the benchmark suite spans eight diverse datasets across multiple domains, real-world industrial databases may exhibit complex relational schemas not captured in flat tabular benchmarks.
- **Construct Validity:** Quality dimensions were constructed to reflect operational machine learning failure modes; however, specific domain metrics (e.g., fairness, clinical compliance) require future specialized extensions.
- **Conclusion Validity:** Cross-dataset error pooling was eliminated by adopting scale-free relative error reduction ($R$), and dataset was strictly maintained as the independent unit of analysis ($N=8$).

---

## 28. Conclusion

DataTrust AI v3 establishes a task-conditioned, decision-theoretic, and cryptographically auditable framework for dataset fitness assessment.

### Summary of Authoritative Findings:
1. **Full-System Differentiation:** Full DataTrust produces statistically significant, materially different structural fitness decisions compared to static equal-weight aggregation ($p = 0.0015, d = -1.79$).
2. **Defect Policy Dominance:** The non-compensatory defect policy is the primary structural driver of this differentiation ($p = 0.0024, d = -1.64$), successfully eliminating compensatory masking.
3. **Deterministic Safety Protection:** Non-compensatory safety vetoes trigger deterministically under adversarial corruptions (6/6 passed, 100%).
4. **Nested Outer-Test Generalization:** In the three uncapped benchmark datasets evaluated, calibration learned from the calibration partition was associated with reduced outer-test prediction error ($N=3$ exploratory subgroup, mean $R = 44.65\%$).
5. **Sequestered Protocol Separation:** Sequestered 60/20/20 partitioning successfully decouples outcome-adaptive calibration from honest outer-test validation.
6. **Reproducibility:** End-to-end execution is 100% deterministic, verifying 0 numerical mismatches across sequential runs and satisfying 11/11 research invariants.

By replacing static uniform assertions with task-specific reciprocal AHP weighting, enforcing non-compensatory defect policy circuit breakers, eliminating data leakage via train-only remediation, and decoupling empirical calibration from sequestered outer-test validation, DataTrust AI provides a principled foundation for trustworthy data-centric artificial intelligence.

---

## 29. References

1. **Baylor, D., Breck, E., Chiu, H. T., Fenu, G., Floratou, S., Joshi, G., ... & Zinkevich, M.** (2017). TFX: A TensorFlow-based production-scale machine learning platform. In *Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 1387-1395).
2. **Breck, E., Polyzotis, N., Roy, S., Whang, S., & Zinkevich, M.** (2019). Data validation for machine learning. In *Proceedings of the 2nd MLSys Conference*.
3. **Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J.** (2009). Modeling wine preferences by data mining from physicochemical properties. *Decision Support Systems*, 47(4), 547-553.
4. **De Vito, S., Massera, E., Piga, M., Martinotto, L., & Di Francia, G.** (2008). On field calibration of an electronic nose for benzene estimation in an urban pollution monitoring scenario. *Sensors and Actuators B: Chemical*, 129(2), 750-757.
5. **Efron, B., Hastie, T., Johnstone, I., & Tibshirani, R.** (2004). Least angle regression. *The Annals of Statistics*, 32(2), 407-499.
6. **Fishburn, P. C.** (1974). Lexicographic orders, utilities and decision rules: A survey. *Management Science*, 20(11), 1442-1471.
7. **Pace, R. K., & Barry, R.** (1997). Sparse spatial autoregressions. *Statistics & Probability Letters*, 33(3), 291-297.
8. **Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, É.** (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.
9. **Saaty, T. L.** (1977). A scaling method for priorities in hierarchical structures. *Journal of Mathematical Psychology*, 15(3), 234-281.
10. **Saaty, T. L.** (1980). *The Analytic Hierarchy Process: Planning, Priority Setting, Resource Allocation*. McGraw-Hill, New York.
11. **Schelter, S., Lange, D., Schmidt, P., Celikel, M., Biessmann, F., & Grafberger, A.** (2018). Automating data quality verification at scale. *Proceedings of the VLDB Endowment*, 11(12), 1781-1794.
12. **Schelter, S., Rukat, F., & Biessmann, F.** (2019). Unit testing data with Deequ. In *Proceedings of the 2019 International Conference on Management of Data (SIGMOD)* (pp. 1993-1996).
13. **Sculley, D., Holt, G., Golovin, D., Davydov, E., Phillips, T., Ebner, D., ... & Dennison, D.** (2015). Hidden technical debt in machine learning systems. In *Advances in Neural Information Processing Systems (NeurIPS)* (pp. 2503-2511).
14. **Street, W. N., Wolberg, W. H., & Mangasarian, O. L.** (1993). Nuclear feature extraction for breast tumor diagnosis. In *Biomedical Image Processing and Biomedical Visualization* (Vol. 1905, pp. 861-870). SPIE.
15. **Superconductive / Great Expectations Open Source Project.** (2020). *Great Expectations: Always know what to expect from your data*. Technical Documentation and Open Source Repository.
16. **Tversky, A.** (1972). Elimination by aspects: A theory of choice. *Psychological Review*, 79(4), 281-299.
17. **Wang, R. Y., & Strong, D. M.** (1996). Beyond accuracy: What data quality means to data consumers. *Journal of Management Information Systems*, 12(4), 5-33.
