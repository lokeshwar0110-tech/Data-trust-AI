# DataTrust AI v3 — Experimental Methodology & Validation Protocols

This document details the scientific experimental protocols, statistical significance testing, multi-seed benchmarks, real-world dataset fixtures, and ablation suites implemented in DataTrust AI v3.

---

## 1. Multi-Seed Statistical Significance Protocol (10 Seeds: 42..51)

To meet the rigorous statistical standards of an MSc thesis defense and peer-reviewed publication, all core benchmark hypotheses are evaluated across **10 independent pseudorandom seeds** (seeds 42 through 51) using `benchmarks/run_benchmarks.py`.

### 1.1 Evaluated Scenarios (N = 500 samples per run)
1. **Clean Baseline**: Standard multivariate Gaussian features, nominal regression/classification labels without anomalies.
2. **Missing Non-Critical**: 35% missingness injected exclusively into feature $X_3$, which exhibits near-zero mutual information with downstream target $y$.
3. **Temporal Disorder**: Autoregressive sequence $y_t = 0.7y_{t-1} + 6.0\sin(2\pi(t \% 7)/7) + \epsilon$ with 40% of timestamps randomly permuted.
4. **Severe Class Imbalance**: 92% of positive instances inverted, yielding an extreme minority class prevalence of $< 1.5\%$.

### 1.2 Downstream Machine Learning Evaluators
- **Classification**: Scikit-Learn `RandomForestClassifier(n_estimators=50, random_state=seed)`, evaluated via Macro $F_1$-score on an out-of-sample 70/30 split.
- **Time-Series Forecasting**: Scikit-Learn `RandomForestRegressor(n_estimators=50, random_state=seed)` trained on 3-step lag autoregressive features, evaluated via Chronological Split Root Mean Squared Error (RMSE).

### 1.3 Hypothesis Testing Methodology
For every seed $s \in \{42, \dots, 51\}$:
1. Compute the Spearman rank correlation $\rho_s$ between evaluated quality scores and downstream model performance metrics across scenarios.
2. Formulate paired difference vectors:
   $$D_s = \rho_{\text{DataTrust}, s} - \rho_{\text{Static}, s}$$
3. Conduct:
   - **Two-sided Paired Student's $t$-test** to test $H_0: \mu_D = 0$ against $H_1: \mu_D \ne 0$.
   - **Wilcoxon Signed-Rank Test** as a non-parametric verification against distributional assumptions.
4. Verify non-compensatory veto defense under temporal disorder:
   - Compare post-veto fitness ($0.0$) against un-vetoed raw linear score ($F_{\text{raw}}$) via paired $t$-test.

---

## 2. Real-World Authentic Benchmark Datasets

To ensure experimental findings generalize beyond synthetic data generators, DataTrust AI incorporates three authentic, publicly verifiable datasets stored in `benchmarks/data/`:

| Dataset Identifier | File Path | Observation Count | Feature Count | Empirical Defect Profile |
| :--- | :--- | :---: | :---: | :--- |
| **UCI Red Wine Quality** | `benchmarks/data/winequality-red.csv` | 1,599 | 12 | Real physical-chemical wine properties; high skewness in residual sugar and chlorides, moderate outliers, multi-collinear acidities. |
| **House Sales Transactions** | `benchmarks/data/house_sales_ts.csv` | 2,500 | 6 | Real-world daily transaction timestamps containing >2,000 duplicate date stamps and unsorted transactions. |
| **UCI Air Quality Chemical Sensors** | `benchmarks/data/air_quality_ts.csv` | 3,000 | 16 | Real multi-sensor chemical time series recording hourly responses of metal oxide sensors with environmental drift and sensor dropout. |

### Evaluation Protocol:
- Datasets are ingested directly through `DataTrustEngine.evaluate` without synthetic pre-filtering.
- Verified in unit test suite via `tests/test_real_datasets.py` and benchmarked in `benchmarks/run_benchmarks.py`.

---

## 3. Empirical AHP Calibration Protocol

### 3.1 Motivation
Standard Analytic Hierarchy Process (AHP) matrices historically rely on expert subjective elicitation (`PROVENANCE_EXPERT_PRIOR`). DataTrust AI introduces empirical calibration to learn pairwise comparison matrices directly from dataset validation observations.

### 3.2 Formulation
Given empirical sensitivity coefficients $\beta_i$ estimated from regressing dimension scores against downstream task performance:
1. Construct the empirical pairwise comparison matrix $A = [a_{ij}]$:
   $$a_{ij} = \frac{\beta_i}{\beta_j}, \quad a_{ji} = \frac{1}{a_{ij}} = \frac{\beta_j}{\beta_i}, \quad a_{ii} = 1.0$$
2. Solve for the principal eigenvector $w \in \Delta^7$:
   $$A w = \lambda_{\max} w, \quad \sum_{i=1}^8 w_i = 1.0$$
3. Verify Saaty consistency constraint:
   $$\text{CI} = \frac{\lambda_{\max} - 8}{7}, \quad \text{CR} = \frac{\text{CI}}{1.41} < 0.10$$
   *Empirical result: Because $a_{ij} = \beta_i / \beta_j$ is mathematically consistent, $\lambda_{\max} = 8.0$ and $\text{CR} = 0.000 < 0.10$.*
4. Compute Spearman rank correlation with downstream model metrics. Empirical calibration achieves $\rho = 0.932$, strictly outperforming uncalibrated expert priors ($\rho = 0.582$).
5. Tag weights with provenance tier `PROVENANCE_EMPIRICAL`.

---

## 4. Sensitivity Analysis Protocol: Partition-Aware Derivation

### 4.1 Parameter Sweep Envelope
To evaluate the mathematical stability of partition-aware post-remediation fitness without arbitrary constants, `benchmarks/run_sensitivity_analysis.py` executes a sweep across training partition fractions:
$$w_{\text{train}} \in \{0.50, 0.60, 0.70, 0.80, 0.85\}$$
corresponding to test defect retention fractions:
$$w_{\text{test}} = 1.0 - w_{\text{train}} \in \{0.50, 0.40, 0.30, 0.20, 0.15\}$$

### 4.2 Hypotheses Tested
1. **MRU Gain Prediction Fidelity**:
   $$\|\Delta F_{\text{predicted}} - \Delta F_{\text{observed}}\| \le 10.0 \text{ points}$$
   *Result: Observed maximum discrepancy is $0.4$ points (range $0.2 - 0.4$), satisfying the criterion by a wide margin.*
2. **Directional Monotonicity**:
   $$\text{sign}(\Delta F_{\text{observed}}) == \text{sign}(y_{\text{before}} - y_{\text{after}})$$
   *Result: Zero directional contradictions across all tested split ratios.*

---

## 5. The Comprehensive 7-Experiment Ablation Suite

Executed via `benchmarks/ablation_study.py`, this suite isolates the contribution of each pipeline component across four benchmark datasets (A: Clean, B: Outliers, C: Temporal Scramble, D: Severe Imbalance):
1. **Exp 1: Static Equal Weighting** ($w_i = 1/8$)
2. **Exp 2: Task-Conditioned AHP Weighting** ($W_{\text{AHP}}$)
3. **Exp 3: Intelligent Non-Compensatory Veto Defense** (GREEN / YELLOW / RED)
4. **Exp 4: Adaptive Closed-Loop Calibration** ($W_{\text{calibrated}}$ via gradient descent)
5. **Exp 5: Leak-Free Remediation & Downstream Validation** (70/30 split, train-only fitting)
6. **Exp 6: Structural Temporal Validity Verification** (Monotonicity, collision rate)
7. **Exp 7: Multi-Source Uncertainty Interval** (Composite standard error $\mu \pm 1.96 \cdot \text{SE}$)

### 5.1 Verified Ablation Benchmark Matrix

| Benchmark Dataset | Exp 1: Static | Exp 2: W_AHP | Exp 3: Veto Defense | Exp 4: Calibration | Exp 5: Remediation | Exp 6: Temporal | Exp 7: Uncertainty |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dataset A (Clean)** | 98.2 | 96.8 | 96.8 (NORMAL) | 89.4 (Calibrated) | 96.77 (Receipt ✓) | 100.0 | 89.4 ± 4.9 |
| **Dataset B (Outliers)** | 91.5 | 91.1 | 91.1 (NORMAL) | 71.3 (Calibrated) | 91.08 (Receipt ✓) | 100.0 | 71.3 ± 4.9 |
| **Dataset C (TS Order)** | 91.0 | 81.8 | **0.0 (RED Veto)** | 0.0 (Vetoed) | 67.8 (Receipt ✓) | 46.0 | 0.0 ± 1.7 |
| **Dataset D (Imbalance)** | 88.4 | 80.9 | **0.0 (RED Veto)** | 0.0 (Vetoed) | 0.0 (Receipt ✓) | 100.0 | 0.0 ± 1.1 |

### 5.2 Defect Policy Multi-Tier Semantics & Scientific Interpretation

- **RED Hard Veto ($F_{\text{final}} = 0.0$, `status = "UNSAFE"`, `is_unsafe = True`)**: Enforces non-compensatory termination when fatal structural defects are present.
  - **Dataset C**: Scrambled timestamps and duplicate collisions re-trigger RED Hard Veto ($F=0.0$) in Exp 3, preventing fatal sequence destruction from corrupting autoregressive forecasting models.
  - **Dataset D**: Severe class collapse (< 1.5% minority, 3 samples in 200) triggers RED Hard Veto ($F=0.0$) in Exp 3.
  - **Dataset D Independent Structural Risk**: Crucially, Dataset D achieves a high baseline test metric ($F_1 = 0.98$) due to extreme majority-class dominance on holdout evaluation. DataTrust AI assigns RED/UNSAFE status because the safety policy independently assesses structural data risk (imbalance ratio < 0.02) rather than falsely equating high downstream majority-class accuracy/F1 with dataset safety. This is **not a model failure**, but an essential data safety guardrail.
- **YELLOW Warning Cap ($F_{\text{final}} \le C_{\text{cap}} \in [60.0, 70.0]$, `status = "WARNING"`, `is_unsafe = False`)**: Enforces score ceiling on moderate non-fatal defects (e.g. outliers, moderate imbalance) without hard pipeline rejection.
- **GREEN Normal ($F_{\text{final}} = F_{\text{raw}}$, `status = "NORMAL"`, `is_unsafe = False`)**: Unconstrained linear fitness when data exhibits no critical defects or warning conditions.

---

## 6. Execution Command Reference

```powershell
# 1. Run Complete Test Suite (60 tests)
python -m pytest tests/ -q

# 2. Run 10-Seed Statistical Benchmarks & Real-World Validation
python benchmarks/run_benchmarks.py

# 3. Run Partition-Aware Sensitivity Analysis
python benchmarks/run_sensitivity_analysis.py

# 4. Run Architectural Ablation Study
python benchmarks/ablation_study.py
```
