# DataTrust AI

**DataTrust AI** is a decision-support and data validation platform that evaluates whether a dataset is fit for a specific downstream analytical or machine learning task, moving beyond generic, static data quality checks.

---

## The Four Pillars

| Pillar | Location | Description |
| :--- | :--- | :--- |
| **SaaS MVP Application** | [`backend/`](backend/) & [`datatrust/`](datatrust/) | FastAPI REST backend + interactive web dashboard + PDF audit exporter. |
| **Empirical Benchmarks** | [`benchmarks/run_benchmarks.py`](benchmarks/run_benchmarks.py) | Multi-seed benchmark suite demonstrating Spearman rank alignment ($\rho = 0.793 \pm 0.244$ vs $\rho = -0.272 \pm 0.544$ for static profiling, paired t-test $p = 2.89 \times 10^{-5}$) and empirical calibration ($\rho = 0.7167$). |
| **MSc Thesis** | [`docs/thesis/msc_thesis.md`](docs/thesis/msc_thesis.md) | Complete thesis on context-aware MCDM data fitness evaluation. |
| **Research Paper** | [`docs/paper/research_paper.md`](docs/paper/research_paper.md) | Publication-ready paper draft for NeurIPS/SIGMOD/TKDE. |
| **Patent Draft** | [`docs/patent/patent_draft.md`](docs/patent/patent_draft.md) | Formal patent specification and claims covering dynamic task-weighting and veto architecture. |

---

## Quick Start

### 1. Installation
```bash
cd C:\Users\Lenovo\.gemini\antigravity\scratch\datatrust-ai
pip install -e .
```

### 2. Run Test Suite
```bash
pytest tests/ -v -W default
```
All 77 tests verify task inference, AHP Consistency Ratios ($CR < 0.10$), multi-dimensional quality metrics, non-compensatory vetoes, strict test holdout immutability, adaptive calibration, and REST API contracts.

### 3. Launch the SaaS MVP Web Dashboard
```bash
python backend/main.py
```
Or with uvicorn directly:
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
Open your browser at **`http://localhost:8000`** to access:
- File Ingestion (drag-and-drop CSV, Parquet, Excel)
- Task Objective Selector (`Supervised Classification`, `Supervised Regression`, `Time Series Forecasting`, `Descriptive BI`)
- Dynamic Trust Score Gauge (0–100) & Confidence Tier
- Radar Dimension Chart & Factor Attribution Waterfall
- Actionable Remediation Guidance Checklist
- One-Click PDF Audit Report Export

### 4. Run Empirical Benchmarks
```bash
python benchmarks/run_benchmarks.py
```
Demonstrates how DataTrust AI dynamically penalizes temporal leakage and class collapse while preserving scores under benign non-critical missingness.
