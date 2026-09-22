import pytest
import pandas as pd
import numpy as np
from datatrust.engine import DataTrustEngine
from datatrust.models import TaskType
from datatrust.reporting import ReportGenerator


def test_full_evaluation_pipeline_and_pdf():
    # Synthetic tabular dataset
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    df = pd.DataFrame({
        "timestamp": dates,
        "feature_a": np.random.normal(0, 1, n),
        "feature_b": np.random.uniform(10, 50, n),
        "category": np.random.choice(["X", "Y", "Z"], n),
        "target": np.random.choice([0, 1], n, p=[0.6, 0.4])
    })

    # Add 5 missing values
    df.loc[10:14, "feature_a"] = np.nan

    report = DataTrustEngine.evaluate(
        df=df,
        task=TaskType.SUPERVISED_CLASSIFICATION,
        dataset_name="TestCreditRisk",
        target_column="target",
        time_column="timestamp"
    )

    assert report.trust_score > 60.0
    assert report.confidence_tier in ["High Fitness / Low Risk", "Moderate Fitness / Conditional"]
    assert len(report.factor_attributions) == 8
    assert report.ahp_consistency_ratio < 0.10

    # Test PDF generation
    pdf_bytes = ReportGenerator.generate_pdf(report)
    assert len(pdf_bytes) > 1000
    assert pdf_bytes.startswith(b"%PDF")
