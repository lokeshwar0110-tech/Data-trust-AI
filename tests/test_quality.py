import pytest
import pandas as pd
import numpy as np
from datatrust.quality_engine import QualityEngine
from datatrust.models import TaskType, QualityDimension


def test_completeness_evaluation():
    # 50 rows, 4 columns = 200 cells
    data = {
        "a": [1.0] * 40 + [np.nan] * 10,
        "b": [2.0] * 50,
        "c": [np.nan] * 50,  # 100% missing column
        "d": [4.0] * 45 + [np.nan] * 5
    }
    df = pd.DataFrame(data)
    engine = QualityEngine(df)
    res = engine.evaluate_completeness()

    assert res.score < 80.0
    assert "c" in res.raw_metrics["completely_empty_cols"]
    assert res.raw_metrics["total_missing_cells"] == 65


def test_temporal_monotonicity_check():
    # Non-monotonic timestamps
    dates = pd.date_range("2024-01-01", periods=20, freq="D").tolist()
    dates[5], dates[6] = dates[6], dates[5]  # swap to break monotonicity

    df = pd.DataFrame({
        "timestamp": dates,
        "value": np.random.randn(20)
    })

    engine = QualityEngine(df, time_column="timestamp")
    res = engine.evaluate_temporal_validity(TaskType.TIME_SERIES_FORECASTING)

    assert res.raw_metrics["is_monotonic_increasing"] is False
    assert res.score < 70.0


def test_outlier_detection():
    # Normal data with 3 extreme anomalies
    np.random.seed(42)
    vals = np.random.normal(10, 2, 100).tolist()
    vals[0] = 99999.0
    vals[1] = -88888.0

    df = pd.DataFrame({"feat1": vals, "feat2": np.random.normal(5, 1, 100)})
    engine = QualityEngine(df)
    res = engine.evaluate_outliers()

    assert len(res.raw_metrics["extreme_kurtosis_cols"]) > 0
    assert res.score < 95.0


def test_configurable_imbalance_policy():
    from datatrust.models import ImbalancePolicy

    # Create dataset with 4% minority
    # Under standard 2% threshold, 4% is NOT vetoed.
    # Under a strict 5% threshold, 4% IS vetoed.
    n = 100
    df = pd.DataFrame({
        "feat": np.random.randn(n),
        "target": [0] * 96 + [1] * 4  # 4.0%
    })

    policy_lenient = ImbalancePolicy(minority_threshold_pct=2.0, veto_enabled=True)
    engine_lenient = QualityEngine(df, target_column="target", imbalance_policy=policy_lenient)
    res_lenient = engine_lenient.evaluate_bias_imbalance(TaskType.SUPERVISED_CLASSIFICATION)
    assert res_lenient.is_veto_triggered is False

    policy_strict = ImbalancePolicy(minority_threshold_pct=5.0, veto_enabled=True)
    engine_strict = QualityEngine(df, target_column="target", imbalance_policy=policy_strict)
    res_strict = engine_strict.evaluate_bias_imbalance(TaskType.SUPERVISED_CLASSIFICATION)
    assert res_strict.is_veto_triggered is True
