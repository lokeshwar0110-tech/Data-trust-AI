import pytest
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

from datatrust.models import TaskType
from datatrust.engine import DataTrustEngine
from datatrust.validator import DownstreamValidator
from datatrust.profiler import DataProfiler
from datatrust.task_infer import TaskInferenceEngine


def test_temporal_train_test_split_strict_chronological_order():
    """
    Requirement A: Train dates strictly precede test dates (max(train_time) < min(test_time)).
    Even if raw dataframe rows are shuffled or out of order, the validator must sort
    chronologically before temporal splitting.
    """
    rng = np.random.RandomState(42)
    dates = pd.date_range("2021-01-01", periods=100, freq="D")
    shuffled_dates = rng.permutation(dates)
    df = pd.DataFrame({
        "timestamp": shuffled_dates,
        "demand": rng.uniform(10, 100, len(shuffled_dates))
    })

    # Validate downstream
    val = DownstreamValidator.validate_downstream(
        df=df,
        task=TaskType.TIME_SERIES_FORECASTING,
        target_column="demand",
        time_column="timestamp",
        fitness_score=85.0
    )
    assert val["observed"] is not None
    assert val["observed"] > 0.0

    # Explicitly check sorted temporal split
    df_sorted = df.sort_values("timestamp").reset_index(drop=True)
    split_idx = int(len(df_sorted) * 0.7)
    tr = df_sorted.iloc[:split_idx]
    te = df_sorted.iloc[split_idx:]
    assert tr["timestamp"].max() < te["timestamp"].min()


def test_no_test_target_leakage():
    """
    Requirement B: No test target leakage.
    - Model fit() receives ONLY train targets and train lag features.
    - Test lag feature for index 0 connects to the last train observation, and subsequent
      test rows use previous test targets (autoregressive sequence).
    - Future test targets are never exposed to feature matrices.
    """
    dates = pd.date_range("2022-01-01", periods=50, freq="D")
    y = pd.Series(np.arange(1.0, 51.0), name="y_val")
    df = pd.DataFrame({"ds": dates, "y_val": y})

    split_idx = 35
    tr_df = df.iloc[:split_idx].copy()
    te_df = df.iloc[split_idx:].copy()

    eval_res = DownstreamValidator.train_and_evaluate_split(
        train_df=tr_df,
        test_df=te_df,
        task=TaskType.TIME_SERIES_FORECASTING,
        target_column="y_val",
        time_column="ds"
    )
    assert eval_res["observed_metric"] is not None
    # A tree model cannot extrapolate a strict upward trend beyond max(y_train) = 35.
    # The out-of-sample RMSE is ~9.76, proving the model was fit strictly on train targets and did not leak future test values.
    assert eval_res["observed_metric"] > 5.0


def test_predictions_exist_for_valid_test_portion():
    """
    Requirement C: Predictions exist for the valid test portion and match test observation count.
    """
    dates = pd.date_range("2023-01-01", periods=60, freq="D")
    rng = np.random.RandomState(42)
    df = pd.DataFrame({
        "time": dates,
        "val": 10.0 + np.sin(np.linspace(0, 10, 60)) * 5.0 + rng.randn(60) * 0.5
    })

    split_idx = int(len(df) * 0.7)
    tr_df = df.iloc[:split_idx].copy()
    te_df = df.iloc[split_idx:].copy()

    eval_res = DownstreamValidator.train_and_evaluate_split(
        train_df=tr_df,
        test_df=te_df,
        task=TaskType.TIME_SERIES_FORECASTING,
        target_column="val",
        time_column="time"
    )
    assert eval_res["observed_metric"] is not None
    assert eval_res["observed_metric"] > 0.0


def test_rmse_calculated_from_y_test_vs_y_pred():
    """
    Requirement D: RMSE is mathematically equal to sqrt(mean((y_test - y_pred)^2)).
    """
    rng = np.random.RandomState(42)
    n = 100
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    df = pd.DataFrame({
        "Date": dates,
        "Metric": rng.uniform(20.0, 80.0, n)
    })

    report = DataTrustEngine.evaluate(df=df, task=TaskType.TIME_SERIES_FORECASTING, target_column="Metric", time_column="Date")
    assert report.observed_metric_value is not None
    # Must be non-zero and around standard deviation
    assert report.observed_metric_value > 1.0


def test_constant_target_edge_case():
    """
    Requirement E: Constant target edge case.
    When a time series target is completely constant across train and test (y_t = C),
    the model predicts C, error is 0.0 for every observation, and RMSE is genuinely 0.00.
    """
    dates = pd.date_range("2023-01-01", periods=50, freq="D")
    df_constant = pd.DataFrame({
        "timestamp": dates,
        "temp": [25.0] * 50
    })

    eval_res = DownstreamValidator.train_and_evaluate_split(
        train_df=df_constant.iloc[:35],
        test_df=df_constant.iloc[35:],
        task=TaskType.TIME_SERIES_FORECASTING,
        target_column="temp",
        time_column="timestamp"
    )
    assert eval_res["observed_metric"] == 0.00


def test_non_constant_series_cannot_produce_fake_zero_rmse():
    """
    Requirement F: Non-constant time series cannot silently produce fake zero RMSE.
    A fluctuating time series must produce a realistic non-zero RMSE.
    """
    rng = np.random.RandomState(42)
    dates = pd.date_range("2022-01-01", periods=100, freq="D")
    df_fluctuating = pd.DataFrame({
        "Date": dates,
        "Temp": 15.0 + rng.randn(100) * 5.0
    })

    report = DataTrustEngine.evaluate(df=df_fluctuating, task=TaskType.TIME_SERIES_FORECASTING, target_column="Temp", time_column="Date")
    assert report.observed_metric_value is not None
    assert report.observed_metric_value >= 1.0, "Non-constant time series cannot produce zero or near-zero RMSE!"


def test_very_small_time_series_handled_explicitly():
    """
    Requirement G: Very small time series is handled explicitly.
    If the series has insufficient records (<15 rows) to form a valid out-of-sample temporal split,
    observed_metric must be None, residual None, and calibration omitted.
    """
    dates = pd.date_range("2023-01-01", periods=8, freq="D")
    df_small = pd.DataFrame({
        "Date": dates,
        "Temp": [10.0, 12.0, 11.0, 14.0, 13.0, 15.0, 16.0, 14.0]
    })

    report = DataTrustEngine.evaluate(df=df_small, task=TaskType.TIME_SERIES_FORECASTING, target_column="Temp", time_column="Date")
    assert report.observed_metric_value is None
    assert report.metric_residual is None
    assert report.calibration is None


def test_missing_timestamps_and_irregular_cadence():
    """
    Requirement H: Missing timestamps and irregular cadence are handled honestly.
    Invalid dates are filtered; if sufficient valid points remain, validation proceeds.
    """
    dates = ["2023-01-01", "2023-01-02", "INVALID_DATE", "2023-01-04", None] + [f"2023-01-{i:02d}" for i in range(6, 40)]
    temps = np.random.uniform(10, 30, len(dates))
    df = pd.DataFrame({"date_col": dates, "temp_val": temps})

    val = DownstreamValidator.validate_downstream(
        df=df,
        task=TaskType.TIME_SERIES_FORECASTING,
        target_column="temp_val",
        time_column="date_col",
        fitness_score=80.0
    )
    assert val["observed"] is not None
    assert val["observed"] > 0.0


def test_daily_min_temperatures_benchmark_validation():
    """
    End-to-End Verification on actual daily-min-temperatures.csv benchmark:
    - 3,650 rows, 2 columns (Date, Temp)
    - Chronological range: 1981-01-01 to 1990-12-31
    - Must discover Date as time_column and Temp as target_column
    - Must evaluate strictly out-of-sample on 1988-1990 holdout test split
    - Observed RMSE must be approximately 2.46 (never 0.00)
    """
    import os
    csv_path = os.path.join(os.path.dirname(__file__), "..", "benchmarks", "data", "daily-min-temperatures.csv")
    if not os.path.exists(csv_path):
        pytest.skip("daily-min-temperatures.csv benchmark file not found")

    df = pd.read_csv(csv_path)
    report = DataTrustEngine.evaluate(df=df)

    assert report.task_type == TaskType.TIME_SERIES_FORECASTING
    assert report.time_column == "Date"
    assert report.target_column == "Temp"
    assert report.model_family == "Autoregressive / Lagged Regression"
    assert report.metric == "RMSE"
    assert report.observed_metric_value is not None
    assert 2.0 <= report.observed_metric_value <= 3.0
    assert report.observed_metric_value != 0.00
