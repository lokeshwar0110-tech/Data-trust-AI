import os
import pytest
import pandas as pd
from datatrust.models import TaskType, QualityDimension
from datatrust.engine import DataTrustEngine

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "benchmarks", "data")


def test_real_wine_quality_assessment():
    """Issue 2: Verify 8-dimensional assessment on authentic UCI Red Wine data."""
    wine_path = os.path.join(DATA_DIR, "winequality-red.csv")
    assert os.path.exists(wine_path), f"Missing real dataset: {wine_path}"
    df = pd.read_csv(wine_path)
    assert len(df) == 1599
    assert "quality" in df.columns

    report = DataTrustEngine.evaluate(
        df=df,
        task=TaskType.SUPERVISED_REGRESSION,
        target_column="quality",
        dataset_name="UCI_Red_Wine_Real"
    )

    # 1. 8 Canonical dimensions present
    assert len(report.dimensions) == 8
    assert report.raw_fitness > 0.0
    fit = report.final_fitness if report.final_fitness is not None else report.raw_fitness
    assert 0.0 <= fit <= 100.0

    # 2. Validity is high (clean tabular measurements)
    val_score = report.dimensions[QualityDimension.VALIDITY.value].score
    assert val_score > 70.0

    # 3. Defect policy reflects distribution characteristics
    assert report.defect_policy is not None


def test_real_house_sales_time_series_veto():
    """Issue 2: Verify temporal defect policy / veto triggers on real transaction timestamps."""
    house_path = os.path.join(DATA_DIR, "house_sales_ts.csv")
    assert os.path.exists(house_path), f"Missing real dataset: {house_path}"
    df = pd.read_csv(house_path)
    assert len(df) >= 1000
    assert "Datesold" in df.columns
    assert "Price" in df.columns

    report = DataTrustEngine.evaluate(
        df=df,
        task=TaskType.TIME_SERIES_FORECASTING,
        target_column="Price",
        time_column="Datesold",
        dataset_name="House_Sales_Real_TS"
    )

    # In real estate transaction data, multiple sales occur on the same date -> collisions & non-monotonicity
    timeliness_score = report.dimensions[QualityDimension.TIMELINESS.value].score
    assert timeliness_score < 50.0, f"Expected low timeliness due to duplicate dates, got {timeliness_score}"

    # Defect policy must trigger RED or YELLOW veto
    assert report.defect_policy is not None
    assert report.defect_policy.tier in ["RED", "YELLOW"]
    fit = report.final_fitness if report.final_fitness is not None else 0.0
    assert fit < report.raw_fitness or fit == 0.0


def test_real_air_quality_sensor_pipeline():
    """Issue 2: Verify multi-dimensional evaluation on UCI chemical sensor time-series."""
    air_path = os.path.join(DATA_DIR, "air_quality_ts.csv")
    assert os.path.exists(air_path), f"Missing real dataset: {air_path}"
    df = pd.read_csv(air_path)
    assert len(df) >= 1000
    assert "timestamp" in df.columns
    assert "CO(GT)" in df.columns

    report = DataTrustEngine.evaluate(
        df=df,
        task=TaskType.TIME_SERIES_FORECASTING,
        target_column="CO(GT)",
        time_column="timestamp",
        dataset_name="UCI_Air_Quality_Real_TS"
    )

    assert len(report.dimensions) == 8
    # Air quality has valid monotonic hourly intervals
    timeliness_score = report.dimensions[QualityDimension.TIMELINESS.value].score
    assert timeliness_score > 60.0
    fit = report.final_fitness if report.final_fitness is not None else report.raw_fitness
    assert fit > 50.0
