import os
import sys
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing, load_diabetes, load_breast_cancer, load_iris

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from datatrust.models import TaskType
from datatrust.engine import DataTrustEngine
from datatrust.task_infer import TaskInferenceEngine

rng = np.random.RandomState(42)

datasets = [
    # A1. California Housing (Regression)
    {
        "id": "A1_cal_housing",
        "name": "California Housing",
        "category": "A. Regression",
        "df": fetch_california_housing(as_frame=True).frame.head(1000), # 1000 for fast eval
        "target": "MedHouseVal",
        "time": None,
        "task": TaskType.SUPERVISED_REGRESSION
    },
    # A2. Diabetes (Regression)
    {
        "id": "A2_diabetes",
        "name": "Diabetes Progression",
        "category": "A. Regression",
        "df": load_diabetes(as_frame=True).frame.rename(columns={"target": "disease_progression"}),
        "target": "disease_progression",
        "time": None,
        "task": TaskType.SUPERVISED_REGRESSION
    },
    # B1. Breast Cancer (Classification)
    {
        "id": "B1_breast_cancer",
        "name": "Breast Cancer Diagnostic",
        "category": "B. Classification",
        "df": load_breast_cancer(as_frame=True).frame.rename(columns={"target": "biopsy_result"}),
        "target": "biopsy_result",
        "time": None,
        "task": TaskType.SUPERVISED_CLASSIFICATION
    },
    # B2. Iris (Classification)
    {
        "id": "B2_iris",
        "name": "Iris Species Morphometrics",
        "category": "B. Classification",
        "df": pd.concat([pd.DataFrame(load_iris().data, columns=load_iris().feature_names), pd.Series(load_iris().target_names[load_iris().target], name="variety")], axis=1),
        "target": "variety",
        "time": None,
        "task": TaskType.SUPERVISED_CLASSIFICATION
    },
    # C1. Melbourne Daily Temperatures (Time Series)
    {
        "id": "C1_melb_temperatures",
        "name": "Melbourne Daily Temperatures",
        "category": "C. Time Series",
        "df": pd.read_csv("benchmarks/data/daily-min-temperatures.csv"),
        "target": "Temp",
        "time": "Date",
        "task": TaskType.TIME_SERIES_FORECASTING
    },
    # C2. Air Quality (Time Series)
    {
        "id": "C2_air_quality",
        "name": "Air Quality Sensors",
        "category": "C. Time Series",
        "df": pd.read_csv("benchmarks/data/air_quality_ts.csv").head(1000),
        "target": "CO(GT)",
        "time": "timestamp",
        "task": TaskType.TIME_SERIES_FORECASTING
    },
    # D1. Unlabeled Coordinates (Clustering)
    {
        "id": "D1_coord_clustering",
        "name": "Unlabeled Spatial Clusters",
        "category": "D. Clustering",
        "df": pd.DataFrame(rng.randn(200, 4), columns=[f"coord_{i}" for i in range(1, 5)]),
        "target": None,
        "time": None,
        "task": TaskType.CLUSTERING
    },
    # D2. Unlabeled Universities (Clustering)
    {
        "id": "D2_univ_clustering",
        "name": "Academic Institution Profiles",
        "category": "D. Clustering",
        "df": pd.read_csv("C:/Users/Lenovo/Clustered_Universities.csv").drop(columns=[c for c in pd.read_csv("C:/Users/Lenovo/Clustered_Universities.csv").columns if 'cluster' in c.lower() or c == 'University']),
        "target": None,
        "time": None,
        "task": TaskType.CLUSTERING
    },
    # E1. Retail Transactions (Transactional with Date)
    {
        "id": "E1_retail_transactions",
        "name": "Retail Store Sales",
        "category": "E. Transaction / Tabular with Date",
        "df": pd.DataFrame({
            "order_date": rng.choice(pd.date_range("2023-01-01", periods=15, freq="D"), 300),
            "store_id": [f"Store_{i%8}" for i in range(300)],
            "category": rng.choice(["Apparel", "Home", "Grocery", "Electronics"], 300),
            "discount_pct": rng.uniform(0.0, 0.40, 300),
            "units": rng.randint(1, 20, 300),
            "sales_amount": rng.uniform(15.0, 850.0, 300)
        }),
        "target": "sales_amount",
        "time": "order_date",
        "task": TaskType.SUPERVISED_REGRESSION
    },
    # F1. Messy Defective Sensor Telemetry (Noisy / Messy)
    {
        "id": "F1_messy_telemetry",
        "name": "Messy Sensor Telemetry",
        "category": "F. Noisy / Messy",
        "df": pd.DataFrame({
            "temp": [25.0 + rng.randn() if rng.rand() > 0.15 else np.nan for _ in range(250)],
            "pressure": [1.0 + rng.randn()*0.1 if rng.rand() > 0.08 else 850.0 for _ in range(250)], # extreme outlier
            "vibration": rng.uniform(0.01, 0.05, 250),
            "failure_status": [rng.choice([0, 1], p=[0.92, 0.08]) for _ in range(250)]
        }),
        "target": "failure_status",
        "time": None,
        "task": TaskType.SUPERVISED_CLASSIFICATION
    }
]

print("Evaluating all 10 datasets...")
for d in datasets:
    df = d["df"]
    target = d["target"]
    time_col = d["time"]
    task = d["task"]
    
    # 1. Autonomous Task Inference
    inf = TaskInferenceEngine.infer_task(df)
    
    # 2. Engine Evaluation
    report = DataTrustEngine.evaluate(
        df=df,
        task=task,
        target_column=target,
        time_column=time_col
    )
    
    cal = report.calibration
    unc = report.fitness_uncertainty
    
    pred_m = report.predicted_metric_value
    obs_m = report.observed_metric_value
    res_m = report.metric_residual
    
    print(f"\n[{d['id']}] {d['name']} ({d['category']})")
    print(f"  Shape: {df.shape}, Inferred: {inf.inferred_task.value} (conf={inf.confidence:.2f}), Target: {inf.detected_target}")
    print(f"  Designated Task: {task.value}, Target: {target}")
    print(f"  Fitness: Raw={report.raw_fitness:.1f}, Final={report.final_fitness}, Status={report.fitness_status}, Tier={report.defect_policy.tier if report.defect_policy else 'GREEN'}")
    print(f"  Validation: Model={report.model_family}, Metric={report.metric}, Pred={pred_m}, Obs={obs_m}, Res={res_m}")
    unc_str = f"[{unc.ci_lower:.1f}, {unc.ci_upper:.1f}] (SE={unc.composite_se:.2f})" if unc else "N/A"
    print(f"  Calibration: {cal.calibration_status if cal else 'N/A'}, Provenance: {cal.provenance if cal else 'N/A'}")
    print(f"  Uncertainty: {unc_str}")
