import os
import sys
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing, load_diabetes, load_breast_cancer, load_iris

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from datatrust.task_infer import TaskInferenceEngine
from datatrust.engine import DataTrustEngine
from datatrust.models import TaskType

# 1. Cal Housing
cal = fetch_california_housing(as_frame=True).frame
res = TaskInferenceEngine.infer_task(cal)
print("Cal Housing -> Inferred:", res.inferred_task, "Target:", res.detected_target, "Conf:", res.confidence)

# 2. Diabetes with non-keyword target 'disease_progression'
diab = load_diabetes(as_frame=True).frame.rename(columns={"target": "disease_progression"})
res = TaskInferenceEngine.infer_task(diab)
print("Diabetes -> Inferred:", res.inferred_task, "Target:", res.detected_target, "Conf:", res.confidence)

# 3. Breast Cancer with non-keyword target 'biopsy_result'
bc = load_breast_cancer(as_frame=True).frame.rename(columns={"target": "biopsy_result"})
res = TaskInferenceEngine.infer_task(bc)
print("Breast Cancer -> Inferred:", res.inferred_task, "Target:", res.detected_target, "Conf:", res.confidence)

# 4. Iris with categorical species labels
iris_raw = load_iris()
iris = pd.DataFrame(iris_raw.data, columns=iris_raw.feature_names)
iris['variety'] = iris_raw.target_names[iris_raw.target]
res = TaskInferenceEngine.infer_task(iris)
print("Iris -> Inferred:", res.inferred_task, "Target:", res.detected_target, "Conf:", res.confidence)

# 5. Melbourne Daily Temperatures
melb = pd.read_csv("benchmarks/data/daily-min-temperatures.csv")
res = TaskInferenceEngine.infer_task(melb)
print("Melbourne TS -> Inferred:", res.inferred_task, "Target:", res.detected_target, "Time:", res.candidate_timestamps[0] if res.candidate_timestamps else None, "Conf:", res.confidence)

# 6. Air Quality TS
air = pd.read_csv("benchmarks/data/air_quality_ts.csv")
res = TaskInferenceEngine.infer_task(air)
print("Air Quality TS -> Inferred:", res.inferred_task, "Target:", res.detected_target, "Time:", res.candidate_timestamps[0] if res.candidate_timestamps else None, "Conf:", res.confidence)

# 7. Unlabeled Synthetic Coordinates
rng = np.random.RandomState(42)
df_coords = pd.DataFrame(rng.randn(150, 5), columns=[f"coord_{i}" for i in range(1, 6)])
res = TaskInferenceEngine.infer_task(df_coords)
print("Coordinates Clustering -> Inferred:", res.inferred_task, "Target:", res.detected_target, "Conf:", res.confidence)

# 8. Unlabeled Universities
univ = pd.read_csv("C:/Users/Lenovo/Clustered_Universities.csv").drop(columns=[c for c in pd.read_csv("C:/Users/Lenovo/Clustered_Universities.csv").columns if 'cluster' in c.lower()])
res = TaskInferenceEngine.infer_task(univ)
print("Universities -> Inferred:", res.inferred_task, "Target:", res.detected_target, "Conf:", res.confidence)

# 9. Retail Transactions with Dates
dates = pd.date_range("2023-01-01", periods=15, freq="D")
n = 300
df_retail = pd.DataFrame({
    "order_date": rng.choice(dates, n),
    "store_id": [f"Store_{i%10}" for i in range(n)],
    "category": rng.choice(["Apparel", "Home", "Grocery", "Electronics"], n),
    "discount_pct": rng.uniform(0.0, 0.40, n),
    "units": rng.randint(1, 20, n),
    "sales_amount": rng.uniform(15.0, 850.0, n)
})
res = TaskInferenceEngine.infer_task(df_retail)
print("Retail Transactions -> Inferred:", res.inferred_task, "Target:", res.detected_target, "Time:", res.candidate_timestamps[0] if res.candidate_timestamps else None, "Conf:", res.confidence)

# 10. Messy Sensor Log
df_messy = pd.DataFrame({
    "temp": [25.0 + rng.randn() if rng.rand() > 0.15 else np.nan for _ in range(200)],
    "pressure": [1.0 + rng.randn()*0.1 if rng.rand() > 0.10 else 999.0 for _ in range(200)], # extreme outlier
    "rpm": [1500 + rng.randint(-100, 100) if rng.rand() > 0.05 else np.nan for _ in range(200)],
    "vibration": rng.uniform(0.01, 0.05, 200),
    "failure_status": [rng.choice([0, 1], p=[0.9, 0.1]) for _ in range(200)]
})
res = TaskInferenceEngine.infer_task(df_messy)
print("Messy Sensor -> Inferred:", res.inferred_task, "Target:", res.detected_target, "Conf:", res.confidence)
