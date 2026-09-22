import os
import sys
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing, load_diabetes, load_breast_cancer, load_iris

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from datatrust.task_infer import TaskInferenceEngine
from datatrust.profiler import DataProfiler

print("--- California Housing ---")
cal = fetch_california_housing(as_frame=True).frame
res_cal = TaskInferenceEngine.infer_task(cal)
print("Cal Housing Inferred:", res_cal.inferred_task, res_cal.confidence, "P(T|D):", res_cal.p_task_given_data)
candidates_cal = DataProfiler.find_candidate_target_details(cal)
print("Candidates:", candidates_cal[:3])

print("\n--- Breast Cancer ---")
bc = load_breast_cancer(as_frame=True).frame
# Rename target to 'diagnostic_outcome' to test non-keyword discovery
bc_renamed = bc.rename(columns={"target": "diagnostic_outcome"})
res_bc = TaskInferenceEngine.infer_task(bc_renamed)
print("BC Inferred:", res_bc.inferred_task, res_bc.confidence, "P(T|D):", res_bc.p_task_given_data)
candidates_bc = DataProfiler.find_candidate_target_details(bc_renamed)
print("Candidates:", candidates_bc[:3])

print("\n--- Universities (Unlabeled Clustering) ---")
univ = pd.read_csv("C:/Users/Lenovo/Clustered_Universities.csv")
# Note: Clustered_Universities.csv has 'Cluster' column. Let's drop it to test purely unlabeled clustering!
univ_unlabeled = univ.drop(columns=[c for c in univ.columns if 'cluster' in c.lower()])
res_univ = TaskInferenceEngine.infer_task(univ_unlabeled)
print("Univ Inferred:", res_univ.inferred_task, res_univ.confidence, "P(T|D):", res_univ.p_task_given_data)
candidates_univ = DataProfiler.find_candidate_target_details(univ_unlabeled)
print("Candidates:", candidates_univ[:3])

print("\n--- Melbourne Daily Temperatures ---")
melb = pd.read_csv("benchmarks/data/daily-min-temperatures.csv")
res_melb = TaskInferenceEngine.infer_task(melb)
print("Melb Inferred:", res_melb.inferred_task, res_melb.confidence, "P(T|D):", res_melb.p_task_given_data)
candidates_melb = DataProfiler.find_candidate_target_details(melb)
print("Candidates:", candidates_melb[:3])
