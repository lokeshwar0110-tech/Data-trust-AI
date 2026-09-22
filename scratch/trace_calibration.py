import os, sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.getcwd())
from datatrust.models import TaskType, QualityDimension
from datatrust.engine import DataTrustEngine
from datatrust.calibration import PerformanceCalibrationEngine
from benchmarks.run_main_controlled_experiment import load_benchmark_datasets

datasets = load_benchmark_datasets()
print(f"Tracing {len(datasets)} datasets...")

canon_keys = [dim.value for dim in QualityDimension]

for d in datasets:
    rep = DataTrustEngine.evaluate(
        df=d['df'].copy(),
        task=d['task'],
        target_column=d['target'],
        time_column=d['time']
    )
    cal = rep.calibration
    dp = rep.defect_policy
    w_before = rep.weights_ahp
    w_after = rep.weights_calibrated
    
    deltas = [abs(w_after.get(k, 0.0) - w_before.get(k, 0.0)) for k in canon_keys]
    max_delta = max(deltas) if deltas else 0.0
    sum_before = sum(w_before.get(k, 0.0) for k in canon_keys)
    sum_after = sum(w_after.get(k, 0.0) for k in canon_keys)
    
    print("="*70)
    print(f"Dataset: {d['name']} ({d['task'].value})")
    print(f"Defect Tier: {dp.tier}, Cap Applied: {dp.cap_applied}")
    print(f"Raw Fitness: {rep.raw_fitness}")
    print(f"Pre-Calibration Fitness (v1): {dp.final_fitness}")
    print(f"Calibration Status: {cal.calibration_status if cal else 'None'}")
    print(f"Calibration Provenance: {cal.provenance if cal else 'None'}")
    print(f"Calibrated Trust Score: {cal.calibrated_trust_score if cal else 'None'}")
    print(f"Reported Final Fitness: {rep.final_fitness}")
    print(f"Max Absolute Weight Delta: {max_delta:.4f}")
    print(f"Sum W Before: {sum_before:.4f}, Sum W After: {sum_after:.4f}")
    print(f"Predicted Metric Before: {cal.surrogate_predicted_performance if cal else 'None'}")
    print(f"Predicted Metric After: {cal.calibrated_predicted_performance if cal else 'None'}")
    print(f"Observed Metric: {cal.observed_performance if cal else 'None'}")
