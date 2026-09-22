import os, sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.getcwd())
from datatrust.models import TaskType, QualityDimension
from datatrust.engine import DataTrustEngine
from datatrust.remediator import DataRemediator
from datatrust.reproducibility import hash_index
from benchmarks.run_main_controlled_experiment import load_benchmark_datasets, evaluate_controlled_split

datasets = load_benchmark_datasets()
print(f"Tracing remediation on {len(datasets)} benchmark datasets...")

for d in datasets:
    d_name = d["name"]
    task = d["task"]
    target = d["target"]
    time_col = d["time"]
    df = d["df"].copy()
    
    # 1. Evaluate DataTrust report on raw dataset
    rep = DataTrustEngine.evaluate(df=df, task=task, target_column=target, time_column=time_col)
    raw_fit = rep.raw_fitness
    final_fit = rep.final_fitness
    opt_actions = [a.action_id for a in rep.optimized_remediations] if rep.optimized_remediations else []
    
    # 2. Downstream metric before remediation (seed 42)
    obs_metric_before, _, _, tr_idx, te_idx = evaluate_controlled_split(
        df=df, task=task, target_column=target, time_column=time_col, seed=42
    )
    
    # Split train/test exactly as in evaluate_controlled_split
    clean_sub = df.dropna(subset=[target]).copy() if target else df.copy()
    if task == TaskType.TIME_SERIES_FORECASTING and time_col and time_col in clean_sub.columns:
        t_parsed = pd.to_datetime(clean_sub[time_col], errors='coerce')
        valid_mask = t_parsed.notna()
        clean_sub = clean_sub.loc[valid_mask].copy()
        clean_sub['__t__'] = t_parsed.loc[valid_mask]
        clean_sub = clean_sub.sort_values('__t__').reset_index(drop=True)
        split_idx = int(len(clean_sub) * 0.70)
        train_df = clean_sub.iloc[:split_idx].copy().drop(columns=['__t__'])
        test_df = clean_sub.iloc[split_idx:].copy().drop(columns=['__t__'])
    else:
        df_shuffled = clean_sub.sample(frac=1.0, random_state=42).reset_index(drop=True)
        split_idx = int(len(df_shuffled) * 0.70)
        train_df = df_shuffled.iloc[:split_idx].copy()
        test_df = df_shuffled.iloc[split_idx:].copy()
        
    hash_before = hash_index(test_df.index)
    
    # Apply remediation
    rem_tr, rem_te, log, fitted = DataRemediator.fit_and_transform_splits(
        train_df=train_df,
        test_df=test_df,
        task=task,
        action_ids=opt_actions if opt_actions else None,
        target_column=target,
        time_column=time_col
    )
    
    hash_after = hash_index(rem_te.index)
    test_rows_modified = int(len(test_df) - len(rem_te))
    train_rows_modified = int(abs(len(train_df) - len(rem_tr)))
    
    print("="*70)
    print(f"Dataset: {d_name} ({task.value})")
    print(f"Optimized Recommended Actions: {opt_actions}")
    print(f"Executed Actions in Log: {[l for l in log if 'EXECUTED' in l or 'TRAIN:' in l]}")
    print(f"Raw Fitness: {raw_fit} -> Final Fitness: {final_fit}")
    print(f"Observed Metric (Seed 42): {obs_metric_before}")
    print(f"Train Rows Modified: {train_rows_modified}")
    print(f"Test Rows Modified: {test_rows_modified} (Must be 0)")
    print(f"Holdout Hash Match: {hash_before == hash_after} ({hash_before[:8]}... == {hash_after[:8]}...)")
