"""
DataTrust AI Empirical Benchmark Suite (Multi-Seed & Real-World Validation)
Evaluates correlation between DataTrust Task-Aware Scores vs Static Profiling Scores
against true downstream model performance degradation (F1-score / RMSE) across 10 random seeds,
and evaluates real-world datasets in a dedicated separate benchmark section.
"""

import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon, spearmanr
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, mean_squared_error

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datatrust.engine import DataTrustEngine
from datatrust.models import TaskType, QualityDimension
from datatrust.weighting import AHPWeightingEngine

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def generate_benchmark_scenarios(n=500, seed=42):
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    X1 = rng.normal(0, 1, n)
    X2 = rng.normal(5, 2, n)
    X3 = rng.uniform(10, 50, n)
    
    # Classification target: non-linear boundary
    y_clf = (X1 * 1.5 + X2 * 0.8 + rng.normal(0, 0.5, n) > 4.0).astype(int)
    
    # Time-series target: autoregressive + trend + seasonality
    trend = np.linspace(0, 10, n)
    y_ts = np.zeros(n)
    for t in range(1, n):
        y_ts[t] = 0.7 * y_ts[t-1] + trend[t] + rng.normal(0, 1)

    clean_df = pd.DataFrame({
        "timestamp": dates,
        "x1": X1,
        "x2": X2,
        "x3": X3,
        "y_clf": y_clf,
        "y_ts": y_ts
    })

    scenarios = {}
    scenarios["clean"] = clean_df.copy()

    # Scenario 2: Moderate Missingness in non-critical X3
    df_missing = clean_df.copy()
    missing_indices = rng.choice(n, int(n * 0.35), replace=False)
    df_missing.loc[missing_indices, "x3"] = np.nan
    scenarios["missing_non_critical"] = df_missing

    # Scenario 3: Severe Temporal Leakage / Disorder
    df_leak = clean_df.copy()
    scrambled_idx = rng.permutation(n)
    df_leak["timestamp"] = df_leak["timestamp"].iloc[scrambled_idx].values
    scenarios["temporal_disorder"] = df_leak

    # Scenario 4: Severe Class Imbalance (<2% positive)
    df_imbalance = clean_df.copy()
    pos_indices = df_imbalance[df_imbalance["y_clf"] == 1].index
    flip_indices = rng.choice(pos_indices, int(len(pos_indices) * 0.92), replace=False)
    df_imbalance.loc[flip_indices, "y_clf"] = 0
    scenarios["severe_imbalance"] = df_imbalance

    return scenarios


def evaluate_downstream_models(df: pd.DataFrame, seed=42):
    clean_sub = df.dropna()
    results = {}

    # 1. Classification F1
    try:
        X = clean_sub[["x1", "x2"]]
        y = clean_sub["y_clf"]
        if len(np.unique(y)) > 1:
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=seed)
            clf = RandomForestClassifier(random_state=seed)
            clf.fit(X_tr, y_tr)
            preds = clf.predict(X_te)
            results["clf_f1"] = round(float(f1_score(y_te, preds, zero_division=0)), 3)
        else:
            results["clf_f1"] = 0.0
    except Exception:
        results["clf_f1"] = 0.0

    # 2. Time-Series RMSE
    try:
        df_sorted = clean_sub.sort_values("timestamp")
        split_point = int(len(df_sorted) * 0.7)
        train_ts = df_sorted.iloc[:split_point]
        test_ts = df_sorted.iloc[split_point:]

        X_tr = train_ts[["x1", "x2"]]
        y_tr = train_ts["y_ts"]
        X_te = test_ts[["x1", "x2"]]
        y_te = test_ts["y_ts"]

        reg = RandomForestRegressor(random_state=seed)
        reg.fit(X_tr, y_tr)
        preds = reg.predict(X_te)
        rmse = np.sqrt(mean_squared_error(y_te, preds))
        results["ts_rmse"] = round(float(rmse), 3)
    except Exception:
        results["ts_rmse"] = 999.0

    return results


def run_multi_seed_benchmarks(seeds=range(42, 52)):
    print("\n" + "=" * 95)
    print(f"PART 1: MULTI-SEED STATISTICAL SIGNIFICANCE BENCHMARKS ({len(seeds)} SEEDS: {min(seeds)}..{max(seeds)})")
    print("=" * 95)

    all_seed_results = []
    comparison_pairs = {
        "dt_vs_static_clf": {"dt": [], "static": []},
        "dt_vs_static_ts": {"dt": [], "static": []},
        "dt_with_veto_vs_no_veto": {"veto": [], "no_veto": []},
    }

    spearman_rhos = {"datatrust": [], "static": []}

    for s in seeds:
        scenarios = generate_benchmark_scenarios(n=500, seed=s)
        f1_list = []
        rmse_list = []
        dt_clf_list = []
        dt_ts_list = []
        static_list = []

        for name, df in scenarios.items():
            perf = evaluate_downstream_models(df, seed=s)
            missing_rate = df.isna().sum().sum() / (df.shape[0] * df.shape[1])
            static_score = round(max(0.0, 100.0 - (missing_rate * 200)), 1)

            rep_clf = DataTrustEngine.evaluate(
                df=df,
                task=TaskType.SUPERVISED_CLASSIFICATION,
                dataset_name=name,
                target_column="y_clf"
            )
            rep_ts = DataTrustEngine.evaluate(
                df=df,
                task=TaskType.TIME_SERIES_FORECASTING,
                dataset_name=name,
                target_column="y_ts",
                time_column="timestamp"
            )

            fit_clf = rep_clf.final_fitness if rep_clf.final_fitness is not None else rep_clf.raw_fitness
            fit_ts = rep_ts.final_fitness if rep_ts.final_fitness is not None else rep_ts.raw_fitness

            all_seed_results.append({
                "seed": s,
                "scenario": name,
                "clf_f1": perf["clf_f1"],
                "ts_rmse": perf["ts_rmse"],
                "static_score": static_score,
                "dt_clf_fitness": fit_clf,
                "dt_clf_raw": rep_clf.raw_fitness,
                "dt_ts_fitness": fit_ts,
                "dt_ts_raw": rep_ts.raw_fitness
            })

            f1_list.append(perf["clf_f1"])
            rmse_list.append(perf["ts_rmse"])
            dt_clf_list.append(fit_clf)
            dt_ts_list.append(fit_ts)
            static_list.append(static_score)

            comparison_pairs["dt_vs_static_clf"]["dt"].append(fit_clf)
            comparison_pairs["dt_vs_static_clf"]["static"].append(static_score)
            comparison_pairs["dt_vs_static_ts"]["dt"].append(fit_ts)
            comparison_pairs["dt_vs_static_ts"]["static"].append(static_score)
            comparison_pairs["dt_with_veto_vs_no_veto"]["veto"].append(fit_ts)
            comparison_pairs["dt_with_veto_vs_no_veto"]["no_veto"].append(rep_ts.raw_fitness)

        # Correlation per seed: alignment of score with classification F1
        rho_dt, _ = spearmanr(dt_clf_list, f1_list)
        rho_st, _ = spearmanr(static_list, f1_list)
        if not np.isnan(rho_dt):
            spearman_rhos["datatrust"].append(rho_dt)
        if not np.isnan(rho_st):
            spearman_rhos["static"].append(rho_st)

    df_res = pd.DataFrame(all_seed_results)

    # Compute Summary Aggregations
    summary = df_res.groupby("scenario").agg({
        "clf_f1": ["mean", "std"],
        "ts_rmse": ["mean", "std"],
        "static_score": ["mean", "std"],
        "dt_clf_fitness": ["mean", "std"],
        "dt_ts_fitness": ["mean", "std"]
    }).round(2)

    print("\nBenchmark Metrics Across 10 Seeds (Mean +/- Std):")
    print(summary.to_string())

    # Statistical Significance Testing (Paired t-test and Wilcoxon)
    print("\n" + "-" * 95)
    print("HYPOTHESIS TESTING & STATISTICAL SIGNIFICANCE (p < 0.05)")
    print("-" * 95)

    # 1. DataTrust vs Static Correlation with downstream performance
    dt_rhos = np.array(spearman_rhos["datatrust"])
    st_rhos = np.array(spearman_rhos["static"])
    t_stat_rho, p_val_rho = ttest_rel(dt_rhos, st_rhos)
    w_stat_rho, p_wilc_rho = wilcoxon(dt_rhos, st_rhos)
    print(f"Spearman Alignment with Downstream F1:")
    print(f"  DataTrust rho: {np.mean(dt_rhos):.3f} +/- {np.std(dt_rhos):.3f}")
    print(f"  Static rho:    {np.mean(st_rhos):.3f} +/- {np.std(st_rhos):.3f}")
    print(f"  Paired t-test: t = {t_stat_rho:.3f}, p = {p_val_rho:.4e} (p < 0.05: {p_val_rho < 0.05})")
    print(f"  Wilcoxon test: W = {w_stat_rho:.3f}, p = {p_wilc_rho:.4e} (p < 0.05: {p_wilc_rho < 0.05})")

    # 2. DataTrust with Veto vs Without Veto under catastrophic temporal defect
    ts_veto = df_res[df_res["scenario"] == "temporal_disorder"]["dt_ts_fitness"]
    ts_no_veto = df_res[df_res["scenario"] == "temporal_disorder"]["dt_ts_raw"]
    t_stat_v, p_val_v = ttest_rel(ts_veto, ts_no_veto)
    print(f"\nNon-Compensatory Veto Circuit Breaker (Temporal Disorder Scenario):")
    print(f"  With Veto Fitness:    {ts_veto.mean():.2f} +/- {ts_veto.std():.2f} (Hard Ceiling Enforced)")
    print(f"  Without Veto (Raw):   {ts_no_veto.mean():.2f} +/- {ts_no_veto.std():.2f} (Masked Vulnerability)")
    print(f"  Paired t-test: t = {t_stat_v:.3f}, p = {p_val_v:.4e} (p < 0.05: {p_val_v < 0.05})")

    return df_res, {
        "rho_dt_mean": float(np.mean(dt_rhos)),
        "rho_st_mean": float(np.mean(st_rhos)),
        "p_val_rho": float(p_val_rho),
        "p_val_veto": float(p_val_v)
    }


def run_real_world_benchmarks():
    print("\n" + "=" * 95)
    print("PART 2: REAL-WORLD DATASET VALIDATION (AUTHENTIC BENCHMARK FIXTURES)")
    print("=" * 95)

    real_datasets = [
        {
            "name": "UCI Red Wine Quality",
            "file": "winequality-red.csv",
            "task": TaskType.SUPERVISED_REGRESSION,
            "target": "quality",
            "time_col": None,
            "domain": "Enological Physical-Chemical Profiling"
        },
        {
            "name": "House Sales Transactions",
            "file": "house_sales_ts.csv",
            "task": TaskType.TIME_SERIES_FORECASTING,
            "target": "Price",
            "time_col": "Datesold",
            "domain": "Real Estate Real-World Transaction Series (Duplicate Timestamps)"
        },
        {
            "name": "UCI Air Quality Sensors",
            "file": "air_quality_ts.csv",
            "task": TaskType.TIME_SERIES_FORECASTING,
            "target": "CO(GT)",
            "time_col": "timestamp",
            "domain": "Environmental Chemical Metal-Oxide Sensor Network"
        }
    ]

    real_results = []
    for r in real_datasets:
        path = os.path.join(DATA_DIR, r["file"])
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        report = DataTrustEngine.evaluate(
            df=df,
            task=r["task"],
            target_column=r["target"],
            time_column=r["time_col"],
            dataset_name=r["name"]
        )

        fit = report.final_fitness if report.final_fitness is not None else report.raw_fitness
        tier = report.defect_policy.tier if report.defect_policy else "GREEN"

        real_results.append({
            "Dataset": r["name"],
            "Task": r["task"].value,
            "Rows": len(df),
            "Cols": len(df.columns),
            "Raw_Fitness": round(report.raw_fitness, 1),
            "Final_Fitness": round(fit, 1),
            "Defect_Tier": tier,
            "Fitness_Status": report.fitness_status,
            "Domain": r["domain"]
        })

    df_real = pd.DataFrame(real_results)
    print(df_real.to_string(index=False))
    return df_real


if __name__ == "__main__":
    df_synth, stats = run_multi_seed_benchmarks(range(42, 52))
    df_real = run_real_world_benchmarks()
