import io
import pandas as pd
import numpy as np
from typing import Optional, List
from pydantic import BaseModel
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Response
from fastapi.responses import StreamingResponse

from datatrust.models import (
    TaskType,
    QualityDimension,
    TrustReport,
    TaskInferenceResult,
    CalibrationResult,
    RemediationActionV2,
    ReproducibilityReceipt
)
from datatrust.reproducibility import ReproducibilityEngine
from datatrust.engine import DataTrustEngine
from datatrust.task_infer import TaskInferenceEngine
from datatrust.sensitivity import TaskQualitySensitivityMatrix
from datatrust.calibration import PerformanceCalibrationEngine
from datatrust.optimizer import RemediationOptimizer
from datatrust.reporting import ReportGenerator

router = APIRouter(prefix="/api/v1", tags=["Evaluation"])


def parse_uploaded_file(file: UploadFile, content: bytes) -> pd.DataFrame:
    filename = file.filename.lower()
    buffer = io.BytesIO(content)
    try:
        if filename.endswith(".csv"):
            return pd.read_csv(buffer)
        elif filename.endswith(".parquet"):
            return pd.read_parquet(buffer)
        elif filename.endswith((".xls", ".xlsx")):
            return pd.read_excel(buffer)
        else:
            return pd.read_csv(buffer)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse file '{file.filename}': {str(e)}")


@router.get("/tasks")
def get_supported_tasks():
    descriptions = {
        TaskType.SUPERVISED_CLASSIFICATION: "Optimized for binary/multiclass classification. Sensitive to label noise, severe imbalance, and feature drift.",
        TaskType.SUPERVISED_REGRESSION: "Optimized for continuous target prediction. Highly sensitive to extreme outliers, heavy tails, and multicollinearity.",
        TaskType.TIME_SERIES_FORECASTING: "Optimized for chronological sequence models. Strictly vetoes temporal leakage, duplicate timestamps, and cadence breaks.",
        TaskType.CLUSTERING: "Optimized for unsupervised clustering and segmentation. Prioritizes completeness, uniqueness, and feature support."
    }
    sens_matrix = TaskQualitySensitivityMatrix.get_matrix_dataframe()

    result = []
    seen_tasks = set()
    for t in TaskType:
        if t.value in seen_tasks:
            continue
        seen_tasks.add(t.value)
        weights = TaskQualitySensitivityMatrix.get_normalized_weights(t)
        desc = descriptions.get(t, "Unsupervised exploratory clustering.")
        result.append({
            "task_id": t.value,
            "name": t.value.replace("_", " ").title(),
            "description": desc,
            "sensitivities": sens_matrix.get(t.value, {}),
            "weights": {dim.value: round(w, 3) for dim, w in weights.items()}
        })
    return {"tasks": result, "sensitivity_matrix": sens_matrix}


@router.get("/sensitivities")
def get_sensitivity_matrix():
    return TaskQualitySensitivityMatrix.get_matrix_dataframe()


@router.post("/auto-detect", response_model=TaskInferenceResult)
async def auto_detect_task(
    file: UploadFile = File(...),
    target_column: Optional[str] = Form(None),
    time_column: Optional[str] = Form(None)
):
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    df = parse_uploaded_file(file, content)
    result = TaskInferenceEngine.infer_task(
        df,
        explicit_target=target_column if target_column and target_column.strip() else None,
        explicit_time=time_column if time_column and time_column.strip() else None
    )
    return result


@router.post("/evaluate", response_model=TrustReport)
async def evaluate_dataset(
    file: UploadFile = File(...),
    task: Optional[str] = Form(None),
    target_column: Optional[str] = Form(None),
    time_column: Optional[str] = Form(None),
    observed_performance: Optional[float] = Form(None),
    minority_threshold_pct: Optional[float] = Form(None),
    veto_enabled: Optional[bool] = Form(None)
):
    from datatrust.models import ImbalancePolicy
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    df = parse_uploaded_file(file, content)
    
    # Handle auto-detection if task is None or "auto"
    task_enum = None
    if task and task.strip().lower() != "auto":
        try:
            task_enum = TaskType(task.strip())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid task type: {task}")

    policy = None
    if minority_threshold_pct is not None or veto_enabled is not None:
        kwargs = {}
        if minority_threshold_pct is not None:
            kwargs["minority_threshold_pct"] = minority_threshold_pct
        if veto_enabled is not None:
            kwargs["veto_enabled"] = veto_enabled
        policy = ImbalancePolicy(**kwargs)

    report = DataTrustEngine.evaluate(
        df=df,
        task=task_enum,
        dataset_name=file.filename,
        target_column=target_column if target_column and target_column.strip() else None,
        time_column=time_column if time_column and time_column.strip() else None,
        observed_performance=observed_performance,
        imbalance_policy=policy
    )
    return report


@router.post("/calibrate", response_model=CalibrationResult)
def calibrate_performance(
    task: TaskType = Form(...),
    initial_trust_score: float = Form(...),
    observed_performance: float = Form(...),
    # Canonical v3 dimension scores
    completeness_score: Optional[float] = Form(None),
    validity_score: Optional[float] = Form(None),
    consistency_score: Optional[float] = Form(None),
    uniqueness_score: Optional[float] = Form(None),
    timeliness_score: Optional[float] = Form(None),
    outlier_anomaly_score: Optional[float] = Form(None),
    distribution_balance_score: Optional[float] = Form(None),
    coverage_representativeness_score: Optional[float] = Form(None),
    # Legacy backward-compatibility parameters
    missingness_score: Optional[float] = Form(None),
    outlier_resilience_score: Optional[float] = Form(None),
    drift_stability_score: Optional[float] = Form(None),
    label_integrity_score: Optional[float] = Form(None),
    leakage_score: Optional[float] = Form(None),
    bias_imbalance_score: Optional[float] = Form(None),
    cardinality_schema_score: Optional[float] = Form(None),
    temporal_validity_score: Optional[float] = Form(None)
):
    initial_weights = TaskQualitySensitivityMatrix.get_normalized_weights(task)
    
    # Resolve canonical dimension scores with legacy fallbacks
    c_cmp = completeness_score if completeness_score is not None else (missingness_score if missingness_score is not None else 80.0)
    c_val = validity_score if validity_score is not None else (leakage_score if leakage_score is not None else (cardinality_schema_score if cardinality_schema_score is not None else 80.0))
    c_cns = consistency_score if consistency_score is not None else 80.0
    c_unq = uniqueness_score if uniqueness_score is not None else 80.0
    c_tim = timeliness_score if timeliness_score is not None else (temporal_validity_score if temporal_validity_score is not None else (drift_stability_score if drift_stability_score is not None else 80.0))
    c_out = outlier_anomaly_score if outlier_anomaly_score is not None else (outlier_resilience_score if outlier_resilience_score is not None else 80.0)
    c_bal = distribution_balance_score if distribution_balance_score is not None else (bias_imbalance_score if bias_imbalance_score is not None else (label_integrity_score if label_integrity_score is not None else 80.0))
    c_cov = coverage_representativeness_score if coverage_representativeness_score is not None else 80.0

    dim_scores = {
        QualityDimension.COMPLETENESS: c_cmp,
        QualityDimension.VALIDITY: c_val,
        QualityDimension.CONSISTENCY: c_cns,
        QualityDimension.UNIQUENESS: c_unq,
        QualityDimension.TIMELINESS: c_tim,
        QualityDimension.OUTLIER_ANOMALY: c_out,
        QualityDimension.DISTRIBUTION_BALANCE: c_bal,
        QualityDimension.COVERAGE_REPRESENTATIVENESS: c_cov
    }

    cal_result = PerformanceCalibrationEngine.calibrate(
        task=task,
        initial_weights=initial_weights,
        dimension_scores=dim_scores,
        initial_trust_score=initial_trust_score,
        observed_performance=observed_performance
    )
    return cal_result


@router.post("/validate-downstream")
async def validate_downstream_model(
    file: Optional[UploadFile] = File(None),
    scenario: Optional[str] = Form(None),
    task: Optional[str] = Form(None),
    target_column: Optional[str] = Form(None),
    time_column: Optional[str] = Form(None)
):
    if file and file.filename:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        df = parse_uploaded_file(file, content)
        name = file.filename
        task_enum = None
        if task and task.strip().lower() != "auto":
            try:
                task_enum = TaskType(task.strip())
            except ValueError:
                pass
    elif scenario:
        df, demo_task, name, demo_target, demo_time = get_demo_scenario_data(scenario)
        task_enum = demo_task
        if not target_column:
            target_column = demo_target
        if not time_column:
            time_column = demo_time
    else:
        raise HTTPException(status_code=400, detail="Either a file upload or a scenario preset must be provided.")

    report = DataTrustEngine.evaluate(
        df=df,
        task=task_enum,
        dataset_name=name,
        target_column=target_column if target_column and target_column.strip() else None,
        time_column=time_column if time_column and time_column.strip() else None
    )
    return {
        "metric_name": report.predicted_metric_name,
        "predicted": report.predicted_metric_value,
        "observed": report.observed_metric_value,
        "residual": report.metric_residual,
        "task_conditioned_fitness": report.task_conditioned_fitness,
        "status": "Empirical model validation completed."
    }


class SimulateRequest(BaseModel):
    current_trust_score: float
    actions: List[RemediationActionV2]
    applied_action_ids: List[str]


@router.post("/simulate-remediation")
def simulate_remediation(req: SimulateRequest):
    projected = RemediationOptimizer.simulate_remediations(
        current_trust_score=req.current_trust_score,
        actions=req.actions,
        applied_action_ids=req.applied_action_ids
    )
    return {
        "current_trust_score": req.current_trust_score,
        "projected_trust_score": projected,
        "fitness_gain": round(projected - req.current_trust_score, 1)
    }


@router.post("/report/pdf")
async def download_pdf_report(
    file: Optional[UploadFile] = File(None),
    scenario: Optional[str] = Form(None),
    task: Optional[str] = Form(None),
    target_column: Optional[str] = Form(None),
    time_column: Optional[str] = Form(None)
):
    if file and file.filename:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        df = parse_uploaded_file(file, content)
        name = file.filename
        task_enum = None
        if task and task.strip().lower() != "auto":
            try:
                task_enum = TaskType(task.strip())
            except ValueError:
                pass
    elif scenario:
        df, demo_task, name, demo_target, demo_time = get_demo_scenario_data(scenario)
        task_enum = demo_task
        if not target_column:
            target_column = demo_target
        if not time_column:
            time_column = demo_time
    else:
        raise HTTPException(status_code=400, detail="Either a file upload or a scenario preset must be provided.")

    report = DataTrustEngine.evaluate(
        df=df,
        task=task_enum,
        dataset_name=name,
        target_column=target_column if target_column and target_column.strip() else None,
        time_column=time_column if time_column and time_column.strip() else None
    )

    pdf_bytes = ReportGenerator.generate_pdf(report)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"attachment; filename=datatrust_audit_{report.task_type.value}.pdf"
        }
    )


def get_demo_scenario_data(scenario: str, task: Optional[TaskType] = None) -> tuple[pd.DataFrame, TaskType, str, Optional[str], Optional[str]]:
    rng = np.random.RandomState(42)
    n = 200

    if scenario in ["temporal_leakage", "temporal_order", "dataset_c"]:
        # Dataset C: Structural Temporal Sequence Inversion & Duplicate Timestamps (Distinguished from feature leakage)
        # True chronological autoregressive + weekly seasonality sequence
        base_dates = pd.date_range("2024-01-01", periods=n, freq="D").tolist()
        y = np.zeros(n)
        y[0] = 50.0
        for i in range(1, n):
            y[i] = 0.70 * y[i - 1] + 6.0 * np.sin(2 * np.pi * (i % 7) / 7.0) + rng.normal(0, 1.0)

        seasonality = np.array([float(i % 7) for i in range(n)])
        sensor_reading = rng.normal(100, 5, n)

        # Inject 12 duplicate timestamps
        for i in [10, 25, 40, 55, 70, 85, 100, 115, 130, 145, 160, 175]:
            base_dates[i] = base_dates[i - 1]

        # Scramble chronological order to create temporal order defect
        shuffle_idx = rng.permutation(n)
        df = pd.DataFrame({
            "timestamp": [base_dates[idx] for idx in shuffle_idx],
            "day_of_week": seasonality[shuffle_idx],
            "sensor_reading": sensor_reading[shuffle_idx],
            "target": y[shuffle_idx]
        })
        return df, TaskType.TIME_SERIES_FORECASTING, "Dataset_C_Temporal_Order_and_Duplicates.csv", "target", "timestamp"

    elif scenario in ["class_imbalance", "dataset_d"]:
        # Dataset D: Severe Class Imbalance (<2% minority, RED Hard Veto F_final = 0.0, UNSAFE)
        minority_count = int(n * 0.015)  # 3 positive samples in 200 rows = 1.5%
        targets = [0] * (n - minority_count) + [1] * minority_count
        df = pd.DataFrame({
            "feature1": rng.randn(len(targets)),
            "feature2": rng.randn(len(targets)),
            "fraud_label": targets
        })
        return df, TaskType.SUPERVISED_CLASSIFICATION, "Dataset_D_Class_Imbalance.csv", "fraud_label", None

    elif scenario in ["extreme_outliers", "dataset_b"]:
        # Mandate 7: Dataset B: Extreme Outliers (target & feature kurtosis > 5, Winsorization reduces RMSE from ~32k to <10)
        vals = rng.normal(50, 5, n).tolist()
        vals[2] = 500000.0
        vals[15] = -250000.0
        df = pd.DataFrame({
            "sales": vals,
            "store_id": rng.choice(["Store_A", "Store_B", "Store_C"], n),
            "price": rng.uniform(10, 100, n),
            "customer_id": [f"ID_{i}" for i in range(n)]
        })
        return df, TaskType.SUPERVISED_REGRESSION, "Dataset_B_Extreme_Outliers.csv", "sales", None

    elif scenario in ["clean_classification"] or (scenario == "clean" and task == TaskType.SUPERVISED_CLASSIFICATION):
        # Dataset A (Classification): Clean Balanced Binary Classification
        labels = [0] * (n // 2) + [1] * (n // 2)
        rng.shuffle(labels)
        df = pd.DataFrame({
            "feature1": rng.normal(0, 1, n),
            "feature2": rng.normal(5, 2, n),
            "feature3": rng.uniform(10, 50, n),
            "label": labels
        })
        return df, TaskType.SUPERVISED_CLASSIFICATION, "Dataset_A_Clean_Classification.csv", "label", None

    else:
        # Dataset A: Clean (Regression)
        df = pd.DataFrame({
            "revenue": rng.normal(5000, 250, n),
            "visitors": rng.poisson(300, n),
            "conversion_rate": rng.uniform(0.02, 0.05, n),
            "ad_spend": rng.uniform(100, 500, n)
        })
        return df, TaskType.SUPERVISED_REGRESSION, "Dataset_A_Clean.csv", "revenue", None


@router.get("/sensitivity-matrix")
def get_sensitivity_matrix():
    return TaskQualitySensitivityMatrix.get_full_sensitivity_matrix()


@router.post("/reproducibility-receipt")
def get_reproducibility_receipt(report: TrustReport) -> ReproducibilityReceipt:
    return ReproducibilityEngine.generate_receipt(report)


@router.post("/remediate-and-validate")
async def remediate_and_validate_dataset(
    file: Optional[UploadFile] = File(None),
    scenario: Optional[str] = Form(None),
    task: Optional[str] = Form(None),
    target_column: Optional[str] = Form(None),
    time_column: Optional[str] = Form(None)
):
    if file and file.filename:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        df = parse_uploaded_file(file, content)
        task_enum = None
        if task and task.strip().lower() != "auto":
            try:
                task_enum = TaskType(task.strip())
            except ValueError:
                pass
    elif scenario:
        df, demo_task, _, demo_target, demo_time = get_demo_scenario_data(scenario)
        task_enum = demo_task
        if not target_column:
            target_column = demo_target
        if not time_column:
            time_column = demo_time
    else:
        raise HTTPException(status_code=400, detail="Either a file upload or a scenario preset must be provided.")

    report_before, report_after, val_res = DataTrustEngine.remediate_and_validate(
        df=df,
        task=task_enum,
        target_column=target_column if target_column and target_column.strip() else None,
        time_column=time_column if time_column and time_column.strip() else None
    )
    return {
        "report_before": report_before,
        "report_after": report_after,
        "validation": val_res
    }


@router.get("/demo/{scenario}")
def evaluate_demo_scenario(scenario: str, task: Optional[str] = "auto"):
    task_enum = None
    if task and task.strip().lower() != "auto":
        try:
            task_enum = TaskType(task.strip())
        except ValueError:
            pass

    df, demo_task, name, target_col, time_col = get_demo_scenario_data(scenario, task=task_enum)
    if task_enum is None:
        task_enum = demo_task

    report = DataTrustEngine.evaluate(
        df=df,
        task=task_enum,
        dataset_name=name,
        target_column=target_col,
        time_column=time_col
    )
    return report
