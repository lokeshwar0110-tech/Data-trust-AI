import pytest
import pandas as pd
import numpy as np
from datatrust.validator import DownstreamValidator
from datatrust.optimizer import RemediationOptimizer
from datatrust.models import TaskType, RemediationActionV2, QualityDimension


def test_downstream_regression_validation():
    np.random.seed(42)
    n = 60
    df = pd.DataFrame({
        "feat1": np.random.randn(n),
        "feat2": np.random.randn(n),
        "target_sales": np.random.normal(50, 10, n)
    })
    res = DownstreamValidator.validate_downstream(
        df=df,
        task=TaskType.SUPERVISED_REGRESSION,
        target_column="target_sales",
        time_column=None,
        fitness_score=92.0
    )
    assert res["metric_name"] == "RMSE"
    assert res["predicted"] > 0
    assert res["observed"] is not None
    assert res["residual"] is not None


def test_downstream_classification_validation():
    np.random.seed(42)
    n = 60
    df = pd.DataFrame({
        "feat1": np.random.randn(n),
        "feat2": np.random.randn(n),
        "target_class": np.random.choice([0, 1], n)
    })
    res = DownstreamValidator.validate_downstream(
        df=df,
        task=TaskType.SUPERVISED_CLASSIFICATION,
        target_column="target_class",
        time_column=None,
        fitness_score=85.0
    )
    assert res["metric_name"] == "F1"
    assert 0.0 <= res["predicted"] <= 1.0
    assert 0.0 <= res["observed"] <= 1.0


def test_what_if_scenarios_generation():
    actions = [
        RemediationActionV2(
            id="act_1",
            dimension=QualityDimension.CORRELATION_INTEGRITY,
            issue="Leakage",
            recommendation="Drop",
            severity="critical",
            fitness_improvement=25.0,
            operational_cost=1,
            mru=25.0,
            priority_rank=1
        ),
        RemediationActionV2(
            id="act_2",
            dimension=QualityDimension.COMPLETENESS,
            issue="Missing",
            recommendation="Impute",
            severity="medium",
            fitness_improvement=8.0,
            operational_cost=2,
            mru=4.0,
            priority_rank=2
        )
    ]
    scenarios = RemediationOptimizer.generate_what_if_scenarios(
        current_trust_score=65.0,
        actions=actions
    )
    assert len(scenarios) == 3
    assert scenarios[0].name.startswith("Scenario A")
    assert scenarios[2].is_recommended is True
    assert scenarios[2].simulated_fitness >= 65.0


def test_test_set_integrity_and_audit_receipt():
    from datatrust.remediator import DataRemediator
    from datatrust.reproducibility import ReproducibilityEngine

    # Create imbalanced dataset
    np.random.seed(42)
    n_train, n_test = 100, 40
    train_df = pd.DataFrame({
        "feat": np.random.randn(n_train),
        "target": [0] * 95 + [1] * 5
    })
    test_df = pd.DataFrame({
        "feat": np.random.randn(n_test),
        "target": [0] * 38 + [1] * 2
    })

    test_len_before = len(test_df)
    test_pos_before = int((test_df["target"] == 1).sum())

    rem_train, rem_test, log, params = DataRemediator.fit_and_transform_splits(
        train_df=train_df,
        test_df=test_df,
        task=TaskType.SUPERVISED_CLASSIFICATION,
        action_ids=["act_resample"],
        target_column="target"
    )

    # TEST SET INTEGRITY VERIFICATION:
    # 1. Test set must NOT be resampled or altered in length
    assert len(rem_test) == test_len_before
    # 2. Test set class counts must remain exactly the original distribution
    assert int((rem_test["target"] == 1).sum()) == test_pos_before
    # 3. Train set DID get resampled
    assert len(rem_train) > n_train
    # 4. Test index hash immutability
    assert params["test_indices_immutable"] is True
    assert params["test_index_hash_before"] == params["test_index_hash_after"]
    assert list(rem_test.index) == list(test_df.index)

    # Audit receipt verification
    receipt = ReproducibilityEngine.generate_downstream_audit_receipt(
        train_indices=list(range(n_train)),
        test_indices=list(range(n_test)),
        feature_columns=["feat"],
        target_column="target",
        time_column=None,
        preprocessing_operations=["70/30 split"],
        remediation_operations=log,
        model_family="Random Forest Classifier",
        hyperparameters={"max_depth": 6},
        baseline_metric=0.50,
        remediated_metric=0.75,
        metric_name="F1",
        test_index_hash_before=params["test_index_hash_before"],
        test_index_hash_after=params["test_index_hash_after"]
    )
    assert receipt.leakage_free_verified is True
    assert receipt.test_set_resampled is False
    assert receipt.test_indices_immutable is True
    assert receipt.test_index_hash_before == receipt.test_index_hash_after
    assert len(receipt.train_indices_hash) == 64
    assert len(receipt.test_indices_hash) == 64


def test_time_series_test_set_immutability():
    from datatrust.remediator import DataRemediator
    import hashlib

    # Time series with duplicates and scrambled sequence
    train_dates = ["2024-01-03", "2024-01-01", "2024-01-01", "2024-01-02"]
    test_dates = ["2024-01-05", "2024-01-04", "2024-01-05"]
    train_df = pd.DataFrame({
        "timestamp": train_dates,
        "target": [10.0, 20.0, 20.0, 30.0]
    })
    test_df = pd.DataFrame({
        "timestamp": test_dates,
        "target": [40.0, 35.0, 40.0]
    }, index=[100, 101, 102])

    test_hash_before = hashlib.sha256(str(list(test_df.index)).encode('utf-8')).hexdigest()

    rem_train, rem_test, log, params = DataRemediator.fit_and_transform_splits(
        train_df=train_df,
        test_df=test_df,
        task=TaskType.TIME_SERIES_FORECASTING,
        action_ids=["act_sort"],
        target_column="target",
        time_column="timestamp"
    )

    # Verify:
    # 1. Train duplicates removed and sorted
    assert len(rem_train) == 3
    # 2. Test set IMMUTABLE: no rows dropped, no deduplication, index unchanged
    assert len(rem_test) == 3
    assert list(rem_test.index) == [100, 101, 102]
    test_hash_after = hashlib.sha256(str(list(rem_test.index)).encode('utf-8')).hexdigest()
    assert test_hash_before == test_hash_after
    assert params["test_indices_immutable"] is True
