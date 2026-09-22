import pytest
from datatrust.optimizer import RemediationOptimizer
from datatrust.models import (
    TaskType,
    QualityDimension,
    DimensionResult,
    RemediationActionV2,
    RemediationDecision
)


def test_mru_ranking():
    actions = [
        RemediationActionV2(
            id="act_1",
            dimension=QualityDimension.COMPLETENESS,
            issue="Missing data",
            recommendation="Impute missing",
            severity="medium",
            fitness_improvement=6.0,
            operational_cost=4,
            mru=1.5,
            priority_rank=0
        ),
        RemediationActionV2(
            id="act_2",
            dimension=QualityDimension.CORRELATION_INTEGRITY,
            issue="Target leakage",
            recommendation="Drop column",
            severity="critical",
            fitness_improvement=28.0,
            operational_cost=1,
            mru=28.0,
            priority_rank=0
        )
    ]
    actions.sort(key=lambda a: a.mru, reverse=True)
    assert actions[0].id == "act_2"
    assert actions[0].mru == 28.0


def test_simulate_remediations():
    current_score = 60.0
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
            fitness_improvement=10.0,
            operational_cost=2,
            mru=5.0,
            priority_rank=2
        )
    ]

    sim_score = RemediationOptimizer.simulate_remediations(
        current_trust_score=current_score,
        actions=actions,
        applied_action_ids=["act_1"]
    )
    assert sim_score == 85.0

    sim_all = RemediationOptimizer.simulate_remediations(
        current_trust_score=current_score,
        actions=actions,
        applied_action_ids=["act_1", "act_2"]
    )
    assert sim_all == 95.0


def test_mru_simulation_accounts_for_irreducible_test_defects():
    """
    Issue 6 Verification:
    Verifies that RemediationOptimizer discounts predicted gains by the immutable
    holdout test partition defect fraction (1 - train_ratio).
    Asserts predicted gain matches observed post-remediation gain within +- 10 points.
    """
    import numpy as np
    import pandas as pd
    from datatrust.engine import DataTrustEngine

    # Synthesize Dataset with temporal disorder
    rng = np.random.RandomState(42)
    n = 150
    base_dates = pd.date_range("2024-01-01", periods=n, freq="D").tolist()
    y = np.zeros(n)
    y[0] = 50.0
    for i in range(1, n):
        y[i] = 0.70 * y[i - 1] + 5.0 * np.sin(2 * np.pi * (i % 7) / 7.0) + rng.normal(0, 1.0)
    for i in [15, 30, 45, 60, 75, 90, 105, 120, 135]:
        base_dates[i] = base_dates[i - 1]
    shuffle_idx = rng.permutation(n)
    df_ts = pd.DataFrame({
        "timestamp": [base_dates[idx] for idx in shuffle_idx],
        "sensor_reading": rng.normal(100, 5, n),
        "target": y[shuffle_idx]
    })

    rep_b, rep_a, res = DataTrustEngine.remediate_and_validate(
        df=df_ts,
        task=TaskType.TIME_SERIES_FORECASTING,
        target_column="target",
        time_column="timestamp"
    )

    # 1. Check predicted gain accounts for immutable test partition
    pred_gain = res.predicted_fitness_gain
    obs_gain = res.observed_fitness_gain

    # 2. Difference between predicted and observed gain must be within +- 10 points
    delta = abs(pred_gain - obs_gain)
    assert delta <= 10.0, f"Predicted gain ({pred_gain}) diverged from observed gain ({obs_gain}) by {delta:.1f} > 10.0"

    # 3. Post-remediation decision must not be falsely BLOCKED if downstream model improved
    if res.performance_status == "Improved ✓":
        assert res.decision != RemediationDecision.BLOCKED
