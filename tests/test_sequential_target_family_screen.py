import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.sequential_target_family_screen import (
    TargetFamilyScreenError,
    materialize_target_family_labels,
    nested_oof_fold_plan,
    reconcile_quantiles,
    soft_triple_barrier_distribution,
)


def _frame() -> pd.DataFrame:
    decision = pd.to_datetime([
        "2024-01-01T00:00:00Z", "2024-02-01T00:00:00Z", "2024-03-01T00:00:00Z",
    ])
    return pd.DataFrame({
        "candidate_id": ["a", "b", "c"], "__ts__": decision, "__decision_ts__": decision,
        "__label_available_at__": decision + pd.Timedelta(hours=12),
        "execution_gross_ev_12h": [0.02, -0.01, 0.015],
        "execution_cost_return": [0.01, 0.01, 0.01],
        "execution_net_ev_12h": [0.01, -0.02, 0.005],
        "__first_touch_target_soft__": [0.9, 0.1, 0.5],
        "clean_economic_favorable_first": [1, 0, 0], "adverse_first": [0, 1, 0], "timeout": [0, 0, 1],
        "same_minute_favorable_adverse_conflict": [0, 0, 0], "competing_risk_atr_fraction": [0.01, 0.01, 0.01],
        "endpoint_favorable_margin_return": [0.01, -0.02, -0.005],
        "endpoint_adverse_margin_return": [0.02, -0.01, 0.005],
        "oof_fold": ["warmup", "base", "test"],
    })


def test_soft_barrier_preserves_hits_and_is_simplex():
    values = soft_triple_barrier_distribution(_frame())
    assert np.allclose(values.sum(axis=1), 1.0)
    assert np.array_equal(values[0], np.array([1.0, 0.0, 0.0]))
    assert np.array_equal(values[1], np.array([0.0, 1.0, 0.0]))
    assert np.all(values[2] > 0.0)


def test_g1_fails_closed_without_ordered_path():
    with pytest.raises(TargetFamilyScreenError, match="ordered H12 minute paths"):
        soft_triple_barrier_distribution(_frame(), geometry="G1")


def test_materialized_labels_preserve_cost_and_atr_normalization():
    labels, manifest = materialize_target_family_labels(_frame())
    assert np.allclose(labels.target_t4_net_atr, labels.execution_net_ev_12h / labels.competing_risk_atr_fraction)
    assert manifest["candidate_count"] == 3
    assert set(("target_t0_control", "target_t1_net_return", "target_t2_upper_soft", "target_t3_net_return", "target_t4_net_atr")).issubset(labels)


def test_quantiles_reconcile_and_meta_plan_is_nested():
    fixed = reconcile_quantiles(np.array([[2.0, 1.0, 3.0, 2.5, 4.0]]))
    assert np.all(np.diff(fixed, axis=1) >= 0.0)
    plan = nested_oof_fold_plan(_frame())
    assert not bool(plan.iloc[1].meta_scored)
    assert bool(plan.iloc[2].meta_scored)
    assert plan.iloc[2].meta_train_folds_with_base_oof == ["base"]
