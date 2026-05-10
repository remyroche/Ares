import pytest
import pandas as pd
import numpy as np
import tempfile
import os

from extreme_price_movements.slice_plan_store import (
    restrict_stage_symbols,
    restrict_stage_period,
    apply_stage_usage_limits,
    compute_event_fingerprint,
    slice_plan_is_stale,
    _deserialize_timestamp,
)


def test_restrict_stage_symbols():
    view = {"stage_name": "test", "symbols": ["A", "C", "B", "D"]}

    # max_assets None
    assert restrict_stage_symbols(view, None)["symbols"] == ["A", "C", "B", "D"]

    # max_assets 0
    assert restrict_stage_symbols(view, 0)["symbols"] == ["A", "C", "B", "D"]

    # max_assets less than total
    # notice the sorting should occur inside the function
    res = restrict_stage_symbols(view, 2)
    assert res["symbols"] == ["A", "B"]

    # max_assets more than total
    res2 = restrict_stage_symbols(view, 10)
    assert set(res2["symbols"]) == {"A", "B", "C", "D"}


def test_restrict_stage_period():
    view = {
        "stage_name": "test",
        "allowed_start_ts": "2020-01-01T00:00:00+00:00",
        "allowed_end_ts": "2021-01-01T00:00:00+00:00",
    }

    # max_months None
    assert (
        restrict_stage_period(view, None)["allowed_start_ts"]
        == "2020-01-01T00:00:00+00:00"
    )

    # max_months 6
    res = restrict_stage_period(view, 6)
    # 2021-01-01 - 6 months approx 2020-07-01
    assert "2020-07-01" in res["allowed_start_ts"]

    # max_months 24 (exceeds span)
    res2 = restrict_stage_period(view, 24)
    assert res2["allowed_start_ts"] == "2020-01-01T00:00:00+00:00"


def test_apply_stage_usage_limits():
    view = {
        "stage_name": "test",
        "symbols": ["A", "C", "B"],
        "allowed_start_ts": "2020-01-01T00:00:00+00:00",
        "allowed_end_ts": "2021-01-01T00:00:00+00:00",
    }

    res = apply_stage_usage_limits(view, max_assets=1, max_months=1)
    assert res["symbols"] == ["A"]
    assert "2020-12-01" in res["allowed_start_ts"]


def test_compute_event_fingerprint():
    df = pd.DataFrame(
        {
            "symbol": ["A", "B"],
            "t0": [pd.Timestamp("2021-01-01"), pd.Timestamp("2021-01-02")],
        }
    )
    fp = compute_event_fingerprint(df)
    assert fp["n_events"] == 2
    assert fp["n_symbols"] == 2
    assert fp["hash"] is not None


def test_slice_plan_is_stale():
    existing = {
        "version": 2,
        "event_fingerprint": {"hash": "abc"},
        "planner": {"preset": "fast"},
    }

    # Not stale
    assert not slice_plan_is_stale(existing, {"hash": "abc"}, {"preset": "fast"})

    # Fingerprint changed
    assert slice_plan_is_stale(existing, {"hash": "def"}, {"preset": "fast"})

    # Config changed
    assert slice_plan_is_stale(existing, {"hash": "abc"}, {"preset": "robust"})


def test_build_stage_view_holdout_combined():
    from extreme_price_movements.periods_symbols_management import ConsumerSlicePlan

    cp1 = ConsumerSlicePlan(
        tag="t1",
        fit_idx=np.array([]),
        predict_idx=np.array([]),
        symbols_fit={"A"},
        symbols_predict={"B"},
        metadata={},
        consumer_role="test",
        outer_fold_id="f1",
        inner_fold_id=None,
        oof_target_idx=None,
    )
    cp2 = ConsumerSlicePlan(
        tag="t2",
        fit_idx=np.array([]),
        predict_idx=np.array([]),
        symbols_fit={"C"},
        symbols_predict={"D"},
        metadata={},
        consumer_role="test",
        outer_fold_id="f1",
        inner_fold_id=None,
        oof_target_idx=None,
    )

    from extreme_price_movements.slice_plan_store import _build_stage_view

    view = _build_stage_view(
        "holdout_strategy_eval", [cp1, cp2], 0.1, ["policy_optimiser", "backtest_eval"]
    )

    assert "source_roles" in view
    assert view["source_roles"] == ["policy_optimiser", "backtest_eval"]
    assert set(view["symbols"]) == {"A", "B", "C", "D"}


def test_restrict_stage_period_with_missing_dates():
    view = {
        "stage_name": "test",
        "allowed_start_ts": None,
        "allowed_end_ts": None,
        "symbols": ["A", "B"],
    }

    from extreme_price_movements.slice_plan_store import restrict_stage_period

    res = restrict_stage_period(view, 5)
    assert res["allowed_start_ts"] is None
    assert res["symbols"] == ["A", "B"]
