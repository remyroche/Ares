from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.materialize_long_exact_h12_raw_base_panel import START, END, validate_panel


def _frame() -> pd.DataFrame:
    timestamps = pd.date_range(START, END - pd.Timedelta(hours=1), freq="MS", tz="UTC")
    return pd.DataFrame({
        "candidate_id": [f"id-{index}" for index in range(len(timestamps))],
        "__ts__": timestamps, "__symbol__": "BTC", "side_name": "long",
        "frozen_base_score": 0.0, "__decision_ts__": timestamps + pd.Timedelta(hours=1),
        "__label_end_ts__": timestamps + pd.Timedelta(hours=13),
        "__label_available_at__": timestamps + pd.Timedelta(hours=13),
        "execution_label_end_utc": timestamps + pd.Timedelta(hours=13),
        "execution_label_available_at": timestamps + pd.Timedelta(hours=13),
        "execution_gross_ev_12h": 0.02, "execution_cost_return": 0.01,
        "execution_net_ev_12h": 0.01, "__opportunity_occurred_12h__": 1.0,
        "__favorable_payoff_return_12h__": 0.02, "__adverse_competing_risk_12h__": 0.0,
        "__timeout_outcome_12h__": 0.0, "__exit_conversion_loss_return_12h__": 0.0,
        "__peak_mfe_atr_12h__": 2.0, "__time_to_first_meaningful_mfe_hours_12h__": 1.0,
        "__mae_before_meaningful_mfe_atr_12h__": 0.1,
        "__bars_before_price_stops_decreasing_12h__": 3.0,
        "__future_slope_atr_per_hour_12h__": 0.1, "raw_pit": np.arange(len(timestamps), dtype=float),
    })


def test_validates_requested_20_month_exact_h12_panel() -> None:
    validate_panel(_frame(), ["raw_pit"])


def test_rejects_label_availability_before_endpoint() -> None:
    frame = _frame()
    frame.loc[0, "execution_label_available_at"] = frame.loc[0, "__decision_ts__"]
    try:
        validate_panel(frame, ["raw_pit"])
    except ValueError as error:
        assert "availability" in str(error)
    else:
        raise AssertionError("expected strict availability rejection")


def test_rejects_frozen_geometry_or_selection_inputs() -> None:
    frame = _frame()
    frame["dae_b16_00"] = 0.0
    try:
        validate_panel(frame, ["raw_pit", "dae_b16_00"])
    except ValueError as error:
        assert "unsupported frozen geometry" in str(error)
    else:
        raise AssertionError("expected frozen-geometry rejection")
