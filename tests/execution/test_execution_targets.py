from __future__ import annotations

import pandas as pd

from scripts.execution.train_execution_risk import reshape_surface_target
from src.execution.surface import add_causal_transition_features, add_future_deterioration_targets


def _surface() -> pd.DataFrame:
    times = pd.date_range("2026-01-01T00:00:00Z", periods=6, freq="min")
    return pd.DataFrame({
        "symbol": ["BTC/USD"] * len(times), "state_minute": times, "book_valid": [True] * len(times),
        "spread_bps": [10., 11., 12., 13., 14., 15.],
        "bid_depth_50bps": [1000., 990., 980., 970., 960., 950.],
        "ask_depth_50bps": [1000.] * len(times),
        "sell_book_cost_bps_n100": [10., 20., 30., 40., 50., 60.],
        "buy_book_cost_bps_n100": [11., 21., 31., 41., 51., 61.],
    })


def test_causal_transition_values_do_not_change_when_future_rows_change() -> None:
    original = add_causal_transition_features(_surface())
    mutated = _surface()
    mutated.loc[mutated.index >= 4, "spread_bps"] = 9_999.0
    changed = add_causal_transition_features(mutated)
    assert original.loc[3, "spread_bps_change_1m"] == changed.loc[3, "spread_bps_change_1m"]
    # The 30-minute median is intentionally unavailable on this short panel;
    # it is still identical rather than filled from later observations.
    assert pd.isna(original.loc[3, "spread_vs_recent_median"])
    assert pd.isna(changed.loc[3, "spread_vs_recent_median"])


def test_future_labels_are_explicit_and_missing_path_is_not_imputed() -> None:
    surface = _surface().drop(index=2).reset_index(drop=True)
    labelled = add_future_deterioration_targets(surface)
    assert "deterioration_sell_1m_n100" in labelled.columns
    first = labelled.iloc[0]
    # The exact 1m target can use 00:01, while a broken future path invalidates
    # the next-3m maximum rather than filling through 00:02.
    assert first["deterioration_sell_1m_n100"] == 10.0
    assert pd.isna(first["max_cost_next_3m_n100"])
    assert labelled["execution_label_source"].eq("future_l2_only").all()


def test_spread_and_max_cost_targets_are_future_only_and_do_not_cross_gaps() -> None:
    surface = _surface().drop(index=2).reset_index(drop=True)
    labelled = add_future_deterioration_targets(surface)
    first = labelled.iloc[0]
    # Terminal labels retain their declared horizon semantics.
    assert first["spread_widening_1m"] == 1.0
    # Max-path labels require every intermediate minute and must not bridge the
    # missing 00:02 state.
    assert pd.isna(first["max_spread_widening_next_3m"])
    assert pd.isna(first["max_deterioration_sell_3m_n100"])


def test_future_labels_never_expand_from_derived_cost_transitions() -> None:
    causal = add_causal_transition_features(_surface())
    assert "sell_book_cost_bps_n100_change_1m" in causal.columns
    labelled = add_future_deterioration_targets(causal)
    assert "sell_book_cost_bps_n100_future_1m" in labelled.columns
    assert not any("_change_1m_future_" in column for column in labelled.columns)
    assert not any("_change_1m" in column and column.startswith("deterioration_") for column in labelled.columns)
    # The longer requested horizon is part of the fixed label contract even
    # when this short synthetic panel cannot resolve it.
    assert "spread_widening_30m" in labelled.columns


def test_one_minute_maximum_targets_reuse_identical_terminal_labels() -> None:
    labelled = add_future_deterioration_targets(_surface())
    max_spread = reshape_surface_target(
        labelled, horizon_minutes=1, sides=("sell",), target_kind="max_spread_delta"
    )
    direct_spread = reshape_surface_target(
        labelled, horizon_minutes=1, sides=("sell",), target_kind="spread_delta"
    )
    assert max_spread["deterioration_target_bps"].equals(direct_spread["deterioration_target_bps"])
    max_cost = reshape_surface_target(
        labelled, horizon_minutes=1, sides=("sell",), target_kind="max_book_cost_delta"
    )
    direct_cost = reshape_surface_target(
        labelled, horizon_minutes=1, sides=("sell",), target_kind="book_cost_delta"
    )
    assert max_cost["deterioration_target_bps"].equals(direct_cost["deterioration_target_bps"])


def test_five_minute_snapshot_contract_never_fills_missing_source_bars() -> None:
    surface = _surface().iloc[:4].copy()
    surface["state_minute"] = pd.date_range("2026-01-01T00:00:00Z", periods=4, freq="5min")
    causal = add_causal_transition_features(surface, cadence_minutes=5)
    labelled = add_future_deterioration_targets(causal, cadence_minutes=5)
    assert "spread_bps_change_5m" in causal.columns
    assert "spread_widening_5m" in labelled.columns
    assert "spread_widening_1m" not in labelled.columns
    assert labelled.loc[0, "spread_widening_5m"] == 1.0

    broken = surface.drop(index=1).reset_index(drop=True)
    broken_labelled = add_future_deterioration_targets(broken, cadence_minutes=5)
    # 00:00 must not look through the missing 00:05 snapshot to 00:10.
    assert pd.isna(broken_labelled.loc[0, "spread_widening_10m"])
