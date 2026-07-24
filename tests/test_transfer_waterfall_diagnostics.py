from __future__ import annotations

import numpy as np
import pandas as pd

from extreme_price_movements.diagnostics.transfer_waterfall import (
    common_key_intersection,
    grouped_transfer_reports,
    stage_waterfall_metrics,
)


KEYS = ("timestamp", "symbol")


def _ledgers() -> dict[str, pd.DataFrame]:
    timestamp = pd.to_datetime(
        ["2026-01-31T23:00:00Z", "2026-02-01T01:00:00Z", "2026-02-02T01:00:00Z"], utc=True
    )
    base = pd.DataFrame(
        {
            "timestamp": timestamp,
            "symbol": ["AAA", "BBB", "CCC"],
            "side": ["long", "short", "long"],
            "baseline_side": ["long", "long", "long"],
            "archetype": ["trend", "mean_revert", "trend"],
            "score": [0.9, 0.8, 0.7],
            "gross_return": [0.030, 0.020, 0.010],
            "net_return": [0.020, 0.010, 0.000],
            "mfe": [0.040, 0.030, 0.020],
            "mae": [-0.010, -0.020, -0.030],
            "exit_reason": ["tp", "sl", "timeout"],
        }
    )
    policy = base.iloc[[0, 1]].copy()
    policy["side"] = ["short", "short"]
    policy["gross_return"] = [0.040, 0.030]
    policy["net_return"] = [0.030, 0.020]
    policy["exit_reason"] = ["hard_tp", "timeout"]
    return {"base": base, "policy": policy}


def test_common_intersection_makes_stage_metrics_identical_row_comparisons() -> None:
    ledgers = _ledgers()

    common = common_key_intersection(ledgers, key_cols=KEYS)
    metrics = stage_waterfall_metrics(ledgers, key_cols=KEYS)

    assert common["symbol"].tolist() == ["AAA", "BBB"]
    assert metrics["trade_count"].tolist() == [2, 2]
    assert metrics.loc[0, "ev_per_trade"] == 0.015
    assert metrics.loc[1, "ev_per_trade"] == 0.025


def test_cost_reconciliation_uses_supplied_net_once() -> None:
    metrics = stage_waterfall_metrics(_ledgers(), key_cols=KEYS).set_index("stage")

    # The base stage's intersection rows are gross 5% and supplied net 3%; no
    # extra fee/spread column is used to reduce net a second time.
    assert metrics.loc["base", "net_return_sum"] == 0.03
    assert np.isclose(metrics.loc["base", "cost_drag_sum"], 0.02)
    assert np.isclose(metrics.loc["base", "gross_to_net_ratio"], 0.6)


def test_outcome_flips_compare_previous_stage_and_stored_baseline() -> None:
    metrics = stage_waterfall_metrics(_ledgers(), key_cols=KEYS).set_index("stage")

    assert np.isnan(metrics.loc["base", "flip_vs_previous_rate"])
    assert metrics.loc["base", "flip_vs_baseline_rate"] == 0.0
    # BBB changes from +1% to +2%, so both remain positive. AAA also remains
    # positive; there are no binary economic-outcome flips.
    assert metrics.loc["policy", "flip_vs_previous_rate"] == 0.0
    assert metrics.loc["policy", "flip_vs_baseline_rate"] == 0.0


def test_grouped_reports_cover_side_month_archetype_and_global_score_tails() -> None:
    report = grouped_transfer_reports(
        _ledgers(),
        key_cols=KEYS,
        score_tail_bounds=(("top_half", 0.0, 0.5), ("bottom_half", 0.5, 1.0)),
    )

    assert set(report) == {"overall", "side", "month", "archetype", "global_score_tail"}
    assert set(report["side"].loc[report["side"]["stage"].eq("base"), "side"]) == {"long", "short"}
    assert set(report["month"].loc[report["month"]["stage"].eq("base"), "month"]) == {"2026-01", "2026-02"}
    assert set(report["archetype"].loc[report["archetype"]["stage"].eq("base"), "archetype"]) == {"trend", "mean_revert"}
    tails = report["global_score_tail"].query("stage == 'base'").set_index("score_tail")
    assert tails.loc["top_half", "trade_count"] == 1
    assert tails.loc["bottom_half", "trade_count"] == 1
