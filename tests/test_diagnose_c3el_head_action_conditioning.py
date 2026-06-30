import numpy as np
import pandas as pd

from scripts.diagnose_c3el_head_action_conditioning import build_slice_report, build_week_report, load_actions


def test_load_actions_filters_head_nonbaseline_and_binding_actions(tmp_path):
    path = tmp_path / "actions.csv"
    pd.DataFrame(
        [
            {
                "timestamp": "2026-06-15T00:00:00+00:00",
                "strategy_id": "short_boll_a",
                "multiplier": 0.5,
                "action_binds": 1,
                "delta_full_J": 10.0,
                "delta_immediate_J": 1.0,
                "delta_full_net_pnl": 10.0,
                "delta_full_cost_pnl": 0.0,
                "delta_full_turnover": 0.0,
                "affected_notional": 100.0,
                "strategy_rank_q90": 0.8,
            },
            {
                "timestamp": "2026-06-15T01:00:00+00:00",
                "strategy_id": "short_boll_a",
                "multiplier": 1.0,
                "action_binds": 1,
                "delta_full_J": 99.0,
                "delta_immediate_J": 1.0,
                "delta_full_net_pnl": 99.0,
                "delta_full_cost_pnl": 0.0,
                "delta_full_turnover": 0.0,
                "affected_notional": 100.0,
                "strategy_rank_q90": 0.9,
            },
            {
                "timestamp": "2026-06-15T02:00:00+00:00",
                "strategy_id": "short_asset_a",
                "multiplier": 0.5,
                "action_binds": 1,
                "delta_full_J": 99.0,
                "delta_immediate_J": 1.0,
                "delta_full_net_pnl": 99.0,
                "delta_full_cost_pnl": 0.0,
                "delta_full_turnover": 0.0,
                "affected_notional": 100.0,
                "strategy_rank_q90": 0.9,
            },
            {
                "timestamp": "2026-06-15T03:00:00+00:00",
                "strategy_id": "short_boll_b",
                "multiplier": 0.5,
                "action_binds": 0,
                "delta_full_J": 99.0,
                "delta_immediate_J": 1.0,
                "delta_full_net_pnl": 99.0,
                "delta_full_cost_pnl": 0.0,
                "delta_full_turnover": 0.0,
                "affected_notional": 100.0,
                "strategy_rank_q90": 0.9,
            },
        ]
    ).to_csv(path, index=False)

    frame = load_actions(path, head="short_boll")

    assert len(frame) == 1
    assert frame.iloc[0]["strategy_id"] == "short_boll_a"
    assert frame.iloc[0]["week_start"].isoformat() == "2026-06-15T00:00:00+00:00"


def test_build_slice_report_finds_condition_with_positive_action_value():
    rows = []
    for i in range(60):
        good = i < 30
        rows.append(
            {
                "timestamp": pd.Timestamp("2026-06-15", tz="UTC") + pd.Timedelta(hours=i),
                "strategy_id": f"short_boll_{i % 3}",
                "multiplier": 0.5,
                "head": "short_boll",
                "delta_full_J": 100.0 if good else -50.0,
                "delta_immediate_J": 50.0 if good else -25.0,
                "delta_full_net_pnl": 100.0 if good else -50.0,
                "delta_full_cost_pnl": 0.0,
                "delta_full_turnover": 0.0,
                "strategy_candidate_count": i,
                "strategy_rank_q90": 0.2 + 0.01 * i,
                "noise_feature": np.sin(i),
                "week_start": pd.Timestamp("2026-06-15", tz="UTC"),
            }
        )
    frame = pd.DataFrame(rows)

    report = build_slice_report(frame, min_rows=10, epsilon=50.0, quantiles=[0.5])
    weeks = build_week_report(frame, epsilon=50.0)
    top = report.iloc[0]

    assert not report.empty
    assert top["feature"] == "strategy_candidate_count"
    assert top["direction"] == "low"
    assert top["selected_sum_delta_full_J"] > 0
    assert top["selected_positive_share"] == 1.0
    assert weeks.iloc[0]["sum_delta_full_J"] == 1500.0
