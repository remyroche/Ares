import pandas as pd

from scripts.diagnose_c3el_weekly_conditioning import (
    load_weekly_deltas,
    load_weekly_features,
    score_weekly_conditions,
)


def test_weekly_conditioning_finds_feature_that_separates_positive_weeks(tmp_path):
    weekly_path = tmp_path / "weekly.csv"
    feature_path = tmp_path / "features.csv"
    rows = []
    for week, base_pnl, cand_pnl in [
        ("2026-06-01T00:00:00+00:00", 100.0, 150.0),
        ("2026-06-08T00:00:00+00:00", 100.0, 140.0),
        ("2026-06-15T00:00:00+00:00", 100.0, 80.0),
        ("2026-06-22T00:00:00+00:00", 100.0, 70.0),
    ]:
        rows.append(
            {
                "arm": "C0_baseline",
                "week_start": week,
                "net_pnl": base_pnl,
                "trade_count": 10,
                "net_hit_rate_pct": 50.0,
                "full_sl_rate_pct": 30.0,
                "timeout_rate_pct": 20.0,
            }
        )
        rows.append(
            {
                "arm": "C3el_head_native",
                "week_start": week,
                "net_pnl": cand_pnl,
                "trade_count": 9,
                "net_hit_rate_pct": 51.0,
                "full_sl_rate_pct": 29.0,
                "timeout_rate_pct": 20.0,
            }
        )
    pd.DataFrame(rows).to_csv(weekly_path, index=False)

    feature_rows = []
    for week, signal in [
        ("2026-06-01T00:00:00+00:00", 10.0),
        ("2026-06-08T00:00:00+00:00", 11.0),
        ("2026-06-15T00:00:00+00:00", 1.0),
        ("2026-06-22T00:00:00+00:00", 2.0),
    ]:
        for hour in range(3):
            feature_rows.append(
                {
                    "timestamp": pd.Timestamp(week) + pd.Timedelta(hours=hour),
                    "strategy_id": "short_boll_test",
                    "multiplier": 0.5,
                    "action_binds": 1,
                    "state_signal": signal,
                    "noise": hour,
                }
            )
    pd.DataFrame(feature_rows).to_csv(feature_path, index=False)

    weekly = load_weekly_deltas(weekly_path, arm="C3el_head_native")
    features = load_weekly_features(feature_path, head="short_boll")
    joined = weekly.merge(features, on="week_start", how="left")
    report = score_weekly_conditions(joined)
    top = report.iloc[0]

    assert weekly["delta_net_pnl"].tolist() == [50.0, 40.0, -20.0, -30.0]
    assert top["feature"] == "state_signal__mean"
    assert top["direction"] == "high"
    assert top["selected_positive_week_share"] == 1.0
    assert top["selected_delta_net_pnl_sum"] == 90.0
