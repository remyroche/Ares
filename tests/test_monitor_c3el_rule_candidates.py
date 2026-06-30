from pathlib import Path

import pandas as pd

from scripts.monitor_c3el_rule_candidates import load_scored_features, summarize_rules, tag_rules


def test_tag_rules_sets_strict_conjunctive_and_weak_flags() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-06-15", periods=4, freq="h", tz="UTC"),
            "strategy_id": ["short_asset_a"] * 4,
            "action_family": ["size"] * 4,
            "action_value": [0.0] * 4,
            "p_intervene": [0.90, 0.91, 0.75, 0.60],
            "pred_action_delta_J": [330.0, 340.0, 150.0, 500.0],
            "cooldown_count": [30.0, 50.0, 10.0, 10.0],
            "timestamp_rank_q90": [0.80, 0.90, 0.70, 0.70],
            "strategy_candidate_open_or_cooldown_symbol_share": [0.30, 0.50, 0.20, 0.20],
            "strategy_rank_max": [0.80, 0.95, 0.70, 0.70],
            "feature_row_matched": [True] * 4,
        }
    )

    tagged = tag_rules(frame)

    assert tagged["rule_strict_p80_d320"].tolist() == [True, True, False, False]
    assert tagged["rule_p80_d320_cooldown_lte_38_5"].tolist() == [True, False, False, False]
    assert tagged["rule_p80_d320_cooldown_lte_38_5_open_or_cooldown_share_lte_0_3949"].tolist() == [
        True,
        False,
        False,
        False,
    ]
    assert tagged["rule_p80_d320_timestamp_rank_q90_lte_0_8641"].tolist() == [True, False, False, False]
    assert tagged["rule_p80_d320_open_or_cooldown_share_lte_0_3949"].tolist() == [True, False, False, False]
    assert tagged["rule_p80_d320_strategy_rank_max_lte_0_9054"].tolist() == [True, False, False, False]
    assert tagged["rule_p80_d320_at_least_4_conditions"].tolist() == [True, False, False, False]
    assert tagged["rule_weak_p70_d100"].tolist() == [True, True, True, False]
    assert tagged["monitor_condition_count"].tolist() == [4, 0, 4, 4]

    summary, by_day = summarize_rules(tagged)
    strict_rows = summary.loc[summary["rule"].eq("rule_strict_p80_d320"), "rows"].iloc[0]
    cooldown_rows = summary.loc[summary["rule"].eq("rule_p80_d320_cooldown_lte_38_5"), "rows"].iloc[0]
    assert strict_rows == 2
    assert cooldown_rows == 1
    assert not by_day.empty


def test_load_scored_features_filters_head_action_and_joins_features(tmp_path: Path) -> None:
    ts = pd.Timestamp("2026-06-15 00:00:00", tz="UTC")
    scores_path = tmp_path / "scores.csv"
    features_path = tmp_path / "features.parquet"
    pd.DataFrame(
        {
            "timestamp": [ts, ts + pd.Timedelta(hours=1), ts + pd.Timedelta(hours=2), ts + pd.Timedelta(hours=3)],
            "strategy_id": ["short_asset_a", "short_asset_a", "short_boll_a", "short_asset_a"],
            "head": ["short_asset", "short_asset", "short_boll", "short_asset"],
            "action_family": ["size", "size", "size", "size"],
            "action_value": [0.0, 0.5, 0.0, 0.0],
            "p_intervene": [0.9, 0.9, 0.9, 0.8],
            "pred_action_delta_J": [330.0, 330.0, 330.0, 100.0],
        }
    ).to_csv(scores_path, index=False)
    pd.DataFrame(
        {
            "timestamp": [ts, ts + pd.Timedelta(hours=1), ts + pd.Timedelta(hours=2), ts + pd.Timedelta(hours=3)],
            "strategy_id": ["short_asset_a", "short_asset_a", "short_boll_a", "short_asset_a"],
            "multiplier": [0.0, 0.5, 0.0, 0.0],
            "cooldown_count": [10.0, 20.0, 30.0, 40.0],
        }
    ).to_parquet(features_path, index=False)

    joined = load_scored_features(scores_path, features_path, action_value=0.0, head="short_asset")

    assert len(joined) == 2
    assert set(joined["timestamp"]) == {ts, ts + pd.Timedelta(hours=3)}
    assert joined["feature_row_matched"].tolist() == [True, True]
    assert joined["cooldown_count"].tolist() == [10.0, 40.0]
