from pathlib import Path

import pandas as pd

from scripts.run_c3el_forward_readiness_check import run_check


def test_forward_readiness_check_runs_monitor_and_targets(tmp_path: Path) -> None:
    ts = pd.Timestamp("2026-06-15 00:00:00", tz="UTC")
    scores = tmp_path / "scores.csv"
    features = tmp_path / "features.parquet"
    existing = tmp_path / "existing.csv"
    out_dir = tmp_path / "out"

    pd.DataFrame(
        {
            "timestamp": [ts, ts + pd.Timedelta(hours=1), ts + pd.Timedelta(hours=2)],
            "strategy_id": ["short_asset_a", "short_asset_a", "short_asset_a"],
            "head": ["short_asset", "short_asset", "short_asset"],
            "action_family": ["size", "size", "size"],
            "action_value": [0.0, 0.0, 0.0],
            "p_intervene": [0.95, 0.96, 0.40],
            "pred_action_delta_J": [350.0, 360.0, 500.0],
        }
    ).to_csv(scores, index=False)
    pd.DataFrame(
        {
            "timestamp": [ts, ts + pd.Timedelta(hours=1), ts + pd.Timedelta(hours=2)],
            "strategy_id": ["short_asset_a", "short_asset_a", "short_asset_a"],
            "multiplier": [0.0, 0.0, 0.0],
            "cooldown_count": [20.0, 25.0, 10.0],
            "timestamp_rank_q90": [0.80, 0.80, 0.80],
            "strategy_candidate_open_or_cooldown_symbol_share": [0.20, 0.20, 0.20],
            "strategy_rank_max": [0.80, 0.80, 0.80],
        }
    ).to_parquet(features, index=False)
    pd.DataFrame(
        {
            "timestamp": [ts],
            "strategy_id": ["short_asset_a"],
            "action_family": ["size"],
            "action_value": [0.0],
        }
    ).to_csv(existing, index=False)

    manifest = run_check(
        scores=scores,
        action_features=features,
        existing_labels=[existing],
        out_dir=out_dir,
        max_targets=10,
        max_per_day=10,
    )

    assert manifest["decision"] == "run_exact_state_replay"
    assert manifest["rules"] == [
        "rule_p80_d320_cooldown_lte_38_5",
        "rule_p80_d320_cooldown_lte_38_5_open_or_cooldown_share_lte_0_3949",
    ]
    assert manifest["rule_rows_by_rule"]["rule_p80_d320_cooldown_lte_38_5"] == 2
    assert (
        manifest["rule_rows_by_rule"]["rule_p80_d320_cooldown_lte_38_5_open_or_cooldown_share_lte_0_3949"]
        == 2
    )
    assert manifest["rule_rows_before_existing_exclusion"] == 2
    assert manifest["rule_rows_before_existing_exclusion_sum_by_rule"] == 4
    assert manifest["candidate_pool_rows"] == 1
    assert manifest["target_rows"] == 1
    assert (out_dir / "monitor" / "tagged_score_rows.csv").exists()
    assert (out_dir / "targets" / "target_actions.csv").exists()
    assert (
        out_dir / "targets_p80_d320_cooldown_lte_38_5_open_or_cooldown_share_lte_0_3949" / "target_actions.csv"
    ).exists()

    targets = pd.read_csv(out_dir / "targets" / "target_actions.csv")
    assert len(targets) == 1
    assert pd.Timestamp(targets.iloc[0]["timestamp"]) == ts + pd.Timedelta(hours=1)
