from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.audit_market_state_rank_priority_starvation import (
    _comparison_rows,
    candidate_starvation_stats,
    priority_replay_stats,
)


def _write_artifact(root: Path, *, rank: float, accepted: bool) -> None:
    simple = root / "simple_policy_optimiser"
    simple.mkdir(parents=True)
    ts = pd.Timestamp("2026-06-20T00:00:00Z")
    broad = pd.DataFrame(
        {
            "timestamp": [ts, ts],
            "symbol": ["A/USD:USD", "B/USD:USD"],
            "side": ["short", "short"],
            "strategy_id": ["short_boll_s1", "short_boll_s1"],
            "head": ["short_boll", "short_boll"],
            "normalized_rank_score": [rank, 0.65],
            "base_strategy_threshold": [0.70, 0.70],
            "net_return": [0.02, -0.01],
            "simple_policy_exit_reason": ["tp", "full_sl"],
        }
    )
    deployable = broad.loc[broad["normalized_rank_score"] >= broad["base_strategy_threshold"]].copy()
    decisions = deployable.copy()
    decisions["accepted"] = bool(accepted)
    decisions["rejection_reason"] = "accepted" if accepted else "max_new_entries_per_bar_reached"
    accepted_rows = deployable.copy() if accepted else deployable.iloc[0:0].copy()
    broad.to_parquet(simple / "simple_policy_candidates_broad.parquet", index=False)
    deployable.to_parquet(simple / "simple_policy_candidates.parquet", index=False)
    decisions.to_parquet(simple / "portfolio_decisions.parquet", index=False)
    accepted_rows.to_parquet(simple / "accepted_trades.parquet", index=False)


def test_candidate_starvation_stats_separates_threshold_pass_and_acceptance(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    _write_artifact(artifact, rank=0.75, accepted=False)

    stats, reasons = candidate_starvation_stats("test_contract", artifact)

    row = stats.iloc[0]
    assert row["head"] == "short_boll"
    assert int(row["broad_rows"]) == 2
    assert int(row["base_threshold_pass_rows"]) == 1
    assert int(row["deployable_rows"]) == 1
    assert int(row["accepted_rows"]) == 0
    assert reasons.iloc[0]["rejection_reason"] == "max_new_entries_per_bar_reached"


def test_comparison_rows_identifies_global_threshold_starvation(tmp_path: Path) -> None:
    timestamp_artifact = tmp_path / "timestamp"
    global_artifact = tmp_path / "global"
    _write_artifact(timestamp_artifact, rank=0.95, accepted=True)
    _write_artifact(global_artifact, rank=0.69, accepted=False)

    timestamp_stats, _ = candidate_starvation_stats("timestamp_rank_t1", timestamp_artifact)
    global_stats, _ = candidate_starvation_stats("global_rank_challenger", global_artifact)
    comparison = _comparison_rows(pd.concat([timestamp_stats, global_stats], ignore_index=True))

    row = comparison.iloc[0]
    assert row["head"] == "short_boll"
    assert int(row["delta_base_threshold_pass_rows"]) == -1
    assert int(row["delta_deployable_rows"]) == -1
    assert int(row["delta_accepted_rows"]) == -1


def test_priority_replay_stats_reports_no_accepted_set_movement(tmp_path: Path) -> None:
    priority_dir = tmp_path / "priority"
    priority_dir.mkdir()
    pd.DataFrame(
        [
            {"arm": "P0_static_priority", "trade_count": 2, "net_pnl": 1.0, "full_sl_rate": 0.0},
            {"arm": "L1_lgbm_learned_priority", "trade_count": 2, "net_pnl": 1.0, "full_sl_rate": 0.0},
        ]
    ).to_csv(priority_dir / "head_priority_learning_replay_summary.csv", index=False)
    pd.DataFrame(
        [
            {"arm": "P0_static_priority", "head": "short_boll", "trade_count": 2},
            {"arm": "L1_lgbm_learned_priority", "head": "short_boll", "trade_count": 2},
        ]
    ).to_csv(priority_dir / "head_priority_learning_by_head.csv", index=False)
    pd.DataFrame(
        [
            {
                "arm": "L1_lgbm_learned_priority",
                "jaccard_vs_baseline": 1.0,
                "baseline_only": 0,
                "arm_only": 0,
            }
        ]
    ).to_csv(priority_dir / "head_priority_learning_accepted_overlap.csv", index=False)

    stats = priority_replay_stats("global_rank_priority", priority_dir)

    assert stats["delta_net_pnl"] == 0.0
    assert stats["accepted_jaccard"] == 1.0
    assert stats["baseline_only"] == 0
    assert stats["priority_only"] == 0
