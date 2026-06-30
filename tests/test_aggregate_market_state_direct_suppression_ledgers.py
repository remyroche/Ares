from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.aggregate_market_state_direct_suppression_ledgers import (
    aggregate_direct_suppression_ledgers,
    write_combined_ledger,
)
from scripts.build_market_state_direct_suppression_ledger import BASELINE_ARM


def _accepted_row(
    *,
    arm: str,
    timestamp: pd.Timestamp,
    symbol: str,
    strategy_id: str,
    rank: float,
    net_return: float,
) -> dict[str, object]:
    return {
        "arm": arm,
        "timestamp": timestamp,
        "symbol": symbol,
        "side": "short",
        "strategy_id": strategy_id,
        "head": strategy_id,
        "normalized_rank_score": rank,
        "effective_rank_score": rank,
        "net_return": net_return,
        "gross_return": net_return + 0.001,
        "net_pnl": net_return * 1000.0,
        "gross_pnl": (net_return + 0.001) * 1000.0,
        "cost_pnl": 1.0,
        "simple_policy_exit_reason": "full_sl" if net_return < 0 else "trailing",
        "position_size": 1000.0,
    }


def _schedule_row(
    *,
    timestamp: pd.Timestamp,
    strategy_id: str,
    arm: str | None,
    fold: int | None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "timestamp": timestamp,
        "strategy_id": strategy_id,
        "head": strategy_id,
        "base_threshold": 0.70,
        "state_threshold": 0.75,
        "raw_state_threshold": 0.75,
        "controller_mode": "threshold_raise_only",
        "threshold_action_enabled": True,
        "force_base_threshold": False,
        "risk_severity": 0.7,
        "controller_reason": "test",
        "prediction_coverage": 1.0,
        "state_ood_share": 0.0,
        "mean_pred_utility": -0.02,
        "mean_pred_full_sl": 0.5,
        "mean_pred_timeout": 0.1,
        "base_candidate_count": 2,
        "frontier_candidate_count": 2,
        "frontier_upper_rank": 0.80,
        "predicted_removed_loss_avoided": 0.02,
        "predicted_removed_winner_sacrificed": 0.0,
        "predicted_action_edge": 0.02,
    }
    if arm is not None:
        row["arm"] = arm
    if fold is not None:
        row["fold"] = fold
    return row


def _write_walkforward_source(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    ts = pd.Timestamp("2026-06-23T00:00:00Z")
    pd.DataFrame(
        [
            _accepted_row(
                arm=BASELINE_ARM,
                timestamp=ts,
                symbol="BTC-PERP",
                strategy_id="short_asset",
                rank=0.72,
                net_return=-0.02,
            ),
            _accepted_row(
                arm=BASELINE_ARM,
                timestamp=ts,
                symbol="ETH-PERP",
                strategy_id="short_asset",
                rank=0.77,
                net_return=0.01,
            ),
        ]
    ).to_parquet(root / "accepted_trades.parquet", index=False)
    pd.DataFrame(
        [
            _schedule_row(
                timestamp=ts,
                strategy_id="short_asset",
                arm="S1_observed_axes_shared_response",
                fold=1,
            )
        ]
    ).to_parquet(root / "strategy_threshold_schedule.parquet", index=False)


def _write_later_source(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    ts = pd.Timestamp("2026-06-24T00:00:00Z")
    pd.DataFrame(
        [
            _accepted_row(
                arm="S1_observed_axes_shared_response__post_selection_overlay",
                timestamp=ts,
                symbol="SOL-PERP",
                strategy_id="short_boll",
                rank=0.73,
                net_return=-0.03,
            )
        ]
    ).to_parquet(root / "accepted_trades.parquet", index=False)
    pd.DataFrame(
        [
            _schedule_row(
                timestamp=ts,
                strategy_id="short_boll",
                arm=None,
                fold=None,
            )
        ]
    ).to_parquet(root / "strategy_threshold_schedule.parquet", index=False)
    (root / "manifest.json").write_text(
        '{"selected_arm":"S1_observed_axes_shared_response__post_selection_overlay"}\n',
        encoding="utf-8",
    )


def test_aggregate_direct_suppression_ledgers_normalizes_later_score_dirs(
    tmp_path: Path,
) -> None:
    walk = tmp_path / "walk"
    later = tmp_path / "later"
    out = tmp_path / "out"
    _write_walkforward_source(walk)
    _write_later_source(later)

    ledger, by_group, by_strategy, by_source, by_source_strategy, summary = (
        aggregate_direct_suppression_ledgers(
        [
            {
                "source_dir": str(walk),
                "source_kind": "walkforward",
                "source_window_id": "walk",
                "accepted_arm_mode": "filter_baseline_arm",
            },
            {
                "source_dir": str(later),
                "source_kind": "later_shadow",
                "source_window_id": "later",
                "accepted_arm_mode": "all_accepted_as_baseline",
                "controller_arm_fallback": "S1_observed_axes_shared_response__post_selection_overlay",
            },
        ]
    )
    )

    assert summary["aggregation_contract"] == "combined_direct_accepted_frontier_training_ledger_v1"
    assert summary["row_count"] == 3
    assert summary["source_with_rows_count"] == 2
    assert sorted(ledger["fold"].unique().tolist()) == [1, 2]
    assert set(ledger["source_kind"]) == {"walkforward", "later_shadow"}
    assert int(ledger["direct_suppression_profitable"].sum()) == 2
    assert int(by_source["rows"].sum()) == 3
    assert set(by_group["head"]) == {"short_asset", "short_boll"}
    assert set(by_strategy["strategy_id"]) == {"short_asset", "short_boll"}
    assert set(by_source_strategy["strategy_id"]) == {"short_asset", "short_boll"}

    outputs = write_combined_ledger(
        ledger,
        by_group,
        by_strategy,
        by_source,
        by_source_strategy,
        summary,
        out,
    )
    for path in outputs.values():
        assert Path(path).exists()
    assert "by_strategy_csv" in outputs
    assert "by_source_strategy_csv" in outputs
