from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.audit_market_state_head_priority_window_transfer import load_window


def _write_window(root: Path) -> None:
    root.mkdir(parents=True)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "candidate_universe": {
                    "rows": 10,
                    "timestamp_count": 2,
                    "timestamp_min": "2026-06-23T00:00:00+00:00",
                    "timestamp_max": "2026-06-23T01:00:00+00:00",
                }
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "arm": ["P0_static_priority", "L1_lgbm_learned_priority"],
            "trade_count": [3, 4],
            "net_pnl": [10.0, 14.0],
            "full_sl_rate": [0.2, 0.1],
            "timeout_rate": [0.0, 0.1],
        }
    ).to_csv(root / "head_priority_learning_replay_summary.csv", index=False)
    pd.DataFrame(
        {
            "arm": ["L1_lgbm_learned_priority"],
            "jaccard_vs_baseline": [0.95],
        }
    ).to_csv(root / "head_priority_learning_accepted_overlap.csv", index=False)
    pd.DataFrame(
        {
            "arm": ["L1_lgbm_learned_priority"],
            "scope": ["all"],
            "entrants": [2],
            "removed": [1],
            "entrant_net_pnl": [5.0],
            "removed_net_pnl": [1.0],
            "net_replacement_pnl": [4.0],
            "net_action_pnl_delta": [4.0],
            "defensive_success": [2.0],
        }
    ).to_csv(root / "head_priority_learning_accepted_swap_utility.csv", index=False)
    pd.DataFrame(
        {
            "arm": ["P0_static_priority", "L1_lgbm_learned_priority"],
            "head": ["short_boll", "short_boll"],
            "trade_count": [1, 2],
            "win_rate": [0.0, 0.5],
            "net_pnl": [2.0, 6.0],
            "full_sl_rate": [0.0, 0.0],
            "timeout_rate": [0.0, 0.0],
        }
    ).to_csv(root / "head_priority_learning_by_head.csv", index=False)
    pd.DataFrame(
        {
            "arm": ["L1_lgbm_learned_priority"],
            "selection_gate_passed": [True],
            "selection_objective": [0.7],
            "config_max_adjustment": [0.6],
        }
    ).to_csv(root / "head_priority_learning_model_diagnostics.csv", index=False)
    pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-06-23T00:00:00Z", "2026-06-23T00:00:00Z"],
                utc=True,
            ),
            "head": ["short_asset", "short_boll"],
            "portfolio_priority_adjustment": [-0.1, 0.1],
            "coverage": [1.0, 1.0],
        }
    ).to_parquet(root / "head_priority_learned_schedule.parquet", index=False)


def test_load_window_computes_priority_transfer_deltas(tmp_path: Path) -> None:
    root = tmp_path / "window"
    _write_window(root)

    row, by_head = load_window(root, label="window")

    assert row["delta_net_pnl"] == 4.0
    assert row["delta_full_sl_rate"] == -0.1
    assert row["delta_timeout_rate"] == 0.1
    assert row["accepted_jaccard"] == 0.95
    assert row["entrants"] == 2
    assert row["removed"] == 1
    assert row["coverage"] == 1.0
    assert by_head.iloc[0]["delta_net_pnl"] == 4.0
