from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.report_market_state_rank_scope_blend_validation import (
    aggregate_rank_scope_windows,
    summarize_rank_scope_window,
)


def _write_window(root: Path, label: str, *, blend_delta: float, blend_jaccard: float = 0.80) -> None:
    root.mkdir(parents=True)
    base_net = 100.0
    summary = pd.DataFrame(
        [
            {
                "arm": "R0_t1_timestamp_short_boll",
                "trade_count": 10,
                "net_pnl": base_net,
                "full_sl_rate": 0.20,
                "timeout_rate": 0.10,
                "worst_24h_net_pnl": -10.0,
            },
            {
                "arm": "R1_global_short_boll",
                "trade_count": 9,
                "net_pnl": base_net - 5.0,
                "full_sl_rate": 0.20,
                "timeout_rate": 0.10,
                "worst_24h_net_pnl": -11.0,
            },
            {
                "arm": "R2_state_switch_short_boll",
                "trade_count": 10,
                "net_pnl": base_net + 1.0,
                "full_sl_rate": 0.20,
                "timeout_rate": 0.10,
                "worst_24h_net_pnl": -9.0,
            },
            {
                "arm": "R3_state_blended_short_boll",
                "trade_count": 10,
                "net_pnl": base_net + float(blend_delta),
                "full_sl_rate": 0.20,
                "timeout_rate": 0.10,
                "worst_24h_net_pnl": -8.0,
            },
        ]
    )
    summary.to_csv(root / "rank_scope_switch_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "arm": "R0_t1_timestamp_short_boll",
                "jaccard_vs_baseline": 1.0,
                "arm_only": 0,
                "baseline_only": 0,
            },
            {
                "arm": "R1_global_short_boll",
                "jaccard_vs_baseline": 0.60,
                "arm_only": 2,
                "baseline_only": 3,
            },
            {
                "arm": "R2_state_switch_short_boll",
                "jaccard_vs_baseline": 0.70,
                "arm_only": 2,
                "baseline_only": 2,
            },
            {
                "arm": "R3_state_blended_short_boll",
                "jaccard_vs_baseline": blend_jaccard,
                "arm_only": 1,
                "baseline_only": 1,
            },
        ]
    ).to_csv(root / "rank_scope_switch_accepted_overlap.csv", index=False)
    pd.DataFrame(
        [
            {
                "arm": "R3_state_blended_short_boll",
                "timestamp": "2026-06-23T00:00:00+00:00",
                "short_boll_rank_scope": "state_blend",
                "short_boll_timestamp_weight": 0.5,
            },
            {
                "arm": "R3_state_blended_short_boll",
                "timestamp": "2026-06-23T01:00:00+00:00",
                "short_boll_rank_scope": "state_blend",
                "short_boll_timestamp_weight": 0.9,
            },
        ]
    ).to_csv(root / "rank_scope_switch_schedule.csv", index=False)
    (root / "rank_scope_switch_manifest.json").write_text(
        json.dumps(
            {
                "params": {"evaluation_period": label},
                "contract": {
                    "production_eligible": False,
                    "promotion_status": "shadow_only",
                },
                "rank_contract_candidate_parity": {"passed": True},
            }
        )
    )


def test_summarize_rank_scope_window_reports_blend_delta(tmp_path: Path) -> None:
    window = tmp_path / "window"
    _write_window(window, "jun23_00_08", blend_delta=3.5)

    row = summarize_rank_scope_window(window)

    assert row["window_label"] == "jun23_00_08"
    assert row["blend_delta_net_pnl"] == 3.5
    assert row["blend_accepted_jaccard_vs_t1"] == 0.80
    assert row["blend_timestamp_weight_mean"] == 0.70


def test_aggregate_blocks_promotion_when_later_windows_are_mixed(tmp_path: Path) -> None:
    dev = tmp_path / "jun15_22"
    later_a = tmp_path / "jun23_a"
    later_b = tmp_path / "jun23_b"
    _write_window(dev, "jun15_22", blend_delta=20.0)
    _write_window(later_a, "jun23_00_08", blend_delta=0.0)
    _write_window(later_b, "jun23_09_jun24_08", blend_delta=-5.0)

    payload = aggregate_rank_scope_windows(
        [dev, later_a, later_b],
        output_dir=tmp_path / "out",
        min_later_windows=2,
        min_positive_later_share=0.75,
        min_accepted_jaccard=0.75,
    )
    rollup = payload["rollup"]

    assert rollup["later_window_count"] == 2
    assert rollup["later_blend_positive_delta_share"] == 0.0
    assert rollup["shadow_promotion_gate_passed"] is False
    assert "later_positive_delta_share_below_gate" in rollup["shadow_promotion_failures"]
    assert "later_median_delta_not_positive" in rollup["shadow_promotion_failures"]
    assert (tmp_path / "out" / "rank_scope_blend_validation_report.md").exists()
