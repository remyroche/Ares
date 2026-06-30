from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.report_market_state_rank_router_priority_validation import aggregate_windows


def _write_window(path: Path, *, label: str, base: float, combo: float) -> Path:
    root = path / label
    root.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "arm": "R0_t1_timestamp_short_boll",
                "trade_count": 10,
                "net_pnl": base,
                "full_sl_rate": 0.1,
                "timeout_rate": 0.2,
            },
            {
                "arm": "R3_state_blended_short_boll",
                "trade_count": 10,
                "net_pnl": base + 1.0,
                "full_sl_rate": 0.1,
                "timeout_rate": 0.2,
            },
            {
                "arm": "R4_t1_timestamp_plus_priority",
                "trade_count": 10,
                "net_pnl": base + 2.0,
                "full_sl_rate": 0.1,
                "timeout_rate": 0.2,
            },
            {
                "arm": "R5_state_blended_plus_priority",
                "trade_count": 10,
                "net_pnl": combo,
                "full_sl_rate": 0.1,
                "timeout_rate": 0.2,
            },
        ]
    ).to_csv(root / "rank_scope_switch_summary.csv", index=False)
    pd.DataFrame(
        [
            {"arm": "R3_state_blended_short_boll", "jaccard_vs_baseline": 0.9},
            {"arm": "R4_t1_timestamp_plus_priority", "jaccard_vs_baseline": 0.95},
            {"arm": "R5_state_blended_plus_priority", "jaccard_vs_baseline": 0.85},
        ]
    ).to_csv(root / "rank_scope_switch_accepted_overlap.csv", index=False)
    return root


def test_rank_router_priority_validation_fails_when_later_combo_underperforms(tmp_path: Path) -> None:
    dev = _write_window(tmp_path, label="jun15_22", base=100.0, combo=120.0)
    later_a = _write_window(tmp_path, label="jun23_00_08", base=10.0, combo=10.0)
    later_b = _write_window(tmp_path, label="jun23_09_24_08", base=10.0, combo=5.0)

    payload = aggregate_windows(
        [dev, later_a, later_b],
        output_dir=tmp_path / "out",
        development_labels={"jun15_22"},
    )

    rollup = payload["rollup"]
    assert rollup["rank_plus_priority_promotion_gate_passed"] is False
    assert rollup["rank_plus_priority_should_remain_shadow"] is True
    assert "rank_plus_priority_later_median_delta_not_positive" in rollup[
        "rank_plus_priority_failures"
    ]
    assert (tmp_path / "out" / "rank_router_priority_validation_report.md").exists()
