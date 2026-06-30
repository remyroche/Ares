from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from scripts.audit_market_state_priority_actionability import (
    audit_cap_sweep_dir,
    frontier_blockers,
    frontier_blocker_summary,
    main as actionability_main,
    timestamp_actionability,
)


def _decisions(*, shadow: bool, changed: bool = False) -> pd.DataFrame:
    timestamp = pd.Timestamp("2026-06-20T00:00:00Z")
    accepted = [True, False, False]
    if shadow and changed:
        accepted = [False, True, False]
    priorities = [1.00, 0.90, 0.20]
    if shadow:
        priorities = [0.95, 1.05, 0.20]
    return pd.DataFrame(
        {
            "candidate_index": [0, 1, 2],
            "timestamp": [timestamp, timestamp, timestamp],
            "symbol": ["A", "B", "C"],
            "side": ["short", "short", "short"],
            "strategy_id": ["s0", "s1", "s2"],
            "portfolio_priority": priorities,
            "accepted": accepted,
            "rejection_reason": [
                "accepted" if value else "max_new_entries_per_bar_reached"
                for value in accepted
            ],
        }
    )


def _candidates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "head": ["short_asset", "short_boll", "short_asset"],
            "net_return": [0.01, -0.02, 0.03],
            "portfolio_priority_adjustment": [-0.05, 0.15, -0.05],
        }
    )


def _schedule() -> pd.DataFrame:
    timestamp = pd.Timestamp("2026-06-20T00:00:00Z")
    return pd.DataFrame(
        {
            "timestamp": [timestamp, timestamp],
            "head": ["short_asset", "short_boll"],
            "portfolio_priority_adjustment": [-0.05, 0.15],
            "priority_arm": ["cap", "cap"],
        }
    )


def test_timestamp_actionability_marks_active_frontier_cross_without_trade_change() -> None:
    rows = timestamp_actionability(
        static_decisions=_decisions(shadow=False),
        shadow_decisions=_decisions(shadow=True, changed=False),
        shadow_candidates=_candidates(),
        schedule=_schedule(),
        window_label="w",
    )

    assert len(rows) == 1
    row = rows.iloc[0]
    assert row["active_schedule_rows"] == 2
    assert bool(row["accepted_set_changed"]) is False
    assert bool(row["active_no_action"]) is True
    assert bool(row["frontier_crossed_on_static_set"]) is True
    assert row["frontier_gap_static"] > 0
    assert row["frontier_gap_after_adjustment_on_static_set"] < 0


def test_timestamp_actionability_counts_entrants_and_removed() -> None:
    rows = timestamp_actionability(
        static_decisions=_decisions(shadow=False),
        shadow_decisions=_decisions(shadow=True, changed=True),
        shadow_candidates=_candidates(),
        schedule=_schedule(),
        window_label="w",
    )

    row = rows.iloc[0]
    assert bool(row["accepted_set_changed"]) is True
    assert row["entrants"] == 1
    assert row["removed"] == 1
    assert bool(row["active_no_action"]) is False


def test_frontier_blockers_classifies_top_rejected_rows() -> None:
    blockers = frontier_blockers(
        static_decisions=_decisions(shadow=False),
        shadow_decisions=_decisions(shadow=True, changed=False),
        shadow_candidates=_candidates(),
        schedule=_schedule(),
        window_label="w",
        top_n_rejected_per_timestamp=2,
    )
    summary = frontier_blocker_summary(blockers)

    assert len(blockers) == 2
    top = blockers.sort_values("frontier_rejected_rank").iloc[0]
    assert top["static_rejection_reason"] == "max_new_entries_per_bar_reached"
    assert top["blocker_category"] == "ordering_capacity_blocked"
    assert bool(top["above_static_frontier"]) is False
    assert "ordering_capacity_blocked" in set(summary["blocker_category"])


def test_audit_cap_sweep_dir_reads_window_artifacts(tmp_path) -> None:
    cap_dir = tmp_path / "cap"
    cap_dir.mkdir()
    arm = "L1_lgbm_learned_priority_cap_0p15_zge_0p5"
    pd.DataFrame(
        {
            "arm": [arm],
            "delta_net_pnl": [1.25],
            "accepted_jaccard": [0.95],
            "coverage": [1.0],
            "full_sl_rate": [0.1],
            "delta_full_sl_rate": [-0.01],
            "timeout_rate": [0.2],
            "delta_timeout_rate": [0.0],
            "defensive_success": [0.5],
        }
    ).to_csv(cap_dir / "head_priority_cap_sweep_metrics.csv", index=False)
    _schedule().to_parquet(cap_dir / "head_priority_cap_sweep_schedules.parquet", index=False)
    _decisions(shadow=False).to_parquet(cap_dir / "P0_static_priority_decisions.parquet", index=False)
    _decisions(shadow=True, changed=True).to_parquet(cap_dir / f"{arm}_decisions.parquet", index=False)
    _candidates().to_parquet(cap_dir / f"{arm}_candidates.parquet", index=False)

    summary, by_timestamp, by_head, blockers = audit_cap_sweep_dir(
        cap_dir,
        arm_contains="cap_0p15_zge_0p5",
        window_label="window",
    )

    assert summary["delta_net_pnl"].iloc[0] == 1.25
    assert summary["action_timestamps"].iloc[0] == 1
    assert by_timestamp["entrants"].iloc[0] == 1
    assert set(by_head["head"]) == {"short_asset", "short_boll"}
    assert not blockers.empty


def test_actionability_main_can_use_selected_challenger(tmp_path: Path, monkeypatch) -> None:
    cap_dir = tmp_path / "cap"
    cap_dir.mkdir()
    default_arm = "L1_lgbm_learned_priority_cap_0p10_zge_0p5"
    selected_arm = "L1_lgbm_learned_priority_cap_0p15_zge_0p5"
    pd.DataFrame(
        {
            "arm": [default_arm, selected_arm],
            "delta_net_pnl": [-1.0, 1.25],
            "accepted_jaccard": [0.90, 0.95],
            "coverage": [1.0, 1.0],
            "full_sl_rate": [0.11, 0.10],
            "delta_full_sl_rate": [0.01, -0.01],
            "timeout_rate": [0.20, 0.20],
            "delta_timeout_rate": [0.0, 0.0],
            "defensive_success": [-0.5, 0.5],
        }
    ).to_csv(cap_dir / "head_priority_cap_sweep_metrics.csv", index=False)
    (cap_dir / "selected_shadow_challenger.json").write_text(
        json.dumps({"selected": True, "arm": selected_arm}),
        encoding="utf-8",
    )
    _schedule().to_parquet(cap_dir / "head_priority_cap_sweep_schedules.parquet", index=False)
    _decisions(shadow=False).to_parquet(cap_dir / "P0_static_priority_decisions.parquet", index=False)
    for arm in [default_arm, selected_arm]:
        _decisions(shadow=True, changed=arm == selected_arm).to_parquet(
            cap_dir / f"{arm}_decisions.parquet",
            index=False,
        )
        _candidates().to_parquet(cap_dir / f"{arm}_candidates.parquet", index=False)
    out_dir = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_market_state_priority_actionability.py",
            "--cap-sweep-dir",
            str(cap_dir),
            "--use-selected-challenger",
            "--output-dir",
            str(out_dir),
        ],
    )

    actionability_main()

    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    summary = pd.read_csv(out_dir / "market_state_priority_actionability_by_window.csv")
    assert manifest["params"]["resolved_arm_contains"] == "cap_0p15_zge_0p5"
    assert manifest["params"]["arm_selector_source"].startswith("selected_shadow_challenger:")
    assert summary["arm"].iloc[0] == selected_arm
