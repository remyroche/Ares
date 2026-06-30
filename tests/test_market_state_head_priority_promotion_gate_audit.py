from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.audit_market_state_head_priority_promotion_gates import audit_priority_gates


def _write_priority_artifact(
    root: Path,
    *,
    selection_gate_passed: bool,
    net_delta: float = 10.0,
    full_sl_delta: float = 0.0,
    timeout_delta: float = 0.0,
    entrant_net_pnl: float = 8.0,
    removed_net_pnl: float = 1.0,
) -> None:
    root.mkdir(parents=True)
    base_net = 100.0
    pd.DataFrame(
        [
            {
                "arm": "P0_static_priority",
                "trade_count": 10,
                "net_pnl": base_net,
                "full_sl_rate": 0.20,
                "timeout_rate": 0.10,
            },
            {
                "arm": "L0_selected_lgbm_priority",
                "trade_count": 10,
                "net_pnl": base_net + float(net_delta),
                "full_sl_rate": 0.20 + float(full_sl_delta),
                "timeout_rate": 0.10 + float(timeout_delta),
            },
        ]
    ).to_csv(root / "head_priority_learning_replay_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "arm": "L0_selected_lgbm_priority",
                "jaccard_vs_baseline": 0.95,
                "baseline_only": 1,
                "arm_only": 1,
            }
        ]
    ).to_csv(root / "head_priority_learning_accepted_overlap.csv", index=False)
    pd.DataFrame(
        [
            {
                "arm": "L0_selected_lgbm_priority",
                "scope": "all",
                "scope_value": "all",
                "entrants": 1,
                "removed": 1,
                "entrant_net_pnl": float(entrant_net_pnl),
                "removed_net_pnl": float(removed_net_pnl),
                "net_replacement_pnl": float(entrant_net_pnl) - float(removed_net_pnl),
                "same_key_net_pnl_delta": float(net_delta) - (float(entrant_net_pnl) - float(removed_net_pnl)),
                "net_action_pnl_delta": float(net_delta),
                "removed_loss_avoided": 0.0,
                "removed_winner_pnl_sacrificed": float(max(removed_net_pnl, 0.0)),
                "defensive_success": -float(max(removed_net_pnl, 0.0)),
            }
        ]
    ).to_csv(root / "head_priority_learning_accepted_swap_utility.csv", index=False)
    pd.DataFrame(
        [
            {
                "arm": "L0_selected_lgbm_priority",
                "selection_gate_passed": bool(selection_gate_passed),
                "selection_objective": 0.75,
                "fold_action_positive_delta_share": 1.0,
                "fold_mean_action_utility_delta": 0.01,
            }
        ]
    ).to_csv(root / "head_priority_learning_model_diagnostics.csv", index=False)
    pd.DataFrame(
        [
            {
                "selection_gate_passed": bool(selection_gate_passed),
                "selection_objective": 0.75,
            }
        ]
    ).to_csv(root / "head_priority_config_selection.csv", index=False)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "generated_by": "run_market_state_head_priority_learning",
                "params": {"selection_gate_mode": "defensive"},
            }
        ),
        encoding="utf-8",
    )


def test_head_priority_gate_promotes_candidate_with_selection_and_swap_quality(tmp_path: Path) -> None:
    priority_dir = tmp_path / "priority"
    _write_priority_artifact(priority_dir, selection_gate_passed=True)

    report = audit_priority_gates(priority_dir)

    assert report["passing_candidate_count"] == 1
    assert report["single_window_replay_gate_passed"] is True
    assert report["production_passing_candidate_count"] == 0
    assert report["priority_should_remain_shadow"] is True
    assert "fewer_than_3_replay_windows" in report["production_blockers"]
    best = report["best_raw_candidate"]
    assert best["candidate_promotable"] is True
    assert best["fail_reasons"] == ""


def test_head_priority_gate_rejects_forced_positive_candidate_without_selection_gate(
    tmp_path: Path,
) -> None:
    priority_dir = tmp_path / "priority"
    _write_priority_artifact(priority_dir, selection_gate_passed=False, net_delta=20.0)

    report = audit_priority_gates(priority_dir)

    assert report["passing_candidate_count"] == 0
    assert report["priority_should_remain_shadow"] is True
    best = report["best_raw_candidate"]
    assert best["net_pnl_delta"] == 20.0
    assert "selection_gate_not_passed" in best["fail_reasons"]


def test_head_priority_gate_auto_detects_opportunity_mode_from_manifest(tmp_path: Path) -> None:
    priority_dir = tmp_path / "priority"
    _write_priority_artifact(
        priority_dir,
        selection_gate_passed=True,
        net_delta=15.0,
        timeout_delta=0.006,
    )
    (priority_dir / "manifest.json").write_text(
        json.dumps(
            {
                "generated_by": "run_market_state_head_priority_learning",
                "params": {"selection_gate_mode": "opportunity"},
            }
        ),
        encoding="utf-8",
    )

    defensive = audit_priority_gates(priority_dir, gate_mode="defensive")
    opportunity = audit_priority_gates(priority_dir)

    assert defensive["passing_candidate_count"] == 0
    assert "timeout_rate_worsened" in defensive["best_raw_candidate"]["fail_reasons"]
    assert opportunity["gate_mode"] == "opportunity"
    assert opportunity["passing_candidate_count"] == 1
    assert opportunity["single_window_replay_gate_passed"] is True


def test_head_priority_gate_blocks_rank_prior_june_broad_candidate_from_production(
    tmp_path: Path,
) -> None:
    priority_dir = tmp_path / "priority"
    _write_priority_artifact(priority_dir, selection_gate_passed=True)
    (priority_dir / "manifest.json").write_text(
        json.dumps(
            {
                "generated_by": "run_market_state_head_priority_learning",
                "params": {"selection_gate_mode": "opportunity"},
                "contract": {
                    "changes_scores_or_ranks": True,
                    "changes_thresholds": False,
                    "changes_position_sizing": False,
                    "execution_enabled": False,
                    "operational_status": "shadow_only",
                    "qfail_active": False,
                    "head_health_active": False,
                    "market_state_threshold_controller_active": False,
                },
                "static_baseline_candidate_parity": {
                    "promotion_grade_scope": False,
                },
                "candidate_universe": {
                    "timestamp_min": "2026-06-15T17:00:00+00:00",
                    "timestamp_max": "2026-06-22T23:00:00+00:00",
                },
            }
        ),
        encoding="utf-8",
    )

    report = audit_priority_gates(priority_dir)

    assert report["passing_candidate_count"] == 1
    assert report["production_passing_candidate_count"] == 0
    assert report["priority_should_remain_shadow"] is True
    blockers = set(report["production_blockers"])
    assert "changes_scores_or_ranks_rank_prior_shadow_only" in blockers
    assert "candidate_universe_not_promotion_grade" in blockers
    assert "june_15_22_development_window_not_promotion_oos" in blockers


def test_head_priority_gate_handles_no_selected_candidate_empty_diagnostics(
    tmp_path: Path,
) -> None:
    priority_dir = tmp_path / "priority"
    priority_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "arm": "P0_static_priority",
                "trade_count": 10,
                "net_pnl": 100.0,
                "full_sl_rate": 0.20,
                "timeout_rate": 0.10,
            }
        ]
    ).to_csv(priority_dir / "head_priority_learning_replay_summary.csv", index=False)
    (priority_dir / "head_priority_learning_model_diagnostics.csv").write_text(
        "",
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "selection_gate_passed": False,
                "selection_objective": 0.50,
            }
        ]
    ).to_csv(priority_dir / "head_priority_config_selection.csv", index=False)
    pd.DataFrame().to_csv(priority_dir / "head_priority_learning_accepted_overlap.csv", index=False)
    pd.DataFrame().to_csv(
        priority_dir / "head_priority_learning_accepted_swap_utility.csv",
        index=False,
    )
    (priority_dir / "manifest.json").write_text(
        json.dumps(
            {
                "generated_by": "run_market_state_head_priority_learning",
                "params": {"selection_gate_mode": "opportunity"},
            }
        ),
        encoding="utf-8",
    )

    report = audit_priority_gates(priority_dir)

    assert report["candidate_count"] == 0
    assert report["passing_candidate_count"] == 0
    assert report["single_window_replay_gate_passed"] is False
    assert report["priority_should_remain_shadow"] is True
    assert report["best_raw_candidate"] == {}
