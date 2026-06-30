from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.reselect_market_state_accepted_frontier import reselect_accepted_frontier


def _aggregate_row(
    arm: str,
    *,
    median: float,
    q25: float,
    positive_share: float,
    overlay: bool = False,
    baseline: bool = False,
) -> dict:
    return {
        "arm": arm,
        "folds": 3,
        "median_delta_net_pnl": median,
        "mean_delta_net_pnl": median,
        "q25_delta_net_pnl": q25,
        "positive_delta_share": positive_share,
        "median_delta_max_drawdown": 0.0,
        "median_delta_worst_24h": 0.0,
        "median_trade_count": 10,
        "median_trade_retention_share": 1.0,
        "median_delta_full_sl_rate": -0.01,
        "base_arm": arm.replace("__post_selection_overlay", ""),
        "is_post_selection_overlay": overlay,
        "is_baseline": baseline,
        "complexity": 99 if baseline else 1,
    }


def _suppression_row(
    arm: str,
    *,
    suppressed: int,
    defensive_success: float,
    positive_share: float,
) -> dict:
    return {
        "arm": arm,
        "scope": "all",
        "scope_value": "all",
        "suppressed_candidates": suppressed,
        "realized_defensive_success": defensive_success,
        "positive_suppression_fold_share": positive_share,
        "suppressed_loss_avoided": max(defensive_success, 0.0),
        "suppressed_winner_pnl_sacrificed": max(-defensive_success, 0.0),
    }


def _write_source(
    root: Path,
    *,
    overlay_direct_success: float,
    overlay_direct_share: float,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    full = "S1_observed_axes_shared_response"
    overlay = f"{full}__post_selection_overlay"
    pd.DataFrame(
        [
            _aggregate_row(
                "S0_baseline_static_thresholds",
                median=0.0,
                q25=0.0,
                positive_share=0.0,
                baseline=True,
            ),
            _aggregate_row(full, median=0.0, q25=-1.0, positive_share=1 / 3),
            _aggregate_row(overlay, median=10.0, q25=5.0, positive_share=1.0, overlay=True),
        ]
    ).to_csv(root / "walkforward_aggregate_delta.csv", index=False)
    pd.DataFrame(
        [
            _suppression_row(full, suppressed=10, defensive_success=5.0, positive_share=1.0),
            _suppression_row(overlay, suppressed=0, defensive_success=0.0, positive_share=0.0),
        ]
    ).to_csv(root / "walkforward_threshold_candidate_suppression_aggregate.csv", index=False)
    pd.DataFrame(
        [
            _suppression_row(full, suppressed=1, defensive_success=0.01, positive_share=1.0),
            _suppression_row(
                overlay,
                suppressed=1 if overlay_direct_success > 0 else 0,
                defensive_success=overlay_direct_success,
                positive_share=overlay_direct_share,
            ),
        ]
    ).to_csv(root / "walkforward_threshold_baseline_accepted_suppression_aggregate.csv", index=False)
    pd.DataFrame(
        [
            {
                "arm": overlay,
                "scope": "all",
                "scope_value": "all",
                "folds_with_action": 3,
                "action_entrants": 0,
                "action_removed": 5,
                "action_removed_loss_avoided": 20.0,
                "action_removed_winner_pnl_sacrificed": 0.0,
                "action_defensive_success": 20.0,
                "positive_action_fold_share": 1.0,
                "mean_action_net_pnl_delta": 20.0,
            }
        ]
    ).to_csv(root / "walkforward_threshold_action_utility_aggregate.csv", index=False)
    pd.DataFrame(
        [
            {
                "arm": full,
                "mean_prediction_coverage": 1.0,
                "mean_state_ood_share": 0.0,
                "force_base_share": 0.0,
            },
            {
                "arm": overlay,
                "mean_prediction_coverage": 1.0,
                "mean_state_ood_share": 0.0,
                "force_base_share": 0.0,
            },
        ]
    ).to_csv(root / "walkforward_controller_state_diagnostics.csv", index=False)
    (root / "walkforward_selected_controller_candidate.json").write_text(
        json.dumps(
            {
                "selected_arm": overlay,
                "selection_policy": {
                    "min_positive_delta_share": 0.5,
                    "min_median_delta_net_pnl": 0.0,
                    "min_q25_delta_net_pnl": 0.0,
                    "min_defensive_success": 0.0,
                    "min_positive_suppression_share": 0.5,
                    "max_mean_state_ood_share": 0.1,
                    "min_median_delta_max_drawdown": 0.0,
                    "min_median_delta_worst_24h": 0.0,
                    "max_median_delta_full_sl_rate": 0.0,
                    "min_median_trade_retention_share": 0.8,
                    "median_delta_tie_abs_tol": 1.0,
                    "median_delta_tie_rel_tol": 0.05,
                    "require_post_selection_confirmation": True,
                    "select_no_backfill_overlay_only": True,
                },
            }
        ),
        encoding="utf-8",
    )


def test_accepted_frontier_reselection_rejects_action_only_overlay(tmp_path: Path) -> None:
    source = tmp_path / "source"
    out = tmp_path / "out"
    _write_source(source, overlay_direct_success=0.0, overlay_direct_share=0.0)

    payload = reselect_accepted_frontier(source, out)

    assert payload["selected_arm"] is None
    assert payload["reason"] == "no_arm_passed_selection_gates"
    assert payload["selection_policy"]["suppression_gate_source"] == "baseline_accepted_suppression"
    assert payload["selection_policy"]["overlay_gate_uses_action_metrics"] is False
    table = pd.read_csv(out / "accepted_frontier_controller_candidate_selection.csv")
    overlay = table.loc[
        table["arm"].eq("S1_observed_axes_shared_response__post_selection_overlay")
    ].iloc[0]
    assert "defensive_success_not_positive" in overlay["selection_fail_reasons"]
    assert "suppression_not_recurrent" in overlay["selection_fail_reasons"]
    assert (out / "accepted_frontier_reselection_report.md").exists()


def test_accepted_frontier_reselection_can_select_direct_recurrent_overlay(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    out = tmp_path / "out"
    _write_source(source, overlay_direct_success=2.0, overlay_direct_share=1.0)

    payload = reselect_accepted_frontier(source, out)

    assert payload["selected_arm"] == "S1_observed_axes_shared_response__post_selection_overlay"
    assert payload["selected_metrics"]["realized_defensive_success"] == 2.0
    assert payload["selection_policy"]["suppression_gate_source"] == "baseline_accepted_suppression"
    summary = pd.read_csv(out / "accepted_frontier_selection_summary.csv")
    selected = summary.loc[
        summary["arm"].eq("S1_observed_axes_shared_response__post_selection_overlay")
    ].iloc[0]
    assert bool(selected["passed_selection_gates"])
