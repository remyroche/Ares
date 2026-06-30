import json
from pathlib import Path

import pandas as pd

from scripts.audit_market_state_controller_promotion_gates import (
    _controller_arm_complexity,
    audit_promotion_gates,
)


def _selection_row(
    arm: str,
    *,
    median: float,
    q25: float,
    positive_share: float,
    defensive_success: float,
    suppression_share: float,
    post_defensive_success: float,
    post_suppression_share: float,
    mean_ood: float = 0.0,
    delta_max_drawdown: float = 0.0,
    delta_worst_24h: float = 0.0,
    trade_retention: float = 1.0,
    delta_full_sl: float = -0.01,
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
        "median_delta_max_drawdown": delta_max_drawdown,
        "median_delta_worst_24h": delta_worst_24h,
        "median_trade_retention_share": trade_retention,
        "median_delta_full_sl_rate": delta_full_sl,
        "base_arm": arm.replace("__post_selection_overlay", ""),
        "is_post_selection_overlay": overlay,
        "is_baseline": baseline,
        "complexity": 99 if baseline else 1,
        "suppressed_candidates": 10,
        "realized_defensive_success": defensive_success,
        "positive_suppression_fold_share": suppression_share,
        "suppressed_loss_avoided": 1.0,
        "suppressed_winner_pnl_sacrificed": 0.5,
        "post_selection_median_delta_net_pnl": median,
        "post_selection_mean_delta_net_pnl": median,
        "post_selection_q25_delta_net_pnl": q25,
        "post_selection_positive_delta_share": positive_share,
        "post_selection_median_delta_max_drawdown": delta_max_drawdown,
        "post_selection_median_delta_worst_24h": delta_worst_24h,
        "post_selection_median_trade_retention_share": trade_retention,
        "post_selection_median_delta_full_sl_rate": delta_full_sl,
        "post_selection_realized_defensive_success": post_defensive_success,
        "post_selection_positive_suppression_fold_share": post_suppression_share,
        "post_selection_suppressed_loss_avoided": 1.0,
        "post_selection_suppressed_winner_pnl_sacrificed": 0.5,
        "mean_prediction_coverage": 1.0,
        "mean_state_ood_share": mean_ood,
        "max_state_ood_share": mean_ood,
        "mean_force_base_share": 0.0,
    }


def _write_bundle(
    root: Path,
    rows: list[dict],
    selected_arm=None,
    reason="no_arm_passed_selection_gates",
    policy_updates: dict | None = None,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    policy = {
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
    }
    if policy_updates:
        policy.update(policy_updates)
    frame = pd.DataFrame(rows)
    reasons = []
    passed = []
    scores = []
    from scripts.audit_market_state_controller_promotion_gates import _compute_fail_reasons, _score_row

    for _, row in frame.iterrows():
        row_reasons = _compute_fail_reasons(row, policy)
        reasons.append(";".join(row_reasons))
        passed.append(not row_reasons)
        scores.append(_score_row(row))
    frame["passed_selection_gates"] = passed
    frame["selection_fail_reasons"] = reasons
    frame["selection_score"] = scores
    frame.to_csv(root / "walkforward_controller_candidate_selection.csv", index=False)
    (root / "walkforward_selected_controller_candidate.json").write_text(
        json.dumps({"selected_arm": selected_arm, "reason": reason, "selection_policy": policy}),
        encoding="utf-8",
    )
    (root / "strategy_threshold_controller_config.json").write_text(
        json.dumps({"selection": {"selected_arm": selected_arm, "reason": reason, "selection_policy": policy}}),
        encoding="utf-8",
    )


def test_promotion_gate_audit_accepts_no_passing_controller(tmp_path: Path) -> None:
    _write_bundle(
        tmp_path,
        [
            _selection_row(
                "S1_observed_axes_shared_response",
                median=10.0,
                q25=-1.0,
                positive_share=2 / 3,
                defensive_success=-0.1,
                suppression_share=2 / 3,
                post_defensive_success=-0.2,
                post_suppression_share=0.0,
            )
        ],
    )

    payload, selection, failures = audit_promotion_gates(tmp_path)

    assert failures == []
    assert payload["passed"] is True
    assert payload["promotion_gate_passed"] is False
    assert payload["controller_should_remain_disabled"] is True
    assert bool(selection.loc[0, "recomputed_passed_selection_gates"]) is False
    assert "q25_delta_below_gate" in selection.loc[0, "recomputed_selection_fail_reasons"]


def test_promotion_gate_audit_enriches_freed_capacity_action_metrics(tmp_path: Path) -> None:
    _write_bundle(
        tmp_path,
        [
            _selection_row(
                "S1_observed_axes_shared_response",
                median=10.0,
                q25=-1.0,
                positive_share=2 / 3,
                defensive_success=0.5,
                suppression_share=2 / 3,
                post_defensive_success=-0.2,
                post_suppression_share=0.0,
            ),
            _selection_row(
                "S1_observed_axes_shared_response__post_selection_overlay",
                median=5.0,
                q25=1.0,
                positive_share=1.0,
                defensive_success=-0.2,
                suppression_share=0.0,
                post_defensive_success=-0.2,
                post_suppression_share=0.0,
                overlay=True,
            ),
        ],
    )
    pd.DataFrame(
        [
            {
                "arm": "S1_observed_axes_shared_response",
                "fold": 1,
                "entrants": 2,
                "removed": 2,
                "entrant_net_pnl": 3.0,
                "removed_net_pnl": -1.0,
                "net_replacement_pnl": 4.0,
                "same_key_net_pnl_delta": 0.5,
                "net_action_pnl_delta": 4.5,
            },
            {
                "arm": "S1_observed_axes_shared_response",
                "fold": 2,
                "entrants": 1,
                "removed": 1,
                "entrant_net_pnl": -1.0,
                "removed_net_pnl": 2.0,
                "net_replacement_pnl": -3.0,
                "same_key_net_pnl_delta": 0.0,
                "net_action_pnl_delta": -3.0,
            },
            {
                "arm": "S1_observed_axes_shared_response__post_selection_overlay",
                "fold": 1,
                "entrants": 1,
                "removed": 1,
                "entrant_net_pnl": 1.0,
                "removed_net_pnl": 0.0,
                "net_replacement_pnl": 1.0,
                "same_key_net_pnl_delta": 0.0,
                "net_action_pnl_delta": 1.0,
            },
        ]
    ).to_csv(tmp_path / "strategy_threshold_action_audit.csv", index=False)

    payload, selection, failures = audit_promotion_gates(tmp_path)

    assert failures == []
    row = selection.loc[selection["arm"].eq("S1_observed_axes_shared_response")].iloc[0]
    assert float(row["freed_capacity_entrant_count"]) == 3.0
    assert float(row["freed_capacity_net_replacement_pnl"]) == 1.0
    assert float(row["freed_capacity_net_action_pnl_delta"]) == 1.5
    assert float(row["positive_freed_capacity_fold_share"]) == 0.5
    assert float(row["post_selection_freed_capacity_net_replacement_pnl"]) == 1.0
    assert payload["best_raw_candidate"]["freed_capacity_entrant_count"] == 3.0
    assert payload["best_raw_candidate"]["freed_capacity_net_action_pnl_delta"] == 1.5


def test_promotion_gate_audit_blocks_replacement_dependent_promotion(tmp_path: Path) -> None:
    _write_bundle(
        tmp_path,
        [
            _selection_row(
                "S1_observed_axes_shared_response",
                median=10.0,
                q25=5.0,
                positive_share=1.0,
                defensive_success=0.1,
                suppression_share=1.0,
                post_defensive_success=0.1,
                post_suppression_share=1.0,
            )
        ],
        selected_arm="S1_observed_axes_shared_response",
        reason="selected",
    )
    pd.DataFrame(
        [
            {
                "arm": "S1_observed_axes_shared_response",
                "fold": 1,
                "entrants": 3,
                "removed": 1,
                "entrant_net_pnl": 8.0,
                "removed_net_pnl": 0.0,
                "net_replacement_pnl": 8.0,
                "same_key_net_pnl_delta": 0.0,
                "net_action_pnl_delta": 8.0,
            }
        ]
    ).to_csv(tmp_path / "strategy_threshold_action_audit.csv", index=False)

    payload, selection, failures = audit_promotion_gates(tmp_path)

    assert failures == []
    assert payload["promotion_gate_passed"] is True
    assert payload["controller_promotion_ready"] is False
    assert payload["controller_should_remain_disabled"] is True
    assert payload["action_attribution_gate"]["passed"] is False
    assert "replay_lift_depends_on_replacement_or_backfill" in payload["action_attribution_gate"]["failures"]
    row = selection.loc[selection["arm"].eq("S1_observed_axes_shared_response")].iloc[0]
    assert bool(row["replacement_dependent_lift"]) is True
    assert float(row["direct_suppression_value_share"]) < 0.02


def test_pruned_state_pack_is_known_controller_arm() -> None:
    assert _controller_arm_complexity("S7_pruned_state_pack") == 2


def test_promotion_gate_audit_rejects_wrong_selected_null_when_arm_passes(tmp_path: Path) -> None:
    _write_bundle(
        tmp_path,
        [
            _selection_row(
                "S1_observed_axes_shared_response",
                median=10.0,
                q25=5.0,
                positive_share=1.0,
                defensive_success=1.0,
                suppression_share=1.0,
                post_defensive_success=1.0,
                post_suppression_share=1.0,
            )
        ],
        selected_arm=None,
    )

    payload, _selection, failures = audit_promotion_gates(tmp_path)

    assert payload["promotion_gate_passed"] is True
    assert "selected_arm None != expected 'S1_observed_axes_shared_response'" in failures
    assert payload["passed"] is False


def test_stage1_observed_only_scope_ignores_out_of_scope_selected_forecast_arm(tmp_path: Path) -> None:
    _write_bundle(
        tmp_path,
        [
            _selection_row(
                "S1_observed_axes_shared_response",
                median=8.0,
                q25=4.0,
                positive_share=1.0,
                defensive_success=1.0,
                suppression_share=1.0,
                post_defensive_success=1.0,
                post_suppression_share=1.0,
            ),
            _selection_row(
                "S2_observed_forecast_shared_response",
                median=12.0,
                q25=6.0,
                positive_share=1.0,
                defensive_success=1.0,
                suppression_share=1.0,
                post_defensive_success=1.0,
                post_suppression_share=1.0,
            ),
        ],
        selected_arm="S2_observed_forecast_shared_response",
        reason="selected",
    )

    payload, selection, failures = audit_promotion_gates(
        tmp_path,
        allowed_base_arms={"S1_observed_axes_shared_response"},
        audit_scope_name="stage1_observed_only",
    )

    assert payload["promotion_gate_passed"] is True
    assert payload["expected_selected_arm"] == "S1_observed_axes_shared_response"
    assert payload["audit_scope"]["stored_selected_arm_in_scope"] is False
    assert payload["audit_scope"]["unfiltered_candidate_count"] == 2
    assert payload["audit_scope"]["filtered_candidate_count"] == 1
    assert selection["arm"].tolist() == ["S1_observed_axes_shared_response"]
    assert failures == []


def test_promotion_gate_audit_rejects_stale_fail_reasons(tmp_path: Path) -> None:
    _write_bundle(
        tmp_path,
        [
            _selection_row(
                "S1_observed_axes_shared_response",
                median=10.0,
                q25=-5.0,
                positive_share=1.0,
                defensive_success=1.0,
                suppression_share=1.0,
                post_defensive_success=1.0,
                post_suppression_share=1.0,
            )
        ],
    )
    frame = pd.read_csv(tmp_path / "walkforward_controller_candidate_selection.csv")
    frame.loc[0, "selection_fail_reasons"] = ""
    frame.to_csv(tmp_path / "walkforward_controller_candidate_selection.csv", index=False)

    payload, _selection, failures = audit_promotion_gates(tmp_path)

    assert payload["passed"] is False
    assert any("selection fail reasons mismatch" in failure for failure in failures)


def test_promotion_gate_audit_rejects_unsafe_risk_or_flow_metrics(tmp_path: Path) -> None:
    _write_bundle(
        tmp_path,
        [
            _selection_row(
                "S1_observed_axes_shared_response",
                median=10.0,
                q25=5.0,
                positive_share=1.0,
                defensive_success=1.0,
                suppression_share=1.0,
                post_defensive_success=1.0,
                post_suppression_share=1.0,
                delta_max_drawdown=-0.01,
                delta_worst_24h=-1.0,
                trade_retention=0.5,
                delta_full_sl=0.02,
            )
        ],
    )

    payload, selection, failures = audit_promotion_gates(tmp_path)

    assert failures == []
    assert payload["promotion_gate_passed"] is False
    reasons = selection.loc[0, "recomputed_selection_fail_reasons"]
    assert "max_drawdown_worsened" in reasons
    assert "worst_24h_worsened" in reasons
    assert "full_sl_rate_worsened" in reasons
    assert "insufficient_trade_retention" in reasons


def test_promotion_gate_audit_accepts_no_backfill_overlay_policy(tmp_path: Path) -> None:
    _write_bundle(
        tmp_path,
        [
            _selection_row(
                "S1_observed_axes_shared_response",
                median=0.0,
                q25=-1.0,
                positive_share=1 / 3,
                defensive_success=1.0,
                suppression_share=1.0,
                post_defensive_success=1.0,
                post_suppression_share=1.0,
            ),
            _selection_row(
                "S1_observed_axes_shared_response__post_selection_overlay",
                median=10.0,
                q25=5.0,
                positive_share=1.0,
                defensive_success=1.0,
                suppression_share=1.0,
                post_defensive_success=1.0,
                post_suppression_share=1.0,
                overlay=True,
            ),
        ],
        selected_arm="S1_observed_axes_shared_response__post_selection_overlay",
        reason="selected",
        policy_updates={"select_no_backfill_overlay_only": True},
    )

    payload, selection, failures = audit_promotion_gates(tmp_path)

    assert failures == []
    assert payload["promotion_gate_passed"] is True
    assert payload["expected_selected_arm"] == "S1_observed_axes_shared_response__post_selection_overlay"
    full_replay_reasons = selection.loc[
        selection["arm"].eq("S1_observed_axes_shared_response"),
        "recomputed_selection_fail_reasons",
    ].iloc[0]
    overlay_reasons = selection.loc[
        selection["arm"].eq("S1_observed_axes_shared_response__post_selection_overlay"),
        "recomputed_selection_fail_reasons",
    ].iloc[0]
    assert "full_replay_can_promote_replacements" in full_replay_reasons
    assert "post_selection_overlay_audit_arm" not in overlay_reasons
    assert bool(
        selection.loc[
            selection["arm"].eq("S1_observed_axes_shared_response__post_selection_overlay"),
            "recomputed_passed_selection_gates",
        ].iloc[0]
    )


def test_promotion_gate_audit_rejects_later_action_phases_even_when_metrics_pass(tmp_path: Path) -> None:
    row = _selection_row(
        "S1_observed_axes_shared_response",
        median=10.0,
        q25=5.0,
        positive_share=1.0,
        defensive_success=1.0,
        suppression_share=1.0,
        post_defensive_success=1.0,
        post_suppression_share=1.0,
    )
    row.update(
        {
            "controller_action_scope": "bounded_score_correction",
            "controller_action_phase": "C4",
            "changes_scores_or_ranks": True,
            "changes_auction_ordering": True,
            "changes_position_sizing": True,
            "allows_threshold_reductions": True,
            "promotes_replacement_candidates": True,
        }
    )
    _write_bundle(tmp_path, [row])

    payload, selection, failures = audit_promotion_gates(tmp_path)

    assert failures == []
    assert payload["promotion_gate_passed"] is False
    reasons = selection.loc[0, "recomputed_selection_fail_reasons"]
    assert "non_threshold_raise_action_scope" in reasons
    assert "later_action_phase_requires_prior_threshold_only_promotion" in reasons
    assert "controller_changes_scores_or_ranks" in reasons
    assert "controller_changes_auction_ordering" in reasons
    assert "position_sizing_requires_prior_threshold_only_promotion" in reasons
    assert "controller_can_lower_thresholds" in reasons
    assert "controller_promotes_replacement_candidates" in reasons
    assert payload["best_raw_candidate"]["controller_action_scope"] == "bounded_score_correction"
    assert payload["best_raw_candidate"]["changes_scores_or_ranks"] is True
