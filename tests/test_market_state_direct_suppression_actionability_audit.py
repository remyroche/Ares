import json
from pathlib import Path

import pandas as pd

from scripts.audit_market_state_direct_suppression_actionability import audit_actionability


def _write_training_bundle(root: Path, policy_rows: list[dict]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "direct_suppression_training_summary.json").write_text(
        json.dumps(
            {
                "ledger_rows": 164,
                "unique_decision_keys": 49,
                "timestamp_count": 39,
                "active_heads": ["short_asset", "short_boll"],
                "feature_count": 24,
                "oof": {
                    "oof_rows": 98,
                    "oof_unique_decision_keys": 38,
                    "prob_auc": 0.87,
                    "prob_average_precision": 0.91,
                    "utility_spearman": 0.82,
                },
                "policy_grid": {
                    "min_suppressed_rows": 2,
                    "min_suppressed_folds": 2,
                },
                "selection": {
                    "selected_arm": None,
                    "reason": "no_policy_grid_row_passed_diagnostic_gate",
                    "best_attempt": {
                        "controller_arm": "S7_pruned_state_pack__post_selection_overlay",
                        "suppressed_rows": 1,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(policy_rows).to_csv(root / "direct_suppression_policy_grid.csv", index=False)
    pd.DataFrame(
        {
            "fold": [1, 2],
            "valid_unique_decision_keys": [20, 18],
        }
    ).to_csv(root / "direct_suppression_fold_report.csv", index=False)
    pd.DataFrame(
        {
            "feature": ["state_shock_down", "state_liquidity_stress_proxy"],
            "classifier_importance": [4.0, 1.0],
            "regressor_importance": [2.0, 3.0],
        }
    ).to_csv(root / "direct_suppression_feature_importance.csv", index=False)


def _policy_row(**updates: object) -> dict:
    row = {
        "policy_scope": "controller_arm",
        "controller_arm": "S1_observed_axes_shared_response",
        "target_head": "short_asset",
        "probability_cutoff": 0.7,
        "utility_cutoff": 0.0,
        "max_delta": 0.05,
        "suppressed_rows": 0,
        "suppressed_folds": 0,
        "loss_avoided": 0.0,
        "winner_pnl_sacrificed": 0.0,
        "defensive_success": 0.0,
        "positive_fold_share": 0.0,
        "passes_diagnostic_gate": False,
        "selection_score": 0.0,
    }
    row.update(updates)
    return row


def test_actionability_audit_identifies_nonrecurrent_positive_action_folds(tmp_path: Path) -> None:
    training_dir = tmp_path / "training"
    output_dir = tmp_path / "audit"
    _write_training_bundle(
        training_dir,
        [
            _policy_row(
                suppressed_rows=2,
                suppressed_folds=2,
                loss_avoided=1.0,
                winner_pnl_sacrificed=0.1,
                defensive_success=0.9,
                positive_fold_share=1 / 3,
                selection_score=10.0,
            ),
            _policy_row(
                policy_scope="controller_arm_head_strategy",
                controller_arm="S7_pruned_state_pack__post_selection_overlay",
                target_head="short_boll",
                suppressed_rows=1,
                suppressed_folds=1,
                loss_avoided=0.06,
                winner_pnl_sacrificed=0.0,
                defensive_success=0.06,
                positive_fold_share=1.0,
                selection_score=8.0,
            ),
        ],
    )

    payload = audit_actionability(training_dir, output_dir)

    assert payload["dominant_blocker"] == "nonrecurrent_positive_action_folds"
    assert payload["passing_policy_rows"] == 0
    assert payload["positive_suppression_policy_rows"] == 2
    assert payload["recurrent_support_policy_rows"] == 1
    assert payload["max_suppressed_rows"] == 2
    assert payload["max_suppressed_folds"] == 2
    assert payload["max_defensive_success"] == 0.9
    assert payload["blocker_counts"]["positive_fold_share_below_50pct"] == 1
    assert (output_dir / "direct_suppression_actionability_audit.md").exists()
    assert (output_dir / "direct_suppression_actionability_by_scope.csv").exists()


def test_actionability_audit_identifies_insufficient_recurrent_support(tmp_path: Path) -> None:
    training_dir = tmp_path / "training"
    output_dir = tmp_path / "audit"
    _write_training_bundle(
        training_dir,
        [
            _policy_row(
                policy_scope="controller_arm_head_strategy",
                target_head="short_boll",
                suppressed_rows=1,
                suppressed_folds=1,
                loss_avoided=0.06,
                winner_pnl_sacrificed=0.0,
                defensive_success=0.06,
                positive_fold_share=1.0,
                selection_score=8.0,
            )
        ],
    )

    payload = audit_actionability(training_dir, output_dir)

    assert payload["dominant_blocker"] == "insufficient_recurrent_action_support"
    assert payload["passing_policy_rows"] == 0
    assert payload["positive_suppression_policy_rows"] == 1
    assert payload["recurrent_support_policy_rows"] == 0
    assert payload["max_suppressed_rows"] == 1
    assert payload["max_suppressed_folds"] == 1
    assert payload["blocker_counts"]["suppressed_rows_below_min"] == 1
    assert payload["blocker_counts"]["suppressed_folds_below_min"] == 1
