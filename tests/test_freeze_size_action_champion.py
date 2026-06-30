from pathlib import Path

import pandas as pd

from scripts.freeze_size_action_champion import _gate_status, _summarize_arm


def test_freeze_manifest_metrics_and_gates(tmp_path: Path) -> None:
    run_dir = tmp_path / "size_action_run"
    run_dir.mkdir()
    arm = "C3em_bagged_safety_c3ed_or_high_value_zero_classifier_expanded_union_gate"

    pd.DataFrame(
        [
            {
                "arm": arm,
                "folds": 2,
                "median_delta_net_pnl": 10.0,
                "q25_delta_net_pnl": 8.0,
                "mean_delta_net_pnl": 11.0,
                "positive_delta_net_pnl_share": 1.0,
                "median_delta_cost_pnl": -1.0,
                "median_exposure_ratio": 0.99,
                "median_multiplier": 0.98,
            }
        ]
    ).to_csv(run_dir / "size_action_promotion_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "arm": arm,
                "fold_id": fold,
                "delta_net_pnl": 10.0,
                "intervention_count": 5,
                "positive_action_count": 5,
                "realized_delta_full_J_sum": 12.0,
                "realized_delta_full_net_pnl_sum": 11.0,
                "sequential_replay_positive": True,
                "independent_label_positive": True,
                "sequential_replay_disagrees_with_label": False,
            }
            for fold in (0, 1)
        ]
    ).to_csv(run_dir / "size_action_replay_vs_label_audit.csv", index=False)
    pd.DataFrame(
        [
            {
                "arm": arm,
                "fold_id": fold,
                "oracle_gain_capture_ratio": 0.20,
                "oracle_positive_group_capture_rate": 0.25,
            }
            for fold in (0, 1)
        ]
    ).to_csv(run_dir / "size_action_action_quality.csv", index=False)
    pd.DataFrame(
        [
            {
                "fold_id": fold,
                "timestamp": f"2026-05-01 0{idx}:00:00+00:00",
                "strategy_id": "short_asset",
                "split": "eval",
                "group_can_bind": 1.0,
            }
            for fold in (0, 1)
            for idx in range(100)
        ]
    ).to_csv(run_dir / "size_action_exact_panel.csv", index=False)
    pd.DataFrame(
        [
            {"noop_decision_signature_equal": True},
            {"noop_decision_signature_equal": True},
        ]
    ).to_csv(run_dir / "size_action_noop_parity.csv", index=False)

    metrics = _summarize_arm(run_dir, arm)
    gates = _gate_status(metrics)

    assert metrics["interventions"] == 10
    assert metrics["positive_actions"] == 10
    assert metrics["precision_total"] == 1.0
    assert metrics["binding_opportunity_groups"] == 200
    assert metrics["binding_intervention_rate_total"] == 0.05
    assert gates["research_ready"] is True
    assert gates["production_ready"] is False
    assert "true_prospective_frozen_dual_scoring_not_completed" in gates["production_blockers"]


def test_freeze_manifest_accepts_external_parity_evidence(tmp_path: Path) -> None:
    run_dir = tmp_path / "size_action_run"
    run_dir.mkdir()
    arm = "C3em_bagged_safety_c3ed_or_high_value_zero_classifier_expanded_union_gate"

    pd.DataFrame([{"arm": arm, "q25_delta_net_pnl": 1.0, "median_exposure_ratio": 0.99}]).to_csv(
        run_dir / "size_action_promotion_summary.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "arm": arm,
                "fold_id": 0,
                "delta_net_pnl": 1.0,
                "intervention_count": 1,
                "positive_action_count": 1,
                "sequential_replay_positive": True,
                "independent_label_positive": True,
                "sequential_replay_disagrees_with_label": False,
            }
        ]
    ).to_csv(run_dir / "size_action_replay_vs_label_audit.csv", index=False)
    pd.DataFrame([{"arm": arm, "fold_id": 0}]).to_csv(run_dir / "size_action_action_quality.csv", index=False)
    pd.DataFrame(
        [
            {"fold_id": 0, "timestamp": f"2026-05-01 {idx:02d}:00:00+00:00", "strategy_id": "s", "split": "eval", "group_can_bind": 1.0}
            for idx in range(20)
        ]
    ).to_csv(run_dir / "size_action_exact_panel.csv", index=False)
    pd.DataFrame(columns=["fold_id", "split"]).to_csv(run_dir / "size_action_noop_parity.csv", index=False)
    external = tmp_path / "external_noop_parity.csv"
    pd.DataFrame([{"noop_decision_signature_equal": True}]).to_csv(external, index=False)

    metrics = _summarize_arm(run_dir, arm, parity_artifact=external)
    gates = _gate_status(metrics)

    assert metrics["noop_parity_source"] == "external"
    assert metrics["noop_parity_rows"] == 1
    assert metrics["noop_decision_signature_all_equal"] is True
    assert gates["checks"]["noop_parity"] is True
