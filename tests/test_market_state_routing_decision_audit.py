from __future__ import annotations

import pandas as pd

from scripts.audit_market_state_routing_decision import (
    combined_decision,
    head_priority_decision,
    load_priority_gate_audit,
    priority_gate_shadow_decision,
    shadow_window_priority_decision,
    state_head_activation_summary,
)


def test_state_head_activation_summary_shadow_logs_candidates_when_controller_disabled() -> None:
    activation = pd.DataFrame(
        {
            "state_head": ["forecast_h6_shock_up", "state_shock", "latent_gmm_probabilities"],
            "state_level": ["forecast", "observed_axis", "latent"],
            "component_group": ["return_shock", "return_shock", "latent_shadow"],
            "recommended_status": ["active_candidate", "disabled_candidate", "shadow"],
            "loo_median_increment_net_pnl": [10.0, -1.0, 0.0],
            "loo_q25_increment_net_pnl": [2.0, -2.0, 0.0],
            "loo_positive_increment_share": [1.0, 0.0, 0.0],
            "activation_disable_reason": ["", "no_positive_leave_one_out_increment", "shadow_disabled"],
        }
    )

    summary = state_head_activation_summary(activation, controller_enabled=False)

    assert summary["executable_state_heads"] == []
    assert summary["shadow_state_heads"] == ["forecast_h6_shock_up", "latent_gmm_probabilities"]
    assert summary["disabled_state_heads"] == ["state_shock"]
    assert summary["status_counts"]["active_candidate"] == 1
    assert summary["by_level"]["forecast"]["active_candidate"] == 1


def test_state_head_activation_summary_exposes_candidates_when_controller_enabled() -> None:
    activation = pd.DataFrame(
        {
            "state_head": ["forecast_h6_shock_up"],
            "state_level": ["forecast"],
            "component_group": ["return_shock"],
            "recommended_status": ["active_candidate"],
        }
    )

    summary = state_head_activation_summary(activation, controller_enabled=True)

    assert summary["executable_state_heads"] == ["forecast_h6_shock_up"]
    assert summary["shadow_state_heads"] == []


def test_combined_routing_decision_keeps_static_t1_when_both_modulators_fail() -> None:
    threshold = {
        "active_status": "disabled_noop",
        "selected_arm": None,
        "best_raw_candidate": {
            "arm": "S2",
            "median_delta_net_pnl": 100.0,
            "q25_delta_net_pnl": -1.0,
            "realized_defensive_success": 0.1,
            "post_selection_realized_defensive_success": -0.2,
            "freed_capacity_entrant_count": 3.0,
            "freed_capacity_net_replacement_pnl": 10.0,
            "freed_capacity_net_action_pnl_delta": 9.0,
            "fail_reasons": "q25_delta_below_gate",
        },
    }
    priority = {
        "active_status": "shadow_rejected",
        "candidate_arm": "L0",
        "net_pnl_delta_vs_static": -10.0,
        "full_sl_delta_vs_static": 0.02,
        "trade_count_delta_vs_static": 1,
        "accepted_jaccard_vs_static": 0.94,
    }

    report = combined_decision(threshold, priority)

    assert report["recommendation"] == "keep_static_T1_active_log_market_state_shadow_only"
    assert report["promotion_allowed"] is False
    assert report["production_active_stack"]["threshold_controller"] == "disabled_noop"
    assert report["production_active_stack"]["head_priority_modulation"] == "shadow_only_rejected"
    assert report["production_active_stack"]["executable_market_state_heads"] == []
    assert report["threshold_controller"]["best_raw_candidate"]["freed_capacity_net_action_pnl_delta"] == 9.0
    assert "no_auction_priority_modulation_until_threshold_only_passes" in report["required_next_validation"]


def test_combined_routing_decision_flags_review_when_nonstatic_candidate_exists() -> None:
    threshold = {
        "active_status": "candidate_selected",
        "selected_arm": "S1",
        "best_raw_candidate": {
            "arm": "S1",
            "median_delta_net_pnl": 10.0,
            "q25_delta_net_pnl": 1.0,
            "realized_defensive_success": 0.1,
            "post_selection_realized_defensive_success": 0.1,
            "fail_reasons": "",
        },
    }
    priority = {
        "active_status": "shadow_rejected",
        "candidate_arm": "L0",
        "net_pnl_delta_vs_static": -10.0,
        "full_sl_delta_vs_static": 0.02,
        "trade_count_delta_vs_static": 1,
        "accepted_jaccard_vs_static": 0.94,
    }

    report = combined_decision(threshold, priority)

    assert report["recommendation"] == "review_non_static_market_state_candidate_before_any_promotion"
    assert report["promotion_allowed"] is None


def test_combined_routing_decision_uses_priority_gate_to_keep_shadow() -> None:
    threshold = {
        "active_status": "disabled_noop",
        "selected_arm": None,
        "best_raw_candidate": {
            "arm": None,
            "median_delta_net_pnl": float("nan"),
            "q25_delta_net_pnl": float("nan"),
            "realized_defensive_success": float("nan"),
            "post_selection_realized_defensive_success": float("nan"),
            "fail_reasons": None,
        },
        "state_head_activation": {"executable_state_heads": [], "shadow_state_heads": []},
    }
    priority = {
        "active_status": "shadow_candidate_needs_later_oos",
        "candidate_arm": "L0",
        "net_pnl_delta_vs_static": 12.0,
        "full_sl_delta_vs_static": 0.0,
        "trade_count_delta_vs_static": 0,
        "accepted_jaccard_vs_static": 0.98,
    }
    gate = {
        "priority_should_remain_shadow": True,
        "passing_candidate_count": 0,
        "best_raw_candidate": {"arm": "L0", "fail_reasons": "selection_gate_not_passed"},
    }

    report = combined_decision(threshold, priority, gate)

    assert report["recommendation"] == "keep_static_T1_active_log_market_state_shadow_only"
    assert report["production_active_stack"]["head_priority_modulation"] == "shadow_only_rejected"
    assert report["head_priority_promotion_gate"]["passing_candidate_count"] == 0


def test_priority_gate_shadow_decision_prefers_opportunity_gate_payload() -> None:
    gate = {
        "passed": False,
        "opportunity_routing_passed": False,
        "opportunity_should_remain_shadow": True,
        "opportunity_routing_gate": {
            "passed": False,
            "failures": ["fewer_than_2_action_windows"],
            "action_window_count": 1,
        },
        "defensive_suppression_gate": {"passed": True, "failures": []},
    }

    status = priority_gate_shadow_decision(gate)

    assert status["gate_family"] == "opportunity_routing"
    assert status["gate_passed"] is False
    assert status["should_remain_shadow"] is True
    assert status["failures"] == ["fewer_than_2_action_windows"]


def test_load_priority_gate_audit_reads_shadow_promotion_filename(tmp_path) -> None:
    audit_dir = tmp_path / "audit"
    audit_dir.mkdir()
    (audit_dir / "market_state_priority_shadow_promotion_gate.json").write_text(
        """
        {
          "opportunity_routing_passed": false,
          "opportunity_should_remain_shadow": true,
          "opportunity_routing_gate": {
            "failures": ["fewer_than_2_action_windows"]
          }
        }
        """,
        encoding="utf-8",
    )

    payload = load_priority_gate_audit(audit_dir)

    assert payload is not None
    assert payload["opportunity_should_remain_shadow"] is True
    assert payload["opportunity_routing_gate"]["failures"] == ["fewer_than_2_action_windows"]


def test_load_priority_gate_audit_reads_shadow_window_root(tmp_path) -> None:
    audit_dir = tmp_path / "shadow"
    (audit_dir / "promotion_audit").mkdir(parents=True)
    (audit_dir / "promotion_audit" / "market_state_priority_shadow_promotion_gate.json").write_text(
        """
        {
          "opportunity_routing_passed": false,
          "opportunity_should_remain_shadow": true,
          "opportunity_routing_gate": {
            "failures": ["accepted_jaccard_below_required_95pct"]
          }
        }
        """,
        encoding="utf-8",
    )

    payload = load_priority_gate_audit(audit_dir)

    assert payload is not None
    assert payload["opportunity_should_remain_shadow"] is True
    assert payload["opportunity_routing_gate"]["failures"] == ["accepted_jaccard_below_required_95pct"]


def test_shadow_window_priority_decision_reads_aggregate_and_gate(tmp_path) -> None:
    root = tmp_path / "shadow"
    audit = root / "promotion_audit"
    audit.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "arm": "L1_lgbm_learned_priority_cap_0p15_zge_0p5",
                "trade_count": 117,
                "delta_net_pnl": 21.5,
                "delta_full_sl_rate": -0.01,
                "delta_timeout_rate": 0.0,
                "accepted_jaccard": 0.90,
                "gate_passed": True,
            },
            {
                "arm": "L1_lgbm_learned_priority_cap_0p15_zge_0p5",
                "trade_count": 11,
                "delta_net_pnl": 20.0,
                "delta_full_sl_rate": 0.08,
                "delta_timeout_rate": 0.0,
                "accepted_jaccard": 0.50,
                "gate_passed": False,
            },
        ]
    ).to_csv(audit / "market_state_priority_shadow_window_summary.csv", index=False)
    pd.DataFrame([{"arm": "L1_lgbm_learned_priority_cap_0p15_zge_0p5", "head": "short_boll"}]).to_csv(
        audit / "market_state_priority_shadow_by_head.csv",
        index=False,
    )
    (audit / "market_state_priority_shadow_promotion_gate.json").write_text(
        """
        {
          "opportunity_routing_passed": false,
          "opportunity_should_remain_shadow": true,
          "opportunity_routing_gate": {
            "failures": ["accepted_jaccard_below_required_95pct"]
          }
        }
        """,
        encoding="utf-8",
    )
    (root / "manifest.json").write_text(
        """
        {
          "contract": {
            "active_baseline": "static_T1",
            "rank_contract_preserved": "within_timestamp",
            "candidate_rank_contract_names": ["short_boll_timestamp_rank"]
          },
          "static_baseline_parity": {
            "promotion_grade": true,
            "checked_windows": 2,
            "passed_windows": 2
          },
          "windows": [
            {
              "static_baseline_parity": {
                "observed": {"trade_count": 117}
              }
            },
            {
              "static_baseline_parity": {
                "observed": {"trade_count": 10}
              }
            }
          ]
        }
        """,
        encoding="utf-8",
    )

    decision = shadow_window_priority_decision(root)

    assert decision["source_format"] == "shadow_window_aggregate"
    assert decision["active_status"] == "shadow_rejected"
    assert decision["candidate_arm"] == "L1_lgbm_learned_priority_cap_0p15_zge_0p5"
    assert decision["selection_gate_pass_count"] == 1
    assert decision["net_pnl_delta_vs_static"] == 20.75
    assert decision["full_sl_delta_vs_static"] == 0.08
    assert decision["accepted_jaccard_vs_static"] == 0.50
    assert decision["candidate_trade_count_sum"] == 128
    assert decision["base_trade_count_sum"] == 127
    assert decision["trade_count_delta_vs_static"] == 1
    assert decision["selected_config"]["rank_contract_preserved"] == "within_timestamp"
    assert decision["selected_config"]["static_baseline_parity_promotion_grade"] is True


def test_combined_routing_decision_blocks_priority_when_threshold_controller_disabled() -> None:
    threshold = {
        "active_status": "disabled_noop",
        "selected_arm": None,
        "best_raw_candidate": {
            "arm": None,
            "median_delta_net_pnl": float("nan"),
            "q25_delta_net_pnl": float("nan"),
            "realized_defensive_success": float("nan"),
            "post_selection_realized_defensive_success": float("nan"),
            "fail_reasons": None,
        },
        "state_head_activation": {"executable_state_heads": [], "shadow_state_heads": []},
    }
    priority = {
        "active_status": "shadow_candidate_needs_later_oos",
        "candidate_arm": "L0",
        "net_pnl_delta_vs_static": 12.0,
        "full_sl_delta_vs_static": 0.0,
        "trade_count_delta_vs_static": 0,
        "accepted_jaccard_vs_static": 0.98,
    }
    gate = {
        "opportunity_routing_passed": True,
        "opportunity_should_remain_shadow": False,
        "opportunity_routing_gate": {"passed": True, "failures": []},
        "defensive_suppression_gate": {
            "passed": False,
            "failures": ["action_window_defensive_success_not_positive"],
        },
    }

    report = combined_decision(threshold, priority, gate)

    assert report["recommendation"] == "keep_static_T1_active_log_market_state_shadow_only"
    assert report["promotion_allowed"] is False
    assert report["priority_blocked_by_threshold_controller"] is True
    assert (
        report["production_active_stack"]["head_priority_modulation"]
        == "shadow_blocked_until_threshold_controller_promoted"
    )
    assert report["head_priority_gate_status"]["gate_family"] == "opportunity_routing"
    assert "head_priority_opportunity_routing_recurrent_positive_replacement_value" in report["required_next_validation"]
    assert "threshold_only_controller_promotion_before_priority_modulation" in report["required_next_validation"]


def test_head_priority_decision_rejects_positive_forced_run_without_gate(tmp_path) -> None:
    priority_dir = tmp_path / "priority"
    priority_dir.mkdir()
    pd.DataFrame(
        [
            {
                "arm": "P0_static_priority",
                "trade_count": 10,
                "net_pnl": 100.0,
                "full_sl_rate": 0.20,
            },
            {
                "arm": "L0_forced_priority",
                "trade_count": 10,
                "net_pnl": 120.0,
                "full_sl_rate": 0.20,
            },
        ]
    ).to_csv(priority_dir / "head_priority_learning_replay_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "arm": "L0_forced_priority",
                "jaccard_vs_baseline": 0.95,
                "baseline_only": 1,
                "arm_only": 1,
            }
        ]
    ).to_csv(priority_dir / "head_priority_learning_accepted_overlap.csv", index=False)
    pd.DataFrame(
        [
            {
                "backend": "lgbm",
                "selection_objective": 0.70,
                "selection_gate_passed": False,
            }
        ]
    ).to_csv(priority_dir / "head_priority_learning_model_diagnostics.csv", index=False)
    pd.DataFrame(
        [
            {
                "backend": "lgbm",
                "selection_objective": 0.70,
                "selection_gate_passed": False,
            }
        ]
    ).to_csv(priority_dir / "head_priority_config_selection.csv", index=False)
    pd.DataFrame(
        [
            {
                "arm": "P0_static_priority",
                "head": "short_asset",
                "trade_count": 10,
            },
            {
                "arm": "L0_forced_priority",
                "head": "short_asset",
                "trade_count": 10,
            },
        ]
    ).to_csv(priority_dir / "head_priority_learning_by_head.csv", index=False)

    decision = head_priority_decision(priority_dir)

    assert decision["net_pnl_delta_vs_static"] == 20.0
    assert decision["selection_gate_pass_count"] == 0
    assert decision["active_status"] == "shadow_rejected"
