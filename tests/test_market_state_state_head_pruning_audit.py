from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.audit_market_state_state_head_pruning import audit_state_head_pruning


def _base_row(**overrides):
    row = {
        "state_level": "forecast",
        "state_head": "forecast_h6_shock_up",
        "component_group": "return_shock",
        "aggregate_status": "active",
        "recommended_status": "active_candidate",
        "activation_disable_reason": "",
        "forecast_skill_gate_pass": True,
        "response_gate_pass": True,
        "action_gate_pass": True,
        "leave_one_out_gate_pass": True,
        "defensive_action_gate_pass": True,
        "loo_median_increment_net_pnl": 10.0,
        "loo_q25_increment_net_pnl": 2.0,
        "loo_positive_increment_share": 0.75,
        "loo_state_head_defensive_success": 4.0,
        "loo_state_head_loss_avoided": 5.0,
        "loo_state_head_winner_pnl_sacrificed": 1.0,
        "max_abs_spearman_corr": 0.5,
        "redundant_with": "",
        "redundancy_group": "return_shock",
        "redundancy_flag": False,
        "activation_registry_version": "market_state_activation_registry_v1",
    }
    row.update(overrides)
    return row


def _write_registry(root: Path, rows: list[dict]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(root / "market_state_activation_registry.csv", index=False)
    (root / "manifest.json").write_text('{"market_mode":"perps"}', encoding="utf-8")


def test_state_head_pruning_accepts_consistent_registry(tmp_path: Path) -> None:
    rows = [
        _base_row(),
        _base_row(
            state_head="forecast_h6_rv_ratio",
            component_group="volatility_tail",
            recommended_status="disabled_candidate",
            leave_one_out_gate_pass=False,
            defensive_action_gate_pass=False,
            activation_disable_reason="no_positive_leave_one_out_increment;state_action_sacrifices_winners",
            loo_median_increment_net_pnl=-1.0,
            loo_q25_increment_net_pnl=-2.0,
            loo_state_head_defensive_success=-3.0,
            loo_state_head_loss_avoided=0.0,
            loo_state_head_winner_pnl_sacrificed=3.0,
        ),
        _base_row(
            state_level="latent",
            state_head="latent_gmm_probabilities",
            component_group="latent_shadow",
            aggregate_status="shadow_disabled",
            recommended_status="shadow",
            response_gate_pass=False,
            leave_one_out_gate_pass=False,
            defensive_action_gate_pass=False,
            activation_disable_reason=(
                "shadow_disabled;weak_response_effect;"
                "no_positive_leave_one_out_increment;state_action_sacrifices_winners"
            ),
        ),
    ]
    _write_registry(tmp_path, rows)

    payload, audit = audit_state_head_pruning(tmp_path)

    assert payload["passed"] is True
    assert payload["active_candidate_count"] == 1
    assert payload["disabled_candidate_count"] == 1
    assert payload["shadow_count"] == 1
    assert set(audit["expected_status"]) == {"active_candidate", "disabled_candidate", "shadow"}


def test_state_head_pruning_rejects_status_mismatch(tmp_path: Path) -> None:
    rows = [
        _base_row(
            recommended_status="active_candidate",
            leave_one_out_gate_pass=False,
            defensive_action_gate_pass=False,
            activation_disable_reason="",
        )
    ]
    _write_registry(tmp_path, rows)

    payload, audit = audit_state_head_pruning(tmp_path)

    assert payload["passed"] is False
    assert bool(audit.loc[0, "status_mismatch"])
    assert any("expected disabled_candidate" in failure for failure in payload["failures"])


def test_state_head_pruning_rejects_active_candidate_with_bad_q25(tmp_path: Path) -> None:
    rows = [_base_row(loo_q25_increment_net_pnl=-0.1)]
    _write_registry(tmp_path, rows)

    payload, _audit = audit_state_head_pruning(tmp_path)

    assert payload["passed"] is False
    assert any("q25 LOO increment below gate" in failure for failure in payload["failures"])
