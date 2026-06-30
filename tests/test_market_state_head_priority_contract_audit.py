from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.audit_market_state_head_priority_contract import audit_head_priority_contract


def _write_priority_artifact(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "generated_by": "run_market_state_head_priority_learning",
        "purpose": "learned_market_state_head_priority_modulation_shadow_ablation",
        "contract": {
            "changes_scores_or_ranks": False,
            "changes_thresholds": False,
            "changes_position_sizing": False,
            "changes_auction_ordering": True,
            "qfail_active": False,
            "head_health_active": False,
            "market_state_threshold_controller_active": False,
            "operational_status": "shadow_only",
            "execution_enabled": False,
            "production_eligible": False,
            "requires_promotion_gate": True,
            "market_state_encoder_uses_candidate_features": False,
            "priority_adjustment_column": "portfolio_priority_adjustment",
            "priority_multiplier_column": "portfolio_priority_multiplier",
        },
        "params": {
            "max_adjustment": 0.20,
            "max_priority_multiplier": 1.50,
            "selection_replay_top_n": 5,
        },
        "state_head_activation_filter": {
            "enabled": False,
            "reason": "test_legacy_full_pack",
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    timestamps = pd.to_datetime(
        ["2026-06-15T00:00:00Z", "2026-06-15T01:00:00Z"],
        utc=True,
    )
    train_rows = []
    score_rows = []
    schedule_rows = []
    for fold, timestamp, shock in zip([1, 1], timestamps, [0.2, 0.4]):
        for head, adj in [("short_asset", -0.05), ("short_boll", 0.05)]:
            train_rows.append(
                {
                    "fold": fold,
                    "timestamp": timestamp,
                    "head": head,
                    "state_shock_down": shock,
                    "state_drift_score": 0.05,
                    "forecast_h6_deleveraging": shock + 0.1,
                    "priority_target": adj,
                }
            )
            score_rows.append(
                {
                    "timestamp": timestamp,
                    "head": head,
                    "state_shock_down": shock,
                    "state_drift_score": 0.05,
                    "forecast_h6_deleveraging": shock + 0.1,
                }
            )
            schedule_rows.append(
                {
                    "timestamp": timestamp,
                    "head": head,
                    "portfolio_priority_adjustment": adj,
                    "portfolio_priority_multiplier": 1.0,
                }
            )
    pd.DataFrame(train_rows).to_parquet(root / "head_priority_training_targets.parquet", index=False)
    pd.DataFrame(score_rows).to_parquet(root / "head_priority_score_rows.parquet", index=False)
    pd.DataFrame(schedule_rows).to_parquet(root / "head_priority_learned_schedule.parquet", index=False)

    pd.DataFrame(
        {
            "arm": ["P0_static_priority", "L0_selected_lgbm_priority"],
            "trade_count": [10, 10],
            "net_pnl": [1.0, 2.0],
            "full_sl_rate": [0.1, 0.1],
            "timeout_rate": [0.0, 0.0],
        }
    ).to_csv(root / "head_priority_learning_replay_summary.csv", index=False)
    pd.DataFrame({"arm": ["P0_static_priority"], "head": ["short_boll"], "net_pnl": [1.0]}).to_csv(
        root / "head_priority_learning_by_head.csv",
        index=False,
    )
    pd.DataFrame(
        {
            "arm": ["L0_selected_lgbm_priority"],
            "selection_gate_passed": [False],
            "selection_objective": [0.5],
        }
    ).to_csv(root / "head_priority_learning_model_diagnostics.csv", index=False)
    pd.DataFrame(
        {
            "arm": ["L0_selected_lgbm_priority"],
            "jaccard_vs_baseline": [0.98],
        }
    ).to_csv(root / "head_priority_learning_accepted_overlap.csv", index=False)
    pd.DataFrame(
        {
            "arm": ["L0_selected_lgbm_priority"],
            "scope": ["all"],
            "entrants": [1],
            "removed": [1],
            "net_replacement_pnl": [1.0],
            "net_action_pnl_delta": [1.0],
        }
    ).to_csv(root / "head_priority_learning_accepted_swap_utility.csv", index=False)
    pd.DataFrame(
        {
            "feature": ["state_shock_down"],
            "present": [True],
            "finite_share": [1.0],
            "filled_with_training_median": [False],
        }
    ).to_csv(root / "head_priority_score_feature_coverage.csv", index=False)
    pd.DataFrame(
        {
            "selection_gate_passed": [False],
            "selection_objective": [0.5],
        }
    ).to_csv(root / "head_priority_config_selection.csv", index=False)
    pd.DataFrame({"validation_fold": [1], "validation_spearman": [0.1]}).to_csv(
        root / "head_priority_config_fold_validation.csv",
        index=False,
    )


def test_head_priority_contract_audit_accepts_shadow_artifact(tmp_path: Path) -> None:
    _write_priority_artifact(tmp_path)

    payload = audit_head_priority_contract(tmp_path)

    assert payload["passed"] is True
    assert payload["training_state_feature_count"] == 3
    assert payload["score_state_feature_count"] == 3
    assert payload["replay_aware_selection"] is True


def test_head_priority_contract_audit_accepts_fixed_config_empty_selection(tmp_path: Path) -> None:
    _write_priority_artifact(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    manifest["params"]["select_config_grid"] = False
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "head_priority_config_selection.csv").write_text("", encoding="utf-8")

    payload = audit_head_priority_contract(tmp_path)

    assert payload["passed"] is True


def test_head_priority_contract_audit_rejects_non_marketwide_state_values(tmp_path: Path) -> None:
    _write_priority_artifact(tmp_path)
    frame = pd.read_parquet(tmp_path / "head_priority_score_rows.parquet")
    mask = frame["head"].eq("short_boll") & frame["timestamp"].eq(frame["timestamp"].iloc[0])
    frame.loc[mask, "state_shock_down"] = 0.99
    frame.to_parquet(tmp_path / "head_priority_score_rows.parquet", index=False)

    payload = audit_head_priority_contract(tmp_path)

    assert payload["passed"] is False
    assert any("head_priority_score_rows.state_shock_down varies within" in failure for failure in payload["failures"])


def test_head_priority_contract_audit_rejects_orderbook_like_state_features(tmp_path: Path) -> None:
    _write_priority_artifact(tmp_path)
    for name in ["head_priority_training_targets.parquet", "head_priority_score_rows.parquet"]:
        frame = pd.read_parquet(tmp_path / name)
        frame["state_bid_depth_proxy"] = 1.0
        frame.to_parquet(tmp_path / name, index=False)

    payload = audit_head_priority_contract(tmp_path)

    assert payload["passed"] is False
    assert any("order-book-like" in failure for failure in payload["failures"])


def test_head_priority_contract_audit_rejects_rank_or_model_state_features(tmp_path: Path) -> None:
    _write_priority_artifact(tmp_path)
    for name in ["head_priority_training_targets.parquet", "head_priority_score_rows.parquet"]:
        frame = pd.read_parquet(tmp_path / name)
        frame["state_policy_rank_pct"] = 0.8
        frame["forecast_net_pnl"] = 1.0
        frame["state_anchor_score"] = 0.9
        frame.to_parquet(tmp_path / name, index=False)

    payload = audit_head_priority_contract(tmp_path)

    assert payload["passed"] is False
    assert any("strategy/model/performance-like" in failure for failure in payload["failures"])


def test_head_priority_contract_audit_accepts_activation_filtered_artifact(tmp_path: Path) -> None:
    _write_priority_artifact(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    manifest["state_head_activation_filter"] = {
        "enabled": True,
        "path": "market_state_activation_registry.csv",
        "allowed_statuses": ["active_candidate"],
        "allowed_state_heads": ["forecast_h6_deleveraging"],
        "registry_rows": 3,
        "allowed_state_head_count": 1,
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    for name in ["head_priority_training_targets.parquet", "head_priority_score_rows.parquet"]:
        frame = pd.read_parquet(tmp_path / name)
        frame = frame[
            [
                col
                for col in frame.columns
                if col
                not in {
                    "state_shock_down",
                    "state_drift_score",
                }
            ]
        ]
        frame.to_parquet(tmp_path / name, index=False)

    payload = audit_head_priority_contract(tmp_path)

    assert payload["passed"] is True
    assert payload["training_state_feature_count"] == 1
    assert payload["score_state_feature_count"] == 1


def test_head_priority_contract_audit_rejects_disabled_state_heads_when_filter_enabled(
    tmp_path: Path,
) -> None:
    _write_priority_artifact(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    manifest["state_head_activation_filter"] = {
        "enabled": True,
        "path": "market_state_activation_registry.csv",
        "allowed_statuses": ["active_candidate"],
        "allowed_state_heads": ["forecast_h6_deleveraging"],
        "registry_rows": 3,
        "allowed_state_head_count": 1,
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    payload = audit_head_priority_contract(tmp_path)

    assert payload["passed"] is False
    assert any("outside activation filter" in failure for failure in payload["failures"])


def test_head_priority_contract_audit_rejects_executable_priority_artifact(tmp_path: Path) -> None:
    _write_priority_artifact(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    manifest["contract"]["execution_enabled"] = True
    manifest["contract"]["production_eligible"] = True
    manifest["contract"]["operational_status"] = "active"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    payload = audit_head_priority_contract(tmp_path)

    assert payload["passed"] is False
    assert "manifest.contract.operational_status is not shadow_only" in payload["failures"]
    assert "manifest.contract.execution_enabled is not false" in payload["failures"]
    assert "manifest.contract.production_eligible is not false" in payload["failures"]


def test_head_priority_contract_audit_rejects_unbounded_schedule(tmp_path: Path) -> None:
    _write_priority_artifact(tmp_path)
    schedule = pd.read_parquet(tmp_path / "head_priority_learned_schedule.parquet")
    schedule.loc[0, "portfolio_priority_adjustment"] = 0.50
    schedule.to_parquet(tmp_path / "head_priority_learned_schedule.parquet", index=False)

    payload = audit_head_priority_contract(tmp_path)

    assert payload["passed"] is False
    assert any("exceeds max_adjustment" in failure for failure in payload["failures"])


def test_head_priority_contract_audit_rejects_unbounded_multiplier(tmp_path: Path) -> None:
    _write_priority_artifact(tmp_path)
    schedule = pd.read_parquet(tmp_path / "head_priority_learned_schedule.parquet")
    schedule.loc[0, "portfolio_priority_multiplier"] = 2.0
    schedule.to_parquet(tmp_path / "head_priority_learned_schedule.parquet", index=False)

    payload = audit_head_priority_contract(tmp_path)

    assert payload["passed"] is False
    assert any("exceeds max_priority_multiplier" in failure for failure in payload["failures"])
