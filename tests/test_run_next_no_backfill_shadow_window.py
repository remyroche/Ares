from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.run_next_no_backfill_shadow_window import (
    DEFAULT_BUNDLE,
    _existing_monitor_score_dirs,
    _latest_score_manifest_paths,
    _update_runner_config,
    build_runner_plan,
)


def _config(tmp_path: Path) -> dict:
    score_dir = tmp_path / "reports" / "prior_score"
    score_dir.mkdir(parents=True)
    (score_dir / "manifest.json").write_text(
        json.dumps(
            {
                "bundle": str(DEFAULT_BUNDLE),
                "policy_manifest": "data_perp/reports/reliability_blend_component_arm_portfolio_ablation_20260625/A0_anchor_only/portfolio_policy_ablation_manifest.json",
                "train_deployable_candidates": "data_perp/artifacts/reliability_blend_native_simple_policy_replay_20260624_floor070/simple_policy_optimiser/simple_policy_candidates.parquet",
            }
        ),
        encoding="utf-8",
    )
    return {
        "active_stack": {
            "rank_contract": "anchor_global_policy_rank_reference",
            "rank_scope": "global_over_time",
            "enabled_heads": ["short_asset", "short_boll"],
            "disabled_heads": ["long_bars", "long_dist"],
            "qfail_active": False,
            "market_state_threshold_controller_active": False,
        },
        "market_state_controller_validation": {
            "global_rank_threshold_controller_no_backfill_shadow_monitor": {
                "windows": [
                    {
                        "period_start": "2026-06-25T09:00:00+00:00",
                        "period_end": "2026-06-26T07:00:00+00:00",
                        "score_dir": str(score_dir),
                    }
                ]
            },
            "global_rank_threshold_controller_no_backfill_shadow_window_discovery": {
                "appendable_candidate_count": 0,
                "latest_discovered_window_end": "2026-06-26T07:00:00+00:00",
            },
            "global_rank_threshold_controller_no_backfill_shadow_score_latest": {
                "score_dir": str(score_dir),
                "bundle": str(DEFAULT_BUNDLE),
            },
        },
    }


def _feature_store(tmp_path: Path, end: str) -> Path:
    feature_store = tmp_path / "features" / "20260627_010000"
    feature_store.mkdir(parents=True)
    timestamps = pd.date_range("2026-06-26T00:00:00Z", end, freq="1h")
    pd.DataFrame({"ts": timestamps, "x": range(len(timestamps))}).to_parquet(
        feature_store / "symbol=BTC_USD:USD.parquet",
        index=False,
    )
    return feature_store


def _plan(tmp_path: Path, *, end: str, allow_partial: bool) -> dict:
    feature_store = _feature_store(tmp_path, end)
    return build_runner_plan(
        config=_config(tmp_path),
        config_path=tmp_path / "stack.json",
        data_root=tmp_path,
        feature_store_dir=feature_store,
        output_dir=tmp_path / "runner",
        readiness_output_dir=tmp_path / "runner" / "readiness",
        maturity_buffer_hours=16,
        target_window_hours=24,
        min_timestamp_count=3,
        min_feature_timestamp_coverage=0.95,
        allow_partial_window=allow_partial,
        run_id="",
        score_output_dir=None,
        discovery_output_dir=None,
        monitor_output_dir=None,
        bundle=None,
        policy_manifest=None,
        train_deployable_candidates=None,
        policy_variant="refit_bar4_strategy_bar2",
        market_mode="perps",
        exchange="krakenfutures",
        rank_reference_run_id="reliability_blend_anchor_rank_reference_20260625_prejune",
        policy_artifact_run_id="20260617_090000_no_mkt4_labelhpo_final_fit",
        model_artifact_run_id="20260618_081800_current4_final_fit",
        include_monitor_step=True,
    )


def test_runner_does_not_plan_steps_when_next_window_is_immature(tmp_path: Path) -> None:
    plan = _plan(tmp_path, end="2026-06-27T00:00:00Z", allow_partial=True)

    assert plan["status"] == "not_scoreable_yet"
    assert plan["steps"] == []
    assert plan["readiness"]["mature_timestamp_count_available"] == 1
    assert plan["readiness"]["scoreable_min_window_now"] is False
    assert plan["readiness"]["minimum_window_feature_coverage_ready"] is True
    assert plan["paths"]["runner_output_dir"] == str(tmp_path / "runner")
    assert plan["paths"]["runner_manifest"].endswith(
        "next_no_backfill_shadow_runner_manifest.json"
    )


def test_runner_requires_full_window_by_default_when_only_partial_is_ready(tmp_path: Path) -> None:
    plan = _plan(tmp_path, end="2026-06-27T03:00:00Z", allow_partial=False)

    assert plan["status"] == "full_window_not_scoreable_yet"
    assert plan["reason"] == "full_window_required_but_only_partial_window_scoreable"
    assert plan["steps"] == []
    assert plan["readiness"]["scoreable_min_window_now"] is True
    assert plan["readiness"]["scoreable_full_window_now"] is False
    assert plan["readiness"]["minimum_window_feature_coverage_ready"] is True


def test_runner_plans_partial_score_chain_with_current_global_rank_contract(tmp_path: Path) -> None:
    plan = _plan(tmp_path, end="2026-06-27T03:00:00Z", allow_partial=True)

    assert plan["status"] == "scoreable_now"
    assert plan["reason"] == "partial_window_scoreable"
    assert plan["window"]["start"] == "2026-06-26T08:00:00+00:00"
    assert plan["window"]["end"] == "2026-06-26T11:00:00+00:00"
    assert [step["name"] for step in plan["steps"]] == [
        "materialize_t1_anchor_candidates",
        "score_market_state_no_backfill_shadow_bundle",
        "discover_appendable_no_backfill_shadow_windows",
        "refresh_no_backfill_shadow_monitor",
    ]
    score_step = plan["steps"][1]
    assert "--bundle" in score_step["command"]
    assert "--eval-candidates" in score_step["command"]
    assert "--window-start" in score_step["command"]
    assert "--window-end" in score_step["command"]
    assert "scripts/score_market_state_controller_bundle.py" in score_step["command"]
    monitor_step = plan["steps"][-1]
    assert monitor_step["command"].count("--score-dir") == 2


def test_runner_preserves_monitor_summary_score_dirs(tmp_path: Path) -> None:
    prior_1 = tmp_path / "reports" / "prior_1"
    prior_2 = tmp_path / "reports" / "prior_2"
    prior_1.mkdir(parents=True)
    prior_2.mkdir(parents=True)
    monitor_dir = tmp_path / "reports" / "monitor"
    monitor_dir.mkdir(parents=True)
    metrics_csv = monitor_dir / "no_backfill_shadow_window_metrics.csv"
    pd.DataFrame({"score_dir": [str(prior_1), str(prior_2)]}).to_csv(
        metrics_csv,
        index=False,
    )
    summary_json = monitor_dir / "no_backfill_shadow_monitor_summary.json"
    summary_json.write_text(
        json.dumps(
            {
                "window_metrics_csv": str(metrics_csv),
                "windows": [{"score_dir": str(prior_1)}],
            }
        ),
        encoding="utf-8",
    )
    config = _config(tmp_path)
    controller = config["market_state_controller_validation"]
    controller["global_rank_threshold_controller_no_backfill_shadow_monitor"] = {
        "summary_json": str(summary_json),
    }

    score_dirs = _existing_monitor_score_dirs(config)

    assert score_dirs == [prior_1, prior_2]


def test_runner_resolves_active_bundle_from_bundle_dir(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "reports" / "active_bundle"
    bundle_dir.mkdir(parents=True)
    (bundle_dir / "market_state_controller_bundle.joblib").write_bytes(b"bundle")
    config = {
        "market_state_controller_validation": {
            "global_rank_threshold_controller_no_backfill_shadow_score_latest": {
                "bundle_dir": str(bundle_dir),
            }
        }
    }

    defaults = _latest_score_manifest_paths(config)

    assert defaults["bundle"] == bundle_dir / "market_state_controller_bundle.joblib"


def test_runner_update_config_records_latest_score_and_monitor(tmp_path: Path) -> None:
    config_path = tmp_path / "stack.json"
    score_dir = tmp_path / "score"
    monitor_dir = tmp_path / "monitor"
    bundle = tmp_path / "bundle" / "market_state_controller_bundle.joblib"
    score_dir.mkdir(parents=True)
    monitor_dir.mkdir(parents=True)
    bundle.parent.mkdir(parents=True)
    bundle.write_bytes(b"bundle")
    (score_dir / "manifest.json").write_text(
        json.dumps(
            {
                "generated_by": "score_market_state_controller_bundle",
                "generated_at_utc": "2026-06-27T22:52:19+00:00",
                "score_manifest_contract_version": "market_state_controller_score_manifest_v2",
                "bundle": str(bundle),
                "bundle_sha256": "a" * 64,
                "policy_manifest": "policy.json",
                "policy_manifest_sha256": "b" * 64,
                "eval_candidates": "eval.parquet",
                "eval_candidates_sha256": "c" * 64,
                "train_deployable_candidates": "train.parquet",
                "train_deployable_candidates_sha256": "d" * 64,
                "selected_arm": "S1_observed_axes_shared_response__post_selection_overlay",
                "rank_contract": "anchor_global_policy_rank_reference",
                "rank_reference_run_id": "rank_ref",
                "active_heads": ["short_asset", "short_boll"],
                "disabled_heads": ["long_bars", "long_dist"],
                "controller_execution_enabled": False,
                "shadow_controller_only": True,
                "shadow_no_backfill_replay_available": True,
                "shadow_direct_threshold_only_available": True,
                "shadow_locked_accepted_overlay_available": True,
                "source_contract_audit": {"overall_passed": True},
                "output_sha256": {"manifest": "e" * 64},
            }
        ),
        encoding="utf-8",
    )
    (monitor_dir / "no_backfill_shadow_monitor_summary.json").write_text(
        json.dumps(
            {
                "generated_by": "report_market_state_no_backfill_shadow_monitor",
                "generated_at_utc": "2026-06-27T22:52:20+00:00",
                "status": "not_promoted_negative_later_windows",
                "window_metrics_csv": str(monitor_dir / "metrics.csv"),
                "by_head_csv": str(monitor_dir / "by_head.csv"),
                "window_count": 5,
                "promotion_gate_passed": False,
                "promotion_gate_failures": ["negative_median_later_window_total_delta_net_pnl"],
                "direct_threshold_only_promotion_gate_passed": False,
                "direct_threshold_only_promotion_gate_failures": ["direct_gate"],
                "locked_accepted_overlay_promotion_gate_passed": False,
                "locked_accepted_overlay_promotion_gate_failures": ["locked_gate"],
                "interpretation": "remain disabled",
                "positive_delta_window_share": 0.4,
                "median_total_net_pnl_delta": -5.0,
                "q25_total_net_pnl_delta": -6.0,
                "sum_total_net_pnl_delta": -97.0,
                "sum_direct_threshold_only_net_pnl_delta": 28.0,
                "min_eval_feature_store_timestamp_coverage": 1.0,
                "min_eval_source_feature_count": 1632,
                "all_score_manifest_artifact_hashes_complete": True,
                "all_source_contracts_passed": True,
                "rank_contracts": ["anchor_global_policy_rank_reference"],
                "selected_arms": ["S1_observed_axes_shared_response__post_selection_overlay"],
            }
        ),
        encoding="utf-8",
    )
    plan = {
        "generated_by": "run_next_no_backfill_shadow_window",
        "generated_at_utc": "2026-06-27T22:52:21+00:00",
        "status": "scoreable_now",
        "reason": "partial_window_scoreable",
        "window": {"start": "2026-06-26T20:00:00+00:00", "end": "2026-06-26T22:00:00+00:00"},
        "steps": [{"name": "score_market_state_no_backfill_shadow_bundle"}],
        "completed_steps": [{"name": "score_market_state_no_backfill_shadow_bundle", "returncode": 0}],
        "paths": {
            "runner_output_dir": str(tmp_path / "runner"),
            "runner_manifest": str(tmp_path / "runner" / "next_no_backfill_shadow_runner_manifest.json"),
            "readiness_output_dir": str(tmp_path / "runner" / "readiness"),
            "score_output_dir": str(score_dir),
            "discovery_output_dir": str(tmp_path / "discovery"),
            "monitor_output_dir": str(monitor_dir),
        },
    }
    config: dict = {}

    _update_runner_config(config, config_path, plan)

    updated = json.loads(config_path.read_text(encoding="utf-8"))
    controller = updated["market_state_controller_validation"]
    latest = controller["global_rank_threshold_controller_no_backfill_shadow_score_latest"]
    monitor = controller["global_rank_threshold_controller_no_backfill_shadow_monitor"]
    assert latest["score_dir"] == str(score_dir)
    assert latest["bundle_dir"] == str(bundle.parent)
    assert latest["shadow_controller_only"] is True
    assert latest["shadow_no_backfill_replay_available"] is True
    assert latest["source_contract_overall_passed"] is True
    assert monitor["monitor_dir"] == str(monitor_dir)
    assert monitor["controller_should_remain_disabled"] is True
    assert monitor["sum_total_net_pnl_delta"] == -97.0
