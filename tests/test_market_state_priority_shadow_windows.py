from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

from scripts.run_market_state_priority_shadow_windows import (
    _audit_command,
    _cap_sweep_command,
    _needs_run,
    _priority_command,
    _read_window,
    _resolve_run_rank_contract,
    _run_readiness_preflight,
    _score_command,
    _shadow_priority_contract_fields,
    _static_baseline_parity,
    _slug,
)
from scripts.run_market_state_head_priority_learning import (
    BASELINE_ARM,
    _load_static_baseline_artifacts,
)


def _args(tmp_path: Path) -> Namespace:
    return Namespace(
        bundle=tmp_path / "bundle.joblib",
        walkforward_dir=tmp_path / "walkforward",
        feature_store_dir=tmp_path / "features",
        policy_manifest=tmp_path / "policy.json",
        train_deployable_candidates=tmp_path / "train.parquet",
        policy_variant="refit_bar4_strategy_bar2",
        market_mode="perps",
        state_arm="S2_observed_forecast_shared_response",
        state_head_statuses="active_candidate",
        use_all_state_heads=False,
        backend="lgbm",
        target_mode="head_top_candidate",
        min_rank=0.50,
        frontier_gamma=3.0,
        frontier_bandwidth=0.08,
        sl_penalty=0.0,
        timeout_penalty=0.002,
        rank_residual_weight=1.0,
        validation_mode="fold_aware",
        cap=0.15,
        max_priority_multiplier=1.0,
        max_rank_adjustment=0.0,
        priority_action="adjustment",
        cap_grid="0.05,0.15,0.30",
        min_abs_z=0.50,
        min_abs_z_grid="0.0,0.5",
        selection_gate_mode="opportunity",
        selection_min_accepted_jaccard=0.95,
        selection_max_full_sl_delta=0.005,
        selection_max_timeout_delta=0.0,
        arm_contains="cap_0p15_zge_0p5",
        use_selected_challenger=True,
        readiness_existing_manifest=None,
        readiness_output_dir=None,
        readiness_min_timestamp_count=3,
        readiness_min_rows=1,
    )


def test_slug_normalizes_window_labels() -> None:
    assert _slug("Jun 23 09:00-Jun 24 08:00 UTC") == "jun_23_09_00_jun_24_08_00_utc"
    assert _slug("  ") == "window"


def test_needs_run_requires_all_outputs_unless_forced(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.write_text("ok", encoding="utf-8")

    assert _needs_run([first, second], force=False) is True
    second.write_text("ok", encoding="utf-8")
    assert _needs_run([first, second], force=False) is False
    assert _needs_run([first, second], force=True) is True


def test_score_command_uses_list_form_and_preserves_rank_contract_inputs(tmp_path: Path) -> None:
    args = _args(tmp_path)
    cmd = _score_command(args, candidate=tmp_path / "candidates.parquet", output_dir=tmp_path / "score")

    assert isinstance(cmd, list)
    assert "score_market_state_controller_bundle.py" in cmd[2]
    assert "--eval-candidates" in cmd
    assert str(tmp_path / "candidates.parquet") in cmd
    assert "--eval-feature-store-dir" in cmd
    assert str(args.feature_store_dir) in cmd
    assert "--train-deployable-candidates" in cmd


def test_priority_and_cap_commands_keep_fixed_shadow_parameters(tmp_path: Path) -> None:
    args = _args(tmp_path)
    learning_cmd = _priority_command(args, score_dir=tmp_path / "score", output_dir=tmp_path / "priority")
    cap_cmd = _cap_sweep_command(args, priority_dir=tmp_path / "priority", output_dir=tmp_path / "cap")

    assert "run_market_state_head_priority_learning.py" in learning_cmd[2]
    assert "--state-arm" in learning_cmd
    assert learning_cmd[learning_cmd.index("--state-arm") + 1] == "S2_observed_forecast_shared_response"
    assert "--state-head-statuses" in learning_cmd
    assert learning_cmd[learning_cmd.index("--state-head-statuses") + 1] == "active_candidate"
    assert "--use-all-state-heads" not in learning_cmd
    assert "--max-adjustment" in learning_cmd
    assert learning_cmd[learning_cmd.index("--max-adjustment") + 1] == "0.15"
    assert "--rank-residual-weight" in learning_cmd
    assert learning_cmd[learning_cmd.index("--rank-residual-weight") + 1] == "1.0"
    assert "--max-priority-multiplier" in learning_cmd
    assert learning_cmd[learning_cmd.index("--max-priority-multiplier") + 1] == "1.0"
    assert "--max-rank-adjustment" in learning_cmd
    assert learning_cmd[learning_cmd.index("--max-rank-adjustment") + 1] == "0.0"
    assert "--priority-action" in learning_cmd
    assert learning_cmd[learning_cmd.index("--priority-action") + 1] == "adjustment"
    assert "--caps" in cap_cmd
    assert cap_cmd[cap_cmd.index("--caps") + 1] == "0.05,0.15,0.30"
    assert "--min-abs-z-thresholds" in cap_cmd
    assert cap_cmd[cap_cmd.index("--min-abs-z-thresholds") + 1] == "0.0,0.5"
    assert "--selection-gate-mode" in cap_cmd
    assert cap_cmd[cap_cmd.index("--selection-gate-mode") + 1] == "opportunity"
    assert "--selection-min-accepted-jaccard" in cap_cmd
    assert cap_cmd[cap_cmd.index("--selection-min-accepted-jaccard") + 1] == "0.95"
    assert "--selection-max-full-sl-delta" in cap_cmd
    assert cap_cmd[cap_cmd.index("--selection-max-full-sl-delta") + 1] == "0.005"
    assert "--selection-max-timeout-delta" in cap_cmd
    assert cap_cmd[cap_cmd.index("--selection-max-timeout-delta") + 1] == "0.0"
    assert "replay_market_state_learned_priority_cap_sweep.py" in cap_cmd[2]

    manifest = tmp_path / "t1_manifest.json"
    learning_with_manifest = _priority_command(
        args,
        score_dir=tmp_path / "score",
        output_dir=tmp_path / "priority",
        static_baseline_manifest=str(manifest),
    )
    cap_with_manifest = _cap_sweep_command(
        args,
        priority_dir=tmp_path / "priority",
        output_dir=tmp_path / "cap",
        static_baseline_manifest=str(manifest),
    )
    assert "--static-baseline-manifest" in learning_with_manifest
    assert learning_with_manifest[learning_with_manifest.index("--static-baseline-manifest") + 1] == str(manifest)
    assert "--static-baseline-manifest" in cap_with_manifest
    assert cap_with_manifest[cap_with_manifest.index("--static-baseline-manifest") + 1] == str(manifest)


def test_priority_command_can_use_all_observed_state_heads(tmp_path: Path) -> None:
    args = _args(tmp_path)
    args.state_arm = "S1_observed_axes_shared_response"
    args.state_head_statuses = "active,disabled_candidate"
    args.use_all_state_heads = True

    learning_cmd = _priority_command(args, score_dir=tmp_path / "score", output_dir=tmp_path / "priority")

    assert "--state-arm" in learning_cmd
    assert learning_cmd[learning_cmd.index("--state-arm") + 1] == "S1_observed_axes_shared_response"
    assert "--use-all-state-heads" in learning_cmd
    assert "--state-head-statuses" in learning_cmd
    assert learning_cmd[learning_cmd.index("--state-head-statuses") + 1] == "active,disabled_candidate"


def test_audit_command_pairs_cap_dirs_and_labels(tmp_path: Path) -> None:
    args = _args(tmp_path)
    cmd = _audit_command(
        args,
        cap_dirs=[tmp_path / "cap1", tmp_path / "cap2"],
        labels=["one", "two"],
        output_dir=tmp_path / "audit",
    )

    assert cmd.count("--cap-sweep-dir") == 2
    assert cmd.count("--window-label") == 2
    assert "--arm-contains" in cmd
    assert cmd[cmd.index("--arm-contains") + 1] == "cap_0p15_zge_0p5"
    assert "--use-selected-challenger" in cmd


def test_shadow_priority_contract_fields_are_not_executable() -> None:
    contract = _shadow_priority_contract_fields()

    assert contract["operational_status"] == "shadow_only"
    assert contract["execution_enabled"] is False
    assert contract["production_eligible"] is False
    assert contract["requires_promotion_gate"] is True
    assert contract["qfail_active"] is False
    assert contract["head_health_active"] is False
    assert contract["market_state_threshold_controller_active"] is False
    assert contract["priority_action"] == "portfolio_priority_adjustment_shadow_only"
    assert contract["changes_scores_or_thresholds"] is False
    assert contract["changes_auction_ordering"] is True


def test_read_window_infers_candidate_rank_contract(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    candidate_dir = artifact / "simple_policy_optimiser"
    candidate_dir.mkdir(parents=True)
    candidate = candidate_dir / "simple_policy_candidates.parquet"
    pd = __import__("pandas")
    pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-06-24T09:00:00Z", "2026-06-24T10:00:00Z"]),
            "head": ["short_asset", "short_boll"],
        }
    ).to_parquet(candidate)
    (artifact / "t1_repaired_static_baseline_manifest.json").write_text(
        """{
          "generated_by": "materialize_t1_repaired_static_baseline",
          "active_stack": {
            "rank_contract": "anchor_global_policy_rank_reference",
            "rank_scope": "global_over_time",
            "rank_reference_run_id": "prejune_rankref",
            "promotion_status": "rank_contract_challenger",
            "enabled_heads": ["short_asset", "short_boll"],
            "disabled_heads": ["long_bars", "long_dist"],
            "qfail_active": false,
            "head_health_active": false,
            "market_state_threshold_controller_active": false,
            "policy_variant": "refit_bar4_strategy_bar2",
            "auction": "global_auction"
          }
        }""",
        encoding="utf-8",
    )

    window = _read_window(candidate)

    assert window["rows"] == 2
    assert window["heads"] == ["short_asset", "short_boll"]
    assert window["contract"]["rank_scope"] == "global_over_time"
    assert window["contract"]["rank_contract"] == "anchor_global_policy_rank_reference"
    assert window["contract"]["rank_reference_run_id"] == "prejune_rankref"


def test_static_baseline_parity_checks_materialized_manifest_summary(tmp_path: Path) -> None:
    pd = __import__("pandas")
    artifact = tmp_path / "artifact"
    manifest = artifact / "t1_repaired_static_baseline_manifest.json"
    _write_json(
        manifest,
        {
            "summary": {
                "trade_count": 7,
                "net_pnl": 12.5,
                "gross_pnl": 20.0,
                "cost_pnl": 7.5,
                "full_sl_rate": 0.1,
                "timeout_rate": 0.2,
                "worst_24h_net_pnl": -3.0,
            }
        },
    )
    priority_dir = tmp_path / "priority"
    priority_dir.mkdir()
    pd.DataFrame(
        [
            {
                "arm": "P0_static_priority",
                "trade_count": 7,
                "net_pnl": 12.5,
                "gross_pnl": 20.0,
                "cost_pnl": 7.5,
                "full_sl_rate": 0.1,
                "timeout_rate": 0.2,
                "worst_24h_net_pnl": -3.0,
            }
        ]
    ).to_csv(priority_dir / "head_priority_learning_replay_summary.csv", index=False)
    window = {"candidate": {"contract": {"manifest_path": str(manifest)}}}

    parity = _static_baseline_parity(window=window, priority_dir=priority_dir)

    assert parity["checked"] is True
    assert parity["passed"] is True
    assert parity["failures"] == []

    pd.DataFrame(
        [
            {
                "arm": "P0_static_priority",
                "trade_count": 8,
                "net_pnl": 11.0,
                "gross_pnl": 20.0,
                "cost_pnl": 7.5,
                "full_sl_rate": 0.1,
                "timeout_rate": 0.2,
                "worst_24h_net_pnl": -3.0,
            }
        ]
    ).to_csv(priority_dir / "head_priority_learning_replay_summary.csv", index=False)

    mismatch = _static_baseline_parity(window=window, priority_dir=priority_dir)

    assert mismatch["checked"] is True
    assert mismatch["passed"] is False
    assert mismatch["failures"] == ["trade_count", "net_pnl"]
    assert mismatch["deltas"]["trade_count"] == 1.0
    assert mismatch["deltas"]["net_pnl"] == -1.5


def test_load_static_baseline_artifacts_relabels_frozen_t1_arm(tmp_path: Path) -> None:
    pd = __import__("pandas")
    root = tmp_path / "artifact"
    policy_dir = root / "simple_policy_optimiser"
    policy_dir.mkdir(parents=True)
    summary_path = policy_dir / "portfolio_replay_summary.csv"
    accepted_path = policy_dir / "accepted_trades.parquet"
    decisions_path = policy_dir / "portfolio_decisions.parquet"
    equity_path = policy_dir / "equity_curve.parquet"
    by_head_path = policy_dir / "portfolio_by_head.csv"
    pd.DataFrame(
        [
            {
                "arm": "production_T1_repaired_static_baseline",
                "trade_count": 1,
                "net_pnl": 2.5,
                "gross_pnl": 3.0,
                "cost_pnl": 0.5,
            }
        ]
    ).to_csv(summary_path, index=False)
    pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2026-06-24T00:00:00Z"),
                "symbol": "BTC/USD:USD",
                "side": "short",
                "strategy_id": "short_boll_test",
                "arm": "production_T1_repaired_static_baseline",
                "net_pnl": 2.5,
            }
        ]
    ).to_parquet(accepted_path, index=False)
    pd.DataFrame([{"accepted": True}]).to_parquet(decisions_path, index=False)
    pd.DataFrame([{"timestamp": pd.Timestamp("2026-06-24T00:00:00Z"), "wallet": 10002.5}]).to_parquet(
        equity_path,
        index=False,
    )
    pd.DataFrame([{"arm": "production_T1_repaired_static_baseline", "head": "short_boll"}]).to_csv(
        by_head_path,
        index=False,
    )
    manifest = root / "t1_repaired_static_baseline_manifest.json"
    _write_json(
        manifest,
        {
            "generated_by": "materialize_t1_repaired_static_baseline",
            "active_stack": {"rank_contract": "short_boll_timestamp_rank", "rank_scope": "within_timestamp"},
            "outputs": {
                "summary": str(summary_path),
                "accepted_trades": str(accepted_path),
                "decisions": str(decisions_path),
                "equity_curve": str(equity_path),
                "by_head": str(by_head_path),
            },
        },
    )

    loaded = _load_static_baseline_artifacts(manifest, arm=BASELINE_ARM)

    assert loaded is not None
    decisions, equity, accepted, summary, by_head, info = loaded
    assert int(summary.iloc[0]["trade_count"]) == 1
    assert summary.iloc[0]["arm"] == BASELINE_ARM
    assert accepted.iloc[0]["arm"] == BASELINE_ARM
    assert by_head.iloc[0]["arm"] == BASELINE_ARM
    assert len(decisions) == 1
    assert len(equity) == 1
    assert info["rank_contract"] == "short_boll_timestamp_rank"
    assert info["accepted_rows"] == 1


def test_resolve_run_rank_contract_rejects_mixed_rank_scopes() -> None:
    windows = [
        {"candidate": {"contract": {"rank_scope": "global_over_time", "rank_contract": "global"}}},
        {"candidate": {"contract": {"rank_scope": "within_timestamp", "rank_contract": "timestamp"}}},
    ]

    try:
        _resolve_run_rank_contract(windows)
    except ValueError as exc:
        assert "mixed candidate rank scopes" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("expected mixed rank scopes to fail")

    resolved = _resolve_run_rank_contract(windows, allow_mixed_rank_contracts=True)
    assert resolved["rank_contract_preserved"] == "mixed"
    assert resolved["candidate_rank_scopes"] == ["global_over_time", "within_timestamp"]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_preflight_candidate(root: Path, timestamps: list[str]) -> Path:
    pd = __import__("pandas")
    candidate_dir = root / "simple_policy_optimiser"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for ts in timestamps:
        for head in ["short_asset", "short_boll"]:
            rows.append(
                {
                    "timestamp": pd.Timestamp(ts),
                    "head": head,
                    "symbol": f"{head}_BTC",
                    "strategy_id": f"{head}_strategy",
                    "side": "short",
                }
            )
    candidate = candidate_dir / "simple_policy_candidates.parquet"
    pd.DataFrame(rows).to_parquet(candidate)
    _write_json(
        root / "t1_repaired_static_baseline_manifest.json",
        {
            "active_stack": {
                "rank_contract": "anchor_global_policy_rank_reference",
                "rank_scope": "global_over_time",
                "rank_reference_run_id": "prejune_rankref",
                "promotion_status": "rank_contract_challenger",
                "policy_variant": "refit_bar4_strategy_bar2",
                "enabled_heads": ["short_asset", "short_boll"],
                "disabled_heads": ["long_bars", "long_dist"],
                "qfail_active": False,
                "head_health_active": None,
                "market_state_threshold_controller_active": False,
            }
        },
    )
    return candidate


def _write_preflight_existing_manifest(path: Path) -> None:
    _write_json(
        path,
        {
            "contract": {
                "candidate_rank_contracts": [
                    {
                        "rank_contract": "anchor_global_policy_rank_reference",
                        "rank_scope": "global_over_time",
                        "rank_reference_run_id": "prejune_rankref",
                    }
                ]
            },
            "windows": [
                {
                    "label": "existing",
                    "candidate": {
                        "sha256": "oldsha",
                        "start": "2026-06-15T00:00:00+00:00",
                        "end": "2026-06-15T02:00:00+00:00",
                        "contract": {"policy_variant": "refit_bar4_strategy_bar2"},
                    },
                }
            ],
        },
    )


def test_readiness_preflight_is_opt_in(tmp_path: Path) -> None:
    args = _args(tmp_path)
    assert _run_readiness_preflight(args, [tmp_path / "missing.parquet"]) is None


def test_readiness_preflight_passes_fresh_compatible_window(tmp_path: Path) -> None:
    args = _args(tmp_path)
    existing = tmp_path / "existing" / "manifest.json"
    _write_preflight_existing_manifest(existing)
    candidate = _write_preflight_candidate(
        tmp_path / "candidate",
        ["2026-06-24T09:00:00Z", "2026-06-24T10:00:00Z", "2026-06-24T11:00:00Z"],
    )
    args.readiness_existing_manifest = existing
    args.readiness_output_dir = tmp_path / "readiness"

    summary = _run_readiness_preflight(args, [candidate])

    assert summary is not None
    assert summary["passed"] is True
    assert (args.readiness_output_dir / "market_state_priority_window_readiness_report.md").exists()


def test_readiness_preflight_fails_overlapping_append_window(tmp_path: Path) -> None:
    args = _args(tmp_path)
    existing = tmp_path / "existing" / "manifest.json"
    _write_preflight_existing_manifest(existing)
    candidate = _write_preflight_candidate(
        tmp_path / "candidate",
        ["2026-06-15T00:00:00Z", "2026-06-15T01:00:00Z", "2026-06-15T02:00:00Z"],
    )
    args.readiness_existing_manifest = existing
    args.readiness_output_dir = tmp_path / "readiness"

    try:
        _run_readiness_preflight(args, [candidate])
    except RuntimeError as exc:
        assert "window readiness preflight failed" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("expected readiness preflight to fail")
