from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts import run_t1_rank_validation_period as runner


def _args(tmp_path: Path, **overrides):
    defaults = {
        "start": "2026-06-23T09:00:00Z",
        "end": "2026-06-24T08:00:00Z",
        "run_id": "unit_t1_validation",
        "data_root": tmp_path / "data_perp",
        "feature_store_dir": tmp_path / "features",
        "feature_run_id": "features_unit",
        "policy_artifact_run_id": "policy_unit",
        "model_artifact_run_id": "model_unit",
        "reference_candidates": tmp_path / "reference_candidates.parquet",
        "train_deployable_candidates": tmp_path / "train_deployable.parquet",
        "policy_manifest": tmp_path / "policy_manifest.json",
        "policy_variant": "refit_bar4_strategy_bar2",
        "rank_reference_run_id": "rankref_unit",
        "pre_june_walkforward_dir": tmp_path / "prejune",
        "later_comparison_dir": [tmp_path / "older_later_block"],
        "active_head": [],
        "symbols": "",
        "max_symbols": 0,
        "chunk_rows": 128,
        "market_mode": "perps",
        "exchange": "krakenfutures",
        "cached_matrix_only": False,
        "min_later_timestamps": 24,
        "min_later_base_trades": 30,
        "min_later_challenger_trades": 30,
        "dry_run": True,
    }
    defaults.update(overrides)
    return type("Args", (), defaults)()


def _cmd_by_name(steps: list[runner.Step]) -> dict[str, list[str]]:
    return {step.name: step.command for step in steps}


def test_build_steps_preserves_fixed_t1_contract(tmp_path: Path) -> None:
    args = _args(tmp_path)
    paths = runner.period_paths(data_root=args.data_root, run_id=args.run_id)

    steps = runner.build_steps(args, paths)
    commands = _cmd_by_name(steps)

    assert [step.name for step in steps] == [
        "build_sample_ledger",
        "score_live_finalfit_anchor_meta",
        "materialize_policy_candidates_from_anchor_scores",
        "bridge_to_t1_anchor_scored_candidates",
        "audit_t1_rank_validation_inputs",
        "replay_timestamp_rank_t1",
        "replay_global_rank_challenger",
        "compare_timestamp_vs_global",
        "audit_combined_rank_contract_evidence",
    ]
    assert commands["replay_timestamp_rank_t1"][
        commands["replay_timestamp_rank_t1"].index("--rank-contract") + 1
    ] == "short_boll_timestamp_rank"
    assert commands["replay_global_rank_challenger"][
        commands["replay_global_rank_challenger"].index("--rank-contract") + 1
    ] == "anchor_global_policy_rank_reference"
    assert commands["replay_timestamp_rank_t1"][
        commands["replay_timestamp_rank_t1"].index("--disable-heads") + 1
    ] == "long_bars,long_dist"
    assert "--market-state-controller-bundle" not in commands[
        "materialize_policy_candidates_from_anchor_scores"
    ]
    assert "--score-column" in commands["materialize_policy_candidates_from_anchor_scores"]
    assert "calibrated_score" in commands["materialize_policy_candidates_from_anchor_scores"]
    assert commands["materialize_policy_candidates_from_anchor_scores"][
        commands["materialize_policy_candidates_from_anchor_scores"].index("--exchange") + 1
    ] == "krakenfutures"


def test_default_reference_candidates_use_floor070_t1_policy_root() -> None:
    assert "reliability_blend_native_simple_policy_replay_20260624_floor070" in str(
        runner.DEFAULT_REFERENCE_CANDIDATES
    )
    assert "longdist050" not in str(runner.DEFAULT_REFERENCE_CANDIDATES)


def test_live_scoring_uses_only_active_head_strategy_ids(tmp_path: Path) -> None:
    args = _args(tmp_path, active_head=["short_boll"])
    paths = runner.period_paths(data_root=args.data_root, run_id=args.run_id)

    commands = _cmd_by_name(runner.build_steps(args, paths))
    score_cmd = commands["score_live_finalfit_anchor_meta"]
    strategy_values = [
        score_cmd[idx + 1]
        for idx, value in enumerate(score_cmd)
        if value == "--strategy-id"
    ]

    assert strategy_values == [runner.STRATEGY_IDS["short_boll"]]
    assert runner.STRATEGY_IDS["short_asset"] not in strategy_values


def test_evidence_audit_includes_new_comparison_first_and_extra_later_blocks(tmp_path: Path) -> None:
    args = _args(tmp_path)
    paths = runner.period_paths(data_root=args.data_root, run_id=args.run_id)

    commands = _cmd_by_name(runner.build_steps(args, paths))
    evidence_cmd = commands["audit_combined_rank_contract_evidence"]
    later_values = [
        evidence_cmd[idx + 1]
        for idx, value in enumerate(evidence_cmd)
        if value == "--later-comparison-dir"
    ]

    assert later_values[0] == str(paths.comparison_dir)
    assert later_values[1:] == [str(tmp_path / "older_later_block")]


def test_write_manifest_records_contract_and_commands(tmp_path: Path) -> None:
    args = _args(tmp_path)
    paths = runner.period_paths(data_root=args.data_root, run_id=args.run_id)
    steps = runner.build_steps(args, paths)

    runner.write_manifest(
        args=args,
        paths=paths,
        all_steps=steps,
        selected_steps=steps[:2],
        dry_run=True,
        completed=[],
    )

    payload = json.loads(paths.manifest_path.read_text(encoding="utf-8"))
    assert payload["dry_run"] is True
    assert payload["contract"]["fixed_policy_contract"]["qfail_active"] is False
    assert payload["contract"]["fixed_policy_contract"]["market_state_threshold_controller_active"] is False
    assert payload["contract"]["baseline"] == "short_boll_timestamp_rank"
    assert payload["contract"]["challenger"] == "anchor_global_policy_rank_reference"
    assert payload["paths"]["comparison_dir"] == str(paths.comparison_dir)
    assert len(payload["steps"]) == len(steps)
    assert [step["name"] for step in payload["selected_steps"]] == [
        "build_sample_ledger",
        "score_live_finalfit_anchor_meta",
    ]


def test_default_run_id_is_period_specific() -> None:
    start = pd.Timestamp("2026-06-23T09:00:00Z")
    end = pd.Timestamp("2026-06-24T08:00:00Z")

    assert runner.default_run_id(start, end) == "t1_rank_validation_20260623_0900_20260624_0800"


def test_step_outputs_exist_requires_all_outputs(tmp_path: Path) -> None:
    one = tmp_path / "one.txt"
    two = tmp_path / "two.txt"
    step = runner.Step("example", ["python", "example.py"], {"one": str(one), "two": str(two)})

    one.write_text("ok", encoding="utf-8")
    assert runner._step_outputs_exist(step) is False

    two.write_text("ok", encoding="utf-8")
    assert runner._step_outputs_exist(step) is True


def test_slice_steps_selects_contiguous_named_range(tmp_path: Path) -> None:
    args = _args(tmp_path)
    paths = runner.period_paths(data_root=args.data_root, run_id=args.run_id)
    steps = runner.build_steps(args, paths)

    selected = runner.slice_steps(
        steps,
        start_at="bridge_to_t1_anchor_scored_candidates",
        stop_after="replay_timestamp_rank_t1",
    )

    assert [step.name for step in selected] == [
        "bridge_to_t1_anchor_scored_candidates",
        "audit_t1_rank_validation_inputs",
        "replay_timestamp_rank_t1",
    ]


def test_slice_steps_rejects_inverted_range(tmp_path: Path) -> None:
    args = _args(tmp_path)
    paths = runner.period_paths(data_root=args.data_root, run_id=args.run_id)
    steps = runner.build_steps(args, paths)

    try:
        runner.slice_steps(
            steps,
            start_at="compare_timestamp_vs_global",
            stop_after="build_sample_ledger",
        )
    except ValueError as exc:
        assert "at or before" in str(exc)
    else:
        raise AssertionError("slice_steps accepted an inverted range")
