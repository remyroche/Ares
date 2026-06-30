#!/usr/bin/env python3
"""Run an end-to-end fixed-contract T1 rank-validation period.

This orchestrates the manual chain used for later T1 evidence blocks:

1. build a concrete active-head sample ledger from the feature store;
2. score it with the live final-fit anchor/meta stack;
3. materialize policy candidate outcomes from those anchor scores;
4. bridge the materialized candidates into an explicit T1 anchor-scored root;
5. replay timestamp-rank T1 and the causal global-rank challenger;
6. compare both fixed-policy artifacts and refresh the rank-contract evidence.

The script deliberately does not enable q-fail, native reliability blend scoring,
HeadHealth, or the market-state threshold controller.  The only replay arm
difference is the short_boll rank contract.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_t1_feature_store_sample_ledger import DEFAULT_ACTIVE_HEADS, STRATEGY_IDS


DEFAULT_MODEL_ARTIFACT_RUN_ID = "20260618_081800_current4_final_fit"
DEFAULT_POLICY_ARTIFACT_RUN_ID = DEFAULT_MODEL_ARTIFACT_RUN_ID
DEFAULT_FEATURE_RUN_ID = "20260627_120000"
DEFAULT_RANK_REFERENCE_RUN_ID = "reliability_blend_anchor_rank_reference_20260625_prejune"
DEFAULT_REFERENCE_CANDIDATES = Path(
    "data_perp/artifacts/reliability_blend_native_simple_policy_replay_20260624_floor070"
    "/simple_policy_optimiser/simple_policy_candidates_broad.parquet"
)
DEFAULT_TRAIN_DEPLOYABLE_CANDIDATES = Path(
    "data_perp/artifacts/reliability_blend_native_simple_policy_replay_20260624_floor070"
    "/simple_policy_optimiser/simple_policy_candidates.parquet"
)
DEFAULT_POLICY_MANIFEST = Path(
    "data_perp/reports/reliability_blend_component_arm_portfolio_ablation_20260625"
    "/A0_anchor_only/portfolio_policy_ablation_manifest.json"
)
DEFAULT_PREJUNE_WALKFORWARD_DIR = Path(
    "data_perp/reports/t1_rank_contract_walkforward_20260626_prejune_timestamp_vs_global_v2"
)


@dataclass(frozen=True)
class Step:
    name: str
    command: list[str]
    outputs: dict[str, str]


@dataclass(frozen=True)
class PeriodPaths:
    run_id: str
    report_dir: Path
    sample_ledger: Path
    live_score_dir: Path
    live_score_ledger: Path
    policy_candidate_run_id: str
    policy_candidate_root: Path
    policy_candidate_broad: Path
    t1_anchor_candidate_root: Path
    t1_anchor_candidate_broad: Path
    input_audit_dir: Path
    timestamp_t1_root: Path
    global_t1_root: Path
    comparison_dir: Path
    evidence_dir: Path
    manifest_path: Path


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _timestamp(value: str, *, name: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    ts = ts.tz_convert("UTC")
    if pd.isna(ts):
        raise ValueError(f"{name} is not a valid timestamp: {value!r}")
    return ts


def _slug_timestamp(value: pd.Timestamp) -> str:
    ts = value.tz_convert("UTC")
    return ts.strftime("%Y%m%d_%H%M")


def default_run_id(start: pd.Timestamp, end: pd.Timestamp) -> str:
    return f"t1_rank_validation_{_slug_timestamp(start)}_{_slug_timestamp(end)}"


def period_paths(*, data_root: Path, run_id: str) -> PeriodPaths:
    report_dir = data_root / "reports" / run_id
    live_score_dir = report_dir / "live_finalfit_anchor_scores"
    sample_ledger = report_dir / "t1_active_heads_sample_ledger.parquet"
    policy_candidate_run_id = f"{run_id}_anchor_policy_candidates_tmp"
    policy_candidate_root = data_root / "artifacts" / policy_candidate_run_id
    t1_anchor_candidate_root = data_root / "artifacts" / f"{run_id}_anchor_scored_candidates"
    timestamp_t1_root = data_root / "artifacts" / f"{run_id}_timestamp_rank_t1"
    global_t1_root = data_root / "artifacts" / f"{run_id}_global_rank_challenger"
    comparison_dir = data_root / "reports" / f"{run_id}_timestamp_vs_global"
    evidence_dir = data_root / "reports" / f"{run_id}_evidence_audit"
    return PeriodPaths(
        run_id=run_id,
        report_dir=report_dir,
        sample_ledger=sample_ledger,
        live_score_dir=live_score_dir,
        live_score_ledger=live_score_dir / "combined_prediction_ledger.parquet",
        policy_candidate_run_id=policy_candidate_run_id,
        policy_candidate_root=policy_candidate_root,
        policy_candidate_broad=policy_candidate_root
        / "simple_policy_optimiser"
        / "simple_policy_candidates_broad.parquet",
        t1_anchor_candidate_root=t1_anchor_candidate_root,
        t1_anchor_candidate_broad=t1_anchor_candidate_root
        / "simple_policy_optimiser"
        / "simple_policy_candidates_broad.parquet",
        input_audit_dir=data_root / "reports" / f"{run_id}_input_audit",
        timestamp_t1_root=timestamp_t1_root,
        global_t1_root=global_t1_root,
        comparison_dir=comparison_dir,
        evidence_dir=evidence_dir,
        manifest_path=report_dir / "t1_rank_validation_period_manifest.json",
    )


def _python(script: str) -> list[str]:
    return [sys.executable, "-u", script]


def _append_many(command: list[str], flag: str, values: list[str]) -> None:
    for value in values:
        command.extend([flag, str(value)])


def build_steps(args: argparse.Namespace, paths: PeriodPaths) -> list[Step]:
    start = _timestamp(str(args.start), name="start")
    end = _timestamp(str(args.end), name="end")
    heads = tuple(args.active_head or DEFAULT_ACTIVE_HEADS)
    unknown = sorted(set(heads).difference(STRATEGY_IDS))
    if unknown:
        raise ValueError(f"Unknown active head(s): {unknown}")
    strategy_ids = [STRATEGY_IDS[head] for head in heads]

    feature_store_dir = args.feature_store_dir or (Path(args.data_root) / "features" / args.feature_run_id)
    sample_cmd = _python("scripts/build_t1_feature_store_sample_ledger.py") + [
        "--feature-store-dir",
        str(feature_store_dir),
        "--start",
        start.isoformat(),
        "--end",
        end.isoformat(),
        "--output",
        str(paths.sample_ledger),
    ]
    _append_many(sample_cmd, "--head", list(heads))
    if args.symbols:
        sample_cmd.extend(["--symbols", str(args.symbols)])
    if int(args.max_symbols) > 0:
        sample_cmd.extend(["--max-symbols", str(int(args.max_symbols))])

    score_cmd = _python("scripts/generate_live_finalfit_oos_predictions.py") + [
        "--data-root",
        str(args.data_root),
        "--policy-artifact-run-id",
        str(args.policy_artifact_run_id),
        "--model-artifact-run-id",
        str(args.model_artifact_run_id),
        "--feature-run-id",
        str(args.feature_run_id),
        "--sample-ledger",
        str(paths.sample_ledger),
        "--min-timestamp",
        start.isoformat(),
        "--max-timestamp",
        end.isoformat(),
        "--chunk-rows",
        str(int(args.chunk_rows)),
        "--output-dir",
        str(paths.live_score_dir),
    ]
    _append_many(score_cmd, "--strategy-id", strategy_ids)
    if bool(args.cached_matrix_only):
        score_cmd.append("--cached-matrix-only")

    materialize_cmd = _python("scripts/materialize_live_ledger_blend_native_candidates.py") + [
        "--ledger",
        str(paths.live_score_ledger),
        "--score-column",
        "calibrated_score",
        "--allow-ledger-score",
        "--reference-candidates",
        str(args.reference_candidates),
        "--data-root",
        str(args.data_root),
        "--market-mode",
        str(args.market_mode),
        "--exchange",
        str(args.exchange),
        "--output-run-id",
        paths.policy_candidate_run_id,
        "--start",
        start.isoformat(),
        "--end",
        end.isoformat(),
        "--rank-reference-run-id",
        str(args.rank_reference_run_id),
    ]

    bridge_cmd = _python("scripts/materialize_t1_anchor_scored_candidates.py") + [
        "--candidates",
        str(paths.policy_candidate_broad),
        "--score-ledger",
        str(paths.live_score_ledger),
        "--output-dir",
        str(paths.t1_anchor_candidate_root),
    ]
    _append_many(bridge_cmd, "--active-head", list(heads))

    audit_cmd = _python("scripts/audit_t1_rank_validation_inputs.py") + [
        "--candidate-root",
        str(paths.t1_anchor_candidate_root),
        "--min-timestamp",
        start.isoformat(),
        "--output-dir",
        str(paths.input_audit_dir),
    ]

    timestamp_cmd = _python("scripts/materialize_t1_repaired_static_baseline.py") + [
        "--eval-candidates",
        str(paths.t1_anchor_candidate_broad),
        "--train-deployable-candidates",
        str(args.train_deployable_candidates),
        "--policy-manifest",
        str(args.policy_manifest),
        "--policy-variant",
        str(args.policy_variant),
        "--rank-contract",
        "short_boll_timestamp_rank",
        "--rank-reference-run-id",
        str(args.rank_reference_run_id),
        "--data-root",
        str(args.data_root),
        "--disable-heads",
        "long_bars,long_dist",
        "--output-dir",
        str(paths.timestamp_t1_root),
        "--market-mode",
        str(args.market_mode),
    ]
    global_cmd = list(timestamp_cmd)
    global_cmd[global_cmd.index("short_boll_timestamp_rank")] = "anchor_global_policy_rank_reference"
    global_cmd[global_cmd.index(str(paths.timestamp_t1_root))] = str(paths.global_t1_root)

    compare_cmd = _python("scripts/compare_t1_rank_contracts.py") + [
        "--base-dir",
        str(paths.timestamp_t1_root),
        "--challenger-dir",
        str(paths.global_t1_root),
        "--output-dir",
        str(paths.comparison_dir),
        "--base-name",
        "timestamp_rank_t1",
        "--challenger-name",
        "global_rank_challenger",
    ]

    later_dirs = [str(paths.comparison_dir)] + [str(path) for path in args.later_comparison_dir]
    evidence_cmd = _python("scripts/audit_t1_rank_contract_evidence.py") + [
        "--pre-june-walkforward-dir",
        str(args.pre_june_walkforward_dir),
        "--output-dir",
        str(paths.evidence_dir),
        "--min-later-timestamps",
        str(int(args.min_later_timestamps)),
        "--min-later-base-trades",
        str(int(args.min_later_base_trades)),
        "--min-later-challenger-trades",
        str(int(args.min_later_challenger_trades)),
    ]
    _append_many(evidence_cmd, "--later-comparison-dir", later_dirs)

    return [
        Step("build_sample_ledger", sample_cmd, {"sample_ledger": str(paths.sample_ledger)}),
        Step("score_live_finalfit_anchor_meta", score_cmd, {"score_ledger": str(paths.live_score_ledger)}),
        Step(
            "materialize_policy_candidates_from_anchor_scores",
            materialize_cmd,
            {"policy_candidate_broad": str(paths.policy_candidate_broad)},
        ),
        Step(
            "bridge_to_t1_anchor_scored_candidates",
            bridge_cmd,
            {"t1_anchor_candidate_broad": str(paths.t1_anchor_candidate_broad)},
        ),
        Step("audit_t1_rank_validation_inputs", audit_cmd, {"input_audit": str(paths.input_audit_dir)}),
        Step("replay_timestamp_rank_t1", timestamp_cmd, {"timestamp_t1_root": str(paths.timestamp_t1_root)}),
        Step("replay_global_rank_challenger", global_cmd, {"global_t1_root": str(paths.global_t1_root)}),
        Step("compare_timestamp_vs_global", compare_cmd, {"comparison_dir": str(paths.comparison_dir)}),
        Step("audit_combined_rank_contract_evidence", evidence_cmd, {"evidence_dir": str(paths.evidence_dir)}),
    ]


def slice_steps(steps: list[Step], *, start_at: str = "", stop_after: str = "") -> list[Step]:
    """Return the requested contiguous step slice by step name."""

    names = [step.name for step in steps]
    start_idx = 0
    stop_idx = len(steps)
    if start_at:
        if start_at not in names:
            raise ValueError(f"Unknown --start-at step {start_at!r}. Valid steps: {names}")
        start_idx = names.index(start_at)
    if stop_after:
        if stop_after not in names:
            raise ValueError(f"Unknown --stop-after step {stop_after!r}. Valid steps: {names}")
        stop_idx = names.index(stop_after) + 1
    if start_idx >= stop_idx:
        raise ValueError("--start-at must refer to a step at or before --stop-after")
    return steps[start_idx:stop_idx]


def write_manifest(
    *,
    args: argparse.Namespace,
    paths: PeriodPaths,
    all_steps: list[Step],
    selected_steps: list[Step],
    dry_run: bool,
    completed: list[dict[str, Any]],
) -> None:
    paths.report_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_by": "run_t1_rank_validation_period",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dry_run": bool(dry_run),
        "run_id": paths.run_id,
        "contract": {
            "baseline": "short_boll_timestamp_rank",
            "baseline_rank_scope": "within_timestamp",
            "challenger": "anchor_global_policy_rank_reference",
            "challenger_rank_scope": "global_over_time",
            "fixed_policy_contract": {
                "score_path": "anchor_meta_calibrated_score",
                "active_heads": ["short_asset", "short_boll"],
                "disabled_heads": ["long_bars", "long_dist"],
                "qfail_active": False,
                "native_reliability_blend_active": False,
                "market_state_threshold_controller_active": False,
                "static_base_thresholds": True,
                "policy_variant": str(args.policy_variant),
                "auction": "global_auction",
            },
        },
        "args": {
            key: _json_safe(value)
            for key, value in vars(args).items()
            if key not in {"func"}
        },
        "paths": _json_safe(asdict(paths)),
        "steps": [
            {
                "name": step.name,
                "command": step.command,
                "outputs": step.outputs,
            }
            for step in all_steps
        ],
        "selected_steps": [
            {
                "name": step.name,
                "command": step.command,
                "outputs": step.outputs,
            }
            for step in selected_steps
        ],
        "completed_steps": completed,
    }
    paths.manifest_path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _step_outputs_exist(step: Step) -> bool:
    if not step.outputs:
        return False
    return all(Path(path).exists() for path in step.outputs.values())


def run_steps(steps: list[Step], *, skip_existing: bool = False) -> list[dict[str, Any]]:
    completed: list[dict[str, Any]] = []
    for idx, step in enumerate(steps, start=1):
        if skip_existing and _step_outputs_exist(step):
            print(f"[{idx}/{len(steps)}] {step.name} (skipped; outputs exist)")
            completed.append(
                {
                    "name": step.name,
                    "returncode": 0,
                    "skipped": True,
                    "skip_reason": "outputs_exist",
                    "outputs": step.outputs,
                }
            )
            continue
        print(f"[{idx}/{len(steps)}] {step.name}")
        result = subprocess.run(step.command, check=False)
        completed.append(
            {
                "name": step.name,
                "returncode": int(result.returncode),
                "skipped": False,
                "outputs": step.outputs,
            }
        )
        if result.returncode != 0:
            raise SystemExit(result.returncode)
    return completed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--data-root", type=Path, default=Path("data_perp"))
    parser.add_argument("--feature-store-dir", type=Path, default=None)
    parser.add_argument("--feature-run-id", default=DEFAULT_FEATURE_RUN_ID)
    parser.add_argument("--policy-artifact-run-id", default=DEFAULT_POLICY_ARTIFACT_RUN_ID)
    parser.add_argument("--model-artifact-run-id", default=DEFAULT_MODEL_ARTIFACT_RUN_ID)
    parser.add_argument("--reference-candidates", type=Path, default=DEFAULT_REFERENCE_CANDIDATES)
    parser.add_argument("--train-deployable-candidates", type=Path, default=DEFAULT_TRAIN_DEPLOYABLE_CANDIDATES)
    parser.add_argument("--policy-manifest", type=Path, default=DEFAULT_POLICY_MANIFEST)
    parser.add_argument("--policy-variant", default="refit_bar4_strategy_bar2")
    parser.add_argument("--rank-reference-run-id", default=DEFAULT_RANK_REFERENCE_RUN_ID)
    parser.add_argument("--pre-june-walkforward-dir", type=Path, default=DEFAULT_PREJUNE_WALKFORWARD_DIR)
    parser.add_argument("--later-comparison-dir", action="append", type=Path, default=[])
    parser.add_argument("--active-head", action="append", default=[])
    parser.add_argument("--symbols", default="")
    parser.add_argument("--max-symbols", type=int, default=0)
    parser.add_argument("--chunk-rows", type=int, default=512)
    parser.add_argument("--market-mode", default="perps")
    parser.add_argument("--exchange", default="krakenfutures")
    parser.add_argument("--cached-matrix-only", action="store_true")
    parser.add_argument("--min-later-timestamps", type=int, default=24)
    parser.add_argument("--min-later-base-trades", type=int, default=30)
    parser.add_argument("--min-later-challenger-trades", type=int, default=30)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help=(
            "Skip a step when all of its declared output paths already exist. "
            "This is intended for resuming an interrupted validation period "
            "without recomputing expensive scoring/materialization artifacts."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--start-at",
        default="",
        help="Start execution at this step name. Dry-run manifests still include the full plan.",
    )
    parser.add_argument(
        "--stop-after",
        default="",
        help="Stop execution after this step name. Useful for staged validation runs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = _timestamp(str(args.start), name="start")
    end = _timestamp(str(args.end), name="end")
    if end < start:
        raise SystemExit("--end must be >= --start")
    run_id = str(args.run_id or default_run_id(start, end))
    paths = period_paths(data_root=Path(args.data_root), run_id=run_id)
    all_steps = build_steps(args, paths)
    selected_steps = slice_steps(
        all_steps,
        start_at=str(args.start_at or ""),
        stop_after=str(args.stop_after or ""),
    )
    completed: list[dict[str, Any]] = []
    write_manifest(
        args=args,
        paths=paths,
        all_steps=all_steps,
        selected_steps=selected_steps,
        dry_run=bool(args.dry_run),
        completed=completed,
    )
    if bool(args.dry_run):
        print(f"Wrote dry-run manifest: {paths.manifest_path}")
        for idx, step in enumerate(selected_steps, start=1):
            print(f"[{idx}/{len(selected_steps)}] {step.name}: {' '.join(step.command)}")
        return
    completed = run_steps(selected_steps, skip_existing=bool(args.skip_existing))
    write_manifest(
        args=args,
        paths=paths,
        all_steps=all_steps,
        selected_steps=selected_steps,
        dry_run=False,
        completed=completed,
    )
    print(f"Wrote T1 rank-validation period manifest: {paths.manifest_path}")
    if len(selected_steps) == len(all_steps) and selected_steps[0].name == all_steps[0].name:
        print(f"Comparison report: {paths.comparison_dir / 't1_rank_contract_comparison_report.md'}")
        print(f"Evidence audit: {paths.evidence_dir / 't1_rank_contract_evidence_audit.md'}")
    else:
        selected_names = [step.name for step in selected_steps]
        all_names = [step.name for step in all_steps]
        last_idx = all_names.index(selected_names[-1])
        next_step = all_names[last_idx + 1] if last_idx + 1 < len(all_names) else ""
        resume_hint = (
            f"Resume with --start-at {next_step} --skip-existing."
            if next_step
            else "All selected steps are complete."
        )
        print(
            "Partial validation run complete. "
            f"{resume_hint} Omit --start-at/--stop-after for the full chain."
        )


if __name__ == "__main__":
    main()
