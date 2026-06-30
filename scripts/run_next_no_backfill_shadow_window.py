#!/usr/bin/env python3
"""Guarded runner for the next global-rank no-backfill shadow window.

This script is intentionally conservative.  It first runs the readiness logic
from ``audit_next_no_backfill_shadow_window_readiness`` and only materializes a
new T1 candidate window when enough mature feature history is available under
the active global-over-time rank contract.  If the next window is not ready, it
writes a manifest and exits without scoring anything.
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

from scripts.audit_next_no_backfill_shadow_window_readiness import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULT_DATA_ROOT,
    _latest_feature_store_dir,
    _load_json,
    build_readiness,
    update_config as update_readiness_config,
    write_readiness,
)
from scripts.run_t1_rank_validation_period import (  # noqa: E402
    DEFAULT_MODEL_ARTIFACT_RUN_ID,
    DEFAULT_POLICY_ARTIFACT_RUN_ID,
    DEFAULT_RANK_REFERENCE_RUN_ID,
)
from scripts.run_t1_rank_validation_period import (  # noqa: E402
    DEFAULT_POLICY_MANIFEST as DEFAULT_T1_POLICY_MANIFEST,
)
from scripts.run_t1_rank_validation_period import (  # noqa: E402
    DEFAULT_TRAIN_DEPLOYABLE_CANDIDATES,
)


DEFAULT_OUTPUT_DIR = Path("data_perp/reports/market_state_next_no_backfill_shadow_runner")
DEFAULT_BUNDLE = Path(
    "data_perp/reports/"
    "market_state_controller_bundle_globalrank_no_backfill_20260627_v1"
    "/market_state_controller_bundle.joblib"
)


@dataclass(frozen=True)
class Step:
    name: str
    command: list[str]
    outputs: dict[str, str]


@dataclass(frozen=True)
class PlannedPaths:
    runner_output_dir: Path
    readiness_output_dir: Path
    t1_run_id: str
    t1_anchor_candidates: Path
    score_output_dir: Path
    discovery_output_dir: Path
    monitor_output_dir: Path
    runner_manifest: Path


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
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _as_utc(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if pd.isna(ts):
        raise ValueError(f"Invalid timestamp: {value!r}")
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _slug(ts: pd.Timestamp) -> str:
    return _as_utc(ts).strftime("%Y%m%d_%H")


def _window_slug(start: pd.Timestamp, end: pd.Timestamp) -> str:
    return f"{_slug(start)}_{_slug(end)}"


def _feature_run_id(feature_store_dir: Path) -> str:
    return Path(feature_store_dir).name


def _latest_score_manifest_paths(config: dict[str, Any]) -> dict[str, Path]:
    controller = dict(config.get("market_state_controller_validation") or {})
    latest = dict(
        controller.get("global_rank_threshold_controller_no_backfill_shadow_score_latest")
        or {}
    )
    out: dict[str, Path] = {}
    score_dir_raw = latest.get("score_dir")
    if score_dir_raw:
        manifest_path = Path(str(score_dir_raw)) / "manifest.json"
        if manifest_path.exists():
            manifest = _load_json(manifest_path)
            for key in ("bundle", "policy_manifest", "train_deployable_candidates"):
                value = manifest.get(key)
                if value:
                    out[key] = Path(str(value))
    if latest.get("bundle"):
        out.setdefault("bundle", Path(str(latest["bundle"])))
    bundle_dir = latest.get("bundle_dir")
    if bundle_dir:
        out.setdefault(
            "bundle",
            Path(str(bundle_dir)) / "market_state_controller_bundle.joblib",
        )
    walkforward = dict(
        controller.get("global_rank_threshold_controller_no_backfill_walkforward")
        or {}
    )
    if walkforward.get("bundle_dir"):
        out.setdefault(
            "bundle",
            Path(str(walkforward["bundle_dir"])) / "market_state_controller_bundle.joblib",
        )
    return out


def _append_unique_score_dir(out: list[Path], seen: set[str], value: Any) -> None:
    if not value:
        return
    path = Path(str(value))
    key = str(path)
    if key in seen:
        return
    out.append(path)
    seen.add(key)


def _score_dirs_from_monitor_summary(summary_path: Path) -> list[Path]:
    if not summary_path.exists():
        return []
    try:
        summary = _load_json(summary_path)
    except Exception:
        return []
    out: list[Path] = []
    seen: set[str] = set()
    for window in summary.get("windows") or []:
        if isinstance(window, dict):
            _append_unique_score_dir(out, seen, window.get("score_dir"))
    metrics_csv_raw = summary.get("window_metrics_csv")
    if metrics_csv_raw:
        metrics_csv = Path(str(metrics_csv_raw))
        if metrics_csv.exists():
            try:
                frame = pd.read_csv(metrics_csv, usecols=["score_dir"])
            except Exception:
                frame = pd.DataFrame()
            if "score_dir" in frame.columns:
                for value in frame["score_dir"].dropna().tolist():
                    _append_unique_score_dir(out, seen, value)
    return out


def _existing_monitor_score_dirs(config: dict[str, Any]) -> list[Path]:
    controller = dict(config.get("market_state_controller_validation") or {})
    monitor = dict(
        controller.get("global_rank_threshold_controller_no_backfill_shadow_monitor")
        or {}
    )
    out: list[Path] = []
    seen: set[str] = set()
    for window in monitor.get("windows") or []:
        if not isinstance(window, dict):
            continue
        _append_unique_score_dir(out, seen, window.get("score_dir"))
    summary_json_raw = monitor.get("summary_json")
    if summary_json_raw:
        for path in _score_dirs_from_monitor_summary(Path(str(summary_json_raw))):
            _append_unique_score_dir(out, seen, path)
    monitor_dir_raw = monitor.get("monitor_dir")
    if monitor_dir_raw:
        summary_path = Path(str(monitor_dir_raw)) / "no_backfill_shadow_monitor_summary.json"
        for path in _score_dirs_from_monitor_summary(summary_path):
            _append_unique_score_dir(out, seen, path)
    metrics_csv_raw = monitor.get("window_metrics_csv")
    if metrics_csv_raw:
        metrics_csv = Path(str(metrics_csv_raw))
        if metrics_csv.exists():
            try:
                frame = pd.read_csv(metrics_csv, usecols=["score_dir"])
            except Exception:
                frame = pd.DataFrame()
            if "score_dir" in frame.columns:
                for value in frame["score_dir"].dropna().tolist():
                    _append_unique_score_dir(out, seen, value)
    return out


def _default_paths(
    *,
    data_root: Path,
    output_dir: Path,
    readiness_output_dir: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    run_id: str,
    score_output_dir: Path | None,
    discovery_output_dir: Path | None,
    monitor_output_dir: Path | None,
    feature_store_dir: Path,
) -> PlannedPaths:
    slug = _window_slug(start, end)
    t1_run_id = run_id or f"t1_rank_validation_next_no_backfill_{slug}_shadow_window"
    score_dir = score_output_dir or (
        data_root
        / "reports"
        / (
            "market_state_controller_bundle_score_globalrank_no_backfill_shadow_allstates_"
            f"{_feature_run_id(feature_store_dir)}_{slug}"
        )
    )
    discovery_dir = discovery_output_dir or (
        data_root / "reports" / f"market_state_no_backfill_shadow_window_discovery_{slug}"
    )
    monitor_dir = monitor_output_dir or (
        data_root / "reports" / f"market_state_no_backfill_shadow_monitor_globalrank_{slug}"
    )
    return PlannedPaths(
        runner_output_dir=output_dir,
        readiness_output_dir=readiness_output_dir,
        t1_run_id=t1_run_id,
        t1_anchor_candidates=(
            data_root
            / "artifacts"
            / f"{t1_run_id}_anchor_scored_candidates"
            / "simple_policy_optimiser"
            / "simple_policy_candidates_broad.parquet"
        ),
        score_output_dir=score_dir,
        discovery_output_dir=discovery_dir,
        monitor_output_dir=monitor_dir,
        runner_manifest=output_dir / "next_no_backfill_shadow_runner_manifest.json",
    )


def build_steps(
    *,
    config: dict[str, Any],
    config_path: Path,
    data_root: Path,
    feature_store_dir: Path,
    paths: PlannedPaths,
    start: pd.Timestamp,
    end: pd.Timestamp,
    bundle: Path,
    policy_manifest: Path,
    train_deployable_candidates: Path,
    policy_variant: str,
    market_mode: str,
    exchange: str,
    rank_reference_run_id: str,
    policy_artifact_run_id: str,
    model_artifact_run_id: str,
    min_timestamp_count: int,
    include_monitor_step: bool,
) -> list[Step]:
    py = [sys.executable, "-u"]
    common_start = start.isoformat()
    common_end = end.isoformat()
    t1_cmd = py + [
        "scripts/run_t1_rank_validation_period.py",
        "--start",
        common_start,
        "--end",
        common_end,
        "--run-id",
        paths.t1_run_id,
        "--data-root",
        str(data_root),
        "--feature-store-dir",
        str(feature_store_dir),
        "--feature-run-id",
        _feature_run_id(feature_store_dir),
        "--policy-artifact-run-id",
        str(policy_artifact_run_id),
        "--model-artifact-run-id",
        str(model_artifact_run_id),
        "--policy-manifest",
        str(policy_manifest),
        "--train-deployable-candidates",
        str(train_deployable_candidates),
        "--policy-variant",
        str(policy_variant),
        "--rank-reference-run-id",
        str(rank_reference_run_id),
        "--market-mode",
        str(market_mode),
        "--exchange",
        str(exchange),
        "--stop-after",
        "bridge_to_t1_anchor_scored_candidates",
        "--skip-existing",
    ]
    score_cmd = py + [
        "scripts/score_market_state_controller_bundle.py",
        "--bundle",
        str(bundle),
        "--eval-candidates",
        str(paths.t1_anchor_candidates),
        "--eval-feature-store-dir",
        str(feature_store_dir),
        "--output-dir",
        str(paths.score_output_dir),
        "--policy-manifest",
        str(policy_manifest),
        "--policy-variant",
        str(policy_variant),
        "--train-deployable-candidates",
        str(train_deployable_candidates),
        "--market-mode",
        str(market_mode),
        "--window-start",
        common_start,
        "--window-end",
        common_end,
    ]
    discovery_cmd = py + [
        "scripts/discover_market_state_no_backfill_shadow_windows.py",
        "--config",
        str(config_path),
        "--search-root",
        str(data_root / "reports"),
        "--output-dir",
        str(paths.discovery_output_dir),
        "--include-regex",
        "globalrank.*no_backfill",
        "--min-timestamp-count",
        str(int(min_timestamp_count)),
    ]
    steps = [
        Step(
            "materialize_t1_anchor_candidates",
            t1_cmd,
            {"t1_anchor_candidates": str(paths.t1_anchor_candidates)},
        ),
        Step(
            "score_market_state_no_backfill_shadow_bundle",
            score_cmd,
            {"score_manifest": str(paths.score_output_dir / "manifest.json")},
        ),
        Step(
            "discover_appendable_no_backfill_shadow_windows",
            discovery_cmd,
            {
                "discovery_json": str(
                    paths.discovery_output_dir
                    / "market_state_no_backfill_shadow_window_discovery.json"
                )
            },
        ),
    ]
    if include_monitor_step:
        monitor_score_dirs = _existing_monitor_score_dirs(config)
        monitor_score_dirs.append(paths.score_output_dir)
        monitor_cmd = py + [
            "scripts/report_market_state_no_backfill_shadow_monitor.py",
            "--output-dir",
            str(paths.monitor_output_dir),
            "--expected-rank-contract",
            "anchor_global_policy_rank_reference",
            "--expected-selected-arm",
            "S1_observed_axes_shared_response__post_selection_overlay",
        ]
        for score_dir in monitor_score_dirs:
            monitor_cmd.extend(["--score-dir", str(score_dir)])
        steps.append(
            Step(
                "refresh_no_backfill_shadow_monitor",
                monitor_cmd,
                {
                    "monitor_summary": str(
                        paths.monitor_output_dir / "no_backfill_shadow_monitor_summary.json"
                    )
                },
            )
        )
    return steps


def build_runner_plan(
    *,
    config: dict[str, Any],
    config_path: Path,
    data_root: Path,
    feature_store_dir: Path,
    output_dir: Path,
    readiness_output_dir: Path,
    maturity_buffer_hours: int,
    target_window_hours: int,
    min_timestamp_count: int,
    min_feature_timestamp_coverage: float,
    allow_partial_window: bool,
    run_id: str,
    score_output_dir: Path | None,
    discovery_output_dir: Path | None,
    monitor_output_dir: Path | None,
    bundle: Path | None,
    policy_manifest: Path | None,
    train_deployable_candidates: Path | None,
    policy_variant: str,
    market_mode: str,
    exchange: str,
    rank_reference_run_id: str,
    policy_artifact_run_id: str,
    model_artifact_run_id: str,
    include_monitor_step: bool,
) -> dict[str, Any]:
    readiness = build_readiness(
        config=config,
        config_path=config_path,
        data_root=data_root,
        feature_store_dir=feature_store_dir,
        output_dir=readiness_output_dir,
        maturity_buffer_hours=int(maturity_buffer_hours),
        target_window_hours=int(target_window_hours),
        min_timestamp_count=int(min_timestamp_count),
        min_feature_timestamp_coverage=float(min_feature_timestamp_coverage),
    )
    start_raw = readiness.get("next_window_start")
    end_raw = (
        readiness.get("proposed_scoreable_window_end")
        if allow_partial_window
        else readiness.get("target_window_end")
    )
    status = "not_scoreable_yet"
    reason = readiness.get("next_action")
    steps: list[Step] = []
    paths: PlannedPaths | None = None
    start = _as_utc(start_raw) if start_raw else None
    end = _as_utc(end_raw) if end_raw else None

    if readiness.get("scoreable_min_window_now") and (
        allow_partial_window or readiness.get("scoreable_full_window_now")
    ):
        if start is None or end is None or end < start:
            status = "not_scoreable_yet"
            reason = "invalid_next_window_bounds"
        else:
            status = "scoreable_now"
            reason = (
                "partial_window_scoreable"
                if allow_partial_window and not readiness.get("scoreable_full_window_now")
                else "full_window_scoreable"
            )
            defaults = _latest_score_manifest_paths(config)
            resolved_bundle = bundle or defaults.get("bundle") or DEFAULT_BUNDLE
            resolved_policy_manifest = (
                policy_manifest
                or defaults.get("policy_manifest")
                or DEFAULT_T1_POLICY_MANIFEST
            )
            resolved_train = (
                train_deployable_candidates
                or defaults.get("train_deployable_candidates")
                or DEFAULT_TRAIN_DEPLOYABLE_CANDIDATES
            )
            paths = _default_paths(
                data_root=data_root,
                output_dir=output_dir,
                readiness_output_dir=readiness_output_dir,
                start=start,
                end=end,
                run_id=run_id,
                score_output_dir=score_output_dir,
                discovery_output_dir=discovery_output_dir,
                monitor_output_dir=monitor_output_dir,
                feature_store_dir=feature_store_dir,
            )
            steps = build_steps(
                config=config,
                config_path=config_path,
                data_root=data_root,
                feature_store_dir=feature_store_dir,
                paths=paths,
                start=start,
                end=end,
                bundle=Path(resolved_bundle),
                policy_manifest=Path(resolved_policy_manifest),
                train_deployable_candidates=Path(resolved_train),
                policy_variant=policy_variant,
                market_mode=market_mode,
                exchange=exchange,
                rank_reference_run_id=rank_reference_run_id,
                policy_artifact_run_id=policy_artifact_run_id,
                model_artifact_run_id=model_artifact_run_id,
                min_timestamp_count=min_timestamp_count,
                include_monitor_step=include_monitor_step,
            )
    elif readiness.get("scoreable_min_window_now") and not allow_partial_window:
        status = "full_window_not_scoreable_yet"
        reason = "full_window_required_but_only_partial_window_scoreable"

    path_payload = (
        _json_safe(asdict(paths))
        if paths is not None
        else {
            "runner_output_dir": str(output_dir),
            "readiness_output_dir": str(readiness_output_dir),
            "runner_manifest": str(
                output_dir / "next_no_backfill_shadow_runner_manifest.json"
            ),
        }
    )
    return {
        "generated_by": "run_next_no_backfill_shadow_window",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "reason": reason,
        "readiness": readiness,
        "window": {
            "start": start.isoformat() if start is not None else None,
            "end": end.isoformat() if end is not None else None,
            "allow_partial_window": bool(allow_partial_window),
        },
        "paths": path_payload,
        "steps": [
            {"name": step.name, "command": step.command, "outputs": step.outputs}
            for step in steps
        ],
        "completed_steps": [],
    }


def write_runner_manifest(plan: dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "next_no_backfill_shadow_runner_manifest.json"
    path.write_text(
        json.dumps(_json_safe(plan), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Next No-Backfill Shadow Runner",
        "",
        f"- Status: `{plan['status']}`",
        f"- Reason: `{plan['reason']}`",
        f"- Window start: `{plan['window']['start']}`",
        f"- Window end: `{plan['window']['end']}`",
        f"- Partial window allowed: `{plan['window']['allow_partial_window']}`",
        f"- Planned steps: `{len(plan['steps'])}`",
        "",
        "## Readiness",
        "",
        f"- Feature max: `{plan['readiness'].get('feature_timestamp_max')}`",
        f"- Mature timestamps available: `{plan['readiness'].get('mature_timestamp_count_available')}`",
        f"- Scoreable minimum window: `{plan['readiness'].get('scoreable_min_window_now')}`",
        f"- Scoreable full window: `{plan['readiness'].get('scoreable_full_window_now')}`",
        f"- Next action: `{plan['readiness'].get('next_action')}`",
        "",
        "## Steps",
        "",
    ]
    if plan["steps"]:
        for idx, step in enumerate(plan["steps"], start=1):
            lines.append(f"{idx}. `{step['name']}`")
    else:
        lines.append("_No execution steps planned._")
    (output_dir / "next_no_backfill_shadow_runner_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    return path


def _outputs_exist(step: Step) -> bool:
    return bool(step.outputs) and all(Path(path).exists() for path in step.outputs.values())


def run_steps(steps: list[Step], *, skip_existing: bool) -> list[dict[str, Any]]:
    completed: list[dict[str, Any]] = []
    for idx, step in enumerate(steps, start=1):
        if skip_existing and _outputs_exist(step):
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


def _update_runner_config(config: dict[str, Any], config_path: Path, plan: dict[str, Any]) -> None:
    controller = config.setdefault("market_state_controller_validation", {})
    paths = dict(plan.get("paths") or {})
    controller["global_rank_threshold_controller_no_backfill_next_window_runner"] = {
        "generated_by": plan["generated_by"],
        "generated_at_utc": plan["generated_at_utc"],
        "status": plan["status"],
        "reason": plan["reason"],
        "window": plan["window"],
        "runner_dir": str(paths.get("runner_output_dir") or ""),
        "runner_manifest": str(paths.get("runner_manifest") or ""),
        "readiness_dir": str(paths.get("readiness_output_dir") or ""),
        "score_output_dir": str(paths.get("score_output_dir") or ""),
        "discovery_output_dir": str(paths.get("discovery_output_dir") or ""),
        "monitor_output_dir": str(paths.get("monitor_output_dir") or ""),
        "planned_step_count": int(len(plan.get("steps") or [])),
        "completed_steps": plan.get("completed_steps") or [],
    }
    score_dir_raw = paths.get("score_output_dir")
    if score_dir_raw:
        score_dir = Path(str(score_dir_raw))
        score_manifest_path = score_dir / "manifest.json"
        if score_manifest_path.exists():
            score_manifest = _load_json(score_manifest_path)
            bundle_path = score_manifest.get("bundle")
            output_hashes = score_manifest.get("output_sha256")
            missing_output_hashes = []
            if isinstance(output_hashes, dict):
                missing_output_hashes = [
                    key for key, value in output_hashes.items() if not str(value or "").strip()
                ]
            controller["global_rank_threshold_controller_no_backfill_shadow_score_latest"] = {
                "generated_by": score_manifest.get("generated_by"),
                "generated_at_utc": score_manifest.get("generated_at_utc"),
                "score_dir": str(score_dir),
                "manifest": str(score_manifest_path),
                "bundle": str(bundle_path or ""),
                "bundle_dir": str(Path(str(bundle_path)).parent) if bundle_path else "",
                "bundle_sha256": score_manifest.get("bundle_sha256"),
                "policy_manifest": score_manifest.get("policy_manifest"),
                "policy_manifest_sha256": score_manifest.get("policy_manifest_sha256"),
                "eval_candidates": score_manifest.get("eval_candidates"),
                "eval_candidates_sha256": score_manifest.get("eval_candidates_sha256"),
                "train_deployable_candidates": score_manifest.get("train_deployable_candidates"),
                "train_deployable_candidates_sha256": score_manifest.get(
                    "train_deployable_candidates_sha256"
                ),
                "score_manifest_contract_version": score_manifest.get(
                    "score_manifest_contract_version"
                ),
                "score_manifest_artifact_hashes_complete": not missing_output_hashes,
                "selected_arm": score_manifest.get("selected_arm"),
                "rank_contract": score_manifest.get("rank_contract"),
                "rank_reference_run_id": score_manifest.get("rank_reference_run_id"),
                "active_heads": score_manifest.get("active_heads"),
                "disabled_heads": score_manifest.get("disabled_heads"),
                "controller_execution_enabled": bool(
                    score_manifest.get("controller_execution_enabled")
                ),
                "shadow_controller_only": bool(score_manifest.get("shadow_controller_only")),
                "shadow_no_backfill_replay_available": bool(
                    score_manifest.get("shadow_no_backfill_replay_available")
                ),
                "shadow_direct_threshold_only_available": bool(
                    score_manifest.get("shadow_direct_threshold_only_available")
                ),
                "shadow_locked_accepted_overlay_available": bool(
                    score_manifest.get("shadow_locked_accepted_overlay_available")
                ),
                "source_contract_overall_passed": bool(
                    (score_manifest.get("source_contract_audit") or {}).get("overall_passed")
                ),
                "missing_output_hash_keys": missing_output_hashes,
            }
    monitor_dir_raw = paths.get("monitor_output_dir")
    if monitor_dir_raw:
        monitor_dir = Path(str(monitor_dir_raw))
        monitor_summary_path = monitor_dir / "no_backfill_shadow_monitor_summary.json"
        if monitor_summary_path.exists():
            summary = _load_json(monitor_summary_path)
            controller["global_rank_threshold_controller_no_backfill_shadow_monitor"] = {
                "generated_by": summary.get("generated_by"),
                "generated_at_utc": summary.get("generated_at_utc"),
                "status": summary.get("status"),
                "monitor_dir": str(monitor_dir),
                "summary_json": str(monitor_summary_path),
                "report": str(monitor_dir / "no_backfill_shadow_monitor_report.md"),
                "window_metrics_csv": summary.get("window_metrics_csv"),
                "by_head_csv": summary.get("by_head_csv"),
                "window_count": summary.get("window_count"),
                "ignored_empty_eval_window_count": summary.get(
                    "ignored_empty_eval_window_count"
                ),
                "promotion_gate_passed": bool(summary.get("promotion_gate_passed")),
                "promotion_gate_failures": summary.get("promotion_gate_failures") or [],
                "direct_threshold_only_promotion_gate_passed": bool(
                    summary.get("direct_threshold_only_promotion_gate_passed")
                ),
                "direct_threshold_only_promotion_gate_failures": summary.get(
                    "direct_threshold_only_promotion_gate_failures"
                )
                or [],
                "locked_accepted_overlay_promotion_gate_passed": bool(
                    summary.get("locked_accepted_overlay_promotion_gate_passed")
                ),
                "locked_accepted_overlay_promotion_gate_failures": summary.get(
                    "locked_accepted_overlay_promotion_gate_failures"
                )
                or [],
                "controller_should_remain_disabled": not bool(
                    summary.get("promotion_gate_passed")
                ),
                "interpretation": summary.get("interpretation"),
                "positive_delta_window_share": summary.get("positive_delta_window_share"),
                "median_total_net_pnl_delta": summary.get("median_total_net_pnl_delta"),
                "q25_total_net_pnl_delta": summary.get("q25_total_net_pnl_delta"),
                "sum_total_net_pnl_delta": summary.get("sum_total_net_pnl_delta"),
                "sum_direct_threshold_only_net_pnl_delta": summary.get(
                    "sum_direct_threshold_only_net_pnl_delta"
                ),
                "min_eval_feature_store_timestamp_coverage": summary.get(
                    "min_eval_feature_store_timestamp_coverage"
                ),
                "min_eval_source_feature_count": summary.get("min_eval_source_feature_count"),
                "all_score_manifest_artifact_hashes_complete": bool(
                    summary.get("all_score_manifest_artifact_hashes_complete")
                ),
                "all_source_contracts_passed": bool(summary.get("all_source_contracts_passed")),
                "rank_contracts": summary.get("rank_contracts") or [],
                "selected_arms": summary.get("selected_arms") or [],
            }
    config_path.write_text(
        json.dumps(_json_safe(config), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--feature-store-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--readiness-output-dir", type=Path, default=None)
    parser.add_argument("--maturity-buffer-hours", type=int, default=16)
    parser.add_argument("--target-window-hours", type=int, default=24)
    parser.add_argument("--min-timestamp-count", type=int, default=3)
    parser.add_argument("--min-feature-timestamp-coverage", type=float, default=0.95)
    parser.add_argument("--allow-partial-window", action="store_true")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--score-output-dir", type=Path, default=None)
    parser.add_argument("--discovery-output-dir", type=Path, default=None)
    parser.add_argument("--monitor-output-dir", type=Path, default=None)
    parser.add_argument("--bundle", type=Path, default=None)
    parser.add_argument("--policy-manifest", type=Path, default=None)
    parser.add_argument("--train-deployable-candidates", type=Path, default=None)
    parser.add_argument("--policy-variant", default="refit_bar4_strategy_bar2")
    parser.add_argument("--market-mode", default="perps")
    parser.add_argument("--exchange", default="krakenfutures")
    parser.add_argument("--rank-reference-run-id", default=DEFAULT_RANK_REFERENCE_RUN_ID)
    parser.add_argument("--policy-artifact-run-id", default=DEFAULT_POLICY_ARTIFACT_RUN_ID)
    parser.add_argument("--model-artifact-run-id", default=DEFAULT_MODEL_ARTIFACT_RUN_ID)
    parser.add_argument("--no-monitor-step", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--update-config", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _load_json(args.config)
    feature_store_dir = args.feature_store_dir or _latest_feature_store_dir(args.data_root)
    if feature_store_dir is None or not Path(feature_store_dir).exists():
        raise SystemExit("No feature-store directory found; pass --feature-store-dir.")
    readiness_output_dir = args.readiness_output_dir or (args.output_dir / "readiness")
    plan = build_runner_plan(
        config=config,
        config_path=args.config,
        data_root=args.data_root,
        feature_store_dir=Path(feature_store_dir),
        output_dir=args.output_dir,
        readiness_output_dir=readiness_output_dir,
        maturity_buffer_hours=int(args.maturity_buffer_hours),
        target_window_hours=int(args.target_window_hours),
        min_timestamp_count=int(args.min_timestamp_count),
        min_feature_timestamp_coverage=float(args.min_feature_timestamp_coverage),
        allow_partial_window=bool(args.allow_partial_window),
        run_id=str(args.run_id or ""),
        score_output_dir=args.score_output_dir,
        discovery_output_dir=args.discovery_output_dir,
        monitor_output_dir=args.monitor_output_dir,
        bundle=args.bundle,
        policy_manifest=args.policy_manifest,
        train_deployable_candidates=args.train_deployable_candidates,
        policy_variant=str(args.policy_variant),
        market_mode=str(args.market_mode),
        exchange=str(args.exchange),
        rank_reference_run_id=str(args.rank_reference_run_id),
        policy_artifact_run_id=str(args.policy_artifact_run_id),
        model_artifact_run_id=str(args.model_artifact_run_id),
        include_monitor_step=not bool(args.no_monitor_step),
    )
    write_readiness(plan["readiness"], readiness_output_dir)
    if bool(args.update_config):
        update_readiness_config(config, args.config, plan["readiness"])
    steps = [
        Step(
            name=str(step["name"]),
            command=[str(part) for part in step["command"]],
            outputs={str(k): str(v) for k, v in dict(step["outputs"]).items()},
        )
        for step in plan["steps"]
    ]
    manifest_path = write_runner_manifest(plan, args.output_dir)
    if plan["status"] != "scoreable_now":
        print(json.dumps(_json_safe({k: plan[k] for k in ("status", "reason", "window")}), indent=2))
        print(f"Wrote runner manifest: {manifest_path}")
        if bool(args.update_config):
            _update_runner_config(config, args.config, plan)
        return
    if bool(args.dry_run):
        print(f"Wrote dry-run runner manifest: {manifest_path}")
        for idx, step in enumerate(steps, start=1):
            print(f"[{idx}/{len(steps)}] {step.name}: {' '.join(step.command)}")
        if bool(args.update_config):
            _update_runner_config(config, args.config, plan)
        return
    completed = run_steps(steps, skip_existing=bool(args.skip_existing))
    plan["completed_steps"] = completed
    write_runner_manifest(plan, args.output_dir)
    if bool(args.update_config):
        _update_runner_config(config, args.config, plan)
    print(f"Wrote runner manifest: {manifest_path}")


if __name__ == "__main__":
    main()
