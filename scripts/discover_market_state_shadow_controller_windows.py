#!/usr/bin/env python3
"""Discover scored shadow threshold-controller bundles ready for monitoring.

This script is an operational helper for Stage 2.  It scans report roots for
`score_market_state_controller_bundle` outputs, verifies that they are still
shadow-only threshold-controller bundles under the active T1 contract, and
classifies each bundle as appendable, already monitored, or failed.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import report_market_state_shadow_controller_monitor as monitor  # noqa: E402


DEFAULT_OUTPUT_DIR = Path(
    "data_perp/reports/market_state_shadow_controller_window_discovery"
)
EXPECTED_RANK_CONTRACT = "short_boll_timestamp_rank"
EXPECTED_ACTIVE_HEADS = ["short_asset", "short_boll"]
EXPECTED_DISABLED_HEADS = ["long_bars", "long_dist"]
REQUIRED_FILES = (
    "manifest.json",
    "market_state_feature_contract.json",
    "market_state_timestamp_panel.parquet",
    "strategy_threshold_schedule.parquet",
    "strategy_threshold_action_audit.csv",
    "shadow_controller_proposed_schedule.parquet",
    "shadow_threshold_action_audit.csv",
    "shadow_threshold_candidate_suppression_utility.csv",
    "controller_replay_summary.csv",
    "controller_replay_by_head.csv",
    "accepted_trades.parquet",
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
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


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    view = frame.fillna("").astype(str)
    columns = list(view.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(row[col] for col in columns) + " |")
    return "\n".join(lines)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _normalised_path(path: Path) -> str:
    try:
        return str(path.resolve())
    except Exception:
        return str(path)


def existing_monitored_bundle_dirs(monitor_dir: Path | None) -> set[str]:
    if monitor_dir is None:
        return set()
    path = Path(monitor_dir) / "shadow_controller_monitor_bundles.csv"
    if not path.exists():
        return set()
    frame = pd.read_csv(path)
    if "bundle_dir" not in frame.columns:
        return set()
    return {_normalised_path(Path(value)) for value in frame["bundle_dir"].dropna().astype(str)}


def discover_bundle_dirs(
    roots: list[Path],
    *,
    include_regex: str | None = None,
    exclude_regex: str | None = None,
) -> list[Path]:
    include = re.compile(include_regex) if include_regex else None
    exclude = re.compile(exclude_regex) if exclude_regex else None
    found: dict[str, Path] = {}
    for root in roots:
        if not root.exists():
            continue
        for manifest in root.rglob("manifest.json"):
            bundle_dir = manifest.parent
            text = str(bundle_dir)
            if include and not include.search(text):
                continue
            if exclude and exclude.search(text):
                continue
            payload = _load_json(manifest)
            if payload.get("generated_by") != "score_market_state_controller_bundle":
                continue
            found[_normalised_path(bundle_dir)] = bundle_dir
    return [found[key] for key in sorted(found)]


def _timestamp_count(bundle_dir: Path) -> int:
    path = bundle_dir / "market_state_timestamp_panel.parquet"
    if not path.exists():
        return 0
    frame = pd.read_parquet(path, columns=["timestamp"])
    timestamps = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    return int(timestamps.nunique())


def _safe_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value]
    if value in (None, ""):
        return []
    return [str(value)]


def _contract_failures(bundle_dir: Path, manifest: dict[str, Any], feature_contract: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    missing_files = [name for name in REQUIRED_FILES if not (bundle_dir / name).exists()]
    if missing_files:
        failures.append(f"missing_required_files:{','.join(missing_files)}")
    if manifest.get("generated_by") != "score_market_state_controller_bundle":
        failures.append("not_scored_controller_bundle")
    controller = dict(manifest.get("controller") or {})
    if bool(manifest.get("controller_execution_enabled", controller.get("controller_execution_enabled", True))):
        failures.append("controller_execution_enabled")
    if not bool(manifest.get("shadow_controller_only", controller.get("shadow_controller_only", False))):
        failures.append("not_shadow_controller_only")
    if bool(controller.get("changes_scores_or_ranks", False)):
        failures.append("controller_changes_scores_or_ranks")
    if bool(controller.get("changes_auction_ordering", False)):
        failures.append("controller_changes_auction_ordering")
    if feature_contract.get("rank_contract") != EXPECTED_RANK_CONTRACT:
        failures.append("rank_contract_mismatch")
    if _safe_list(feature_contract.get("active_heads")) != EXPECTED_ACTIVE_HEADS:
        failures.append("active_heads_mismatch")
    if _safe_list(feature_contract.get("disabled_heads")) != EXPECTED_DISABLED_HEADS:
        failures.append("disabled_heads_mismatch")
    invariants = dict(feature_contract.get("invariants") or {})
    for key in (
        "one_market_state_row_per_timestamp",
        "state_join_timestamp_constant",
    ):
        if invariants.get(key) is not True:
            failures.append(f"invariant_not_true:{key}")
    for key in (
        "market_state_uses_strategy_ids",
        "market_state_uses_model_predictions",
        "market_state_uses_ranks",
        "market_state_uses_candidate_counts",
        "market_state_uses_portfolio_pnl",
        "market_state_uses_realized_strategy_outcomes",
        "actual_order_book_features_allowed",
        "controller_changes_scores_or_ranks",
        "controller_changes_auction_ordering",
        "controller_can_lower_thresholds",
    ):
        if invariants.get(key) is not False:
            failures.append(f"invariant_not_false:{key}")
    return failures


def readiness_for_bundle(
    bundle_dir: Path,
    *,
    existing_dirs: set[str],
    min_timestamp_count: int,
    min_start_after: pd.Timestamp | None = None,
    run_artifact_audit: bool = False,
) -> dict[str, Any]:
    bundle_dir = Path(bundle_dir)
    manifest = _load_json(bundle_dir / "manifest.json")
    feature_contract = _load_json(bundle_dir / "market_state_feature_contract.json")
    failures = _contract_failures(bundle_dir, manifest, feature_contract)
    timestamp_count = _timestamp_count(bundle_dir)
    if timestamp_count < int(min_timestamp_count):
        failures.append("insufficient_timestamp_count")
    already_monitored = _normalised_path(bundle_dir) in existing_dirs
    summary: dict[str, Any] = {}
    if not failures:
        try:
            result = monitor.summarize_shadow_bundle(
                bundle_dir,
                run_artifact_audit=bool(run_artifact_audit),
            )
            summary = dict(result["summary"])
            if not bool(summary.get("applied_noop_pass")):
                failures.append("applied_noop_parity_failed")
            if not bool(summary.get("shadow_schedule_safe")):
                failures.append("shadow_schedule_not_safe")
            if not bool(summary.get("coverage_ok")):
                failures.append("coverage_not_ok")
            if run_artifact_audit and not bool(summary.get("artifact_audit_passed")):
                failures.append("artifact_audit_failed")
        except Exception as exc:
            failures.append(f"summarizer_failed:{type(exc).__name__}")
    if min_start_after is not None and summary:
        start = pd.to_datetime(summary.get("start_timestamp"), utc=True, errors="coerce")
        if pd.isna(start):
            failures.append("missing_start_timestamp")
        elif pd.Timestamp(start) < pd.Timestamp(min_start_after):
            failures.append("before_min_start_after")
    status = "failed"
    if already_monitored:
        status = "already_monitored"
    elif failures == ["before_min_start_after"]:
        status = "excluded"
    elif not failures:
        status = "appendable"
    return {
        "bundle_dir": str(bundle_dir),
        "status": status,
        "failures": failures,
        "already_monitored": bool(already_monitored),
        "generated_by": manifest.get("generated_by"),
        "selected_arm": manifest.get("selected_arm"),
        "rank_contract": feature_contract.get("rank_contract"),
        "active_heads": _safe_list(feature_contract.get("active_heads")),
        "disabled_heads": _safe_list(feature_contract.get("disabled_heads")),
        "timestamp_count": int(timestamp_count),
        "start_timestamp": summary.get("start_timestamp"),
        "end_timestamp": summary.get("end_timestamp"),
        "shadow_suppressed_candidates": summary.get("shadow_suppressed_candidates"),
        "shadow_realized_defensive_success": summary.get("shadow_realized_defensive_success"),
        "defensive_positive": summary.get("defensive_positive"),
        "applied_noop_pass": summary.get("applied_noop_pass"),
        "shadow_schedule_safe": summary.get("shadow_schedule_safe"),
        "coverage_ok": summary.get("coverage_ok"),
    }


def discover_shadow_controller_windows(
    *,
    roots: list[Path],
    output_dir: Path,
    existing_monitor_dir: Path | None = None,
    include_regex: str | None = None,
    exclude_regex: str | None = None,
    min_timestamp_count: int = 3,
    min_start_after: str | None = None,
    max_candidates: int = 0,
    run_artifact_audit: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = discover_bundle_dirs(
        roots,
        include_regex=include_regex,
        exclude_regex=exclude_regex,
    )
    if max_candidates and max_candidates > 0:
        candidates = candidates[: int(max_candidates)]
    existing_dirs = existing_monitored_bundle_dirs(existing_monitor_dir)
    min_start_ts = (
        pd.Timestamp(pd.to_datetime(min_start_after, utc=True, errors="raise"))
        if min_start_after
        else None
    )
    rows = [
        readiness_for_bundle(
            path,
            existing_dirs=existing_dirs,
            min_timestamp_count=int(min_timestamp_count),
            min_start_after=min_start_ts,
            run_artifact_audit=bool(run_artifact_audit),
        )
        for path in candidates
    ]
    frame = pd.DataFrame(rows)
    if frame.empty:
        frame = pd.DataFrame(
            columns=[
                "bundle_dir",
                "status",
                "failures",
                "timestamp_count",
                "start_timestamp",
                "end_timestamp",
            ]
        )
    appendable = frame.loc[frame["status"].astype(str).eq("appendable")].copy()
    readiness_csv = output_dir / "shadow_controller_window_readiness.csv"
    appendable_csv = output_dir / "appendable_shadow_controller_windows.csv"
    frame.to_csv(readiness_csv, index=False)
    appendable.to_csv(appendable_csv, index=False)
    latest_end = None
    if "end_timestamp" in frame.columns:
        ends = pd.to_datetime(frame["end_timestamp"], utc=True, errors="coerce").dropna()
        latest_end = ends.max().isoformat() if not ends.empty else None
    status_counts = {
        str(k): int(v)
        for k, v in frame["status"].value_counts(dropna=False).sort_index().items()
    }
    summary = {
        "generated_by": "discover_market_state_shadow_controller_windows",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "search_roots": [str(root) for root in roots],
        "existing_monitor_dir": str(existing_monitor_dir) if existing_monitor_dir is not None else None,
        "include_regex": include_regex,
        "exclude_regex": exclude_regex,
        "min_timestamp_count": int(min_timestamp_count),
        "min_start_after": min_start_after,
        "max_candidates": int(max_candidates),
        "run_artifact_audit": bool(run_artifact_audit),
        "discovered_candidate_count": int(len(candidates)),
        "appendable_candidate_count": int(len(appendable)),
        "already_monitored_count": int((frame["status"].astype(str) == "already_monitored").sum()),
        "excluded_candidate_count": int((frame["status"].astype(str) == "excluded").sum()),
        "failed_candidate_count": int((frame["status"].astype(str) == "failed").sum()),
        "status_counts": status_counts,
        "latest_discovered_window_end": latest_end,
        "readiness_csv": str(readiness_csv),
        "appendable_csv": str(appendable_csv),
    }
    (output_dir / "market_state_shadow_controller_window_discovery.json").write_text(
        json.dumps(_json_safe(summary), indent=2) + "\n",
        encoding="utf-8",
    )
    _write_report(summary, frame, appendable, output_dir)
    return {**summary, "readiness": frame, "appendable": appendable}


def _write_report(
    summary: dict[str, Any],
    frame: pd.DataFrame,
    appendable: pd.DataFrame,
    output_dir: Path,
) -> None:
    lines = [
        "# Market-State Shadow Controller Window Discovery",
        "",
        "This report discovers scored threshold-controller shadow bundles that can be appended to the Stage-2 monitor.",
        "",
        "## Summary",
        "",
        f"- Discovered candidates: `{summary['discovered_candidate_count']}`",
        f"- Appendable candidates: `{summary['appendable_candidate_count']}`",
        f"- Already monitored: `{summary['already_monitored_count']}`",
        f"- Excluded candidates: `{summary['excluded_candidate_count']}`",
        f"- Failed candidates: `{summary['failed_candidate_count']}`",
        f"- Latest discovered end: `{summary['latest_discovered_window_end']}`",
        "",
        "## Appendable Windows",
        "",
        _markdown_table(appendable) if not appendable.empty else "_No appendable windows found._",
        "",
        "## Readiness",
        "",
        _markdown_table(frame) if not frame.empty else "_No scored bundles discovered._",
        "",
        "## Interpretation",
        "",
        (
            "Appendable windows are shadow-only scored bundles under the active T1 "
            "timestamp-rank contract with disabled execution, safe proposed "
            "threshold raises, complete suppression utility, and no applied schedule changes."
        ),
        "",
    ]
    (output_dir / "market_state_shadow_controller_window_discovery_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-root", action="append", type=Path, default=None)
    parser.add_argument("--existing-monitor-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--include-regex", default=r"market_state_controller_bundle_score")
    parser.add_argument("--exclude-regex", default=None)
    parser.add_argument("--min-timestamp-count", type=int, default=3)
    parser.add_argument(
        "--min-start-after",
        default=None,
        help="Optional UTC timestamp; bundles starting before this are classified as excluded.",
    )
    parser.add_argument("--max-candidates", type=int, default=0)
    parser.add_argument("--run-artifact-audit", action="store_true")
    parser.add_argument("--fail-if-none", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    roots = list(args.search_root or [Path("data_perp/reports")])
    summary = discover_shadow_controller_windows(
        roots=roots,
        output_dir=args.output_dir,
        existing_monitor_dir=args.existing_monitor_dir,
        include_regex=args.include_regex,
        exclude_regex=args.exclude_regex,
        min_timestamp_count=int(args.min_timestamp_count),
        min_start_after=args.min_start_after,
        max_candidates=int(args.max_candidates),
        run_artifact_audit=bool(args.run_artifact_audit),
    )
    print(json.dumps(_json_safe({k: v for k, v in summary.items() if k not in {"readiness", "appendable"}}), indent=2))
    if bool(args.fail_if_none) and int(summary["appendable_candidate_count"]) <= 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
