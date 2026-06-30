#!/usr/bin/env python3
"""Discover global-rank no-backfill shadow-score windows ready for monitoring.

This is the no-backfill/global-over-time counterpart to the older timestamp
rank shadow-controller discovery helper. It scans scored
``score_market_state_controller_bundle`` output directories, verifies that they
match the active T1 global rank contract, and classifies each window as:

* ``appendable``: valid, not already monitored, and after any requested cutoff;
* ``already_monitored``: valid but already present in the monitor;
* ``excluded``: valid but intentionally before the cutoff;
* ``failed``: missing files or contract/hash/parity failures.

The script is read-only and never promotes a controller.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts import report_market_state_no_backfill_shadow_monitor as monitor


DEFAULT_CONFIG = Path("config/reliability_blend_production_stack.json")
DEFAULT_OUTPUT_DIR = Path(
    "data_perp/reports/market_state_no_backfill_shadow_window_discovery"
)
DEFAULT_EXPECTED_RANK_CONTRACT = "anchor_global_policy_rank_reference"
DEFAULT_EXPECTED_SELECTED_ARM = "S1_observed_axes_shared_response__post_selection_overlay"
DEFAULT_EXPECTED_ACTIVE_HEADS = ["short_asset", "short_boll"]
DEFAULT_EXPECTED_DISABLED_HEADS = ["long_bars", "long_dist"]

REQUIRED_FILES = (
    "manifest.json",
    "market_state_feature_contract.json",
    "market_state_timestamp_panel.parquet",
    "market_state_feature_coverage.csv",
    "strategy_threshold_schedule.parquet",
    "strategy_threshold_action_audit.csv",
    "strategy_threshold_controller_config.json",
    "controller_replay_summary.csv",
    "controller_replay_by_head.csv",
    "accepted_trades.parquet",
    "shadow_no_backfill_scored_candidates.parquet",
    "shadow_no_backfill_decisions.parquet",
    "shadow_no_backfill_accepted_trades.parquet",
    "shadow_no_backfill_replay_summary.csv",
    "shadow_no_backfill_replay_by_head.csv",
    "shadow_no_backfill_accepted_trade_delta.csv",
    "shadow_direct_threshold_only_summary.csv",
    "shadow_direct_threshold_only_accepted_trade_delta.csv",
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


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


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


def _normalised_path(path: Path) -> str:
    try:
        return str(path.resolve())
    except Exception:
        return str(path)


def _safe_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value]
    if value in (None, ""):
        return []
    return [str(value)]


def _valid_sha256(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def monitored_score_dirs_from_config(config: dict[str, Any]) -> set[str]:
    controller = dict(config.get("market_state_controller_validation") or {})
    monitor_payload = dict(
        controller.get("global_rank_threshold_controller_no_backfill_shadow_monitor") or {}
    )
    out: set[str] = set()
    for window in monitor_payload.get("windows") or []:
        if isinstance(window, dict) and window.get("score_dir"):
            out.add(_normalised_path(Path(str(window["score_dir"]))))
    latest = dict(
        controller.get("global_rank_threshold_controller_no_backfill_shadow_score_latest")
        or {}
    )
    if latest.get("score_dir"):
        out.add(_normalised_path(Path(str(latest["score_dir"]))))
    return out


def monitored_score_dirs_from_dir(monitor_dir: Path | None) -> set[str]:
    if monitor_dir is None:
        return set()
    path = monitor_dir / "no_backfill_shadow_window_metrics.csv"
    if not path.exists():
        return set()
    frame = pd.read_csv(path)
    if "score_dir" not in frame.columns:
        return set()
    return {_normalised_path(Path(value)) for value in frame["score_dir"].dropna().astype(str)}


def discover_score_dirs(
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
            score_dir = manifest.parent
            text = str(score_dir)
            if include and not include.search(text):
                continue
            if exclude and exclude.search(text):
                continue
            payload = _load_json(manifest)
            if payload.get("generated_by") != "score_market_state_controller_bundle":
                continue
            if payload.get("shadow_no_backfill_replay_available") is not True:
                continue
            found[_normalised_path(score_dir)] = score_dir
    return [found[key] for key in sorted(found)]


def _score_period(score_dir: Path, manifest: dict[str, Any]) -> tuple[str | None, str | None]:
    eval_candidates = manifest.get("eval_candidates")
    if not eval_candidates:
        return None, None
    path = Path(str(eval_candidates))
    if not path.exists():
        return None, None
    timestamps = pd.to_datetime(
        pd.read_parquet(path, columns=["timestamp"])["timestamp"],
        utc=True,
        errors="coerce",
    ).dropna()
    if timestamps.empty:
        return None, None
    return timestamps.min().isoformat(), timestamps.max().isoformat()


def _hash_failures(manifest: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    for field in monitor.REQUIRED_SCORE_INPUT_HASH_FIELDS:
        if not _valid_sha256(manifest.get(field)):
            failures.append(f"missing_or_invalid_input_hash:{field}")
    hashes = manifest.get("output_sha256")
    if not isinstance(hashes, dict):
        failures.append("missing_output_sha256")
        return failures
    for key in monitor.REQUIRED_SCORE_OUTPUT_HASH_KEYS:
        if not _valid_sha256(hashes.get(key)):
            failures.append(f"missing_or_invalid_output_hash:{key}")
    return failures


def readiness_for_score_dir(
    score_dir: Path,
    *,
    already_monitored: set[str],
    expected_rank_contract: str,
    expected_selected_arm: str,
    expected_active_heads: list[str],
    expected_disabled_heads: list[str],
    min_timestamp_count: int,
    min_start_after: pd.Timestamp | None = None,
    require_complete_hashes: bool = True,
) -> dict[str, Any]:
    score_dir = Path(score_dir)
    manifest = _load_json(score_dir / "manifest.json")
    failures: list[str] = []
    missing_files = [name for name in REQUIRED_FILES if not (score_dir / name).exists()]
    if missing_files:
        failures.append("missing_required_files:" + ",".join(missing_files))
    if manifest.get("generated_by") != "score_market_state_controller_bundle":
        failures.append("not_scored_controller_bundle")
    if manifest.get("score_manifest_contract_version") != "market_state_controller_score_manifest_v2":
        failures.append("score_manifest_contract_mismatch")
    if manifest.get("shadow_no_backfill_replay_available") is not True:
        failures.append("missing_shadow_no_backfill_replay")
    if manifest.get("rank_contract") != expected_rank_contract:
        failures.append("rank_contract_mismatch")
    if manifest.get("selected_arm") != expected_selected_arm:
        failures.append("selected_arm_mismatch")
    if _safe_list(manifest.get("active_heads")) != expected_active_heads:
        failures.append("active_heads_mismatch")
    if _safe_list(manifest.get("disabled_heads")) != expected_disabled_heads:
        failures.append("disabled_heads_mismatch")
    if bool(manifest.get("controller_execution_enabled", True)):
        failures.append("controller_execution_enabled")
    if bool(manifest.get("shadow_controller_only")) is not True:
        failures.append("not_shadow_controller_only")
    controller = dict(manifest.get("controller") or {})
    if bool(controller.get("changes_scores_or_ranks", True)):
        failures.append("controller_changes_scores_or_ranks")
    if bool(controller.get("changes_auction_ordering", True)):
        failures.append("controller_changes_auction_ordering")
    if require_complete_hashes:
        failures.extend(_hash_failures(manifest))

    start, end = _score_period(score_dir, manifest)
    timestamp_count = 0
    try:
        if start and end:
            eval_candidates = Path(str(manifest.get("eval_candidates")))
            timestamps = pd.to_datetime(
                pd.read_parquet(eval_candidates, columns=["timestamp"])["timestamp"],
                utc=True,
                errors="coerce",
            ).dropna()
            timestamp_count = int(timestamps.nunique())
    except Exception as exc:
        failures.append(f"timestamp_read_failed:{type(exc).__name__}")
    if timestamp_count < int(min_timestamp_count):
        failures.append("insufficient_timestamp_count")
    if min_start_after is not None and start is not None:
        start_ts = pd.Timestamp(pd.to_datetime(start, utc=True))
        if start_ts < pd.Timestamp(min_start_after):
            failures.append("before_min_start_after")

    summary: dict[str, Any] = {}
    if not failures or failures == ["before_min_start_after"]:
        try:
            summary = monitor._window_row(score_dir)
        except Exception as exc:
            failures.append(f"window_summary_failed:{type(exc).__name__}")

    normalized = _normalised_path(score_dir)
    already = normalized in already_monitored
    status = "failed"
    non_cutoff_failures = [f for f in failures if f != "before_min_start_after"]
    if already and not non_cutoff_failures:
        status = "already_monitored"
    elif not non_cutoff_failures and "before_min_start_after" in failures:
        status = "excluded"
    elif not failures:
        status = "appendable"
    return {
        "score_dir": str(score_dir),
        "status": status,
        "failures": failures,
        "already_monitored": bool(already),
        "selected_arm": manifest.get("selected_arm"),
        "rank_contract": manifest.get("rank_contract"),
        "active_heads": _safe_list(manifest.get("active_heads")),
        "disabled_heads": _safe_list(manifest.get("disabled_heads")),
        "timestamp_count": int(timestamp_count),
        "period_start": start,
        "period_end": end,
        "baseline_trade_count": summary.get("baseline_trade_count"),
        "shadow_trade_count": summary.get("shadow_trade_count"),
        "total_net_pnl_delta": summary.get("total_net_pnl_delta"),
        "direct_threshold_only_total_net_pnl_delta": summary.get(
            "direct_threshold_only_total_net_pnl_delta"
        ),
        "score_manifest_artifact_hashes_complete": summary.get(
            "score_manifest_artifact_hashes_complete"
        ),
    }


def discover_no_backfill_shadow_windows(
    *,
    roots: list[Path],
    output_dir: Path,
    config: Path | None = DEFAULT_CONFIG,
    existing_monitor_dir: Path | None = None,
    include_regex: str | None = None,
    exclude_regex: str | None = None,
    expected_rank_contract: str = DEFAULT_EXPECTED_RANK_CONTRACT,
    expected_selected_arm: str = DEFAULT_EXPECTED_SELECTED_ARM,
    expected_active_heads: list[str] | None = None,
    expected_disabled_heads: list[str] | None = None,
    min_timestamp_count: int = 3,
    min_start_after: str | None = None,
    max_candidates: int = 0,
    require_complete_hashes: bool = True,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    config_payload = _load_json(config) if config is not None else {}
    already = monitored_score_dirs_from_config(config_payload)
    already.update(monitored_score_dirs_from_dir(existing_monitor_dir))
    candidates = discover_score_dirs(
        roots,
        include_regex=include_regex,
        exclude_regex=exclude_regex,
    )
    if max_candidates and max_candidates > 0:
        candidates = candidates[: int(max_candidates)]
    min_start_ts = (
        pd.Timestamp(pd.to_datetime(min_start_after, utc=True, errors="raise"))
        if min_start_after
        else None
    )
    rows = [
        readiness_for_score_dir(
            path,
            already_monitored=already,
            expected_rank_contract=expected_rank_contract,
            expected_selected_arm=expected_selected_arm,
            expected_active_heads=expected_active_heads or DEFAULT_EXPECTED_ACTIVE_HEADS,
            expected_disabled_heads=expected_disabled_heads or DEFAULT_EXPECTED_DISABLED_HEADS,
            min_timestamp_count=min_timestamp_count,
            min_start_after=min_start_ts,
            require_complete_hashes=require_complete_hashes,
        )
        for path in candidates
    ]
    frame = pd.DataFrame(rows)
    if frame.empty:
        frame = pd.DataFrame(
            columns=[
                "score_dir",
                "status",
                "failures",
                "timestamp_count",
                "period_start",
                "period_end",
            ]
        )
    appendable = frame.loc[frame["status"].astype(str).eq("appendable")].copy()
    readiness_csv = output_dir / "no_backfill_shadow_window_readiness.csv"
    appendable_csv = output_dir / "appendable_no_backfill_shadow_windows.csv"
    frame.to_csv(readiness_csv, index=False)
    appendable.to_csv(appendable_csv, index=False)
    latest_end = None
    if "period_end" in frame.columns:
        ends = pd.to_datetime(frame["period_end"], utc=True, errors="coerce").dropna()
        latest_end = ends.max().isoformat() if not ends.empty else None
    status_counts = {
        str(k): int(v)
        for k, v in frame["status"].value_counts(dropna=False).sort_index().items()
    }
    summary = {
        "generated_by": "discover_market_state_no_backfill_shadow_windows",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "search_roots": [str(root) for root in roots],
        "config": str(config) if config is not None else None,
        "existing_monitor_dir": str(existing_monitor_dir) if existing_monitor_dir else None,
        "include_regex": include_regex,
        "exclude_regex": exclude_regex,
        "expected_rank_contract": expected_rank_contract,
        "expected_selected_arm": expected_selected_arm,
        "expected_active_heads": expected_active_heads or DEFAULT_EXPECTED_ACTIVE_HEADS,
        "expected_disabled_heads": expected_disabled_heads or DEFAULT_EXPECTED_DISABLED_HEADS,
        "min_timestamp_count": int(min_timestamp_count),
        "min_start_after": min_start_after,
        "max_candidates": int(max_candidates),
        "require_complete_hashes": bool(require_complete_hashes),
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
    (output_dir / "market_state_no_backfill_shadow_window_discovery.json").write_text(
        json.dumps(_json_safe(summary), indent=2) + "\n",
        encoding="utf-8",
    )
    report = [
        "# No-Backfill Shadow Window Discovery",
        "",
        "This report discovers scored global-rank no-backfill threshold-controller windows.",
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
        _markdown_table(frame) if not frame.empty else "_No scored windows discovered._",
        "",
        "## Interpretation",
        "",
        (
            "Appendable windows are shadow-only no-backfill score outputs under "
            "the active T1 global-over-time rank contract with complete hashes "
            "and replay artifacts. They can be appended to the no-backfill "
            "monitor, but they do not imply controller promotion."
        ),
        "",
    ]
    (output_dir / "market_state_no_backfill_shadow_window_discovery_report.md").write_text(
        "\n".join(report),
        encoding="utf-8",
    )
    return {**summary, "readiness": frame, "appendable": appendable}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-root", action="append", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--existing-monitor-dir", type=Path, default=None)
    parser.add_argument("--include-regex", default=r"globalrank.*no_backfill")
    parser.add_argument("--exclude-regex", default=None)
    parser.add_argument("--expected-rank-contract", default=DEFAULT_EXPECTED_RANK_CONTRACT)
    parser.add_argument("--expected-selected-arm", default=DEFAULT_EXPECTED_SELECTED_ARM)
    parser.add_argument("--expected-active-head", action="append", default=None)
    parser.add_argument("--expected-disabled-head", action="append", default=None)
    parser.add_argument("--min-timestamp-count", type=int, default=3)
    parser.add_argument("--min-start-after", default=None)
    parser.add_argument("--max-candidates", type=int, default=0)
    parser.add_argument("--allow-incomplete-hashes", action="store_true")
    parser.add_argument("--fail-if-none", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    roots = list(args.search_root or [Path("data_perp/reports")])
    summary = discover_no_backfill_shadow_windows(
        roots=roots,
        output_dir=args.output_dir,
        config=args.config,
        existing_monitor_dir=args.existing_monitor_dir,
        include_regex=args.include_regex,
        exclude_regex=args.exclude_regex,
        expected_rank_contract=str(args.expected_rank_contract),
        expected_selected_arm=str(args.expected_selected_arm),
        expected_active_heads=args.expected_active_head or DEFAULT_EXPECTED_ACTIVE_HEADS,
        expected_disabled_heads=args.expected_disabled_head or DEFAULT_EXPECTED_DISABLED_HEADS,
        min_timestamp_count=int(args.min_timestamp_count),
        min_start_after=args.min_start_after,
        max_candidates=int(args.max_candidates),
        require_complete_hashes=not bool(args.allow_incomplete_hashes),
    )
    printable = {k: v for k, v in summary.items() if k not in {"readiness", "appendable"}}
    print(json.dumps(_json_safe(printable), indent=2))
    if bool(args.fail_if_none) and int(summary["appendable_candidate_count"]) <= 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
