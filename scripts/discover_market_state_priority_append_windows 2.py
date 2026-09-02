#!/usr/bin/env python3
"""Discover candidate ledgers that can extend priority shadow validation.

This script scans artifact/report roots for `simple_policy_candidates.parquet`,
filters to T1-like candidate ledgers with manifests, then runs the same
readiness audit used by the safe-grid runner.  It is a discovery/reporting tool:
by default it exits successfully even when no appendable window exists.
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

from scripts.audit_market_state_priority_window_readiness import (
    DEFAULT_EXISTING_MANIFEST,
    audit_window_readiness,
)


DEFAULT_OUTPUT_DIR = Path(
    "data_perp/reports/market_state_priority_append_window_discovery_20260626"
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


def _load_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
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


def _artifact_root_for_candidate(path: Path) -> Path:
    return path.parent.parent if len(path.parents) >= 2 else path.parent


def _candidate_manifest(path: Path) -> Path | None:
    root = _artifact_root_for_candidate(path)
    for name in [
        "t1_repaired_static_baseline_manifest.json",
        "t1_anchor_scored_candidate_manifest.json",
        "live_ledger_native_materialization_manifest.json",
    ]:
        manifest = root / name
        if manifest.exists():
            return manifest
    manifests = sorted(root.glob("*manifest*.json"))
    return manifests[0] if manifests else None


def _is_t1_candidate(path: Path) -> bool:
    manifest = _candidate_manifest(path)
    if manifest is None:
        return False
    payload = _load_json(manifest)
    active_stack = dict(payload.get("active_stack") or {})
    generated_by = str(payload.get("generated_by") or "")
    if generated_by == "materialize_t1_repaired_static_baseline":
        return True
    if active_stack.get("name") == "T1_repaired_static_baseline":
        return True
    rank_contract = str(active_stack.get("rank_contract") or "")
    return bool(rank_contract and {"short_asset", "short_boll"}.issubset(set(active_stack.get("enabled_heads") or [])))


def discover_candidates(
    roots: list[Path],
    *,
    include_regex: str | None = None,
    exclude_regex: str | None = None,
) -> list[Path]:
    include = re.compile(include_regex) if include_regex else None
    exclude = re.compile(exclude_regex) if exclude_regex else None
    paths: dict[str, Path] = {}
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("simple_policy_candidates.parquet"):
            text = str(path)
            if include and not include.search(text):
                continue
            if exclude and exclude.search(text):
                continue
            if not _is_t1_candidate(path):
                continue
            paths[text] = path
    return [paths[key] for key in sorted(paths)]


def discover_append_windows(
    *,
    roots: list[Path],
    existing_manifest: Path,
    output_dir: Path,
    include_regex: str | None = None,
    exclude_regex: str | None = None,
    min_timestamp_count: int = 3,
    min_rows: int = 1,
    max_candidates: int = 0,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = discover_candidates(
        roots,
        include_regex=include_regex,
        exclude_regex=exclude_regex,
    )
    if max_candidates and max_candidates > 0:
        candidates = candidates[: int(max_candidates)]
    readiness_dir = output_dir / "readiness"
    readiness = audit_window_readiness(
        candidates=candidates,
        existing_manifest=existing_manifest,
        output_dir=readiness_dir,
        min_timestamp_count=min_timestamp_count,
        min_rows=min_rows,
    )
    frame = pd.DataFrame(readiness.get("candidate_rows") or [])
    appendable = frame.loc[frame["status"].eq("pass")].copy() if not frame.empty else pd.DataFrame()
    latest_end = None
    if not frame.empty and "end" in frame:
        ends = pd.to_datetime(frame["end"], utc=True, errors="coerce").dropna()
        latest_end = ends.max().isoformat() if not ends.empty else None
    appendable_path = output_dir / "appendable_market_state_priority_windows.csv"
    appendable.to_csv(appendable_path, index=False)
    summary = {
        "generated_by": "discover_market_state_priority_append_windows",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "search_roots": [str(root) for root in roots],
        "existing_manifest": str(existing_manifest),
        "include_regex": include_regex,
        "exclude_regex": exclude_regex,
        "min_timestamp_count": int(min_timestamp_count),
        "min_rows": int(min_rows),
        "max_candidates": int(max_candidates),
        "discovered_candidate_count": int(len(candidates)),
        "appendable_candidate_count": int(len(appendable)),
        "latest_discovered_window_end": latest_end,
        "readiness_output_dir": str(readiness_dir),
        "appendable_csv": str(appendable_path),
        "readiness_passed_all_candidates": bool(readiness.get("passed")),
    }
    (output_dir / "market_state_priority_append_window_discovery.json").write_text(
        json.dumps(_json_safe(summary), indent=2) + "\n",
        encoding="utf-8",
    )
    report = [
        "# Market-State Priority Append Window Discovery",
        "",
        f"Discovered candidates: `{len(candidates)}`",
        f"Appendable candidates: `{len(appendable)}`",
        f"Latest discovered end: `{latest_end}`",
        "",
        "## Appendable Windows",
        "",
        _markdown_table(appendable) if not appendable.empty else "_No appendable windows found._",
        "",
        "## Readiness Output",
        "",
        f"- `{readiness_dir}`",
        "",
        "## Interpretation",
        "",
        (
            "Appendable windows are candidates that pass the fixed safe-grid "
            "readiness contract. Non-appendable candidates may still be useful "
            "for other experiments, but they should not be added to this "
            "market-state priority shadow validation."
        ),
        "",
    ]
    (output_dir / "market_state_priority_append_window_discovery_report.md").write_text(
        "\n".join(report),
        encoding="utf-8",
    )
    return {**summary, "readiness": readiness}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-root", action="append", type=Path, default=None)
    parser.add_argument("--existing-manifest", type=Path, default=DEFAULT_EXISTING_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--include-regex",
        default=r"T1|t1|global_rank|rank_validation",
        help="Optional regex applied to candidate paths before manifest inspection.",
    )
    parser.add_argument("--exclude-regex", default=None)
    parser.add_argument("--min-timestamp-count", type=int, default=3)
    parser.add_argument("--min-rows", type=int, default=1)
    parser.add_argument("--max-candidates", type=int, default=0)
    parser.add_argument("--fail-if-none", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    roots = list(args.search_root or [Path("data_perp/artifacts")])
    summary = discover_append_windows(
        roots=roots,
        existing_manifest=args.existing_manifest,
        output_dir=args.output_dir,
        include_regex=args.include_regex,
        exclude_regex=args.exclude_regex,
        min_timestamp_count=int(args.min_timestamp_count),
        min_rows=int(args.min_rows),
        max_candidates=int(args.max_candidates),
    )
    print(json.dumps(_json_safe(summary), indent=2))
    if bool(args.fail_if_none) and int(summary["appendable_candidate_count"]) <= 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
