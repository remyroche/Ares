#!/usr/bin/env python3
"""Audit candidate ledgers for T1 rank-contract validation eligibility.

The T1 rank-contract comparison must use the same score path as the active
anchor/meta stack. A later-period ledger generated from a native reliability
blend is useful for sensitivity work, but it is not valid promotion evidence for
the T1 timestamp-vs-global rank contract.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_CANDIDATE_COLUMNS = (
    "timestamp",
    "symbol",
    "side",
    "strategy_id",
    "head",
    "calibrated_score",
    "normalized_rank_score",
    "strategy_rank_pct",
    "policy_rank_pct",
    "base_strategy_threshold",
    "entry_price",
    "exit_price",
    "exit_timestamp",
    "net_return",
    "gross_return",
    "holding_bars",
    "simple_policy_exit_reason",
)

ANCHOR_SCORE_COLUMNS = {
    "reliability_anchor_only_score",
    "anchor_score",
    "meta_score",
}
T1_ACTIVE_STACK_SCORE_COLUMNS = ANCHOR_SCORE_COLUMNS | {"calibrated_score"}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if np.isfinite(out) else None


def audit_candidate_root(root: Path, *, min_timestamp: pd.Timestamp | None = None) -> dict[str, Any]:
    policy_dir = root / "simple_policy_optimiser"
    broad_path = policy_dir / "simple_policy_candidates_broad.parquet"
    deployable_path = policy_dir / "simple_policy_candidates.parquet"
    manifest_path = root / "live_ledger_native_materialization_manifest.json"
    t1_manifest_path = root / "t1_repaired_static_baseline_manifest.json"
    t1_anchor_manifest_path = root / "t1_anchor_scored_candidate_manifest.json"
    manifest = _read_json(manifest_path)
    t1_manifest = _read_json(t1_manifest_path)
    t1_anchor_manifest = _read_json(t1_anchor_manifest_path)
    row: dict[str, Any] = {
        "root": str(root),
        "broad_path": str(broad_path),
        "broad_exists": broad_path.exists(),
        "deployable_exists": deployable_path.exists(),
        "manifest_path": str(manifest_path) if manifest_path.exists() else "",
        "t1_manifest_path": str(t1_manifest_path) if t1_manifest_path.exists() else "",
        "t1_anchor_manifest_path": str(t1_anchor_manifest_path) if t1_anchor_manifest_path.exists() else "",
        "generated_by": manifest.get("generated_by")
        or t1_manifest.get("generated_by")
        or t1_anchor_manifest.get("generated_by"),
        "candidate_rows": None,
        "timestamp_min": None,
        "timestamp_max": None,
        "timestamp_count": None,
        "heads": "",
        "missing_required_columns": "",
        "score_column": None,
        "score_source": None,
        "score_path": None,
        "score_path_exists": None,
        "score_is_anchor_compatible": False,
        "period_after_min_timestamp": None,
        "eligible_for_t1_rank_validation": False,
        "rejection_reasons": "",
    }
    reasons: list[str] = []
    if not broad_path.exists():
        reasons.append("missing_broad_candidates")
    else:
        frame = pd.read_parquet(broad_path)
        row["candidate_rows"] = int(len(frame))
        missing = [col for col in REQUIRED_CANDIDATE_COLUMNS if col not in frame.columns]
        row["missing_required_columns"] = ";".join(missing)
        if missing:
            reasons.append("missing_required_columns")
        ts = pd.to_datetime(frame.get("timestamp"), utc=True, errors="coerce")
        if ts.notna().any():
            row["timestamp_min"] = ts.min().isoformat()
            row["timestamp_max"] = ts.max().isoformat()
            row["timestamp_count"] = int(ts.nunique())
            if min_timestamp is not None:
                after_min = bool(ts.min() >= min_timestamp)
                row["period_after_min_timestamp"] = after_min
                if not after_min:
                    reasons.append("period_not_after_min_timestamp")
        else:
            reasons.append("missing_finite_timestamps")
        if "head" in frame.columns:
            row["heads"] = ",".join(sorted(frame["head"].dropna().astype(str).unique()))
        score = pd.to_numeric(frame.get("calibrated_score"), errors="coerce")
        row["calibrated_score_min"] = _safe_float(score.min())
        row["calibrated_score_max"] = _safe_float(score.max())
        row["calibrated_score_mean"] = _safe_float(score.mean())

    score_diag = manifest.get("score_diagnostics") if isinstance(manifest, dict) else {}
    active_stack = t1_manifest.get("active_stack") if isinstance(t1_manifest, dict) else {}
    score_contract = (
        t1_anchor_manifest.get("score_contract")
        if isinstance(t1_anchor_manifest, dict)
        else {}
    )
    score_column = None
    score_source = None
    score_path = None
    if isinstance(score_diag, dict) and score_diag:
        score_column = score_diag.get("score_column")
        score_source = score_diag.get("score_source")
        score_path = manifest.get("score_path")
    elif isinstance(active_stack, dict) and active_stack:
        score_column = active_stack.get("active_score_column")
        score_source = active_stack.get("score_path")
        score_path = None
    elif isinstance(score_contract, dict) and score_contract:
        score_column = score_contract.get("score_column")
        score_source = score_contract.get("score_source")
        score_path = t1_anchor_manifest.get("score_ledger_path")
    row["score_column"] = score_column
    row["score_source"] = score_source
    row["score_path"] = score_path
    if score_path:
        row["score_path_exists"] = Path(str(score_path)).exists()
        if not row["score_path_exists"]:
            reasons.append("referenced_score_path_missing")
    elif manifest_path.exists() and not t1_manifest_path.exists() and not t1_anchor_manifest_path.exists():
        reasons.append("missing_score_path")

    score_is_anchor = False
    if isinstance(score_diag, dict) and score_diag:
        # Native/materialized ledgers may rewrite ``calibrated_score`` to a
        # blend score.  Only explicit anchor/meta score arms are accepted from
        # score-diagnostics manifests; generic calibrated_score is accepted
        # only from the T1 active-stack manifest below.
        if score_column in ANCHOR_SCORE_COLUMNS:
            score_is_anchor = True
        elif score_column == "calibrated_score":
            reasons.append("generic_calibrated_score_requires_t1_manifest")
    elif score_column in T1_ACTIVE_STACK_SCORE_COLUMNS:
        score_is_anchor = True
    if isinstance(active_stack, dict) and active_stack.get("score_path") == "anchor_meta_calibrated_score":
        score_is_anchor = True
    if isinstance(score_contract, dict) and score_contract:
        if (
            score_contract.get("score_source") == "live_finalfit_anchor_meta_score"
            and bool(score_contract.get("native_reliability_blend_active")) is False
            and bool(score_contract.get("qfail_active")) is False
            and bool(score_contract.get("market_state_threshold_controller_active")) is False
        ):
            score_is_anchor = True
    row["score_is_anchor_compatible"] = bool(score_is_anchor)
    if not score_is_anchor:
        reasons.append("score_not_anchor_compatible")

    if not row["broad_exists"]:
        pass
    if not row["deployable_exists"]:
        reasons.append("missing_deployable_candidates")
    row["eligible_for_t1_rank_validation"] = len(reasons) == 0
    row["rejection_reasons"] = ";".join(dict.fromkeys(reasons))
    return row


def discover_candidate_roots(search_roots: list[Path]) -> list[Path]:
    """Find artifact roots containing simple-policy broad candidate ledgers."""

    roots: set[Path] = set()
    for search_root in search_roots:
        if not search_root.exists():
            continue
        if search_root.is_file():
            if search_root.name == "simple_policy_candidates_broad.parquet":
                roots.add(search_root.parent.parent)
            continue
        for path in search_root.rglob("simple_policy_candidates_broad.parquet"):
            if path.parent.name != "simple_policy_optimiser":
                continue
            roots.add(path.parent.parent)
    return sorted(roots, key=lambda p: str(p))


def _render_report(output_dir: Path, audit: pd.DataFrame) -> str:
    eligible = int(audit["eligible_for_t1_rank_validation"].fillna(False).sum()) if not audit.empty else 0
    latest_ts = None
    if not audit.empty and "timestamp_max" in audit.columns:
        ts = pd.to_datetime(audit["timestamp_max"], utc=True, errors="coerce").dropna()
        latest_ts = ts.max().isoformat() if not ts.empty else None
    reason_counts: dict[str, int] = {}
    if not audit.empty and "rejection_reasons" in audit.columns:
        for value in audit["rejection_reasons"].fillna("").astype(str):
            for reason in [part for part in value.split(";") if part]:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
    reason_lines = [
        f"| {reason} | {count} |"
        for reason, count in sorted(reason_counts.items(), key=lambda item: (-item[1], item[0]))[:10]
    ]
    lines = [
        "# T1 Rank-Validation Candidate Input Audit",
        "",
        "A ledger is eligible only when it has required outcome/cost fields, covers the requested later period, and uses an anchor/meta-compatible score path. Native reliability-blend scored ledgers are rejected for T1 rank-contract promotion evidence.",
        "",
        "## Summary",
        "",
        f"- Audited roots: `{int(len(audit))}`",
        f"- Eligible roots: `{eligible}`",
        f"- Latest audited timestamp: `{latest_ts}`",
        "",
        "| rejection reason | count |",
        "|---|---:|",
        *(reason_lines if reason_lines else ["| none | 0 |"]),
        "",
        "## Candidate Roots",
        "",
        "| eligible | timestamp_min | timestamp_max | rows | score_column | score_path_exists | rejection_reasons | root |",
        "|---:|---|---|---:|---|---:|---|---|",
    ]
    for _, row in audit.iterrows():
        lines.append(
            f"| {bool(row['eligible_for_t1_rank_validation'])} | {row.get('timestamp_min', '')} | "
            f"{row.get('timestamp_max', '')} | {row.get('candidate_rows', '')} | "
            f"{row.get('score_column', '')} | {row.get('score_path_exists', '')} | "
            f"{row.get('rejection_reasons', '')} | `{row.get('root', '')}` |"
        )
    lines.extend(
        [
            "",
            "## Verdict",
            "",
            (
                f"Eligible ledgers found: `{eligible}`."
                if eligible
                else "No eligible later-period T1 anchor/meta candidate ledger was found in the audited inputs."
            ),
            "",
            "Generated files:",
            f"- `{output_dir / 't1_rank_validation_input_audit.csv'}`",
            f"- `{output_dir / 't1_rank_validation_input_audit_report.md'}`",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", action="append", type=Path, default=[])
    parser.add_argument(
        "--discover-under",
        action="append",
        type=Path,
        default=[],
        help=(
            "Recursively discover artifact roots containing "
            "simple_policy_optimiser/simple_policy_candidates_broad.parquet."
        ),
    )
    parser.add_argument("--min-timestamp", default="2026-06-22T23:59:59Z")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    min_ts = pd.Timestamp(args.min_timestamp)
    min_ts = min_ts.tz_localize("UTC") if min_ts.tzinfo is None else min_ts.tz_convert("UTC")
    candidate_roots = sorted(
        set([Path(root) for root in args.candidate_root] + discover_candidate_roots(list(args.discover_under))),
        key=lambda p: str(p),
    )
    if not candidate_roots:
        raise SystemExit("No candidate roots were provided or discovered")
    rows = [audit_candidate_root(root, min_timestamp=min_ts) for root in candidate_roots]
    audit = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = args.output_dir / "t1_rank_validation_input_audit.csv"
    report_path = args.output_dir / "t1_rank_validation_input_audit_report.md"
    audit.to_csv(audit_path, index=False)
    report_path.write_text(_render_report(args.output_dir, audit), encoding="utf-8")
    print(f"Wrote T1 rank-validation input audit: {report_path}")


if __name__ == "__main__":
    main()
