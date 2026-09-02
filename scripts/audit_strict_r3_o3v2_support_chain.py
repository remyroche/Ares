#!/usr/bin/env python3
"""Audit sealed O3-v2 support-funnel score receipts without refitting models.

The support funnel deliberately writes target-free held score panels and only
joins policy outcomes for diagnostics.  This auditor verifies that contract at
the receipt level across every stage and records one immutable correctness
receipt outside the model-output roots.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

import run_strict_r3_o3v2_target_funnel as target


REQUIRED = frozenset({
    "candidate_id", "__decision_ts__", "side_name", "enhanced_base_routed",
    "enhanced_base_bps", "base_rank_ts", "conditional_consensus_rank",
    "ordinary_shadow_consensus_rank", "head_agreement_std",
})


def _exclusive_json(path: Path, payload: object) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _audit_root(root: Path) -> list[dict[str, object]]:
    manifest = root / "run_manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(f"missing sealed manifest: {manifest}")
    rows: list[dict[str, object]] = []
    for path in sorted((root / "target_free_scores").rglob("*.parquet")):
        frame = pd.read_parquet(path)
        leaked = sorted(target.PROHIBITED_SCORE_COLUMNS.intersection(frame.columns))
        missing = sorted(REQUIRED - set(frame.columns))
        duplicate_ids = int(frame["candidate_id"].duplicated().sum()) if "candidate_id" in frame else len(frame)
        base_values = frame["enhanced_base_bps"] if "enhanced_base_bps" in frame else pd.Series(index=frame.index, dtype=float)
        routed_values = frame["enhanced_base_routed"] if "enhanced_base_routed" in frame else pd.Series(index=frame.index, dtype=bool)
        base_coverage = float(pd.to_numeric(base_values, errors="coerce").notna().mean()) if len(frame) else 0.0
        routed_coverage = float(routed_values.fillna(False).astype(bool).mean()) if len(frame) else 0.0
        rows.append({
            "root": str(root), "receipt": str(path.relative_to(root)), "rows": int(len(frame)),
            "duplicate_candidate_ids": duplicate_ids, "base_feature_coverage": base_coverage,
            "routed_coverage": routed_coverage, "forbidden_outcome_fields": leaked,
            "missing_required_fields": missing,
            "passed": bool(not leaked and not missing and duplicate_ids == 0 and base_coverage >= 0.90),
        })
    if not rows:
        raise AssertionError(f"no target-free score receipts under {root}")
    return rows


def run(*, roots: tuple[Path, ...], selection: Path, out: Path) -> None:
    if out.exists():
        raise FileExistsError(out)
    selected = json.loads(selection.read_text())
    development = tuple(selected.get("development_months", ()))
    # Selection is intentionally date-agnostic: successor contracts are
    # allowed to choose their own predeclared development window.  The old
    # literal 2025-Q4 check made this audit unusable for otherwise valid
    # March--April 2026 physical/support selections.
    if not development or any(not isinstance(month, str) or len(month) != 7 for month in development):
        raise AssertionError(f"selection has no valid development months: {development}")
    rows = [row for root in roots for row in _audit_root(root)]
    failed = [row for row in rows if not row["passed"]]
    out.mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(out / "support_receipt_audit.parquet", index=False, compression="zstd")
    _exclusive_json(out / "correctness_report.json", {
        "schema": "strict_r3_o3v2_support_chain_audit_v1",
        "passed": not failed,
        "receipts": len(rows),
        "failed_receipts": failed,
        "selection": str(selection),
        "development_months": list(development),
        "checks": {
            "target_free": "no policy/semantic outcome columns in held score files",
            "identity": "candidate IDs unique within every receipt",
            "coverage": "enhanced base bps present on at least 90% of every receipt",
            "required_score_contract": sorted(REQUIRED),
        },
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(roots=tuple(args.root), selection=args.selection, out=args.out)


if __name__ == "__main__":
    main()
