#!/usr/bin/env python3
"""Materialize a fail-closed native-L2 historical backfill readiness audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

# Make direct ``python scripts/<runner>.py`` execution use the repository
# package, matching the command form used by the roadmap.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.native_l2_backfill_readiness import (
    DEFAULT_SCAN_ROOTS,
    aggregate_inventory,
    assess_candidate_window,
    discover_parquet_files,
    inventory_parquet_file,
)


DEFAULT_OVERLAP_MANIFEST = ROOT / "data_perp/artifacts/native_l2_candidate_overlap_audit_20260801_v2/run_manifest.json"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/native_l2_backfill_readiness_20260801_v1"


def _read_overlap_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"panels": [], "status": "MISSING_OVERLAP_MANIFEST"}
    data = json.loads(path.read_text(encoding="utf-8"))
    panels = data.get("panels", [])
    if not isinstance(panels, list):
        raise ValueError("overlap manifest panels must be a list")
    return {"panels": panels, "status": data.get("status", "UNKNOWN")}


def _write_report(
    output: Path,
    summary: dict[str, Any],
    window: dict[str, Any],
    panels: list[dict[str, Any]],
    overlap_status: str,
) -> None:
    lines = [
        "# Native-L2 historical backfill readiness",
        "",
        "Status: `RESEARCH_ONLY_NATIVE_L2_BACKFILL_REQUIRED`",
        "",
        "This is a source and coverage audit only. It reads parquet metadata plus source, product-identity, and timestamp columns. It does not read labels, model scores, costs, or portfolio fields, and it does not create a training panel.",
        "",
        "## Decision",
        "",
        f"- Declared candidate window: **{window.get('required_candidate_min_ts') or 'unknown'}** to **{window.get('required_candidate_max_ts') or 'unknown'}**.",
        f"- Exact native-L2 window found locally: **{summary.get('native_min_ts') or 'none'}** to **{summary.get('native_max_ts') or 'none'}**.",
        f"- Native source contains the full declared candidate window: **{bool(window.get('native_window_contains_declared_candidate_window'))}**.",
        f"- Historical native-L2 backfill required: **{bool(window.get('historical_native_backfill_required'))}**.",
        "",
        "## Local source inventory",
        "",
        f"- Files scanned: **{int(summary.get('files', 0)):,}**.",
        f"- Rows scanned: **{int(summary.get('rows', 0)):,}**.",
        f"- Exact native rows (`kraken_futures_l2_snapshot`): **{int(summary.get('native_rows', 0)):,}**.",
        f"- Native product-file identities: **{int(summary.get('native_product_file_identities', 0)):,}** (file-key count; not a base-asset collapse).",
        f"- Explicit proxy rows (`local_ohlcv_summary`): **{int(summary.get('proxy_rows', 0)):,}**.",
        f"- Files containing native rows: **{int(summary.get('native_files', 0)):,}**.",
        f"- Proxy-only files: **{int(summary.get('proxy_only_files', 0)):,}**.",
        f"- Native UTC calendar days with observations: **{int(summary.get('native_coverage_days', 0)):,}**.",
        f"- Native UTC calendar gaps inside the observed span: **{len(summary.get('native_missing_calendar_days', [])):,}**.",
        "",
        "| native UTC day | native rows |",
        "|---|---:|",
    ]
    for day, count in sorted(dict(summary.get("native_day_counts") or {}).items()):
        lines.append(f"| {day} | {int(count):,} |")
    lines.extend(
        [
            "",
            "| panel | rows | symbols | required start | required end |",
            "|---|---:|---:|---|---|",
        ]
    )
    for panel in sorted(panels, key=lambda value: str(value.get("panel_id", ""))):
        lines.append(
            f"| {panel.get('panel_id', '')} | {int(panel.get('rows', 0)):,} | {int(panel.get('symbols', 0)):,} | {panel.get('min_candidate_ts', '')} | {panel.get('max_candidate_ts', '')} |"
        )
    lines.extend(
        [
            "",
            "## Blocking evidence",
            "",
            "- The native cohort begins after the earliest declared candidate panels, so it cannot support historical strict OOF joins for the full roadmap window.",
            "- Proxy OHLCV rows are inventory evidence only and are not admitted as native depth, flow, or resilience features.",
            "- The existing overlap manifest is a readiness diagnostic (`%s`); it does not authorize model fitting." % overlap_status,
            "- A future backfill must preserve exact product identity, source timestamps, and observed/publication semantics; forward filling or proxy substitution is forbidden.",
            "",
            "## Next admissible action",
            "",
            "1. Acquire or materialize a longer timestamped native-L2 feed covering the declared candidate window.",
            "2. Re-run the existing native sidecar generator with the same source allow-list and bounded lag contract.",
            "3. Re-run the exact-product backward as-of overlap audit before reading labels or fitting any model.",
            "4. Only if coverage is adequate, build strict OOF `retain | clear` labels and evaluate one pooled-global top-k book with latest-month, worst-month, side, and cost gates.",
            "",
            "## Fail-closed gates",
            "",
            "- `candidate_joined`: false.",
            "- `labels_used`: false.",
            "- `model_fitted`: false.",
            "- `promotion_eligible`: false.",
            "- `portfolio_constraints_in_scope`: false.",
        ]
    )
    (output / "NATIVE_L2_BACKFILL_READINESS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan-root", action="append", type=Path, dest="scan_roots")
    parser.add_argument("--overlap-manifest", type=Path, default=DEFAULT_OVERLAP_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    roots = args.scan_roots or [ROOT / root for root in DEFAULT_SCAN_ROOTS]
    paths = discover_parquet_files(roots)
    records = [inventory_parquet_file(path) for path in paths]
    summary = aggregate_inventory(records)
    overlap = _read_overlap_manifest(args.overlap_manifest)
    panels = list(overlap.get("panels", []))
    window = assess_candidate_window(summary, panels)

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_json(output / "source_file_inventory.jsonl", orient="records", lines=True)
    pd.DataFrame(records).to_csv(output / "source_file_inventory.csv", index=False)
    (output / "source_inventory_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "candidate_window_readiness.json").write_text(
        json.dumps({"window": window, "panels": panels}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "status": "RESEARCH_ONLY_NATIVE_L2_BACKFILL_REQUIRED",
        "promotion_eligible": False,
        "candidate_joined": False,
        "labels_used": False,
        "model_fitted": False,
        "portfolio_constraints_in_scope": False,
        "source_allow_list": ["kraken_futures_l2_snapshot"],
        "proxy_sources_excluded": ["local_ohlcv_summary"],
        "scan_roots": [str(root) for root in roots],
        "files_scanned": len(records),
        "inventory_summary": "source_inventory_summary.json",
        "candidate_window": "candidate_window_readiness.json",
        "overlap_manifest": str(args.overlap_manifest),
        "window": window,
        "native_coverage_days": int(summary.get("native_coverage_days", 0)),
        "native_missing_calendar_days": list(summary.get("native_missing_calendar_days", [])),
        "native_day_counts": dict(summary.get("native_day_counts") or {}),
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_report(output, summary, window, panels, str(overlap.get("status", "UNKNOWN")))
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
