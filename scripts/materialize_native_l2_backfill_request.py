#!/usr/bin/env python3
"""Materialize a label-free native-L2 historical backfill request."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.native_l2_backfill_request import build_backfill_request


DEFAULT_OVERLAP = ROOT / "data_perp/artifacts/native_l2_candidate_overlap_audit_20260801_v3/run_manifest.json"
DEFAULT_SIDECAR = ROOT / "data_perp/artifacts/native_l2_continuation_sidecar_20260801_v3/native_l2_continuation_features.parquet"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/native_l2_backfill_request_20260801_v1"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overlap-manifest", type=Path, default=DEFAULT_OVERLAP)
    parser.add_argument("--native-sidecar", type=Path, default=DEFAULT_SIDECAR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    manifest = json.loads(args.overlap_manifest.read_text(encoding="utf-8"))
    panels = manifest.get("panels") or []
    requirements, summary = build_backfill_request(
        panels,
        root=ROOT,
        native_sidecar=args.native_sidecar,
    )
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    requirements.to_csv(output / "native_l2_backfill_request.csv", index=False)
    requirements.loc[~requirements["native_coverage"]].to_csv(
        output / "native_l2_missing_symbol_days.csv", index=False
    )
    (output / "run_manifest.json").write_text(
        json.dumps(
            {
                **summary,
                "overlap_manifest": str(args.overlap_manifest),
                "output_rows": int(len(requirements)),
                "outputs": ["native_l2_backfill_request.csv", "native_l2_missing_symbol_days.csv"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Native-L2 backfill request",
        "",
        "Status: `RESEARCH_ONLY_BACKFILL_REQUEST_NO_MODEL`",
        "",
        "This request was built from candidate identity and availability timestamps plus the existing native sidecar's product/timestamp columns. Labels, scores, costs, portfolio fields, and model outputs were not loaded.",
        "",
        f"- Required candidate window: **{summary['required_candidate_min_day']}** through **{summary['required_candidate_max_day']}**.",
        f"- Candidate symbols: **{summary['candidate_symbol_count']:,}**.",
        f"- Candidate symbol/day pairs: **{summary['candidate_symbol_day_pairs']:,}**.",
        f"- Pairs currently covered by native snapshots: **{summary['currently_covered_symbol_day_pairs']:,}**.",
        f"- Missing symbol/day pairs requested: **{summary['missing_symbol_day_pairs']:,}**.",
        "",
        "The provider must return factual native order-book snapshots with exact product identity, observed/publication timestamps, and no OHLCV proxy substitution. After acquisition, rerun the native sidecar and exact-product backward as-of overlap before any labels, OOF fitting, HPO, or economics.",
    ]
    (output / "NATIVE_L2_BACKFILL_REQUEST.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
