#!/usr/bin/env python3
"""Materialize the complete side-local three-month Geometry/K9 warm-up."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.strict_r3_canonical_v2 import _geometry_definition_months  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger-root", type=Path, required=True)
    parser.add_argument("--start", default="2024-10-01T00:00:00Z")
    parser.add_argument("--end-exclusive", default="2025-01-01T00:00:00Z")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    start, end, expected_months = _geometry_definition_months(args.start, args.end_exclusive)
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output already exists: {args.out_dir}")
    months = list(pd.date_range(start, end, freq="MS", inclusive="left"))
    paths = [args.ledger_root / "ledger" / f"month={month:%Y-%m}" / "prequential_base_ledger.parquet" for month in months]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"geometry warm-up is missing ledger partitions: {missing}")
    frame = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
    if frame.candidate_id.duplicated().any():
        raise ValueError("geometry warm-up duplicate candidate identities")
    if not frame.side_name.astype(str).str.lower().eq("short").all():
        raise ValueError("short geometry warm-up contains non-short rows")
    observed = set(frame.__decision_ts__.dt.strftime("%Y-%m"))
    if observed != expected_months:
        raise ValueError(f"unexpected warm-up month coverage: {sorted(observed)}")
    required = (
        "stack_is_prequential", "geometry_definition_population_complete",
        "prequential_base_anchor_bps", "h12_label_valid",
        "h12_label_available_ts", "h12_tp6_sl4_net_bps",
    )
    missing_fields = sorted(set(required).difference(frame.columns))
    if missing_fields:
        raise KeyError(f"warm-up ledger is missing {missing_fields}")
    if not frame.stack_is_prequential.fillna(False).astype(bool).all():
        raise ValueError("geometry warm-up contains a non-prequential base score")
    if not frame.geometry_definition_population_complete.fillna(False).astype(bool).all():
        raise ValueError("geometry warm-up did not preserve the complete target-free population")
    audit = frame.assign(__month__=frame.__decision_ts__.dt.strftime("%Y-%m")).groupby("__month__", as_index=False).agg(
        target_free_rows=("candidate_id", "size"),
        finite_base_anchor_rows=("prequential_base_anchor_bps", lambda value: int(np.isfinite(pd.to_numeric(value, errors="coerce")).sum())),
        h12_valid_rows=("h12_label_valid", lambda value: int(value.fillna(False).astype(bool).sum())),
        h12_available_before_definition_end=("h12_label_available_ts", lambda value: int((pd.to_datetime(value, utc=True, errors="coerce") < end).sum())),
    )
    args.out_dir.mkdir(parents=True)
    frame.sort_values(["__decision_ts__", "candidate_id"], kind="stable").to_parquet(
        args.out_dir / "geometry_warmup_ledger.parquet", index=False, compression="zstd"
    )
    audit.to_parquet(args.out_dir / "geometry_warmup_audit.parquet", index=False, compression="zstd")
    (args.out_dir / "run_manifest.json").write_text(json.dumps({
        "schema": "strict_r3_short_p0_f90_geometry_warmup_v1",
        "side": "short", "definition_start": start.isoformat(),
        "definition_end_exclusive": end.isoformat(), "source_ledger": str(args.ledger_root),
        "target_free_population_preserved": True,
        "base_scores": "strict-prequential P0/F90 only",
        "geometry_target": "exact H12 TP6/SL4 net > prequential P0 base anchor",
    }, indent=2) + "\n")
    print(args.out_dir)


if __name__ == "__main__":
    main()
