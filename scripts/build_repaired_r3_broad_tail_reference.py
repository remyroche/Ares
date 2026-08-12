#!/usr/bin/env python3
"""Create an immutable R3 broad-to-tail reference manifest with repaired months.

The source manifest is never overwritten.  Every supplied recovery directory
must be complete for *both* sides of exactly one month; the two sides are
replaced together so a run cannot mix minute-source vintages within a month.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.materialize_packb_tp6_sl4_h12_labels import COST_BPS, IDENTITY_COLUMNS


TBM_PATH_COLUMNS = (
    "first_tp4_minute",
    "first_tp6_minute",
    "first_sl4_minute",
    "first_sl6_minute",
)


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _month(value: object) -> str:
    return pd.Timestamp(value, tz="UTC").strftime("%Y-%m")


def _load_historical_expected(source: Path, month: str, side: str) -> pd.DataFrame:
    frame = pd.read_parquet(source, columns=list(IDENTITY_COLUMNS))
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="raise")
    start = pd.Timestamp(f"{month}-01", tz="UTC")
    end = start + pd.offsets.MonthBegin(1)
    frame = frame.loc[frame.side_name.eq(side) & frame.__ts__.ge(start) & frame.__ts__.lt(end)]
    if frame.empty or frame.candidate_id.duplicated().any():
        raise ValueError(f"historical expected population is invalid for {month}/{side}")
    return frame.loc[:, list(IDENTITY_COLUMNS)].sort_values("candidate_id", kind="stable").reset_index(drop=True)


def _validate_part(path: Path, expected: pd.DataFrame, *, month: str, side: str) -> None:
    required = {
        *IDENTITY_COLUMNS,
        "label_valid",
        "target_invalid",
        "t4_tp6_sl4_gross_bps",
        "t4_tp6_sl4_net_bps",
        *TBM_PATH_COLUMNS,
    }
    frame = pd.read_parquet(path)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"repaired part missing columns {missing}: {path}")
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="raise")
    observed = frame.loc[:, list(IDENTITY_COLUMNS)].sort_values("candidate_id", kind="stable").reset_index(drop=True)
    if len(observed) != len(expected) or not observed.equals(expected):
        raise ValueError(f"repaired part changed canonical candidate identity: {month}/{side}")
    valid = frame.label_valid.astype(bool)
    if not frame.target_invalid.astype(bool).eq(~valid).all():
        raise ValueError(f"repaired part has invalid validity complement: {month}/{side}")
    if frame.loc[~valid, ["t4_tp6_sl4_gross_bps", "t4_tp6_sl4_net_bps"]].notna().any().any():
        raise ValueError(f"invalid path received economic target: {month}/{side}")
    if frame.loc[~valid, list(TBM_PATH_COLUMNS)].notna().any().any():
        raise ValueError(f"invalid path received first-touch TBM target: {month}/{side}")
    if not np.isfinite(frame.loc[valid, list(TBM_PATH_COLUMNS)].to_numpy(dtype=float)).all():
        raise ValueError(f"valid path lacks exact first-touch TBM fields: {month}/{side}")
    if not np.allclose(
        frame.loc[valid, "t4_tp6_sl4_gross_bps"].to_numpy(float) - COST_BPS,
        frame.loc[valid, "t4_tp6_sl4_net_bps"].to_numpy(float), rtol=0.0, atol=2e-3,
    ):
        raise ValueError(f"repaired part cost arithmetic failed: {month}/{side}")


def _recovery_month(root: Path) -> tuple[str, dict[str, Any]]:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"recovery manifest absent: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "complete" or not manifest.get("complete"):
        raise ValueError(f"recovery is not complete: {root}")
    cells = manifest.get("cells", {})
    months = cells.get("months", [])
    sides = cells.get("sides", [])
    if len(months) != 1 or set(sides) != {"long", "short"}:
        raise ValueError(f"recovery must be exactly one complete both-side month: {root}")
    if manifest.get("candidate_source_kind") != "historical":
        raise ValueError(f"this repair builder currently accepts historical recovery only: {root}")
    return str(months[0]), manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--recovery-dir", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"immutable repaired reference already exists: {args.output}")
    reference = pd.read_parquet(args.reference)
    required = {"path", "source_month", "population"}
    missing = sorted(required.difference(reference.columns))
    if missing:
        raise ValueError(f"reference manifest missing {missing}")
    replacement: dict[str, dict[str, Any]] = {}
    for recovery in args.recovery_dir:
        month, manifest = _recovery_month(recovery)
        if month in replacement:
            raise ValueError(f"duplicate repair month {month}")
        replacement[month] = {"root": recovery.resolve(), "manifest": manifest}
    output = reference.copy()
    audit: list[dict[str, Any]] = []
    for month, item in sorted(replacement.items()):
        source = Path(item["manifest"]["source_candidates"])
        if not source.is_file():
            raise FileNotFoundError(f"recovery candidate source unavailable: {source}")
        month_rows = output.source_month.astype(str).eq(month)
        if int(month_rows.sum()) != 2:
            raise ValueError(f"reference does not have exactly two rows for repair month {month}")
        for side in ("long", "short"):
            part = item["root"] / "parts" / f"month={month}" / f"side={side}.parquet"
            expected = _load_historical_expected(source, month, side)
            _validate_part(part, expected, month=month, side=side)
            selector = month_rows & output.path.astype(str).str.endswith(f"side={side}.parquet")
            if int(selector.sum()) != 1:
                raise ValueError(f"reference path naming cannot select {month}/{side}")
            output.loc[selector, "path"] = str(part)
            audit.append({
                "month": month, "side": side, "replacement_path": str(part),
                "rows": int(len(expected)), "part_sha256": _sha(part),
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(args.output, index=False, compression="zstd")
    manifest = {
        "schema": "r3_broad_tail_repaired_reference_v2",
        "status": "complete",
        "source_reference": str(args.reference.resolve()),
        "source_reference_sha256": _sha(args.reference),
        "output_reference": str(args.output.resolve()),
        "output_reference_sha256": _sha(args.output),
        "replacements": audit,
        "replacement_policy": "complete both-side monthly exact-path rebuilds only; no mixed minute-source vintage within a repaired month",
        "t3_contract": {
            "status": "exact_first_touch_required",
            "path_fields": list(TBM_PATH_COLUMNS),
            "invalid_rows": "all four fields null",
            "valid_rows": "all four fields finite; -1 denotes an exact no-touch outcome",
        },
    }
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "complete", "output": str(args.output), "repaired_months": sorted(replacement)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
