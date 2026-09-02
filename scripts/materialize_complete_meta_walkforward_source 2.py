#!/usr/bin/env python3
"""Splice a historical meta handoff into a corrected complete-period source.

Rows before the cutover come from the long historical OOS base stream. Rows at
or after the cutover come exclusively from the corrected handoff. Only the
frozen meta feature contract and required label/context columns are retained,
which avoids loading or rewriting the full raw feature universe.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from scripts.run_s52_train_meta_regime_handoff_smoke import (
    KEY_COLUMNS,
    LEDGER_CONTEXT_COLUMNS,
    OUTCOME_COLUMNS,
    _load_fixed_selected_features,
    _load_fixed_selected_features_by_side,
    _projected_handoff_columns_for_selected,
)


def _read_filtered(
    path: Path,
    columns: list[str],
    *,
    cutover: pd.Timestamp,
    before: bool,
) -> pa.Table:
    available = set(pq.read_schema(path).names)
    selected = [column for column in columns if column in available]
    missing_keys = sorted(set(KEY_COLUMNS).difference(selected))
    if missing_keys:
        raise ValueError(f"{path} is missing key columns: {missing_keys}")
    table = pq.read_table(path, columns=selected)
    timestamp_type = table.schema.field("__ts__").type
    boundary = pa.scalar(cutover.to_pydatetime(), type=timestamp_type)
    mask = pc.less(table["__ts__"], boundary) if before else pc.greater_equal(
        table["__ts__"], boundary
    )
    return table.filter(mask)


def _concat_write(tables: list[pa.Table], path: Path) -> int:
    table = pa.concat_tables(tables, promote_options="default")
    pq.write_table(table, path, compression="zstd", use_dictionary=True)
    return int(table.num_rows)


def _coverage(path: Path) -> dict[str, object]:
    ts = pd.to_datetime(pd.read_parquet(path, columns=["__ts__"])["__ts__"], utc=True)
    month = ts.dt.strftime("%Y-%m")
    daily = ts.dt.floor("D")
    return {
        "rows": int(len(ts)),
        "start": ts.min().isoformat(),
        "end": ts.max().isoformat(),
        "days": int(daily.nunique()),
        "rows_by_month": {
            str(key): int(value) for key, value in month.value_counts().sort_index().items()
        },
        "days_by_month": {
            str(key): int(value)
            for key, value in pd.DataFrame({"month": month, "day": daily})
            .drop_duplicates()
            .groupby("month", observed=True)["day"]
            .size()
            .items()
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-handoff-dir", type=Path, required=True)
    parser.add_argument("--corrected-handoff-dir", type=Path, required=True)
    parser.add_argument("--selected-features-csv", type=Path, required=True)
    parser.add_argument("--cutover", default="2026-05-01T00:00:00Z")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cutover = pd.Timestamp(args.cutover)
    if cutover.tzinfo is None:
        cutover = cutover.tz_localize("UTC")
    else:
        cutover = cutover.tz_convert("UTC")

    historical_handoff = args.historical_handoff_dir / "train_meta_regime_handoff.parquet"
    corrected_handoff = args.corrected_handoff_dir / "train_meta_regime_handoff.parquet"
    historical_ledger = args.historical_handoff_dir / "s52_trailing_regime_scored_ledger.parquet"
    corrected_ledger = args.corrected_handoff_dir / "s52_trailing_regime_scored_ledger.parquet"

    fixed_features = list(_load_fixed_selected_features(args.selected_features_csv) or [])
    for features in (_load_fixed_selected_features_by_side(args.selected_features_csv) or {}).values():
        fixed_features.extend(features)
    fixed_features = list(dict.fromkeys(fixed_features))
    projected = sorted(
        set(
            _projected_handoff_columns_for_selected(corrected_handoff, fixed_features)
            or []
        )
        | set(
            _projected_handoff_columns_for_selected(historical_handoff, fixed_features)
            or []
        )
        | set(KEY_COLUMNS)
        | {"month", "score", "selected_top30"}
    )
    ledger_columns = sorted(
        set(KEY_COLUMNS)
        | {"month", "score", "selected_top30"}
        | set(OUTCOME_COLUMNS)
        | set(LEDGER_CONTEXT_COLUMNS)
    )

    handoff_rows = _concat_write(
        [
            _read_filtered(historical_handoff, projected, cutover=cutover, before=True),
            _read_filtered(corrected_handoff, projected, cutover=cutover, before=False),
        ],
        args.output_dir / "train_meta_regime_handoff.parquet",
    )
    ledger_rows = _concat_write(
        [
            _read_filtered(historical_ledger, ledger_columns, cutover=cutover, before=True),
            _read_filtered(corrected_ledger, ledger_columns, cutover=cutover, before=False),
        ],
        args.output_dir / "s52_trailing_regime_scored_ledger.parquet",
    )
    if handoff_rows != ledger_rows:
        raise AssertionError(
            f"handoff/ledger row mismatch: {handoff_rows:,} != {ledger_rows:,}"
        )

    coverage = _coverage(args.output_dir / "train_meta_regime_handoff.parquet")
    manifest = {
        "schema": "complete_meta_walkforward_source_v1",
        "historical_handoff_dir": str(args.historical_handoff_dir),
        "corrected_handoff_dir": str(args.corrected_handoff_dir),
        "cutover": cutover.isoformat(),
        "selected_features_csv": str(args.selected_features_csv),
        "frozen_feature_count": len(fixed_features),
        "materialized_handoff_columns": projected,
        "historical_rows_used_only_before_cutover": True,
        "corrected_rows_used_at_or_after_cutover": True,
        "coverage": coverage,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
