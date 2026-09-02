#!/usr/bin/env python3
"""Hydrate the selected carry feature in a spliced meta final-fit handoff."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from extreme_price_movements.data_store import read_symbol_features
from extreme_price_movements.fast_funcs import numba_rolling_zscore_fused


FEATURE = "carry_adj_ret_self_z_10h"
SOURCE_COLUMNS = [FEATURE, "ret10h", "fund_rate"]


def _feature_path(root: Path, symbol: str) -> Path:
    return root / f"symbol={symbol.replace('/', '_')}.parquet"


def _candidate_keys(handoff: Path, cutover: pd.Timestamp) -> pd.DataFrame:
    keys = pd.read_parquet(
        handoff, columns=["__ts__", "__symbol__", FEATURE]
    )
    keys["__ts__"] = pd.to_datetime(keys["__ts__"], utc=True, errors="coerce")
    keys = keys.loc[keys["__ts__"].ge(cutover), ["__ts__", "__symbol__", FEATURE]]
    return keys.drop_duplicates(["__ts__", "__symbol__"], keep="last")


def _symbol_carry(
    feature_root: Path,
    symbol: str,
    timestamps: pd.DatetimeIndex,
) -> pd.Series:
    path = _feature_path(feature_root, symbol)
    if not path.exists() or timestamps.empty:
        return pd.Series(dtype="float32")
    start = timestamps.min() - pd.Timedelta(days=15)
    end = timestamps.max()
    values = read_symbol_features(
        str(path), columns=SOURCE_COLUMNS, start_ts=start, end_ts=end
    )
    if values.empty:
        return pd.Series(dtype="float32")
    values.index = pd.to_datetime(values.index, utc=True, errors="coerce")
    values = values.loc[~values.index.duplicated(keep="last")].sort_index()
    direct = (
        pd.to_numeric(values[FEATURE], errors="coerce")
        if FEATURE in values
        else pd.Series(np.nan, index=values.index, dtype="float32")
    )
    ret = (
        pd.to_numeric(values["ret10h"], errors="coerce")
        if "ret10h" in values
        else pd.Series(np.nan, index=values.index, dtype="float32")
    )
    funding = (
        pd.to_numeric(values["fund_rate"], errors="coerce")
        if "fund_rate" in values
        else pd.Series(np.nan, index=values.index, dtype="float32")
    )
    if ret.notna().any() and funding.notna().any():
        raw = (ret - funding * np.float32(10.0 / 8.0)).astype(np.float32)
        derived = numba_rolling_zscore_fused(raw, 14 * 24).clip(-6.0, 6.0)
        direct = direct.where(direct.notna(), derived)
    return direct.reindex(timestamps).astype(np.float32)


def _build_lookup(
    handoff: Path,
    feature_root: Path,
    cutover: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, object]]:
    keys = _candidate_keys(handoff, cutover)
    parts: list[pd.DataFrame] = []
    requested = 0
    for symbol, group in keys.groupby("__symbol__", sort=False, observed=True):
        timestamps = pd.DatetimeIndex(group["__ts__"].dropna().sort_values().unique())
        requested += len(timestamps)
        carry = _symbol_carry(feature_root, str(symbol), timestamps)
        if carry.empty:
            continue
        parts.append(
            pd.DataFrame(
                {
                    "__ts__": carry.index,
                    "__symbol__": str(symbol),
                    FEATURE: carry.to_numpy(dtype=np.float32, copy=False),
                }
            )
        )
    lookup = (
        pd.concat(parts, ignore_index=True, copy=False)
        if parts
        else pd.DataFrame(columns=["__ts__", "__symbol__", FEATURE])
    )
    lookup = lookup.dropna(subset=[FEATURE]).drop_duplicates(
        ["__ts__", "__symbol__"], keep="last"
    )
    return lookup, {
        "requested_symbol_timestamps": int(requested),
        "hydrated_symbol_timestamps": int(len(lookup)),
        "hydrated_coverage": float(len(lookup) / max(requested, 1)),
        "symbols_requested": int(keys["__symbol__"].nunique()),
        "symbols_hydrated": int(lookup["__symbol__"].nunique()) if len(lookup) else 0,
    }


def _rewrite(handoff: Path, lookup: pd.DataFrame, cutover: pd.Timestamp) -> dict[str, int]:
    source = pq.ParquetFile(handoff)
    temp = handoff.with_suffix(".hydrating.parquet")
    lookup_series = lookup.set_index(["__ts__", "__symbol__"])[FEATURE]
    writer: pq.ParquetWriter | None = None
    filled = 0
    recent = 0
    try:
        for row_group in range(source.num_row_groups):
            table = source.read_row_group(row_group)
            frame = table.select(["__ts__", "__symbol__", FEATURE]).to_pandas()
            ts = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
            current = pd.to_numeric(frame[FEATURE], errors="coerce").to_numpy(
                dtype=np.float32, copy=True
            )
            recent_mask = ts.ge(cutover).to_numpy()
            recent += int(recent_mask.sum())
            if recent_mask.any():
                index = pd.MultiIndex.from_arrays(
                    [ts.loc[recent_mask], frame.loc[recent_mask, "__symbol__"].astype(str)]
                )
                replacement = lookup_series.reindex(index).to_numpy(
                    dtype=np.float32, copy=False
                )
                target = np.flatnonzero(recent_mask)
                use = ~np.isfinite(current[target]) & np.isfinite(replacement)
                current[target[use]] = replacement[use]
                filled += int(use.sum())
            field_index = table.schema.get_field_index(FEATURE)
            replacement_array = pa.array(current, type=pa.float32())
            table = table.set_column(field_index, FEATURE, replacement_array)
            if writer is None:
                writer = pq.ParquetWriter(
                    temp, table.schema, compression="zstd", use_dictionary=True
                )
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    os.replace(temp, handoff)
    return {"recent_rows": int(recent), "filled_rows": int(filled)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff-dir", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--cutover", default="2026-06-01T00:00:00Z")
    args = parser.parse_args()
    cutover = pd.Timestamp(args.cutover)
    cutover = (
        cutover.tz_localize("UTC") if cutover.tzinfo is None else cutover.tz_convert("UTC")
    )
    handoff = args.handoff_dir / "train_meta_regime_handoff.parquet"
    lookup, lookup_metrics = _build_lookup(handoff, args.feature_root, cutover)
    rewrite_metrics = _rewrite(handoff, lookup, cutover)
    payload = {
        "schema": "meta_finalfit_selected_feature_hydration_v1",
        "feature": FEATURE,
        "definition": (
            "rolling_z_336(ret10h - fund_rate * 10/8), clipped to [-6,6]"
        ),
        "cutover": cutover.isoformat(),
        "feature_root": str(args.feature_root),
        **lookup_metrics,
        **rewrite_metrics,
    }
    (args.handoff_dir / "selected_feature_hydration.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
