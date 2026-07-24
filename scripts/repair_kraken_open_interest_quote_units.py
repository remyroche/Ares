#!/usr/bin/env python3
"""Repair legacy Kraken OI rows stored in native rather than quote units.

The live ticker persists ``openInterestValue`` (quote notional), while the
analytics endpoint returns native contract/base amounts. This tool compares
both possible representations against quote-notional observations bracketing a
known suspect interval and only converts rows when doing so materially improves
continuity. It is intentionally bounded by explicit start/end timestamps.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from extreme_price_movements.data_store import (
    PartitionedOHLCVStore,
    _kraken_oi_to_quote_notional,
)


def _load_series(path: Path) -> pd.Series:
    frame = pd.read_parquet(path)
    frame.index = pd.to_datetime(frame.index, utc=True, errors="coerce").floor("h")
    values = pd.to_numeric(frame["open_interest"], errors="coerce")
    values = values.where(np.isfinite(values) & (values > 0.0))
    return values[~values.index.duplicated(keep="last")].sort_index()


def _expected_log_path(
    series: pd.Series,
    target_index: pd.DatetimeIndex,
) -> np.ndarray | None:
    before = series.loc[series.index < target_index.min()].dropna()
    after = series.loc[series.index > target_index.max()].dropna()
    if before.empty or after.empty:
        return None
    left = float(np.log(before.iloc[-1]))
    right = float(np.log(after.iloc[0]))
    return np.linspace(left, right, len(target_index) + 2, dtype=np.float64)[1:-1]


def _repair_symbol(
    *,
    sidecar_path: Path,
    price_store: PartitionedOHLCVStore,
    start: pd.Timestamp,
    end: pd.Timestamp,
    minimum_log_improvement: float,
    apply: bool,
    backup_dir: Path | None,
) -> dict[str, object]:
    symbol_key = sidecar_path.stem
    symbol = symbol_key.replace("_USD_USD", "/USD:USD")
    oi = _load_series(sidecar_path)
    target_index = pd.DatetimeIndex(
        oi.index[(oi.index >= start) & (oi.index <= end)]
    )
    record: dict[str, object] = {
        "symbol": symbol,
        "rows_in_interval": int(len(target_index)),
        "converted_rows": 0,
        "embedded_converted_rows": 0,
        "embedded_native_unit_rows_remaining": 0,
        "mean_log_improvement": 0.0,
        "applied": bool(apply),
    }
    if target_index.empty:
        return record

    expected = _expected_log_path(oi, target_index)
    if expected is None:
        record["reason"] = "missing_bracketing_quote_observations"
        return record
    prices = price_store.load(
        symbol,
        start_ts=start - pd.Timedelta(hours=1),
        end_ts=end + pd.Timedelta(hours=1),
    )
    if prices.empty or not {"mark_close", "close"}.intersection(prices.columns):
        record["reason"] = "missing_perp_close"
        return record
    raw = oi.reindex(target_index).astype(np.float64)
    converted = _kraken_oi_to_quote_notional(raw, prices).astype(np.float64)
    raw_error = np.abs(np.log(raw) - expected)
    converted_error = np.abs(np.log(converted) - expected)
    improvement = raw_error - converted_error
    convert_mask = (
        np.isfinite(converted)
        & (converted > 0.0)
        & np.isfinite(improvement)
        & (improvement >= float(minimum_log_improvement))
    )
    converted_rows = int(convert_mask.sum())
    record["converted_rows"] = converted_rows
    record["mean_log_improvement"] = (
        float(improvement.loc[convert_mask].mean()) if converted_rows else 0.0
    )
    if not converted_rows or not apply:
        return record

    if backup_dir is not None:
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup_path = backup_dir / "open_interest_hourly" / sidecar_path.name
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        if not backup_path.exists():
            shutil.copy2(sidecar_path, backup_path)
    converted_index = target_index[convert_mask.to_numpy()]
    converted_values = converted.loc[convert_mask]
    original_values = raw.loc[convert_mask]
    oi.loc[converted_index] = converted_values
    out = oi.rename("open_interest").to_frame().astype(np.float32)
    tmp = sidecar_path.with_suffix(".parquet.tmp")
    out.to_parquet(tmp, compression="zstd")
    tmp.replace(sidecar_path)

    # Historical feature generation reads embedded OI from the hourly OHLCV
    # partitions, while live residual hydration may read the sidecar. Repair
    # both contracts, but only where the embedded value still matches the
    # diagnosed native-unit observation. Existing quote-notional duplicates
    # are deliberately left byte-for-byte unchanged.
    symbol_dir = Path(price_store._get_symbol_dir(symbol))
    embedded_converted = 0
    original_by_ts = original_values.to_dict()
    converted_by_ts = converted_values.to_dict()
    for part_path in sorted(symbol_dir.glob("year=*/*.parquet")):
        try:
            frame = pd.read_parquet(part_path)
        except Exception:
            continue
        if frame.empty or "ts" not in frame.columns or "open_interest" not in frame:
            continue
        timestamps = pd.to_datetime(frame["ts"], utc=True, errors="coerce").dt.floor(
            "h"
        )
        values = pd.to_numeric(frame["open_interest"], errors="coerce")
        changed = np.zeros(len(frame), dtype=bool)
        replacements = values.to_numpy(dtype=np.float64, copy=True)
        for row_pos, ts in enumerate(timestamps):
            native_value = original_by_ts.get(ts)
            quote_value = converted_by_ts.get(ts)
            current = replacements[row_pos]
            if native_value is None or quote_value is None:
                continue
            if not (np.isfinite(current) and current > 0.0):
                continue
            if abs(np.log(current) - np.log(float(native_value))) > 0.05:
                continue
            replacements[row_pos] = float(quote_value)
            changed[row_pos] = True
        if not bool(changed.any()):
            continue
        if backup_dir is not None:
            relative = part_path.relative_to(Path(price_store.root_dir))
            backup_path = backup_dir / relative
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            if not backup_path.exists():
                shutil.copy2(part_path, backup_path)
        frame.loc[changed, "open_interest"] = replacements[changed].astype(np.float32)
        tmp_part = part_path.with_suffix(".parquet.tmp")
        frame.to_parquet(tmp_part, index=False, compression="zstd")
        tmp_part.replace(part_path)
        embedded_converted += int(changed.sum())
    record["embedded_converted_rows"] = embedded_converted
    reloaded = price_store.load(symbol, start_ts=start, end_ts=end)
    if "open_interest" in reloaded.columns:
        observed = pd.to_numeric(
            reloaded["open_interest"], errors="coerce"
        ).reindex(converted_index)
        observed_raw_error = np.abs(np.log(observed) - np.log(original_values))
        observed_quote_error = np.abs(np.log(observed) - np.log(converted_values))
        native_remaining = (
            np.isfinite(observed_raw_error)
            & np.isfinite(observed_quote_error)
            & (observed_raw_error < observed_quote_error)
        )
        record["embedded_native_unit_rows_remaining"] = int(native_remaining.sum())
        if bool(native_remaining.any()):
            raise RuntimeError(
                f"Embedded OI repair did not win load precedence for {symbol}: "
                f"{int(native_remaining.sum())} native-unit rows remain"
            )
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--perp-root",
        type=Path,
        default=Path("data_perp/exchanges/krakenfutures"),
    )
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--minimum-log-improvement", type=float, default=0.35)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--backup-dir", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
    sidecar_root = args.perp_root / "open_interest_hourly"
    price_store = PartitionedOHLCVStore(
        root_dir=str(args.perp_root), timeframe="1h"
    )
    rows = [
        _repair_symbol(
            sidecar_path=path,
            price_store=price_store,
            start=start,
            end=end,
            minimum_log_improvement=float(args.minimum_log_improvement),
            apply=bool(args.apply),
            backup_dir=args.backup_dir,
        )
        for path in sorted(sidecar_root.glob("*.parquet"))
    ]
    report = pd.DataFrame(rows)
    summary = {
        "symbols": int(len(report)),
        "symbols_converted": int((report["converted_rows"] > 0).sum()),
        "converted_rows": int(report["converted_rows"].sum()),
        "embedded_converted_rows": int(report["embedded_converted_rows"].sum()),
        "embedded_native_unit_rows_remaining": int(
            report["embedded_native_unit_rows_remaining"].sum()
        ),
        "applied": bool(args.apply),
        "start": start.isoformat(),
        "end": end.isoformat(),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(args.report, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
