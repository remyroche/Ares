#!/usr/bin/env python3
"""Augment existing causal label ledgers with 12-hour path targets only.

This deliberately bypasses the full label pipeline.  The source label ledger
owns row identity and the ATR value, while the canonical hourly OHLCV store
owns the post-decision path.  No market features, strategy masks, or policy
rollouts are recomputed.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.data_store import PartitionedOHLCVStore  # noqa: E402
from extreme_price_movements.path_auxiliary_targets import (  # noqa: E402
    ALL_SUPPORTIVE_LABEL_COLUMNS,
    TARGET_COLUMNS,
    TARGET_SCHEMA,
    build_path_auxiliary_targets,
)

SCHEMA = "materialize_path_auxiliary_targets_v1_target_only"
IDENTITY_COLUMNS = ("__ts__", "__symbol__", "side_name")
LABEL_RESOLUTION_COLUMN = "__label_end_ts__"
VALIDITY_COLUMNS = {
    "__path_auxiliary_target_valid__",
    "__time_to_first_meaningful_mfe_target_valid__",
    "__meaningful_mfe_reached_12h__",
}
RAW_TARGET_COLUMNS = (
    "__peak_mfe_return_12h__",
    "__peak_mfe_atr_12h__",
    "__time_to_first_meaningful_mfe_hours_12h__",
    "__mae_before_meaningful_mfe_atr_12h__",
    "__bars_before_price_stops_decreasing_12h__",
    "__future_slope_atr_per_hour_12h__",
)
OUTPUT_TARGET_COLUMNS = tuple(
    dict.fromkeys(
        (
            *RAW_TARGET_COLUMNS,
            *TARGET_COLUMNS.values(),
            *ALL_SUPPORTIVE_LABEL_COLUMNS,
            *sorted(VALIDITY_COLUMNS),
        )
    )
)


@dataclass(frozen=True)
class SymbolBars:
    index_ns: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray


def _utc_ns(values: Iterable[Any]) -> np.ndarray:
    return pd.to_datetime(values, utc=True, errors="coerce").astype("int64").to_numpy()


def _target_output(rows: int) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for column in OUTPUT_TARGET_COLUMNS:
        if column in VALIDITY_COLUMNS:
            output[column] = np.zeros(rows, dtype=np.int8)
        else:
            output[column] = np.full(rows, np.nan, dtype=np.float32)
    # Hourly bars are timestamped at bar open.  A 12-bar path beginning at the
    # decision timestamp is fully observable only when the twelfth bar closes,
    # i.e. decision_ts + 12h.  Persist this separately from the numeric target
    # arrays so downstream purging never falls back to an older base-label
    # horizon.
    output[LABEL_RESOLUTION_COLUMN] = np.full(rows, np.datetime64("NaT"), dtype="datetime64[ns]")
    return output


def materialize_batch_targets(
    frame: pd.DataFrame,
    bars_by_symbol: dict[str, SymbolBars],
    *,
    decision_delay_hours: int = 1,
    horizon_hours: int = 12,
) -> dict[str, np.ndarray]:
    """Return target columns aligned one-for-one with ``frame`` rows."""

    required = {"__ts__", "__symbol__", "__path_auxiliary_atr_fraction__"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"label batch missing required columns: {sorted(missing)}")
    rows = len(frame)
    output = _target_output(rows)
    if rows == 0:
        return output

    signal_ns = _utc_ns(frame["__ts__"])
    decision_ns = signal_ns + int(pd.Timedelta(hours=decision_delay_hours).value)
    symbols = frame["__symbol__"].astype(str).to_numpy()
    atr = pd.to_numeric(
        frame["__path_auxiliary_atr_fraction__"], errors="coerce"
    ).to_numpy(np.float64)
    if "side_name" in frame:
        side_text = frame["side_name"].astype(str).str.lower().to_numpy()
    elif "side" in frame:
        raw_side = frame["side"].to_numpy()
        side_text = np.where(
            pd.to_numeric(pd.Series(raw_side), errors="coerce").fillna(0).to_numpy()
            < 0,
            "short",
            "long",
        )
    else:
        raise ValueError("label batch needs side_name or side")
    hour_ns = int(pd.Timedelta(hours=1).value)
    horizon = int(horizon_hours)

    for symbol in pd.unique(symbols):
        bars = bars_by_symbol.get(str(symbol))
        if bars is None or len(bars.index_ns) < horizon:
            continue
        local_rows = np.flatnonzero(symbols == symbol)
        positions = np.searchsorted(bars.index_ns, decision_ns[local_rows])
        bounded = np.minimum(positions, len(bars.index_ns) - 1)
        exact = (positions < len(bars.index_ns)) & (
            bars.index_ns[bounded] == decision_ns[local_rows]
        )
        mature = exact & ((positions + horizon) <= len(bars.index_ns))
        if not np.any(mature):
            continue
        selected_rows = local_rows[mature]
        pos = positions[mature]
        high_windows = np.lib.stride_tricks.sliding_window_view(
            bars.high, horizon
        )[pos]
        low_windows = np.lib.stride_tricks.sliding_window_view(bars.low, horizon)[
            pos
        ]
        if horizon > 1:
            contiguous_windows = np.lib.stride_tricks.sliding_window_view(
                np.diff(bars.index_ns) == hour_ns, horizon - 1
            )[pos]
            contiguous = np.all(contiguous_windows, axis=1)
        else:
            contiguous = np.ones(len(pos), dtype=bool)
        if not np.any(contiguous):
            continue
        selected_rows = selected_rows[contiguous]
        pos = pos[contiguous]
        output[LABEL_RESOLUTION_COLUMN][selected_rows] = (
            decision_ns[selected_rows] + horizon * hour_ns
        ).astype("datetime64[ns]")
        side_sign = np.where(side_text[selected_rows] == "short", -1.0, 1.0)
        targets = build_path_auxiliary_targets(
            entry_price=bars.open[pos],
            future_high=high_windows[contiguous],
            future_low=low_windows[contiguous],
            atr_fraction=atr[selected_rows],
            side_sign=side_sign,
            bar_minutes=60,
            horizon_hours=horizon,
        )
        for column, values in targets.as_columns().items():
            if column not in output:
                continue
            output[column][selected_rows] = values
    return output


def _scan_sources(
    sources: list[Path], *, start: pd.Timestamp | None, end: pd.Timestamp | None
) -> tuple[list[str], pd.Timestamp, pd.Timestamp]:
    symbols: set[str] = set()
    minimum: pd.Timestamp | None = None
    maximum: pd.Timestamp | None = None
    for source in sources:
        parquet = pq.ParquetFile(source)
        for batch in parquet.iter_batches(
            batch_size=250_000, columns=["__ts__", "__symbol__"]
        ):
            frame = batch.to_pandas()
            ts = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
            mask = ts.notna()
            if start is not None:
                mask &= ts >= start
            if end is not None:
                mask &= ts < end
            if not mask.any():
                continue
            kept_ts = ts[mask]
            symbols.update(frame.loc[mask, "__symbol__"].dropna().astype(str))
            batch_min = kept_ts.min()
            batch_max = kept_ts.max()
            minimum = batch_min if minimum is None else min(minimum, batch_min)
            maximum = batch_max if maximum is None else max(maximum, batch_max)
    if minimum is None or maximum is None or not symbols:
        raise ValueError("no source rows remain after date filtering")
    return sorted(symbols), minimum, maximum


def _load_bars(
    root: Path,
    symbols: list[str],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    decision_delay_hours: int,
    horizon_hours: int,
) -> dict[str, SymbolBars]:
    store = PartitionedOHLCVStore(root_dir=str(root), timeframe="1h")
    read_start = start + pd.Timedelta(hours=decision_delay_hours)
    read_end = end + pd.Timedelta(hours=decision_delay_hours + horizon_hours)
    output: dict[str, SymbolBars] = {}
    for index, symbol in enumerate(symbols, start=1):
        frame = store.load(
            symbol,
            columns=["open", "high", "low"],
            start_ts=read_start,
            end_ts=read_end,
        )
        if frame.empty:
            continue
        frame = frame.sort_index()
        index_ns = _utc_ns(frame.index)
        finite_ts = index_ns != np.iinfo(np.int64).min
        if not np.any(finite_ts):
            continue
        output[symbol] = SymbolBars(
            index_ns=index_ns[finite_ts],
            open=pd.to_numeric(frame["open"], errors="coerce")
            .to_numpy(np.float64)[finite_ts],
            high=pd.to_numeric(frame["high"], errors="coerce")
            .to_numpy(np.float64)[finite_ts],
            low=pd.to_numeric(frame["low"], errors="coerce")
            .to_numpy(np.float64)[finite_ts],
        )
        if index % 25 == 0 or index == len(symbols):
            print(
                f"Loaded canonical hourly bars: {index}/{len(symbols)} "
                f"symbols ({len(output)} non-empty)",
                flush=True,
            )
    return output


def _json_safe(value: Any) -> Any:
    if isinstance(value, (Path, pd.Timestamp)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def materialize(
    source_dir: Path,
    output_dir: Path,
    *,
    ohlcv_root: Path,
    start: str | None,
    end: str | None,
    batch_rows: int,
    decision_delay_hours: int,
    horizon_hours: int,
    verify_existing_heads: bool,
) -> dict[str, Any]:
    sources = sorted(source_dir.glob("train_global_*.parquet"))
    if not sources:
        raise FileNotFoundError(f"no train_global parquet files under {source_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {output_dir}")
    labels_dir = output_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    start_ts = pd.Timestamp(start, tz="UTC") if start else None
    end_ts = pd.Timestamp(end, tz="UTC") if end else None
    symbols, minimum, maximum = _scan_sources(
        sources, start=start_ts, end=end_ts
    )
    bars = _load_bars(
        ohlcv_root,
        symbols,
        start=minimum,
        end=maximum,
        decision_delay_hours=decision_delay_hours,
        horizon_hours=horizon_hours,
    )

    datasets: dict[str, Any] = {}
    target_names = {*OUTPUT_TARGET_COLUMNS, LABEL_RESOLUTION_COLUMN}
    for source in sources:
        output_path = labels_dir / source.name
        writer: pq.ParquetWriter | None = None
        rows_written = 0
        max_peak_delta = 0.0
        max_timing_delta = 0.0
        parquet = pq.ParquetFile(source)
        try:
            for batch_number, batch in enumerate(
                parquet.iter_batches(batch_size=int(batch_rows)), start=1
            ):
                frame = batch.to_pandas()
                ts = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
                keep = ts.notna()
                if start_ts is not None:
                    keep &= ts >= start_ts
                if end_ts is not None:
                    keep &= ts < end_ts
                if not keep.any():
                    continue
                frame = frame.loc[keep].reset_index(drop=True)
                previous_peak = pd.to_numeric(
                    frame["__peak_mfe_atr_12h__"], errors="coerce"
                ).to_numpy(np.float64)
                previous_timing = pd.to_numeric(
                    frame["__time_to_first_meaningful_mfe_hours_12h__"],
                    errors="coerce",
                ).to_numpy(np.float64)
                targets = materialize_batch_targets(
                    frame,
                    bars,
                    decision_delay_hours=decision_delay_hours,
                    horizon_hours=horizon_hours,
                )
                frame = frame.drop(
                    columns=[column for column in target_names if column in frame],
                    errors="ignore",
                )
                for column, values in targets.items():
                    frame[column] = values
                new_peak = targets["__peak_mfe_atr_12h__"]
                new_timing = targets["__time_to_first_meaningful_mfe_hours_12h__"]
                peak_valid = np.isfinite(previous_peak) & np.isfinite(new_peak)
                timing_valid = np.isfinite(previous_timing) & np.isfinite(new_timing)
                if peak_valid.any():
                    max_peak_delta = max(
                        max_peak_delta,
                        float(np.max(np.abs(previous_peak[peak_valid] - new_peak[peak_valid]))),
                    )
                if timing_valid.any():
                    max_timing_delta = max(
                        max_timing_delta,
                        float(
                            np.max(
                                np.abs(
                                    previous_timing[timing_valid]
                                    - new_timing[timing_valid]
                                )
                            )
                        ),
                    )
                if verify_existing_heads:
                    if peak_valid.any() and float(
                        np.max(np.abs(previous_peak[peak_valid] - new_peak[peak_valid]))
                    ) > 1e-5:
                        raise ValueError(
                            f"{source.name} recomputed peak-MFE target differs from source"
                        )
                    if timing_valid.any() and float(
                        np.max(
                            np.abs(
                                previous_timing[timing_valid]
                                - new_timing[timing_valid]
                            )
                        )
                    ) > 1e-6:
                        raise ValueError(
                            f"{source.name} recomputed timing target differs from source"
                        )
                table = pa.Table.from_pandas(frame, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(
                        output_path, table.schema, compression="zstd"
                    )
                writer.write_table(table)
                rows_written += len(frame)
                if batch_number % 10 == 0:
                    print(
                        f"{source.name}: batches={batch_number} rows={rows_written:,}",
                        flush=True,
                    )
        finally:
            if writer is not None:
                writer.close()
        if rows_written == 0:
            raise ValueError(f"no rows written for {source}")
        schema_names = pq.read_schema(output_path).names
        datasets[source.stem] = {
            "file": source.name,
            "rows": rows_written,
            "columns": schema_names,
            "max_abs_delta_vs_source_peak_mfe_atr": max_peak_delta,
            "max_abs_delta_vs_source_time_to_meaningful_mfe_hours": max_timing_delta,
        }

    manifest = {
        "schema": SCHEMA,
        "target_schema": TARGET_SCHEMA,
        "source_labels": str(source_dir),
        "canonical_ohlcv_root": str(ohlcv_root),
        "utc_identity": list(IDENTITY_COLUMNS),
        "signal_timestamp_column": "__ts__",
        "decision_timestamp_contract": "__ts__ + 1h",
        "first_path_timestamp_contract": "decision_timestamp",
        "decision_delay_hours": decision_delay_hours,
        "horizon_hours": horizon_hours,
        "start": minimum,
        "end": maximum,
        "symbols_requested": len(symbols),
        "symbols_loaded": len(bars),
        "datasets": datasets,
        "target_columns": list(OUTPUT_TARGET_COLUMNS),
        "label_resolution_column": LABEL_RESOLUTION_COLUMN,
        "label_resolution_contract": (
            "decision_timestamp + horizon_hours; hourly bar timestamps denote bar open"
        ),
        "verify_existing_heads": bool(verify_existing_heads),
    }
    (labels_dir / "labels_manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True) + "\n"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--ohlcv-root",
        type=Path,
        default=Path("data_perp/exchanges/krakenfutures"),
    )
    parser.add_argument("--start", default="2025-02-01T00:00:00Z")
    parser.add_argument("--end", default=None)
    parser.add_argument("--batch-rows", type=int, default=100_000)
    parser.add_argument("--decision-delay-hours", type=int, default=1)
    parser.add_argument("--horizon-hours", type=int, default=12)
    parser.add_argument(
        "--no-verify-existing-heads",
        action="store_false",
        dest="verify_existing_heads",
        help="Do not fail when recomputed peak/timing targets differ from source.",
    )
    parser.set_defaults(verify_existing_heads=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = materialize(
        args.source_labels,
        args.output_dir,
        ohlcv_root=args.ohlcv_root,
        start=args.start,
        end=args.end,
        batch_rows=args.batch_rows,
        decision_delay_hours=args.decision_delay_hours,
        horizon_hours=args.horizon_hours,
        verify_existing_heads=args.verify_existing_heads,
    )
    print(json.dumps(_json_safe(manifest), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
