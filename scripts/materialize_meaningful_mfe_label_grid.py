#!/usr/bin/env python3
"""Materialize a fixed 12h/24h meaningful-MFE label grid from hourly paths."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.data_store import PartitionedOHLCVStore  # noqa: E402
from extreme_price_movements.meaningful_mfe_label_grid import (  # noqa: E402
    SCHEMA as LABEL_SCHEMA,
    MeaningfulMFEGridSpec,
    build_meaningful_mfe_grid_labels,
)


SCHEMA = "materialize_meaningful_mfe_label_grid_v1"
IDENTITY = ("__ts__", "__symbol__", "side_name", "candidate_id")


@dataclass(frozen=True)
class HourlyBars:
    timestamp_ns: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(_safe(payload), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _timestamp_ns(values: Sequence[Any]) -> np.ndarray:
    return pd.to_datetime(values, utc=True, errors="raise").astype("int64").to_numpy()


def load_hourly_bars(
    root: Path,
    symbols: Sequence[str],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    max_horizon_hours: int,
) -> dict[str, HourlyBars]:
    store = PartitionedOHLCVStore(root_dir=str(root), timeframe="1h")
    output: dict[str, HourlyBars] = {}
    for index, symbol in enumerate(symbols, start=1):
        frame = store.load(
            symbol,
            columns=["open", "high", "low", "close"],
            start_ts=start,
            end_ts=end + pd.Timedelta(hours=max_horizon_hours + 1),
        )
        if frame.empty:
            continue
        frame = frame.sort_index()
        output[str(symbol)] = HourlyBars(
            timestamp_ns=_timestamp_ns(frame.index),
            open=pd.to_numeric(frame["open"], errors="coerce").to_numpy(np.float64),
            high=pd.to_numeric(frame["high"], errors="coerce").to_numpy(np.float64),
            low=pd.to_numeric(frame["low"], errors="coerce").to_numpy(np.float64),
            close=pd.to_numeric(frame["close"], errors="coerce").to_numpy(np.float64),
        )
        if index % 25 == 0 or index == len(symbols):
            print(
                f"loaded hourly bars {index}/{len(symbols)} "
                f"({len(output)} non-empty)",
                flush=True,
            )
    return output


def materialize_symbol_labels(
    frame: pd.DataFrame,
    bars: HourlyBars,
    specs: Sequence[MeaningfulMFEGridSpec],
    *,
    decision_column: str,
    atr_column: str,
) -> pd.DataFrame:
    """Return valid, exact, contiguous path labels for one symbol."""

    if frame.empty:
        return pd.DataFrame()
    max_horizon = max(spec.horizon_hours for spec in specs)
    decision_ns = _timestamp_ns(frame[decision_column])
    positions = np.searchsorted(bars.timestamp_ns, decision_ns)
    bounded = np.minimum(positions, max(len(bars.timestamp_ns) - 1, 0))
    exact = (positions < len(bars.timestamp_ns)) & (
        bars.timestamp_ns[bounded] == decision_ns
    )
    mature = exact & ((positions + max_horizon) <= len(bars.timestamp_ns))
    if max_horizon > 1 and mature.any():
        hour_ns = int(pd.Timedelta(hours=1).value)
        contiguous_windows = np.lib.stride_tricks.sliding_window_view(
            np.diff(bars.timestamp_ns) == hour_ns,
            max_horizon - 1,
        )
        candidate = np.flatnonzero(mature)
        mature[candidate] &= contiguous_windows[positions[candidate]].all(axis=1)
    selected = np.flatnonzero(mature)
    if not len(selected):
        return pd.DataFrame()
    position = positions[selected]
    high = np.lib.stride_tricks.sliding_window_view(
        bars.high, max_horizon
    )[position]
    low = np.lib.stride_tricks.sliding_window_view(
        bars.low, max_horizon
    )[position]
    close = np.lib.stride_tricks.sliding_window_view(
        bars.close, max_horizon
    )[position]
    atr = pd.to_numeric(frame.iloc[selected][atr_column], errors="coerce").to_numpy(float)
    side = np.where(
        frame.iloc[selected]["side_name"].astype(str).str.lower().eq("short"),
        -1.0,
        1.0,
    )
    base_columns = [
        *IDENTITY,
        decision_column,
        atr_column,
        *[
            column
            for column in ("execution_net_ev_12h", "execution_gross_ev_12h")
            if column in frame
        ],
    ]
    base = frame.iloc[selected].loc[:, list(dict.fromkeys(base_columns))].reset_index(drop=True)
    parts: list[pd.DataFrame] = []
    for spec in specs:
        labels = build_meaningful_mfe_grid_labels(
            entry_price=bars.open[position],
            future_high=high[:, : spec.horizon_hours],
            future_low=low[:, : spec.horizon_hours],
            future_close=close[:, : spec.horizon_hours],
            atr_fraction=atr,
            side_sign=side,
            spec=spec,
        )
        labels["label_resolution_utc"] = pd.to_datetime(
            decision_ns[selected] + int(pd.Timedelta(hours=spec.horizon_hours).value),
            utc=True,
        )
        parts.append(pd.concat([base, labels], axis=1))
    return pd.concat(parts, ignore_index=True)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    source = pd.read_parquet(args.input)
    required = [*IDENTITY, args.decision_column, args.atr_column]
    missing = [column for column in required if column not in source]
    if missing:
        raise ValueError("label-grid source is missing columns: " + ", ".join(missing))
    if source.duplicated(list(IDENTITY)).any():
        raise ValueError("label-grid source identities must be unique")
    source[args.decision_column] = pd.to_datetime(
        source[args.decision_column], utc=True, errors="raise"
    )
    source = source.sort_values(
        [args.decision_column, "__symbol__", "side_name"], kind="mergesort"
    ).reset_index(drop=True)
    specs = [
        MeaningfulMFEGridSpec(
            horizon_hours=horizon,
            upper_atr=upper,
            upper_return_floor=float(args.upper_return_floor),
            lower_atr=float(args.lower_atr),
            temperature=float(args.temperature),
            round_trip_cost=float(args.round_trip_cost),
        )
        for horizon in args.horizon_hours
        for upper in args.upper_atr
    ]
    bars = load_hourly_bars(
        args.ohlcv_root,
        sorted(source["__symbol__"].astype(str).unique()),
        start=source[args.decision_column].min(),
        end=source[args.decision_column].max(),
        max_horizon_hours=max(spec.horizon_hours for spec in specs),
    )
    parts: list[pd.DataFrame] = []
    for symbol, positions in source.groupby("__symbol__", sort=True).indices.items():
        if str(symbol) not in bars:
            continue
        labels = materialize_symbol_labels(
            source.iloc[np.asarray(positions, dtype=int)].reset_index(drop=True),
            bars[str(symbol)],
            specs,
            decision_column=args.decision_column,
            atr_column=args.atr_column,
        )
        if not labels.empty:
            parts.append(labels)
    if not parts:
        raise ValueError("no complete exact hourly paths were materialized")
    labels = pd.concat(parts, ignore_index=True)
    args.output_dir.mkdir(parents=True)
    output_path = args.output_dir / "meaningful_mfe_label_grid.parquet"
    labels.to_parquet(output_path, index=False)
    coverage = (
        labels.groupby("grid_name", observed=True)
        .agg(
            rows=("candidate_id", "size"),
            valid_rows=("label_valid", "sum"),
            favorable_first_rate=("favorable_first", "mean"),
            adverse_first_rate=("adverse_first", "mean"),
            timeout_rate=("timeout", "mean"),
            mean_soft_label=("soft_label", "mean"),
            mean_early_path_quality=("early_3bar_path_quality", "mean"),
            mean_time_to_80pct_mfe_hours=("time_to_80pct_mfe_hours", "mean"),
            reaches_80pct_economic_barrier_rate=(
                "reaches_80pct_economic_barrier",
                "mean",
            ),
            mean_time_to_80pct_economic_barrier_hours=(
                "time_to_80pct_economic_barrier_hours",
                "mean",
            ),
            mean_economic_barrier_time_quality=(
                "economic_barrier_time_quality",
                "mean",
            ),
            mean_future_slope_atr_per_hour_clip_10=(
                "future_close_slope_atr_per_hour_clip_10",
                "mean",
            ),
        )
        .reset_index()
    )
    coverage["source_rows"] = len(source)
    coverage["path_coverage_rate"] = coverage["rows"] / len(source)
    coverage_path = args.output_dir / "label_grid_coverage.csv"
    coverage.to_csv(coverage_path, index=False)
    manifest = {
        "schema": SCHEMA,
        "label_schema": LABEL_SCHEMA,
        "status": "materialized_research_labels_not_model_evidence",
        "contract": {
            "identity": list(IDENTITY),
            "decision_timestamp": args.decision_column,
            "path_start": "hourly bar whose open timestamp equals execution decision",
            "path_requirement": "exact timestamp and contiguous complete hourly horizon",
            "conflict": "same-hour favorable/adverse touch is adverse",
            "cost": "reported once as favorable barrier minus round-trip cost; event labels do not replay execution",
            "label_use": "targets/support labels only; never same-row inference features",
        },
        "input": {
            "path": str(args.input),
            "sha256": _sha256(args.input),
            "rows": int(len(source)),
        },
        "ohlcv_root": str(args.ohlcv_root),
        "specs": [asdict(spec) | {"name": spec.name} for spec in specs],
        "symbols_requested": int(source["__symbol__"].nunique()),
        "symbols_loaded": int(len(bars)),
        "outputs": {
            "labels": {"path": str(output_path), "sha256": _sha256(output_path)},
            "coverage": {"path": str(coverage_path), "sha256": _sha256(coverage_path)},
        },
    }
    _write_json(args.output_dir / "manifest.json", manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--decision-column", default="execution_decision_utc")
    parser.add_argument("--atr-column", default="oof_entry_atr_fraction")
    parser.add_argument("--ohlcv-root", type=Path, default=Path("data_perp/exchanges/krakenfutures"))
    parser.add_argument("--horizon-hours", type=int, action="append", default=[])
    parser.add_argument("--upper-atr", type=float, action="append", default=[])
    parser.add_argument("--upper-return-floor", type=float, default=0.015)
    parser.add_argument("--lower-atr", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.35)
    parser.add_argument("--round-trip-cost", type=float, default=0.01)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not args.horizon_hours:
        args.horizon_hours = [12, 24]
    if not args.upper_atr:
        args.upper_atr = [1.5, 2.0]
    return args


if __name__ == "__main__":
    print(json.dumps(run(_parser()), indent=2, default=str))
