#!/usr/bin/env python3
"""Materialize the canonical five supportive path-label families for shorts.

This target-only sidecar reuses the exact target-free short candidate identities
and their frozen decision-time entry/ATR from the short H12 label substrate.
It reopens the same post-decision 720x1-minute OHLC path used for exact short
labels, then calls the shared side-normalized v6 auxiliary kernel.  All output
columns are supervised labels: none is a live inference feature.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.path_auxiliary_targets import (  # noqa: E402
    ALL_SUPPORTIVE_LABEL_COLUMNS,
    TARGET_COLUMNS,
    TARGET_SCHEMA,
    build_path_auxiliary_targets,
)
from scripts.materialize_packb_tp6_sl4_h12_labels import (  # noqa: E402
    _minute_path_pruned,
    _packb_to_kraken_symbol,
)


SCHEMA = "strict_r3_short_supportive_path_labels_v2_exact_entry_parity"
HORIZON_MINUTES = 12 * 60
IDENTITY = ("candidate_id", "__ts__", "__decision_ts__", "__symbol__", "side_name")
DEFAULT_LABELS = ROOT / "data_perp/artifacts/strict_r3_short_target_labels_2024_20260820_v1"
DEFAULT_MINUTES = ROOT / "data_perp/exchanges/krakenfutures/execution_1m/ohlcv"
PRIMARY_AUXILIARY_COLUMNS = (
    "__peak_mfe_return_12h__",
    "__peak_mfe_atr_12h__",
    "__time_to_first_meaningful_mfe_hours_12h__",
    "__mae_before_meaningful_mfe_atr_12h__",
    "__bars_before_price_stops_decreasing_12h__",
    "__future_slope_atr_per_hour_12h__",
    *TARGET_COLUMNS.values(),
)
VALIDITY_COLUMNS = (
    "__path_auxiliary_target_valid__",
    "__time_to_first_meaningful_mfe_target_valid__",
    "__meaningful_mfe_reached_12h__",
)
SIDECARE_TARGET_COLUMNS = tuple(dict.fromkeys([
    *PRIMARY_AUXILIARY_COLUMNS, *ALL_SUPPORTIVE_LABEL_COLUMNS,
]))
DERIVED_CONVERSION_LABEL_COLUMNS = (
    "__early_mfe_1h_atr__", "__early_mfe_2h_atr__", "__early_mfe_3h_atr__",
    "__reaches_2atr_within_12h__", "__time_to_2atr_minutes__",
    "__max_adverse_before_activation_atr__",
    "__squeeze_adjusted_mfe3h_l025__", "__squeeze_adjusted_mfe3h_l050__", "__squeeze_adjusted_mfe3h_l100__",
    "__activation_before_adverse_grade__",
)
ALL_SIDECAR_TARGET_COLUMNS = tuple(dict.fromkeys([*SIDECARE_TARGET_COLUMNS, *DERIVED_CONVERSION_LABEL_COLUMNS]))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, utc=True, errors="raise")


def _month_part(root: Path, month: pd.Timestamp) -> Path:
    path = root / "parts" / f"month={month:%Y-%m}" / "side=short.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _paths_for_group(
    minute: pd.DataFrame, decisions: pd.Series
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    starts = minute.index.get_indexer(pd.DatetimeIndex(decisions)).astype(np.int64)
    offsets = np.arange(HORIZON_MINUTES, dtype=np.int64)[None, :]
    positions = starts[:, None] + offsets
    in_range = (starts >= 0) & (positions[:, -1] < len(minute))
    safe = np.clip(positions, 0, max(len(minute) - 1, 0))
    path_open = minute.open.to_numpy(dtype=np.float64)[safe[:, 0]]
    high = minute.high.to_numpy(dtype=np.float64)[safe]
    low = minute.low.to_numpy(dtype=np.float64)[safe]
    complete = (
        in_range
        & np.isfinite(path_open)
        & np.isfinite(high).all(axis=1)
        & np.isfinite(low).all(axis=1)
    )
    return path_open, high, low, complete


def _derived_conversion_labels(*, entry: np.ndarray, high: np.ndarray, low: np.ndarray, atr_fraction: np.ndarray) -> dict[str, np.ndarray]:
    """Vectorised short-only early-path conversion labels from a complete H12 path.

    These are target-only labels.  They intentionally use the realised path,
    while the model that learns them receives only decision-time features.
    """
    atr = np.maximum(np.asarray(atr_fraction, dtype=np.float64), 1e-8)
    favourable = np.maximum(0.0, 1.0 - low / entry[:, None]) / atr[:, None]
    adverse = np.maximum(0.0, high / entry[:, None] - 1.0) / atr[:, None]
    result: dict[str, np.ndarray] = {}
    for minutes, name in ((60, "1h"), (120, "2h"), (180, "3h")):
        result[f"__early_mfe_{name}_atr__"] = np.max(favourable[:, :minutes], axis=1).astype(np.float32)
    hit2 = favourable >= 2.0
    reaches2 = hit2.any(axis=1)
    first2 = np.argmax(hit2, axis=1).astype(np.float64) + 1.0
    result["__reaches_2atr_within_12h__"] = reaches2.astype(np.float32)
    result["__time_to_2atr_minutes__"] = np.where(reaches2, first2, np.nan).astype(np.float32)
    early_fav = favourable[:, :180]
    early_adv = adverse[:, :180]
    peak_index = np.argmax(early_fav, axis=1)
    running_adverse = np.maximum.accumulate(early_adv, axis=1)
    adverse_before_peak = running_adverse[np.arange(len(entry)), peak_index]
    result["__max_adverse_before_activation_atr__"] = adverse_before_peak.astype(np.float32)
    early_mfe = np.max(early_fav, axis=1)
    for multiplier, name in ((0.25, "l025"), (0.50, "l050"), (1.0, "l100")):
        result[f"__squeeze_adjusted_mfe3h_{name}__"] = (early_mfe - multiplier * adverse_before_peak).astype(np.float32)
    activation = favourable >= 0.5
    stop = adverse >= 3.0
    activated = activation.any(axis=1)
    first_activation = np.argmax(activation, axis=1)
    first_stop = np.argmax(stop, axis=1)
    stopped = stop.any(axis=1)
    stop_before_activation = stopped & ((~activated) | (first_stop < first_activation))
    activation_adverse = np.where(activated, np.maximum.accumulate(adverse, axis=1)[np.arange(len(entry)), first_activation], np.nan)
    full_mfe = np.max(favourable, axis=1)
    grade = np.ones(len(entry), dtype=np.float32)  # no meaningful activation
    grade[stop_before_activation] = 0.0
    clean = activated & ~stop_before_activation & (activation_adverse < 0.5)
    grade[activated & ~stop_before_activation & ~clean] = 2.0
    grade[clean] = 3.0
    grade[clean & (full_mfe >= 1.5)] = 4.0
    grade[clean & (full_mfe >= 2.5)] = 5.0
    result["__activation_before_adverse_grade__"] = grade
    return result


def _materialize_month(source: pd.DataFrame, minute_root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    source = source.reset_index(drop=True)
    if source.candidate_id.duplicated().any():
        raise ValueError("short source contains duplicate candidate identities")
    if not source.side_name.astype(str).str.lower().eq("short").all():
        raise ValueError("supportive path source is not short-only")
    for column in ("__ts__", "__decision_ts__", "__label_available_at__"):
        source[column] = _utc(source[column])
    if not source["__decision_ts__"].eq(source["__ts__"] + pd.Timedelta(hours=1)).all():
        raise ValueError("short support labels require signal close + one-hour decision entries")
    expected_available = source["__decision_ts__"] + pd.Timedelta(hours=12)
    if not source["__label_available_at__"].eq(expected_available).all():
        raise ValueError("short support label availability must be decision + 12 hours")
    identity = source.loc[:, [*IDENTITY, "__label_available_at__", "label_valid", "target_invalid"]].copy()
    # Allocate every target column once.  The prior implementation inserted
    # each field into the pandas frame per symbol, creating fragmented blocks
    # and unnecessary copies.  This columnar sidecar is fixed-width and only
    # writes each group into preallocated arrays.
    values: dict[str, np.ndarray] = {
        column: np.full(len(source), np.nan, dtype=np.float32)
        for column in ALL_SIDECAR_TARGET_COLUMNS
    }
    values.update({column: np.zeros(len(source), dtype=np.int8) for column in VALIDITY_COLUMNS})
    valid = source.label_valid.astype(bool) & ~source.target_invalid.astype(bool)
    if valid.any():
        required = source.loc[valid, ["tp6_sl4_entry_price", "atr_1h"]].apply(pd.to_numeric, errors="coerce")
        if required.isna().any().any() or (required <= 0.0).any().any():
            raise ValueError("valid short label row lacks a positive frozen entry or ATR")
    entry_open_parity_rows = 0
    for symbol, group in source.loc[valid].groupby("__symbol__", sort=True):
        decisions = group["__decision_ts__"]
        minute = _minute_path_pruned(
            minute_root, _packb_to_kraken_symbol(str(symbol)),
            decisions.min(), decisions.max() + pd.Timedelta(minutes=HORIZON_MINUTES),
        )
        path_open, high, low, complete = _paths_for_group(minute, decisions)
        if not complete.all():
            failed = group.loc[~complete, "candidate_id"].astype(str).head(5).tolist()
            raise AssertionError(f"exact short supportive path missing for valid labels: {symbol}: {failed}")
        frozen_entry = pd.to_numeric(group["tp6_sl4_entry_price"], errors="coerce").to_numpy(float)
        if not np.allclose(path_open, frozen_entry, rtol=0.0, atol=1e-12):
            mismatch = group.loc[~np.isclose(path_open, frozen_entry, rtol=0.0, atol=1e-12), "candidate_id"]
            raise AssertionError(
                "frozen short entry is not the reopened decision-minute open: "
                f"{symbol}: {mismatch.astype(str).head(5).tolist()}"
            )
        entry_open_parity_rows += len(group)
        targets = build_path_auxiliary_targets(
            entry_price=frozen_entry,
            future_high=high,
            future_low=low,
            atr_fraction=(
                pd.to_numeric(group["atr_1h"], errors="coerce").to_numpy(float)
                / pd.to_numeric(group["tp6_sl4_entry_price"], errors="coerce").to_numpy(float)
            ),
            side_sign=np.full(len(group), -1.0, dtype=np.float32),
            bar_minutes=1,
            horizon_hours=12,
            include_supportive_columns=True,
        ).as_columns()
        if not np.asarray(targets["__path_auxiliary_target_valid__"], dtype=bool).all():
            raise AssertionError(f"shared short path kernel invalidated complete H12 rows for {symbol}")
        indices = group.index.to_numpy()
        for column in SIDECARE_TARGET_COLUMNS:
            values[column][indices] = np.asarray(targets[column], dtype=np.float32)
        derived = _derived_conversion_labels(
            entry=frozen_entry, high=high, low=low,
            atr_fraction=(pd.to_numeric(group["atr_1h"], errors="coerce").to_numpy(float) / frozen_entry),
        )
        for column in DERIVED_CONVERSION_LABEL_COLUMNS:
            values[column][indices] = derived[column]
        for column in VALIDITY_COLUMNS:
            values[column][indices] = np.asarray(targets[column], dtype=np.int8)
    output = pd.concat([identity, pd.DataFrame(values)], axis=1, copy=False)
    # Supervised invalidity must never become an ordinary zero-valued path.
    invalid = ~valid
    if output.loc[invalid, list(ALL_SIDECAR_TARGET_COLUMNS)].notna().any().any():
        raise AssertionError("invalid short path labels were encoded as economic target values")
    if not output.loc[valid, "__path_auxiliary_target_valid__"].astype(bool).all():
        raise AssertionError("valid short H12 labels lost path-target validity")
    record: dict[str, Any] = {
        "rows": int(len(output)),
        "source_valid_rows": int(valid.sum()),
        "source_invalid_rows": int((~valid).sum()),
        "auxiliary_valid_rows": int(output["__path_auxiliary_target_valid__"].sum()),
        "direct_reopened_entry_open_parity_rows": int(entry_open_parity_rows),
        "meaningful_mfe_rows": int(output["__meaningful_mfe_reached_12h__"].sum()),
    }
    for name, column in TARGET_COLUMNS.items():
        values = pd.to_numeric(output.loc[valid, column], errors="coerce")
        record[f"{name}_finite_rows"] = int(values.notna().sum())
        record[f"{name}_mean"] = float(values.mean())
        record[f"{name}_std"] = float(values.std(ddof=0))
    return output, record


def run(*, labels_root: Path, minute_root: Path, out: Path, start: pd.Timestamp, end: pd.Timestamp) -> Path:
    if out.exists():
        raise FileExistsError(f"immutable output already exists: {out}")
    if start.tzinfo is None or end.tzinfo is None or start.day != 1 or end.day != 1 or start.hour != 0 or end.hour != 0 or start >= end:
        raise ValueError("start/end must be increasing UTC month boundaries")
    records: list[dict[str, Any]] = []
    for month in pd.date_range(start, end, freq="MS", inclusive="left"):
        source = pd.read_parquet(
            _month_part(labels_root, month),
            columns=[
                *IDENTITY, "__label_available_at__", "label_valid", "target_invalid",
                "tp6_sl4_entry_price", "atr_1h",
            ],
        )
        sidecar, record = _materialize_month(source, minute_root)
        record["month"] = f"{month:%Y-%m}"
        destination = out / "parts" / f"month={month:%Y-%m}" / "side=short.parquet"
        destination.parent.mkdir(parents=True, exist_ok=True)
        sidecar.to_parquet(destination, index=False, compression="zstd")
        records.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
    coverage = pd.DataFrame(records)
    coverage.to_parquet(out / "coverage_by_month.parquet", index=False, compression="zstd")
    manifest = {
        "schema": SCHEMA,
        "status": "complete",
        "side": "short",
        "months": [record["month"] for record in records],
        "identity": list(IDENTITY),
        "source_labels": str(labels_root.resolve()),
        "source_labels_manifest_sha256": _sha256(labels_root / "run_manifest.json"),
        "minute_root": str(minute_root.resolve()),
        "path_kernel": TARGET_SCHEMA,
        "entry": "frozen exact one-minute open at signal close + one hour; exactly reverified against reopened decision-minute OHLCV",
        "path": "complete post-decision 720x1-minute high/low path",
        "label_available_at": "decision + 12 hours",
        "side_normalization": "side_sign=-1; favourable=entry-low and adverse=high-entry",
        "target_families": TARGET_COLUMNS,
        "supportive_columns": list(ALL_SUPPORTIVE_LABEL_COLUMNS),
        "derived_conversion_labels": list(DERIVED_CONVERSION_LABEL_COLUMNS),
        "invalidity": "source-invalid rows remain target-invalid with null support targets",
        "inference": "all materialized columns are supervised labels only and prohibited from inference features",
        "coverage": records,
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-root", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--minute-root", type=Path, default=DEFAULT_MINUTES)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--start", default="2024-01-01T00:00:00Z")
    parser.add_argument("--end", default="2024-07-01T00:00:00Z")
    args = parser.parse_args()
    print(run(
        labels_root=args.labels_root.resolve(), minute_root=args.minute_root.resolve(), out=args.out.resolve(),
        start=pd.to_datetime(args.start, utc=True), end=pd.to_datetime(args.end, utc=True),
    ))


if __name__ == "__main__":
    main()
