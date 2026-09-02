#!/usr/bin/env python3
"""Materialise exact short O hazard/competing-risk labels from 1-minute paths.

The producer is target-only.  It joins the immutable target-free short P0
identities to their frozen decision-minute entry/ATR labels and reopens the
same complete post-entry 720 x one-minute Kraken path used by the rich-policy
labels.  No output field is an inference feature.

Favourable means ``entry - low > 250 bps``.  Adverse means
``high - entry >= {1.5, 2.0, 3.0} ATR``.  A same-minute touch is conservatively
adverse-first for each competing-risk contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import run_strict_r3_short_p0_oc_k0_round3_c_targets as r3  # noqa: E402
from scripts.materialize_packb_tp6_sl4_h12_labels import (  # noqa: E402
    _minute_path_pruned as _legacy_minute_path_pruned,
    _overlapping_minute_fragments,
    _packb_to_kraken_symbol,
)
from scripts.materialize_strict_r3_short_p0_rich_path_labels import HORIZON_MINUTES, _path_matrices  # noqa: E402


SCHEMA = "strict_r3_short_p0_oc_k0_event_timing_labels_v1"
SIDE = "short"
FAVOURABLE_BPS = 250.0
ADVERSE_ATR = (1.5, 2.0, 3.0)
PATH_READ_WORKERS = 4
START = pd.Timestamp("2024-05-01T00:00:00Z")
END = pd.Timestamp("2026-08-01T00:00:00Z")
RICH_ROOT = ROOT / "data_perp/artifacts/strict_r3_short_p0_rich_path_labels_apr2024_jul2026_20260821_v1"
MINUTE_ROOT = ROOT / "data_perp/exchanges/krakenfutures/execution_1m/ohlcv"
OUT = ROOT / "data_perp/artifacts/strict_r3_short_p0_oc_k0_event_timing_labels_202405_202607_20260822_v1"
IDENTITY = ("candidate_id", "__ts__", "__decision_ts__", "__symbol__", "side_name")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _first(
    values: np.ndarray, threshold: np.ndarray | float, *, inclusive: bool = True
) -> np.ndarray:
    """One-based first hit minute, or NaN where never hit.

    The opportunity contract is deliberately strict: favourable movement must
    be *greater than* 250 bps.  Adverse barriers retain the conventional
    greater-than-or-equal first-touch definition.  Keeping the comparison
    explicit prevents a quiet semantic drift at the exact boundary.
    """
    boundary = np.asarray(threshold, dtype=float)
    hit = values >= boundary if inclusive else values > boundary
    any_hit = hit.any(axis=1)
    return np.where(any_hit, np.argmax(hit, axis=1).astype(float) + 1.0, np.nan)


def _derive_labels(
    *, entry: np.ndarray, atr: np.ndarray, high: np.ndarray, low: np.ndarray
) -> dict[str, np.ndarray]:
    """Vectorised side-normalised 6-hour favourable/adverse first touches."""
    entry = np.asarray(entry, dtype=float)
    atr = np.asarray(atr, dtype=float)
    high = np.asarray(high, dtype=float)[:, :360]
    low = np.asarray(low, dtype=float)[:, :360]
    favourable_bps = np.maximum(0.0, 1.0 - low / entry[:, None]) * 10_000.0
    adverse_atr = np.maximum(0.0, high / entry[:, None] - 1.0) / np.maximum(atr[:, None] / entry[:, None], 1e-12)
    # The extra micro-bps is numerical protection for ratios such as
    # ``1 - 97.5 / 100`` that are mathematically 250 bps but marginally above
    # it in binary floating point.  It is not an economic buffer.
    first_favourable = _first(favourable_bps, FAVOURABLE_BPS + 1e-6, inclusive=True)
    result: dict[str, np.ndarray] = {
        "first_favourable_250bps_minute": first_favourable.astype(np.float32),
        "favourable_hit_1h": (np.isfinite(first_favourable) & (first_favourable <= 60.0)).astype(np.int8),
        "favourable_hit_2h": (np.isfinite(first_favourable) & (first_favourable <= 120.0)).astype(np.int8),
        "favourable_hit_4h": (np.isfinite(first_favourable) & (first_favourable <= 240.0)).astype(np.int8),
        "favourable_hit_6h": np.isfinite(first_favourable).astype(np.int8),
    }
    for threshold in ADVERSE_ATR:
        tag = f"{threshold:.1f}".replace(".", "p")
        first_adverse = _first(adverse_atr, threshold)
        favourable_first = np.isfinite(first_favourable) & (~np.isfinite(first_adverse) | (first_favourable < first_adverse))
        # Tie belongs to adverse-first: OHLC bars cannot prove intra-bar order.
        adverse_first = np.isfinite(first_adverse) & (~np.isfinite(first_favourable) | (first_adverse <= first_favourable))
        event = np.zeros(len(entry), dtype=np.int8)
        event[favourable_first] = 1
        event[adverse_first] = 2
        result[f"first_adverse_{tag}atr_minute"] = first_adverse.astype(np.float32)
        result[f"event_{tag}atr"] = event
        result[f"favourable_first_{tag}atr"] = favourable_first.astype(np.int8)
        result[f"adverse_first_{tag}atr"] = adverse_first.astype(np.int8)
        result[f"censored_{tag}atr"] = (event == 0).astype(np.int8)
    return result


def _utc_timestamp(value: object) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _row_groups_for_window(reader: pq.ParquetFile, start: pd.Timestamp, end_exclusive: pd.Timestamp) -> list[int] | None:
    """Return timestamp-overlapping parquet row groups, or ``None`` safely.

    ``None`` explicitly means that the immutable file has no usable timestamp
    statistics; callers then read the whole fragment, preserving the original
    reader's exact semantics.
    """
    try:
        column = reader.schema_arrow.get_field_index("ts")
        if column < 0:
            return None
        selected: list[int] = []
        for index in range(reader.num_row_groups):
            stats = reader.metadata.row_group(index).column(column).statistics
            if stats is None or not stats.has_min_max:
                return None
            low = _utc_timestamp(stats.min)
            high = _utc_timestamp(stats.max)
            if high >= start and low < end_exclusive:
                selected.append(index)
        return selected
    except (OSError, pa.ArrowInvalid, TypeError, ValueError, OverflowError):
        return None


def _minute_path_pruned(root: Path, symbol: str, start: pd.Timestamp, end_exclusive: pd.Timestamp) -> pd.DataFrame:
    """Exact minute reader with safe parquet row-group pruning.

    The fragment list, columns, timestamp predicate, deduplication order and
    final complete grid are byte-for-byte equivalent to the established
    materialiser.  Only row groups whose immutable statistics cannot overlap
    the requested window are skipped.  Files without usable statistics retain
    the legacy full-fragment path.
    """
    fragments = _overlapping_minute_fragments(root, symbol, start, end_exclusive)
    tables: list[pa.Table] = []
    unreadable: list[str] = []
    for path in fragments:
        try:
            reader = pq.ParquetFile(path)
            groups = _row_groups_for_window(reader, start, end_exclusive)
            if groups == []:
                continue
            if groups is None:
                table = reader.read(columns=["ts", "open", "high", "low", "close"])
            else:
                table = reader.read_row_groups(groups, columns=["ts", "open", "high", "low", "close"])
            tables.append(table)
        except (OSError, pa.ArrowInvalid) as exc:
            unreadable.append(f"{path.name}: {type(exc).__name__}")
    if tables:
        raw = pa.concat_tables(tables, promote_options="permissive").to_pandas()
        raw["ts"] = pd.to_datetime(raw["ts"], utc=True, errors="raise")
        raw = raw.loc[raw["ts"].ge(start) & raw["ts"].lt(end_exclusive)]
        raw = raw.drop_duplicates("ts", keep="last").set_index("ts").sort_index()
    else:
        raw = pd.DataFrame(columns=["open", "high", "low", "close"], index=pd.DatetimeIndex([], tz="UTC"))
    grid = pd.date_range(start.floor("min"), (end_exclusive - pd.Timedelta(minutes=1)).floor("min"), freq="min", tz="UTC")
    result = raw.reindex(grid)
    result.attrs["unreadable_minute_fragments"] = tuple(unreadable)
    return result


def _h12_by_month() -> dict[str, Path]:
    payload = json.loads((RICH_ROOT / "run_manifest.json").read_text())
    roots = [Path(value) for value in payload["sources"]["h12_roots"]]
    result: dict[str, Path] = {}
    for root in roots:
        for part in root.glob("parts/month=*/side=short.parquet"):
            month = part.parent.name.removeprefix("month=")
            if month in result:
                raise AssertionError(f"duplicate H12 source for {month}")
            result[month] = part
    return result


def _month_input(frame: pd.DataFrame, h12_parts: dict[str, Path], month: pd.Timestamp) -> pd.DataFrame:
    key = f"{month:%Y-%m}"
    held = frame.loc[frame["__decision_ts__"].dt.strftime("%Y-%m").eq(key)].copy()
    if held.empty:
        return held
    part = h12_parts.get(key)
    if part is None:
        raise FileNotFoundError(f"H12 label part missing for {key}")
    h12 = pd.read_parquet(part, columns=[
        *IDENTITY, "tp6_sl4_entry_price", "atr_1h", "label_valid", "target_invalid",
    ])
    output = held.loc[:, [*IDENTITY, "__label_available_at__", "rich_path_label_valid", "rich_path_target_invalid", "policy_path_valid", "policy_net_bps"]].merge(
        h12, on=list(IDENTITY), how="left", validate="one_to_one",
    )
    if len(output) != len(held) or output.candidate_id.duplicated().any():
        raise AssertionError(f"target-free identity mismatch while joining H12 {key}")
    output["__decision_ts__"] = pd.to_datetime(output["__decision_ts__"], utc=True)
    output["__label_available_at__"] = pd.to_datetime(output["__label_available_at__"], utc=True)
    expected = output["__decision_ts__"] + pd.Timedelta(hours=12)
    if not output["__label_available_at__"].eq(expected).all():
        raise AssertionError(f"{key}: label availability is not decision + H12")
    if not output.side_name.astype(str).str.lower().eq(SIDE).all():
        raise AssertionError(f"{key}: non-short label source")
    return output.reset_index(drop=True)


def _blank(source: pd.DataFrame) -> pd.DataFrame:
    out = source.loc[:, [*IDENTITY, "__label_available_at__", "policy_net_bps"]].copy()
    out["event_timing_label_valid"] = False
    out["event_timing_target_invalid"] = True
    numeric = ("first_favourable_250bps_minute", *[f"first_adverse_{threshold:.1f}".replace(".", "p") + "atr_minute" for threshold in ADVERSE_ATR])
    integer = (
        "favourable_hit_1h", "favourable_hit_2h", "favourable_hit_4h", "favourable_hit_6h",
        *[f"{name}_{threshold:.1f}".replace(".", "p") + "atr" for threshold in ADVERSE_ATR for name in ("event", "favourable_first", "adverse_first", "censored")],
    )
    for name in numeric:
        out[name] = np.nan
    for name in integer:
        out[name] = np.nan
    return out


def _materialize_month(source: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = _blank(source)
    valid = (
        source["rich_path_label_valid"].fillna(False).astype(bool)
        & ~source["rich_path_target_invalid"].fillna(True).astype(bool)
        & source["policy_path_valid"].fillna(False).astype(bool)
        & source["label_valid"].fillna(False).astype(bool)
        & ~source["target_invalid"].fillna(True).astype(bool)
        & pd.to_numeric(source["tp6_sl4_entry_price"], errors="coerce").gt(0.0)
        & pd.to_numeric(source["atr_1h"], errors="coerce").gt(0.0)
    )
    parity_rows = 0
    for symbol, part in source.loc[valid].groupby("__symbol__", sort=True):
        idx = part.index.to_numpy()
        decisions = pd.to_datetime(part["__decision_ts__"], utc=True)
        minute = _minute_path_pruned(
            MINUTE_ROOT, _packb_to_kraken_symbol(str(symbol)),
            decisions.min(), decisions.max() + pd.Timedelta(minutes=HORIZON_MINUTES),
        )
        opened, high, low, _close, complete = _path_matrices(minute, decisions)
        if not complete.all():
            missed = part.loc[~complete, "candidate_id"].astype(str).head(5).tolist()
            raise AssertionError(f"valid source lacks complete minute path for {symbol}: {missed}")
        entry = pd.to_numeric(part["tp6_sl4_entry_price"], errors="coerce").to_numpy(float)
        atr = pd.to_numeric(part["atr_1h"], errors="coerce").to_numpy(float)
        if not np.allclose(opened, entry, rtol=0.0, atol=1e-12):
            raise AssertionError(f"{symbol}: frozen entry diverges from exact decision-minute open")
        labels = _derive_labels(entry=entry, atr=atr, high=high, low=low)
        out.loc[idx, "event_timing_label_valid"] = True
        out.loc[idx, "event_timing_target_invalid"] = False
        for column, values in labels.items():
            out.loc[idx, column] = values
        parity_rows += len(part)
    invalid = ~valid
    event_columns = [column for column in out if column not in {*IDENTITY, "__label_available_at__", "policy_net_bps", "event_timing_label_valid", "event_timing_target_invalid"}]
    if out.loc[invalid, event_columns].notna().any().any():
        raise AssertionError("invalid event-path rows were assigned an ordinary timing/event label")
    records: dict[str, Any] = {
        "rows": int(len(out)), "valid_rows": int(valid.sum()), "invalid_rows": int((~valid).sum()),
        "direct_entry_open_parity_rows": int(parity_rows),
    }
    for threshold in ADVERSE_ATR:
        tag = f"{threshold:.1f}".replace(".", "p")
        current = out.loc[valid]
        records[f"favourable_first_rate_{tag}atr"] = float(pd.to_numeric(current[f"favourable_first_{tag}atr"], errors="coerce").mean())
        records[f"adverse_first_rate_{tag}atr"] = float(pd.to_numeric(current[f"adverse_first_{tag}atr"], errors="coerce").mean())
        records[f"censored_rate_{tag}atr"] = float(pd.to_numeric(current[f"censored_{tag}atr"], errors="coerce").mean())
    return out, records


def _materialize_all(source: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Materialise every month with exact symbol-month source windows.

    Historical one-minute sources contain append-only overlapping fragments.
    Some symbols have many small fragments spanning a year, so a symbol-year
    batch would needlessly read all of them for a sparse candidate population.
    Grouping by the source month is exactly the original label contract:
    every group asks the same immutable reader for the same timestamp window
    as the former month-by-month implementation, while keeping one vectorised
    result frame and avoiding a second population load.
    """
    source = source.reset_index(drop=True).copy()
    out = _blank(source)
    valid = (
        source["rich_path_label_valid"].fillna(False).astype(bool)
        & ~source["rich_path_target_invalid"].fillna(True).astype(bool)
        & source["policy_path_valid"].fillna(False).astype(bool)
        & source["label_valid"].fillna(False).astype(bool)
        & ~source["target_invalid"].fillna(True).astype(bool)
        & pd.to_numeric(source["tp6_sl4_entry_price"], errors="coerce").gt(0.0)
        & pd.to_numeric(source["atr_1h"], errors="coerce").gt(0.0)
    )
    source["_month"] = pd.to_datetime(source["__decision_ts__"], utc=True).dt.strftime("%Y-%m")
    parity = np.zeros(len(source), dtype=bool)
    groups = list(source.loc[valid].groupby(["_month", "__symbol__"], sort=True))
    group_total = len(groups)

    def derive_group(item: tuple[tuple[str, str], pd.DataFrame]) -> list[tuple[np.ndarray, dict[str, np.ndarray]]]:
        (month, symbol), part = item
        # One candidate's exact path ends at decision + 12h.  Splitting sparse
        # symbol-months where that path cannot overlap avoids reading unrelated
        # days of an append-heavy archive.  It is semantically identical to a
        # single broad read because a fragment can affect a candidate only when
        # it overlaps that candidate's own complete path window.
        ordered = part.sort_values(["__decision_ts__", "candidate_id"], kind="stable").copy()
        decisions = pd.to_datetime(ordered["__decision_ts__"], utc=True)
        segment = decisions.diff().gt(pd.Timedelta(minutes=HORIZON_MINUTES)).cumsum()
        result: list[tuple[np.ndarray, dict[str, np.ndarray]]] = []
        for _segment, local in ordered.groupby(segment, sort=True):
            idx = local.index.to_numpy()
            local_decisions = pd.to_datetime(local["__decision_ts__"], utc=True)
            minute = _minute_path_pruned(
                MINUTE_ROOT, _packb_to_kraken_symbol(str(symbol)),
                local_decisions.min(), local_decisions.max() + pd.Timedelta(minutes=HORIZON_MINUTES),
            )
            opened, high, low, _close, complete = _path_matrices(minute, local_decisions)
            if not complete.all():
                missed = local.loc[~complete, "candidate_id"].astype(str).head(5).tolist()
                raise AssertionError(f"valid source lacks complete minute path for {symbol}/{month}: {missed}")
            entry = pd.to_numeric(local["tp6_sl4_entry_price"], errors="coerce").to_numpy(float)
            atr = pd.to_numeric(local["atr_1h"], errors="coerce").to_numpy(float)
            if not np.allclose(opened, entry, rtol=0.0, atol=1e-12):
                raise AssertionError(f"{symbol}/{month}: frozen entry diverges from exact decision-minute open")
            result.append((idx, _derive_labels(entry=entry, atr=atr, high=high, low=low)))
        return result

    # These reads are independent and bounded.  The parent still writes every
    # group in deterministic source order, so output identities and values do
    # not depend on completion order.
    with ThreadPoolExecutor(max_workers=PATH_READ_WORKERS, thread_name_prefix="short-event-path") as executor:
        for group_index, segments in enumerate(executor.map(derive_group, groups), start=1):
            for idx, labels in segments:
                out.loc[idx, "event_timing_label_valid"] = True
                out.loc[idx, "event_timing_target_invalid"] = False
                for column, values in labels.items():
                    out.loc[idx, column] = values
                parity[idx] = True
            if group_index % 50 == 0 or group_index == group_total:
                print(json.dumps({"progress": "symbol_month_paths", "completed_groups": group_index, "total_groups": group_total}), flush=True)
    invalid = ~valid
    event_columns = [column for column in out if column not in {*IDENTITY, "__label_available_at__", "policy_net_bps", "event_timing_label_valid", "event_timing_target_invalid"}]
    if out.loc[invalid, event_columns].notna().any().any():
        raise AssertionError("invalid event-path rows were assigned an ordinary timing/event label")
    if not out.loc[valid, "event_timing_label_valid"].all() or not parity[valid.to_numpy()].all():
        raise AssertionError("complete source path failed to receive a timing label")
    out["__decision_ts__"] = pd.to_datetime(out["__decision_ts__"], utc=True)
    records: list[dict[str, Any]] = []
    for month, local in out.groupby(out["__decision_ts__"].dt.strftime("%Y-%m"), sort=True):
        current_valid = local["event_timing_label_valid"].astype(bool)
        record: dict[str, Any] = {
            "month": month, "rows": int(len(local)), "valid_rows": int(current_valid.sum()),
            "invalid_rows": int((~current_valid).sum()),
            "direct_entry_open_parity_rows": int(parity[local.index.to_numpy()].sum()),
        }
        for threshold in ADVERSE_ATR:
            tag = f"{threshold:.1f}".replace(".", "p")
            valid_local = local.loc[current_valid]
            record[f"favourable_first_rate_{tag}atr"] = float(pd.to_numeric(valid_local[f"favourable_first_{tag}atr"], errors="coerce").mean())
            record[f"adverse_first_rate_{tag}atr"] = float(pd.to_numeric(valid_local[f"adverse_first_{tag}atr"], errors="coerce").mean())
            record[f"censored_rate_{tag}atr"] = float(pd.to_numeric(valid_local[f"censored_{tag}atr"], errors="coerce").mean())
        records.append(record)
    return out.drop(columns=["_month"], errors="ignore"), pd.DataFrame(records)


def run(out: Path = OUT) -> Path:
    if out.exists():
        raise FileExistsError(f"immutable output exists: {out}")
    frame, _, _, source_hashes = r3._load_frame()
    frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True)
    frame = frame.loc[frame["__decision_ts__"].ge(START) & frame["__decision_ts__"].lt(END)].copy()
    h12_parts = _h12_by_month()
    source_parts: list[pd.DataFrame] = []
    for month in pd.date_range(START, END, freq="MS", inclusive="left"):
        source = _month_input(frame, h12_parts, month)
        if source.empty:
            continue
        source_parts.append(source)
    if not source_parts:
        raise RuntimeError("no target-free short candidates for timing labels")
    result, coverage = _materialize_all(pd.concat(source_parts, ignore_index=True))
    out.mkdir(parents=True)
    records = coverage.to_dict("records")
    for month, part in result.groupby(result["__decision_ts__"].dt.strftime("%Y-%m"), sort=True):
        path = out / "parts" / f"month={month}" / "side=short.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        part.to_parquet(path, index=False, compression="zstd")
    for record in records:
        print(json.dumps(record), flush=True)
    coverage.to_parquet(out / "coverage.parquet", index=False, compression="zstd")
    manifest = {
        "schema": SCHEMA, "status": "complete", "side": SIDE,
        "period": {"start": START.isoformat(), "end_exclusive": END.isoformat()},
        "entry": "frozen exact decision-minute open at signal close + one hour",
        "horizon": "complete post-decision 720 x one-minute bars; event window first 360 minutes",
        "favourable": f"entry - low > {FAVOURABLE_BPS:g} bps",
        "adverse": [f"high - entry >= {value:g} ATR" for value in ADVERSE_ATR],
        "tie_policy": "same-minute favourable/adverse OHLC touch is adverse-first",
        "label_availability": "decision + 12h; labels are supervised-only and never inference inputs",
        "invalidity": "incomplete/invalid paths remain null-labelled and excluded from fitting",
        "sources": {"rich_label_manifest_sha256": _sha256(RICH_ROOT / "run_manifest.json"), **source_hashes},
        "coverage": records,
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    print(run(args.out))


if __name__ == "__main__":
    main()
