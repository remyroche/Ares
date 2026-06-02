#!/usr/bin/env python3
"""Compare live-fetched OHLCV against cached historical references.

The live inference fetcher appends fresh hourly candles under the exchange market
data root. This audit checks those saved rows against any overlapping cached
historical-hourly rows and, when available, against execution 1m bars aggregated
back to hourly candles.

The important part of this script is overlap discovery. A fixed symbol list can
make a single stale or differently-sourced cached row look like a general live
fetch problem, so the default mode now discovers common symbols, samples them
deterministically, compares every overlapping hour in the requested window, and
writes both pair-level summaries and concrete mismatch examples.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from extreme_price_movements.data_store import PartitionedOHLCVStore


OHLCV_COLS = ("open", "high", "low", "close", "volume")
DEFAULT_PRIORITY_SYMBOLS = (
    "BTC/USD:USD",
    "ETH/USD:USD",
    "AR/USD:USD",
    "DEGEN/USD:USD",
    "SOL/USD:USD",
    "XRP/USD:USD",
    "HBAR/USD:USD",
    "NEAR/USD:USD",
)


def _parse_ts(value: str) -> pd.Timestamp:
    ts = pd.to_datetime(value, utc=True, errors="raise")
    return pd.Timestamp(ts)


def _normalise_frame(df: pd.DataFrame, *, floor: str | None = "h") -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=list(OHLCV_COLS))
    out = df.copy()
    if "ts" in out.columns:
        out.index = pd.to_datetime(out["ts"], utc=True, errors="coerce")
    else:
        out.index = pd.to_datetime(out.index, utc=True, errors="coerce")
    out = out.loc[pd.notna(out.index)].sort_index()
    if floor:
        out.index = out.index.floor(floor)
    keep = [c for c in OHLCV_COLS if c in out.columns]
    out = out[keep]
    return out[~out.index.duplicated(keep="last")]


def _symbol_from_partition_dir(path: Path) -> str | None:
    name = path.name
    if not name.startswith("symbol="):
        return None
    safe = name.split("=", 1)[1]
    if not safe:
        return None
    if "_USD:USD" in safe:
        return safe.replace("_USD:USD", "/USD:USD")
    if "_" in safe:
        base, quote = safe.split("_", 1)
        return f"{base}/{quote}"
    return safe


def _discover_symbols(store: PartitionedOHLCVStore) -> list[str]:
    root = Path(store.ohlcv_dir)
    if not root.exists():
        return []
    symbols: list[str] = []
    for path in root.glob("symbol=*"):
        if not path.is_dir():
            continue
        symbol = _symbol_from_partition_dir(path)
        if symbol:
            symbols.append(symbol)
    return sorted(set(symbols))


def _stable_symbol_sample(
    symbols: Sequence[str],
    *,
    priority_symbols: Sequence[str],
    max_symbols: int,
) -> list[str]:
    unique = sorted(set(symbols))
    if max_symbols <= 0 or len(unique) <= max_symbols:
        return unique
    selected: list[str] = []
    available = set(unique)
    for symbol in priority_symbols:
        if symbol in available and symbol not in selected:
            selected.append(symbol)
    remaining_budget = max(0, max_symbols - len(selected))
    ranked = sorted(
        (s for s in unique if s not in selected),
        key=lambda s: hashlib.sha256(s.encode("utf-8")).hexdigest(),
    )
    selected.extend(ranked[:remaining_budget])
    return sorted(selected)


def _load_store(
    store: PartitionedOHLCVStore,
    symbol: str,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    floor: str | None = "h",
) -> pd.DataFrame:
    """Load raw partitioned OHLCV without store sidecar overlays.

    PartitionedOHLCVStore.load() intentionally overlays actual-volume sidecars in
    normal production use. This audit is about candle parity, so it reads the
    partition files directly and compares the raw saved OHLCV values.
    """
    sym_dir = Path(store._get_symbol_dir(symbol))
    if not sym_dir.exists():
        return pd.DataFrame(columns=list(OHLCV_COLS))
    files: list[str] = []
    start_sec = int(start.timestamp())
    end_sec = int(end.timestamp())
    for path in sym_dir.rglob("*.parquet"):
        try:
            parts = path.stem.split("-")
            if len(parts) >= 3:
                f_min = int(parts[-2])
                f_max = int(parts[-1])
                if f_min > end_sec or f_max < start_sec:
                    continue
        except Exception:
            pass
        files.append(str(path))
    if not files:
        return pd.DataFrame(columns=list(OHLCV_COLS))
    try:
        frames: list[pd.DataFrame] = []
        for file_path in sorted(files):
            try:
                part = pd.read_parquet(
                    file_path,
                    columns=["ts", *[c for c in OHLCV_COLS]],
                )
            except Exception:
                try:
                    part = pd.read_parquet(file_path)
                except Exception:
                    continue
            frames.append(part)
        if not frames:
            return pd.DataFrame(columns=list(OHLCV_COLS))
        df = pd.concat(frames, ignore_index=True)
        out = _normalise_frame(df, floor=floor)
        if not out.empty:
            out = out.loc[(out.index >= start) & (out.index <= end)]
        return out
    except Exception:
        return pd.DataFrame(columns=list(OHLCV_COLS))


def _summarise_diff(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    price_tol_abs: float,
    volume_tol_abs: float,
) -> Dict[str, object]:
    common = left.index.intersection(right.index)
    out: Dict[str, object] = {
        "overlap_rows": int(len(common)),
        "first_overlap_ts": common.min().isoformat() if len(common) else None,
        "last_overlap_ts": common.max().isoformat() if len(common) else None,
    }
    if len(common) == 0:
        return out
    left_aligned = left.loc[common]
    right_aligned = right.loc[common]
    mismatch_any = np.zeros(len(common), dtype=bool)
    price_mismatch_any = np.zeros(len(common), dtype=bool)
    volume_mismatch_any = np.zeros(len(common), dtype=bool)
    for col in OHLCV_COLS:
        if col not in left_aligned.columns or col not in right_aligned.columns:
            continue
        diff = (
            left_aligned[col].astype(float) - right_aligned[col].astype(float)
        ).replace([np.inf, -np.inf], np.nan)
        valid = diff.dropna()
        tol = volume_tol_abs if col == "volume" else price_tol_abs
        col_mismatch = diff.abs().fillna(np.inf).to_numpy() > tol
        mismatch_any |= col_mismatch
        if col == "volume":
            volume_mismatch_any |= col_mismatch
        else:
            price_mismatch_any |= col_mismatch
        out[f"{col}_max_abs_diff"] = (
            None if valid.empty else float(valid.abs().max())
        )
        out[f"{col}_mean_abs_diff"] = (
            None if valid.empty else float(valid.abs().mean())
        )
        out[f"{col}_mismatch_count_gt_1e-9"] = int((diff.abs() > 1e-9).sum())
        out[f"{col}_mismatch_count_gt_tol"] = int(col_mismatch.sum())
    out["exact_rows_within_tol"] = int((~mismatch_any).sum())
    out["mismatch_rows_gt_tol"] = int(mismatch_any.sum())
    out["price_mismatch_rows_gt_tol"] = int(price_mismatch_any.sum())
    out["volume_mismatch_rows_gt_tol"] = int(volume_mismatch_any.sum())
    return out


def _aggregate_1m_to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=list(OHLCV_COLS))
    out = df.copy()
    if "ts" in out.columns:
        out.index = pd.to_datetime(out["ts"], utc=True, errors="coerce")
    else:
        out.index = pd.to_datetime(out.index, utc=True, errors="coerce")
    out = out.loc[pd.notna(out.index)].sort_index()
    # Some saved 1m rows include a small seconds offset from the exchange.
    out.index = out.index.floor("min")
    agg = out.resample("1h", label="left", closed="left").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    return _normalise_frame(agg.dropna(how="any"))


def _compare_pair(
    *,
    pair_name: str,
    symbols: Iterable[str],
    left_store: PartitionedOHLCVStore,
    right_store: PartitionedOHLCVStore,
    start: pd.Timestamp,
    end: pd.Timestamp,
    right_is_1m: bool = False,
    price_tol_abs: float = 1e-9,
    volume_tol_abs: float = 1e-9,
    max_examples: int = 50,
) -> tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    rows: List[Dict[str, object]] = []
    examples: List[Dict[str, object]] = []
    for symbol in symbols:
        left = _load_store(left_store, symbol, start=start, end=end, floor="h")
        right_raw = _load_store(
            right_store,
            symbol,
            start=start,
            end=end,
            floor=None if right_is_1m else "h",
        )
        right = _aggregate_1m_to_hourly(right_raw) if right_is_1m else right_raw
        row: Dict[str, object] = {
            "pair": pair_name,
            "symbol": symbol,
            "live_rows": int(len(left)),
            "reference_rows": int(len(right)),
            "live_first": left.index.min().isoformat() if len(left) else None,
            "live_last": left.index.max().isoformat() if len(left) else None,
            "reference_first": right.index.min().isoformat() if len(right) else None,
            "reference_last": right.index.max().isoformat() if len(right) else None,
        }
        summary = _summarise_diff(
            left,
            right,
            price_tol_abs=price_tol_abs,
            volume_tol_abs=volume_tol_abs,
        )
        row.update(summary)
        common = left.index.intersection(right.index)
        if len(common) and len(examples) < max_examples:
            left_aligned = left.loc[common]
            right_aligned = right.loc[common]
            for col in OHLCV_COLS:
                if col not in left_aligned.columns or col not in right_aligned.columns:
                    continue
                tol = volume_tol_abs if col == "volume" else price_tol_abs
                diff = (
                    left_aligned[col].astype(float)
                    - right_aligned[col].astype(float)
                ).replace([np.inf, -np.inf], np.nan)
                bad = diff.abs() > tol
                if not bool(bad.any()):
                    continue
                for ts, delta in diff.loc[bad].abs().sort_values(ascending=False).items():
                    left_value = left_aligned.at[ts, col]
                    right_value = right_aligned.at[ts, col]
                    denom = abs(float(right_value)) if pd.notna(right_value) else np.nan
                    examples.append(
                        {
                            "pair": pair_name,
                            "symbol": symbol,
                            "ts": pd.Timestamp(ts).isoformat(),
                            "column": col,
                            "live_value": None if pd.isna(left_value) else float(left_value),
                            "reference_value": None
                            if pd.isna(right_value)
                            else float(right_value),
                            "abs_diff": None if pd.isna(delta) else float(delta),
                            "rel_diff_bps": None
                            if not np.isfinite(denom) or denom == 0
                            else float(10000.0 * float(delta) / denom),
                        }
                    )
                    if len(examples) >= max_examples:
                        break
                if len(examples) >= max_examples:
                    break
        rows.append(row)
    for row in rows:
        row["_mismatch_examples_ref"] = f"{pair_name}_mismatch_examples.csv"
    return rows, examples


def _execution_1m_completeness(
    *,
    symbols: Iterable[str],
    live_store: PartitionedOHLCVStore,
    execution_1m_store: PartitionedOHLCVStore,
    start: pd.Timestamp,
    end: pd.Timestamp,
    price_tol_abs: float,
    volume_tol_abs: float,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows: list[dict[str, object]] = []
    for symbol in symbols:
        live = _load_store(live_store, symbol, start=start, end=end, floor="h")
        raw_1m = _load_store(execution_1m_store, symbol, start=start, end=end, floor=None)
        if live.empty or raw_1m.empty:
            continue
        raw_1m = raw_1m.copy()
        raw_1m.index = pd.to_datetime(raw_1m.index, utc=True, errors="coerce").floor("min")
        raw_1m = raw_1m.loc[pd.notna(raw_1m.index)].sort_index()
        if raw_1m.empty:
            continue
        hour_key = raw_1m.index.floor("h")
        minute_counts = raw_1m.groupby(hour_key).agg(
            raw_1m_rows=("open", "size"),
            unique_1m_minutes=("open", lambda s: int(s.index.nunique())),
        )
        aggregated = _aggregate_1m_to_hourly(raw_1m)
        common = live.index.intersection(aggregated.index)
        for ts in common:
            row: dict[str, object] = {
                "symbol": symbol,
                "ts": pd.Timestamp(ts).isoformat(),
                "raw_1m_rows": int(minute_counts.at[ts, "raw_1m_rows"])
                if ts in minute_counts.index
                else 0,
                "unique_1m_minutes": int(minute_counts.at[ts, "unique_1m_minutes"])
                if ts in minute_counts.index
                else 0,
            }
            row["is_complete_hour"] = bool(row["unique_1m_minutes"] >= 60)
            mismatch_any = False
            price_mismatch_any = False
            volume_mismatch_any = False
            for col in OHLCV_COLS:
                if col not in live.columns or col not in aggregated.columns:
                    continue
                left_value = live.at[ts, col]
                right_value = aggregated.at[ts, col]
                diff = (
                    np.nan
                    if pd.isna(left_value) or pd.isna(right_value)
                    else abs(float(left_value) - float(right_value))
                )
                tol = volume_tol_abs if col == "volume" else price_tol_abs
                is_bad = (not np.isfinite(diff)) or diff > tol
                row[f"{col}_abs_diff"] = None if not np.isfinite(diff) else float(diff)
                row[f"{col}_mismatch"] = bool(is_bad)
                mismatch_any = mismatch_any or is_bad
                if col == "volume":
                    volume_mismatch_any = volume_mismatch_any or is_bad
                else:
                    price_mismatch_any = price_mismatch_any or is_bad
            row["exact_within_tol"] = not mismatch_any
            row["price_mismatch"] = bool(price_mismatch_any)
            row["volume_mismatch"] = bool(volume_mismatch_any)
            rows.append(row)
    if not rows:
        return rows, {
            "rows": 0,
            "symbols": 0,
            "complete_hour_rows": 0,
            "incomplete_hour_rows": 0,
        }
    df = pd.DataFrame(rows)
    complete = df["is_complete_hour"].fillna(False).astype(bool)
    mismatch = ~df["exact_within_tol"].fillna(False).astype(bool)
    summary = {
        "rows": int(len(df)),
        "symbols": int(df["symbol"].nunique()),
        "complete_hour_rows": int(complete.sum()),
        "incomplete_hour_rows": int((~complete).sum()),
        "mismatch_rows": int(mismatch.sum()),
        "mismatch_rate": float(mismatch.mean()),
        "complete_hour_mismatch_rows": int((complete & mismatch).sum()),
        "incomplete_hour_mismatch_rows": int(((~complete) & mismatch).sum()),
        "unique_1m_minutes_min": int(pd.to_numeric(df["unique_1m_minutes"], errors="coerce").min()),
        "unique_1m_minutes_median": float(
            pd.to_numeric(df["unique_1m_minutes"], errors="coerce").median()
        ),
        "unique_1m_minutes_max": int(pd.to_numeric(df["unique_1m_minutes"], errors="coerce").max()),
        "rows_by_unique_1m_minutes": {
            str(int(k)): int(v)
            for k, v in df["unique_1m_minutes"].value_counts().sort_index().items()
        },
    }
    return rows, summary


def _pair_summary(rows: list[dict[str, object]], *, pair_name: str) -> dict[str, object]:
    df = pd.DataFrame(rows)
    if df.empty:
        return {
            "pair": pair_name,
            "symbols": 0,
            "symbols_with_overlap": 0,
            "overlap_rows": 0,
            "mismatch_rows_gt_tol": 0,
            "mismatch_rate_gt_tol": None,
        }
    overlap = pd.to_numeric(
        df["overlap_rows"] if "overlap_rows" in df else pd.Series(0, index=df.index),
        errors="coerce",
    ).fillna(0)
    mismatches = pd.to_numeric(
        df["mismatch_rows_gt_tol"]
        if "mismatch_rows_gt_tol" in df
        else pd.Series(0, index=df.index),
        errors="coerce",
    ).fillna(0)
    total_overlap = int(overlap.sum())
    total_mismatch = int(mismatches.sum())
    return {
        "pair": pair_name,
        "symbols": int(len(df)),
        "symbols_with_live_rows": int((pd.to_numeric(df["live_rows"], errors="coerce") > 0).sum()),
        "symbols_with_reference_rows": int((pd.to_numeric(df["reference_rows"], errors="coerce") > 0).sum()),
        "symbols_with_overlap": int((overlap > 0).sum()),
        "symbols_with_mismatch": int((mismatches > 0).sum()),
        "overlap_rows": total_overlap,
        "exact_rows_within_tol": int(
            pd.to_numeric(
                df["exact_rows_within_tol"]
                if "exact_rows_within_tol" in df
                else pd.Series(0, index=df.index),
                errors="coerce",
            )
            .fillna(0)
            .sum()
        ),
        "mismatch_rows_gt_tol": total_mismatch,
        "price_mismatch_rows_gt_tol": int(
            pd.to_numeric(
                df["price_mismatch_rows_gt_tol"]
                if "price_mismatch_rows_gt_tol" in df
                else pd.Series(0, index=df.index),
                errors="coerce",
            )
            .fillna(0)
            .sum()
        ),
        "volume_mismatch_rows_gt_tol": int(
            pd.to_numeric(
                df["volume_mismatch_rows_gt_tol"]
                if "volume_mismatch_rows_gt_tol" in df
                else pd.Series(0, index=df.index),
                errors="coerce",
            )
            .fillna(0)
            .sum()
        ),
        "mismatch_rate_gt_tol": None
        if total_overlap == 0
        else float(total_mismatch / total_overlap),
    }
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data_perp")
    parser.add_argument("--run-id", default="20260525_010004_nopenalty")
    parser.add_argument(
        "--symbols",
        default="auto",
        help="Comma-separated symbols, or 'auto' to discover a deterministic sample.",
    )
    parser.add_argument("--max-symbols", type=int, default=64)
    parser.add_argument("--price-tol-abs", type=float, default=1e-6)
    parser.add_argument("--volume-tol-abs", type=float, default=1e-4)
    parser.add_argument("--max-mismatch-examples", type=int, default=200)
    parser.add_argument("--start", default="2026-05-01T00:00:00Z")
    parser.add_argument("--end", default="2026-05-31T00:00:00Z")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    start = _parse_ts(args.start)
    end = _parse_ts(args.end)

    live_store = PartitionedOHLCVStore(
        root_dir=data_root / "exchanges" / "krakenfutures",
        timeframe="1h",
    )
    historical_store = PartitionedOHLCVStore(
        root_dir=data_root
        / "exchanges"
        / "krakenfutures"
        / "exchanges"
        / "krakenfutures",
        timeframe="1h",
    )
    nested_execution_1m_store = PartitionedOHLCVStore(
        root_dir=data_root
        / "exchanges"
        / "krakenfutures"
        / "exchanges"
        / "krakenfutures"
        / "execution_1m",
        timeframe="1m",
    )
    execution_1m_store = PartitionedOHLCVStore(
        root_dir=data_root / "exchanges" / "krakenfutures" / "execution_1m",
        timeframe="1m",
    )

    if str(args.symbols).strip().lower() == "auto":
        live_symbols = set(_discover_symbols(live_store))
        reference_symbols = (
            set(_discover_symbols(historical_store))
            | set(_discover_symbols(execution_1m_store))
            | set(_discover_symbols(nested_execution_1m_store))
        )
        common = sorted(live_symbols & reference_symbols)
        symbols = _stable_symbol_sample(
            common or sorted(live_symbols),
            priority_symbols=DEFAULT_PRIORITY_SYMBOLS,
            max_symbols=int(args.max_symbols),
        )
    else:
        symbols = [s.strip() for s in str(args.symbols).split(",") if s.strip()]

    comparisons: dict[str, list[dict[str, object]]] = {}
    mismatch_examples: dict[str, list[dict[str, object]]] = {}
    pairs = [
        (
            "hourly_live_vs_historical",
            historical_store,
            False,
        ),
        (
            "hourly_live_vs_execution_1m_aggregate",
            execution_1m_store,
            True,
        ),
        (
            "hourly_live_vs_nested_execution_1m_aggregate",
            nested_execution_1m_store,
            True,
        ),
    ]
    pair_summaries = []
    for pair_name, right_store, right_is_1m in pairs:
        rows, examples = _compare_pair(
            pair_name=pair_name,
            symbols=symbols,
            left_store=live_store,
            right_store=right_store,
            start=start,
            end=end,
            right_is_1m=right_is_1m,
            price_tol_abs=float(args.price_tol_abs),
            volume_tol_abs=float(args.volume_tol_abs),
            max_examples=int(args.max_mismatch_examples),
        )
        comparisons[pair_name] = rows
        mismatch_examples[pair_name] = examples
        pair_summaries.append(_pair_summary(rows, pair_name=pair_name))
    execution_1m_completeness_rows, execution_1m_completeness_summary = (
        _execution_1m_completeness(
            symbols=symbols,
            live_store=live_store,
            execution_1m_store=execution_1m_store,
            start=start,
            end=end,
            price_tol_abs=float(args.price_tol_abs),
            volume_tol_abs=float(args.volume_tol_abs),
        )
    )

    report = {
        "data_root": str(data_root),
        "run_id": args.run_id,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "symbols": symbols,
        "symbol_count": len(symbols),
        "price_tol_abs": float(args.price_tol_abs),
        "volume_tol_abs": float(args.volume_tol_abs),
        "stores": {
            "live_hourly": str(Path(live_store.ohlcv_dir)),
            "historical_hourly": str(Path(historical_store.ohlcv_dir)),
            "execution_1m": str(Path(execution_1m_store.ohlcv_dir)),
            "nested_execution_1m": str(Path(nested_execution_1m_store.ohlcv_dir)),
        },
        "pair_summaries": pair_summaries,
        "execution_1m_completeness_summary": execution_1m_completeness_summary,
        **comparisons,
    }

    out_dir = (
        data_root
        / "artifacts"
        / str(args.run_id)
        / "live_ohlcv_parity"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ohlcv_live_vs_historical_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True)
    )
    pd.DataFrame(pair_summaries).to_csv(out_dir / "pair_summaries.csv", index=False)
    for key in comparisons:
        pd.DataFrame(report[key]).to_csv(out_dir / f"{key}.csv", index=False)
        pd.DataFrame(mismatch_examples[key]).to_csv(
            out_dir / f"{key}_mismatch_examples.csv", index=False
        )
    pd.DataFrame(execution_1m_completeness_rows).to_csv(
        out_dir / "execution_1m_hourly_completeness.csv",
        index=False,
    )
    (out_dir / "execution_1m_hourly_completeness_summary.json").write_text(
        json.dumps(execution_1m_completeness_summary, indent=2, sort_keys=True)
        + "\n"
    )
    print(f"wrote {out_dir}")
    for summary in pair_summaries:
        print(json.dumps(summary, sort_keys=True))
    print(json.dumps(execution_1m_completeness_summary, sort_keys=True))


if __name__ == "__main__":
    main()
