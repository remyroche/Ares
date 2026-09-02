#!/usr/bin/env python3
"""Materialise the target-free causal intraday feature sidecar for P8U probes.

The sidecar is deliberately separate from policy labels and Router outputs.
It reads only the fixed candidate identity universe plus historical completed
15-minute/hourly market data, then writes one immutable monthly file.  Outcome
labels are joined only by the probe runner after this candidate population is
sealed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.opportunity_probe_features import (  # noqa: E402
    OPPORTUNITY_PROBE_INTRADAY_FEATURE_KEYS,
    add_opportunity_probe_cross_sectional_features,
    canonical_15m_file_stem,
    materialize_opportunity_probe_hourly_derivative_features,
    materialize_opportunity_probe_intraday_features,
)


SCHEMA = "strict_r3_p8u_probe_causal_intraday_sidecar_v2"
IDENTITY = ("candidate_id", "__decision_ts__", "__symbol__", "side_name")
INTRADAY_ROOT = ROOT / "15m_ohlcv_perp"
HOURLY_ROOT = ROOT / "data_perp" / "ohlcv"


def _utc(value: object) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _month_starts(start: object, end: object) -> list[pd.Timestamp]:
    start_ts, end_ts = _utc(start), _utc(end)
    return list(pd.date_range(start_ts.normalize().replace(day=1), end_ts.normalize().replace(day=1), freq="MS", tz="UTC"))


def _source_for(month: pd.Timestamp, sources: list[dict[str, Any]]) -> dict[str, Any]:
    for source in sources:
        if _utc(source["start"]) <= month <= _utc(source["end"]):
            return source
    raise KeyError(f"no feature source declared for {month:%Y-%m}")


def _candidate_path(config: dict[str, Any], month: pd.Timestamp) -> Path:
    source = _source_for(month, config["feature_sources"])
    return ROOT / source["root"] / f"month={month:%Y-%m}" / "causal_feature_universe.parquet"


def _load_candidates(config: dict[str, Any], month: pd.Timestamp) -> pd.DataFrame:
    path = _candidate_path(config, month)
    if not path.exists():
        raise FileNotFoundError(path)
    available = set(pq.ParquetFile(path).schema.names)
    missing = set(IDENTITY).difference(available)
    if missing:
        raise AssertionError(f"{path} lacks identity columns {sorted(missing)}")
    frame = pd.read_parquet(path, columns=list(IDENTITY))
    frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
    frame["__symbol__"] = frame["__symbol__"].astype(str)
    frame["side_name"] = frame["side_name"].astype(str).str.lower()
    if frame["candidate_id"].duplicated().any():
        raise AssertionError(f"duplicate candidate IDs in {path}")
    return frame.loc[frame["side_name"].eq(str(config["side"]).lower())].copy()


def _read_15m(symbol: str, *, start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame | None, str | None]:
    path = INTRADAY_ROOT / f"{canonical_15m_file_stem(symbol)}_15m.parquet"
    if not path.exists():
        return None, "missing_15m_file"
    try:
        raw = pd.read_parquet(path)
    except Exception as exc:  # malformed legacy cache is a source failure, not a zero signal.
        return None, f"read_15m_failed:{type(exc).__name__}"
    if not isinstance(raw.index, pd.DatetimeIndex):
        if "ts" not in raw.columns:
            return None, "15m_timestamp_missing"
        raw = raw.set_index("ts")
    raw.index = pd.to_datetime(raw.index, utc=True, errors="coerce")
    raw = raw.loc[~raw.index.isna() & ~raw.index.duplicated(keep="last")].sort_index(kind="stable")
    # 60 days lets Wilder-14h settle before the first emitted month row while
    # keeping per-symbol reads bounded even for multi-year research panels.
    return raw.loc[(raw.index >= start - pd.Timedelta(days=60)) & (raw.index <= end)].copy(), None


def _hourly_path(symbol: str) -> Path | None:
    base = str(symbol).split("/", 1)[0].strip().upper()
    preferred = (f"{base}_USDT", f"{base}_USDC", f"{base}_USD", f"{base}_USD:USD")
    for key in preferred:
        candidate = HOURLY_ROOT / f"symbol={key}"
        if candidate.exists():
            return candidate
    # A nonstandard historical quote is acceptable only when it is the unique
    # local perpetual source for that base.  Ambiguity fails closed to null.
    matches = sorted(HOURLY_ROOT.glob(f"symbol={base}_*"))
    return matches[0] if len(matches) == 1 else None


def _read_hourly(symbol: str, *, start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame | None, str | None]:
    directory = _hourly_path(symbol)
    if directory is None:
        return None, "missing_hourly_file"
    parts = sorted(directory.glob("year=*/*.parquet"))
    if not parts:
        return None, "empty_hourly_directory"
    rows: list[pd.DataFrame] = []
    wanted = ("ts", "open", "close", "open_interest", "funding_rate", "mark_price", "index_price", "premium_index")
    for part in parts:
        try:
            available = set(pq.ParquetFile(part).schema.names)
            columns = [column for column in wanted if column in available]
            if {"ts", "open", "close"}.difference(columns):
                continue
            rows.append(pd.read_parquet(part, columns=columns))
        except Exception:
            continue
    if not rows:
        return None, "hourly_read_failed"
    raw = pd.concat(rows, ignore_index=True)
    raw["ts"] = pd.to_datetime(raw["ts"], utc=True, errors="coerce")
    raw = raw.loc[raw["ts"].notna()].drop_duplicates("ts", keep="last").set_index("ts").sort_index(kind="stable")
    return raw.loc[(raw.index >= start - pd.Timedelta(days=60)) & (raw.index <= end)].copy(), None


def _symbol_features(group: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    symbol = str(group["__symbol__"].iloc[0])
    start, end = group["__decision_ts__"].min(), group["__decision_ts__"].max()
    output = group.loc[:, list(IDENTITY)].copy()
    for field in OPPORTUNITY_PROBE_INTRADAY_FEATURE_KEYS:
        output[field] = np.nan
    status: list[str] = []
    bars_15m, error_15m = _read_15m(symbol, start=start, end=end)
    if error_15m is None and bars_15m is not None and not bars_15m.empty:
        intraday = materialize_opportunity_probe_intraday_features(bars_15m)
        output = output.merge(
            intraday.reset_index(), on="__decision_ts__", how="left", validate="many_to_one", suffixes=("", "__raw")
        )
        for field in OPPORTUNITY_PROBE_INTRADAY_FEATURE_KEYS:
            raw_field = f"{field}__raw"
            if raw_field in output:
                output[field] = output.pop(raw_field)
        status.append("15m_ok")
    else:
        status.append(error_15m or "empty_15m")
    hourly, error_hourly = _read_hourly(symbol, start=start, end=end)
    if error_hourly is None and hourly is not None and not hourly.empty:
        hourly_features = materialize_opportunity_probe_hourly_derivative_features(hourly)
        output = output.merge(
            hourly_features.reset_index(), on="__decision_ts__", how="left", validate="many_to_one", suffixes=("", "__hourly")
        )
        for field in hourly_features.columns:
            duplicate = f"{field}__hourly"
            if duplicate in output:
                output[field] = output.pop(duplicate)
        status.append("hourly_ok")
    else:
        status.append(error_hourly or "empty_hourly")
    core = ("probe_atr_bps_14h", "probe_path_efficiency_1h", "probe_relative_volume_1h_4h")
    output["probe_intraday_source_ready"] = output.loc[:, list(core)].notna().all(axis=1)
    output["probe_intraday_source_status"] = ";".join(status)
    return output, {
        "symbol": symbol, "rows": int(len(output)), "source_ready_rows": int(output["probe_intraday_source_ready"].sum()),
        "status": ";".join(status),
    }


def _benchmark_returns(decision_timestamps: pd.Series) -> pd.DataFrame:
    """Materialise BTC/ETH completed-bar returns even when they are not candidates."""
    timestamps = pd.DatetimeIndex(pd.to_datetime(decision_timestamps, utc=True, errors="raise").unique()).sort_values()
    if timestamps.empty:
        return pd.DataFrame()
    columns: dict[str, pd.Series] = {}
    for symbol in ("BTC/USD:USD", "ETH/USD:USD"):
        bars, error = _read_15m(symbol, start=timestamps.min(), end=timestamps.max())
        if error is not None or bars is None or bars.empty:
            continue
        intraday = materialize_opportunity_probe_intraday_features(bars)
        columns[symbol] = intraday.reindex(timestamps)["probe_return_1h"]
    return pd.DataFrame(columns, index=timestamps)


def _write_parquet_exclusive(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=str, default=None, help="inclusive UTC month/date")
    parser.add_argument("--end", type=str, default=None, help="inclusive UTC month/date")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"immutable output already exists: {output}")
    start = args.start or config["research_period"][0]
    end = args.end or config["research_period"][1]
    months = _month_starts(start, end)
    output.mkdir(parents=True, exist_ok=False)
    audits: list[dict[str, Any]] = []
    for month in months:
        candidates = _load_candidates(config, month)
        pieces: list[pd.DataFrame] = []
        symbol_audit: list[dict[str, Any]] = []
        for _, group in candidates.groupby("__symbol__", sort=True, observed=True):
            materialized, audit = _symbol_features(group.reset_index(drop=True))
            pieces.append(materialized)
            symbol_audit.append(audit)
        joined = pd.concat(pieces, ignore_index=True)
        if joined["candidate_id"].duplicated().any() or len(joined) != len(candidates):
            raise AssertionError(f"{month:%Y-%m}: sidecar changed target-free candidate identity")
        joined = add_opportunity_probe_cross_sectional_features(
            joined, benchmark_returns=_benchmark_returns(joined["__decision_ts__"])
        )
        for field in OPPORTUNITY_PROBE_INTRADAY_FEATURE_KEYS:
            if field not in joined:
                raise AssertionError(f"{month:%Y-%m}: missing materialized field {field}")
        ordered = joined.loc[:, [*IDENTITY, *OPPORTUNITY_PROBE_INTRADAY_FEATURE_KEYS, "probe_intraday_source_ready", "probe_intraday_source_status"]].copy()
        _write_parquet_exclusive(ordered, output / f"month={month:%Y-%m}" / "causal_probe_intraday_features.parquet")
        field_coverage = {field: float(ordered[field].notna().mean()) for field in OPPORTUNITY_PROBE_INTRADAY_FEATURE_KEYS}
        audits.append({
            "month": f"{month:%Y-%m}", "candidate_rows": int(len(candidates)), "sidecar_rows": int(len(ordered)),
            "source_ready_rows": int(ordered["probe_intraday_source_ready"].sum()),
            "source_ready_fraction": float(ordered["probe_intraday_source_ready"].mean()),
            "field_coverage": field_coverage, "symbols": symbol_audit,
        })
        print(json.dumps({k: audits[-1][k] for k in ("month", "candidate_rows", "source_ready_rows", "source_ready_fraction")}, sort_keys=True))
    manifest = {
        "schema": SCHEMA, "status": "complete", "side": config["side"], "config": str(config_path),
        "config_sha256": _sha256(config_path), "start": str(_utc(start)), "end": str(_utc(end)),
        "feature_fields": OPPORTUNITY_PROBE_INTRADAY_FEATURE_KEYS,
        "candidate_identity_contract": list(IDENTITY),
        "causal_rule": "decision T uses only bars ending no later than T; 15m source is shifted +15m and hourly source +1h",
        "audits": audits,
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
