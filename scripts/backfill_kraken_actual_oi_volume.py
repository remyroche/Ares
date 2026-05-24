#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from extreme_price_movements.data_store import (
    _fetch_kraken_futures_open_interest_analytics,
    make_perp_exchange,
)
from extreme_price_movements.kraken_actual_data import (
    aggregate_trades_to_hourly,
    fetch_trades_paged,
    load_partitioned_ohlcv_symbol,
    load_sidecar_frame,
    load_sidecar_series,
    load_verified_perp_symbols,
    plan_symbol_coverage,
    safe_symbol,
    symbol_key_from_symbol,
    write_actual_volume_sidecar,
)
from extreme_price_movements.utils import tprint


VOLUME_MODEL_CONTRACT_COLUMNS = (
    "dist_vwap_norm",
    "vwap_zone_1d_atr",
    "vwap_zone_7d_atr",
    "dist_vwap_12_atr",
    "dist_vwap_24_atr",
    "dist_vwap_96_atr",
    "trapped_longs_12",
    "trapped_longs_24",
    "trapped_longs_96",
    "dist_stack",
    "oi_rel_vol_2h",
    "oi_rel_vol_4h",
    "oi_rel_vol_8h",
    "oi_value_log_1d_robust_z",
    "oi_value_log_7d_robust_z",
    "oi_chg_2h_robust_z",
    "oi_chg_4h_robust_z",
    "oi_chg_8h_robust_z",
    "unwind_score",
)


def _load_manifest_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("symbols") if isinstance(payload, dict) else payload
    out: list[dict[str, Any]] = []
    for row in rows or []:
        if isinstance(row, dict):
            if row.get("perp_symbol"):
                out.append(dict(row))
        elif row:
            out.append({"perp_symbol": str(row)})
    return out


def _filter_rows(
    rows: list[dict[str, Any]],
    *,
    symbols: str,
    limit_symbols: int,
    partition_count: int,
    partition_id: int,
) -> list[dict[str, Any]]:
    allow = {s.strip() for s in str(symbols or "").split(",") if s.strip()}
    if allow:
        rows = [row for row in rows if str(row.get("perp_symbol")) in allow]
    partition_count = max(1, int(partition_count))
    partition_id = int(partition_id)
    if partition_id < 0 or partition_id >= partition_count:
        raise ValueError(f"invalid partition {partition_id}/{partition_count}")
    rows = [row for i, row in enumerate(rows) if i % partition_count == partition_id]
    if limit_symbols > 0:
        rows = rows[: int(limit_symbols)]
    return rows


def _load_symbol_plan(perp_root: Path, symbol: str, max_gap_hours: int, retry_unavailable: bool):
    symbol_key = symbol_key_from_symbol(symbol)
    raw = load_partitioned_ohlcv_symbol(perp_root / "ohlcv", symbol_key)
    oi = load_sidecar_series(perp_root / "open_interest_hourly", symbol, "open_interest")
    funding = load_sidecar_series(perp_root / "funding_hourly", symbol, "funding_rate")
    actual_volume = load_sidecar_frame(perp_root / "actual_volume_hourly", symbol)
    return plan_symbol_coverage(
        symbol_key=symbol_key,
        ohlcv=raw,
        oi_sidecar=oi,
        actual_volume_sidecar=actual_volume,
        funding_sidecar=funding,
        max_gap_hours=max_gap_hours,
        retry_unavailable_volume=retry_unavailable,
    )


def _write_oi_sidecar(path: Path, existing: pd.Series, incoming: pd.Series) -> tuple[int, int]:
    before = int(existing.notna().sum()) if existing is not None else 0
    pieces = []
    if existing is not None and not existing.empty:
        pieces.append(existing)
    if incoming is not None and not incoming.empty:
        pieces.append(incoming)
    if not pieces:
        return before, before
    merged = pd.concat(pieces).sort_index().groupby(level=0).last()
    merged = merged.replace([np.inf, -np.inf], np.nan).where(lambda s: s > 0.0).dropna()
    out = merged.rename("open_interest").to_frame().astype({"open_interest": "float32"})
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    out.to_parquet(tmp, compression="zstd")
    tmp.replace(path)
    return before, int(out["open_interest"].notna().sum())


def _run_report(args: argparse.Namespace, rows: list[dict[str, Any]]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    perp_root = Path(args.perp_root)
    for row in rows:
        symbol = str(row["perp_symbol"])
        coverage, oi_ranges, volume_ranges = _load_symbol_plan(
            perp_root,
            symbol,
            int(args.max_gap_hours),
            bool(args.retry_unavailable_volume),
        )
        rec = dict(coverage.__dict__)
        rec["perp_symbol"] = symbol
        rec["oi_gap_ranges"] = len(oi_ranges)
        rec["volume_gap_ranges"] = len(volume_ranges)
        records.append(rec)
    report = pd.DataFrame(records)
    if not report.empty:
        totals = {
            "symbols": int(len(report)),
            "price_rows": int(report["price_rows"].sum()),
            "missing_oi": int(report["missing_oi"].sum()),
            "missing_volume": int(report["missing_volume"].sum()),
            "missing_funding": int(report["missing_funding"].sum()),
            "missing_any": int(report["missing_any"].sum()),
            "valid_all": int(report["valid_all"].sum()),
            "oi_gap_ranges": int(report["oi_gap_ranges"].sum()),
            "volume_gap_ranges": int(report["volume_gap_ranges"].sum()),
            "actual_trades": int(report["actual_trades"].sum()),
            "confirmed_no_trades": int(report["confirmed_no_trades"].sum()),
            "source_conflict_volume": int(report.get("source_conflict_volume", pd.Series(dtype=int)).sum()),
            "unavailable_volume": int(report["unavailable_volume"].sum()),
        }
    else:
        totals = {"symbols": 0}
    print(json.dumps({"totals": totals}, indent=2, sort_keys=True), flush=True)
    if args.report:
        path = Path(args.report)
        if not args.dry_run:
            path.parent.mkdir(parents=True, exist_ok=True)
            report.to_csv(path, index=False)
        else:
            print(f"dry-run: report not written to {path}", flush=True)
    return report


def _run_oi_backfill(args: argparse.Namespace, rows: list[dict[str, Any]]) -> dict[str, Any]:
    perp_root = Path(args.perp_root)
    out_dir = perp_root / "open_interest_hourly"
    exchange = None
    stats = {"symbols": len(rows), "updated": 0, "no_gaps": 0, "fetched_rows": 0, "failed": []}
    for i, row in enumerate(rows, start=1):
        symbol = str(row["perp_symbol"])
        try:
            coverage, oi_ranges, _volume_ranges = _load_symbol_plan(
                perp_root,
                symbol,
                int(args.max_gap_hours),
                bool(args.retry_unavailable_volume),
            )
            if not oi_ranges:
                stats["no_gaps"] += 1
                continue
            tprint(f"[OI {i:04d}/{len(rows):04d}] {symbol}: gap_ranges={len(oi_ranges)}")
            if args.dry_run:
                continue
            if exchange is None:
                exchange = make_perp_exchange()
                exchange.rateLimit = max(int(getattr(exchange, "rateLimit", 0) or 0), int(args.rate_limit_ms))
            parts = []
            for start, end in oi_ranges:
                series = _fetch_kraken_futures_open_interest_analytics(
                    exchange,
                    symbol,
                    int(start.value // 10**6),
                    int(end.value // 10**6),
                    timeframe="1h",
                )
                if not series.empty:
                    parts.append(series)
                time.sleep(max(0.0, float(args.sleep)))
            if not parts:
                continue
            incoming = pd.concat(parts).sort_index().groupby(level=0).last()
            incoming = incoming.replace([np.inf, -np.inf], np.nan).where(lambda s: s > 0.0).dropna()
            if incoming.empty:
                continue
            existing = load_sidecar_series(out_dir, symbol, "open_interest")
            before, after = _write_oi_sidecar(out_dir / f"{safe_symbol(symbol)}.parquet", existing, incoming)
            stats["updated"] += 1
            stats["fetched_rows"] += int(max(0, after - before))
            tprint(f"  {symbol}: oi_rows={before}->{after}")
        except Exception as exc:
            msg = f"{symbol}: {exc.__class__.__name__}: {exc}"
            stats["failed"].append(msg)
            tprint(f"[OI {i:04d}/{len(rows):04d}] FAIL {msg}")
    print(json.dumps(stats, indent=2, sort_keys=True), flush=True)
    return stats


def _unavailable_hours_frame(
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    source: str,
    reason: str,
) -> pd.DataFrame:
    hours = pd.date_range(start.floor("h"), end.floor("h") - pd.Timedelta(hours=1), freq="1h", tz="UTC")
    out = pd.DataFrame(index=hours)
    out.index.name = "ts"
    out["volume"] = np.nan
    out["quote_volume"] = np.nan
    out["trade_count"] = np.nan
    out["vwap"] = np.nan
    out["source"] = source
    out["coverage_status"] = "unavailable"
    out["reason"] = reason
    return out


def _coalesce_fetch_ranges(
    ranges: list[tuple[pd.Timestamp, pd.Timestamp]],
    *,
    max_window_hours: int,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if not ranges:
        return []
    max_delta = pd.Timedelta(hours=max(1, int(max_window_hours)))
    out: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    cur_start, cur_end = ranges[0]
    for start, end in ranges[1:]:
        if end - cur_start <= max_delta:
            cur_end = max(cur_end, end)
            continue
        out.append((cur_start, cur_end))
        cur_start, cur_end = start, end
    out.append((cur_start, cur_end))
    return out


def _run_perp_volume_backfill(args: argparse.Namespace, rows: list[dict[str, Any]]) -> dict[str, Any]:
    perp_root = Path(args.perp_root)
    out_dir = perp_root / "actual_volume_hourly"
    exchange = None
    stats = {
        "symbols": len(rows),
        "updated": 0,
        "no_gaps": 0,
        "actual_trade_hours": 0,
        "confirmed_no_trade_hours": 0,
        "unavailable_hours": 0,
        "failed": [],
    }
    for i, row in enumerate(rows, start=1):
        symbol = str(row["perp_symbol"])
        try:
            _coverage, _oi_ranges, volume_ranges = _load_symbol_plan(
                perp_root,
                symbol,
                int(args.max_gap_hours),
                bool(args.retry_unavailable_volume),
            )
            if not volume_ranges:
                stats["no_gaps"] += 1
                continue
            fetch_ranges = _coalesce_fetch_ranges(
                volume_ranges,
                max_window_hours=int(args.volume_fetch_window_hours),
            )
            tprint(
                f"[VOL {i:04d}/{len(rows):04d}] {symbol}: "
                f"gap_ranges={len(volume_ranges)} fetch_windows={len(fetch_ranges)}"
            )
            if args.dry_run:
                continue
            if exchange is None:
                exchange = make_perp_exchange()
                exchange.rateLimit = max(int(getattr(exchange, "rateLimit", 0) or 0), int(args.rate_limit_ms))
            pieces = []
            for j, (start, end) in enumerate(fetch_ranges, start=1):
                if j == 1 or j % int(args.progress_every) == 0 or j == len(fetch_ranges):
                    tprint(
                        f"  {symbol}: fetch_window {j}/{len(fetch_ranges)} "
                        f"{start.isoformat()}->{end.isoformat()}"
                    )
                trades, complete, reason = fetch_trades_paged(
                    exchange,
                    symbol,
                    start_ts=start,
                    end_ts=end,
                    limit=int(args.trade_limit),
                    max_pages=int(args.max_trade_pages),
                    sleep_seconds=float(args.sleep),
                )
                if complete:
                    fill_empty = bool(trades) or args.empty_fetch_status == "confirmed_no_trades"
                    empty_status = "confirmed_no_trades" if fill_empty else "unavailable"
                    if fill_empty:
                        frame = aggregate_trades_to_hourly(
                            trades,
                            start_ts=start,
                            end_ts=end,
                            source="krakenfutures_fetch_trades",
                            fill_empty_hours=True,
                            empty_status=empty_status,
                        )
                    else:
                        frame = _unavailable_hours_frame(
                            start,
                            end,
                            source="krakenfutures_fetch_trades",
                            reason=f"empty_fetch:{reason}",
                        )
                else:
                    frame = _unavailable_hours_frame(
                        start,
                        end,
                        source="krakenfutures_fetch_trades",
                        reason=reason,
                    )
                if not frame.empty:
                    pieces.append(frame)
                time.sleep(max(0.0, float(args.sleep)))
            if not pieces:
                continue
            incoming = pd.concat(pieces).sort_index().groupby(level=0).last()
            before, after = write_actual_volume_sidecar(
                out_dir / f"{safe_symbol(symbol)}.parquet",
                incoming,
            )
            stats["updated"] += 1
            status = incoming.get("coverage_status", pd.Series(dtype=str)).astype(str)
            stats["actual_trade_hours"] += int(status.eq("actual_trades").sum())
            stats["confirmed_no_trade_hours"] += int(status.eq("confirmed_no_trades").sum())
            stats["unavailable_hours"] += int(status.eq("unavailable").sum())
            tprint(f"  {symbol}: actual_volume_rows={before}->{after}")
        except Exception as exc:
            msg = f"{symbol}: {exc.__class__.__name__}: {exc}"
            stats["failed"].append(msg)
            tprint(f"[VOL {i:04d}/{len(rows):04d}] FAIL {msg}")
    print(json.dumps(stats, indent=2, sort_keys=True), flush=True)
    return stats


def _spot_archive_pair_keys(spot_symbol: str) -> list[str]:
    if not spot_symbol or "/" not in spot_symbol:
        return []
    base, raw_quote = spot_symbol.split("/", 1)
    quote = raw_quote.split(":", 1)[0]
    bases = [base.upper()]
    if base.upper() == "BTC":
        bases.append("XBT")
    if base.upper() == "DOGE":
        bases.append("XDG")
    quotes = [quote.upper()]
    out = []
    for b in bases:
        for q in quotes:
            out.append(f"{b}{q}_60.csv")
    return list(dict.fromkeys(out))


def _run_spot_archive_volume(args: argparse.Namespace, rows: list[dict[str, Any]]) -> dict[str, Any]:
    archive_path = Path(args.spot_ohlcvt_zip)
    stats = {"symbols": len(rows), "updated": 0, "missing_member": 0, "failed": []}
    if not archive_path.exists():
        raise FileNotFoundError(f"spot OHLCVT archive not found: {archive_path}")
    out_dir = Path(args.spot_root) / "actual_volume_hourly"
    with zipfile.ZipFile(archive_path) as archive:
        names = {Path(name).name.upper(): name for name in archive.namelist() if name.lower().endswith(".csv")}
        for i, row in enumerate(rows, start=1):
            spot_symbol = str(row.get("spot_symbol") or "")
            if not spot_symbol:
                stats["missing_member"] += 1
                continue
            member = None
            for key in _spot_archive_pair_keys(spot_symbol):
                member = names.get(key.upper())
                if member:
                    break
            if member is None:
                stats["missing_member"] += 1
                continue
            try:
                with archive.open(member) as handle:
                    df = pd.read_csv(
                        handle,
                        header=None,
                        names=["ts", "open", "high", "low", "close", "volume", "trades"],
                        usecols=[0, 4, 5, 6],
                    )
                if df.empty:
                    continue
                unit = "ms" if pd.to_numeric(df["ts"], errors="coerce").max() > 10**11 else "s"
                df["ts"] = pd.to_datetime(df["ts"], unit=unit, utc=True, errors="coerce")
                df = df.dropna(subset=["ts"]).set_index("ts").sort_index()
                out = pd.DataFrame(index=df.index.floor("h"))
                out["volume"] = pd.to_numeric(df["volume"], errors="coerce").to_numpy()
                out["trade_count"] = pd.to_numeric(df["trades"], errors="coerce").to_numpy()
                close = pd.to_numeric(df["close"], errors="coerce")
                out["quote_volume"] = out["volume"].to_numpy() * close.to_numpy()
                out["vwap"] = close.to_numpy()
                out["source"] = "kraken_spot_ohlcvt_archive"
                out["coverage_status"] = np.where(out["trade_count"].gt(0), "actual_trades", "confirmed_no_trades")
                out = out.groupby(level=0).agg(
                    volume=("volume", "sum"),
                    quote_volume=("quote_volume", "sum"),
                    trade_count=("trade_count", "sum"),
                    vwap=("vwap", "last"),
                    source=("source", "last"),
                    coverage_status=("coverage_status", "last"),
                )
                if args.dry_run:
                    tprint(f"[SPOT {i:04d}/{len(rows):04d}] {spot_symbol}: archive_rows={len(out)} dry-run")
                    continue
                before, after = write_actual_volume_sidecar(out_dir / f"{safe_symbol(spot_symbol)}.parquet", out)
                stats["updated"] += 1
                tprint(f"[SPOT {i:04d}/{len(rows):04d}] {spot_symbol}: rows={before}->{after}")
            except Exception as exc:
                stats["failed"].append(f"{spot_symbol}: {exc.__class__.__name__}: {exc}")
    print(json.dumps(stats, indent=2, sort_keys=True), flush=True)
    return stats


def _latest_feature_dir(data_root: Path) -> Path:
    feature_root = data_root / "features"
    candidates = [path for path in feature_root.iterdir() if path.is_dir()] if feature_root.exists() else []
    if not candidates:
        raise FileNotFoundError(f"no feature directories under {feature_root}")
    return sorted(candidates, key=lambda p: p.name)[-1]


def _run_volume_feature_recompute(args: argparse.Namespace) -> dict[str, Any]:
    feature_dir = Path(args.feature_dir) if args.feature_dir else _latest_feature_dir(Path(args.data_root))
    report_dir = Path(args.feature_report_dir)
    if not args.feature_report_dir:
        run_id = feature_dir.name
        report_dir = Path(args.data_root) / "artifacts" / run_id / "features"
    if not args.dry_run:
        report_dir.mkdir(parents=True, exist_ok=True)
    raw_root = Path(args.perp_root)
    commands = [
        [
            sys.executable,
            "-u",
            "scripts/recompute_kraken_ohlcv_contract_features.py",
            "--feature-dir",
            str(feature_dir),
            "--raw-root",
            str(raw_root),
            "--columns",
            ",".join(VOLUME_MODEL_CONTRACT_COLUMNS),
            "--report",
            str(report_dir / "actual_volume_model_contract_recompute_report.csv"),
        ],
        [
            sys.executable,
            "-u",
            "scripts/recompute_kraken_oi_crowding_features.py",
            "--feature-dir",
            str(feature_dir),
            "--raw-root",
            str(raw_root),
            "--report",
            str(report_dir / "actual_volume_oi_crowding_recompute_report.csv"),
        ],
    ]
    if args.dry_run:
        for command in commands:
            command.append("--dry-run")
    results = []
    for command in commands:
        tprint("Running targeted feature recompute: " + " ".join(command))
        completed = subprocess.run(command, cwd=Path.cwd(), check=False)
        results.append({"command": command, "returncode": int(completed.returncode)})
        if completed.returncode != 0:
            break
    summary = {
        "feature_dir": str(feature_dir),
        "report_dir": str(report_dir),
        "dry_run": bool(args.dry_run),
        "results": results,
    }
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data_perp/exchanges/krakenfutures/manifests/kraken_dual_market_verified_universe_latest.json")
    parser.add_argument("--perp-root", default="data_perp/exchanges/krakenfutures")
    parser.add_argument("--spot-root", default="data_spot/exchanges/kraken")
    parser.add_argument(
        "--action",
        action="append",
        choices=("report", "oi", "perp-volume", "spot-volume", "recompute-volume-features", "all"),
        default=None,
    )
    parser.add_argument("--data-root", default="data_perp")
    parser.add_argument("--feature-dir", default="")
    parser.add_argument("--feature-report-dir", default="")
    parser.add_argument("--symbols", default="")
    parser.add_argument("--limit-symbols", type=int, default=0)
    parser.add_argument("--partition-count", type=int, default=1)
    parser.add_argument("--partition-id", type=int, default=0)
    parser.add_argument("--max-gap-hours", type=int, default=720)
    parser.add_argument("--retry-unavailable-volume", action="store_true")
    parser.add_argument("--rate-limit-ms", type=int, default=250)
    parser.add_argument("--sleep", type=float, default=0.05)
    parser.add_argument("--trade-limit", type=int, default=1000)
    parser.add_argument("--max-trade-pages", type=int, default=200)
    parser.add_argument("--volume-fetch-window-hours", type=int, default=720)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument(
        "--empty-fetch-status",
        choices=("unavailable", "confirmed_no_trades"),
        default="unavailable",
    )
    parser.add_argument("--spot-ohlcvt-zip", default="data_spot/exchanges/kraken/raw/Kraken_OHLCVT.zip")
    parser.add_argument("--report", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rows = _filter_rows(
        _load_manifest_rows(Path(args.manifest)),
        symbols=args.symbols,
        limit_symbols=int(args.limit_symbols),
        partition_count=int(args.partition_count),
        partition_id=int(args.partition_id),
    )
    actions = args.action or ["report"]
    if "all" in actions:
        actions = ["report", "oi", "perp-volume", "spot-volume", "recompute-volume-features"]

    failed = False
    if "report" in actions:
        _run_report(args, rows)
    if "oi" in actions:
        failed = bool(_run_oi_backfill(args, rows).get("failed")) or failed
    if "perp-volume" in actions:
        failed = bool(_run_perp_volume_backfill(args, rows).get("failed")) or failed
    if "spot-volume" in actions:
        failed = bool(_run_spot_archive_volume(args, rows).get("failed")) or failed
    if "recompute-volume-features" in actions:
        result = _run_volume_feature_recompute(args)
        failed = any(item.get("returncode") != 0 for item in result.get("results", [])) or failed
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
