"""Backfill canonical Kraken PF OI/funding sidecars.

This is deliberately separate from ``PartitionedOHLCVStore.update_symbol_perp``.
The latter writes event-time auxiliary columns into the OHLCV store, while the
canonical sidecar contract is indexed at ``availability_ts = observation_ts +
1h``.  This script downloads only PF-linear USD products, shifts newly fetched
observations exactly once, and fills missing sidecar timestamps without
overwriting existing canonical values.

The script is safe to resume: each symbol/family is merged atomically and the
audit records endpoint errors and skipped products rather than synthesising
values.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import urllib3


HOUR = pd.Timedelta(hours=1)
OI_API_FLOOR = pd.Timestamp("2023-03-07", tz="UTC")
USER_AGENT = "Ares-canonical-oi-funding-backfill/1.0"
PF_SUFFIX = ":USD"
ALIASES = {"BTC": "XBT", "XBT": "XBT"}


@dataclass(frozen=True)
class Product:
    feature_symbol: str
    product_id: str | None
    sidecar_key: str
    status: str
    reason: str = ""


def _utc(value: Any) -> pd.Timestamp:
    return pd.Timestamp(value, tz="UTC") if not isinstance(value, pd.Timestamp) else value.tz_convert("UTC")


def _safe_key(symbol: str) -> str:
    return str(symbol).replace("/", "_").replace(":", "_")


def _parse_pf_product(symbol: str) -> Product:
    # Accept both the feature-store spelling (``BTC_USD:USD``) and the frozen
    # inference-universe spelling (``BTC/USD:USD``).  They denote the same PF
    # linear product and collapse to the same canonical sidecar key.
    text = str(symbol).strip().upper().replace("/", "_")
    key = _safe_key(text)
    # Only the canonical linear USD contract is admitted automatically.
    if not text.endswith(PF_SUFFIX):
        return Product(text, None, key, "SKIP", "non_pf_linear_product")
    base = text.rsplit("_USD:USD", 1)[0].split("_", 1)[0]
    base = ALIASES.get(base, base)
    if not re.fullmatch(r"[A-Z0-9]+", base):
        return Product(text, None, key, "SKIP", "invalid_base")
    return Product(text, f"PF_{base}USD", key, "READY")


def _read_products(
    feature_dir: Path,
    symbols_file: Path | None,
    excluded_symbols: set[str] | None = None,
) -> list[Product]:
    if symbols_file is not None:
        text = symbols_file.read_text()
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            if isinstance(payload.get("source_map"), dict):
                symbols = list(payload["source_map"])
            else:
                symbols = list(
                    payload.get("symbols")
                    or payload.get("canonical_symbols")
                    or payload.get("expanded_symbols")
                    or []
                )
        elif isinstance(payload, list):
            symbols = list(payload)
        else:
            symbols = [x.strip() for x in text.splitlines() if x.strip()]
    else:
        symbols = [p.stem.removeprefix("symbol=") for p in sorted(feature_dir.glob("symbol=*.parquet"))]
    out: list[Product] = []
    seen: set[str] = set()
    excluded = set(excluded_symbols or ())
    for symbol in symbols:
        if symbol in excluded:
            continue
        product = _parse_pf_product(symbol)
        if product.feature_symbol in seen:
            continue
        seen.add(product.feature_symbol)
        out.append(product)
    return out


def _http_get(session: requests.Session, url: str, params: dict[str, Any]) -> requests.Response:
    try:
        response = session.get(url, params=params, timeout=45, headers={"User-Agent": USER_AGENT})
    except requests.exceptions.SSLError:
        # macOS worker environments may not expose the system CA bundle.  The
        # endpoint is public; record the fallback explicitly in the manifest.
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        response = session.get(url, params=params, timeout=45, headers={"User-Agent": USER_AGENT}, verify=False)
    response.raise_for_status()
    return response


def _parse_oi_payload(payload: dict[str, Any]) -> pd.Series:
    result = payload.get("result", {}) if isinstance(payload, dict) else {}
    timestamps = result.get("timestamp", []) if isinstance(result, dict) else []
    data = result.get("data", []) if isinstance(result, dict) else []
    rows: list[tuple[pd.Timestamp, float]] = []
    if not isinstance(timestamps, list) or not isinstance(data, list):
        return pd.Series(dtype="float32")
    for raw_ts, raw_value in zip(timestamps, data):
        try:
            ts = pd.to_datetime(float(raw_ts), unit="s", utc=True).floor("h")
        except Exception:
            continue
        values = raw_value if isinstance(raw_value, (list, tuple)) else [raw_value]
        value = None
        for item in reversed(values):
            try:
                candidate = float(item)
            except Exception:
                continue
            if np.isfinite(candidate) and candidate > 0:
                value = candidate
                break
        if value is not None:
            rows.append((ts, value))
    if not rows:
        return pd.Series(dtype="float32")
    return pd.Series(dict(rows), dtype="float32").sort_index()


def _parse_funding_payload(payload: dict[str, Any]) -> pd.Series:
    rates = payload.get("rates", []) if isinstance(payload, dict) else []
    rows: list[tuple[pd.Timestamp, float]] = []
    if not isinstance(rates, list):
        return pd.Series(dtype="float32")
    for item in rates:
        if not isinstance(item, dict):
            continue
        ts = pd.to_datetime(item.get("timestamp"), utc=True, errors="coerce")
        if pd.isna(ts):
            continue
        raw = item.get("relativeFundingRate", item.get("fundingRate"))
        try:
            value = float(raw)
        except Exception:
            continue
        if np.isfinite(value):
            rows.append((pd.Timestamp(ts).floor("h"), value))
    if not rows:
        return pd.Series(dtype="float32")
    return pd.Series(dict(rows), dtype="float32").sort_index()


def _download_oi(product: Product, start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.Series, dict[str, Any]]:
    if product.product_id is None:
        return pd.Series(dtype="float32"), {"status": "SKIP", "reason": product.reason}
    # The public chart API has no useful rows before the current OI retention
    # boundary.  Starting at that floor avoids an empty response terminating
    # the pagination loop before later history is reached.
    cursor = max(start, OI_API_FLOOR)
    rows: list[pd.Series] = []
    calls = 0
    more = True
    session = requests.Session()
    endpoint = f"https://futures.kraken.com/api/charts/v1/analytics/{product.product_id}/open-interest"
    try:
        while cursor < end and more:
            payload = _http_get(session, endpoint, {"since": int(cursor.timestamp()), "to": int(end.timestamp()), "interval": 3600}).json()
            calls += 1
            series = _parse_oi_payload(payload)
            if not series.empty:
                series = series[(series.index >= cursor) & (series.index < end)]
                if not series.empty:
                    rows.append(series)
                    cursor = series.index.max() + HOUR
            result = payload.get("result", {}) if isinstance(payload, dict) else {}
            more = bool(result.get("more")) if isinstance(result, dict) else False
            if series.empty:
                # Do not spin on empty pages.  A product with no rows in the
                # requested window is recorded as unavailable.
                break
            time.sleep(0.1)
    except Exception as exc:
        return pd.Series(dtype="float32"), {"status": "ERROR", "error": f"{type(exc).__name__}: {exc}", "calls": calls}
    if not rows:
        return pd.Series(dtype="float32"), {"status": "EMPTY", "calls": calls, "more_final": more}
    out = pd.concat(rows).groupby(level=0).last().sort_index()
    return out.astype("float32"), {"status": "OK", "calls": calls, "rows": int(len(out)), "more_final": more}


def _download_funding(product: Product, start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.Series, dict[str, Any]]:
    if product.product_id is None:
        return pd.Series(dtype="float32"), {"status": "SKIP", "reason": product.reason}
    session = requests.Session()
    endpoint = "https://futures.kraken.com/derivatives/api/v3/historical-funding-rates"
    try:
        payload = _http_get(session, endpoint, {"symbol": product.product_id}).json()
        series = _parse_funding_payload(payload)
        if not series.empty:
            series = series[(series.index >= start - HOUR) & (series.index < end)]
        return series.astype("float32"), {"status": "OK" if not series.empty else "EMPTY", "rows": int(len(series)), "endpoint_result": payload.get("result") if isinstance(payload, dict) else None}
    except Exception as exc:
        return pd.Series(dtype="float32"), {"status": "ERROR", "error": f"{type(exc).__name__}: {exc}"}


def _load_funding_archive(product: Product, archive_path: Path | None, start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.Series, dict[str, Any]]:
    """Read the official PF export as observations; its timestamps are shifted below."""
    if archive_path is None or product.product_id is None:
        return pd.Series(dtype="float32"), {"status": "SKIP" if archive_path is None else "EMPTY"}
    member = f"exports/{product.product_id}.csv"
    try:
        with zipfile.ZipFile(archive_path) as archive:
            if member not in archive.namelist():
                return pd.Series(dtype="float32"), {"status": "MISSING_MEMBER", "member": member}
            with archive.open(member) as handle:
                raw = pd.read_csv(handle, usecols=["timestamp", "relative_rate"])
    except Exception as exc:
        return pd.Series(dtype="float32"), {"status": "ERROR", "error": f"{type(exc).__name__}: {exc}", "member": member}
    ts = pd.to_datetime(raw["timestamp"], utc=True, errors="coerce").dt.floor("h")
    values = pd.to_numeric(raw["relative_rate"], errors="coerce")
    valid = ts.notna() & np.isfinite(values.to_numpy(dtype=np.float64, copy=False))
    if not bool(valid.any()):
        return pd.Series(dtype="float32"), {"status": "EMPTY", "member": member}
    frame = pd.DataFrame({"ts": ts[valid], "value": values[valid].to_numpy(dtype="float32", copy=False)})
    series = frame.groupby("ts")["value"].last().sort_index().astype("float32")
    series = series[(series.index >= start - HOUR) & (series.index < end)]
    return series, {"status": "OK" if not series.empty else "EMPTY", "member": member, "rows": int(len(series))}


def _load_existing(path: Path, column: str) -> pd.Series:
    if not path.exists():
        return pd.Series(dtype="float32", index=pd.DatetimeIndex([], tz="UTC"))
    frame = pd.read_parquet(path)
    if frame.empty:
        return pd.Series(dtype="float32", index=pd.DatetimeIndex([], tz="UTC"))
    col = column if column in frame.columns else next(iter(frame.columns), None)
    if col is None:
        return pd.Series(dtype="float32", index=pd.DatetimeIndex([], tz="UTC"))
    idx = pd.to_datetime(frame.index, utc=True, errors="coerce")
    values = pd.to_numeric(frame[col], errors="coerce")
    out = pd.Series(values.to_numpy(dtype="float32"), index=idx, name=column)
    out = out.loc[~out.index.isna() & np.isfinite(out.to_numpy())]
    return out[~out.index.duplicated(keep="last")].sort_index()


def _atomic_write(series: pd.Series, path: Path, column: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = series.rename(column).to_frame()
    frame.index.name = "ts"
    with tempfile.NamedTemporaryFile(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False) as handle:
        tmp = Path(handle.name)
    try:
        frame.to_parquet(tmp, engine="pyarrow", compression="zstd")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _merge_one(
    product: Product,
    family: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    root: Path,
    funding_archive: Path | None = None,
    archive_only: bool = False,
) -> dict[str, Any]:
    column = "open_interest" if family == "open_interest_hourly" else "funding_rate"
    path = root / family / f"{product.sidecar_key}.parquet"
    existing = _load_existing(path, column)
    if family == "open_interest_hourly":
        observations, api = _download_oi(product, start, end)
    else:
        if archive_only:
            observations, api = _load_funding_archive(product, funding_archive, start, end)
        else:
            observations, api = _download_funding(product, start, end)
            if funding_archive is not None:
                archive_observations, archive_meta = _load_funding_archive(product, funding_archive, start, end)
                if not archive_observations.empty:
                    observations = pd.concat([archive_observations, observations]).groupby(level=0).last().sort_index().astype("float32")
                api["archive"] = archive_meta
    # New endpoint rows are observations.  Only after filtering do we shift to
    # the canonical availability index.  Existing canonical rows are retained
    # on duplicate timestamps so the backfill is gap-only and resumable.
    available = observations.copy()
    if not available.empty:
        available.index = pd.DatetimeIndex(available.index + HOUR, name="ts")
        available = available[(available.index >= start) & (available.index <= end)]
    added = available[~available.index.isin(existing.index)] if not available.empty else available
    merged = pd.concat([existing, added]).groupby(level=0).last().sort_index().astype("float32")
    if not merged.empty and (not path.exists() or len(added) > 0):
        _atomic_write(merged, path, column)
    return {
        "feature_symbol": product.feature_symbol,
        "product_id": product.product_id,
        "family": family,
        "path": str(path),
        "status": product.status if api.get("status") == "SKIP" else api.get("status"),
        "reason": product.reason,
        "existing_rows": int(len(existing)),
        "observed_rows": int(len(observations)),
        "available_rows": int(len(available)),
        "added_rows": int(len(added)),
        "final_rows": int(len(merged)),
        "existing_first_ts": existing.index.min().isoformat() if len(existing) else None,
        "existing_last_ts": existing.index.max().isoformat() if len(existing) else None,
        "api_first_observation_ts": observations.index.min().isoformat() if len(observations) else None,
        "api_last_observation_ts": observations.index.max().isoformat() if len(observations) else None,
        "api": api,
        "availability_shift_hours": 1,
        "preserved_existing_duplicates": True,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-dir", type=Path, required=True)
    ap.add_argument("--perp-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--symbols-file", type=Path)
    ap.add_argument("--symbols", default="", help="Comma-separated feature symbols for a bounded pilot.")
    ap.add_argument(
        "--exclude-finalization-json",
        type=Path,
        help="Exclude stale symbols listed by oi_funding_feature_finalization.json.",
    )
    ap.add_argument("--start-ts", default=None)
    ap.add_argument("--end-ts", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--skip-oi", action="store_true")
    ap.add_argument("--skip-funding", action="store_true")
    ap.add_argument("--funding-archive-zip", type=Path, help="Official Kraken PF funding export; rows are observations and shifted +1h.")
    ap.add_argument("--archive-only", action="store_true", help="For funding jobs, use the official export without calling the rolling API.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    excluded_symbols: set[str] = set()
    if args.exclude_finalization_json is not None:
        payload = json.loads(args.exclude_finalization_json.read_text())
        excluded_symbols = set(map(str, payload.get("stale_files_cleared", [])))
    if args.symbols.strip():
        pilot_fd, pilot_name = tempfile.mkstemp(prefix="ares_backfill_symbols_", suffix=".txt")
        os.close(pilot_fd)
        pilot = Path(pilot_name)
        try:
            pilot.write_text("\n".join(x.strip() for x in args.symbols.split(",") if x.strip()) + "\n")
            products = _read_products(args.feature_dir, pilot, excluded_symbols)
        finally:
            pilot.unlink(missing_ok=True)
    else:
        products = _read_products(args.feature_dir, args.symbols_file, excluded_symbols)
    all_indices = []
    for p in sorted(args.feature_dir.glob("symbol=*.parquet")):
        symbol = p.stem.removeprefix("symbol=")
        if symbol not in {x.feature_symbol for x in products}:
            continue
        try:
            frame = pd.read_parquet(p, columns=[])
            all_indices.append(pd.to_datetime(frame.index, utc=True, errors="coerce"))
        except Exception:
            continue
    start = _utc(args.start_ts) if args.start_ts else min((x.min() for x in all_indices if len(x)), default=pd.Timestamp("2022-05-19", tz="UTC"))
    end = _utc(args.end_ts) if args.end_ts else max((x.max() for x in all_indices if len(x)), default=pd.Timestamp.now(tz="UTC").floor("h"))
    ready = [p for p in products if p.status == "READY"]
    skipped = [asdict(p) for p in products if p.status != "READY"]
    manifest: dict[str, Any] = {
        "schema": "kraken_canonical_oi_funding_backfill_v1",
        "status": "DRY_RUN" if args.dry_run else "RUNNING",
        "feature_dir": str(args.feature_dir),
        "perp_root": str(args.perp_root),
        "start_ts": start.isoformat(),
        "end_ts": end.isoformat(),
        "availability_rule": "availability_ts = observation_ts + 1h",
        "product_contract": "PF linear USD only; PI/inverse and unresolved aliases skipped",
        "workers": max(1, int(args.workers)),
        "products_total": len(products),
        "products_ready": len(ready),
        "products_skipped": skipped,
        "excluded_symbols": sorted(excluded_symbols),
        "families": [x for x, enabled in (("open_interest_hourly", not args.skip_oi), ("funding_hourly", not args.skip_funding)) if enabled],
        "funding_archive_zip": str(args.funding_archive_zip) if args.funding_archive_zip else None,
        "funding_archive_only": bool(args.archive_only),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        (args.out_dir / "backfill_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return

    jobs = [(p, family) for p in ready for family in manifest["families"]]
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = [pool.submit(_merge_one, p, family, start, end, args.perp_root, args.funding_archive_zip, args.archive_only) for p, family in jobs]
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda x: (x["feature_symbol"], x["family"]))
    manifest["status"] = "COMPLETE"
    manifest["results"] = results
    status_counts = {status: sum(x["status"] == status for x in results) for status in sorted({x["status"] for x in results})}
    manifest["result_counts"] = {
        "jobs": len(results),
        "ok": sum(x["status"] == "OK" for x in results),
        "empty": sum(x["status"] == "EMPTY" for x in results),
        "error": sum(x["status"] == "ERROR" for x in results),
        "skipped": sum(x["status"] == "SKIP" for x in results),
        "unavailable": sum(x["status"] not in {"OK", "EMPTY", "ERROR", "SKIP"} for x in results),
        "status_counts": status_counts,
        "added_rows": int(sum(x["added_rows"] for x in results)),
    }
    (args.out_dir / "backfill_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"status": manifest["status"], **manifest["result_counts"]}, sort_keys=True))


if __name__ == "__main__":
    main()
