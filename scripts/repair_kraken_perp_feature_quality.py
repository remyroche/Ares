#!/usr/bin/env python3
"""Repair Kraken perp feature files where missing source data was encoded as zero.

The repair is intentionally conservative: it preserves real zero values on rows
with valid source data, and marks rows/features as NaN only when the required
source is unavailable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from extreme_price_movements.features import (
    _canonical_spot_index_from_ohlcv,
    _feature_source_requirements,
)


RANGE_COLUMNS = {12: "range_12h_pct", 16: "range_16h_pct", 24: "range_24h_pct"}


def _file_symbol(symbol: str) -> str:
    return str(symbol).replace("/", "_")


def _funding_file_symbol(symbol: str) -> str:
    return _file_symbol(symbol).replace(":", "_")


def _load_partitioned(root: Path, symbol: str) -> pd.DataFrame:
    files = sorted((root / f"symbol={symbol}").rglob("*.parquet"))
    if not files:
        return pd.DataFrame()
    frames = [pd.read_parquet(path) for path in files]
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    if "ts" not in out.columns:
        return pd.DataFrame()
    out["ts"] = pd.to_datetime(out["ts"], utc=True, errors="coerce")
    out = out.dropna(subset=["ts"]).sort_values("ts")
    out["ts_hour"] = out["ts"].dt.floor("h")
    return out


def _hourly_ohlcv(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    agg: dict[str, tuple[str, str]] = {}
    for col, op in (
        ("open", "first"),
        ("high", "max"),
        ("low", "min"),
        ("close", "last"),
        ("volume", "sum"),
        ("funding_rate", "last"),
        ("open_interest", "last"),
        ("mark_price", "last"),
        ("index_price", "last"),
        ("mark_close", "last"),
        ("index_close", "last"),
        ("spot_open", "first"),
        ("spot_high", "max"),
        ("spot_low", "min"),
        ("spot_close", "last"),
        ("spot_volume", "sum"),
    ):
        if col in raw.columns:
            agg[col] = (col, op)
    if not agg:
        return pd.DataFrame()
    return raw.groupby("ts_hour", sort=True).agg(**agg).astype("float32")


def _load_funding_hourly(root: Path, symbol: str) -> pd.DataFrame:
    path = root / f"{_funding_file_symbol(symbol)}.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        if "ts" not in df.columns:
            return pd.DataFrame()
        df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
        df = df.set_index("ts")
    df.index = pd.to_datetime(df.index, utc=True, errors="coerce").floor("h")
    return df[~df.index.duplicated(keep="last")].sort_index()


def _series(
    frame: pd.DataFrame,
    column: str,
    index: pd.DatetimeIndex,
    *,
    ffill_limit: int | None = None,
) -> pd.Series:
    if frame.empty or column not in frame.columns:
        return pd.Series(np.nan, index=index, dtype="float32")
    out = pd.to_numeric(frame[column], errors="coerce").reindex(index)
    if ffill_limit is not None:
        out = out.ffill(limit=max(0, int(ffill_limit)))
    return out.astype("float32")


def _mask(series: pd.Series, *, positive: bool = False) -> pd.Series:
    out = np.isfinite(series)
    if positive:
        out &= series > 0.0
    return out.fillna(False).astype(bool)


def _rolling_z(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=max(8, min(window, window // 4))).mean()
    std = series.rolling(window, min_periods=max(8, min(window, window // 4))).std()
    return ((series - mean) / std.replace(0.0, np.nan)).clip(-6.0, 6.0).astype("float32")


def _compute_ranges(raw: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    aligned = raw.reindex(index)
    out = pd.DataFrame(index=index)
    close = _series(aligned, "close", index).where(lambda x: x > 0.0)
    high = _series(aligned, "high", index)
    low = _series(aligned, "low", index)
    volume = _series(aligned, "volume", index).fillna(0.0)
    for window, column in RANGE_COLUMNS.items():
        hi = high.rolling(window, min_periods=window).max()
        lo = low.rolling(window, min_periods=window).min()
        vol_sum = volume.rolling(window, min_periods=window).sum()
        valid = _mask(hi) & _mask(lo) & _mask(close, positive=True) & (vol_sum > 0.0)
        out[column] = (((hi - lo) / (close + 1e-12)).replace([np.inf, -np.inf], np.nan)).where(valid).astype("float32")
    return out


def _load_spot_hourly(spot_root: Path, spot_symbol: str) -> pd.DataFrame:
    raw = _load_partitioned(spot_root / "ohlcv", _file_symbol(spot_symbol))
    if raw.empty:
        return pd.DataFrame()
    return _hourly_ohlcv(raw)


def _canonical_from_spot(
    *,
    perp_hourly: pd.DataFrame,
    spot_hourly: pd.DataFrame,
    perp_close: pd.Series,
    index: pd.DatetimeIndex,
) -> tuple[pd.Series, pd.Series]:
    embedded_close = _series(perp_hourly, "spot_close", index)
    external_close = _series(spot_hourly, "close", index)
    embedded_coverage = int((embedded_close > 0.0).sum())
    external_coverage = int((external_close > 0.0).sum())
    use_embedded = bool(
        embedded_coverage > 0
        and (external_coverage <= 0 or embedded_coverage >= int(0.90 * external_coverage))
    )
    source = perp_hourly if use_embedded else spot_hourly
    prefix = "spot_" if use_embedded else ""

    def col(name: str) -> pd.DataFrame:
        s = _series(source, f"{prefix}{name}", index)
        return s.rename("x").to_frame()

    canonical = _canonical_spot_index_from_ohlcv(
        spot_open=col("open"),
        spot_high=col("high"),
        spot_low=col("low"),
        spot_close=col("close"),
        spot_volume=col("volume"),
        fallback_price=col("close"),
        safe_log_eps=1e-9,
        kalman_lambda=0.05,
        bars_per_hour=1,
    )
    spot_close = col("close")["x"].where(lambda x: x > 0.0)
    if canonical is None:
        canonical_ser = spot_close
    else:
        canonical_ser = canonical["x"].reindex(index).where(lambda x: x > 0.0, spot_close)
    spot_available = _mask(spot_close, positive=True)
    canonical_ser = canonical_ser.where(spot_available)
    return canonical_ser.astype("float32"), spot_available


def _source_mask_for_feature(
    name: str,
    *,
    valid_ohlc: pd.Series,
    funding_mask: pd.Series,
    oi_mask: pd.Series,
    spot_mask: pd.Series,
    mark_mask: pd.Series,
    index_mask: pd.Series,
) -> pd.Series:
    out = valid_ohlc.copy()
    reqs = _feature_source_requirements(name)
    if "funding" in reqs:
        out &= funding_mask
    if "open_interest" in reqs:
        out &= oi_mask
    if "spot" in reqs:
        out &= spot_mask
    if "mark_index" in reqs:
        key = str(name)
        if (
            key.startswith("mark_")
            or key.startswith("mark_gap_")
            or key.startswith("mark_trigger_")
            or key.startswith("liq_")
            or "mark_vs" in key
        ):
            out &= mark_mask
        elif (
            key == "canonical_index"
            or key == "index_price"
            or key.startswith("premium_")
            or key.startswith("perp_index_")
        ):
            out &= spot_mask | index_mask
        else:
            out &= mark_mask | index_mask | spot_mask
    if "orderbook" in reqs:
        out &= False
    return out.fillna(False).astype(bool)


def _quality_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.select_dtypes(include=[np.number]).columns:
        s = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        finite = s.dropna()
        if finite.empty:
            rows.append((col, 0, np.nan, np.nan, np.nan, np.nan, np.nan))
            continue
        rows.append(
            (
                col,
                int(finite.size),
                float(finite.std()),
                float((finite == 0.0).mean()),
                float(finite.quantile(0.05)),
                float(finite.quantile(0.50)),
                float(finite.quantile(0.95)),
            )
        )
    return pd.DataFrame(rows, columns=["feature", "finite_n", "std", "zero_frac", "p05", "p50", "p95"])


def repair(args: argparse.Namespace) -> dict[str, object]:
    feature_dir = Path(args.feature_dir)
    perp_ohlcv_root = Path(args.perp_root) / "ohlcv"
    funding_root = Path(args.perp_root) / "funding_hourly"
    spot_root = Path(args.spot_root)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    spot_by_feature_symbol = {
        _file_symbol(item["perp_symbol"]): str(item["spot_symbol"])
        for item in manifest.get("symbols", [])
    }

    report_rows = []
    files = sorted(feature_dir.glob("symbol=*.parquet"))
    for i, path in enumerate(files, start=1):
        symbol = path.stem.removeprefix("symbol=")
        spot_symbol = spot_by_feature_symbol.get(symbol, symbol.removesuffix(":USD").replace("_", "/", 1))

        df = pd.read_parquet(path)
        if not isinstance(df.index, pd.DatetimeIndex):
            if "ts" not in df.columns:
                raise ValueError(f"{path} has no DatetimeIndex or ts column")
            df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
            df = df.set_index("ts")
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
        index = pd.DatetimeIndex(df.index, name="ts")
        df.index = index

        raw = _hourly_ohlcv(_load_partitioned(perp_ohlcv_root, symbol))
        funding = _load_funding_hourly(funding_root, symbol)
        spot = _load_spot_hourly(spot_root, spot_symbol)

        open_s = _series(raw, "open", index)
        high_s = _series(raw, "high", index)
        low_s = _series(raw, "low", index)
        close_s = _series(raw, "close", index)
        valid_ohlc = (
            _mask(open_s, positive=True)
            & _mask(high_s, positive=True)
            & _mask(low_s, positive=True)
            & _mask(close_s, positive=True)
            & (high_s >= low_s)
        )

        funding_s = _series(raw, "funding_rate", index, ffill_limit=12)
        if funding_s.notna().sum() == 0:
            funding_s = _series(funding, "funding_rate", index, ffill_limit=12)
        oi_s = _series(raw, "open_interest", index, ffill_limit=24)
        if oi_s.notna().sum() == 0:
            oi_s = _series(funding, "open_interest", index, ffill_limit=24)
        mark_s = _series(raw, "mark_price", index, ffill_limit=4)
        if mark_s.notna().sum() == 0:
            mark_s = _series(funding, "mark_price", index, ffill_limit=4)
        index_s = _series(raw, "index_price", index, ffill_limit=4)
        if index_s.notna().sum() == 0:
            index_s = _series(funding, "index_price", index, ffill_limit=4)

        funding_mask = _mask(funding_s)
        oi_mask = _mask(oi_s, positive=True)
        mark_mask = _mask(mark_s, positive=True)
        index_mask = _mask(index_s, positive=True)

        canonical, spot_mask = _canonical_from_spot(
            perp_hourly=raw,
            spot_hourly=spot,
            perp_close=close_s,
            index=index,
        )
        premium = ((close_s / (canonical + 1e-12)) - 1.0).clip(-0.10, 0.10)
        premium = premium.where(valid_ohlc & spot_mask)

        before_zero_heavy = 0
        numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != "__symbol__"]
        for col in numeric_cols:
            finite = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if len(finite) >= 1000 and float((finite == 0.0).mean()) >= 0.90:
                before_zero_heavy += 1

        range_df = _compute_ranges(raw, index)
        for col in RANGE_COLUMNS.values():
            if col in df.columns:
                df[col] = range_df[col]

        direct_values = {
            "canonical_index": canonical.where(valid_ohlc & spot_mask),
            "index_price": canonical.where(valid_ohlc & spot_mask),
            "premium_proxy": premium,
            "premium_proxy_bps": (premium * 1e4).clip(-1000, 1000),
            "basis": premium,
            "perp_index_basis": premium,
            "perp_vs_index_bps": (premium * 1e4).clip(-1000, 1000),
            "premium_proxy_z": _rolling_z(premium, 14 * 24),
            "perp_index_basis_z": _rolling_z(premium, 14 * 24),
            "premium_proxy_mom_8h": _rolling_z(premium.diff(8), 14 * 24),
        }
        for col, values in direct_values.items():
            if col in df.columns:
                df[col] = values.astype("float32")
        if "spot_available" in df.columns:
            df["spot_available"] = (valid_ohlc & spot_mask).astype("float32")

        for col in numeric_cols:
            source_mask = _source_mask_for_feature(
                col,
                valid_ohlc=valid_ohlc,
                funding_mask=funding_mask,
                oi_mask=oi_mask,
                spot_mask=spot_mask,
                mark_mask=mark_mask,
                index_mask=index_mask,
            )
            df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).where(source_mask).astype("float32")

        after_zero_heavy = 0
        for col in numeric_cols:
            finite = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if len(finite) >= 1000 and float((finite == 0.0).mean()) >= 0.90:
                after_zero_heavy += 1

        if not args.dry_run:
            tmp = path.with_suffix(".parquet.tmp")
            df.to_parquet(tmp, compression="zstd")
            tmp.replace(path)

        report_rows.append(
            {
                "symbol": symbol,
                "rows": len(df),
                "valid_ohlc_rows": int(valid_ohlc.sum()),
                "spot_rows": int((valid_ohlc & spot_mask).sum()),
                "funding_rows": int((valid_ohlc & funding_mask).sum()),
                "oi_rows": int((valid_ohlc & oi_mask).sum()),
                "mark_rows": int((valid_ohlc & mark_mask).sum()),
                "zero_heavy_features_before": before_zero_heavy,
                "zero_heavy_features_after": after_zero_heavy,
            }
        )
        if i <= 5 or i % 25 == 0 or i == len(files):
            print(f"{i:03d}/{len(files)} {symbol} valid={int(valid_ohlc.sum())} spot={int((valid_ohlc & spot_mask).sum())} zero_heavy {before_zero_heavy}->{after_zero_heavy}", flush=True)

    report = pd.DataFrame(report_rows)
    if args.report_dir:
        out_dir = Path(args.report_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        suffix = "dry_run" if args.dry_run else "repaired"
        report.to_csv(out_dir / f"kraken_perp_feature_source_coverage_{suffix}.csv", index=False)
    return {
        "feature_dir": str(feature_dir),
        "files": len(files),
        "dry_run": bool(args.dry_run),
        "total_valid_ohlc_rows": int(report["valid_ohlc_rows"].sum()) if not report.empty else 0,
        "total_spot_rows": int(report["spot_rows"].sum()) if not report.empty else 0,
        "total_funding_rows": int(report["funding_rows"].sum()) if not report.empty else 0,
        "total_oi_rows": int(report["oi_rows"].sum()) if not report.empty else 0,
        "zero_heavy_features_before": int(report["zero_heavy_features_before"].sum()) if not report.empty else 0,
        "zero_heavy_features_after": int(report["zero_heavy_features_after"].sum()) if not report.empty else 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", default="data_perp/features/20260520_004500")
    parser.add_argument("--perp-root", default="data_perp/exchanges/krakenfutures")
    parser.add_argument("--spot-root", default="data_spot/exchanges/kraken")
    parser.add_argument("--manifest", default="data_perp/exchanges/krakenfutures/manifests/kraken_dual_market_verified_universe_latest.json")
    parser.add_argument("--report-dir", default="reports_perp/kraken_feature_quality")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(json.dumps(repair(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
