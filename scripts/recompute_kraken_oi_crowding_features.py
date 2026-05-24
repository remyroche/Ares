#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from extreme_price_movements.kraken_actual_data import overlay_actual_volume_sidecar
from extreme_price_movements.perp_features import compute_features


TARGET_COLUMNS = (
    "oi_rel_vol_2h",
    "oi_rel_vol_4h",
    "oi_rel_vol_8h",
    "oi_value_log_1d_robust_z",
    "oi_value_log_7d_robust_z",
    "oi_chg_2h_robust_z",
    "oi_chg_4h_robust_z",
    "oi_chg_8h_robust_z",
    "leverage_build",
    "leverage_build_score",
    "unwind_score",
)


def _load_symbol_raw(ohlcv_root: Path, symbol_key: str) -> pd.DataFrame:
    files = sorted((ohlcv_root / f"symbol={symbol_key}").glob("year=*/compact-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no raw OHLCV partitions for {symbol_key}")
    frames = [pd.read_parquet(path) for path in files]
    raw = pd.concat(frames, ignore_index=True)
    if "ts" not in raw.columns:
        raise ValueError(f"raw OHLCV partitions for {symbol_key} have no ts column")
    raw["ts"] = pd.to_datetime(raw["ts"], utc=True, errors="coerce")
    raw = raw.dropna(subset=["ts"]).sort_values("ts")
    raw = raw.drop_duplicates(subset=["ts"], keep="last").set_index("ts")
    return raw


def _load_open_interest_sidecar(raw_root: Path, symbol_key: str, index: pd.DatetimeIndex) -> pd.Series:
    sidecar_root = raw_root / "open_interest_hourly"
    candidates = [
        sidecar_root / f"{symbol_key.replace(':', '_')}.parquet",
        sidecar_root / f"{symbol_key}.parquet",
    ]
    for path in candidates:
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        if "open_interest" not in frame.columns:
            continue
        if "ts" in frame.columns:
            frame["ts"] = pd.to_datetime(frame["ts"], utc=True, errors="coerce")
        else:
            frame = frame.reset_index().rename(columns={frame.index.name or "index": "ts"})
            frame["ts"] = pd.to_datetime(frame["ts"], utc=True, errors="coerce")
        frame = frame.dropna(subset=["ts"]).sort_values("ts")
        frame = frame.drop_duplicates(subset=["ts"], keep="last").set_index("ts")
        return pd.to_numeric(frame["open_interest"], errors="coerce").reindex(index)
    return pd.Series(np.nan, index=index, dtype="float64")


def _series(raw: pd.DataFrame, column: str, index: pd.DatetimeIndex) -> pd.Series:
    if column not in raw.columns:
        return pd.Series(np.nan, index=index, dtype="float64")
    return pd.to_numeric(raw[column], errors="coerce").reindex(index)


def _compute_target_columns(raw: pd.DataFrame, raw_root: Path, symbol_key: str, index: pd.DatetimeIndex) -> pd.DataFrame:
    close = _series(raw, "close", index)
    spot = _series(raw, "spot_close", index)
    mark = _series(raw, "mark_price", index)
    if mark.notna().sum() == 0:
        mark = _series(raw, "mark_close", index)
    raw_oi = _series(raw, "open_interest", index)
    sidecar_oi = _load_open_interest_sidecar(raw_root, symbol_key, index)
    open_interest = raw_oi.combine_first(sidecar_oi)
    df = pd.DataFrame(
        {
            "funding_rate": _series(raw, "funding_rate", index),
            "open_interest": open_interest,
            "perp_price": close,
            "spot_price": spot,
            "mark_price": mark,
            "volume": _series(raw, "volume", index),
            "quote_volume": _series(raw, "volume", index) * close,
            "close": close,
        },
        index=index,
    )
    features = compute_features(df)
    return features.loc[:, list(TARGET_COLUMNS)].astype("float32")


def recompute(args: argparse.Namespace) -> pd.DataFrame:
    feature_dir = Path(args.feature_dir)
    raw_root = Path(args.raw_root)
    ohlcv_root = raw_root / "ohlcv"
    files = sorted(feature_dir.glob("symbol=*.parquet"))
    if not files:
        raise FileNotFoundError(f"no symbol feature parquet files under {feature_dir}")

    rows: list[dict[str, object]] = []
    for i, path in enumerate(files, start=1):
        symbol_key = path.stem.removeprefix("symbol=")
        status = "updated"
        error = ""
        try:
            feat = pd.read_parquet(path)
            if not isinstance(feat.index, pd.DatetimeIndex):
                if "ts" not in feat.columns:
                    raise ValueError("feature parquet has no DatetimeIndex or ts column")
                feat["ts"] = pd.to_datetime(feat["ts"], utc=True, errors="coerce")
                feat = feat.set_index("ts")
            feat.index = pd.DatetimeIndex(pd.to_datetime(feat.index, utc=True), name="ts")
            index = feat.index
            raw = _load_symbol_raw(ohlcv_root, symbol_key)
            raw = overlay_actual_volume_sidecar(raw, root_dir=raw_root, symbol=symbol_key)
            targets = _compute_target_columns(raw, raw_root, symbol_key, index)

            before = {
                col: float(pd.to_numeric(feat.get(col), errors="coerce").replace([np.inf, -np.inf], np.nan).isna().mean())
                if col in feat.columns
                else 1.0
                for col in TARGET_COLUMNS
            }
            after = {
                col: float(targets[col].replace([np.inf, -np.inf], np.nan).isna().mean())
                for col in TARGET_COLUMNS
            }
            if not args.dry_run:
                for col in TARGET_COLUMNS:
                    feat[col] = targets[col]
                tmp = path.with_suffix(".parquet.tmp")
                feat.to_parquet(tmp, compression="zstd")
                tmp.replace(path)
        except Exception as exc:
            status = "failed"
            error = str(exc)
            before = {col: np.nan for col in TARGET_COLUMNS}
            after = {col: np.nan for col in TARGET_COLUMNS}

        row: dict[str, object] = {
            "symbol": symbol_key,
            "status": status,
            "error": error,
        }
        for col in TARGET_COLUMNS:
            row[f"{col}_nan_before"] = before[col]
            row[f"{col}_nan_after"] = after[col]
        rows.append(row)
        if i <= 5 or i % 25 == 0 or i == len(files) or status == "failed":
            msg = f"{i:03d}/{len(files)} {symbol_key} {status}"
            if status == "failed":
                msg += f": {error}"
            print(msg, flush=True)

    report = pd.DataFrame(rows)
    report_path = Path(args.report)
    if not args.dry_run:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(report_path, index=False)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", default="data_perp/features/20260523_015947")
    parser.add_argument("--raw-root", default="data_perp/exchanges/krakenfutures")
    parser.add_argument(
        "--report",
        default="data_perp/artifacts/20260523_015947/features/oi_crowding_recompute_report.csv",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    report = recompute(args)
    ok = report[report["status"].eq("updated")]
    failed = report[report["status"].ne("updated")]
    print(
        f"completed updated={len(ok)} failed={len(failed)} "
        f"report={args.report if not args.dry_run else '<dry-run>'}",
        flush=True,
    )
    if len(failed):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
