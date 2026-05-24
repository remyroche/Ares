#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from extreme_price_movements.data_store import PartitionedOHLCVStore
from extreme_price_movements.features import _canonical_spot_index_from_ohlcv


def _safe_symbol(symbol: str) -> str:
    return str(symbol).replace("/", "_")


def _one_col(df: pd.DataFrame, column: str, name: str, index: pd.DatetimeIndex) -> pd.DataFrame:
    if df.empty or column not in df.columns:
        return pd.DataFrame(np.nan, index=index, columns=[name], dtype=np.float32)
    out = (
        pd.to_numeric(df[column], errors="coerce")
        .reindex(index)
        .ffill()
        .astype(np.float32)
        .rename(name)
        .to_frame()
    )
    return out


def backfill_columns(
    *,
    feature_dir: Path,
    manifest_path: Path,
    perp_root: Path,
    spot_root: Path,
    dry_run: bool = False,
) -> dict[str, object]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    symbols = list(manifest.get("symbols") or [])
    perp_store = PartitionedOHLCVStore(root_dir=str(perp_root), timeframe="1h")
    spot_store = PartitionedOHLCVStore(root_dir=str(spot_root), timeframe="1h")

    updated: list[str] = []
    missing_feature_files: list[str] = []
    failed: list[str] = []

    for i, item in enumerate(symbols, start=1):
        perp_symbol = str(item["perp_symbol"])
        spot_symbol = str(item["spot_symbol"])
        fpath = feature_dir / f"symbol={_safe_symbol(perp_symbol)}.parquet"
        if not fpath.exists():
            missing_feature_files.append(perp_symbol)
            continue
        try:
            feat = pd.read_parquet(fpath)
            if not isinstance(feat.index, pd.DatetimeIndex):
                raise ValueError(f"{fpath} does not have a DatetimeIndex")
            index = pd.DatetimeIndex(pd.to_datetime(feat.index, utc=True), name="ts")
            feat.index = index

            perp = perp_store.load(
                perp_symbol,
                columns=[
                    "close",
                    "spot_open",
                    "spot_high",
                    "spot_low",
                    "spot_close",
                    "spot_volume",
                ],
            )
            spot = spot_store.load(spot_symbol, columns=["open", "high", "low", "close", "volume"])

            col = _safe_symbol(perp_symbol)
            embedded_spot_close = _one_col(perp, "spot_close", col, index)
            external_spot_close = _one_col(spot, "close", col, index)
            embedded_spot_n = int((embedded_spot_close[col] > 0.0).sum())
            external_spot_n = int((external_spot_close[col] > 0.0).sum())
            use_embedded_spot = bool(
                embedded_spot_n > 0
                and (
                    external_spot_n <= 0
                    or embedded_spot_n >= int(0.90 * external_spot_n)
                )
            )
            if use_embedded_spot:
                spot_open = _one_col(perp, "spot_open", col, index)
                spot_high = _one_col(perp, "spot_high", col, index)
                spot_low = _one_col(perp, "spot_low", col, index)
                spot_close = embedded_spot_close
                spot_volume = _one_col(perp, "spot_volume", col, index)
            else:
                spot_open = _one_col(spot, "open", col, index)
                spot_high = _one_col(spot, "high", col, index)
                spot_low = _one_col(spot, "low", col, index)
                spot_close = _one_col(spot, "close", col, index)
                spot_volume = _one_col(spot, "volume", col, index)
            perp_close = _one_col(perp, "close", col, index)

            canonical = _canonical_spot_index_from_ohlcv(
                spot_open=spot_open,
                spot_high=spot_high,
                spot_low=spot_low,
                spot_close=spot_close,
                spot_volume=spot_volume,
                fallback_price=spot_close,
                safe_log_eps=1e-9,
                kalman_lambda=0.05,
                bars_per_hour=1,
            )
            fallback_ser = spot_close[col].where(spot_close[col] > 0.0)
            if canonical is None:
                canonical_ser = fallback_ser.astype(np.float32)
            else:
                canonical_ser = canonical[col].reindex(index).astype(np.float32)
                canonical_ser = canonical_ser.where(canonical_ser > 0.0, fallback_ser)

            premium = ((perp_close[col] / (canonical_ser + 1e-12)) - 1.0).clip(
                -0.10, 0.10
            )
            spot_available = spot_close[col].where(spot_close[col] > 0.0).notna()

            feat["canonical_index"] = (
                canonical_ser.replace([np.inf, -np.inf], np.nan)
                .astype(np.float32)
            )
            feat["premium_proxy"] = (
                premium.replace([np.inf, -np.inf], np.nan)
                .astype(np.float32)
            )
            feat["spot_available"] = spot_available.astype(np.float32)

            if not dry_run:
                tmp_path = fpath.with_suffix(".parquet.tmp")
                feat.to_parquet(tmp_path, compression="zstd")
                tmp_path.replace(fpath)
            updated.append(perp_symbol)
            if i == 1 or i == len(symbols) or i % 25 == 0:
                print(f"updated {len(updated)}/{len(symbols)} symbols", flush=True)
        except Exception as exc:
            failed.append(f"{perp_symbol}: {exc.__class__.__name__}: {exc}")

    return {
        "feature_dir": str(feature_dir),
        "manifest_path": str(manifest_path),
        "updated_count": len(updated),
        "missing_feature_files": missing_feature_files,
        "failed": failed,
        "dry_run": dry_run,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Backfill raw Kraken canonical-index feature columns into existing feature parquet files."
    )
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data_perp/exchanges/krakenfutures/manifests/kraken_dual_market_verified_universe_latest.json"),
    )
    parser.add_argument(
        "--perp-root", type=Path, default=Path("data_perp/exchanges/krakenfutures")
    )
    parser.add_argument("--spot-root", type=Path, default=Path("data_spot/exchanges/kraken"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    result = backfill_columns(
        feature_dir=args.feature_dir,
        manifest_path=args.manifest,
        perp_root=args.perp_root,
        spot_root=args.spot_root,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not result["failed"] and not result["missing_feature_files"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
