#!/usr/bin/env python3
import argparse
import glob
import os

import numpy as np
import pandas as pd

from extreme_price_movements.data_store import PartitionedOHLCVStore
from extreme_price_movements.features import _safe_log_df, _transform_close_fixed_ffd, atr_percent
import extreme_price_movements.fast_funcs as ff


def _symbol_from_file(fpath: str, df: pd.DataFrame) -> str:
    if "__symbol__" in df.columns and not df.empty:
        return str(df["__symbol__"].iloc[0])
    fname = os.path.basename(fpath).replace("symbol=", "").replace(".parquet", "")
    return fname.replace("_", "/", 1)


def _compute_atr_pct_from_ohlcv(
    ohlcv: pd.DataFrame,
    symbol: str,
    atr_n: int,
    ffd_d_base: float,
    ffd_thres: float,
    safe_log_eps: float,
) -> pd.Series:
    if ohlcv.empty:
        return pd.Series(dtype=np.float32)

    high = ohlcv["high"].astype(np.float32).to_frame(name=symbol)
    low = ohlcv["low"].astype(np.float32).to_frame(name=symbol)
    close = ohlcv["close"].astype(np.float32).to_frame(name=symbol)

    h = ff.numba_ewma(_safe_log_df(high, eps=safe_log_eps), 2.0 / 6.0, False)
    l = ff.numba_ewma(_safe_log_df(low, eps=safe_log_eps), 2.0 / 6.0, False)
    c = _transform_close_fixed_ffd(
        close,
        d=ffd_d_base,
        _label=f"repair_{symbol}",
        already_logged=False,
        thres=ffd_thres,
    )
    atr_df = atr_percent(h, l, c, n=atr_n)
    return pd.to_numeric(atr_df.iloc[:, 0], errors="coerce").astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description="Recompute atr_pct for feature parquet files.")
    ap.add_argument("--run-ts", required=True, help="Feature run timestamp (e.g. 20260214_190000)")
    ap.add_argument("--data-root", default="data", help="Data root containing ohlcv/ and features/")
    ap.add_argument("--atr-n", type=int, default=14)
    ap.add_argument("--ffd-d-base", type=float, default=0.4)
    ap.add_argument("--ffd-thres", type=float, default=1e-5)
    ap.add_argument("--safe-log-eps", type=float, default=1e-9)
    ap.add_argument("--force", action="store_true", help="Recompute even when atr_pct exists and has non-NaN values.")
    args = ap.parse_args()

    feat_dir = os.path.join(args.data_root, "features", args.run_ts)
    files = sorted(glob.glob(os.path.join(feat_dir, "symbol=*.parquet")))
    if not files:
        print(f"No feature files found in {feat_dir}")
        return 1

    store = PartitionedOHLCVStore(root_dir=args.data_root, timeframe="1h")

    total = len(files)
    repaired = 0
    skipped = 0
    failed = 0

    for i, fpath in enumerate(files, start=1):
        try:
            df = pd.read_parquet(fpath)
            sym = _symbol_from_file(fpath, df)

            if not args.force and "atr_pct" in df.columns:
                s_old = pd.to_numeric(df["atr_pct"], errors="coerce")
                if bool(s_old.notna().any()):
                    skipped += 1
                    continue

            ohlcv = store.load(sym)
            if ohlcv.empty:
                print(f"[{i}/{total}] {sym}: no OHLCV data, skipped")
                skipped += 1
                continue

            atr_s = _compute_atr_pct_from_ohlcv(
                ohlcv=ohlcv,
                symbol=sym,
                atr_n=max(2, int(args.atr_n)),
                ffd_d_base=float(args.ffd_d_base),
                ffd_thres=float(args.ffd_thres),
                safe_log_eps=float(args.safe_log_eps),
            )

            feat_idx = df.index
            atr_aligned = atr_s.reindex(feat_idx).astype(np.float32)
            df["atr_pct"] = atr_aligned.values
            df.to_parquet(fpath)
            repaired += 1

            if i % 25 == 0 or i == total:
                print(f"Progress {i}/{total}: repaired={repaired}, skipped={skipped}, failed={failed}")
        except Exception as exc:
            failed += 1
            print(f"[{i}/{total}] ERROR {fpath}: {exc}")

    print(f"Done. repaired={repaired}, skipped={skipped}, failed={failed}, total={total}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
