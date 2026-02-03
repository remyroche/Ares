import os
import time
import numpy as np
import pandas as pd
import ccxt
import glob
from datetime import timezone

from extreme_price_movements.utils import tprint

def make_spot_exchange():
    ex = ccxt.binance({"enableRateLimit": True})
    ex.load_markets()
    return ex

def _fetch_ohlcv_paged(exchange, symbol, since_ms, until_ms, timeframe="1h", limit=1000):
    out = []
    since = since_ms
    while True:
        batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        if not batch:
            break
        for row in batch:
            ts = row[0]
            if ts < since_ms:
                continue
            if ts >= until_ms:
                break
            out.append(row)

        last = batch[-1][0]
        if last >= until_ms - 1:
            break
        since = last + 1
        if len(batch) < limit:
            break
        time.sleep(exchange.rateLimit / 1000)

    if not out:
        return pd.DataFrame(columns=["ts","open","high","low","close","volume"]).set_index(
            pd.DatetimeIndex([], tz="UTC", name="ts")
        )

    df = pd.DataFrame(out, columns=["ts","open","high","low","close","volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.drop_duplicates("ts").set_index("ts").sort_index()
    return df

def fetch_ohlcv_all_7d_chunks(exchange, symbol, since_ms, timeframe="1h", limit=1000):
    chunk_ms = int(pd.Timedelta(days=7).total_seconds() * 1000)
    now_ms = int(pd.Timestamp.utcnow().value // 10**6)

    dfs = []
    start = since_ms
    while start < now_ms:
        end = min(start + chunk_ms, now_ms)
        df = _fetch_ohlcv_paged(exchange, symbol, start, end, timeframe=timeframe, limit=limit)
        if len(df):
            dfs.append(df)
        start = end
        time.sleep(exchange.rateLimit / 1000)

    if not dfs:
        return pd.DataFrame(columns=["open","high","low","close","volume"]).set_index(
            pd.DatetimeIndex([], tz="UTC", name="ts")
        )

    out = pd.concat(dfs).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out

class PartitionedOHLCVStore:
    def __init__(self, root_dir="data", timeframe="1h"):
        self.root_dir = root_dir
        self.timeframe = timeframe
        self.ohlcv_dir = os.path.join(root_dir, "ohlcv")
        os.makedirs(self.ohlcv_dir, exist_ok=True)

    def _get_symbol_dir(self, symbol: str) -> str:
        safe_sym = symbol.replace("/", "_")
        return os.path.join(self.ohlcv_dir, f"symbol={safe_sym}")

    def _downcast(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        out = df.copy()
        for col in ["open","high","low","close","volume"]:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce").astype(np.float32)
        return out

    def load(self, symbol: str) -> pd.DataFrame:
        sym_dir = self._get_symbol_dir(symbol)
        if not os.path.exists(sym_dir):
            return pd.DataFrame(columns=["open","high","low","close","volume"]).set_index(
                pd.DatetimeIndex([], tz="UTC", name="ts")
            )

        # Read parquet dataset
        try:
            df = pd.read_parquet(sym_dir)
            if "ts" in df.columns:
                df["ts"] = pd.to_datetime(df["ts"], utc=True)
                df = df.set_index("ts")
            elif df.index.name == "ts":
                pass # already index

            df = df.sort_index()
            # deduplicate
            df = df[~df.index.duplicated(keep="last")]
            return self._downcast(df)
        except Exception:
            # Empty dir or read error
            return pd.DataFrame(columns=["open","high","low","close","volume"]).set_index(
                pd.DatetimeIndex([], tz="UTC", name="ts")
            )

    def save_partitioned(self, symbol: str, df: pd.DataFrame):
        """
        Saves df by partitioning by year/month.
        Appends by writing unique filenames based on timestamp range.
        """
        if df.empty:
            return

        df = self._downcast(df)
        df_reset = df.reset_index().rename(columns={"index": "ts"})
        if "ts" not in df_reset.columns:
             # handle case where index was not named
             df_reset = df.reset_index()
             if df_reset.columns[0] != "ts":
                  df_reset.rename(columns={df_reset.columns[0]: "ts"}, inplace=True)

        df_reset["year"] = df_reset["ts"].dt.year
        df_reset["month"] = df_reset["ts"].dt.month

        sym_dir = self._get_symbol_dir(symbol)
        os.makedirs(sym_dir, exist_ok=True)

        # We use partition_cols for pyarrow dataset structure
        # but to ensure we don't overwrite if we are appending small chunks,
        # we might rely on unique filenames.
        # However, pandas.to_parquet with partition_cols writes files like part.0.parquet
        # inside the folders. If we call it again, it might overwrite or create part.1.parquet?
        # Actually standard to_parquet(..., partition_cols=...) usually writes a hive dataset.
        # Appending to a hive dataset is tricky with plain pandas.
        # Recommendation from user: "write partitioned dataset... then 'append' means 'write new partitions'"
        # But if we have new data for an existing month, we need to handle it.
        # For simplicity and robustness, and since we update incrementally:
        # We can just write the new data as a new file in the partition if we name it uniquely.
        # OR, since the user said "append chunks instead of rewriting",
        # let's assume we are appending strictly new time ranges.

        # Strategy: write to dataset using pyarrow with existing_data_behavior='overwrite_or_ignore'
        # is not directly supported in pandas `to_parquet`.
        # User suggested: "row-group appended" or "partitioned by year/month".

        # Let's try writing with a unique filename based on min_max ts to avoid collisions.
        # But `to_parquet` with `partition_cols` creates directory structure automatically
        # and manages filenames (usually UUIDs or hashes if not specified).

        # To avoid complexity, we will rely on pandas `to_parquet` appending capability if available (not really),
        # OR we write to a temporary dataset and then move files? No.

        # Simple approach for "Append":
        # Since we fetch *new* data (incremental), it likely falls into new hours/days.
        # If it falls into an existing partition (year/month), we want to add to it.
        # If we just write to `sym_dir` with `partition_cols=['year', 'month']`,
        # pandas/pyarrow usually generates unique filenames if we don't specify one?
        # Actually it might clear the directory? No, `to_parquet` on a directory...
        # Pandas `to_parquet` with `partition_cols` works best when writing the whole dataset.

        # Alternative: We load existing, concat, and rewrite?
        # User said "ensure that we append the chunks instead of rewriting".
        # This implies we should NOT load everything and rewrite.

        # So we should write ONLY the new data.
        # df_reset contains only the FRESH data (if we passed fresh data).
        # We generate a unique filename for this batch.

        # We can manually partition.
        for (year, month), group in df_reset.groupby(["year", "month"]):
            part_dir = os.path.join(sym_dir, f"year={year}", f"month={month:02d}")
            os.makedirs(part_dir, exist_ok=True)

            # unique filename using timestamp range
            ts_min = group["ts"].min().value // 10**9
            ts_max = group["ts"].max().value // 10**9
            fname = f"part-{ts_min}-{ts_max}.parquet"
            fpath = os.path.join(part_dir, fname)

            # Drop partition cols before writing
            write_df = group.drop(columns=["year", "month"])
            write_df.to_parquet(fpath, index=False)

    def update_symbol(self, exchange, symbol: str, since_ms: int) -> pd.DataFrame:
        # Load existing to find last TS (we still need to know where to start)
        # But we only need the last timestamp, which we can get efficiently?
        # For now, let's load everything (memory might be an issue if HUGE, but for 1h candles 4 years it's fine ~35k rows)
        existing = self.load(symbol)

        if existing.empty:
            start_ms = since_ms
        else:
            last_ts = existing.index.max()
            start_ms = int(last_ts.value // 10**6) + 1

        now_ms = int(pd.Timestamp.utcnow().value // 10**6)
        if start_ms >= now_ms:
            return existing

        tprint(f"FETCH incr: {symbol} from {pd.to_datetime(start_ms, unit='ms', utc=True)}")
        fresh = fetch_ohlcv_all_7d_chunks(exchange, symbol, start_ms, timeframe=self.timeframe, limit=1000)

        if fresh is not None and not fresh.empty:
            fresh = self._downcast(fresh)
            # Save ONLY the fresh data
            self.save_partitioned(symbol, fresh)

            # Return merged view
            merged = pd.concat([existing, fresh]).sort_index()
            merged = merged[~merged.index.duplicated(keep="last")]
            return merged

        return existing

def check_data_health(df: pd.DataFrame, timeframe="1h") -> dict:
    if df.empty:
        return {"status": "empty", "completeness": 0.0, "missing_count": 0}

    start = df.index.min()
    end = df.index.max()

    # Expected frequency
    if timeframe == "1h":
        freq = "h"
    else:
        freq = timeframe # simplified

    full_idx = pd.date_range(start, end, freq=freq, tz="UTC")
    expected_rows = len(full_idx)
    actual_rows = len(df)

    missing = full_idx.difference(df.index)
    missing_count = len(missing)
    completeness = actual_rows / expected_rows if expected_rows > 0 else 0.0

    gaps = []
    if missing_count > 0:
        # grouping consecutive missing timestamps
        # naive approach: just list first and last
        pass

    return {
        "status": "ok" if missing_count == 0 else "gaps",
        "completeness": completeness,
        "missing_count": missing_count,
        "first_missing": missing[0].isoformat() if missing_count > 0 else None,
        "last_missing": missing[-1].isoformat() if missing_count > 0 else None,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "expected_rows": expected_rows,
        "actual_rows": actual_rows
    }

def to_panel(dfs_by_symbol: dict[str, pd.DataFrame]):
    keys = ["open","high","low","close","volume"]
    panel = {}
    for k in keys:
        panel[k] = pd.concat([df[k].rename(sym) for sym, df in dfs_by_symbol.items()], axis=1).sort_index()
    return panel

def downcast_panel_float32(panel: dict[str, pd.DataFrame]):
    for k in panel:
        panel[k] = panel[k].astype(np.float32)
    return panel

# Alias for compatibility if needed, but we should use PartitionedOHLCVStore
OHLCVStore = PartitionedOHLCVStore
