import os
import time
import json
import numpy as np
import pandas as pd
import ccxt
import glob
import shutil
import fcntl
from datetime import timezone

from extreme_price_movements.utils import tprint, retry_with_backoff

class FileLock:
    """
    Simple file-based lock using fcntl for Unix-like systems.
    """
    def __init__(self, lock_file):
        self.lock_file = lock_file
        self.handle = None

    def __enter__(self):
        try:
            self.handle = open(self.lock_file, 'w')
            # Blocking exclusive lock
            fcntl.flock(self.handle, fcntl.LOCK_EX)
        except Exception as e:
            tprint(f"Error acquiring lock {self.lock_file}: {e}")
            if self.handle:
                self.handle.close()
            raise
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.handle:
            try:
                fcntl.flock(self.handle, fcntl.LOCK_UN)
                self.handle.close()
            except Exception as e:
                tprint(f"Error releasing lock {self.lock_file}: {e}")

def make_spot_exchange():
    tprint(f"Entering function: make_spot_exchange in data_store.py")
    ex = ccxt.binance({"enableRateLimit": True})
    ex.load_markets()
    return ex

@retry_with_backoff(retries=3, backoff_in_seconds=2)
def _fetch_ohlcv_paged(exchange, symbol, since_ms, until_ms, timeframe="1h", limit=1000):
    tprint(f"Entering function: _fetch_ohlcv_paged in data_store.py")
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
    tprint(f"Entering function: fetch_ohlcv_all_7d_chunks in data_store.py")
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
        tprint(f"Entering function: __init__ in data_store.py")
        self.root_dir = root_dir
        self.timeframe = timeframe
        self.ohlcv_dir = os.path.join(root_dir, "ohlcv")
        os.makedirs(self.ohlcv_dir, exist_ok=True)

    def _get_symbol_dir(self, symbol: str) -> str:
        tprint(f"Entering function: _get_symbol_dir in data_store.py")
        safe_sym = symbol.replace("/", "_")
        return os.path.join(self.ohlcv_dir, f"symbol={safe_sym}")

    def _get_meta_path(self, symbol: str) -> str:
        tprint(f"Entering function: _get_meta_path in data_store.py")
        safe_sym = symbol.replace("/", "_")
        return os.path.join(self.ohlcv_dir, f"{safe_sym}.meta.json")

    def _read_meta(self, symbol: str) -> dict:
        tprint(f"Entering function: _read_meta in data_store.py")
        path = self._get_meta_path(symbol)
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _write_meta(self, symbol: str, meta: dict):
        tprint(f"Entering function: _write_meta in data_store.py")
        path = self._get_meta_path(symbol)
        try:
            with open(path, "w") as f:
                json.dump(meta, f)
        except Exception as e:
            tprint(f"Error writing meta for {symbol}: {e}")

    def _downcast(self, df: pd.DataFrame) -> pd.DataFrame:
        tprint(f"Entering function: _downcast in data_store.py")
        if df.empty:
            return df
        out = df.copy()
        for col in ["open","high","low","close","volume"]:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce").astype(np.float32)
        return out

    def load(self, symbol: str, columns=None, start_ts=None, end_ts=None) -> pd.DataFrame:
        """
        Load data for symbol.
        columns: list of columns to read (optimization).
        start_ts: Optional[pd.Timestamp] - inclusive start
        end_ts: Optional[pd.Timestamp] - inclusive end
        """
        tprint(f"Entering function: load in data_store.py")
        sym_dir = self._get_symbol_dir(symbol)
        if not os.path.exists(sym_dir):
            return pd.DataFrame(columns=["open","high","low","close","volume"]).set_index(
                pd.DatetimeIndex([], tz="UTC", name="ts")
            )

        try:
            # 1. Gather all parquet files
            all_files = []
            for root, dirs, files in os.walk(sym_dir):
                for f in files:
                    if f.endswith(".parquet"):
                        all_files.append(os.path.join(root, f))

            if not all_files:
                return pd.DataFrame(columns=["open","high","low","close","volume"]).set_index(
                    pd.DatetimeIndex([], tz="UTC", name="ts")
                )

            # 2. Filter files by time range if provided
            files_to_read = all_files
            if start_ts is not None or end_ts is not None:
                files_to_read = []
                # Timestamps in filename are in seconds
                s_ts_sec = int(start_ts.timestamp()) if start_ts else 0
                e_ts_sec = int(end_ts.timestamp()) if end_ts else 2**63 - 1 # arbitrarily large

                for fpath in all_files:
                    fname = os.path.basename(fpath)
                    # Expected format: part-{min}-{max}.parquet or compact-{min}-{max}.parquet
                    # We can split by '-' and take the last two parts before .parquet
                    try:
                        base = fname.replace(".parquet", "")
                        parts = base.split("-")
                        if len(parts) >= 3:
                            f_min = int(parts[-2])
                            f_max = int(parts[-1])

                            # Check overlap: file range [f_min, f_max] overlaps with [s_ts_sec, e_ts_sec]
                            # if f_min <= e_ts_sec AND f_max >= s_ts_sec
                            if f_min <= e_ts_sec and f_max >= s_ts_sec:
                                files_to_read.append(fpath)
                        else:
                            # If naming convention fails, include it to be safe
                            files_to_read.append(fpath)
                    except Exception:
                        files_to_read.append(fpath)

            if not files_to_read:
                return pd.DataFrame(columns=["open","high","low","close","volume"]).set_index(
                    pd.DatetimeIndex([], tz="UTC", name="ts")
                )

            # 3. Read filtered files
            read_cols = None
            if columns:
                read_cols = list(columns)
                if "ts" not in read_cols:
                    read_cols.append("ts")

            df = pd.read_parquet(files_to_read, columns=read_cols)

            if "ts" in df.columns:
                df["ts"] = pd.to_datetime(df["ts"], utc=True)
                df = df.set_index("ts")
            elif df.index.name == "ts":
                pass

            df = df.sort_index()
            df = df[~df.index.duplicated(keep="last")]

            # 4. Final slice to exact range (since files are coarse)
            if start_ts is not None:
                df = df[df.index >= start_ts]
            if end_ts is not None:
                df = df[df.index <= end_ts]

            return self._downcast(df)
        except Exception as e:
            tprint(f"Error loading {symbol}: {e}")
            return pd.DataFrame(columns=["open","high","low","close","volume"]).set_index(
                pd.DatetimeIndex([], tz="UTC", name="ts")
            )

    def save_partitioned(self, symbol: str, df: pd.DataFrame):
        tprint(f"Entering function: save_partitioned in data_store.py")
        if df.empty:
            return

        df = self._downcast(df)
        df_reset = df.reset_index().rename(columns={"index": "ts"})
        if "ts" not in df_reset.columns:
             df_reset = df.reset_index()
             if df_reset.columns[0] != "ts":
                  df_reset.rename(columns={df_reset.columns[0]: "ts"}, inplace=True)

        df_reset["ts"] = pd.to_datetime(df_reset["ts"], utc=True)
        for c in ["open","high","low","close","volume"]:
            if c in df_reset.columns:
                df_reset[c] = df_reset[c].astype(np.float32)

        df_reset["year"] = df_reset["ts"].dt.year
        df_reset["month"] = df_reset["ts"].dt.month

        sym_dir = self._get_symbol_dir(symbol)

        for (year, month), group in df_reset.groupby(["year", "month"]):
            part_dir = os.path.join(sym_dir, f"year={year}", f"month={month:02d}")
            os.makedirs(part_dir, exist_ok=True)

            ts_min = int(group["ts"].min().value // 10**9)
            ts_max = int(group["ts"].max().value // 10**9)
            fname = f"part-{ts_min}-{ts_max}.parquet"
            fpath = os.path.join(part_dir, fname)

            write_df = group.drop(columns=["year", "month"])
            write_df.to_parquet(fpath, index=False)

            files = [f for f in os.listdir(part_dir) if f.endswith(".parquet")]
            if len(files) > 10:
                # Compaction deferred or async?
                # For safety, synchronous here but only per partition write.
                self.compact_partition(symbol, year, month)

    def compact_partition(self, symbol: str, year: int, month: int):
        tprint(f"Entering function: compact_partition in data_store.py")
        sym_dir = self._get_symbol_dir(symbol)
        part_dir = os.path.join(sym_dir, f"year={year}", f"month={month:02d}")

        if not os.path.exists(part_dir):
            return

        files = glob.glob(os.path.join(part_dir, "*.parquet"))
        if not files:
            return

        try:
            dfs = []
            for f in files:
                dfs.append(pd.read_parquet(f))

            merged = pd.concat(dfs)
            if "ts" in merged.columns:
                merged["ts"] = pd.to_datetime(merged["ts"], utc=True)
                merged = merged.sort_values("ts").drop_duplicates("ts", keep="last")

            ts_min = int(merged["ts"].min().value // 10**9)
            ts_max = int(merged["ts"].max().value // 10**9)
            new_fname = f"compact-{ts_min}-{ts_max}.parquet"
            new_fpath = os.path.join(part_dir, new_fname)
            temp_fpath = new_fpath + ".tmp"

            # Atomic write pattern
            merged.to_parquet(temp_fpath, index=False)
            os.replace(temp_fpath, new_fpath)

            for f in files:
                if f != new_fpath:
                    try:
                        os.remove(f)
                    except OSError:
                        pass # race condition if already deleted

        except Exception as e:
            tprint(f"Error compacting {symbol} {year}-{month}: {e}")

    def update_symbol(self, exchange, symbol: str, since_ms: int) -> pd.DataFrame:
        tprint(f"Entering function: update_symbol in data_store.py")
        # Ensure locking
        sym_dir = self._get_symbol_dir(symbol)
        os.makedirs(sym_dir, exist_ok=True)
        lock_path = os.path.join(sym_dir, ".lock")

        with FileLock(lock_path):
            # Check metadata first to avoid IO
            meta = self._read_meta(symbol)
            last_ts_ms = meta.get("last_ts_ms", 0)

            if last_ts_ms > 0:
                start_ms = last_ts_ms + 1
            else:
                # Fallback to load index if no meta
                # Here load() without args is fine, but we might want to check just the last file?
                # For simplicity, keep as is, but maybe optimize load(columns=['ts'])
                existing_idx = self.load(symbol, columns=["ts"]).index
                if not existing_idx.empty:
                    last_ts = existing_idx.max()
                    start_ms = int(last_ts.value // 10**6) + 1
                else:
                    start_ms = since_ms

            now_ms = int(pd.Timestamp.utcnow().value // 10**6)

            if start_ms >= now_ms:
                return self.load(symbol)

            tprint(f"FETCH incr: {symbol} from {pd.to_datetime(start_ms, unit='ms', utc=True)}")
            fresh = fetch_ohlcv_all_7d_chunks(exchange, symbol, start_ms, timeframe=self.timeframe, limit=1000)

            if fresh is not None and not fresh.empty:
                fresh = self._downcast(fresh)
                self.save_partitioned(symbol, fresh)

                # Update metadata
                new_last = fresh.index.max()
                new_last_ms = int(new_last.value // 10**6)
                if new_last_ms > last_ts_ms:
                    self._write_meta(symbol, {"last_ts_ms": new_last_ms})

                # Reload full to return merged
                return self.load(symbol)

            return self.load(symbol)

def check_data_health(df: pd.DataFrame, timeframe="1h") -> dict:
    tprint(f"Entering function: check_data_health in data_store.py")
    if df.empty:
        return {"status": "empty", "completeness": 0.0, "missing_count": 0}

    start = df.index.min()
    end = df.index.max()

    if timeframe == "1h":
        freq = "h"
    else:
        freq = timeframe

    full_idx = pd.date_range(start, end, freq=freq, tz="UTC")
    expected_rows = len(full_idx)
    actual_rows = len(df)

    missing = full_idx.difference(df.index)
    missing_count = len(missing)
    completeness = actual_rows / expected_rows if expected_rows > 0 else 0.0

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
    tprint(f"Entering function: to_panel in data_store.py")
    keys = ["open","high","low","close","volume"]
    panel = {}
    for k in keys:
        panel[k] = pd.concat([df[k].rename(sym) for sym, df in dfs_by_symbol.items()], axis=1).sort_index()
    return panel

OHLCVStore = PartitionedOHLCVStore
