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
    ex = ccxt.binance({"enableRateLimit": True})
    ex.load_markets()
    return ex

@retry_with_backoff(retries=3, backoff_in_seconds=2)
def _fetch_ohlcv_paged(exchange, symbol, since_ms, until_ms, timeframe="1h", limit=1000):
    # Reduced logging: entry log removed
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

    start = since_ms
    while start < now_ms:
        end = min(start + chunk_ms, now_ms)
        df = _fetch_ohlcv_paged(exchange, symbol, start, end, timeframe=timeframe, limit=limit)
        if len(df):
            yield df
        start = end
        time.sleep(exchange.rateLimit / 1000)

class PartitionedOHLCVStore:
    def __init__(self, root_dir="data", timeframe="1h"):
        self.root_dir = root_dir
        self.timeframe = timeframe
        self.ohlcv_dir = os.path.join(root_dir, "ohlcv")
        os.makedirs(self.ohlcv_dir, exist_ok=True)

    def _get_symbol_dir(self, symbol: str) -> str:
        safe_sym = symbol.replace("/", "_")
        return os.path.join(self.ohlcv_dir, f"symbol={safe_sym}")

    def _get_meta_path(self, symbol: str) -> str:
        safe_sym = symbol.replace("/", "_")
        return os.path.join(self.ohlcv_dir, f"{safe_sym}.meta.json")

    def _read_meta(self, symbol: str) -> dict:
        path = self._get_meta_path(symbol)
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _write_meta(self, symbol: str, meta: dict):
        path = self._get_meta_path(symbol)
        try:
            with open(path, "w") as f:
                json.dump(meta, f)
        except Exception as e:
            tprint(f"Error writing meta for {symbol}: {e}")

    def _downcast(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        out = df.copy()
        for col in ["open","high","low","close","volume"]:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce").astype(np.float32)
        return out

    def load(self, symbol: str, columns=None, start_ts=None, end_ts=None) -> pd.DataFrame:
        """
        Load data for symbol with optimized file filtering.
        columns: list of columns to read (optimization).
        start_ts: Optional[pd.Timestamp] - inclusive start
        end_ts: Optional[pd.Timestamp] - inclusive end
        """
        sym_dir = self._get_symbol_dir(symbol)
        if not os.path.exists(sym_dir):
            return pd.DataFrame(columns=["open","high","low","close","volume"]).set_index(
                pd.DatetimeIndex([], tz="UTC", name="ts")
            )

        try:
            # 1. Gather and filter files by timestamp BEFORE reading
            files_to_read = []
            s_ts_sec = int(start_ts.timestamp()) if start_ts else 0
            e_ts_sec = int(end_ts.timestamp()) if end_ts else 2**63 - 1
            
            for root, dirs, files in os.walk(sym_dir):
                for f in files:
                    if not f.endswith(".parquet"):
                        continue
                    
                    fpath = os.path.join(root, f)
                    
                    # Parse filename for timestamp range
                    if start_ts is not None or end_ts is not None:
                        try:
                            base = f.replace(".parquet", "")
                            parts = base.split("-")
                            if len(parts) >= 3:
                                f_min = int(parts[-2])
                                f_max = int(parts[-1])
                                
                                # Check overlap: [f_min, f_max] ∩ [s_ts_sec, e_ts_sec]
                                if f_min > e_ts_sec or f_max < s_ts_sec:
                                    continue  # Skip non-overlapping files
                        except (ValueError, IndexError):
                            pass  # Include file if parsing fails
                    
                    files_to_read.append(fpath)

            if not files_to_read:
                return pd.DataFrame(columns=["open","high","low","close","volume"]).set_index(
                    pd.DatetimeIndex([], tz="UTC", name="ts")
                )

            # 2. Read only filtered files with column selection
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

    def save_partitioned(self, symbol: str, df: pd.DataFrame, defer_compact: bool = False):
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
        
        sym_dir = self._get_symbol_dir(symbol)
        
        for year, group in df_reset.groupby("year"):
            part_dir = os.path.join(sym_dir, f"year={year}")
            os.makedirs(part_dir, exist_ok=True)
            
            ts_min = int(group["ts"].min().value // 10**9)
            ts_max = int(group["ts"].max().value // 10**9)
            fname = f"part-{ts_min}-{ts_max}.parquet"
            fpath = os.path.join(part_dir, fname)

            write_df = group.drop(columns=["year"])
            write_df.to_parquet(fpath, index=False)

            if not defer_compact:
                self.compact_partition(symbol, year)

    def compact_partition(self, symbol: str, year: int):
        sym_dir = self._get_symbol_dir(symbol)
        part_dir = os.path.join(sym_dir, f"year={year}")

        if not os.path.exists(part_dir):
            return

        files = glob.glob(os.path.join(part_dir, "*.parquet"))
        if not files:
            return

        dfs = []
        for f in files:
            try:
                dfs.append(pd.read_parquet(f))
            except Exception as e:
                tprint(f"Error reading {f} for compaction: {e}")

        if not dfs:
            return

        merged = pd.concat(dfs)
        if "ts" in merged.columns:
            merged["ts"] = pd.to_datetime(merged["ts"], utc=True)
            merged = merged.sort_values("ts").drop_duplicates("ts", keep="last")

        ts_min = int(merged["ts"].min().value // 10**9)
        ts_max = int(merged["ts"].max().value // 10**9)
        new_fname = f"compact-{ts_min}-{ts_max}.parquet"
        new_fpath = os.path.join(part_dir, new_fname)
        temp_fpath = new_fpath + ".tmp"

        try:
            # Atomic write pattern
            merged.to_parquet(temp_fpath, index=False)
            os.replace(temp_fpath, new_fpath)
            
            # Log cumulative stats
            interval_sec = pd.to_timedelta(self.timeframe).total_seconds()
            ts_min_val = merged["ts"].min().value // 10**9
            ts_max_val = merged["ts"].max().value // 10**9
            duration_sec = ts_max_val - ts_min_val + interval_sec
            days_covered = duration_sec / 86400.0
            avg_rows = len(merged) / days_covered if days_covered > 0 else 0
            
            ts_min_dt = pd.Timestamp(ts_min_val, unit='s', tz='UTC').strftime('%Y-%m-%d')
            ts_max_dt = pd.Timestamp(ts_max_val, unit='s', tz='UTC').strftime('%Y-%m-%d')
            tprint(f"Updated {new_fpath}: {len(merged)} rows, {ts_min_dt} -> {ts_max_dt} ({days_covered:.0f}d, ~{avg_rows:.0f} r/d)")

            for f in files:
                if f != new_fpath:
                    try:
                        os.remove(f)
                    except OSError:
                        pass # race condition if already deleted

        except Exception as e:
            tprint(f"Error compacting {symbol} {year}-{month}: {e}")

    def update_symbol(self, exchange, symbol: str, since_ms: int) -> pd.DataFrame:
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

            start_dt = pd.to_datetime(start_ms, unit='ms', utc=True).strftime('%Y-%m-%d %H:%M')
            tprint(f"FETCH incr: {symbol} from {start_dt}")
            
            # Progressive fetch and save (defer compaction to end)
            has_new_data = False
            touched_years = set()
            for chunk_df in fetch_ohlcv_all_7d_chunks(exchange, symbol, start_ms, timeframe=self.timeframe, limit=1000):
                if not chunk_df.empty:
                    has_new_data = True
                    fresh = self._downcast(chunk_df)
                    self.save_partitioned(symbol, fresh, defer_compact=True)
                    touched_years.update(fresh.index.year.unique())
                    
                    # Update metadata incrementally
                    new_last = fresh.index.max()
                    new_last_ms = int(new_last.value // 10**6)
                    if new_last_ms > last_ts_ms:
                         self._write_meta(symbol, {"last_ts_ms": new_last_ms})
                         last_ts_ms = new_last_ms

            # Single compaction pass per year at the end
            for yr in sorted(touched_years):
                self.compact_partition(symbol, yr)


def save_features(feats: dict, ts: pd.Timestamp, root_dir: str):
    """
    Save generated features to disk (Per-Symbol).

    Each symbol is saved as a separate Parquet file with the naming convention:
    symbol={safe_symbol}.parquet

    The original symbol name is preserved in the '__symbol__' column to ensure
    correct restoration of special characters (e.g. slashes) upon loading.

    feats: dict of DataFrames (feature_name -> DataFrame(index=t, cols=syms))
    """
    ts_str = ts.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(root_dir, "features", ts_str)
    os.makedirs(out_dir, exist_ok=True)
    
    tprint(f"Saving features to {out_dir}...")
    
    # 1. Pivot from  Dict[Feat -> DF(Sims)]  to  Dict[Sym -> DF(Feats)]
    # We assume all DFs have same columns (symbols) and index
    first_key = list(feats.keys())[0]
    symbols = feats[first_key].columns
    
    # Pre-extract numpy arrays + index once (avoids repeated pandas overhead)
    feat_keys = [k for k in feats if hasattr(feats[k], "columns")]
    feat_arrays = {}  # key -> (numpy_array, col_list)
    for k in feat_keys:
        df = feats[k]
        feat_arrays[k] = (df.values, list(df.columns))
    time_index = feats[first_key].index
    
    count = 0
    total = len(symbols)
    
    for i, sym in enumerate(symbols):
        try:
            # Fast numpy column extraction
            parts = {}
            for k in feat_keys:
                arr, cols = feat_arrays[k]
                if sym in cols:
                    j = cols.index(sym)
                    parts[k] = arr[:, j]
            
            df_sym = pd.DataFrame(parts, index=time_index)
            df_sym["__symbol__"] = sym

            # Save
            safe_sym = sym.replace("/", "_")
            fname = f"symbol={safe_sym}.parquet"
            fpath = os.path.join(out_dir, fname)
            df_sym.to_parquet(fpath)
            
            count += 1
            if count % 50 == 0:
                tprint(f"Saved features for {count}/{total} symbols...")
                
        except Exception as e:
            tprint(f"Failed to save features for {sym}: {e}")

    tprint(f"Feature save complete. {count}/{total} symbols saved.")

def load_features(ts: pd.Timestamp, root_dir: str) -> dict:
    """
    Load features from disk if they exist for this timestamp.

    Expects files matching 'symbol=*.parquet'. Restores the original symbol name
    from the '__symbol__' column if present, enabling support for symbols with
    special characters (e.g. 'BTC/USDT').

    Returns: dict of DataFrames (feature_name -> DataFrame(index=t, cols=syms)) or None.
    """
    ts_str = ts.strftime("%Y%m%d_%H%M%S")
    in_dir = os.path.join(root_dir, "features", ts_str)
    
    if not os.path.exists(in_dir):
        return None
        
    files = glob.glob(os.path.join(in_dir, "symbol=*.parquet"))
    if not files:
        return None
        
    tprint(f"Found {len(files)} feature files in {in_dir}. Loading...")
    
    # We need to pivot back to Dict[Feat -> DF(Syms)]
    # 1. Read all symbol DFs
    loaded_dfs = {} # Sym -> DF
    
    for fpath in files:
        try:
            fname = os.path.basename(fpath)
            # fname is symbol=XYZ.parquet
            sym = fname.replace("symbol=", "").replace(".parquet", "")
            df = pd.read_parquet(fpath)

            if "__symbol__" in df.columns:
                if not df.empty:
                    real_sym = str(df["__symbol__"].iloc[0])
                    df = df.drop(columns=["__symbol__"])
                    loaded_dfs[real_sym] = df
                else:
                    df = df.drop(columns=["__symbol__"])
                    loaded_dfs[sym] = df
            else:
                # Legacy files without __symbol__: restore slash from underscore
                # e.g. BTC_USDT -> BTC/USDT (first underscore only)
                real_sym = sym.replace("_", "/", 1)
                loaded_dfs[real_sym] = df
        except Exception as e:
            tprint(f"Error loading {fpath}: {e}")
            
    if not loaded_dfs:
        return None

    # 2. Pivot to Feat -> DF
    # All loaded DFs should have same columns (features)
    sample_df = list(loaded_dfs.values())[0]
    feat_keys = sample_df.columns
    
    feats_out = {}
    for k in feat_keys:
        # Construct DF for this feature: Index=Time, Cols=Symbols
        # We can dict comprehension
        data = {sym: df[k] for sym, df in loaded_dfs.items() if k in df.columns}
        feats_out[k] = pd.DataFrame(data).sort_index()
        
    tprint(f"Loaded {len(feats_out)} feature matrices.")
    return feats_out


def check_data_health(df: pd.DataFrame, timeframe="1h") -> dict:
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
    keys = ["open","high","low","close","volume"]
    panel = {}
    for k in keys:
        panel[k] = pd.concat([df[k].rename(sym) for sym, df in dfs_by_symbol.items()], axis=1).sort_index()
    return panel

OHLCVStore = PartitionedOHLCVStore

def get_feature_path(root_dir: str, ts: pd.Timestamp, symbol: str) -> str:
    """
    Returns the expected file path for a symbol's features at a given timestamp.
    """
    ts_str = ts.strftime("%Y%m%d_%H%M%S")
    safe_sym = symbol.replace("/", "_")
    return os.path.join(root_dir, "features", ts_str, f"symbol={safe_sym}.parquet")

def save_artifact_df(df: pd.DataFrame, root_dir: str, run_id: str, category: str, name: str):
    """
    Save a DataFrame as an artifact for a specific run.
    Path: root_dir/artifacts/{run_id}/{category}/{name}.parquet
    """
    out_dir = os.path.join(root_dir, "artifacts", run_id, category)
    os.makedirs(out_dir, exist_ok=True)
    fpath = os.path.join(out_dir, f"{name}.parquet")
    tprint(f"Saving artifact: {fpath}")
    df.to_parquet(fpath)

def load_artifact_df(root_dir: str, run_id: str, category: str, name: str) -> pd.DataFrame:
    """
    Load an artifact DataFrame. Returns None if not found.
    """
    fpath = os.path.join(root_dir, "artifacts", run_id, category, f"{name}.parquet")
    if os.path.exists(fpath):
        tprint(f"Loading artifact: {fpath}")
        return pd.read_parquet(fpath)
    return None
