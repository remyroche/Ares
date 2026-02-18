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
import pyarrow.parquet as pq

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


def _feature_meta_path(parquet_path: str) -> str:
    return parquet_path.replace(".parquet", ".meta.json")


def _write_feature_metadata(parquet_path: str, symbol: str, index: pd.Index):
    meta_path = _feature_meta_path(parquet_path)
    if len(index) == 0:
        first_ts = last_ts = None
    else:
        first_ts = pd.Timestamp(index[0]).isoformat()
        last_ts = pd.Timestamp(index[-1]).isoformat()

    meta = {
        "version": 1,
        "symbol": symbol,
        "rows": int(len(index)),
        "first_ts": first_ts,
        "last_ts": last_ts,
    }

    tmp_meta = meta_path + ".tmp"
    with open(tmp_meta, "w") as fp:
        json.dump(meta, fp)
    os.replace(tmp_meta, meta_path)


def _read_feature_metadata(parquet_path: str) -> dict | None:
    meta_path = _feature_meta_path(parquet_path)
    if not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path, "r") as fp:
            return json.load(fp)
    except Exception:
        return None


def _infer_feature_bounds_from_file(parquet_path: str) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    try:
        pf = pq.ParquetFile(parquet_path)
    except Exception:
        return None, None

    if pf.num_row_groups == 0:
        return None, None

    index_col = None
    for name in pf.schema.names:
        if name.startswith("__index_level_"):
            index_col = name
            break

    if index_col is None:
        return None, None

    try:
        first_group = pf.read_row_group(0, columns=[index_col])
        last_group = pf.read_row_group(pf.num_row_groups - 1, columns=[index_col])
        first_val = first_group.column(0)[0].as_py()
        last_val = last_group.column(0)[-1].as_py()
        return pd.Timestamp(first_val), pd.Timestamp(last_val)
    except Exception:
        return None, None


def get_feature_bounds(parquet_path: str) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    meta = _read_feature_metadata(parquet_path)
    if meta:
        first_ts = pd.Timestamp(meta["first_ts"]) if meta.get("first_ts") else None
        last_ts = pd.Timestamp(meta["last_ts"]) if meta.get("last_ts") else None
        return first_ts, last_ts

    return _infer_feature_bounds_from_file(parquet_path)


def append_symbol_features(parquet_path: str, symbol: str, new_data: pd.DataFrame) -> int:
    if new_data.empty:
        return 0

    new_data = new_data.sort_index()
    numeric_cols = [c for c in new_data.columns if c != "__symbol__"]
    new_data[numeric_cols] = new_data[numeric_cols].astype(np.float32)

    existing = None
    if os.path.exists(parquet_path):
        existing = pd.read_parquet(parquet_path)
        if "__symbol__" in existing.columns:
            existing = existing.drop(columns=["__symbol__"])

    all_cols = sorted(set(new_data.columns) | (set(existing.columns) if existing is not None else set()))
    new_aligned = new_data.reindex(columns=all_cols)

    if existing is not None:
        existing_aligned = existing.reindex(columns=all_cols)
        before_rows = len(existing_aligned)
        combined = pd.concat([existing_aligned, new_aligned])
    else:
        before_rows = 0
        combined = new_aligned

    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    combined["__symbol__"] = symbol
    combined.to_parquet(parquet_path)
    _write_feature_metadata(parquet_path, symbol, combined.index)

    return len(combined) - before_rows


def save_features(
    feats: dict,
    ts: pd.Timestamp,
    root_dir: str,
    min_timestamp_by_symbol: dict[str, pd.Timestamp] | None = None,
    feat_index: pd.Index | None = None,
    feat_columns: list | None = None,
):
    """
    Save generated features to disk (Per-Symbol), streaming one symbol at a time.

    Peak memory ≈ 1 symbol × N_features × T rows (~2 MB).
    No temp chunk dirs, no merge step.

    feats: dict of feature_name -> DataFrame(index=t, cols=syms) OR numpy array (T, S).
           When numpy arrays, feat_index and feat_columns must be provided.
    """
    ts_str = ts.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(root_dir, "features", ts_str)
    os.makedirs(out_dir, exist_ok=True)

    tprint(f"Saving features to {out_dir}...")

    first_key = list(feats.keys())[0]
    first_val = feats[first_key]

    # Detect whether feats contains DataFrames or numpy arrays
    if isinstance(first_val, pd.DataFrame):
        symbols = list(first_val.columns)
        time_index = first_val.index
        feat_keys = [k for k in feats if hasattr(feats[k], "columns")]
        col_maps = {k: {c: j for j, c in enumerate(feats[k].columns)} for k in feat_keys}
        arrays = {k: feats[k].values for k in feat_keys}
    else:
        # Numpy array mode — iterate per-feature first to avoid random access
        # across 469 scattered arrays (which thrashes swap).
        # Phase 1: Extract columns per-feature sequentially, free each array.
        # Phase 2: Write per-symbol from transposed structure.
        import gc as _gc
        assert feat_index is not None and feat_columns is not None, \
            "feat_index and feat_columns required when feats contains numpy arrays"
        symbols = list(feat_columns)
        time_index = feat_index
        feat_keys = [k for k in feats if isinstance(feats[k], np.ndarray) and feats[k].ndim == 2]
        n_feats = len(feat_keys)
        total = len(symbols)

        # Phase 1: Transpose — build per-symbol column dict, free each feature array
        tprint(f"  Transposing {n_feats} features × {total} symbols for save...")
        sym_data = {j: {} for j in range(total)}  # sym_idx -> {feat_name: 1D array}
        for fi, k in enumerate(feat_keys):
            arr = feats[k]
            for j in range(total):
                sym_data[j][k] = arr[:, j].copy()  # copy column to own memory
            feats[k] = None  # free the (T, S) array
            if (fi + 1) % 50 == 0:
                _gc.collect()
                tprint(f"  Transpose progress: {fi+1}/{n_feats}")
        _gc.collect()
        tprint(f"  Transpose complete. Writing {total} symbols...")

        # Phase 2: Write per-symbol
        count = 0
        for j, sym in enumerate(symbols):
            cutoff_ts = None
            if min_timestamp_by_symbol:
                cutoff_ts = min_timestamp_by_symbol.get(sym)

            col_data = sym_data.pop(j)  # pop to free as we go
            if not col_data:
                continue

            df_sym = pd.DataFrame(col_data, index=time_index)
            del col_data
            df_sym = df_sym.astype(np.float32, copy=False)
            if cutoff_ts is not None:
                df_sym = df_sym[df_sym.index > cutoff_ts]
            if df_sym.empty:
                continue

            safe_sym = sym.replace("/", "_")
            final_path = os.path.join(out_dir, f"symbol={safe_sym}.parquet")
            append_symbol_features(final_path, sym, df_sym)
            del df_sym
            count += 1

            if count % 50 == 0:
                tprint(f"  Saved {count}/{total} symbols ({n_feats} features each)")
            if count % 200 == 0:
                _gc.collect()

        tprint(f"Feature save complete. {count}/{total} symbols saved ({n_feats} features).")
        return

    total = len(symbols)
    n_feats = len(feat_keys)

    count = 0
    for sym in symbols:
        cutoff_ts = None
        if min_timestamp_by_symbol:
            cutoff_ts = min_timestamp_by_symbol.get(sym)

        # Build {feat_name: 1-D array} for this symbol
        col_data = {}
        for k in feat_keys:
            j = col_maps[k].get(sym)
            if j is not None:
                col_data[k] = arrays[k][:, j]

        if not col_data:
            continue

        df_sym = pd.DataFrame(col_data, index=time_index)
        df_sym = df_sym.astype(np.float32, copy=False)
        if cutoff_ts is not None:
            df_sym = df_sym[df_sym.index > cutoff_ts]
        if df_sym.empty:
            continue

        safe_sym = sym.replace("/", "_")
        final_path = os.path.join(out_dir, f"symbol={safe_sym}.parquet")
        append_symbol_features(final_path, sym, df_sym)
        del df_sym
        count += 1

        if count % 50 == 0:
            tprint(f"Saved {count}/{total} symbols ({n_feats} features each)")
        if count % 100 == 0:
            _gc.collect()

    tprint(f"Feature save complete. {count}/{total} symbols saved ({n_feats} features).")

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
        
    files = sorted(glob.glob(os.path.join(in_dir, "symbol=*.parquet")))
    if not files:
        return None
        
    tprint(f"Found {len(files)} feature files in {in_dir}. Loading...")
    
    # Build Dict[Feat -> Dict[Symbol -> Series]] incrementally to reduce peak memory.
    # Previous implementation stored all symbol DataFrames first, then pivoted, which
    # could double memory pressure on large universes.
    feat_buffers = {}
    
    start_load = time.time()
    total_files = len(files)
    progress_every = 25 if total_files >= 100 else 10

    for i, fpath in enumerate(files, start=1):
        try:
            fname = os.path.basename(fpath)
            # fname is symbol=XYZ.parquet
            sym = fname.replace("symbol=", "").replace(".parquet", "")
            df = pd.read_parquet(fpath)

            if "__symbol__" in df.columns:
                if not df.empty:
                    real_sym = str(df["__symbol__"].iloc[0])
                    df = df.drop(columns=["__symbol__"])
                else:
                    df = df.drop(columns=["__symbol__"])
                    real_sym = sym
            else:
                # Legacy files without __symbol__: restore slash from underscore
                # e.g. BTC_USDT -> BTC/USDT (first underscore only)
                real_sym = sym.replace("_", "/", 1)

            for k in df.columns:
                if k not in feat_buffers:
                    feat_buffers[k] = {}
                feat_buffers[k][real_sym] = pd.to_numeric(df[k], errors="coerce").astype(np.float32, copy=False)

            del df
            if i % progress_every == 0 or i == total_files:
                elapsed = time.time() - start_load
                tprint(
                    f"Feature load progress: {i}/{total_files} files "
                    f"({(i / total_files) * 100:.1f}%) in {elapsed:.1f}s"
                )
        except Exception as e:
            tprint(f"Error loading {fpath}: {e}")

    # Encourage timely memory reclamation after file ingest loop
    import gc as _gc
    _gc.collect()
            
    if not feat_buffers:
        return None
    
    feats_out = {}
    for k, data in feat_buffers.items():
        # Construct DF for this feature: Index=Time, Cols=Symbols
        feats_out[k] = pd.DataFrame(data).sort_index()

    feat_buffers.clear()
    _gc.collect()
        
    tprint(f"Loaded {len(feats_out)} feature matrices.")
    return feats_out


def load_features_selected(
    ts: pd.Timestamp,
    root_dir: str,
    feature_keys: list[str] | set[str] | tuple[str, ...] | None = None,
    symbols: list[str] | set[str] | tuple[str, ...] | None = None,
) -> dict:
    """
    Load a subset of features/symbols from disk.

    This avoids loading every cached feature matrix into memory when only a
    narrow key set is required by downstream steps (e.g. label generation).
    """
    if feature_keys is None and symbols is None:
        return load_features(ts, root_dir)

    ts_str = ts.strftime("%Y%m%d_%H%M%S")
    in_dir = os.path.join(root_dir, "features", ts_str)
    if not os.path.exists(in_dir):
        return None

    files = sorted(glob.glob(os.path.join(in_dir, "symbol=*.parquet")))
    if not files:
        return None

    feature_set = set(feature_keys) if feature_keys else None
    symbol_set = set(map(str, symbols)) if symbols else None
    feat_buffers: dict[str, dict[str, pd.Series]] = {}

    tprint(
        f"Found {len(files)} feature files in {in_dir}. "
        f"Selective load: keys={len(feature_set) if feature_set else 'ALL'}, "
        f"symbols={len(symbol_set) if symbol_set else 'ALL'}"
    )

    start_load = time.time()
    total_files = len(files)
    progress_every = 25 if total_files >= 100 else 10

    for i, fpath in enumerate(files, start=1):
        try:
            fname = os.path.basename(fpath)
            sym_guess = fname.replace("symbol=", "").replace(".parquet", "").replace("_", "/", 1)
            if symbol_set is not None and sym_guess not in symbol_set:
                if i % progress_every == 0 or i == total_files:
                    elapsed = time.time() - start_load
                    tprint(
                        f"Selective feature load progress: {i}/{total_files} files "
                        f"({(i / total_files) * 100:.1f}%) in {elapsed:.1f}s"
                    )
                continue

            schema_names = set(pq.ParquetFile(fpath).schema.names)
            cols_to_read = []
            if "__symbol__" in schema_names:
                cols_to_read.append("__symbol__")
            if feature_set is None:
                cols_to_read.extend(
                    [c for c in schema_names if c != "__symbol__" and not c.startswith("__index_level_")]
                )
            else:
                cols_to_read.extend([c for c in feature_set if c in schema_names])

            if not cols_to_read or (len(cols_to_read) == 1 and cols_to_read[0] == "__symbol__"):
                if i % progress_every == 0 or i == total_files:
                    elapsed = time.time() - start_load
                    tprint(
                        f"Selective feature load progress: {i}/{total_files} files "
                        f"({(i / total_files) * 100:.1f}%) in {elapsed:.1f}s"
                    )
                continue

            df = pd.read_parquet(fpath, columns=cols_to_read)

            if "__symbol__" in df.columns:
                if not df.empty:
                    real_sym = str(df["__symbol__"].iloc[0])
                else:
                    real_sym = sym_guess
                df = df.drop(columns=["__symbol__"])
            else:
                real_sym = sym_guess

            if symbol_set is not None and real_sym not in symbol_set:
                if i % progress_every == 0 or i == total_files:
                    elapsed = time.time() - start_load
                    tprint(
                        f"Selective feature load progress: {i}/{total_files} files "
                        f"({(i / total_files) * 100:.1f}%) in {elapsed:.1f}s"
                    )
                continue

            for k in df.columns:
                if feature_set is not None and k not in feature_set:
                    continue
                if k not in feat_buffers:
                    feat_buffers[k] = {}
                feat_buffers[k][real_sym] = pd.to_numeric(df[k], errors="coerce").astype(np.float32, copy=False)

            del df
            if i % progress_every == 0 or i == total_files:
                elapsed = time.time() - start_load
                tprint(
                    f"Selective feature load progress: {i}/{total_files} files "
                    f"({(i / total_files) * 100:.1f}%) in {elapsed:.1f}s"
                )
        except Exception as e:
            tprint(f"Error loading {fpath}: {e}")

    if not feat_buffers:
        return None

    feats_out = {}
    for k, data in feat_buffers.items():
        feats_out[k] = pd.DataFrame(data).sort_index()

    tprint(f"Loaded {len(feats_out)} selected feature matrices.")
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
