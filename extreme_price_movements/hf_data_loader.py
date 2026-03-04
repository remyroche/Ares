"""
High-frequency (15m) OHLCV data loader for precise trailing profit simulation.

Downloads 15m data via CCXT and stores in parquet format.
Checks local storage before downloading to avoid redundant API calls.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import ccxt
from extreme_price_movements.utils import tprint


# Storage directory for 15m data
HF_DATA_DIR = Path(os.environ.get("EPM_HF_DATA_DIR", str(Path(__file__).parent / "15m_ohlcv")))
HF_DATA_DIR.mkdir(exist_ok=True)


def _get_parquet_path(symbol: str) -> Path:
    """Get parquet file path for a symbol."""
    # Normalize symbol: BTC/USDT -> btcusdt, BTC_USDT -> btcusdt
    clean_symbol = symbol.replace("/", "").replace("_", "").lower()
    return HF_DATA_DIR / f"{clean_symbol}_15m.parquet"


def _load_existing_data(symbol: str, allow_quote_fallback: bool = True) -> pd.DataFrame:
    """Load existing 15m data from parquet if available.

    Falls back to the USDT-quoted variant if the requested quote currency
    (USDC, BUSD, EUR, etc.) doesn't have a cached parquet yet but the same
    base asset does under USDT. Price series are equivalent for barrier
    refinement purposes.
    """
    path = _get_parquet_path(symbol)
    if path.exists():
        try:
            df = pd.read_parquet(path)
            idx = pd.to_datetime(df.index)
            if idx.tz is None:
                df.index = idx.tz_localize("UTC")
            else:
                df.index = idx.tz_convert("UTC")
            return df
        except Exception as e:
            tprint(f"WARNING: Failed to load {path}: {e}")
            return pd.DataFrame()

    if not allow_quote_fallback:
        return pd.DataFrame()

    # Fallback: try USDT variant for non-USDT quoted symbols
    _FALLBACK_QUOTES = ("USDC", "BUSD", "EUR")
    sym_up = symbol.replace("/", "").replace("_", "").upper()
    for q in _FALLBACK_QUOTES:
        if sym_up.endswith(q):
            base = sym_up[:-len(q)]
            fallback_path = HF_DATA_DIR / f"{base.lower()}usdt_15m.parquet"
            if fallback_path.exists():
                try:
                    df = pd.read_parquet(fallback_path)
                    idx = pd.to_datetime(df.index)
                    if idx.tz is None:
                        df.index = idx.tz_localize("UTC")
                    else:
                        df.index = idx.tz_convert("UTC")
                    return df
                except Exception:
                    pass
            break

    return pd.DataFrame()



def _save_data(symbol: str, df: pd.DataFrame):
    """Save 15m data to parquet with float32 downcasting."""
    if df.empty:
        return
    
    # Downcast to float32
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = df[col].astype(np.float32)
    
    path = _get_parquet_path(symbol)
    df.to_parquet(path, compression='snappy')


def _download_from_exchange(exchange: ccxt.Exchange, symbol: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    """Download 15m OHLCV via CCXT."""
    start_ms = int(start_ts.timestamp() * 1000)
    end_ms = int(end_ts.timestamp() * 1000)
    
    tprint(f"Downloading 15m data for {symbol}: {start_ts} to {end_ts}")
    
    try:
        # CCXT uses '15m' timeframe
        all_ohlcv = []
        current_ms = start_ms
        
        # Fetch in chunks (CCXT has limits per request)
        while current_ms < end_ms:
            ohlcv = exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe='15m',
                since=current_ms,
                limit=1000  # Max per request
            )
            
            if not ohlcv:
                break
            
            all_ohlcv.extend(ohlcv)
            
            # Move to next chunk
            current_ms = ohlcv[-1][0] + 1
            
            # Stop if we've reached the end
            if ohlcv[-1][0] >= end_ms:
                break
        
        if not all_ohlcv:
            tprint(f"WARNING: No data returned for {symbol}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df = df.set_index('timestamp')
        
        # Filter to requested range
        df = df[(df.index >= start_ts) & (df.index <= end_ts)]
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        return df
    
    except Exception as e:
        tprint(f"ERROR: Failed to download {symbol} 15m data: {e}")
        return pd.DataFrame()


def get_15m_ohlcv(exchange: ccxt.Exchange, symbol: str, entry_ts: pd.Timestamp, max_hold_hours: int = 12) -> pd.DataFrame:
    """
    Get 15m OHLCV data for a symbol, downloading if necessary.
    
    Downloads 12 hours of data at once to cover typical holding periods.
    Checks local parquet storage before downloading.
    Overwrites existing data if there's overlap.
    
    Args:
        exchange: CCXT exchange instance (e.g., ccxt.binance())
        symbol: Trading pair in CCXT format (e.g., 'BTC/USDT')
        entry_ts: Entry timestamp
        max_hold_hours: Hours of data to ensure available (default 12)
    
    Returns:
        DataFrame with 15m OHLCV data
    """
    # Ensure UTC
    if entry_ts.tz is None:
        entry_ts = entry_ts.tz_localize('UTC')
    else:
        entry_ts = entry_ts.tz_convert('UTC')
    
    # Download window: 12 hours from entry
    download_start = entry_ts
    download_end = entry_ts + pd.Timedelta(hours=max_hold_hours)
    
    # Load existing data
    existing_df = _load_existing_data(symbol)
    
    # Check if we need to download
    need_download = False
    
    if existing_df.empty:
        need_download = True
    else:
        # Check coverage
        existing_start = existing_df.index.min()
        existing_end = existing_df.index.max()
        
        # Need download if requested range is not fully covered
        if download_start < existing_start or download_end > existing_end:
            need_download = True
    
    if need_download:
        # Download new data
        new_df = _download_from_exchange(exchange, symbol, download_start, download_end)
        
        if not new_df.empty:
            if existing_df.empty:
                # No existing data, just save new
                combined_df = new_df
            else:
                # Merge with existing, overwriting overlaps
                combined_df = pd.concat([existing_df, new_df])
                combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
                combined_df = combined_df.sort_index()
            
            # Save updated data
            _save_data(symbol, combined_df)
            
            # Return requested range
            return combined_df.loc[entry_ts:download_end]
        else:
            # Download failed, return existing data if available
            if not existing_df.empty:
                return existing_df.loc[entry_ts:download_end]
            return pd.DataFrame()
    else:
        # Use existing data
        return existing_df.loc[entry_ts:download_end]


def sync_15m_ohlcv_range(
    exchange: ccxt.Exchange,
    symbol: str,
    since_ts: pd.Timestamp,
    until_ts: pd.Timestamp | None = None,
    full_backfill: bool = True,
) -> pd.DataFrame:
    """Ensure local 15m cache covers [since_ts, until_ts] for symbol."""
    if since_ts.tz is None:
        since_ts = since_ts.tz_localize("UTC")
    else:
        since_ts = since_ts.tz_convert("UTC")

    if until_ts is None:
        until_ts = pd.Timestamp.now(tz="UTC")
    elif until_ts.tz is None:
        until_ts = until_ts.tz_localize("UTC")
    else:
        until_ts = until_ts.tz_convert("UTC")

    if until_ts <= since_ts:
        return pd.DataFrame()

    # Strict check: only this exact symbol/quote cache should determine coverage.
    existing_df = _load_existing_data(symbol, allow_quote_fallback=False)
    if not existing_df.empty:
        ex_start = existing_df.index.min()
        ex_end = existing_df.index.max()
        if ex_start <= since_ts and ex_end >= until_ts:
            return existing_df.loc[(existing_df.index >= since_ts) & (existing_df.index <= until_ts)]

    # Build missing ranges and only download gaps instead of re-downloading covered periods.
    download_ranges: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    if existing_df.empty:
        download_ranges.append((since_ts, until_ts))
    else:
        ex_start = existing_df.index.min()
        ex_end = existing_df.index.max()

        if full_backfill:
            # Legacy behavior: force a single pass from since_ts, may overlap.
            download_ranges.append((since_ts, until_ts))
        else:
            # Missing history before current cache start.
            if since_ts < ex_start:
                left_end = min(until_ts, ex_start - pd.Timedelta(minutes=15))
                if since_ts <= left_end:
                    download_ranges.append((since_ts, left_end))

            # Missing tail after current cache end.
            if until_ts > ex_end:
                right_start = max(since_ts, ex_end + pd.Timedelta(minutes=15))
                if right_start <= until_ts:
                    download_ranges.append((right_start, until_ts))

    if not download_ranges:
        if existing_df.empty:
            return pd.DataFrame()
        return existing_df.loc[(existing_df.index >= since_ts) & (existing_df.index <= until_ts)]

    chunks: list[pd.DataFrame] = []
    for dl_start, dl_end in download_ranges:
        if dl_start >= dl_end:
            continue
        new_df = _download_from_exchange(exchange, symbol, dl_start, dl_end)
        if not new_df.empty:
            chunks.append(new_df)

    if existing_df.empty and not chunks:
        return pd.DataFrame()
    if not chunks:
        return existing_df.loc[(existing_df.index >= since_ts) & (existing_df.index <= until_ts)]

    combined_parts = [existing_df] if not existing_df.empty else []
    combined_parts.extend(chunks)
    combined_df = pd.concat(combined_parts)
    combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
    combined_df = combined_df.sort_index()

    _save_data(symbol, combined_df)
    return combined_df.loc[(combined_df.index >= since_ts) & (combined_df.index <= until_ts)]


def clear_cache(symbol: str = None):
    """
    Clear cached 15m data.
    
    Args:
        symbol: If provided, clear only this symbol. Otherwise clear all.
    """
    if symbol:
        path = _get_parquet_path(symbol)
        if path.exists():
            path.unlink()
            tprint(f"Cleared 15m cache for {symbol}")
    else:
        for path in HF_DATA_DIR.glob("*_15m.parquet"):
            path.unlink()
        tprint("Cleared all 15m cache")


def bulk_sync_15m_universe(
    symbols: list,
    since_ts: pd.Timestamp,
    until_ts: pd.Timestamp | None = None,
    quotes: tuple = ("USDT", "USDC", "BUSD"),
    skip_existing: bool = True,
    symbol_order: str = "alpha_asc",
    partition_count: int = 1,
    partition_id: int = 0,
    exchange_preference: tuple[str, ...] = ("binance", "binanceus"),
) -> dict:
    """Download 15m OHLCV for every symbol in the universe, covering all quote currencies.

    Unlike the previous bulk download which only fetched USDT pairs, this function
    expands each base asset to all requested `quotes` and downloads any variant
    that is missing or stale.

    Args:
        symbols : List of trading pairs in any format, e.g. 'BTC/USDT', 'BTC_USDC', 'BTCUSDT'.
        since_ts: Start of the desired 15m range (UTC).
        until_ts: End of the desired range; defaults to now.
        quotes  : Quote currencies to ensure coverage for.
        skip_existing: If True, skips symbols already fully covered.
        symbol_order: alpha_asc (default) or alpha_desc.
        partition_count: Number of workers partitioning the symbol list.
        partition_id: Zero-based worker id within [0, partition_count).
        exchange_preference: CCXT exchange ids to try in order.

    Returns:
        dict mapping symbol → 'ok' | 'skipped' | 'failed'
    """
    import ccxt as _ccxt

    if until_ts is None:
        until_ts = pd.Timestamp.now(tz="UTC")

    since_ts = since_ts.tz_localize("UTC") if since_ts.tz is None else since_ts.tz_convert("UTC")
    until_ts = until_ts.tz_localize("UTC") if until_ts.tz is None else until_ts.tz_convert("UTC")

    # Extract unique base assets from symbol list
    requested_quotes = tuple(q.upper() for q in quotes)
    _KNOWN_QUOTES = {
        *requested_quotes,
        "USDT", "USDC", "BUSD", "FDUSD", "TUSD", "USDP", "DAI",
        "EUR", "BTC", "ETH", "BNB", "TRY", "BRL",
    }

    def _parse_base_quote(raw_symbol: str) -> tuple[str, str] | tuple[None, None]:
        sym = raw_symbol.strip().upper()
        if not sym:
            return None, None

        if "/" in sym:
            parts = sym.split("/", 1)
            if len(parts) == 2 and parts[0] and parts[1]:
                return parts[0], parts[1]
            return None, None

        if "_" in sym:
            parts = sym.split("_", 1)
            if len(parts) == 2 and parts[0] and parts[1]:
                return parts[0], parts[1]
            return None, None

        compact = sym.replace("-", "")
        for q in sorted(_KNOWN_QUOTES, key=len, reverse=True):
            if compact.endswith(q) and len(compact) > len(q):
                return compact[:-len(q)], q

        # If only base asset is provided (e.g. BTC), caller-supplied quotes expand it.
        if compact.isascii() and compact.isalnum():
            return compact, ""

        return None, None

    base_assets: set[str] = set()
    for sym in symbols:
        base, _quote = _parse_base_quote(str(sym))
        if base and base.isascii() and base.isalnum():
            base_assets.add(base)

    # Build the full set of symbols to sync
    target_symbols: list[str] = []
    for base in sorted(base_assets):
        for q in requested_quotes:
            target_symbols.append(f"{base}/{q}")

    if symbol_order == "alpha_desc":
        target_symbols = sorted(target_symbols, reverse=True)
    else:
        target_symbols = sorted(target_symbols)

    if partition_count < 1:
        raise ValueError("partition_count must be >= 1")
    if partition_id < 0 or partition_id >= partition_count:
        raise ValueError("partition_id must be in [0, partition_count)")

    if partition_count > 1:
        target_symbols = target_symbols[partition_id::partition_count]

    tprint(
        "bulk_sync_15m_universe: "
        f"{len(base_assets)} base assets × {len(requested_quotes)} quotes -> {len(target_symbols)} symbols "
        f"(order={symbol_order}, partition={partition_id}/{partition_count})"
    )

    ex = None
    valid_ccxt = None
    bootstrap_errors: list[str] = []
    for ex_id in exchange_preference:
        try:
            ex_cls = getattr(_ccxt, ex_id)
        except AttributeError:
            bootstrap_errors.append(f"{ex_id}: unknown exchange id")
            continue

        candidate = ex_cls({"enableRateLimit": True})
        try:
            markets = candidate.load_markets()
            ex = candidate
            valid_ccxt = set(markets.keys())
            tprint(f"Using exchange '{ex_id}' with {len(valid_ccxt)} listed markets")
            break
        except Exception as e:
            bootstrap_errors.append(f"{ex_id}: {e}")
            continue

    if ex is None or valid_ccxt is None:
        tprint("ERROR: Could not initialize any exchange. Aborting bulk sync.")
        for err in bootstrap_errors:
            tprint(f"  - {err}")
        return {sym: "bootstrap_failed" for sym in target_symbols}

    results: dict = {}
    skipped = already_ok = failed = ok = 0

    for sym in target_symbols:
        if sym not in valid_ccxt:
            results[sym] = "not_listed"
            skipped += 1
            continue

        if skip_existing:
            # Strict coverage check for this exact symbol (no quote fallback).
            existing = _load_existing_data(sym, allow_quote_fallback=False)
            if not existing.empty and existing.index.min() <= since_ts and existing.index.max() >= until_ts:
                results[sym] = "skipped"
                already_ok += 1
                continue

        try:
            # Incremental mode avoids re-downloading periods already cached.
            df = sync_15m_ohlcv_range(ex, sym, since_ts, until_ts, full_backfill=False)
            if df is not None and not df.empty:
                results[sym] = "ok"
                ok += 1
            else:
                results[sym] = "empty"
                failed += 1
        except Exception as e:
            tprint(f"bulk_sync: failed {sym}: {e}")
            results[sym] = "failed"
            failed += 1

    tprint(f"bulk_sync_15m_universe done: ok={ok} skipped={already_ok} not_listed={skipped} failed/empty={failed}")
    return results
