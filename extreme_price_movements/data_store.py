import os
import time
import numpy as np
import pandas as pd
import ccxt

from utils import tprint

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

class OHLCVStore:
    def __init__(self, root_dir="data", timeframe="1h"):
        self.root_dir = root_dir
        self.timeframe = timeframe
        self.ohlcv_dir = os.path.join(root_dir, "ohlcv")
        os.makedirs(self.ohlcv_dir, exist_ok=True)

    def _sym_path(self, symbol: str) -> str:
        safe = symbol.replace("/", "_")
        return os.path.join(self.ohlcv_dir, f"{safe}.parquet")

    def _downcast(self, df: pd.DataFrame) -> pd.DataFrame:
        if d
