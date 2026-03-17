import numpy as np
import pandas as pd
import time
import cProfile
import pstats
import io
from extreme_price_movements.intraday_crypto_library import build_intraday_crypto_library

def generate_dummy_data(rows=10000):
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=rows, freq="15min")
    close = 100 + np.random.randn(rows).cumsum()
    high = close + np.random.rand(rows) * 2
    low = close - np.random.rand(rows) * 2
    open_p = close - np.random.randn(rows)
    volume = np.random.randint(100, 1000, rows)
    session_id = np.repeat(np.arange(rows // 96 + 1), 96)[:rows]

    df = pd.DataFrame({
        "open": open_p,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "session_id": session_id
    }, index=dates)
    return df

num_assets = 30
dfs = []
for i in range(num_assets):
    df_i = generate_dummy_data(50000)
    dfs.append(df_i)

panel = {}
for col in ['open', 'high', 'low', 'close', 'volume']:
    panel[col] = pd.concat([df[col] for df in dfs], axis=1)
    panel[col].columns = [f"A{i}" for i in range(num_assets)]
panel['session_id'] = dfs[0]['session_id']

pr = cProfile.Profile()
pr.enable()
res2 = build_intraday_crypto_library(panel)
pr.disable()
s = io.StringIO()
sortby = 'cumulative'
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats(30)
print(s.getvalue())
