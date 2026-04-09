import pandas as pd
import numpy as np
import time
from src.training.steps.labeling.profit_labeling.bar_construction import BarConstructor, BarConstructionConfig, BarType

np.random.seed(42)
n_rows = 100000
data = pd.DataFrame({
    'open': np.random.randn(n_rows).cumsum() + 100,
    'high': np.random.randn(n_rows).cumsum() + 100,
    'low': np.random.randn(n_rows).cumsum() + 100,
    'close': np.random.randn(n_rows).cumsum() + 100,
    'volume': np.random.randint(1, 100, n_rows)
}, index=pd.date_range('2023-01-01', periods=n_rows, freq='s'))

data['high'] = data[['open', 'high', 'low', 'close']].max(axis=1) + 1
data['low'] = data[['open', 'high', 'low', 'close']].min(axis=1) - 1

config_vol = BarConstructionConfig(bar_type=BarType.VOLUME, bar_size=1000)
constructor = BarConstructor(config_vol)

# Warmup Numba
_ = constructor.construct_bars(data.iloc[:100])

t0 = time.time()
res_vol = constructor.construct_bars(data)
t1 = time.time()
print(f"Numba Volume Bars: {t1 - t0:.4f}s, {len(res_vol)} bars")

config_dol = BarConstructionConfig(bar_type=BarType.DOLLAR, bar_size=100000)
constructor = BarConstructor(config_dol)

_ = constructor.construct_bars(data.iloc[:100])

t0 = time.time()
res_dol = constructor.construct_bars(data)
t1 = time.time()
print(f"Numba Dollar Bars: {t1 - t0:.4f}s, {len(res_dol)} bars")

config_tick = BarConstructionConfig(bar_type=BarType.TICK, bar_size=1000)
constructor = BarConstructor(config_tick)

_ = constructor.construct_bars(data.iloc[:100], price_change_threshold=0.1)

t0 = time.time()
res_tick = constructor.construct_bars(data, price_change_threshold=0.1)
t1 = time.time()
print(f"Numba Tick Bars: {t1 - t0:.4f}s, {len(res_tick)} bars")
