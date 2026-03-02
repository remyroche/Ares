import numpy as np
from extreme_price_movements.labeling import _numba_triple_barrier

closes = np.array([100, 102, 106, 100], dtype=np.float32)
highs = np.array([100, 102, 106, 100], dtype=np.float32)
lows = np.array([100, 102, 106, 100], dtype=np.float32)
opens = closes.copy()
times = np.array([0, 3600*1e9, 2*3600*1e9, 3*3600*1e9], dtype=np.int64)

tp = np.full(4, 0.05, dtype=np.float32)
sl = np.full(4, 0.02, dtype=np.float32)
horizon = 5
side = 1

lbs, rets, idxs = _numba_triple_barrier(times, opens, highs, lows, closes, tp, sl, horizon, side)
print("Labels:", lbs)
print("Returns:", rets)
print("Idxs:", idxs)
