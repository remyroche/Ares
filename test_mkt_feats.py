import sys
sys.path.append('.')
import pandas as pd
import numpy as np
import warnings

warnings.simplefilter('error', RuntimeWarning)

# Test mkt_rv_pct
df = pd.DataFrame({
    "mkt_rv": [0.0, np.nan, 1.0, 0.0],
})
rv_mean = pd.Series([0.0, 0.0, 1.0, np.nan])
rv_std = pd.Series([np.nan, 0.0, 1.0, 0.0]).clip(lower=1e-6)

try:
    mkt_rv_pct = ((df["mkt_rv"] - rv_mean) / rv_std).clip(-6, 6).fillna(0.0).astype(np.float32)
    mkt_rv_pct = (0.5 * (1.0 + np.vectorize(np.math.erf)(mkt_rv_pct / np.sqrt(2.0)))).astype(np.float32)
    print("mkt_rv_pct works")
except Exception as e:
    print("mkt_rv_pct failed:", e)
