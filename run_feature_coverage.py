import pandas as pd
import numpy as np
import os
from extreme_price_movements.config import CFG
from extreme_price_movements.training_utils import audit_feature_coverage

def mock_data():
    idx = pd.date_range("2024-01-01", periods=1000, freq="15min")
    return pd.DataFrame({
        "open": np.random.randn(1000),
        "high": np.random.randn(1000),
        "low": np.random.randn(1000),
        "close": np.random.randn(1000),
        "volume": np.random.randn(1000)
    }, index=idx)

# We just need to parse the keys generated in `features.py`
# A lot of keys are dynamically generated, so passing dummy data will work to find them.
import extreme_price_movements.features as feats
mkt_gates = pd.DataFrame({
    "mkt_trend": np.random.randn(1000),
    "mkt_rv": np.random.randn(1000),
    "G_TF_TREND": np.random.randn(1000),
    "G_META_EXH": np.random.randn(1000)
}, index=mock_data().index)
d = mock_data()

# Some features expect log returns, let's just make it simple
d['close'] = np.abs(d['close']) + 1.0
d['high'] = d['close'] + 0.1
d['low'] = d['close'] - 0.1
d['open'] = d['close']

try:
    df_features = feats.compute_features_hourly(d, mkt_gates, CFG)
    audit = audit_feature_coverage(df_features, CFG)
    print("BASE UNUSED:")
    for f in audit['base_unused']: print(f)
    print("\nMETA UNUSED:")
    for f in audit['meta_unused']: print(f)
    print("\nGLOBAL UNUSED:")
    for f in audit['global_unused']: print(f)
except Exception as e:
    import traceback
    traceback.print_exc()
