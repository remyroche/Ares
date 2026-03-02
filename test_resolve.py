import pandas as pd
import numpy as np

def create_mock_data():
    idx = pd.date_range('2023-01-01', periods=10, freq='1h')
    df = pd.DataFrame({
        'open': np.ones(10) * 100,
        'high': np.ones(10) * 110,
        'low': np.ones(10) * 90,
        'close': np.ones(10) * 105
    }, index=idx)
    return {'close': df[['close']], 'open': df[['open']], 'high': df[['high']], 'low': df[['low']]}

from extreme_price_movements.labeling import compute_triple_barrier_labels

try:
    panel = create_mock_data()
    tp = 0.05  # 5%
    sl = 0.05  # 5%
    # This should trigger an ambiguous bar because high is 110 (10% up) and low is 90 (10% down)
    labels, returns = compute_triple_barrier_labels(panel, tp, sl, horizon=5, side="long")
    print(labels.iloc[0]['close'])
    print("Test passed without crashing.")
except Exception as e:
    print(f"Error: {e}")
