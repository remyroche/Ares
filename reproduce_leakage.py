
import pandas as pd
import numpy as np
from src.feature_generation.categories.layer3_specific_features import generate_layer3_features

def test_multi_asset_leakage():
    # Create synthetic multi-asset data
    dates = pd.date_range('2024-01-01', periods=100, freq='15min')

    # Asset A: Price 100 -> 110
    df_a = pd.DataFrame({
        'timestamp': dates,
        'asset_id': 'AssetA',
        'close': np.linspace(100, 110, 100),
        'volume': 1000,
        'high': np.linspace(100, 110, 100) + 1,
        'low': np.linspace(100, 110, 100) - 1,
    })

    # Asset B: Price 10 -> 11 (Huge drop from Asset A's 110 if treated continuously)
    df_b = pd.DataFrame({
        'timestamp': dates,
        'asset_id': 'AssetB',
        'close': np.linspace(10, 11, 100),
        'volume': 1000,
        'high': np.linspace(10, 11, 100) + 1,
        'low': np.linspace(10, 11, 100) - 1,
    })

    # Concatenate
    df = pd.concat([df_a, df_b]).reset_index(drop=True)

    # Add dummy base model col
    df['base_prob'] = 0.5

    # Run generation
    try:
        df_out = generate_layer3_features(df, ['base_prob'])

        # Check for leakage at the boundary
        # The first row of Asset B corresponds to index 100
        # If calculating returns/diff without groupby, it will use Asset A's last price (110)
        # against Asset B's first price (10), resulting in ~ -90% return.

        # Check a feature that uses diff/returns, e.g., 'slope_short' (log price diff) or simple diff
        # Let's check 'momentum_short' which is abs(close.diff(12) / close.shift(12))

        # Index 100 is start of Asset B. shift(12) would be index 88 (Asset A).
        # Expected if buggy: mixing A and B.
        # Expected if fixed: NaN for first 12 rows of B.

        feat_val = df_out.loc[100, 'momentum_short']
        print(f"Feature value at boundary (idx 100): {feat_val}")

        # Verify leakage
        # Asset A last close: 110. Asset B first close: 10.
        # diff(12) at idx 100: close[100] (10) - close[88] (approx 108.8) = -98.8
        # momentum_short = abs(-98.8 / 108.8) = ~0.9

        if not np.isnan(feat_val):
            print("Leakage detected! value is not NaN at start of second asset.")
        else:
            print("No leakage detected (or feature is NaN for other reasons).")

    except Exception as e:
        print(f"Execution failed: {e}")

if __name__ == "__main__":
    test_multi_asset_leakage()
