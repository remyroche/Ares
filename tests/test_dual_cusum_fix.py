import pandas as pd
import numpy as np
from src.training.steps.labeling.feature_generation_meta_labeling_step import generate_dual_cusum_signals

def test_dual_cusum_fix():
    print("Testing generate_dual_cusum_signals with Series check...")

    # Create dummy data
    n = 100
    dates = pd.date_range(start="2023-01-01", periods=n, freq="15min")
    close = pd.Series(np.random.normal(100, 1, n), index=dates)

    # Run signal generation
    try:
        signals = generate_dual_cusum_signals(
            close,
            k=0.01,
            window_vol=10,
            window_er=10
        )
        print("Success! DataFrame returned.")
        print("Columns:", signals.columns.tolist())

        if 'trend_signal' in signals.columns:
            print("trend_signal present")
        else:
            print("FAIL: trend_signal missing")
            exit(1)

    except AttributeError as e:
        print(f"FAIL: AttributeError caught: {e}")
        # This confirms if rolling() was called on numpy array
        exit(1)
    except Exception as e:
        print(f"FAIL: Exception caught: {e}")
        exit(1)

if __name__ == "__main__":
    test_dual_cusum_fix()
