import pandas as pd
import numpy as np

def test_rolling_abs():
    print("Testing Series.rolling(20).abs().sum() ...")
    s = pd.Series(np.random.randn(100))
    try:
        # This is the code causing the error
        x = s.rolling(20).abs().sum()
        print("Success (unexpected)")
    except AttributeError as e:
        print(f"Caught expected error: {e}")
    except Exception as e:
        print(f"Caught unexpected error: {type(e).__name__}: {e}")

    print("\nTesting Series.abs().rolling(20).sum() ...")
    try:
        # This is the proposed fix
        x = s.abs().rolling(20).sum()
        print("Success (fix works)")
    except Exception as e:
        print(f"Fix failed: {e}")

if __name__ == "__main__":
    test_rolling_abs()
