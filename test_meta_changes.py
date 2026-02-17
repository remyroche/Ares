import numpy as np
import pandas as pd
from extreme_price_movements.training import compute_meta_target

def test_target():
    n = 100
    r2 = np.random.randn(n) * 0.01
    r4 = np.random.randn(n) * 0.02
    r8 = np.random.randn(n) * 0.03
    vol = np.random.rand(n) * 0.02 + 0.01 # ATR approx 1-3%

    # Old way
    y_old = compute_meta_target(r2, r4, r8)
    print(f"Old Y stats: mean={y_old.mean():.4f}, std={y_old.std():.4f}")

    # New logic simulation
    # Normalize
    n2 = r2 / (vol * np.sqrt(2))
    n4 = r4 / (vol * np.sqrt(4))
    n8 = r8 / (vol * np.sqrt(8))
    raw = 0.4*n2 + 0.35*n4 + 0.25*n8

    # Squash
    c = 2.0
    y_new = np.arcsinh(raw / c)

    print(f"New Y stats: mean={y_new.mean():.4f}, std={y_new.std():.4f}")

    # Check correlation
    corr = np.corrcoef(y_old, y_new)[0,1]
    print(f"Correlation old vs new: {corr:.4f}")

if __name__ == "__main__":
    test_target()
