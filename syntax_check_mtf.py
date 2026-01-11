
import sys
import pandas as pd
import numpy as np
try:
    from src.training.steps.labeling.mtf_feature_generation import create_meta_features, compute_proxy_levels
    from src.training.steps.labeling.causal_specialists import LiquidationSpecialist
    print("Import successful")
except ImportError as e:

    print(f"Import failed: {e}")
    sys.exit(1)
except SyntaxError as e:
    print(f"Syntax error: {e}")
    sys.exit(1)

# Mock data
df = pd.DataFrame({
    'close': np.random.rand(100) + 100,
    'high': np.random.rand(100) + 101,
    'low': np.random.rand(100) + 99,
    'open': np.random.rand(100) + 100,
    'volume': np.random.rand(100) * 1000
})
df.index = pd.date_range(start='2021-01-01', periods=100, freq='H')
signals = pd.DataFrame(index=df.index)

try:
    feats = create_meta_features(df, signals)
    print("Feature generation successful")
    print("New features:", [c for c in feats.columns if 'liquidation' in c or 'proxy' in c])
    
    # Test Specialist Instantiation
    spec = LiquidationSpecialist()
    print(f"Liquidation Specialist instantiated: {spec.name}")
    
except Exception as e:
    print(f"Execution failed: {e}")
    sys.exit(1)
