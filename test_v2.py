import numpy as np
import pandas as pd
from extreme_price_movements.position_sizer_v2 import LayerAPredictor, LayerBPolicyOptimizer, LayerCExecutionOptimizer, make_temporal_splits

print("Testing position_sizer_v2 implementation...")
print("Successfully imported components.")

# Dummy test just for syntactic structure
ts = np.array([1, 2, 3, 4, 5, 6] * 20)
splits = make_temporal_splits(ts, n_samples=120, n_splits=3)
print(f"Created {len(splits)} temporal splits.")

b_opt = LayerBPolicyOptimizer()
print("Initialized Layer B optimizer.")
