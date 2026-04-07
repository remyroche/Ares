import numpy as np
import pandas as pd
from extreme_price_movements.position_sizer_v2 import LayerBPolicyOptimizer
from extreme_price_movements.label_policy_optimizer import LabelPolicy
from extreme_price_movements.position_sizer_v2 import make_temporal_splits

# dummy test to see if we can import and instantiate
opt = LayerBPolicyOptimizer()
print("Successfully imported and instantiated LayerBPolicyOptimizer")
