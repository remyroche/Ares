import numpy as np
import pandas as pd
from extreme_price_movements.position_sizer_v2 import LayerBPolicyOptimizer
from extreme_price_movements.label_policy_optimizer import LabelPolicy
from extreme_price_movements.position_sizer_v2 import make_temporal_splits

opt = LayerBPolicyOptimizer()
assert opt.layerB_score_mode == "stable_absolute"
assert opt.eps_policy_utility == 1e-4

print("Basic tests pass.")
