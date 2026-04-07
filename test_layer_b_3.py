import numpy as np
import pandas as pd
from extreme_price_movements.position_sizer_v2 import LayerBPolicyOptimizer, LabelPolicy, _simulate_policy_batch

# Direct test of score_candidates logic
opt = LayerBPolicyOptimizer()

results = [
    {"policy_obj": LabelPolicy(1.0, 1.0, 10, 1.0, 0.5, 0, 0.0), "net_pnl_day": 0.05, "sortino": 1.5, "maxDD": 0.1, "instability": 0.02, "timeout_rate": 0.1},
    {"policy_obj": LabelPolicy(1.0, 1.0, 20, 1.0, 0.5, 0, 0.0), "net_pnl_day": 0.05, "sortino": 1.5, "maxDD": 0.1, "instability": 0.02, "timeout_rate": 0.1},
]

scored = opt._score_candidates(results)
print([r["rank"] for r in scored])
print(scored[0]["policy_obj"])
