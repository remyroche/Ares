import pandas as pd
df = pd.DataFrame([
    {"canonical_key": "volatility_expansion|z=4", "side": "short", "mean_net_ret": 0.05},
    {"canonical_key": "structure|z=8|p=breakout", "side": "long", "mean_net_ret": 0.06},
    {"canonical_key": "compression_transition|z=8", "side": "mixed", "mean_net_ret": 0.02},
])
df.to_csv("production_lgbm_outputs/combined_accepted_rule_registry.csv", index=False)
