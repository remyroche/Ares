import re

with open("extreme_price_movements/config.py", "r") as f:
    content = f.read()

training_resid_keys = """
TRAINING_RESIDUALIZATION_FEATURE_KEYS = [
    "overext_surprise",
    "blowoff_risk_surprise",
    "exh_qual_surprise",
    "dist_vwap_resid",
    "dist_ema_fast_resid",
    "trend_pct_resid",
]
"""

# Insert TRAINING_RESIDUALIZATION_FEATURE_KEYS after FEATURE_SELECTION_KEYS
idx = content.find("FEATURE_SELECTION_KEYS = [")
idx2 = content.find("]", idx) + 2
content = content[:idx2] + "\n" + training_resid_keys + "\n" + content[idx2:]

with open("extreme_price_movements/config.py", "w") as f:
    f.write(content)
